"""
Ablation: RL-MCF without Autoencoder — Table (ICONIP revision)
===============================================================
Trains the RL agent with an IdentityAutoEncoder (no AE) instead of
the TCN-AE, so the agent operates directly in the raw input space.

Purpose (Reviewer 1223):
    The TCN-AE is claimed to provide an implicit plausibility bias.
    This ablation verifies that claim empirically: removing the AE
    should degrade plausibility (higher anomaly score) while other
    metrics may also shift, justifying the AE's role.

Comparison produced:
    RL-MCF (with AE)  vs  RL-MCF-NoAE (identity AE)

Usage:
    $env:PYTHONPATH = "."
    python scripts/evals/run_ablation_no_ae.py
    python scripts/evals/run_ablation_no_ae.py --seeds 0 1 2 --eval_batches 20

Results saved to:
    assets/results/ablation_no_ae/<model>/seed_<s>_evaluation.json
    assets/results/ablation_no_ae/summary.json
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.utils.train_tools import get_device
from src.data_provider.data_factory import data_provider
from src.models.autoencoder.identity_ae import IdentityAutoEncoder
from src.models.RL.agent import ActorCritic
from src.models.RL.reward_last import CFReward
from src.models.Forecaster.forecaster_wrapper_v2 import ForecasterWrapperV2
from src.evaluation.unified_evaluator import CounterfactualEvaluator
from src.training.RL_trainers.trainer_main import (
    build_temporal_mask,
    filter_batch,
    run_episode_eval,
)

# ─── Default config — ETTh1 / iTransformer (reference config) ─────────────────
FORECAST_CONFIG  = "assets/configs/etth1_dataset/forecasters/itransformer/etth1_96_48_S.json"
AE_CONFIG        = "assets/configs/etth1_dataset/ae/tcn_ae.json"   # used only for bounds params
RL_CONFIG        = "assets/configs/etth1_dataset/RL_ablations/config_itransformer_best.json"
PLAUS_CKPT       = "assets/checkpoints/etth1_chpts/anomaly_detector/plausibility_etth1.pkl"
OUTPUT_DIR       = "assets/results/ablation_no_ae"

SEEDS            = [0, 1, 2]
EVAL_BATCHES     = 20

TABLE_METRICS = [
    "validity_ratio",
    "stepwise_auc",
    "proximity_l2",
    "compactness",
    "temporal_consistency",
    "plausibility_ensemble",
]


# ──────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────────────
class _NormIdentityAE(nn.Module):
    """
    IdentityAutoEncoder with min-max normalisation to [-1, 1].

    Encode: x (B, T, 1) → (x - x_min) / x_range * 2 - 1  → z (B, T)
    Decode: z (B, T)    → (z + 1) / 2 * x_range + x_min  → x (B, T, 1)

    This makes the latent space comparable to the TCN-AE latent space so
    that eta and perturbation magnitudes are directly equivalent.
    """

    def __init__(self, seq_len: int, x_min: float, x_range: float):
        super().__init__()
        self.seq_len  = seq_len
        self.latent_dim = seq_len
        self.register_buffer("x_min",   torch.tensor(x_min,   dtype=torch.float32))
        self.register_buffer("x_range", torch.tensor(x_range, dtype=torch.float32))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1)  →  z: (B, T)  in [-1, 1]
        return ((x.squeeze(-1) - self.x_min) / self.x_range) * 2.0 - 1.0

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, T) in [-1, 1]  →  x: (B, T, 1)
        if z.dim() == 2:
            z = z.unsqueeze(-1)
        return (z + 1.0) / 2.0 * self.x_range + self.x_min

    def forward(self, x: torch.Tensor):
        return self.decode(self.encode(x))


# ──────────────────────────────────────────────────────────────────────────────
class RLNoAETrainer:
    """
    Minimal re-implementation of RLMaskTrainer that swaps the TCN-AE
    for an IdentityAutoEncoder.  Everything else (reward, agent, loops)
    is identical to trainer_main.py so comparisons are fair.
    """

    def __init__(self, cfg_f, cfg_rl, device, ckpt_dir: str):
        self.cfg_f   = cfg_f
        self.cfg_rl  = cfg_rl
        self.device  = device
        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        self.exp_name    = getattr(cfg_rl, "name", "exp") + "_noae"
        self.mask_last_k = getattr(cfg_rl, "mask_last_k", 24)
        self.mask_ramp_k = getattr(cfg_rl, "mask_ramp_k", 8)
        self.filter_q    = getattr(cfg_rl, "filter_quantile", 0.0)

        seq_len = cfg_f.seq_len          # 96

        # ── Frozen forecaster ─────────────────────────────────────────────────
        self.forecaster = ForecasterWrapperV2(cfg_f, device)
        self.forecaster.model.eval()
        for p in self.forecaster.model.parameters():
            p.requires_grad_(False)

        # ── Data (must come before AE — we need train stats) ─────────────────
        _, self.train_loader = data_provider(cfg_f, "train")
        _, self.test_loader  = data_provider(cfg_f, "test")

        # ── Global sigma (same logic as trainer_main) ─────────────────────────
        _x_stds, _xs = [], []
        for i, batch in enumerate(self.train_loader):
            if i >= 50: break
            bx, _, _, _ = batch
            _x_stds.append(bx[:, :, -1].numpy().std(axis=1))
            _xs.append(bx[:, :, -1].numpy())
        self._global_sigma = float(np.concatenate(_x_stds).mean())
        print(f"[NoAE] global_sigma={self._global_sigma:.4f}")

        # ── Identity AE with min-max normalisation to [-1, 1] ────────────────
        # The TCN-AE latent space lives in ~[-1, 1] (clamped in run_episode).
        # We normalise the raw input to the same range so that eta and
        # perturbation magnitudes are directly comparable between runs.
        _xs_all = np.concatenate(_xs, axis=0)
        self._x_min   = float(_xs_all.min())
        self._x_max   = float(_xs_all.max())
        self._x_range = max(self._x_max - self._x_min, 1e-6)
        print(f"[NoAE] x_min={self._x_min:.4f}  x_max={self._x_max:.4f}")

        self.ae_arch = _NormIdentityAE(
            seq_len=seq_len,
            x_min=self._x_min,
            x_range=self._x_range,
        ).to(device)
        self.ae_arch.eval()
        # No learnable parameters — pure normalised pass-through

        # ── Reward ────────────────────────────────────────────────────────────
        self.reward_fn = CFReward(
            fr=getattr(cfg_rl, "fr", 1.0),
            rho=getattr(cfg_rl, "rho", 0.1),
            w_validity=getattr(cfg_rl, "w_validity", 4.0),
            w_proximity=getattr(cfg_rl, "w_proximity", 0.3),
            w_hard_bonus=getattr(cfg_rl, "w_hard_bonus", 3.0),
            w_smooth=getattr(cfg_rl, "w_smooth", 0.5),
            use_validity=getattr(cfg_rl, "use_validity", True),
            use_proximity=getattr(cfg_rl, "use_proximity", True),
            use_smooth=getattr(cfg_rl, "use_smooth", True),
            direction=getattr(cfg_rl, "direction", -1.0),
            global_sigma=self._global_sigma,
        ).to(device)

        # ── Agent — latent_dim = seq_len (no compression) ────────────────────
        self.agent = ActorCritic(
            latent_dim=seq_len,          # 96 instead of 64
            pred_len=cfg_f.pred_len,
            eta=cfg_rl.eta,
            entropy_coef=cfg_rl.entropy_coef,
            direction=getattr(cfg_rl, "direction", -1.0),
        ).to(device)

        params = self.agent.count_parameters()
        print(f"[NoAE] Actor {params['actor']:,}  Critic {params['critic']:,}")

        self.opt_actor  = torch.optim.Adam(self.agent.actor.parameters(),
                                           lr=cfg_rl.lr_actor)
        self.opt_critic = torch.optim.Adam(self.agent.critic.parameters(),
                                           lr=getattr(cfg_rl, "lr_critic", 1e-4))

        decay_step = max(1, cfg_rl.epochs // 2)
        lr_decay   = getattr(cfg_rl, "lr_decay", 0.5)
        self.sched_actor  = torch.optim.lr_scheduler.StepLR(
            self.opt_actor,  step_size=decay_step, gamma=lr_decay)
        self.sched_critic = torch.optim.lr_scheduler.StepLR(
            self.opt_critic, step_size=decay_step, gamma=lr_decay)

        # ── Evaluator with pre-trained plausibility detector ──────────────────
        if os.path.exists(PLAUS_CKPT):
            self.evaluator = CounterfactualEvaluator(
                x_train=None,
                fit_plausibility=False,
                plausibility_checkpoint=PLAUS_CKPT,
            )
        else:
            # Fallback: fit on train set
            x_tr = []
            for i, b in enumerate(self.train_loader):
                if i >= 20: break
                bx, _, _, _ = b
                x_tr.append(bx[:, :, -1:].numpy())
            self.evaluator = CounterfactualEvaluator(
                x_train=np.concatenate(x_tr, axis=0),
                fit_plausibility=True,
            )

    # ── Training loop (mirror of trainer_main.train) ─────────────────────────
    def train(self):
        print(f"\n{'='*60}")
        print(f"[NoAE] Training | {self.cfg_rl.epochs} epochs")
        print(f"{'='*60}")

        best_reward = -float("inf")

        for epoch in range(1, self.cfg_rl.epochs + 1):
            self.agent.train()
            rewards, srs = [], []

            for batch in self.train_loader:
                ep = self._run_episode_train(batch)
                if ep is None:
                    continue

                rs = self.reward_fn.stats(ep["reward_dict"])
                rewards.append(rs["total"])
                srs.append(rs["success"])

                # Actor-Critic update
                loss_dict = self.agent.compute_loss(
                    ep["log_prob"], ep["reward"], ep["value"], ep["entropy"]
                )
                self.opt_actor.zero_grad()
                loss_dict["actor"].backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.agent.actor.parameters(),
                                         self.cfg_rl.grad_clip)
                self.opt_actor.step()

                self.opt_critic.zero_grad()
                (0.5 * loss_dict["critic"]).backward()
                nn.utils.clip_grad_norm_(self.agent.critic.parameters(),
                                         self.cfg_rl.grad_clip)
                self.opt_critic.step()

            mean_r  = float(np.mean(rewards)) if rewards else 0.0
            mean_sr = float(np.mean(srs)) * 100 if srs else 0.0
            print(f"Epoch {epoch:03d}/{self.cfg_rl.epochs} | "
                  f"R={mean_r:.4f} | SR={mean_sr:.1f}%")

            if mean_r > best_reward:
                best_reward = mean_r
                self._save("best")
                print(f"  ✅ best={best_reward:.4f}")

            self.sched_actor.step()
            self.sched_critic.step()

        self._save("final")
        print(f"\n✅ Done | best={best_reward:.4f}")

    # ── Eval ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, n_batches: int = 20):
        print(f"\n[NoAE Eval] {n_batches} batches …")
        self.agent.eval()

        all_x, all_xcf, all_yh, all_ycf = [], [], [], []

        for i, batch in enumerate(self.test_loader):
            if i >= n_batches:
                break
            ep = run_episode_eval(
                batch=batch,
                ae_arch=self.ae_arch,
                forecaster=self.forecaster,
                agent=self.agent,
                reward_fn=self.reward_fn,
                device=self.device,
                use_rl=True,
                mask_last_k=self.mask_last_k,
                mask_ramp_k=self.mask_ramp_k,
                filter_quantile=self.filter_q,
            )
            if ep is None:
                continue
            all_x.append(ep["x_ot"].cpu().numpy())
            all_xcf.append(ep["x_cf"].cpu().numpy())
            all_yh.append(ep["y_hat"].cpu().numpy())
            all_ycf.append(ep["y_cf"].cpu().numpy())

        if not all_x:
            raise RuntimeError("No valid CFs in eval.")

        X    = np.concatenate(all_x,   axis=0)
        Xcf  = np.concatenate(all_xcf, axis=0)
        Yh   = np.concatenate(all_yh,  axis=0)
        Ycf  = np.concatenate(all_ycf, axis=0)

        alphas, betas = self._compute_bounds_np(X, Yh)
        summary = self.evaluator.evaluate(
            X_orig=X, X_cf=Xcf, Y_hat=Yh, Y_cf=Ycf,
            alphas=alphas, betas=betas,
        )
        print("\n── NoAE Evaluation ──────────────────────────────────")
        CounterfactualEvaluator.print_table(summary, label=self.exp_name)
        return summary

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _run_episode_train(self, batch):
        batch_x, _, batch_x_mark, _ = batch
        batch_x      = batch_x.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        x_ot = batch_x[:, :, -1:]

        with torch.no_grad():
            z     = self.ae_arch.encode(x_ot)   # identity: z = x_ot squeezed
            y_hat = self.forecaster.predict_ot(batch_x, batch_x_mark)

        mask_keep = filter_batch(y_hat, self.filter_q)
        if mask_keep.sum() == 0:
            return None

        x_ot         = x_ot[mask_keep]
        batch_x      = batch_x[mask_keep]
        batch_x_mark = batch_x_mark[mask_keep]
        y_hat        = y_hat[mask_keep]
        z            = z[mask_keep]

        s, log_prob, _ = self.agent.actor.sample(self.agent.build_state(z, y_hat))
        z_cf = torch.clamp(z + self.agent.eta * s, -1.0, 1.0)

        x_prop    = self.ae_arch.decode(z_cf)
        delta     = x_prop - x_ot
        temp_mask = build_temporal_mask(
            x_ot.shape[0], x_ot.shape[1], x_ot.shape[2],
            self.mask_last_k, self.mask_ramp_k, self.device,
        )
        x_cf = x_ot + temp_mask * delta

        with torch.no_grad():
            y_cf = self.forecaster.predict_from_ot(
                x_ot=x_cf, x_full=batch_x, x_mark=batch_x_mark)

        reward_dict = self.reward_fn(x_ot, x_cf, y_hat, y_cf, z_cf=z_cf)
        value       = self.agent.evaluate(self.agent.build_state(z, y_hat))
        mu, log_std = self.agent.actor(self.agent.build_state(z, y_hat))
        entropy     = torch.distributions.Normal(mu, log_std.exp()).entropy().sum(-1)

        return dict(
            log_prob=log_prob, reward=reward_dict["total"],
            value=value, entropy=entropy, reward_dict=reward_dict,
        )

    def _compute_bounds_np(self, x_ot, y_hat):
        rho, fr   = self.reward_fn.rho, self.reward_fn.fr
        direction = self.reward_fn.direction
        y2d       = y_hat[:, :, 0]
        sigma     = np.full(x_ot.shape[0], self._global_sigma)
        gap       = (rho * sigma)[:, np.newaxis]
        width     = (fr  * sigma)[:, np.newaxis]
        if direction < 0:
            betas  = y2d - gap
            alphas = betas - width
        else:
            alphas = y2d + gap
            betas  = alphas + width
        return alphas, betas

    def _save(self, tag: str):
        path = os.path.join(self.ckpt_dir, f"{self.exp_name}_agent_{tag}.pt")
        torch.save({
            "actor_state_dict":  self.agent.actor.state_dict(),
            "critic_state_dict": self.agent.critic.state_dict(),
        }, path)
        print(f"[NoAE] Checkpoint → {path}")


# ──────────────────────────────────────────────────────────────────────────────
def run_one_seed(seed: int, cfg_f, cfg_rl, device, eval_batches: int):
    set_seed(seed)
    print(f"\n{'='*65}")
    print(f"  RL-MCF No-AE Ablation  |  Seed={seed}")
    print(f"{'='*65}")

    ckpt_dir = os.path.join(OUTPUT_DIR, "itransformer", f"seed_{seed}", "checkpoints")
    trainer  = RLNoAETrainer(cfg_f, cfg_rl, device, ckpt_dir)
    trainer.train()
    summary = trainer.evaluate(n_batches=eval_batches)
    return summary


def aggregate(results: list) -> dict:
    agg = {}
    for metric in TABLE_METRICS:
        vals = []
        for r in results:
            if r is None: continue
            v = r.get(metric, {})
            vals.append(v.get("mean", float("nan")) if isinstance(v, dict) else float(v))
        vals = [v for v in vals if not np.isnan(v)]
        agg[metric] = {
            "mean_across_seeds": float(np.mean(vals)) if vals else float("nan"),
            "std_across_seeds":  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "per_seed_values":   vals,
        }
    return agg


def print_comparison(with_ae: dict, without_ae: dict):
    col = 16
    print(f"\n{'='*80}")
    print("  ABLATION — AE vs No-AE  (ETTh1 / iTransformer)")
    print(f"{'='*80}")
    print(f"  {'Metric':<26} {'With AE':>{col}} {'No AE':>{col}} {'Δ':>{col}}")
    print("-" * 80)
    for m in TABLE_METRICS:
        mu_ae    = with_ae.get(m, {}).get("mean_across_seeds", float("nan"))
        mu_noae  = without_ae.get(m, {}).get("mean_across_seeds", float("nan"))
        std_noae = without_ae.get(m, {}).get("std_across_seeds", 0.0)
        delta    = mu_noae - mu_ae
        sign     = "↑" if delta > 0 else "↓"
        print(f"  {m:<26} {mu_ae:>{col}.4f} "
              f"{mu_noae:>{col}.4f}±{std_noae:.3f}  {delta:+.4f} {sign}")
    print(f"{'='*80}")
    print("  Plausibility: LOWER = more realistic (DOWN is better)")
    print("  If No-AE plausibility > With-AE → AE role justified.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ablation: RL-MCF without autoencoder (ICONIP revision)"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--eval_batches", type=int, default=EVAL_BATCHES)
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, just evaluate existing checkpoints")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cfg_f  = load_config(FORECAST_CONFIG)
    cfg_rl = load_config(RL_CONFIG)
    device = get_device(cfg_f)

    # Override cfg_rl name to avoid collision with main checkpoints
    cfg_rl.name = "rl_cf_itransformer_noae_etth1"

    seed_results = []
    for seed in args.seeds:
        if args.eval_only:
            # Load existing checkpoint and evaluate
            ckpt_dir = os.path.join(OUTPUT_DIR, "itransformer", f"seed_{seed}", "checkpoints")
            ckpt_path = os.path.join(ckpt_dir, f"rl_cf_itransformer_noae_etth1_agent_best.pt")
            if not os.path.exists(ckpt_path):
                print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
                seed_results.append(None)
                continue
            set_seed(seed)
            trainer = RLNoAETrainer(cfg_f, cfg_rl, device, ckpt_dir)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            trainer.agent.actor.load_state_dict(ckpt["actor_state_dict"])
            trainer.agent.critic.load_state_dict(ckpt["critic_state_dict"])
            summary = trainer.evaluate(n_batches=args.eval_batches)
        else:
            summary = run_one_seed(seed, cfg_f, cfg_rl, device, args.eval_batches)

        out_path = os.path.join(OUTPUT_DIR, "itransformer", f"seed_{seed}_evaluation.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        seed_results.append(summary)

    # Aggregate No-AE results
    noae_agg = aggregate(seed_results)

    # Load reference With-AE results from multiseed experiment (if available)
    with_ae_path = "assets/results/multiseed/itransformer/aggregated.json"
    with_ae_agg = None
    if os.path.exists(with_ae_path):
        with open(with_ae_path) as f:
            with_ae_agg = json.load(f)
    else:
        print(f"\n[Warning] With-AE reference not found at {with_ae_path}")
        print("  Run run_multiseed_table5.py first, or comparison will be skipped.")

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"no_ae": noae_agg, "with_ae": with_ae_agg}, f, indent=2)
    print(f"\nSummary saved → {summary_path}")

    # Print comparison table
    if with_ae_agg:
        print_comparison(with_ae_agg, noae_agg)
    else:
        print("\nNo-AE results:")
        for m, v in noae_agg.items():
            print(f"  {m}: {v['mean_across_seeds']:.4f} ± {v['std_across_seeds']:.4f}")


if __name__ == "__main__":
    main()

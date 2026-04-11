import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from src.models.autoencoder.tcn_ae import TCNAutoEncoder
from src.models.RL_learned_mask.agent import ActorCriticMask
from src.models.RL_learned_mask.reward import CFReward
from src.models.forecaster_wrapper import ForecasterWrapper


def prepare_rl_data(cfg_forecaster, cfg_ae, device):
    from src.data_provider.data_factory import data_provider

    _, train_loader = data_provider(cfg_forecaster, "train")
    _, test_loader = data_provider(cfg_forecaster, "test")

    ckpt_ae = torch.load(
        cfg_ae.checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    scaler = None
    if "scaler_mean" in ckpt_ae and "scaler_std" in ckpt_ae:
        scaler = StandardScaler()
        scaler.mean_ = np.array(ckpt_ae["scaler_mean"], dtype=np.float64)
        scaler.scale_ = np.array(ckpt_ae["scaler_std"], dtype=np.float64)
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = (
            len(scaler.mean_) if np.ndim(scaler.mean_) > 0 else 1
        )
        print("[Data] Scaler loaded from AE checkpoint ✓")
    else:
        print("[Warning] 'scaler_mean' / 'scaler_std' not found in AE checkpoint")
        print("[Warning] Continuing with scaler = None")

    return train_loader, test_loader, scaler


def filter_batch(y_hat, quantile=0.75):
    mean_yhat = y_hat[:, :, 0].mean(dim=1)
    return mean_yhat >= torch.quantile(mean_yhat, quantile)


def _ensure_mask_shape(mask_t, x_ot, mask_min=0.05, mask_max=0.95):
    """
    mask_t attendu en [B,T] ou [B,T,1]
    retourne [B,T,1]
    """
    if mask_t.dim() == 2:
        mask_t = mask_t.unsqueeze(2)
    elif mask_t.dim() != 3:
        raise ValueError(f"mask_t must have shape [B,T] or [B,T,1], got {mask_t.shape}")

    if mask_t.shape[1] != x_ot.shape[1]:
        raise ValueError(
            f"Mask time dimension mismatch: mask_t.shape={mask_t.shape}, x_ot.shape={x_ot.shape}"
        )

    mask_t = torch.clamp(mask_t, min=mask_min, max=mask_max)
    return mask_t


def run_episode(
    batch,
    ae,
    forecaster,
    agent,
    reward_fn,
    device,
    filter_quantile=0.75,
    alpha_hf=0.2,
    mask_min=0.05,
    mask_max=0.95,
):
    batch_x, _, batch_x_mark, _ = batch
    batch_x = batch_x.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    x_ot = batch_x[:, :, -1:]   # (B, seq_len, 1)

    with torch.no_grad():
        z = ae.encode(x_ot)
        y_hat = forecaster.predict_ot(batch_x, batch_x_mark)

    mask_keep = filter_batch(y_hat, filter_quantile)
    if mask_keep.sum() == 0:
        return None

    x_ot = x_ot[mask_keep]
    batch_x = batch_x[mask_keep]
    batch_x_mark = batch_x_mark[mask_keep]
    y_hat = y_hat[mask_keep]
    z = z[mask_keep]

    # état
    s = agent.build_state(z, y_hat)

    # IMPORTANT:
    # on suppose que actor.sample(s) retourne: a, log_prob, mu, mask_t
    a, log_prob, _, mask_t = agent.actor.sample(s)

    # perturbation latente
    z_cf = torch.clamp(z + agent.eta * a, -1.0, 1.0)

    # proposition décodée
    x_prop = ae.decode(z_cf)
    delta = x_prop - x_ot

    # HF skip
    with torch.no_grad():
        x_lf = ae.decode(ae.encode(x_ot))
    x_hf = x_ot - x_lf

    mask_t = _ensure_mask_shape(
        mask_t=mask_t,
        x_ot=x_ot,
        mask_min=mask_min,
        mask_max=mask_max,
    )

    # modification apprise localisée
    masked_delta = delta + alpha_hf * x_hf.detach()
    x_cf = x_ot + mask_t * masked_delta

    with torch.no_grad():
        y_cf = forecaster.predict_from_ot(
            x_ot=x_cf,
            x_full=batch_x,
            x_mark=batch_x_mark
        )

    reward_dict = reward_fn(
        x_ot, x_cf, y_hat, y_cf,
        mask_t=mask_t,
        z_cf=z_cf
    )
    R = reward_dict["total"]
    V = agent.evaluate(s)

    # entropie depuis l'actor
    mu, log_std, _mask_det = agent.actor(s)
    entropy = torch.distributions.Normal(
        mu, log_std.exp()
    ).entropy().sum(dim=-1)

    return {
        "log_prob": log_prob,
        "reward": R,
        "value": V,
        "entropy": entropy,
        "reward_dict": reward_dict,
        "x_ot": x_ot.detach(),
        "x_cf": x_cf.detach(),
        "y_hat": y_hat.detach(),
        "y_cf": y_cf.detach(),
        "z": z.detach(),
        "z_cf": z_cf.detach(),
        "mask_t": mask_t.detach(),
        "n_valid": int(mask_keep.sum()),
    }


class RLLearnedMaskTrainer:
    def __init__(self, cfg_forecaster, cfg_ae, cfg_rl, device):
        self.cfg_f = cfg_forecaster
        self.cfg_ae = cfg_ae
        self.cfg_rl = cfg_rl
        self.device = device

        self.alpha_hf = getattr(cfg_rl, "alpha_hf", 0.2)
        self.mask_min = getattr(cfg_rl, "mask_min", 0.05)
        self.mask_max = getattr(cfg_rl, "mask_max", 0.95)

        for d in [
            cfg_rl.checkpoint_dir_lp,
            cfg_rl.figures_dir_lp,
            cfg_rl.results_dir_lp,
        ]:
            os.makedirs(d, exist_ok=True)

        print("\n[RL-Mask-Learned] Loading frozen models ...")

        self.ae = TCNAutoEncoder.from_checkpoint(
            cfg_ae.checkpoint_path, device=device
        )
        self.ae.eval()
        for p in self.ae.parameters():
            p.requires_grad_(False)

        self.forecaster = ForecasterWrapper(cfg_forecaster, device)
        for p in self.forecaster.model.parameters():
            p.requires_grad_(False)

        self.train_loader, self.test_loader, self.scaler = prepare_rl_data(
            cfg_forecaster, cfg_ae, device
        )

        # state_dim = latent_dim + 6 + 1
        state_dim = cfg_ae.latent_dim + 6 + 1
        seq_len = cfg_forecaster.seq_len

        self.agent = ActorCriticMask(
            state_dim=state_dim,
            latent_dim=cfg_ae.latent_dim,
            seq_len=seq_len,
            eta=cfg_rl.eta,
            entropy_coef=cfg_rl.entropy_coef,
            direction=getattr(cfg_rl, "direction", -1.0),
        ).to(device)

        self.reward_fn = CFReward(
            ae=self.ae,
            rho=cfg_rl.rho,
            alpha=cfg_rl.alpha,
            beta=cfg_rl.beta,
            gamma=getattr(cfg_rl, "gamma", 0.8),
            tau=getattr(cfg_rl, "tau", 0.6),
            lambda_sparse=getattr(cfg_rl, "lambda_sparse", 0.4),
            lambda_tv=getattr(cfg_rl, "lambda_tv", 0.2),
        ).to(device)

        print(f"[RL-Mask-Learned] Objectif       : réduire mean(forecast) de {cfg_rl.rho * 100:.0f}%")
        print(f"[RL-Mask-Learned] eta            = {cfg_rl.eta}")
        print(f"[RL-Mask-Learned] alpha_hf       = {self.alpha_hf}")
        print(f"[RL-Mask-Learned] lambda_sparse  = {getattr(cfg_rl, 'lambda_sparse', 0.4)}")
        print(f"[RL-Mask-Learned] lambda_tv      = {getattr(cfg_rl, 'lambda_tv', 0.2)}")
        print(f"[RL-Mask-Learned] mask bounds    = [{self.mask_min}, {self.mask_max}]")

        params = self.agent.count_parameters()
        print(f"[RL-Mask-Learned] Actor {params['actor']:,}  Critic {params['critic']:,}")

        self.opt_actor = torch.optim.Adam(
            self.agent.actor.parameters(),
            lr=cfg_rl.lr_actor
        )
        self.opt_critic = torch.optim.Adam(
            self.agent.critic.parameters(),
            lr=getattr(cfg_rl, "lr_critic", 1e-4)
        )

        self.history = {k: [] for k in [
            "loss_total", "loss_actor", "loss_critic", "reward_total",
            "r_validity", "r_proximity", "r_reconstruction",
            "r_temporal", "r_mask_sparsity", "r_mask_smoothness",
            "delta_mean", "success_rate", "advantage", "n_valid"
        ]}

    def train(self):
        print(f"\n{'=' * 60}")
        print(
            f"RL-Mask-Learned Training | {self.cfg_rl.epochs} epochs | "
            f"rho={self.cfg_rl.rho} | eta={self.cfg_rl.eta}"
        )
        print(f"{'=' * 60}")

        best_reward = -float("inf")

        for epoch in range(1, self.cfg_rl.epochs + 1):
            t0 = time.time()
            self.agent.train()
            stats = {k: [] for k in self.history}

            for batch in self.train_loader:
                ep = run_episode(
                    batch=batch,
                    ae=self.ae,
                    forecaster=self.forecaster,
                    agent=self.agent,
                    reward_fn=self.reward_fn,
                    device=self.device,
                    alpha_hf=self.alpha_hf,
                    mask_min=self.mask_min,
                    mask_max=self.mask_max,
                )
                if ep is None:
                    continue

                loss_dict = self.agent.compute_loss(
                    ep["log_prob"], ep["reward"],
                    ep["value"], ep["entropy"]
                )

                self.opt_actor.zero_grad()
                loss_dict["actor"].backward(retain_graph=True)
                nn.utils.clip_grad_norm_(
                    self.agent.actor.parameters(), self.cfg_rl.grad_clip
                )
                self.opt_actor.step()

                self.opt_critic.zero_grad()
                (0.5 * loss_dict["critic"]).backward()
                nn.utils.clip_grad_norm_(
                    self.agent.critic.parameters(), self.cfg_rl.grad_clip
                )
                self.opt_critic.step()

                rs = self.reward_fn.stats(ep["reward_dict"])

                stats["loss_total"].append(float(loss_dict["total"].item()))
                stats["loss_actor"].append(float(loss_dict["actor"].item()))
                stats["loss_critic"].append(float(loss_dict["critic"].item()))
                stats["reward_total"].append(rs["total"])
                stats["r_validity"].append(rs["validity"])
                stats["r_proximity"].append(rs["proximity"])
                stats["r_reconstruction"].append(rs["reconstruction"])
                stats["r_temporal"].append(rs["temporal"])
                stats["r_mask_sparsity"].append(rs["mask_sparsity"])
                stats["r_mask_smoothness"].append(rs["mask_smoothness"])
                stats["delta_mean"].append(rs["delta_mean"])
                stats["success_rate"].append(rs["success"])
                stats["advantage"].append(loss_dict["advantage"])
                stats["n_valid"].append(ep["n_valid"])

            means = {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}
            for k, v in means.items():
                self.history[k].append(v)

            print(
                f"Epoch {epoch:03d}/{self.cfg_rl.epochs} | "
                f"R={means['reward_total']:.4f} | "
                f"val={means['r_validity']:.3f} | "
                f"rec={means['r_reconstruction']:.3f} | "
                f"temp={means['r_temporal']:.3f} | "
                f"msp={means['r_mask_sparsity']:.3f} | "
                f"mtv={means['r_mask_smoothness']:.3f} | "
                f"Δ={means['delta_mean']:.4f} | "
                f"SR={means['success_rate'] * 100:.1f}% | "
                f"prox={means['r_proximity']:.3f} | "
                f"{time.time() - t0:.1f}s"
            )

            if means["reward_total"] > best_reward:
                best_reward = means["reward_total"]
                self._save_checkpoint("best")
                print(f"  ✅ best = {best_reward:.4f}")

        self._save_checkpoint("final")
        self._save_history()
        self._plot_training()
        print(f"\n✅ Done | best reward = {best_reward:.4f}")
        return self.history

    @torch.no_grad()
    def evaluate(self, n_batches=20):
        print(f"\n[Eval] Test set ({n_batches} batches) ...")
        self.agent.eval()

        stats = {k: [] for k in [
            "total", "validity", "proximity", "reconstruction",
            "temporal", "mask_sparsity", "mask_smoothness",
            "delta_mean", "success"
        ]}
        cf_examples = []

        for i, batch in enumerate(self.test_loader):
            if i >= n_batches:
                break

            ep = run_episode(
                batch=batch,
                ae=self.ae,
                forecaster=self.forecaster,
                agent=self.agent,
                reward_fn=self.reward_fn,
                device=self.device,
                alpha_hf=self.alpha_hf,
                mask_min=self.mask_min,
                mask_max=self.mask_max,
            )
            if ep is None:
                continue

            rs = self.reward_fn.stats(ep["reward_dict"])
            for k in stats:
                stats[k].append(rs[k])

            if len(cf_examples) < 4:
                cf_examples.append({
                    "x_ot": ep["x_ot"][0].cpu().numpy(),
                    "x_cf": ep["x_cf"][0].cpu().numpy(),
                    "y_hat": ep["y_hat"][0].cpu().numpy(),
                    "y_cf": ep["y_cf"][0].cpu().numpy(),
                    "mask_t": ep["mask_t"][0].cpu().numpy(),
                })

        print(f"\n── Résultats RL-Mask-Learned ───────────────────────────")
        for k, v in stats.items():
            print(f"  {k:20s} = {np.mean(v):.4f}")
        print(f"  Success = {np.mean(stats['success']) * 100:.1f}%")

        self._plot_cf_examples(cf_examples)
        return cf_examples

    def _save_checkpoint(self, tag):
        torch.save({
            "actor_state_dict": self.agent.actor.state_dict(),
            "critic_state_dict": self.agent.critic.state_dict(),
            "history": self.history,
            "cfg_rl": vars(self.cfg_rl),
        }, os.path.join(
            self.cfg_rl.checkpoint_dir_lp,
            f"rl_mask_learned_agent_{tag}.pt"
        ))

    def _save_history(self):
        path = os.path.join(
            self.cfg_rl.results_dir_lp,
            "rl_mask_learned_history.json"
        )
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[RL-Mask-Learned] History → {path}")

    def _plot_training(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))

        axes[0, 0].plot(self.history["reward_total"], lw=1.5)
        axes[0, 0].set_title("Total Reward")
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(self.history["r_validity"], lw=1.5, label="validity")
        axes[0, 1].plot(self.history["r_proximity"], lw=1.5, label="proximity")
        axes[0, 1].plot(self.history["r_reconstruction"], lw=1.5, label="reconstruction")
        axes[0, 1].plot(self.history["r_temporal"], lw=1.5, label="temporal")
        axes[0, 1].set_title("Core Sub-Rewards")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        axes[0, 2].plot(self.history["r_mask_sparsity"], lw=1.5, label="mask_sparsity")
        axes[0, 2].plot(self.history["r_mask_smoothness"], lw=1.5, label="mask_smoothness")
        axes[0, 2].set_title("Mask Rewards")
        axes[0, 2].legend()
        axes[0, 2].grid(alpha=0.3)

        axes[1, 0].plot(self.history["loss_actor"], lw=1.5, label="actor")
        axes[1, 0].plot(self.history["loss_critic"], lw=1.5, label="critic")
        axes[1, 0].set_title("Losses")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].plot(self.history["delta_mean"], lw=1.5, label="Δmean")
        axes[1, 1].plot(self.history["success_rate"], lw=1.5, linestyle="--", label="SR")
        axes[1, 1].axhline(self.cfg_rl.rho, linestyle=":", lw=1.2)
        axes[1, 1].set_title("Δmean & SR")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        axes[1, 2].plot(self.history["n_valid"], lw=1.5)
        axes[1, 2].set_title("Samples valides / epoch")
        axes[1, 2].grid(alpha=0.3)

        plt.suptitle(
            f"RL Mask Learned — rho={self.cfg_rl.rho} | eta={self.cfg_rl.eta}",
            fontsize=13
        )
        plt.tight_layout()
        path = os.path.join(
            self.cfg_rl.figures_dir_lp,
            "rl_mask_learned_training_curves.png"
        )
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[RL-Mask-Learned] Curves → {path}")

    def _plot_cf_examples(self, examples):
        if not examples:
            return

        n = len(examples)
        fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n))
        if n == 1:
            axes = np.array([axes])

        for i, ex in enumerate(examples):
            x_ot = ex["x_ot"][:, 0]
            x_cf = ex["x_cf"][:, 0]
            y_hat = ex["y_hat"][:, 0]
            y_cf = ex["y_cf"][:, 0]
            mask_t = ex["mask_t"][:, 0] if ex["mask_t"].ndim == 2 else ex["mask_t"]

            full_orig = np.concatenate([x_ot, y_hat])
            full_cf = np.concatenate([x_cf, y_cf])
            t_all = np.arange(len(full_orig))
            reduction = ((y_hat.mean() - y_cf.mean()) / (abs(y_hat.mean()) + 1e-8) * 100)
            ok = "✓" if reduction >= self.cfg_rl.rho * 100 else "✗"

            axes[i, 0].plot(t_all, full_orig, lw=1.5, label="x+forecast(x)")
            axes[i, 0].plot(t_all, full_cf, lw=1.5, ls="--", label="x_cf+forecast(x_cf)")
            axes[i, 0].axvline(len(x_ot), linestyle="--", lw=1.2)
            axes[i, 0].fill_between(t_all, full_orig, full_cf, alpha=0.12)
            axes[i, 0].set_title(f"Sample {i+1} — {reduction:+.1f}% {ok}", fontsize=11)
            axes[i, 0].legend(fontsize=9)
            axes[i, 0].grid(alpha=0.3)

            axes[i, 1].plot(mask_t, lw=1.8)
            axes[i, 1].set_ylim(-0.05, 1.05)
            axes[i, 1].set_title("Learned temporal mask", fontsize=11)
            axes[i, 1].grid(alpha=0.3)

        plt.suptitle("CF RL-Mask-Learned — ETTh1", fontsize=13)
        plt.tight_layout()
        path = os.path.join(
            self.cfg_rl.figures_dir_lp,
            "rl_mask_learned_cf_examples.png"
        )
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[RL-Mask-Learned] CF examples → {path}")


if __name__ == "__main__":
    from src.utils.config import load_config
    from src.utils.train_tools import get_device

    CONFIG_FORECASTER = "assets/configs/models/etth1_dataset/itransformer/etth1_96_48_S.json"
    CONFIG_AE = "assets/configs/models/etth1_dataset/ae/tcn_ae.json"
    CONFIG_RL = "assets/configs/models/etth1_dataset/RL/rl_learned_mask.json"

    cfg_f = load_config(CONFIG_FORECASTER)
    cfg_ae = load_config(CONFIG_AE)
    cfg_rl = load_config(CONFIG_RL)

    device = get_device(cfg_f)

    print(f"Device   : {device}")
    print("Pipeline : RL Mask Learned")

    trainer = RLLearnedMaskTrainer(cfg_f, cfg_ae, cfg_rl, device)
    history = trainer.train()
    examples = trainer.evaluate(n_batches=20)

    print("\n✅ Done !")
    print(f"  Checkpoint → {cfg_rl.checkpoint_dir_lp}/rl_learned_mask_agent_best.pt")
    print(f"  Figures    → {cfg_rl.figures_dir_lp}/")
    print(f"  Results    → {cfg_rl.results_dir_lp}/rl_learned_mask_history.json")
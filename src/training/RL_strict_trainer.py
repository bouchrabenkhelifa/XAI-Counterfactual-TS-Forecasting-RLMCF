import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from src.models.tcn_ae import TCNAutoEncoder
from src.models.RL_strict.agent import ActorCritic
from src.models.RL_strict.reward import CFReward
from src.models.RL_strict.latent_plausibility import LatentPlausibility
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


def load_latent_plausibility(path, ae, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    plaus = LatentPlausibility(
        ae=ae,
        alpha=ckpt.get("alpha", 1.0),
        scale=ckpt.get("scale", 8.0),
        latent_weight=ckpt.get("latent_weight", 0.65),
        recon_weight=ckpt.get("recon_weight", 0.35),
    )

    plaus.z_mean = ckpt["z_mean"].to(device)
    plaus.z_std = ckpt["z_std"].to(device)
    plaus.sigma = ckpt.get("sigma", None)
    plaus.fitted_ = True

    print("[RL-Strict] LatentPlausibility loaded ✓")
    return plaus


def filter_batch(y_hat, quantile=0.75):
    mean_yhat = y_hat[:, :, 0].mean(dim=1)
    return mean_yhat >= torch.quantile(mean_yhat, quantile)


def run_episode(
    batch,
    ae,
    forecaster,
    agent,
    reward_fn,
    device,
    filter_quantile=0.75,
    alpha_hf=0.2,
):
    batch_x, _, batch_x_mark, _ = batch
    batch_x = batch_x.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    x_ot = batch_x[:, :, -1:]   # (B, seq_len, 1)

    with torch.no_grad():
        z = ae.encode(x_ot)
        y_hat = forecaster.predict_ot(batch_x, batch_x_mark)

    mask = filter_batch(y_hat, filter_quantile)
    if mask.sum() == 0:
        return None

    x_ot = x_ot[mask]
    batch_x = batch_x[mask]
    batch_x_mark = batch_x_mark[mask]
    y_hat = y_hat[mask]
    z = z[mask]

    s = agent.build_state(z, y_hat)
    a, log_prob, _ = agent.actor.sample(s)
    z_cf = torch.clamp(z + agent.eta * a, -1.0, 1.0)

    x_cf = ae.decode(z_cf)

    # HF skip plus prudent
    with torch.no_grad():
        x_lf = ae.decode(ae.encode(x_ot))
    x_hf = x_ot - x_lf
    x_cf = x_cf + alpha_hf * x_hf.detach()

    with torch.no_grad():
        y_cf = forecaster.predict_from_ot(
            x_ot=x_cf,
            x_full=batch_x,
            x_mark=batch_x_mark
        )

    reward_dict = reward_fn(x_ot, x_cf, y_hat, y_cf, z_cf=z_cf)
    R = reward_dict["total"]
    V = agent.evaluate(s)

    mu, log_std = agent.actor(s)
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
        "n_valid": int(mask.sum()),
    }


class RLStrictTrainer:
    def __init__(self, cfg_forecaster, cfg_ae, cfg_rl, device):
        self.cfg_f = cfg_forecaster
        self.cfg_ae = cfg_ae
        self.cfg_rl = cfg_rl
        self.device = device

        self.alpha_hf = getattr(cfg_rl, "alpha_hf", 0.2)

        for d in [
            cfg_rl.checkpoint_dir_lp,
            cfg_rl.figures_dir_lp,
            cfg_rl.results_dir_lp,
        ]:
            os.makedirs(d, exist_ok=True)

        print("\n[RL-Strict] Loading frozen models ...")

        self.ae = TCNAutoEncoder.from_checkpoint(
            cfg_ae.checkpoint_path, device=device
        )
        self.ae.eval()
        for p in self.ae.parameters():
            p.requires_grad_(False)

        self.forecaster = ForecasterWrapper(cfg_forecaster, device)
        for p in self.forecaster.model.parameters():
            p.requires_grad_(False)

        self.plausibility = load_latent_plausibility(
            cfg_rl.latent_plaus_path, self.ae, device
        )

        self.train_loader, self.test_loader, self.scaler = prepare_rl_data(
            cfg_forecaster, cfg_ae, device
        )

        self.reward_fn = CFReward(
            plausibility=self.plausibility,
            rho=cfg_rl.rho,
            alpha=cfg_rl.alpha,
            beta=cfg_rl.beta,
            gamma=getattr(cfg_rl, "gamma", 0.6),
            kappa=getattr(cfg_rl, "kappa", 0.2),
        ).to(device)

        print(f"[RL-Strict] Objectif  : réduire mean(forecast) de {cfg_rl.rho * 100:.0f}%")
        print(f"[RL-Strict] gamma     = {getattr(cfg_rl, 'gamma', 0.6)}")
        print(f"[RL-Strict] kappa     = {getattr(cfg_rl, 'kappa', 0.2)}")
        print(f"[RL-Strict] alpha_hf  = {self.alpha_hf}")

        self.agent = ActorCritic(
            latent_dim=cfg_ae.latent_dim,
            pred_len=cfg_forecaster.pred_len,
            eta=cfg_rl.eta,
            entropy_coef=cfg_rl.entropy_coef,
            direction=-1.0,
        ).to(device)

        params = self.agent.count_parameters()
        print(f"[RL-Strict] Actor {params['actor']:,}  Critic {params['critic']:,}")

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
            "r_validity", "r_proximity", "r_plausibility",
            "r_plausibility_latent", "r_plausibility_recon",
            "r_latent_drift",
            "delta_mean", "success_rate", "advantage", "n_valid"
        ]}

    def train(self):
        print(f"\n{'=' * 60}")
        print(
            f"RL-Strict Training | {self.cfg_rl.epochs} epochs | "
            f"rho={self.cfg_rl.rho} | eta={self.cfg_rl.eta} | "
            f"gamma={getattr(self.cfg_rl, 'gamma', 0.6)} | "
            f"kappa={getattr(self.cfg_rl, 'kappa', 0.2)} | "
            f"alpha_hf={self.alpha_hf}"
        )
        print(f"{'=' * 60}")

        best_reward = -float("inf")

        for epoch in range(1, self.cfg_rl.epochs + 1):
            t0 = time.time()
            self.agent.train()
            stats = {k: [] for k in self.history}

            for batch in self.train_loader:
                ep = run_episode(
                    batch, self.ae, self.forecaster,
                    self.agent, self.reward_fn, self.device,
                    alpha_hf=self.alpha_hf
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
                stats["r_plausibility"].append(rs["plausibility"])
                stats["r_plausibility_latent"].append(rs["plausibility_latent"])
                stats["r_plausibility_recon"].append(rs["plausibility_recon"])
                stats["r_latent_drift"].append(rs["latent_drift"])
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
                f"pl={means['r_plausibility']:.3f} "
                f"(lat={means['r_plausibility_latent']:.3f}, "
                f"rec={means['r_plausibility_recon']:.3f}) | "
                f"drift={means['r_latent_drift']:.3f} | "
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
            "total", "validity", "proximity", "plausibility",
            "plausibility_latent", "plausibility_recon",
            "latent_drift", "delta_mean", "success"
        ]}
        cf_examples = []

        for i, batch in enumerate(self.test_loader):
            if i >= n_batches:
                break

            ep = run_episode(
                batch, self.ae, self.forecaster,
                self.agent, self.reward_fn, self.device,
                alpha_hf=self.alpha_hf
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
                })

        print(f"\n── Résultats RL-Strict ───────────────────────────")
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
            f"rl_strict_agent_{tag}.pt"
        ))

    def _save_history(self):
        path = os.path.join(
            self.cfg_rl.results_dir_lp,
            "rl_strict_history.json"
        )
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[RL-Strict] History → {path}")

    def _plot_training(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))

        axes[0, 0].plot(self.history["reward_total"], lw=1.5)
        axes[0, 0].set_title("Total Reward")
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(self.history["r_validity"], lw=1.5, label="validity")
        axes[0, 1].plot(self.history["r_proximity"], lw=1.5, label="proximity")
        axes[0, 1].plot(self.history["r_plausibility"], lw=1.5, label="plausibility")
        axes[0, 1].set_title("Sub-Rewards")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        axes[0, 2].plot(self.history["r_plausibility_latent"], lw=1.5, label="pl_latent")
        axes[0, 2].plot(self.history["r_plausibility_recon"], lw=1.5, label="pl_recon")
        axes[0, 2].plot(self.history["r_latent_drift"], lw=1.5, label="drift")
        axes[0, 2].set_title("Strict Plausibility Terms")
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
            f"RL Strict — rho={self.cfg_rl.rho} | "
            f"eta={self.cfg_rl.eta} | "
            f"gamma={getattr(self.cfg_rl, 'gamma', 0.6)} | "
            f"kappa={getattr(self.cfg_rl, 'kappa', 0.2)}",
            fontsize=13
        )
        plt.tight_layout()
        path = os.path.join(
            self.cfg_rl.figures_dir_lp,
            "rl_strict_training_curves.png"
        )
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[RL-Strict] Curves → {path}")

    def _plot_cf_examples(self, examples):
        if not examples:
            return

        n = len(examples)
        fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
        if n == 1:
            axes = [axes]

        for i, ex in enumerate(examples):
            x_ot = ex["x_ot"][:, 0]
            x_cf = ex["x_cf"][:, 0]
            y_hat = ex["y_hat"][:, 0]
            y_cf = ex["y_cf"][:, 0]

            full_orig = np.concatenate([x_ot, y_hat])
            full_cf = np.concatenate([x_cf, y_cf])
            t_all = np.arange(len(full_orig))
            reduction = ((y_hat.mean() - y_cf.mean()) / (abs(y_hat.mean()) + 1e-8) * 100)
            ok = "✓" if reduction >= self.cfg_rl.rho * 100 else "✗"

            axes[i].plot(t_all, full_orig, lw=1.5, label="x+forecast(x)")
            axes[i].plot(t_all, full_cf, lw=1.5, ls="--", label="x_cf+forecast(x_cf)")
            axes[i].axvline(len(x_ot), linestyle="--", lw=1.2)
            axes[i].fill_between(t_all, full_orig, full_cf, alpha=0.12)
            axes[i].set_title(
                f"Sample {i+1} — {reduction:+.1f}% {ok} (strict)",
                fontsize=11
            )
            axes[i].legend(fontsize=9)
            axes[i].grid(alpha=0.3)

        plt.suptitle("CF RL-Strict — ETTh1", fontsize=13)
        plt.tight_layout()
        path = os.path.join(
            self.cfg_rl.figures_dir_lp,
            "rl_strict_cf_examples.png"
        )
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[RL-Strict] CF examples → {path}")


if __name__ == "__main__":
    from src.utils.config import load_config
    from src.utils.train_tools import get_device

    CONFIG_FORECASTER = "assets/configs/models/itransformer/etth1_96_48_S.json"
    CONFIG_AE = "assets/configs/models/ae/tcn_ae.json"
    CONFIG_RL = "assets/configs/models/RL/rl_strict.json"

    cfg_f = load_config(CONFIG_FORECASTER)
    cfg_ae = load_config(CONFIG_AE)
    cfg_rl = load_config(CONFIG_RL)

    device = get_device(cfg_f)

    print(f"Device   : {device}")
    print("Pipeline : RL Strict (hybrid plausibility + latent drift)")

    trainer = RLStrictTrainer(cfg_f, cfg_ae, cfg_rl, device)
    history = trainer.train()
    examples = trainer.evaluate(n_batches=20)

    print("\n✅ Done !")
    print(f"  Checkpoint → {cfg_rl.checkpoint_dir_lp}/rl_strict_agent_best.pt")
    print(f"  Figures    → {cfg_rl.figures_dir_lp}/")
    print(f"  Results    → {cfg_rl.results_dir_lp}/rl_strict_history.json")
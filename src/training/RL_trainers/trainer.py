import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from src.models.autoencoder.tcn_ae import TCNAutoEncoder
from src.models.RL.agent import ActorCritic
from src.models.RL.rewardplz import CFReward
from src.models.Forecaster.forecaster_wrapper import ForecasterWrapper
from src.evaluation.unified_evaluator import CounterfactualEvaluator


def prepare_rl_data(cfg_forecaster, cfg_ae, device):
    from src.data_provider.data_factory import data_provider

    _, train_loader = data_provider(cfg_forecaster, "train")
    _, test_loader = data_provider(cfg_forecaster, "test")
    ckpt_ae = torch.load(cfg_ae.checkpoint_path, map_location="cpu", weights_only=False)
    scaler = None
    if "scaler_mean" in ckpt_ae and "scaler_std" in ckpt_ae:
        scaler = StandardScaler()
        scaler.mean_ = np.array(ckpt_ae["scaler_mean"], dtype=np.float64)
        scaler.scale_ = np.array(ckpt_ae["scaler_std"], dtype=np.float64)
        scaler.var_ = scaler.scale_**2
        scaler.n_features_in_ = len(scaler.mean_) if np.ndim(scaler.mean_) > 0 else 1
        print("[Data] Scaler loaded ✓")
    else:
        print("[Warning] No scaler in AE checkpoint")
    return train_loader, test_loader, scaler


def filter_batch(y_hat, quantile=0.0):
    if quantile == 0.0:
        return torch.ones(y_hat.shape[0], dtype=torch.bool, device=y_hat.device)
    mean_yhat = y_hat[:, :, 0].mean(dim=1)
    return mean_yhat >= torch.quantile(mean_yhat, quantile)


def build_temporal_mask(batch_size, seq_len, channels, last_k, ramp_k, device):
    m = torch.zeros((batch_size, seq_len, channels), device=device)
    start_full = max(0, seq_len - last_k)
    m[:, start_full:, :] = 1.0
    if ramp_k > 0:
        start_ramp = max(0, start_full - ramp_k)
        ramp_len = start_full - start_ramp
        if ramp_len > 0:
            ramp = torch.linspace(0.0, 1.0, ramp_len, device=device).view(
                1, ramp_len, 1
            )
            m[:, start_ramp:start_full, :] = ramp
    return m


def run_episode_train(
    batch,
    ae_arch,
    forecaster,
    agent,
    reward_fn,
    device,
    use_rl=True,
    mask_last_k=24,
    mask_ramp_k=8,
    filter_quantile=0.0,
):
    batch_x, _, batch_x_mark, _ = batch
    batch_x = batch_x.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    x_ot = batch_x[:, :, -1:]

    with torch.no_grad():
        z = ae_arch.encode(x_ot)
        y_hat = forecaster.predict_ot(batch_x, batch_x_mark)

    mask_keep = filter_batch(y_hat, filter_quantile)
    if mask_keep.sum() == 0:
        return None

    x_ot = x_ot[mask_keep]
    batch_x = batch_x[mask_keep]
    batch_x_mark = batch_x_mark[mask_keep]
    y_hat = y_hat[mask_keep]
    z = z[mask_keep]

    if use_rl:
        s = agent.build_state(z, y_hat)
        a, log_prob, _ = agent.actor.sample(s)
        z_cf = torch.clamp(z + agent.eta * a, -1.0, 1.0)
    else:
        s, log_prob, z_cf = None, None, z

    x_prop = ae_arch.decode(z_cf)
    delta = x_prop - x_ot
    temp_mask = build_temporal_mask(
        x_ot.shape[0], x_ot.shape[1], x_ot.shape[2], mask_last_k, mask_ramp_k, device
    )
    x_cf = x_ot + temp_mask * delta

    with torch.no_grad():
        y_cf = forecaster.predict_from_ot(
            x_ot=x_cf, x_full=batch_x, x_mark=batch_x_mark
        )

    reward_dict = reward_fn(x_ot, x_cf, y_hat, y_cf, z_cf=z_cf)
    R = reward_dict["total"]

    entropy = value = None
    if use_rl:
        value = agent.evaluate(s)
        mu, log_std = agent.actor(s)
        entropy = torch.distributions.Normal(mu, log_std.exp()).entropy().sum(dim=-1)

    return dict(
        log_prob=log_prob,
        reward=R,
        value=value,
        entropy=entropy,
        reward_dict=reward_dict,
        x_ot=x_ot.detach(),
        x_cf=x_cf.detach(),
        y_hat=y_hat.detach(),
        y_cf=y_cf.detach(),
        batch_x=batch_x.detach(),
        batch_x_mark=batch_x_mark.detach(),
        z=z.detach(),
        z_cf=z_cf.detach(),
        mask=temp_mask.detach(),
        n_valid=int(mask_keep.sum()),
    )


@torch.no_grad()
def run_episode_eval(
    batch,
    ae_arch,
    forecaster,
    agent,
    reward_fn,
    device,
    use_rl=True,
    mask_last_k=24,
    mask_ramp_k=8,
    filter_quantile=0.0,
):
    batch_x, _, batch_x_mark, _ = batch
    batch_x = batch_x.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    x_ot = batch_x[:, :, -1:]

    z = ae_arch.encode(x_ot)
    y_hat = forecaster.predict_ot(batch_x, batch_x_mark)

    mask_keep = filter_batch(y_hat, filter_quantile)
    if mask_keep.sum() == 0:
        return None

    x_ot = x_ot[mask_keep]
    batch_x = batch_x[mask_keep]
    batch_x_mark = batch_x_mark[mask_keep]
    y_hat = y_hat[mask_keep]
    z = z[mask_keep]

    z_cf = agent.act_deterministic(z, y_hat)[0] if use_rl else z

    x_prop = ae_arch.decode(z_cf)
    delta = x_prop - x_ot
    temp_mask = build_temporal_mask(
        x_ot.shape[0], x_ot.shape[1], x_ot.shape[2], mask_last_k, mask_ramp_k, device
    )
    x_cf = x_ot + temp_mask * delta

    y_cf = forecaster.predict_from_ot(x_ot=x_cf, x_full=batch_x, x_mark=batch_x_mark)
    reward_dict = reward_fn(x_ot, x_cf, y_hat, y_cf, z_cf=z_cf)

    return dict(
        x_ot=x_ot.detach(),
        x_cf=x_cf.detach(),
        y_hat=y_hat.detach(),
        y_cf=y_cf.detach(),
        batch_x=batch_x.detach(),
        batch_x_mark=batch_x_mark.detach(),
        reward_dict=reward_dict,
        n_valid=int(mask_keep.sum()),
    )


class RLMaskTrainer:
    def __init__(self, cfg_forecaster, cfg_ae, cfg_rl, device):
        self.cfg_f = cfg_forecaster
        self.cfg_ae = cfg_ae
        self.cfg_rl = cfg_rl
        self.device = device

        self.exp_name = getattr(cfg_rl, "name", "exp").replace(" ", "_")
        self.mask_last_k = getattr(cfg_rl, "mask_last_k", 24)
        self.mask_ramp_k = getattr(cfg_rl, "mask_ramp_k", 8)
        self.filter_quantile = getattr(cfg_rl, "filter_quantile", 0.0)

        for d in [
            cfg_rl.checkpoint_dir_lp,
            cfg_rl.figures_dir_lp,
            cfg_rl.results_dir_lp,
        ]:
            os.makedirs(d, exist_ok=True)

        print("\n[RL] Loading frozen models …")
        self.ae_arch = TCNAutoEncoder.from_checkpoint(
            cfg_ae.checkpoint_path, device=device
        )
        self.ae_arch.eval()
        for p in self.ae_arch.parameters():
            p.requires_grad_(False)

        self.forecaster = ForecasterWrapper(cfg_forecaster, device)
        self.forecaster.model.eval()
        for p in self.forecaster.model.parameters():
            p.requires_grad_(False)

        self.train_loader, self.test_loader, self.scaler = prepare_rl_data(
            cfg_forecaster, cfg_ae, device
        )

        self.reward_fn = CFReward(
            fr=getattr(cfg_rl, "fr", 0.5),
            w_validity=getattr(cfg_rl, "w_validity", 2.0),
            w_proximity=getattr(cfg_rl, "w_proximity", 0.3),
            w_hard_bonus=getattr(cfg_rl, "w_hard_bonus", 1.0),
            use_validity=getattr(cfg_rl, "use_validity", True),
            use_proximity=getattr(cfg_rl, "use_proximity", True),
            direction=getattr(cfg_rl, "direction", -1.0),
        ).to(device)

        print(
            f"[RL] fr={self.reward_fn.fr}  "
            f"w_val={self.reward_fn.w_validity}  "
            f"w_prox={self.reward_fn.w_proximity}  "
            f"w_bonus={self.reward_fn.w_hard_bonus}  "
            f"direction={self.reward_fn.direction}"
        )

        self.agent = ActorCritic(
            latent_dim=cfg_ae.latent_dim,
            pred_len=cfg_forecaster.pred_len,
            eta=cfg_rl.eta,
            entropy_coef=cfg_rl.entropy_coef,
            direction=getattr(cfg_rl, "direction", -1.0),
        ).to(device)

        params = self.agent.count_parameters()
        print(f"[RL] Actor {params['actor']:,}  Critic {params['critic']:,}")

        self.opt_actor = torch.optim.Adam(
            self.agent.actor.parameters(), lr=cfg_rl.lr_actor
        )
        self.opt_critic = torch.optim.Adam(
            self.agent.critic.parameters(), lr=getattr(cfg_rl, "lr_critic", 1e-4)
        )

        # LR scheduler — reduit le lr de 50% a mi-training pour stabiliser
        lr_decay   = getattr(cfg_rl, "lr_decay", 0.5)
        decay_step = max(1, cfg_rl.epochs // 2)
        self.sched_actor  = torch.optim.lr_scheduler.StepLR(
            self.opt_actor,  step_size=decay_step, gamma=lr_decay
        )
        self.sched_critic = torch.optim.lr_scheduler.StepLR(
            self.opt_critic, step_size=decay_step, gamma=lr_decay
        )

        self.history = {
            k: []
            for k in [
                "loss_total",
                "loss_actor",
                "loss_critic",
                "reward_total",
                "r_validity",
                "r_validity_hard",
                "r_proximity",
                "delta_mean",
                "success_rate",
                "advantage",
                "n_valid",
            ]
        }

        x_train_batches, max_b = [], getattr(cfg_rl, "eval_train_batches", 20)
        for i, batch in enumerate(self.train_loader):
            batch_x, _, _, _ = batch
            x_train_batches.append(batch_x[:, :, -1:].cpu().numpy())
            if i + 1 >= max_b:
                break
        self.x_train_eval = (
            np.concatenate(x_train_batches, axis=0) if x_train_batches else None
        )

        fit_plaus = self.x_train_eval is not None
        if fit_plaus:
            print("[RL] Plausibility will be fitted from x_train ✓")
        else:
            print("[RL] No plausibility (no x_train)")

        self.evaluator = CounterfactualEvaluator(
            x_train=self.x_train_eval,
            fit_plausibility=fit_plaus,
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _compute_bounds_np(self, x_ot, y_hat):
        """
        Numpy mirror de compute_bounds() — bornes relatives a y_hat.
        x_ot  : [K, BH, 1]
        y_hat : [K, H,  1]
        """
        fr        = self.reward_fn.fr
        direction = self.reward_fn.direction
        x2d       = x_ot[:, :, 0]      # [K, BH]
        y2d       = y_hat[:, :, 0]     # [K, H]

        std        = x2d.std(axis=1)   # [K]
        half_width = fr * std          # [K]
        hw         = half_width[:, np.newaxis]  # [K, 1] → broadcast sur H

        if direction < 0:
            alphas = y2d - hw
            betas  = y2d
        else:
            alphas = y2d
            betas  = y2d + hw

        return alphas, betas

    # ─────────────────────────────────────────────────────────────────────────
    def train(self):
        print(f"\n{'='*60}")
        print(f"Training | {self.cfg_rl.epochs} epochs | fr={self.reward_fn.fr}")
        print(f"{'='*60}")

        best_reward = -float("inf")
        use_rl = getattr(self.cfg_rl, "use_rl", True)

        for epoch in range(1, self.cfg_rl.epochs + 1):
            t0 = time.time()
            self.agent.train()
            stats = {k: [] for k in self.history}

            for batch in self.train_loader:
                ep = run_episode_train(
                    batch=batch,
                    ae_arch=self.ae_arch,
                    forecaster=self.forecaster,
                    agent=self.agent,
                    reward_fn=self.reward_fn,
                    device=self.device,
                    use_rl=use_rl,
                    mask_last_k=self.mask_last_k,
                    mask_ramp_k=self.mask_ramp_k,
                    filter_quantile=self.filter_quantile,
                )
                if ep is None:
                    continue

                rs = self.reward_fn.stats(ep["reward_dict"])
                stats["reward_total"].append(rs["total"])
                stats["r_validity"].append(rs["validity"])
                stats["r_validity_hard"].append(rs["validity_hard"])
                stats["r_proximity"].append(rs["proximity"])
                stats["delta_mean"].append(rs["delta_mean"])
                stats["success_rate"].append(rs["success"])
                stats["n_valid"].append(ep["n_valid"])

                if not use_rl:
                    continue

                loss_dict = self.agent.compute_loss(
                    ep["log_prob"], ep["reward"], ep["value"], ep["entropy"]
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

                stats["loss_total"].append(float(loss_dict["total"].item()))
                stats["loss_actor"].append(float(loss_dict["actor"].item()))
                stats["loss_critic"].append(float(loss_dict["critic"].item()))
                stats["advantage"].append(loss_dict["advantage"])

            means = {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}
            for k, v in means.items():
                self.history[k].append(v)

            print(
                f"Epoch {epoch:03d}/{self.cfg_rl.epochs} | "
                f"R={means['reward_total']:.4f} | "
                f"val_soft={means['r_validity']:.3f} | "
                f"val_hard={means['r_validity_hard']:.3f} | "
                f"SR={means['success_rate']*100:.1f}% | "
                f"prox={means['r_proximity']:.3f} | "
                f"{time.time()-t0:.1f}s"
            )

            if means["reward_total"] > best_reward:
                best_reward = means["reward_total"]
                self._save_checkpoint("best")
                print(f"  ✅ best={best_reward:.4f}")

            # LR decay
            self.sched_actor.step()
            self.sched_critic.step()
            lr_now = self.opt_actor.param_groups[0]["lr"]
            if epoch % 10 == 0:
                print(f"  [LR] actor={lr_now:.2e}")

        self._save_checkpoint("final")
        self._save_history()
        self._plot_training()
        print(f"\n✅ Done | best={best_reward:.4f}")
        return self.history

    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, n_batches=20):
        print(f"\n[Eval] {n_batches} batches …")
        self.agent.eval()
        use_rl = getattr(self.cfg_rl, "use_rl", True)

        all_x, all_x_cf, all_y_hat, all_y_cf = [], [], [], []
        cf_examples = []

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
                use_rl=use_rl,
                mask_last_k=self.mask_last_k,
                mask_ramp_k=self.mask_ramp_k,
                filter_quantile=self.filter_quantile,
            )
            if ep is None:
                continue

            all_x.append(ep["x_ot"].cpu().numpy())
            all_x_cf.append(ep["x_cf"].cpu().numpy())
            all_y_hat.append(ep["y_hat"].cpu().numpy())
            all_y_cf.append(ep["y_cf"].cpu().numpy())

            if len(cf_examples) < 4:
                cf_examples.append(
                    {
                        "x_ot": ep["x_ot"][0].cpu().numpy(),
                        "x_cf": ep["x_cf"][0].cpu().numpy(),
                        "y_hat": ep["y_hat"][0].cpu().numpy(),
                        "y_cf": ep["y_cf"][0].cpu().numpy(),
                    }
                )

        if not all_x:
            raise RuntimeError("No valid CFs.")

        all_x     = np.concatenate(all_x,     axis=0)
        all_x_cf  = np.concatenate(all_x_cf,  axis=0)
        all_y_hat = np.concatenate(all_y_hat, axis=0)
        all_y_cf  = np.concatenate(all_y_cf,  axis=0)

        all_alphas, all_betas = self._compute_bounds_np(all_x, all_y_hat)

        summary_std = self.evaluator.evaluate(
            X_orig=all_x,
            X_cf=all_x_cf,
            Y_hat=all_y_hat,
            Y_cf=all_y_cf,
            alphas=all_alphas,
            betas=all_betas,
        )

        print("\n── Evaluation Metrics ───────────────────────────────")
        CounterfactualEvaluator.print_table(summary_std, label=self.exp_name)

        path = os.path.join(
            self.cfg_rl.results_dir_lp, f"{self.exp_name}_evaluation.json"
        )
        with open(path, "w") as f:
            json.dump(summary_std, f, indent=2)
        print(f"[Eval] Saved → {path}")

        self._plot_cf_examples(
            cf_examples,
            os.path.join(
                self.cfg_rl.figures_dir_lp, f"{self.exp_name}_cf_examples.png"
            ),
        )
        return summary_std, cf_examples

    # ─────────────────────────────────────────────────────────────────────────
    def _save_checkpoint(self, tag):
        path = os.path.join(
            self.cfg_rl.checkpoint_dir_lp, f"{self.exp_name}_agent_{tag}.pt"
        )
        torch.save(
            {
                "actor_state_dict": self.agent.actor.state_dict(),
                "critic_state_dict": self.agent.critic.state_dict(),
                "history": self.history,
                "cfg_rl": vars(self.cfg_rl),
            },
            path,
        )
        print(f"[RL] Checkpoint → {path}")

    def _save_history(self):
        path = os.path.join(self.cfg_rl.results_dir_lp, f"{self.exp_name}_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    def _plot_training(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))

        axes[0, 0].plot(self.history["reward_total"], lw=1.5)
        axes[0, 0].set_title("Total Reward")
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(self.history["r_validity"],      lw=1.5, label="validity soft")
        axes[0, 1].plot(self.history["r_validity_hard"], lw=1.5, label="validity hard")
        axes[0, 1].plot(self.history["r_proximity"],     lw=1.5, label="proximity")
        axes[0, 1].set_title("Sub-Rewards")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        axes[0, 2].plot(self.history["success_rate"], lw=1.5)
        axes[0, 2].axhline(0.5, linestyle="--", lw=1.2, color="red", label="50%")
        axes[0, 2].set_title("Success Rate")
        axes[0, 2].legend()
        axes[0, 2].grid(alpha=0.3)

        axes[1, 0].plot(self.history["loss_actor"],  lw=1.5, label="actor")
        axes[1, 0].plot(self.history["loss_critic"], lw=1.5, label="critic")
        axes[1, 0].set_title("Losses")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].plot(self.history["r_validity_hard"], lw=1.5, color="orange")
        axes[1, 1].axhline(0.5, linestyle="--", lw=1.2, color="red", label="50%")
        axes[1, 1].set_title("Validity Hard (Validity Ratio)")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        axes[1, 2].plot(self.history["n_valid"], lw=1.5)
        axes[1, 2].set_title("Samples valides / epoch")
        axes[1, 2].grid(alpha=0.3)

        plt.suptitle(f"RL-CF — {self.exp_name} | fr={self.reward_fn.fr}", fontsize=13)
        plt.tight_layout()
        path = os.path.join(
            self.cfg_rl.figures_dir_lp, f"{self.exp_name}_training_curves.png"
        )
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[RL] Curves → {path}")

    def _plot_cf_examples(self, examples, out_path):
        if not examples:
            return
        n = len(examples)
        fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
        if n == 1:
            axes = [axes]

        for i, ex in enumerate(examples):
            x_ot  = ex["x_ot"][:, 0]
            x_cf  = ex["x_cf"][:, 0]
            y_hat = ex["y_hat"][:, 0]
            y_cf  = ex["y_cf"][:, 0]

            # Bounds — formule multiplicative identique au papier
            x_t  = torch.tensor(ex["x_ot"], dtype=torch.float32).unsqueeze(0)
            yh_t = torch.tensor(ex["y_hat"], dtype=torch.float32).unsqueeze(0)
            alpha_t, beta_t, _ = self.reward_fn.compute_bounds(yh_t, x_ot=x_t)
            alpha_np = alpha_t[0].numpy()  # [T]
            beta_np  = beta_t[0].numpy()   # [T]

            full_orig = np.concatenate([x_ot, y_hat])
            full_cf   = np.concatenate([x_cf, y_cf])
            t_all  = np.arange(len(full_orig))
            t_fore = np.arange(len(x_ot), len(full_orig))

            valid_ratio = float(((y_cf >= alpha_np) & (y_cf <= beta_np)).mean())

            axes[i].plot(t_all, full_orig, lw=1.5, label="original + forecast")
            axes[i].plot(t_all, full_cf,   lw=1.5, ls="--", label="CF + forecast_cf")
            axes[i].fill_between(
                t_fore, alpha_np, beta_np,
                alpha=0.20, color="green", label="α/β bounds"
            )
            axes[i].axvline(len(x_ot), linestyle="--", lw=1.2, color="gray")
            axes[i].set_title(
                f"Sample {i+1} — validity_hard={valid_ratio:.2f}", fontsize=11
            )
            axes[i].legend(fontsize=9)
            axes[i].grid(alpha=0.3)

        plt.suptitle(f"CF examples — {self.exp_name}", fontsize=13)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[RL] CF examples → {out_path}")
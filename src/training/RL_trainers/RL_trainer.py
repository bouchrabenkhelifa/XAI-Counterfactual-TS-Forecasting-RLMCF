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
from src.models.RL.reward import CFReward
from src.models.Forecaster.forecaster_wrapper import ForecasterWrapper

from src.evaluation.evaluator import CounterfactualEvaluator
from src.evaluation.plausibility_metrics import load_plausibility_model


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
        print("[Data] Scaler loaded from AE checkpoint ✓")
    else:
        print("[Warning] 'scaler_mean' / 'scaler_std' not found in AE checkpoint")
        print("[Warning] Continuing with scaler = None")

    return train_loader, test_loader, scaler


def filter_batch(y_hat, quantile=0.75):
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
    filter_quantile=0.75,
    alpha_hf=0.25,
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
        s = None
        log_prob = None
        z_cf = z

    x_prop = ae_arch.decode(z_cf)
    delta = x_prop - x_ot

    with torch.no_grad():
        x_lf = ae_arch.decode(ae_arch.encode(x_ot))
    x_hf = x_ot - x_lf

    temp_mask = build_temporal_mask(
        batch_size=x_ot.shape[0],
        seq_len=x_ot.shape[1],
        channels=x_ot.shape[2],
        last_k=mask_last_k,
        ramp_k=mask_ramp_k,
        device=device,
    )

    if use_rl:
        masked_delta = delta + alpha_hf * x_hf.detach()
        x_cf = x_ot + temp_mask * masked_delta
    else:
        x_cf = x_ot.clone()

    with torch.no_grad():
        y_cf = forecaster.predict_from_ot(
            x_ot=x_cf,
            x_full=batch_x,
            x_mark=batch_x_mark,
        )

    reward_dict = reward_fn(x_ot, x_cf, y_hat, y_cf, z_cf=z_cf)
    R = reward_dict["total"]

    entropy = None
    value = None
    if use_rl:
        value = agent.evaluate(s)
        mu, log_std = agent.actor(s)
        entropy = torch.distributions.Normal(mu, log_std.exp()).entropy().sum(dim=-1)

    return {
        "log_prob": log_prob,
        "reward": R,
        "value": value,
        "entropy": entropy,
        "reward_dict": reward_dict,
        "x_ot": x_ot.detach(),
        "x_cf": x_cf.detach(),
        "y_hat": y_hat.detach(),
        "y_cf": y_cf.detach(),
        "batch_x": batch_x.detach(),
        "batch_x_mark": batch_x_mark.detach(),
        "z": z.detach(),
        "z_cf": z_cf.detach(),
        "mask": temp_mask.detach(),
        "n_valid": int(mask_keep.sum()),
    }


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
    filter_quantile=0.75,
    alpha_hf=0.25,
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

    if use_rl:
        z_cf, _, _ = agent.act_deterministic(z, y_hat)
    else:
        z_cf = z

    x_prop = ae_arch.decode(z_cf)
    delta = x_prop - x_ot

    x_lf = ae_arch.decode(ae_arch.encode(x_ot))
    x_hf = x_ot - x_lf

    temp_mask = build_temporal_mask(
        batch_size=x_ot.shape[0],
        seq_len=x_ot.shape[1],
        channels=x_ot.shape[2],
        last_k=mask_last_k,
        ramp_k=mask_ramp_k,
        device=device,
    )

    if use_rl:
        masked_delta = delta + alpha_hf * x_hf
        x_cf = x_ot + temp_mask * masked_delta
    else:
        x_cf = x_ot.clone()

    y_cf = forecaster.predict_from_ot(
        x_ot=x_cf,
        x_full=batch_x,
        x_mark=batch_x_mark,
    )

    reward_dict = reward_fn(x_ot, x_cf, y_hat, y_cf, z_cf=z_cf)

    return {
        "x_ot": x_ot.detach(),
        "x_cf": x_cf.detach(),
        "y_hat": y_hat.detach(),
        "y_cf": y_cf.detach(),
        "batch_x": batch_x.detach(),
        "batch_x_mark": batch_x_mark.detach(),
        "reward_dict": reward_dict,
        "n_valid": int(mask_keep.sum()),
    }


def forecast_monotonicity_with_context(
    x_ot,
    x_cf,
    x_full,
    x_mark,
    forecaster,
    threshold=5e-2,
):
    device = next(forecaster.model.parameters()).device

    x_ot_np = x_ot.detach().cpu().numpy()
    x_cf_np = x_cf.detach().cpu().numpy()

    x_full = x_full.detach().to(device)
    x_mark = x_mark.detach().to(device)

    B = x_ot_np.shape[0]
    scores = []

    for i in range(B):
        xi = x_ot_np[i : i + 1]
        xi_cf = x_cf_np[i : i + 1]

        xfull_i = x_full[i : i + 1]
        xmark_i = x_mark[i : i + 1]

        diff = np.abs(xi_cf - xi)
        if diff.ndim == 3:
            mask = np.where(np.max(diff[0], axis=-1) > threshold)[0]
        else:
            mask = np.where(diff[0] > threshold)[0]

        if len(mask) == 0:
            scores.append(1.0)
            continue

        xi_t = torch.from_numpy(xi).float().to(device)
        xi_cf_t = torch.from_numpy(xi_cf).float().to(device)

        with torch.no_grad():
            baseline_forecast = forecaster.predict_from_ot(
                x_ot=xi_t,
                x_full=xfull_i,
                x_mark=xmark_i,
            )
        baseline_mean = baseline_forecast.mean().item()

        contributions = []
        for t in mask:
            x_ablated = xi_cf_t.clone()
            x_ablated[0, t, :] = xi_t[0, t, :]

            with torch.no_grad():
                ablated_forecast = forecaster.predict_from_ot(
                    x_ot=x_ablated,
                    x_full=xfull_i,
                    x_mark=xmark_i,
                )

            contribution = baseline_mean - ablated_forecast.mean().item()
            contributions.append(contribution)

        scores.append(float(np.mean(np.array(contributions) > 0)))

    return np.asarray(scores, dtype=np.float32)


class RLMaskTrainer:
    def __init__(self, cfg_forecaster, cfg_ae, cfg_rl, device):
        self.cfg_f = cfg_forecaster
        self.cfg_ae = cfg_ae
        self.cfg_rl = cfg_rl
        self.device = device

        self.alpha_hf = getattr(cfg_rl, "alpha_hf", 0.2)
        self.mask_last_k = getattr(cfg_rl, "mask_last_k", 24)
        self.mask_ramp_k = getattr(cfg_rl, "mask_ramp_k", 8)
        self.filter_quantile = getattr(cfg_rl, "filter_quantile", 0.75)

        for d in [
            cfg_rl.checkpoint_dir_lp,
            cfg_rl.figures_dir_lp,
            cfg_rl.results_dir_lp,
        ]:
            os.makedirs(d, exist_ok=True)

        print("\n[RL-Mask] Loading frozen models ...")

        # AE kept in architecture
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

        reward_ae = self.ae_arch if getattr(cfg_rl, "use_autoencoder", True) else None
        if reward_ae is None:
            print(
                "[RL-Mask] use_autoencoder=False -> reconstruction reward disabled, but AE kept in mask architecture."
            )

        self.reward_fn = CFReward(
            ae=reward_ae,
            rho=cfg_rl.rho,
            alpha=cfg_rl.alpha,
            beta=cfg_rl.beta,
            gamma=getattr(cfg_rl, "gamma", 0.8),
            tau=getattr(cfg_rl, "tau", 0.6),
            use_validity=getattr(cfg_rl, "use_validity", True),
            use_proximity=getattr(cfg_rl, "use_proximity", True),
            use_reconstruction=getattr(cfg_rl, "use_reconstruction", True),
            use_temporal=getattr(cfg_rl, "use_temporal", True),
        ).to(device)

        print(
            f"[RL-Mask] Objectif      : réduire mean(forecast) de {cfg_rl.rho * 100:.0f}%"
        )
        print(f"[RL-Mask] eta           = {cfg_rl.eta}")
        print(f"[RL-Mask] alpha_hf      = {self.alpha_hf}")
        print(f"[RL-Mask] mask_last_k   = {self.mask_last_k}")
        print(f"[RL-Mask] mask_ramp_k   = {self.mask_ramp_k}")
        print(f"[RL-Mask] use_rl        = {getattr(cfg_rl, 'use_rl', True)}")
        print(f"[RL-Mask] name          = {getattr(cfg_rl, 'name', 'unnamed')}")

        self.agent = ActorCritic(
            latent_dim=cfg_ae.latent_dim,
            pred_len=cfg_forecaster.pred_len,
            eta=cfg_rl.eta,
            entropy_coef=cfg_rl.entropy_coef,
            direction=getattr(cfg_rl, "direction", -1.0),
        ).to(device)

        params = self.agent.count_parameters()
        print(f"[RL-Mask] Actor {params['actor']:,}  Critic {params['critic']:,}")

        self.opt_actor = torch.optim.Adam(
            self.agent.actor.parameters(), lr=cfg_rl.lr_actor
        )
        self.opt_critic = torch.optim.Adam(
            self.agent.critic.parameters(), lr=getattr(cfg_rl, "lr_critic", 1e-4)
        )

        self.history = {
            k: []
            for k in [
                "loss_total",
                "loss_actor",
                "loss_critic",
                "reward_total",
                "r_validity",
                "r_proximity",
                "r_reconstruction",
                "r_temporal",
                "delta_mean",
                "success_rate",
                "advantage",
                "n_valid",
            ]
        }

        x_train_batches = []
        max_train_batches = getattr(cfg_rl, "eval_train_batches", 20)
        for i, batch in enumerate(self.train_loader):
            batch_x, _, _, _ = batch
            x_train_batches.append(batch_x[:, :, -1:].cpu().numpy())
            if i + 1 >= max_train_batches:
                break
        self.x_train_eval = (
            np.concatenate(x_train_batches, axis=0)
            if len(x_train_batches) > 0
            else None
        )

        plausibility_model = None
        plaus_path = getattr(cfg_rl, "plausibility_model_path", None)
        if plaus_path is not None and os.path.exists(plaus_path):
            plausibility_model = load_plausibility_model(plaus_path)
            print(f"[RL-Mask] Plausibility model loaded: {plaus_path}")
        else:
            print("[RL-Mask] No plausibility model provided for evaluator")

        self.evaluator = CounterfactualEvaluator(
            plausibility_model=plausibility_model,
            rho=cfg_rl.rho,
            x_train=self.x_train_eval,
        )

    def train(self):
        print(f"\n{'=' * 60}")
        print(
            f"RL-Mask Training | {self.cfg_rl.epochs} epochs | "
            f"rho={self.cfg_rl.rho} | eta={self.cfg_rl.eta} | "
            f"mask_last_k={self.mask_last_k} | ramp={self.mask_ramp_k}"
        )
        print(f"{'=' * 60}")

        best_reward = -float("inf")
        use_rl = getattr(self.cfg_rl, "use_rl", True)

        for epoch in range(1, self.cfg_rl.epochs + 1):
            t0 = time.time()
            self.agent.train()
            stats = {k: [] for k in self.history}
            train_examples = []

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
                    alpha_hf=self.alpha_hf,
                )
                if ep is None:
                    continue

                if len(train_examples) < 4:
                    train_examples.append(
                        {
                            "x_ot": ep["x_ot"][0].cpu().numpy(),
                            "x_cf": ep["x_cf"][0].cpu().numpy(),
                            "y_hat": ep["y_hat"][0].cpu().numpy(),
                            "y_cf": ep["y_cf"][0].cpu().numpy(),
                        }
                    )

                rs = self.reward_fn.stats(ep["reward_dict"])
                stats["reward_total"].append(rs["total"])
                stats["r_validity"].append(rs["validity"])
                stats["r_proximity"].append(rs["proximity"])
                stats["r_reconstruction"].append(rs["reconstruction"])
                stats["r_temporal"].append(rs["temporal"])
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
                f"val={means['r_validity']:.3f} | "
                f"rec={means['r_reconstruction']:.3f} | "
                f"temp={means['r_temporal']:.3f} | "
                f"Δ={means['delta_mean']:.4f} | "
                f"SR={means['success_rate'] * 100:.1f}% | "
                f"prox={means['r_proximity']:.3f} | "
                f"{time.time() - t0:.1f}s"
            )

            is_best = False
            if means["reward_total"] > best_reward:
                best_reward = means["reward_total"]
                self._save_checkpoint("best")
                print(f"  ✅ best = {best_reward:.4f}")
                is_best = True

            if epoch == 1 or epoch == self.cfg_rl.epochs or is_best:
                train_fig_path = os.path.join(
                    self.cfg_rl.figures_dir_lp,
                    f"{getattr(self.cfg_rl, 'name', 'exp')}_train_examples_epoch_{epoch:03d}.png",
                )
                self._plot_cf_examples(
                    train_examples, train_fig_path, title_prefix="Train"
                )

        self._save_checkpoint("final")
        self._save_history()
        self._plot_training()
        print(f"\n✅ Done | best reward = {best_reward:.4f}")
        return self.history

    @torch.no_grad()
    def evaluate(self, n_batches=20):
        print(f"\n[Eval] Test set ({n_batches} batches) ...")
        self.agent.eval()
        use_rl = getattr(self.cfg_rl, "use_rl", True)

        all_x = []
        all_x_cf = []
        all_y_hat = []
        all_y_cf = []
        all_monot = []

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
                alpha_hf=self.alpha_hf,
            )
            if ep is None:
                continue

            all_x.append(ep["x_ot"].cpu().numpy())
            all_x_cf.append(ep["x_cf"].cpu().numpy())
            all_y_hat.append(ep["y_hat"].cpu().numpy())
            all_y_cf.append(ep["y_cf"].cpu().numpy())

            mono = forecast_monotonicity_with_context(
                x_ot=ep["x_ot"],
                x_cf=ep["x_cf"],
                x_full=ep["batch_x"],
                x_mark=ep["batch_x_mark"],
                forecaster=self.forecaster,
            )
            all_monot.append(mono)

            if len(cf_examples) < 4:
                cf_examples.append(
                    {
                        "x_ot": ep["x_ot"][0].cpu().numpy(),
                        "x_cf": ep["x_cf"][0].cpu().numpy(),
                        "y_hat": ep["y_hat"][0].cpu().numpy(),
                        "y_cf": ep["y_cf"][0].cpu().numpy(),
                    }
                )

        if len(all_x) == 0:
            raise RuntimeError("No valid CFs generated during evaluation.")

        all_x = np.concatenate(all_x, axis=0)
        all_x_cf = np.concatenate(all_x_cf, axis=0)
        all_y_hat = np.concatenate(all_y_hat, axis=0)
        all_y_cf = np.concatenate(all_y_cf, axis=0)
        all_monot = np.concatenate(all_monot, axis=0)

        metrics = self.evaluator.evaluate_batch(
            x=all_x,
            x_cf=all_x_cf,
            y_hat=all_y_hat,
            y_cf=all_y_cf,
            include_dtw=False,
            include_reachability=True,
            forecaster=None,
        )
        metrics["forecast_monotonicity"] = all_monot

        summary_std = self.evaluator.summarize_with_std(metrics)
        summary_flat = {k: v["mean"] for k, v in summary_std.items()}

        print("\n── Final Evaluation Metrics ─────────────────────────")
        self.evaluator.pretty_print_summary(summary_flat)

        metrics_path = os.path.join(
            self.cfg_rl.results_dir_lp,
            f"{getattr(self.cfg_rl, 'name', 'exp')}_evaluation.json",
        )
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(summary_std, f, indent=2)
        print(f"\n[Eval] Metrics saved -> {metrics_path}")

        fig_path = os.path.join(
            self.cfg_rl.figures_dir_lp,
            f"{getattr(self.cfg_rl, 'name', 'exp')}_cf_examples.png",
        )
        self._plot_cf_examples(cf_examples, fig_path, title_prefix="Eval")

        return summary_std, cf_examples

    def _save_checkpoint(self, tag):
        torch.save(
            {
                "actor_state_dict": self.agent.actor.state_dict(),
                "critic_state_dict": self.agent.critic.state_dict(),
                "history": self.history,
                "cfg_rl": vars(self.cfg_rl),
            },
            os.path.join(
                self.cfg_rl.checkpoint_dir_lp,
                f"{getattr(self.cfg_rl, 'name', 'exp')}_agent_{tag}.pt",
            ),
        )

    def _save_history(self):
        path = os.path.join(
            self.cfg_rl.results_dir_lp,
            f"{getattr(self.cfg_rl, 'name', 'exp')}_history.json",
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        print(f"[RL-Mask] History → {path}")

    def _plot_training(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))

        axes[0, 0].plot(self.history["reward_total"], lw=1.5)
        axes[0, 0].set_title("Total Reward")
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(self.history["r_validity"], lw=1.5, label="validity")
        axes[0, 1].plot(self.history["r_proximity"], lw=1.5, label="proximity")
        axes[0, 1].plot(
            self.history["r_reconstruction"], lw=1.5, label="reconstruction"
        )
        axes[0, 1].plot(self.history["r_temporal"], lw=1.5, label="temporal")
        axes[0, 1].set_title("Sub-Rewards")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        axes[0, 2].axis("off")

        axes[1, 0].plot(self.history["loss_actor"], lw=1.5, label="actor")
        axes[1, 0].plot(self.history["loss_critic"], lw=1.5, label="critic")
        axes[1, 0].set_title("Losses")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].plot(self.history["delta_mean"], lw=1.5, label="Δmean")
        axes[1, 1].plot(
            self.history["success_rate"], lw=1.5, linestyle="--", label="SR"
        )
        axes[1, 1].axhline(self.cfg_rl.rho, linestyle=":", lw=1.2)
        axes[1, 1].set_title("Δmean & SR")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        axes[1, 2].plot(self.history["n_valid"], lw=1.5)
        axes[1, 2].set_title("Samples valides / epoch")
        axes[1, 2].grid(alpha=0.3)

        plt.suptitle(
            f"RL Mask — {getattr(self.cfg_rl, 'name', 'exp')} | "
            f"rho={self.cfg_rl.rho} | eta={self.cfg_rl.eta} | "
            f"mask_last_k={self.mask_last_k} | ramp={self.mask_ramp_k}",
            fontsize=13,
        )
        plt.tight_layout()
        path = os.path.join(
            self.cfg_rl.figures_dir_lp,
            f"{getattr(self.cfg_rl, 'name', 'exp')}_training_curves.png",
        )
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[RL-Mask] Curves → {path}")

    def _plot_cf_examples(self, examples, out_path, title_prefix="CF"):
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
            reduction = (y_hat.mean() - y_cf.mean()) / (abs(y_hat.mean()) + 1e-8) * 100
            ok = "✓" if reduction >= self.cfg_rl.rho * 100 else "✗"

            axes[i].plot(t_all, full_orig, lw=1.5, label="x + forecast(x)")
            axes[i].plot(t_all, full_cf, lw=1.5, ls="--", label="x_cf + forecast(x_cf)")
            axes[i].axvline(len(x_ot), linestyle="--", lw=1.2)
            axes[i].fill_between(t_all, full_orig, full_cf, alpha=0.12)
            axes[i].set_title(f"Sample {i+1} — {reduction:+.1f}% {ok}", fontsize=11)
            axes[i].legend(fontsize=9)
            axes[i].grid(alpha=0.3)

        plt.suptitle(
            f"{title_prefix} RL-Mask — {getattr(self.cfg_rl, 'name', 'exp')}",
            fontsize=13,
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[RL-Mask] CF examples → {out_path}")

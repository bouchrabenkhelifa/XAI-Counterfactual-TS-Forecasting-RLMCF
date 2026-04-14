import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.models.RL.reward import CFReward
from src.models.RL_wo_ae.agent import NoAEActorCritic
from src.models.Forecaster.forecaster_wrapper import ForecasterWrapper
from src.evaluation.evaluator import CounterfactualEvaluator
from src.evaluation.plausibility_metrics import load_plausibility_model


def build_temporal_mask(batch_size, seq_len, channels, last_k, ramp_k, device):
    m = torch.zeros((batch_size, seq_len, channels), device=device)
    start_full = max(0, seq_len - last_k)
    m[:, start_full:, :] = 1.0

    if ramp_k > 0:
        start_ramp = max(0, start_full - ramp_k)
        ramp_len = start_full - start_ramp
        if ramp_len > 0:
            ramp = torch.linspace(0.0, 1.0, ramp_len, device=device).view(1, ramp_len, 1)
            m[:, start_ramp:start_full, :] = ramp
    return m


def filter_batch(y_hat, quantile=0.75):
    mean_yhat = y_hat[:, :, 0].mean(dim=1)
    return mean_yhat >= torch.quantile(mean_yhat, quantile)


class RLNoAETrainer:
    def __init__(self, cfg_forecaster, cfg_rl, device):
        self.cfg_f = cfg_forecaster
        self.cfg_rl = cfg_rl
        self.device = device
        self.exp_name = getattr(cfg_rl, "name", "exp").replace(" ", "_")

        self.alpha_hf = 0.0
        self.mask_last_k = getattr(cfg_rl, "mask_last_k", 24)
        self.mask_ramp_k = getattr(cfg_rl, "mask_ramp_k", 8)
        self.filter_quantile = getattr(cfg_rl, "filter_quantile", 0.75)

        for d in [
            cfg_rl.checkpoint_dir_lp,
            cfg_rl.figures_dir_lp,
            cfg_rl.results_dir_lp,
        ]:
            os.makedirs(d, exist_ok=True)

        from src.data_provider.data_factory import data_provider
        _, self.train_loader = data_provider(cfg_forecaster, "train")
        _, self.test_loader = data_provider(cfg_forecaster, "test")

        self.forecaster = ForecasterWrapper(cfg_forecaster, device)
        self.forecaster.model.eval()
        for p in self.forecaster.model.parameters():
            p.requires_grad_(False)

        self.reward_fn = CFReward(
            ae=None,
            rho=cfg_rl.rho,
            alpha=cfg_rl.alpha,
            beta=cfg_rl.beta,
            gamma=getattr(cfg_rl, "gamma", 0.8),
            tau=getattr(cfg_rl, "tau", 0.6),
            use_validity=getattr(cfg_rl, "use_validity", True),
            use_proximity=getattr(cfg_rl, "use_proximity", True),
            use_reconstruction=False,
            use_temporal=getattr(cfg_rl, "use_temporal", True),
        ).to(device)

        self.agent = NoAEActorCritic(
            seq_len=cfg_forecaster.seq_len,
            pred_len=cfg_forecaster.pred_len,
            eta=cfg_rl.eta,
            entropy_coef=cfg_rl.entropy_coef,
            direction=getattr(cfg_rl, "direction", -1.0),
        ).to(device)

        self.opt_actor = torch.optim.Adam(self.agent.actor.parameters(), lr=cfg_rl.lr_actor)
        self.opt_critic = torch.optim.Adam(self.agent.critic.parameters(), lr=cfg_rl.lr_critic)

        self.history = {
            k: [] for k in [
                "loss_total", "loss_actor", "loss_critic",
                "reward_total", "r_validity", "r_proximity",
                "r_reconstruction", "r_temporal",
                "delta_mean", "success_rate", "advantage", "n_valid"
            ]
        }

        x_train_batches = []
        max_train_batches = getattr(cfg_rl, "eval_train_batches", 20)
        for i, batch in enumerate(self.train_loader):
            batch_x, _, _, _ = batch
            x_train_batches.append(batch_x[:, :, -1:].cpu().numpy())
            if i + 1 >= max_train_batches:
                break
        self.x_train_eval = np.concatenate(x_train_batches, axis=0) if x_train_batches else None

        plausibility_model = None
        plaus_path = getattr(cfg_rl, "plausibility_model_path", None)
        if plaus_path is not None and os.path.exists(plaus_path):
            plausibility_model = load_plausibility_model(plaus_path)

        self.evaluator = CounterfactualEvaluator(
            plausibility_model=plausibility_model,
            rho=cfg_rl.rho,
            x_train=self.x_train_eval,
        )

    def _make_cf(self, x_ot, y_hat):
        s = self.agent.build_state(x_ot, y_hat)
        a, log_prob, _ = self.agent.actor.sample(s)
        a = a.view(-1, self.cfg_f.seq_len, 1)

        temp_mask = build_temporal_mask(
            batch_size=x_ot.shape[0],
            seq_len=x_ot.shape[1],
            channels=x_ot.shape[2],
            last_k=self.mask_last_k,
            ramp_k=self.mask_ramp_k,
            device=self.device,
        )

        x_cf = x_ot + temp_mask * (self.agent.eta * a)
        return x_cf, log_prob, s

    def train(self):
        best_reward = -float("inf")

        for epoch in range(1, self.cfg_rl.epochs + 1):
            t0 = time.time()
            stats = {k: [] for k in self.history}
            self.agent.train()

            for batch in self.train_loader:
                batch_x, _, batch_x_mark, _ = batch
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                x_ot = batch_x[:, :, -1:]

                with torch.no_grad():
                    y_hat = self.forecaster.predict_ot(batch_x, batch_x_mark)

                keep = filter_batch(y_hat, self.filter_quantile)
                if keep.sum() == 0:
                    continue

                batch_x = batch_x[keep]
                batch_x_mark = batch_x_mark[keep]
                x_ot = x_ot[keep]
                y_hat = y_hat[keep]

                x_cf, log_prob, s = self._make_cf(x_ot, y_hat)

                with torch.no_grad():
                    y_cf = self.forecaster.predict_from_ot(
                        x_ot=x_cf, x_full=batch_x, x_mark=batch_x_mark
                    )

                reward_dict = self.reward_fn(x_ot, x_cf, y_hat, y_cf, z_cf=None)
                R = reward_dict["total"]

                rs = self.reward_fn.stats(reward_dict)
                stats["reward_total"].append(rs["total"])
                stats["r_validity"].append(rs["validity"])
                stats["r_proximity"].append(rs["proximity"])
                stats["r_reconstruction"].append(rs["reconstruction"])
                stats["r_temporal"].append(rs["temporal"])
                stats["delta_mean"].append(rs["delta_mean"])
                stats["success_rate"].append(rs["success"])
                stats["n_valid"].append(int(keep.sum()))

                V = self.agent.evaluate(s)
                mu, log_std = self.agent.actor(s)
                entropy = torch.distributions.Normal(mu, log_std.exp()).entropy().sum(dim=-1)
                loss_dict = self.agent.compute_loss(log_prob, R, V, entropy)

                self.opt_actor.zero_grad()
                loss_dict["actor"].backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.cfg_rl.grad_clip)
                self.opt_actor.step()

                self.opt_critic.zero_grad()
                loss_dict["critic"].backward()
                nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.cfg_rl.grad_clip)
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
                f"prox={means['r_proximity']:.3f} | "
                f"temp={means['r_temporal']:.3f} | "
                f"Δ={means['delta_mean']:.4f} | "
                f"SR={means['success_rate'] * 100:.1f}% | "
                f"{time.time() - t0:.1f}s"
            )

            if means["reward_total"] > best_reward:
                best_reward = means["reward_total"]
                self._save_checkpoint("best")

        self._save_checkpoint("final")
        self._save_history()
        print(f"Done | best reward = {best_reward:.4f}")

    def _save_checkpoint(self, tag):
        path = os.path.join(self.cfg_rl.checkpoint_dir_lp, f"{self.exp_name}_agent_{tag}.pt")
        torch.save(
            {
                "actor_state_dict": self.agent.actor.state_dict(),
                "critic_state_dict": self.agent.critic.state_dict(),
                "history": self.history,
                "cfg_rl": vars(self.cfg_rl),
            },
            path,
        )

    def _save_history(self):
        path = os.path.join(self.cfg_rl.results_dir_lp, f"{self.exp_name}_history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
import os
import json
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from src.models.autoencoder.tcn_ae             import TCNAutoEncoder
from src.models.actor_critic       import ActorCritic, build_state
from src.models.Reward             import CFReward, compute_threshold
from src.models.forecaster_wrapper import ForecasterWrapper
from src.models.anomaly_detector.plausibility       import EnsemblePlausibility
from src.utils.config              import load_config
from src.utils.train_tools         import get_device


def prepare_rl_data(cfg_forecaster, cfg_ae, device):
    import pandas as pd
    from src.data_provider.data_factory import data_provider

    train_data, train_loader = data_provider(cfg_forecaster, "train")
    test_data,  test_loader  = data_provider(cfg_forecaster, "test")

    ckpt_ae      = torch.load(cfg_ae.checkpoint_path, map_location="cpu", weights_only=False)
    scaler       = StandardScaler()
    scaler.mean_ = np.array(ckpt_ae["scaler_mean"], dtype=np.float64)
    scaler.scale_= np.array(ckpt_ae["scaler_std"],  dtype=np.float64)
    scaler.var_  = scaler.scale_ ** 2
    scaler.n_features_in_ = 1
    print(f"[Data] Scaler loaded from AE checkpoint ✓")

    df = pd.read_csv(os.path.join(cfg_forecaster.root_path, cfg_forecaster.data_path))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    ot        = df[["OT"]].values.astype(np.float32)
    ot_s      = scaler.transform(ot[:int(0.70 * len(ot))])
    threshold = compute_threshold(ot_s, k=-0.5)
    print(f"[Data] threshold S = {threshold:.4f}")

    return train_loader, test_loader, scaler, threshold


def run_episode(batch, ae, forecaster, agent, reward_fn, device):
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    batch_x      = batch_x.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    x_ot         = batch_x[:, :, -1:]

    with torch.no_grad():
        z     = ae.encode(x_ot)
        y_hat = forecaster.predict_ot(batch_x, batch_x_mark)

    s              = agent.build_state(z, y_hat)
    a, log_prob, _ = agent.actor.sample(s)
    z_cf           = torch.clamp(z + agent.eta * a, -1.0, 1.0)

    with torch.no_grad():
        x_cf = ae.decode(z_cf)
        y_cf = forecaster.predict_from_ot(x_ot=x_cf, x_full=batch_x, x_mark=batch_x_mark)

    reward_dict = reward_fn(x_ot, x_cf, y_cf)
    R           = reward_dict["total"]
    V           = agent.evaluate(s)

    mu, log_std = agent.actor(s)
    dist        = torch.distributions.Normal(mu, log_std.exp())
    entropy     = dist.entropy().sum(dim=-1)

    return {
        "log_prob"    : log_prob,
        "reward"      : R,
        "value"       : V,
        "entropy"     : entropy,
        "reward_dict" : reward_dict,
        "x_ot"        : x_ot,
        "x_cf"        : x_cf,
        "y_hat"       : y_hat,
        "y_cf"        : y_cf,
        "z"           : z,
        "z_cf"        : z_cf,
    }


class RLTrainer:

    def __init__(self, cfg_forecaster, cfg_ae, cfg_rl, device):
        self.cfg_f  = cfg_forecaster
        self.cfg_ae = cfg_ae
        self.cfg_rl = cfg_rl
        self.device = device

        os.makedirs(cfg_rl.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg_rl.figures_dir,    exist_ok=True)
        os.makedirs(cfg_rl.results_dir,    exist_ok=True)

        print("\n[RL] Loading frozen models ...")

        self.ae = TCNAutoEncoder.from_checkpoint(cfg_ae.checkpoint_path, device=device)
        self.ae.eval()
        for p in self.ae.parameters(): p.requires_grad_(False)

        self.forecaster = ForecasterWrapper(cfg_forecaster, device)
        for p in self.forecaster.model.parameters(): p.requires_grad_(False)

        with open(cfg_rl.plausibility_path, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, dict):
            self.plausibility = (
                obj.get("ensemble") or
                obj.get("model")    or
                obj.get("plausibility")
            )
            if self.plausibility is None:
                print(f"[RL] pkl keys: {list(obj.keys())}")
                raise ValueError("Cannot find EnsemblePlausibility in pkl")
        else:
            self.plausibility = obj

        print(f"[RL] Plausibility loaded ✓  type={type(self.plausibility).__name__}")

        (self.train_loader,
         self.test_loader,
         self.scaler,
         threshold) = prepare_rl_data(cfg_forecaster, cfg_ae, device)

        self.reward_fn = CFReward(
            threshold    = threshold,
            plausibility = self.plausibility,
            alpha        = cfg_rl.alpha,
            beta         = cfg_rl.beta,
            gamma        = cfg_rl.gamma,
        ).to(device)

        self.agent = ActorCritic(
            latent_dim   = cfg_ae.latent_dim,
            pred_len     = cfg_forecaster.pred_len,
            eta          = cfg_rl.eta,
            entropy_coef = cfg_rl.entropy_coef,
            direction    = -1.0,
        ).to(device)

        params = self.agent.count_parameters()
        print(f"[RL] Actor  params : {params['actor']:,}")
        print(f"[RL] Critic params : {params['critic']:,}")

        self.opt_actor  = torch.optim.Adam(self.agent.actor.parameters(),  lr=cfg_rl.lr_actor)
        self.opt_critic = torch.optim.Adam(self.agent.critic.parameters(), lr=cfg_rl.lr_critic)

        self.history = {
            "loss_total"    : [],
            "loss_actor"    : [],
            "loss_critic"   : [],
            "reward_total"  : [],
            "r_validity"    : [],
            "r_proximity"   : [],
            "r_plausibility": [],
            "advantage"     : [],
        }

    def train(self):
        print(f"\n{'='*60}")
        print(f"RL Training | {self.cfg_rl.epochs} epochs")
        print(f"  eta          = {self.cfg_rl.eta}")
        print(f"  lr_actor     = {self.cfg_rl.lr_actor}")
        print(f"  lr_critic    = {self.cfg_rl.lr_critic}")
        print(f"  alpha/beta/γ = {self.cfg_rl.alpha}/{self.cfg_rl.beta}/{self.cfg_rl.gamma}")
        print(f"{'='*60}")

        best_reward = -float("inf")

        for epoch in range(1, self.cfg_rl.epochs + 1):
            t0 = time.time()
            self.agent.train()
            epoch_stats = {k: [] for k in self.history}

            for batch in self.train_loader:
                ep = run_episode(batch, self.ae, self.forecaster,
                                 self.agent, self.reward_fn, self.device)

                loss_dict = self.agent.compute_loss(
                    log_prob = ep["log_prob"],
                    reward   = ep["reward"],
                    value    = ep["value"],
                    entropy  = ep["entropy"],
                )

                self.opt_actor.zero_grad()
                loss_dict["actor"].backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.cfg_rl.grad_clip)
                self.opt_actor.step()

                self.opt_critic.zero_grad()
                (0.5 * loss_dict["critic"]).backward()
                nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.cfg_rl.grad_clip)
                self.opt_critic.step()

                rs = self.reward_fn.stats(ep["reward_dict"])
                epoch_stats["loss_total"].append(float(loss_dict["total"].item()))
                epoch_stats["loss_actor"].append(float(loss_dict["actor"].item()))
                epoch_stats["loss_critic"].append(float(loss_dict["critic"].item()))
                epoch_stats["reward_total"].append(rs["total"])
                epoch_stats["r_validity"].append(rs["validity"])
                epoch_stats["r_proximity"].append(rs["proximity"])
                epoch_stats["r_plausibility"].append(rs["plausibility"])
                epoch_stats["advantage"].append(loss_dict["advantage"])

            means = {k: float(np.mean(v)) for k, v in epoch_stats.items()}
            for k, v in means.items(): self.history[k].append(v)

            dt = time.time() - t0
            print(f"Epoch {epoch:03d}/{self.cfg_rl.epochs} | "
                  f"R={means['reward_total']:.4f} | "
                  f"val={means['r_validity']:.3f} | "
                  f"prox={means['r_proximity']:.3f} | "
                  f"plaus={means['r_plausibility']:.3f} | "
                  f"loss={means['loss_total']:.4f} | "
                  f"{dt:.1f}s")

            if means["reward_total"] > best_reward:
                best_reward = means["reward_total"]
                self._save_checkpoint("best")
                print(f"  ✅ New best reward = {best_reward:.4f}")

        self._save_checkpoint("final")
        self._save_history()
        self._plot_training()
        print(f"\n✅ Training complete | best reward = {best_reward:.4f}")
        return self.history

    @torch.no_grad()
    def evaluate(self, n_batches: int = 10):
        print(f"\n[Eval] Running on test set ({n_batches} batches) ...")
        self.agent.eval()

        all_rewards, all_validity, all_prox, all_plaus = [], [], [], []
        cf_examples = []

        for i, batch in enumerate(self.test_loader):
            if i >= n_batches: break
            ep = run_episode(batch, self.ae, self.forecaster,
                             self.agent, self.reward_fn, self.device)
            rs = self.reward_fn.stats(ep["reward_dict"])
            all_rewards.append(rs["total"])
            all_validity.append(rs["validity"])
            all_prox.append(rs["proximity"])
            all_plaus.append(rs["plausibility"])

            if i < 3:
                cf_examples.append({
                    "x_ot" : ep["x_ot"][0].cpu().numpy(),
                    "x_cf" : ep["x_cf"][0].cpu().numpy(),
                    "y_hat": ep["y_hat"][0].cpu().numpy(),
                    "y_cf" : ep["y_cf"][0].cpu().numpy(),
                })

        print(f"  R_total  = {np.mean(all_rewards):.4f}")
        print(f"  R_valid  = {np.mean(all_validity):.4f}")
        print(f"  R_prox   = {np.mean(all_prox):.4f}")
        print(f"  R_plaus  = {np.mean(all_plaus):.4f}")

        self._plot_cf_examples(cf_examples)
        return cf_examples

    def _save_checkpoint(self, tag: str):
        path = os.path.join(self.cfg_rl.checkpoint_dir, f"rl_agent_{tag}.pt")
        torch.save({
            "actor_state_dict" : self.agent.actor.state_dict(),
            "critic_state_dict": self.agent.critic.state_dict(),
            "history"          : self.history,
            "cfg_rl"           : vars(self.cfg_rl),
        }, path)

    def _save_history(self):
        path = os.path.join(self.cfg_rl.results_dir, "rl_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[RL] History saved → {path}")

    def _plot_training(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        axes[0,0].plot(self.history["reward_total"], color="steelblue", lw=1.5)
        axes[0,0].set_title("Total Reward"); axes[0,0].grid(alpha=0.3)

        axes[0,1].plot(self.history["r_validity"],     label="validity",     color="green",  lw=1.5)
        axes[0,1].plot(self.history["r_proximity"],    label="proximity",    color="orange", lw=1.5)
        axes[0,1].plot(self.history["r_plausibility"], label="plausibility", color="purple", lw=1.5)
        axes[0,1].set_title("Sub-Rewards"); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

        axes[1,0].plot(self.history["loss_actor"],  label="actor",  color="coral",     lw=1.5)
        axes[1,0].plot(self.history["loss_critic"], label="critic", color="steelblue", lw=1.5)
        axes[1,0].set_title("Losses"); axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

        axes[1,1].plot(self.history["advantage"], color="gray", lw=1.5)
        axes[1,1].axhline(0, color="black", ls="--", lw=0.8)
        axes[1,1].set_title("Advantage"); axes[1,1].grid(alpha=0.3)

        plt.suptitle("RL Training — CF Forecasting", fontsize=13)
        plt.tight_layout()
        path = os.path.join(self.cfg_rl.figures_dir, "rl_training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[RL] Training curves → {path}")

    def _plot_cf_examples(self, examples: list):
        n = len(examples)
        fig, axes = plt.subplots(n, 2, figsize=(14, 4*n))
        if n == 1: axes = axes[np.newaxis, :]

        for i, ex in enumerate(examples):
            x_ot  = ex["x_ot"][:, 0]
            x_cf  = ex["x_cf"][:, 0]
            y_hat = ex["y_hat"][:, 0]
            y_cf  = ex["y_cf"][:, 0]

            axes[i,0].plot(x_ot, color="steelblue", lw=1.5, label="Original x")
            axes[i,0].plot(x_cf, color="coral",     lw=1.5, ls="--", label="CF x_cf")
            axes[i,0].set_title(f"Sample {i+1} — Input")
            axes[i,0].legend(fontsize=8); axes[i,0].grid(alpha=0.3)

            axes[i,1].plot(y_hat, color="steelblue", lw=1.5, label="ŷ original")
            axes[i,1].plot(y_cf,  color="coral",     lw=1.5, ls="--", label="ŷ_cf")
            axes[i,1].axhline(self.reward_fn.threshold, color="green", ls=":", lw=1.5, label="threshold S")
            axes[i,1].set_title(f"Sample {i+1} — Forecast")
            axes[i,1].legend(fontsize=8); axes[i,1].grid(alpha=0.3)

        plt.suptitle("CF Examples — Original vs Counterfactual", fontsize=13)
        plt.tight_layout()
        path = os.path.join(self.cfg_rl.figures_dir, "rl_cf_examples.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[RL] CF examples → {path}")
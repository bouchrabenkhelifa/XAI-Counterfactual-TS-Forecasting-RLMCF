import os
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from src.models.tcn_ae import TCNAutoEncoder
from src.models.forecaster_wrapper import ForecasterWrapper
from src.models.RL_without_ae.latent_plausibility import LatentPlausibility
from src.models.optimization_strategy.cf_objective import CounterfactualObjective
from src.models.optimization_strategy.latent_optimizer import (
    LatentCounterfactualOptimizer,
)


def prepare_optim_data(cfg_forecaster, cfg_ae):
    from src.data_provider.data_factory import data_provider

    _, train_loader = data_provider(cfg_forecaster, "train")
    _, test_loader = data_provider(cfg_forecaster, "test")

    ckpt_ae = torch.load(
        cfg_ae.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    scaler = StandardScaler()
    scaler.mean_ = np.array(ckpt_ae["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.array(ckpt_ae["scaler_std"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = 1

    return train_loader, test_loader, scaler


def load_latent_plausibility(path, ae, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    plaus = LatentPlausibility(
        ae=ae,
        alpha=ckpt["alpha"],
        scale=ckpt["scale"],
    )
    plaus.z_mean = ckpt["z_mean"].to(device)
    plaus.z_std = ckpt["z_std"].to(device)
    plaus.sigma = ckpt["sigma"]
    plaus.fitted_ = True
    return plaus


def filter_batch(y_hat, quantile=0.75):
    mean_yhat = y_hat[:, :, 0].mean(dim=1)
    thr = torch.quantile(mean_yhat, quantile)
    return mean_yhat >= thr


class OptimizationTrainer:
    """
    Evaluation d'une stratégie d'optimisation directe du latent.
    """

    def __init__(self, cfg_forecaster, cfg_ae, cfg_optim, device):
        self.cfg_f = cfg_forecaster
        self.cfg_ae = cfg_ae
        self.cfg_optim = cfg_optim
        self.device = device

        for d in [
            cfg_optim.checkpoint_dir,
            cfg_optim.figures_dir,
            cfg_optim.results_dir,
        ]:
            os.makedirs(d, exist_ok=True)

        self.ae = TCNAutoEncoder.from_checkpoint(
            cfg_ae.checkpoint_path,
            device=device,
        )
        self.ae.eval()
        for p in self.ae.parameters():
            p.requires_grad_(False)

        self.forecaster = ForecasterWrapper(cfg_forecaster, device)
        self.forecaster.model.eval()
        for p in self.forecaster.model.parameters():
            p.requires_grad_(False)

        self.plausibility = None
        if getattr(cfg_optim, "latent_plaus_path", None):
            self.plausibility = load_latent_plausibility(
                cfg_optim.latent_plaus_path,
                self.ae,
                device,
            )

        self.train_loader, self.test_loader, self.scaler = prepare_optim_data(
            cfg_forecaster,
            cfg_ae,
        )

        self.objective = CounterfactualObjective(
            rho=cfg_optim.rho,
            alpha=cfg_optim.alpha,
            beta_x=cfg_optim.beta_x,
            beta_z=cfg_optim.beta_z,
            gamma=cfg_optim.gamma,
            plausibility=self.plausibility,
        )

        self.cf_optimizer = LatentCounterfactualOptimizer(
            ae=self.ae,
            forecaster=self.forecaster,
            objective=self.objective,
            steps=cfg_optim.steps,
            lr=cfg_optim.lr,
            clip_latent=True,
            latent_min=-1.0,
            latent_max=1.0,
            patience=cfg_optim.patience,
            tol=cfg_optim.tol,
            device=device,
        )

        self.history = {
            "total": [],
            "goal": [],
            "prox_x": [],
            "prox_z": [],
            "plaus": [],
            "delta_mean": [],
            "success": [],
            "best_step": [],
        }

    def _run_loader(self, loader, max_batches=None, split="test"):
        stats = {k: [] for k in self.history.keys()}
        cf_examples = []

        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            batch_x, _, batch_x_mark, _ = batch
            batch_x = batch_x.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            x_ot = batch_x[:, :, -1:]

            with torch.no_grad():
                y_hat_full = self.forecaster.predict_ot(batch_x, batch_x_mark)

            mask = filter_batch(
                y_hat_full,
                quantile=getattr(self.cfg_optim, "filter_quantile", 0.75),
            )

            if mask.sum() == 0:
                continue

            x_ot = x_ot[mask]
            batch_x_f = batch_x[mask]
            batch_x_mark_f = batch_x_mark[mask]
            y_hat = y_hat_full[mask]

            result = self.cf_optimizer.optimize(
                x_ot=x_ot,
                x_full=batch_x_f,
                x_mark=batch_x_mark_f,
                y_hat=y_hat,
            )

            loss_stats = self.objective.stats(result["loss_dict"])
            stats["total"].append(loss_stats["total"])
            stats["goal"].append(loss_stats["goal"])
            stats["prox_x"].append(loss_stats["prox_x"])
            stats["prox_z"].append(loss_stats["prox_z"])
            stats["plaus"].append(loss_stats["plaus"])
            stats["delta_mean"].append(loss_stats["delta_mean"])
            stats["success"].append(loss_stats["success"])
            stats["best_step"].append(float(result["best_step"]))

            if len(cf_examples) < 4:
                cf_examples.append(
                    {
                        "x_ot": x_ot[0].detach().cpu().numpy(),
                        "x_cf": result["x_cf"][0].detach().cpu().numpy(),
                        "y_hat": result["y_hat"][0].detach().cpu().numpy(),
                        "y_cf": result["y_cf"][0].detach().cpu().numpy(),
                    }
                )

        mean_stats = {
            k: float(np.mean(v)) if len(v) > 0 else 0.0
            for k, v in stats.items()
        }

        print(f"\n[{split.upper()}] Optimization-based CF")
        for k, v in mean_stats.items():
            print(f"  {k:12s}: {v:.4f}")

        return mean_stats, cf_examples

    def evaluate(self, n_batches=20):
        t0 = time.time()

        mean_stats, cf_examples = self._run_loader(
            self.test_loader,
            max_batches=n_batches,
            split="test",
        )

        for k in self.history:
            self.history[k].append(mean_stats[k])

        self._save_history()
        self._plot_cf_examples(cf_examples)

        print(f"\nDone in {time.time() - t0:.1f}s")
        return mean_stats, cf_examples

    def _save_history(self):
        path = os.path.join(
            self.cfg_optim.results_dir,
            "optimization_history.json",
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        print(f"[Optimization] History -> {path}")

    def _plot_cf_examples(self, cf_examples):
        if not cf_examples:
            return

        n = len(cf_examples)
        fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n))
        if n == 1:
            axes = np.array([axes])

        for i, ex in enumerate(cf_examples):
            axes[i, 0].plot(ex["x_ot"].squeeze(), label="x")
            axes[i, 0].plot(ex["x_cf"].squeeze(), label="x_cf")
            axes[i, 0].set_title(f"Input vs Counterfactual #{i + 1}")
            axes[i, 0].legend()
            axes[i, 0].grid(alpha=0.3)

            axes[i, 1].plot(ex["y_hat"].squeeze(), label="y_hat")
            axes[i, 1].plot(ex["y_cf"].squeeze(), label="y_cf")
            axes[i, 1].set_title(f"Forecast vs Counterfactual Forecast #{i + 1}")
            axes[i, 1].legend()
            axes[i, 1].grid(alpha=0.3)

        fig.tight_layout()
        out = os.path.join(
            self.cfg_optim.figures_dir,
            "optimization_cf_examples.png",
        )
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[Optimization] Figure -> {out}")
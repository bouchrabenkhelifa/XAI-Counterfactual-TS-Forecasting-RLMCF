# src/training/optimization_trainer.py
# ─────────────────────────────────────────────────────────────
# Trainer for counterfactual generation via latent optimization.
# Wraps WachterLatent for batch evaluation on ETTh1.
# ─────────────────────────────────────────────────────────────

import os
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from src.models.tcn_ae import TCNAutoEncoder
from src.models.forecaster_wrapper import ForecasterWrapper
from src.models.optimization_strategy.wachter_latent import (
    WachterLatent, WachterLatentConfig
)

def prepare_data(cfg_forecaster, cfg_ae, device):
    from src.data_provider.data_factory import data_provider

    _, train_loader = data_provider(cfg_forecaster, "train")
    _, test_loader  = data_provider(cfg_forecaster, "test")

    print("[Data] Loaders ready ✓")
    return train_loader, test_loader, None

class OptimizationTrainer:
    """
    Evaluates WachterLatent CF generation on the test set.

    Unlike RL, there is no training phase — each CF is generated
    by running an optimization loop per instance at inference time.
    """

    def __init__(self, cfg_forecaster, cfg_ae, cfg_opt, device):
        self.cfg_f   = cfg_forecaster
        self.cfg_ae  = cfg_ae
        self.cfg_opt = cfg_opt
        self.device  = device

        for d in [cfg_opt.figures_dir, cfg_opt.results_dir]:
            os.makedirs(d, exist_ok=True)

        print("\n[OptTrainer] Loading frozen models ...")

        # AE — frozen
        self.ae = TCNAutoEncoder.from_checkpoint(
            cfg_ae.checkpoint_path, device=device)
        self.ae.eval()
        for p in self.ae.parameters():
            p.requires_grad_(False)

        # Forecaster — frozen
        self.forecaster = ForecasterWrapper(cfg_forecaster, device)
        for p in self.forecaster.model.parameters():
            p.requires_grad_(False)

        self.train_loader, self.test_loader, self.scaler = \
            prepare_data(cfg_forecaster, cfg_ae, device)

        # Build WachterLatent
        opt_cfg = WachterLatentConfig(
            rho          = cfg_opt.rho,
            lambda_prox  = cfg_opt.lambda_prox,
            lambda_plaus = cfg_opt.lambda_plaus,
            n_steps      = cfg_opt.n_steps,
            lr           = cfg_opt.lr,
            n_restarts   = cfg_opt.n_restarts,
            noise_std    = cfg_opt.noise_std,
            sigma_plaus  = cfg_opt.sigma_plaus,
        )
        self.generator = WachterLatent(
            ae         = self.ae,
            forecaster = self.forecaster,
            cfg        = opt_cfg,
        )

        # Fit latent statistics
        self.generator.fit(self.train_loader, device)

        # Fit anomaly detectors on train set
        self._fit_detectors()

        print(f"[OptTrainer] Ready | rho={cfg_opt.rho} | "
              f"n_steps={cfg_opt.n_steps} | n_restarts={cfg_opt.n_restarts}")

    # ──────────────────────────────────────────────
    # Fit anomaly detectors
    # ──────────────────────────────────────────────
    def _fit_detectors(self):
        """Fit IF, LOF, OC-SVM on training windows."""
        print("[OptTrainer] Fitting anomaly detectors ...")
        xs = []
        with torch.no_grad():
            for batch in self.train_loader:
                x = batch[0].float()
                x_ot = x[:, :, -1]                    # (B, seq_len)
                xs.append(x_ot.numpy())

        X_train = np.concatenate(xs, axis=0)           # (N, seq_len)

        self.if_model  = IsolationForest(
            contamination=0.1, random_state=42).fit(X_train)
        self.lof_model = LocalOutlierFactor(
            novelty=True, contamination=0.1).fit(X_train)
        self.svm_model = OneClassSVM(
            kernel="rbf", nu=0.1).fit(X_train)

        print(f"[OptTrainer] Detectors fitted on {len(X_train)} windows ✓")

    def _plausibility_scores(self, x_cf_np: np.ndarray) -> dict:
        """
        Compute IF, LOF, OC-SVM plausibility scores.
        Higher = more plausible (normalized to [0,1]).
        """
        # Raw scores (higher = more normal for IF/LOF/SVM)
        if_raw  = self.if_model.score_samples(x_cf_np)
        lof_raw = self.lof_model.score_samples(x_cf_np)
        svm_raw = self.svm_model.score_samples(x_cf_np)

        # Normalize to [0,1] via sigmoid
        def norm(s):
            s = np.array(s, dtype=np.float32)
            return float(torch.sigmoid(
                torch.tensor(s)).mean().item())

        return {
            "plausibility_if"   : norm(if_raw),
            "plausibility_lof"  : norm(lof_raw),
            "plausibility_ocsvm": norm(svm_raw),
            "plausibility"      : (norm(if_raw)
                                   + norm(lof_raw)
                                   + norm(svm_raw)) / 3,
        }

    # ──────────────────────────────────────────────
    # Evaluate on test set
    # ──────────────────────────────────────────────
    def evaluate(self, n_batches: int = 20) -> dict:
        """
        Generate CFs for n_batches test batches and compute metrics.

        Returns:
            results dict with all metrics
        """
        print(f"\n[OptTrainer] Evaluating {n_batches} test batches ...")
        print(f"  n_steps={self.cfg_opt.n_steps} | "
              f"n_restarts={self.cfg_opt.n_restarts} | "
              f"lr={self.cfg_opt.lr}")

        all_metrics = {
            "delta_mean"   : [],
            "relative_red" : [],
            "success"      : [],
            "plausibility" : [],
            "plaus_if"     : [],
            "plaus_lof"    : [],
            "plaus_ocsvm"  : [],
        }
        cf_examples  = []
        t_total      = 0.0

        for i, batch in enumerate(self.test_loader):
            if i >= n_batches:
                break

            t0 = time.time()
            result = self.generator.generate(
                batch[0], batch[2], self.device)
            t_total += time.time() - t0

            # Plausibility via anomaly detectors
            x_cf_np = result["x_cf"].squeeze(-1).numpy()  # (B, seq_len)
            plaus    = self._plausibility_scores(x_cf_np)

            all_metrics["delta_mean"].append(
                float(result["delta_mean"].mean().item()))
            all_metrics["relative_red"].append(
                float(result["rel_red"].mean().item()))
            all_metrics["success"].append(
                float(result["success"].mean().item()))
            all_metrics["plausibility"].append(plaus["plausibility"])
            all_metrics["plaus_if"].append(plaus["plausibility_if"])
            all_metrics["plaus_lof"].append(plaus["plausibility_lof"])
            all_metrics["plaus_ocsvm"].append(plaus["plausibility_ocsvm"])

            if len(cf_examples) < 4:
                cf_examples.append({
                    "x_ot" : result["x_ot"][0].numpy(),
                    "x_cf" : result["x_cf"][0].numpy(),
                    "y_hat": result["y_hat"][0].numpy(),
                    "y_cf" : result["y_cf"][0].numpy(),
                    "success": float(result["success"][0].item()),
                    "red"    : float(result["rel_red"][0].item()) * 100,
                })

            if (i + 1) % 5 == 0:
                print(f"  Batch {i+1:03d}/{n_batches} | "
                      f"SR={np.mean(all_metrics['success'])*100:.1f}% | "
                      f"plaus={np.mean(all_metrics['plausibility']):.4f} | "
                      f"Δ={np.mean(all_metrics['delta_mean']):.4f}")

        # Summary
        summary = {k: float(np.mean(v)) for k, v in all_metrics.items()}
        summary["time_total_s"] = t_total
        summary["time_per_batch_s"] = t_total / max(1, n_batches)

        print(f"\n── Résultats Wachter Latent ──────────────────────────")
        for k, v in summary.items():
            print(f"  {k:25s} = {v:.4f}")

        # Save results
        path = os.path.join(self.cfg_opt.results_dir,
                            "wachter_latent_results.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[OptTrainer] Results → {path}")

        # Plots
        self._plot_cf_examples(cf_examples)

        return summary

    # ──────────────────────────────────────────────
    # Plots
    # ──────────────────────────────────────────────
    def _plot_cf_examples(self, examples):
        if not examples:
            return

        n   = len(examples)
        fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
        if n == 1:
            axes = [axes]

        for i, ex in enumerate(examples):
            x_ot  = ex["x_ot"][:, 0]
            x_cf  = ex["x_cf"][:, 0]
            y_hat = ex["y_hat"][:, 0]
            y_cf  = ex["y_cf"][:, 0]

            full_orig = np.concatenate([x_ot,  y_hat])
            full_cf   = np.concatenate([x_cf,  y_cf])
            t_all     = np.arange(len(full_orig))

            ok = "✓" if ex["success"] else "✗"
            axes[i].plot(t_all, full_orig, color="steelblue",
                         lw=1.5, label="x + forecast(x)")
            axes[i].plot(t_all, full_cf, color="coral",
                         lw=1.5, ls="--", label="x_cf + forecast(x_cf)")
            axes[i].axvline(len(x_ot), color="gray", ls="--", lw=1.2)
            axes[i].fill_between(t_all, full_orig, full_cf,
                                  alpha=0.12, color="coral")
            axes[i].set_title(
                f"Sample {i+1} — {ex['red']:+.1f}% {ok}", fontsize=11)
            axes[i].legend(fontsize=9)
            axes[i].grid(alpha=0.3)

        plt.suptitle(
            f"CF Wachter Latent — ETTh1 | "
            f"rho={self.cfg_opt.rho*100:.0f}% | "
            f"steps={self.cfg_opt.n_steps}",
            fontsize=13
        )
        plt.tight_layout()
        path = os.path.join(
            self.cfg_opt.figures_dir, "wachter_latent_cf_examples.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[OptTrainer] CF examples → {path}")
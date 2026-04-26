"""
PyTorch reimplementation of ForecastCF for ETTh1.

Faithfully reproduces the algorithm from:
  "Counterfactual Explanations for Time Series Forecasting"
  Wang et al., ICDM 2023

Key design choices (matching their paper exactly):
  - generate_bounds: center=median(x), bounds = sv*(1 ± poly_values ± fraction_std*std)
  - CF search: gradient descent on x directly (not in latent space)
  - Metrics: validity_ratio, stepwise_auc, proximity_L2, compactness (atol=0.01)
  - Loss: pred_margin_weight * margin_mse + (1-pred_margin_weight) * weighted_mae
"""

import time
import json
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ─────────────────────────────────────────────
# 1.  Bounds generation  (exact replica of their generate_bounds)
# ─────────────────────────────────────────────

def polynomial_values(shift, change_percent, poly_order, horizon):
    """Replica of ForecastCF polynomial_values()."""
    if horizon == 1:
        return np.asarray([shift + change_percent])
    p_orders = [shift]
    p_orders.extend([0] * poly_order)
    p_orders[-1] = change_percent / ((horizon - 1) ** poly_order)
    p = np.polynomial.Polynomial(p_orders)
    p_coefs = list(reversed(p.coef))
    return np.asarray([np.polyval(p_coefs, i) for i in range(horizon)])


def generate_bounds(input_series, horizon, shift=0.0, change_percent=-0.1,
                    poly_order=1, fraction_std=1.0, center="median"):
    """
    Exact replica of ForecastCF generate_bounds().
    input_series : (back_horizon, 1)  numpy array
    Returns upper, lower : (horizon, 1) numpy arrays
    """
    if center == "median":
        sv = np.median(input_series)
    elif center == "mean":
        sv = np.mean(input_series)
    elif center == "last":
        sv = input_series[-1, 0]
    else:
        sv = np.median(input_series)

    std = np.std(input_series)
    poly = polynomial_values(shift, change_percent, poly_order, horizon)

    upper = sv * (1 + poly + fraction_std * std)   # shape (horizon,)
    lower = sv * (1 + poly - fraction_std * std)

    # swap if sv < 0  (their code does this implicitly via multiplication)
    if sv < 0:
        upper, lower = lower, upper

    return upper[:, np.newaxis], lower[:, np.newaxis]   # (horizon, 1)


# ─────────────────────────────────────────────
# 2.  ForecastCF optimizer  (gradient descent per instance)
# ─────────────────────────────────────────────

class ForecastCF_PyTorch:
    """
    PyTorch reimplementation of ForecastCF._transform_sample().
    Operates directly in input space (no latent space).
    """

    def __init__(self, forecaster, device,
                 max_iter=100, lr=1e-3,
                 pred_margin_weight=0.25,
                 step_weights=None):
        self.forecaster = forecaster
        self.device = device
        self.max_iter = max_iter
        self.lr = lr
        self.pred_margin_weight = pred_margin_weight
        self.weighted_steps_weight = 1.0 - pred_margin_weight
        self.step_weights = step_weights   # (1, back_horizon, 1) or None

    def _margin_mse(self, pred, upper, lower):
        """MSE loss only for out-of-bound timesteps."""
        in_band = (pred <= upper) & (pred >= lower)
        out = ~in_band
        if not out.any():
            return torch.tensor(0.0, device=self.device)
        pred_out = pred[out]
        upper_out = upper[out]
        lower_out = lower[out]
        dist = ((pred_out - upper_out) ** 2 + (pred_out - lower_out) ** 2).mean()
        return dist

    def _weighted_mae(self, x_orig, x_cf, weights):
        diff = (x_orig - x_cf).abs()
        if weights is not None:
            diff = diff * weights
        return diff.mean()

    def transform_sample(self, x_np, upper_np, lower_np, x_full=None, x_mark=None):
        """
        x_np     : (1, back_horizon, 1)  numpy
        upper_np : (horizon, 1)          numpy
        lower_np : (horizon, 1)          numpy
        Returns x_cf_np (1, back_horizon, 1), y_cf_np (1, horizon, 1)
        """
        x_orig = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        upper  = torch.tensor(upper_np[np.newaxis], dtype=torch.float32, device=self.device)  # (1,H,1)
        lower  = torch.tensor(lower_np[np.newaxis], dtype=torch.float32, device=self.device)

        x_cf = nn.Parameter(x_orig.clone())
        optimizer = torch.optim.Adam([x_cf], lr=self.lr)

        weights = None
        if self.step_weights is not None:
            weights = torch.tensor(self.step_weights, dtype=torch.float32, device=self.device)

        for _ in range(self.max_iter):
            optimizer.zero_grad()

            # predict with current x_cf
            with torch.enable_grad():
                if x_full is not None:
                    # replace OT channel in x_full with x_cf
                    x_full_cf = x_full.clone()
                    x_full_cf[:, :, -1:] = x_cf
                    pred = self.forecaster.predict_from_ot(
                        x_ot=x_cf, x_full=x_full_cf, x_mark=x_mark)
                else:
                    pred = self.forecaster(x_cf)

            # check convergence
            in_band = (pred <= upper) & (pred >= lower)
            if in_band.all():
                break

            margin_loss  = self._margin_mse(pred, upper, lower)
            proxim_loss  = self._weighted_mae(x_orig, x_cf, weights)
            loss = self.pred_margin_weight * margin_loss + self.weighted_steps_weight * proxim_loss

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            if x_full is not None:
                x_full_cf = x_full.clone()
                x_full_cf[:, :, -1:] = x_cf
                y_cf = self.forecaster.predict_from_ot(
                    x_ot=x_cf, x_full=x_full_cf, x_mark=x_mark)
            else:
                y_cf = self.forecaster(x_cf)

        return x_cf.detach().cpu().numpy(), y_cf.detach().cpu().numpy()

    def transform(self, X_np, uppers, lowers, X_full=None, X_mark=None):
        """
        X_np   : (N, back_horizon, 1)
        uppers : list of (horizon, 1)
        lowers : list of (horizon, 1)
        Returns X_cf (N, back_horizon, 1), Y_cf (N, horizon, 1)
        """
        N = X_np.shape[0]
        X_cf = np.empty_like(X_np)
        Y_cf = np.empty((N, uppers[0].shape[0], 1))

        for i in range(N):
            if i % 50 == 0:
                print(f"  ForecastCF: {i+1}/{N} samples...")
            x_full_i = X_full[i:i+1] if X_full is not None else None
            x_mark_i = X_mark[i:i+1] if X_mark is not None else None
            xcf_i, ycf_i = self.transform_sample(
                X_np[i:i+1], uppers[i], lowers[i], x_full_i, x_mark_i)
            X_cf[i] = xcf_i[0]
            Y_cf[i] = ycf_i[0]

        return X_cf, Y_cf


# ─────────────────────────────────────────────
# 3.  Metrics  (exact replica of their cf_metrics)
# ─────────────────────────────────────────────

def validity_ratio(y_cf, uppers, lowers):
    """y_cf: (N,H,1), uppers/lowers: (N,H,1)"""
    valid = (y_cf <= uppers) & (y_cf >= lowers)
    return valid.mean(axis=1).mean()


def stepwise_auc(y_cf, uppers, lowers):
    """Replica of cumulative_valid_steps AUC."""
    import pandas as pd
    n_samples, n_steps, _ = y_cf.shape
    in_band = (y_cf <= uppers) & (y_cf >= lowers)   # (N,H,1)
    in_band = in_band[:, :, 0]                        # (N,H)

    until_steps = np.zeros(n_samples)
    for i in range(n_samples):
        count = 0
        for s in range(n_steps):
            if in_band[i, s]:
                count += 1
                until_steps[i] = count
            else:
                until_steps[i] = count
                break

    valid_steps, counts = np.unique(until_steps, return_counts=True)
    cumsum_counts = np.flip(np.cumsum(np.flip(counts)))

    # fillna
    df = pd.DataFrame(
        [{k: v for k, v in zip(valid_steps, cumsum_counts)}],
        columns=list(range(0, n_steps + 1)))
    df = df.sort_index(ascending=True, axis=1)
    df = df.fillna(method="backfill", axis=1)
    df = df.fillna(value=0)
    vs = df.columns.to_numpy()
    cc = df.values[0]

    auc = np.trapz(cc[1:] / n_samples, vs[1:] / n_steps)
    return auc


def proximity_l2(x_orig, x_cf):
    """Mean L2 distance per sample."""
    return np.linalg.norm(x_orig - x_cf, axis=1).mean()


def compactness_score(x_orig, x_cf, atol=0.01):
    """Fraction of unchanged timesteps (atol=0.01 as in their code)."""
    return np.isclose(x_orig, x_cf, atol=atol).mean(axis=1).mean()


# ─────────────────────────────────────────────
# 4.  Main runner
# ─────────────────────────────────────────────

def run_forecastcf_etth1(cfg_forecaster, cfg_ae, cfg_rl, device,
                          desired_change=-0.1, fraction_std=1.0,
                          poly_order=1, shift=0.0, center="median",
                          max_iter=100, lr=1e-3, pred_margin_weight=0.25,
                          n_eval_batches=20, results_dir="assets/results/forecastcf_baseline",
                          figures_dir="assets/figures/forecastcf_baseline"):
    """
    Run ForecastCF (PyTorch) on ETTh1 using the same forecaster and data split
    as our RL method, for a fair comparison.
    """
    from src.training.RL_trainers.trainer_last import RLMaskTrainer, run_episode_eval

    print("=" * 60)
    print("ForecastCF Baseline (PyTorch reimplementation)")
    print(f"  desired_change={desired_change}  fraction_std={fraction_std}")
    print(f"  poly_order={poly_order}  center={center}")
    print(f"  max_iter={max_iter}  lr={lr}")
    print("=" * 60)

    # Load trainer (for data + forecaster)
    trainer = RLMaskTrainer(cfg_forecaster, cfg_ae, cfg_rl, device)
    forecaster = trainer.forecaster
    forecaster.eval()

    # Build CF model
    cf_model = ForecastCF_PyTorch(
        forecaster=forecaster,
        device=device,
        max_iter=max_iter,
        lr=lr,
        pred_margin_weight=pred_margin_weight,
    )

    # Collect test batches
    all_x, all_x_full, all_x_mark, all_yhat = [], [], [], []
    for i, batch in enumerate(trainer.test_loader):
        if i >= n_eval_batches:
            break
        batch_x, _, batch_x_mark, _ = batch
        batch_x      = batch_x.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        x_ot = batch_x[:, :, -1:]
        with torch.no_grad():
            y_hat = forecaster.predict_ot(batch_x, batch_x_mark)
        all_x.append(x_ot.cpu().numpy())
        all_x_full.append(batch_x.cpu().numpy())
        all_x_mark.append(batch_x_mark.cpu().numpy())
        all_yhat.append(y_hat.cpu().numpy())

    all_x      = np.concatenate(all_x)       # (N, BH, 1)
    all_x_full = np.concatenate(all_x_full)  # (N, BH, C)
    all_x_mark = np.concatenate(all_x_mark)  # (N, BH, D)
    all_yhat   = np.concatenate(all_yhat)    # (N, H, 1)
    N, BH, _   = all_x.shape
    H          = all_yhat.shape[1]

    print(f"Generating bounds for {N} samples (H={H}, BH={BH})...")
    uppers, lowers = [], []
    for i in range(N):
        u, l = generate_bounds(
            all_x[i], H,
            shift=shift,
            change_percent=desired_change,
            poly_order=poly_order,
            fraction_std=fraction_std,
            center=center,
        )
        uppers.append(u)
        lowers.append(l)

    uppers_np = np.stack(uppers)   # (N, H, 1)
    lowers_np = np.stack(lowers)   # (N, H, 1)

    print(f"Running ForecastCF gradient search ({max_iter} iters/sample)...")
    t0 = time.time()

    # Convert to torch for forecaster calls
    X_full_t = torch.tensor(all_x_full, dtype=torch.float32, device=device)
    X_mark_t = torch.tensor(all_x_mark, dtype=torch.float32, device=device)

    X_cf, Y_cf = cf_model.transform(
        all_x, uppers, lowers, X_full_t, X_mark_t)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s  ({elapsed/N:.2f}s/sample)")

    # ── Metrics ──
    vr   = validity_ratio(Y_cf, uppers_np, lowers_np)
    sauc = stepwise_auc(Y_cf, uppers_np, lowers_np)
    prox = proximity_l2(all_x, X_cf)
    comp = compactness_score(all_x, X_cf)

    print()
    print("── ForecastCF Metrics (ETTh1, iTransformer) ──")
    print(f"  Validity Ratio  ↑ : {vr:.4f}")
    print(f"  Step AUC        ↑ : {sauc:.4f}")
    print(f"  Proximity L2    ↓ : {prox:.4f}")
    print(f"  Compactness     ↑ : {comp:.4f}")
    print(f"  Runtime           : {elapsed:.1f}s for {N} samples")

    # ── Save results ──
    os.makedirs(results_dir, exist_ok=True)
    results = {
        "method": "ForecastCF_PyTorch",
        "dataset": "ETTh1",
        "forecaster": cfg_forecaster.model,
        "desired_change": desired_change,
        "fraction_std": fraction_std,
        "poly_order": poly_order,
        "center": center,
        "max_iter": max_iter,
        "lr": lr,
        "n_samples": N,
        "runtime_seconds": round(elapsed, 2),
        "metrics": {
            "validity_ratio": round(float(vr),   4),
            "step_auc":       round(float(sauc), 4),
            "proximity_l2":   round(float(prox), 4),
            "compactness":    round(float(comp), 4),
        }
    }
    out_path = os.path.join(results_dir, "forecastcf_etth1_evaluation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n[Saved] {out_path}")

    # ── Figures ──
    os.makedirs(figures_dir, exist_ok=True)
    _plot_cf_examples(all_x, X_cf, all_yhat, Y_cf, uppers_np, lowers_np,
                      figures_dir, n=4)

    return results


def _plot_cf_examples(x, x_cf, y_hat, y_cf, uppers, lowers, figures_dir, n=4):
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), sharey=False)
    BH = x.shape[1]
    H  = y_hat.shape[1]

    for idx in range(n):
        ax = axes[idx]
        t_back = np.arange(BH)
        t_fore = np.arange(BH, BH + H)

        ax.plot(t_back, x[idx, :, 0],    color="steelblue",  lw=1.5, label="x original")
        ax.plot(t_back, x_cf[idx, :, 0], color="darkorange", lw=1.5, linestyle="--", label="x CF")
        ax.plot(t_fore, y_hat[idx, :, 0], color="steelblue",  lw=2,   label="y_hat")
        ax.plot(t_fore, y_cf[idx, :, 0],  color="darkorange", lw=2,   linestyle="--", label="y_cf")
        ax.fill_between(t_fore,
                        lowers[idx, :, 0], uppers[idx, :, 0],
                        alpha=0.25, color="green", label="target band")
        ax.axvline(BH, color="gray", linestyle=":", lw=1)

        vh = ((y_cf[idx] >= lowers[idx]) & (y_cf[idx] <= uppers[idx])).mean()
        ax.set_title(f"Sample {idx+1}  vh={vh:.2f}")
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = os.path.join(figures_dir, "forecastcf_etth1_cf_examples.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[Figure] {out}")

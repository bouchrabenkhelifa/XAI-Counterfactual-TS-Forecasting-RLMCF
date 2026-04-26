import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
CONFIG_FORECASTER = "assets/configs/models/etth1_dataset/itransformer/etth1_96_48_S.json"
CONFIG_AE         = "assets/configs/models/etth1_dataset/ae/tcn_ae.json"
CONFIG_RL         = "assets/configs/models/etth1_dataset/RL_ablations/cf.json"

CHECKPOINT_RL = (
    "assets/checkpoints/etth1_chpts/RL_bounded_goal/"
    "alphabeta_timestep_agent_best.pt"
)

OUTPUT_DIR  = "assets/results/rl_eval"
FIGURES_DIR = "assets/figures/rl_eval"

N_EVAL_BATCHES = 50
RHO            = 0.10
TOL_COMPACT    = 1e-3


# ═════════════════════════════════════════════════════════════════════════════
#  Bounds
# ═════════════════════════════════════════════════════════════════════════════

def compute_bounds_np(y_hat: np.ndarray, rho: float = RHO):
    if y_hat.ndim == 3:
        y_hat = y_hat[:, :, 0]
    target = y_hat - rho * np.abs(y_hat)
    eps    = 0.10 * np.abs(y_hat)
    return target - eps, target + eps, target


def _2d(arr):
    return arr[:, :, 0] if arr.ndim == 3 else arr


# ═════════════════════════════════════════════════════════════════════════════
#  Temporal mask
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _build_mask(B, L, C, last_k, ramp_k, device):
    m = torch.zeros(B, L, C, device=device)
    start = max(0, L - last_k)
    m[:, start:, :] = 1.0
    if ramp_k > 0:
        rs = max(0, start - ramp_k)
        rl = start - rs
        if rl > 0:
            ramp = torch.linspace(0, 1, rl, device=device).view(1, rl, 1)
            m[:, rs:start, :] = ramp
    return m


# ═════════════════════════════════════════════════════════════════════════════
#  Episode runner
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_episode(trainer, batch):
    batch_x, _, batch_x_mark, _ = batch
    batch_x      = batch_x.float().to(trainer.device)
    batch_x_mark = batch_x_mark.float().to(trainer.device)
    x_ot         = batch_x[:, :, -1:]

    z     = trainer.ae_arch.encode(x_ot)
    y_hat = trainer.forecaster.predict_ot(batch_x, batch_x_mark)

    s     = trainer.agent.build_state(z, y_hat)
    mu, _ = trainer.agent.actor(s)
    z_cf  = torch.clamp(z + trainer.agent.eta * mu, -1.0, 1.0)

    x_prop = trainer.ae_arch.decode(z_cf)
    delta  = x_prop - x_ot

    if hasattr(trainer, "mask_last_k"):
        tmask = _build_mask(
            x_ot.shape[0], x_ot.shape[1], x_ot.shape[2],
            trainer.mask_last_k,
            getattr(trainer, "mask_ramp_k", 0),
            trainer.device,
        )
        if getattr(trainer, "alpha_hf", 0) > 0:
            x_lf  = trainer.ae_arch.decode(trainer.ae_arch.encode(x_ot))
            delta = delta + trainer.alpha_hf * (x_ot - x_lf)
        x_cf = x_ot + tmask * delta
    else:
        x_cf = x_prop

    y_cf = trainer.forecaster.predict_from_ot(
        x_ot=x_cf, x_full=batch_x, x_mark=batch_x_mark
    )

    return {
        "x_ot":  x_ot.cpu().numpy(),
        "x_cf":  x_cf.cpu().numpy(),
        "y_hat": y_hat.cpu().numpy(),
        "y_cf":  y_cf.cpu().numpy(),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Collect
# ═════════════════════════════════════════════════════════════════════════════

def collect_rl_data(trainer):
    trainer.agent.eval()
    X_l, Xcf_l, Yh_l, Yc_l = [], [], [], []
    examples = []

    print(f"\n[RL] Collecting {N_EVAL_BATCHES} batches …")
    for i, batch in enumerate(trainer.test_loader):
        if i >= N_EVAL_BATCHES:
            break
        ep = run_episode(trainer, batch)
        X_l.append(ep["x_ot"])
        Xcf_l.append(ep["x_cf"])
        Yh_l.append(ep["y_hat"])
        Yc_l.append(ep["y_cf"])

        if len(examples) < 5:
            a, b, t = compute_bounds_np(ep["y_hat"][0:1])
            examples.append({
                "x_ot":   ep["x_ot"][0],
                "x_cf":   ep["x_cf"][0],
                "y_hat":  ep["y_hat"][0],
                "y_cf":   ep["y_cf"][0],
                "alpha":  a[0],
                "beta":   b[0],
                "target": t[0],
            })

    X    = np.concatenate(X_l,    axis=0)
    Xcf  = np.concatenate(Xcf_l,  axis=0)
    Yhat = np.concatenate(Yh_l,   axis=0)
    Ycf  = np.concatenate(Yc_l,   axis=0)
    print(f"[RL] {len(X)} samples ✓")
    return {"X": X, "X_cf": Xcf, "Y_hat": Yhat, "Y_cf": Ycf}, examples


# ═════════════════════════════════════════════════════════════════════════════
#  Plots
# ═════════════════════════════════════════════════════════════════════════════

def plot_examples(examples, out_path):
    n   = len(examples)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4.5 * n))
    if n == 1:
        axes = [axes]

    for i, ex in enumerate(examples):
        xo  = _2d(ex["x_ot"][np.newaxis])[0]
        xcf = _2d(ex["x_cf"][np.newaxis])[0]
        yh  = _2d(ex["y_hat"][np.newaxis])[0]
        yc  = _2d(ex["y_cf"][np.newaxis])[0]
        a   = ex["alpha"] if ex["alpha"].ndim == 1 else ex["alpha"][:, 0]
        b   = ex["beta"]  if ex["beta"].ndim  == 1 else ex["beta"][:, 0]
        tg  = ex["target"] if ex["target"].ndim == 1 else ex["target"][:, 0]

        T_in = len(xo)
        t_in = np.arange(T_in)
        t_fc = np.arange(T_in, T_in + len(yh))
        red  = (yh.mean() - yc.mean()) / (abs(yh.mean()) + 1e-8) * 100

        # validity ratio pour cet exemple
        vr = float(((yc >= a) & (yc <= b)).mean())

        ax = axes[i]
        ax.plot(t_in, xo,  color="steelblue", lw=1.5, label="x (original)")
        ax.plot(t_in, xcf, color="coral",     lw=1.5, ls="--", label="x_cf")
        ax.plot(t_fc, yh,  color="steelblue", lw=2,   label="forecast(x)")
        ax.plot(t_fc, yc,  color="coral",     lw=2,   ls="--", label="forecast(x_cf)")
        ax.plot(t_fc, tg,  color="green",     lw=1.5, ls=":",  label="target")
        ax.fill_between(t_fc, a, b, alpha=0.18, color="green", label="α/β bounds")
        ax.axvline(T_in, color="gray", ls="--", lw=1)
        ax.set_title(
            f"Sample {i+1}  |  VR = {vr:.2f}  |  Δforecast = {red:+.1f}%",
            fontsize=10,
        )
        ax.legend(fontsize=8, ncol=3)
        ax.grid(alpha=0.3)

    plt.suptitle(
        "RL-CF — ETTh1 / iTransformer\n"
        r"Bounds: target = $\hat{y}(1-\rho)$,  $\alpha/\beta$ = target ± 0.1|$\hat{y}$|",
        fontsize=12,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Examples → {out_path}")


def plot_bar(metrics, out_path):
    keys = [
        ("validity_ratio",       "Validity\nRatio ↑"),
        ("stepwise_auc",         "Step\nAUC ↑"),
        ("compactness",          "Compactness ↑"),
        ("temporal_consistency", "Temp.\nConsist. ↑"),
        ("plausibility_ensemble","Plausib.\n(1-score) ↑"),
    ]
    vals  = []
    errs  = []
    labels = []
    for key, lbl in keys:
        v = metrics.get(key, {}).get("mean", 0.0)
        s = metrics.get(key, {}).get("std",  0.0)
        if "plausibility" in key:
            v = 1.0 - v   # inverser : plus haut = meilleur
        vals.append(v)
        errs.append(s)
        labels.append(lbl)

    colors = ["steelblue", "steelblue", "coral", "coral", "green"]
    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar(labels, vals, yerr=errs, color=colors,
                  alpha=0.85, edgecolor="white", width=0.5, capsize=4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("RL-CF — Summary Metrics (ETTh1 / iTransformer)", fontsize=12)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Bar chart → {out_path}")


def plot_radar(metrics, out_path):
    radar_keys = [
        ("validity_ratio",       True,  "Validity Ratio"),
        ("stepwise_auc",         True,  "Step AUC"),
        ("compactness",          True,  "Compactness"),
        ("temporal_consistency", True,  "Temp. Consist."),
        ("plausibility_ensemble",False, "Plausib. ↓"),
        ("proximity_l1",         False, "Proximity L1 ↓"),
    ]
    labels = [k[2] for k in radar_keys]
    vals   = []
    for key, hb, _ in radar_keys:
        v = np.clip(metrics.get(key, {}).get("mean", 0.0), 0, 1)
        vals.append(v if hb else 1 - v)

    N      = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    vp     = vals + [vals[0]]
    ap     = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(ap, vp, color="steelblue", lw=2)
    ax.fill(ap, vp, color="steelblue", alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title("RL-CF — Key Metrics\n(normalised, higher = better)", size=12, pad=20)
    ax.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Radar → {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    from src.utils.config import load_config
    from src.utils.train_tools import get_device
    from src.training.RL_trainers.RL_trainer_alpha_beta import RLMaskTrainer
    from src.evaluation.unified_evaluator import CounterfactualEvaluator

    cfg_f  = load_config(CONFIG_FORECASTER)
    cfg_ae = load_config(CONFIG_AE)
    cfg_rl = load_config(CONFIG_RL)
    device = get_device(cfg_f)

    # ── 1. Trainer ────────────────────────────────────────────────────────────
    trainer = RLMaskTrainer(cfg_f, cfg_ae, cfg_rl, device)
    ckpt = torch.load(CHECKPOINT_RL, map_location=device, weights_only=False)
    trainer.agent.actor.load_state_dict(ckpt["actor_state_dict"])
    trainer.agent.critic.load_state_dict(ckpt["critic_state_dict"])
    trainer.agent.eval()
    print("[RL] Checkpoint loaded ✓")

    # ── 2. X_train pour détecteurs ────────────────────────────────────────────
    if hasattr(trainer, "x_train_eval") and trainer.x_train_eval is not None:
        x_tr = (trainer.x_train_eval.cpu().numpy()
                if torch.is_tensor(trainer.x_train_eval)
                else trainer.x_train_eval)
    else:
        bufs = []
        for i, b in enumerate(trainer.train_loader):
            if i >= 300:
                break
            bufs.append(b[0].float()[:, :, -1:].numpy())
        x_tr = np.concatenate(bufs, axis=0)

    # ── 3. Évaluateur unifié ──────────────────────────────────────────────────
    ev = CounterfactualEvaluator(
        x_train=x_tr,
        tol_compact=TOL_COMPACT,
        auc_mode="proportion",
        contamination=0.1,
        fit_plausibility=True,
    )

    # ── 4. Collecter les données ──────────────────────────────────────────────
    data, examples = collect_rl_data(trainer)

    # ── 5. Calculer les bornes ────────────────────────────────────────────────
    alphas, betas, _ = compute_bounds_np(data["Y_hat"])

    # ── 6. Évaluer ───────────────────────────────────────────────────────────
    print("\n[Eval] Running unified evaluator …")
    metrics = ev.evaluate(
        X_orig = data["X"],
        X_cf   = data["X_cf"],
        Y_hat  = data["Y_hat"],
        Y_cf   = data["Y_cf"],
        alphas = alphas,
        betas  = betas,
    )
    ev.print_table(metrics, label="RL-CF (ours) — ETTh1 / iTransformer")

    # ── 7. Sauvegarder ───────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # JSON métriques
    out_json = os.path.join(OUTPUT_DIR, "metrics_rl.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[✓] JSON → {out_json}")

    # Arrays numpy (réutilisables pour la comparaison)
    out_npz = os.path.join(OUTPUT_DIR, "rl_arrays.npz")
    np.savez(out_npz, **data)
    print(f"[✓] Arrays → {out_npz}")

    # ── 8. Figures ────────────────────────────────────────────────────────────
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_examples(examples, os.path.join(FIGURES_DIR, "cf_examples.png"))
    plot_bar(metrics,       os.path.join(FIGURES_DIR, "summary_bar.png"))
    plot_radar(metrics,     os.path.join(FIGURES_DIR, "radar.png"))

    print("\n[✓] RL evaluation complete.\n")


if __name__ == "__main__":
    main()
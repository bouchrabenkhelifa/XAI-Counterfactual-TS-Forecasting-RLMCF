import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.anomaly_detector.plausibility import (
    IForestPlausibility,
    LOFPlausibility,
    OCSVMPlausibility,
    EnsemblePlausibility,
)

SANITY_THRESHOLD = 0.4  


def load_config(config_path):
    with open(config_path, "r") as f:
        d = json.load(f)
    return SimpleNamespace(**d)


def load_series(cfg):
    df = pd.read_csv(cfg.data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    ot = df[[cfg.target_col]].values.astype(np.float32)
    T = len(ot)
    t_train = int(cfg.train_ratio * T)
    t_val = int((cfg.train_ratio + cfg.val_ratio) * T)
    ot_train = ot[:t_train]
    ot_val = ot[t_train:t_val]
    ot_test = ot[t_val:]
    scaler = StandardScaler()
    ot_train_s = scaler.fit_transform(ot_train).astype(np.float32)
    ot_val_s = scaler.transform(ot_val).astype(np.float32)
    ot_test_s = scaler.transform(ot_test).astype(np.float32)
    print(
        f"[Data] {cfg.dataset_name} | train={len(ot_train_s)} val={len(ot_val_s)} test={len(ot_test_s)}"
    )
    return ot_train_s, ot_val_s, ot_test_s, scaler


def make_windows_tensor(series, window, n_samples=256, device="cpu"):
    T = len(series)
    idx = np.random.choice(T - window, size=min(n_samples, T - window), replace=False)
    W = np.stack([series[i : i + window] for i in idx], axis=0)
    return torch.from_numpy(W).to(device)


def plot_sanity(scores_train, scores_val, dataset_name, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)
    detectors = ["if", "lof", "ocsvm", "ensemble"]
    labels = ["IForest", "LOF", "OC-SVM", "Ensemble"]
    x = np.arange(len(detectors))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        x - width / 2,
        [scores_train[k] for k in detectors],
        width,
        label="Train",
        color="steelblue",
    )
    ax.bar(
        x + width / 2,
        [scores_val[k] for k in detectors],
        width,
        label="Val",
        color="coral",
    )
    ax.axhline(
        SANITY_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Threshold ({SANITY_THRESHOLD})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Plausibility score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Plausibility sanity check — {dataset_name}")
    ax.legend()
    plt.tight_layout()
    save_path = os.path.join(figures_dir, f"plausibility_sanity_{dataset_name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved → {save_path}")


def separation_test(ensemble, x_real, label="data"):

    x_noise = torch.randn_like(x_real) * 3.0
    score_real = float(ensemble.score(x_real).mean().item())
    score_noise = float(ensemble.score(x_noise).mean().item())
    gap = score_real - score_noise
    status = "✓ good separation" if gap > 0.15 else "✗ poor separation"
    print(
        f"  [{label}] score(real)={score_real:.4f}  score(noise)={score_noise:.4f}  gap={gap:.4f}  {status}"
    )
    return score_real, score_noise, gap


def main(config_path):
    cfg = load_config(config_path)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    ot_train_s, ot_val_s, ot_test_s, scaler = load_series(cfg)

    iforest_cfg = cfg.iforest if hasattr(cfg, "iforest") else {}
    lof_cfg = cfg.lof if hasattr(cfg, "lof") else {}
    ocsvm_cfg = cfg.ocsvm if hasattr(cfg, "ocsvm") else {}

    ensemble = EnsemblePlausibility(window=cfg.window)
    ensemble.iforest = IForestPlausibility(window=cfg.window, **iforest_cfg)
    ensemble.lof = LOFPlausibility(window=cfg.window, **lof_cfg)
    ensemble.ocsvm = OCSVMPlausibility(window=cfg.window, **ocsvm_cfg)

    ot_trainval_s = np.concatenate([ot_train_s, ot_val_s], axis=0)
    ensemble.fit(ot_trainval_s)

    device = "cpu"
    x_train_t = make_windows_tensor(
        ot_train_s, cfg.window, n_samples=512, device=device
    )
    x_val_t = make_windows_tensor(ot_val_s, cfg.window, n_samples=256, device=device)

    print(f"\n── Sanity check on TRAIN (should all be > {SANITY_THRESHOLD}) ──")
    scores_train = ensemble.sanity_check(x_train_t, label="train")

    print(f"\n── Sanity check on VAL (should all be > {SANITY_THRESHOLD}) ──")
    scores_val = ensemble.sanity_check(x_val_t, label="val")

    print("\n── Separation test (real data vs random noise) ──")
    print("  → Gap > 0.15 = detector is useful for RL reward")
    s_real_tr, s_noise_tr, gap_tr = separation_test(ensemble, x_train_t, "train")
    s_real_vl, s_noise_vl, gap_vl = separation_test(ensemble, x_val_t, "val")

    sep_scores = {
        "gap_train": gap_tr,
        "gap_val": gap_vl,
        "score_real_train": s_real_tr,
        "score_noise_train": s_noise_tr,
        "score_real_val": s_real_vl,
        "score_noise_val": s_noise_vl,
    }

    plot_sanity(scores_train, scores_val, cfg.dataset_name, cfg.figures_dir)

    checkpoint = {
        "ensemble": ensemble,
        "scaler": scaler,
        "config": vars(cfg),
        "scores_train": scores_train,
        "scores_val": scores_val,
        "sep_scores": sep_scores,
    }

    ckpt_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)
    with open(ckpt_path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"\nCheckpoint saved → {ckpt_path}")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Dataset    : {cfg.dataset_name}")
    print(f"Train size : {len(ot_train_s)}")
    print(f"Window     : {cfg.window}")
    print(f"\nCheckpoint : {ckpt_path}")


if __name__ == "__main__":
    main("assets/configs/etth1_dataset/anomaly_detector/plausibility.json")

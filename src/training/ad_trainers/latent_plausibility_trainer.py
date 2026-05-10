import os
import pickle
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

from src.utils.config      import load_config
from src.utils.train_tools import get_device
from src.models.autoencoder.tcn_ae     import TCNAutoEncoder
from src.models.RL_without_ae.latent_plausibility import LatentPlausibility


CONFIG_AE = "assets/configs/etth1_dataset/ae/tcn_ae.json"

if __name__ == "__main__":
    cfg_ae = load_config(CONFIG_AE)
    device = get_device(cfg_ae)

    # ── Load AE ───────────────────────────────────────────────
    ae = TCNAutoEncoder.from_checkpoint(cfg_ae.checkpoint_path, device=device)
    ae.eval()
    for p in ae.parameters(): p.requires_grad_(False)

    # ── Load scaler ───────────────────────────────────────────
    ckpt_ae       = torch.load(cfg_ae.checkpoint_path,
                                map_location="cpu", weights_only=False)
    scaler        = StandardScaler()
    scaler.mean_  = np.array(ckpt_ae["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.array(ckpt_ae["scaler_std"],  dtype=np.float64)
    scaler.var_   = scaler.scale_ ** 2
    scaler.n_features_in_ = 1

    # ── Load ETTh1 train + val ────────────────────────────────
    df = pd.read_csv("assets/datasets/ETTh1.csv")
    ot = df[["OT"]].values.astype(np.float32)
    T  = len(ot)
    t_train = int(0.70 * T)
    t_val   = int(0.80 * T)

    ot_train = scaler.transform(ot[:t_train]).astype(np.float32)
    ot_val   = scaler.transform(ot[t_train:t_val]).astype(np.float32)
    ot_tv    = np.concatenate([ot_train, ot_val])

    # ── Build windows ─────────────────────────────────────────
    window = cfg_ae.seq_len
    idxs   = np.arange(0, len(ot_tv) - window, 1)
    W      = np.stack([ot_tv[i:i+window] for i in idxs])
    W_t    = torch.from_numpy(W).float().to(device)

    print(f"Windows pour calibration : {len(W_t)}")

    # ── Fit LatentPlausibility ────────────────────────────────
    plaus = LatentPlausibility(ae=ae, alpha=0.5, scale=10.0)
    plaus.fit(W_t)

    # ── Sanity check ──────────────────────────────────────────
    ot_test   = scaler.transform(ot[t_val:]).astype(np.float32)
    idxs_test = np.arange(0, len(ot_test) - window, 10)
    W_test    = np.stack([ot_test[i:i+window] for i in idxs_test])
    W_test_t  = torch.from_numpy(W_test).float().to(device)

    print("\n── Sanity check ─────────────────────────────────────")
    plaus.sanity_check(W_t[:200],  label="train")
    plaus.sanity_check(W_test_t,   label="test")

    # Tester avec du bruit
    x_noise = torch.randn_like(W_test_t[:50])
    plaus.sanity_check(x_noise, label="bruit")

    # ── Save ──────────────────────────────────────────────────
    os.makedirs("assets/checkpoints/latent_plausibility", exist_ok=True)
    save_path = "assets/checkpoints/latent_plausibility/latent_plaus_etth1.pt"

    torch.save({
        "z_mean" : plaus.z_mean.cpu(),
        "z_std"  : plaus.z_std.cpu(),
        "sigma"  : plaus.sigma,
        "alpha"  : plaus.alpha,
        "scale"  : plaus.scale,
    }, save_path)

    print(f"\n✅ Saved → {save_path}")
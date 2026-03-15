import torch
import torch.nn as nn
import numpy as np


class LatentPlausibility(nn.Module):
    """
    Plausibilité différentiable basée sur l'espace latent de l'AE.

    Score = combinaison de :
      1. Distance latente normalisée : exp(-||z - z_mean|| / sigma)
      2. Reconstruction error       : exp(-MSE(x, decode(encode(x))) * scale)

    Avantages vs IForest/LOF/OC-SVM :
      → différentiable → gradient utile pour l'actor
      → cohérent avec le pipeline latent space
      → pas sensible au lissage du decoder
      → pas de features manuelles
    """

    def __init__(self, ae, alpha=0.5, scale=10.0):
        super().__init__()
        self.ae    = ae
        self.alpha = alpha   # poids distance latente
        self.scale = scale   # scaling reconstruction error

        self.z_mean  = None
        self.z_std   = None
        self.sigma   = None
        self.fitted_ = False

    def fit(self, x_train: torch.Tensor):
        """
        Calibrer sur les données d'entraînement.

        Args:
            x_train : (N, seq_len, 1) tenseur scalé
        """
        self.ae.eval()
        with torch.no_grad():
            z_train = self.ae.encode(x_train)  # (N, 64)

        self.z_mean  = z_train.mean(dim=0)     # (64,)
        self.z_std   = z_train.std(dim=0) + 1e-8
        self.sigma   = float(z_train.std())
        self.fitted_ = True

        print(f"[LatentPlaus] fitted on {len(x_train)} windows")
        print(f"  z_mean norm = {self.z_mean.norm():.4f}")
        print(f"  sigma       = {self.sigma:.4f}")

    def score(self, x_cf: torch.Tensor) -> torch.Tensor:
        """
        Calculer le score de plausibilité.

        Args:
            x_cf : (B, seq_len, 1)

        Returns:
            score : (B,) ∈ [0, 1]
        """
        assert self.fitted_, "Call fit() first"

        z_cf  = self.ae.encode(x_cf)
        x_rec = self.ae.decode(z_cf)

        # 1. Distance latente normalisée
        z_mean = self.z_mean.to(x_cf.device)
        z_std  = self.z_std.to(x_cf.device)
        z_norm = ((z_cf - z_mean) / z_std).norm(dim=1)  # (B,)
        p_latent = torch.exp(-z_norm / (self.sigma * 10 + 1e-8))

        # 2. Reconstruction error
        mse_rec = ((x_cf - x_rec) ** 2).mean(dim=(1, 2))  # (B,)
        p_recon = torch.exp(-mse_rec * self.scale)

        return self.alpha * p_latent + (1 - self.alpha) * p_recon

    def sanity_check(self, x_real: torch.Tensor, label="real"):
        s = self.score(x_real)
        m = float(s.mean().item())
        print(f"[LatentPlaus] sanity({label}) mean={m:.4f} "
              f"{'✓' if m > 0.4 else '✗'}")
        return m
import torch
import torch.nn as nn


class LatentPlausibility(nn.Module):
    """
    Plausibility hybride :
    - score latent (distance au manifold latent)
    - score reconstruction (x_cf doit être bien reconstruit par l'AE)
    """

    def __init__(
        self,
        ae,
        alpha: float = 1.0,
        scale: float = 8.0,
        latent_weight: float = 0.65,
        recon_weight: float = 0.35,
    ):
        super().__init__()
        self.ae = ae
        self.alpha = alpha
        self.scale = scale
        self.latent_weight = latent_weight
        self.recon_weight = recon_weight

        self.z_mean = None
        self.z_std = None
        self.fitted_ = False
        self.sigma = None  # compatibilité avec ton loader

    def fit(self, x_train: torch.Tensor):
        self.ae.eval()
        with torch.no_grad():
            z_train = self.ae.encode(x_train)

        self.z_mean = z_train.mean(dim=0)
        self.z_std = z_train.std(dim=0) + 1e-6
        self.fitted_ = True

        print(f"[LatentPlaus] fitted on {len(x_train)} windows")
        print(f"  z_mean norm = {self.z_mean.norm():.4f}")
        print(f"  z_std mean  = {self.z_std.mean():.4f}")

    def _latent_score_from_z(self, z: torch.Tensor) -> torch.Tensor:
        z_mean = self.z_mean.to(z.device)
        z_std = self.z_std.to(z.device)

        d2 = ((z - z_mean) / z_std).pow(2).mean(dim=1)

        # mapping plus doux que exp(-0.5*d2), pour éviter l'effondrement à ~0.03
        score = 1.0 / (1.0 + self.alpha * d2 / self.scale)
        return torch.clamp(score, 0.0, 1.0)

    def _recon_score_from_x(self, x: torch.Tensor) -> torch.Tensor:
        z = self.ae.encode(x)
        x_rec = self.ae.decode(z)

        recon_err = ((x - x_rec) ** 2).mean(dim=(1, 2))
        score = torch.exp(-self.scale * recon_err)
        return torch.clamp(score, 0.0, 1.0)

    def score_components(
        self,
        x_cf: torch.Tensor = None,
        z_cf: torch.Tensor = None,
    ):
        assert self.fitted_, "Call fit() first"

        if x_cf is None and z_cf is None:
            raise ValueError("Either x_cf or z_cf must be provided")

        latent_score = None
        recon_score = None

        # si x_cf existe, on préfère mesurer sur le x_cf réel
        if x_cf is not None:
            z_from_x = self.ae.encode(x_cf)
            latent_score = self._latent_score_from_z(z_from_x)
            recon_score = self._recon_score_from_x(x_cf)

            total = (
                self.latent_weight * latent_score
                + self.recon_weight * recon_score
            )
            return {
                "total": torch.clamp(total, 0.0, 1.0),
                "latent": latent_score,
                "recon": recon_score,
            }

        # fallback si on n'a que z_cf
        latent_score = self._latent_score_from_z(z_cf)
        return {
            "total": torch.clamp(latent_score, 0.0, 1.0),
            "latent": latent_score,
            "recon": latent_score,   # compatibilité
        }

    def score(self, x_cf: torch.Tensor = None, z_cf: torch.Tensor = None) -> torch.Tensor:
        return self.score_components(x_cf=x_cf, z_cf=z_cf)["total"]

    def sanity_check(self, x_real: torch.Tensor, label: str = "real"):
        s = self.score(x_cf=x_real)
        m = float(s.mean().item())
        print(f"[LatentPlaus] sanity({label}) mean={m:.4f} {'✓' if m > 0.4 else '✗'}")
        return m
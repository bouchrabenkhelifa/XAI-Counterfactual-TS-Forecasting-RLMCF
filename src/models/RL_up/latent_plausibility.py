import torch
import torch.nn as nn


class LatentPlausibility(nn.Module):
    def __init__(self, ae, alpha: float = 1.0, scale: float = 10.0):
        super().__init__()
        self.ae = ae
        self.alpha = alpha
        self.scale = scale

        self.z_mean = None
        self.z_std = None
        self.fitted_ = False

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

    def score(self, x_cf: torch.Tensor = None, z_cf: torch.Tensor = None) -> torch.Tensor:
        assert self.fitted_, "Call fit() first"

        if z_cf is None:
            assert x_cf is not None, "Either x_cf or z_cf must be provided"
            z_cf = self.ae.encode(x_cf)

        z_mean = self.z_mean.to(z_cf.device)
        z_std = self.z_std.to(z_cf.device)

        # squared normalized distance per dimension
        d2 = ((z_cf - z_mean) / z_std).pow(2).mean(dim=1)

        # smoother and well-calibrated plausibility
        p_latent = torch.exp(-0.5 * d2)

        return p_latent

    def sanity_check(self, x_real: torch.Tensor, label: str = "real"):
        s = self.score(x_cf=x_real)
        m = float(s.mean().item())
        print(f"[LatentPlaus] sanity({label}) mean={m:.4f} {'✓' if m > 0.4 else '✗'}")
        return m
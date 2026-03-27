import torch


class LatentCounterfactualOptimizer:
    def __init__(
        self,
        ae,
        forecaster,
        objective,
        steps: int = 200,
        lr: float = 5e-2,
        clip_latent: bool = True,
        latent_min: float = -1.0,
        latent_max: float = 1.0,
        patience: int = 25,
        tol: float = 1e-5,
        device=None,
    ):
        self.ae = ae
        self.forecaster = forecaster
        self.objective = objective
        self.steps = steps
        self.lr = lr
        self.clip_latent = clip_latent
        self.latent_min = latent_min
        self.latent_max = latent_max
        self.patience = patience
        self.tol = tol
        self.device = device

    def optimize(
        self,
        x_ot: torch.Tensor,
        x_full: torch.Tensor,
        x_mark: torch.Tensor,
        y_hat: torch.Tensor = None,
    ):
        with torch.no_grad():
            z = self.ae.encode(x_ot)
            if y_hat is None:
                y_hat = self.forecaster.predict_ot(x_full, x_mark)

        z_cf = z.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([z_cf], lr=self.lr)

        best = {
            "loss": float("inf"),
            "step": -1,
            "z_cf": None,
            "x_cf": None,
            "y_cf": None,
            "loss_dict": None,
        }
        wait = 0
        history = []

        for step in range(1, self.steps + 1):
            optimizer.zero_grad()

            if self.clip_latent:
                z_work = torch.clamp(z_cf, self.latent_min, self.latent_max)
            else:
                z_work = z_cf

            x_cf = self.ae.decode(z_work)
            y_cf = self.forecaster.predict_from_ot(
                x_ot=x_cf,
                x_full=x_full,
                x_mark=x_mark,
            )

            loss_dict = self.objective(
                x=x_ot,
                x_cf=x_cf,
                z=z,
                z_cf=z_work,
                y_hat=y_hat,
                y_cf=y_cf,
            )

            loss = loss_dict["total"].mean()
            loss.backward()
            optimizer.step()

            current = float(loss.item())
            history.append(current)

            if current + self.tol < best["loss"]:
                best["loss"] = current
                best["step"] = step
                best["z_cf"] = z_work.detach().clone()
                best["x_cf"] = x_cf.detach().clone()
                best["y_cf"] = y_cf.detach().clone()
                best["loss_dict"] = {
                    k: (v.detach().clone() if torch.is_tensor(v) else v)
                    for k, v in loss_dict.items()
                }
                wait = 0
            else:
                wait += 1

            if wait >= self.patience:
                break

        return {
            "z": z.detach(),
            "z_cf": best["z_cf"],
            "x_cf": best["x_cf"],
            "y_hat": y_hat.detach(),
            "y_cf": best["y_cf"],
            "loss_dict": best["loss_dict"],
            "best_step": best["step"],
            "history": history,
        }
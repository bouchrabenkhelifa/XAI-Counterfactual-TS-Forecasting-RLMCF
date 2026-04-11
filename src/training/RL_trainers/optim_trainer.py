import os
import torch

from src.models.autoencoder.tcn_ae import TCNAutoEncoder
from src.models.forecaster_wrapper import ForecasterWrapper
from src.models.optimization_strategy.cf_objective import CounterfactualObjective
from src.models.optimization_strategy.latent_optimizer import (
    LatentCounterfactualOptimizer,
)


def get_ae_checkpoint_path(cfg_ae):
    if hasattr(cfg_ae, "checkpoint_path"):
        return cfg_ae.checkpoint_path

    if hasattr(cfg_ae, "checkpoint_dir") and hasattr(cfg_ae, "checkpoint_name"):
        return os.path.join(cfg_ae.checkpoint_dir, cfg_ae.checkpoint_name)

    raise AttributeError(
        "AE config must contain either 'checkpoint_path' or "
        "('checkpoint_dir' and 'checkpoint_name')."
    )


def prepare_optim_data(cfg_forecaster):
    from src.data_provider.data_factory import data_provider

    _, train_loader = data_provider(cfg_forecaster, "train")
    _, test_loader = data_provider(cfg_forecaster, "test")
    return train_loader, test_loader


def estimate_latent_stats(ae, train_loader, device, max_batches=100):
    z_list = []

    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= max_batches:
                break

            batch_x, _, _, _ = batch
            batch_x = batch_x.float().to(device)
            x_ot = batch_x[:, :, -1:]

            z = ae.encode(x_ot)
            z_list.append(z.detach())

    z_all = torch.cat(z_list, dim=0)

    reduce_dims = tuple(range(0, z_all.dim() - 1)) if z_all.dim() > 2 else (0,)
    z_mean = z_all.mean(dim=reduce_dims)
    z_std = z_all.std(dim=reduce_dims)

    z_std = torch.clamp(z_std, min=1e-6)
    return z_mean, z_std


class OptimizationTrainer:
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

        ae_ckpt = get_ae_checkpoint_path(cfg_ae)

        self.ae = TCNAutoEncoder.from_checkpoint(ae_ckpt, device=device)
        self.ae.eval()
        for p in self.ae.parameters():
            p.requires_grad_(False)

        self.forecaster = ForecasterWrapper(cfg_forecaster, device)
        self.forecaster.model.eval()
        for p in self.forecaster.model.parameters():
            p.requires_grad_(False)

        self.train_loader, self.test_loader = prepare_optim_data(cfg_forecaster)

        max_stat_batches = getattr(cfg_optim, "latent_stat_batches", 100)
        self.z_mean, self.z_std = estimate_latent_stats(
            self.ae,
            self.train_loader,
            self.device,
            max_batches=max_stat_batches,
        )

        self.objective = CounterfactualObjective(
            rho=cfg_optim.rho,
            alpha=cfg_optim.alpha,
            beta_x=cfg_optim.beta_x,
            beta_z=cfg_optim.beta_z,
            gamma=cfg_optim.gamma,
            z_mean=self.z_mean,
            z_std=self.z_std,
        )

        self.cf_optimizer = LatentCounterfactualOptimizer(
            ae=self.ae,
            forecaster=self.forecaster,
            objective=self.objective,
            steps=cfg_optim.steps,
            lr=cfg_optim.lr,
            patience=cfg_optim.patience,
            tol=cfg_optim.tol,
            device=device,
        )

    def evaluate(self, n_batches=5):
        for i, batch in enumerate(self.test_loader):
            if i >= n_batches:
                break

            batch_x, _, batch_x_mark, _ = batch
            batch_x = batch_x.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)

            x_ot = batch_x[:, :, -1:]

            with torch.no_grad():
                y_hat = self.forecaster.predict_ot(batch_x, batch_x_mark)

            result = self.cf_optimizer.optimize(
                x_ot=x_ot,
                x_full=batch_x,
                x_mark=batch_x_mark,
                y_hat=y_hat,
            )

            print(f"\nBatch {i}")
            print("delta_mean:", result["loss_dict"]["delta_mean"].mean().item())
            print("success:", result["loss_dict"]["success"].mean().item())
            print("goal:", result["loss_dict"]["goal"].mean().item())
            print("prox_x:", result["loss_dict"]["prox_x"].mean().item())
            print("prox_z:", result["loss_dict"]["prox_z"].mean().item())
            print("plaus:", result["loss_dict"]["plaus"].mean().item())
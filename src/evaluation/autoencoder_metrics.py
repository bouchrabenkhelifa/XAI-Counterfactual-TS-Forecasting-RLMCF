import numpy as np
import torch


def reconstruction_mae(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    return torch.mean(torch.abs(x_hat - x)).item()


def reconstruction_mse(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    return torch.mean((x_hat - x) ** 2).item()


def reconstruction_rmse(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    mse = reconstruction_mse(x_hat, x)
    return float(np.sqrt(mse))
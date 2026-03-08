import json
import os

import numpy as np
import torch


def get_device(configs):
    use_cuda = getattr(configs, "use_gpu", False) and torch.cuda.is_available()
    if use_cuda:
        return torch.device(f"cuda:{configs.gpu}")
    return torch.device("cpu")


def adjust_learning_rate(optimizer, epoch, configs):
    if configs.lradj == "type1":
        lr = configs.learning_rate * (0.5 ** ((epoch - 1) // 1))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print(f"Updating learning rate to {lr}")

    elif configs.lradj == "type2":
        lr_adjust = {
            2: 5e-5,
            4: 1e-5,
            6: 5e-6,
            8: 1e-6,
            10: 5e-7,
            15: 1e-7,
            20: 5e-8,
        }
        if epoch in lr_adjust:
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            print(f"Updating learning rate to {lr}")


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
            )
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def save_history(history: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
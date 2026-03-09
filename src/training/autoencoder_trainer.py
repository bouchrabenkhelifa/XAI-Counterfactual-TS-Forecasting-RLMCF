import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from src.data_provider.data_factory import data_provider
from src.models.autoencoder.ae_wrapper import build_autoencoder
from src.models.autoencoder.conv_ae_wrapper import build_conv_autoencoder
from src.utils.train_tools import get_device, EarlyStopping, save_history


class AutoEncoderTrainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = get_device(configs)

        if configs.model_name == "Conv1DAutoEncoder":
            self.model, _ = build_conv_autoencoder(configs, self.device)
        else:
            self.model, _ = build_autoencoder(configs, self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.configs.learning_rate)

        self.train_data, self.train_loader = data_provider(configs, "train")
        self.val_data, self.val_loader = data_provider(configs, "val")
        self.test_data, self.test_loader = data_provider(configs, "test")

        self.checkpoint_path = os.path.join(
            self.configs.checkpoint_dir,
            self.configs.checkpoint_name,
        )
        self.history_path = os.path.join(
            self.configs.results_dir,
            self.configs.history_name,
        )

    def _compute_batch_loss(self, batch_x):
        batch_x = batch_x.float().to(self.device)
        x_hat = self.model(batch_x)
        loss = self.criterion(x_hat, batch_x)
        return loss, x_hat, batch_x

    def validate(self, loader):
        losses = []
        self.model.eval()

        with torch.no_grad():
            for batch_x, _, _, _ in loader:
                loss, _, _ = self._compute_batch_loss(batch_x)
                losses.append(loss.item())

        self.model.train()
        return float(np.average(losses))

    def train(self):
        early_stopping = EarlyStopping(
            patience=self.configs.patience,
            verbose=True,
        )

        history = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
        }

        train_steps = len(self.train_loader)
        time_now = time.time()

        for epoch in range(self.configs.train_epochs):
            iter_count = 0
            train_losses = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, _, _, _) in enumerate(self.train_loader):
                iter_count += 1
                self.optimizer.zero_grad()

                loss, _, _ = self._compute_batch_loss(batch_x)
                train_losses.append(loss.item())

                loss.backward()
                self.optimizer.step()

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.configs.train_epochs - epoch) * train_steps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")

            train_loss = float(np.average(train_losses))
            val_loss = self.validate(self.val_loader)
            test_loss = self.validate(self.test_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["test_loss"].append(test_loss)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                f"Train Loss: {train_loss:.7f} "
                f"Vali Loss: {val_loss:.7f} "
                f"Test Loss: {test_loss:.7f}"
            )

            early_stopping(val_loss, self.model, self.checkpoint_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        save_history(history, self.history_path)

        return self.model, history
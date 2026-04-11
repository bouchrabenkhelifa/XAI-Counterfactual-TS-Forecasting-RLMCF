import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from src.data_provider.data_factory import data_provider
from src.models.forecaster_wrapper import build_itransformer
from src.utils.train_tools import (
    EarlyStopping,
    adjust_learning_rate,
    get_device,
    save_history,
)


class ITransformerTrainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = get_device(configs)

        self.model, _ = build_itransformer(configs, self.device)
        self.model = self.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.configs.learning_rate,
        )

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

    def _build_decoder_input(self, batch_y):
        dec_inp = torch.zeros_like(batch_y[:, -self.configs.pred_len:, :]).float()
        dec_inp = torch.cat(
            [batch_y[:, :self.configs.label_len, :], dec_inp],
            dim=1,
        )
        return dec_inp.to(self.device)

    def _forward_model(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        if self.configs.output_attention:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return outputs

    def _compute_batch_loss(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        dec_inp = self._build_decoder_input(batch_y)
        outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1 if self.configs.features == "MS" else 0
        outputs = outputs[:, -self.configs.pred_len:, f_dim:]
        target = batch_y[:, -self.configs.pred_len:, f_dim:]

        loss = self.criterion(outputs, target)
        return loss, outputs, target

    def validate(self, loader):
        losses = []
        self.model.eval()

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
                loss, _, _ = self._compute_batch_loss(
                    batch_x, batch_y, batch_x_mark, batch_y_mark
                )
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

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.train_loader):
                iter_count += 1
                self.optimizer.zero_grad()

                loss, _, _ = self._compute_batch_loss(
                    batch_x, batch_y, batch_x_mark, batch_y_mark
                )

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

            adjust_learning_rate(self.optimizer, epoch + 1, self.configs)

        self.model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device)
        )
        save_history(history, self.history_path)

        return self.model, history
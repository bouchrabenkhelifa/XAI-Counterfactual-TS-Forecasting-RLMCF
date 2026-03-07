import os
import json
import torch
import torch.nn as nn

from src.models.forecaster_wrapper import build_itransformer, load_config
from src.datasets.forecast_dataset import build_forecast_dataloaders


def get_device(device_name: str):
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device, label_len, pred_len):
    model.train()
    total_loss = 0.0
    n = 0

    for x_enc, y_full, x_mark_enc, y_mark_dec in loader:
        x_enc = x_enc.to(device).float()                # [B, seq_len, 1]
        y_full = y_full.to(device).float()              # [B, label_len+pred_len, 1]
        x_mark_enc = x_mark_enc.to(device).float()      # [B, seq_len, 4]
        y_mark_dec = y_mark_dec.to(device).float()      # [B, label_len+pred_len, 4]

        # decoder input
        x_dec = torch.zeros_like(y_full).to(device)
        x_dec[:, :label_len, :] = y_full[:, :label_len, :]

        # target = only prediction horizon
        y_target = y_full[:, -pred_len:, :]             # [B, pred_len, 1]

        y_pred = model(x_enc, x_mark_enc, x_dec, y_mark_dec)  # [B, pred_len, 1]
        loss = criterion(y_pred, y_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bsz = x_enc.size(0)
        total_loss += loss.item() * bsz
        n += bsz

    return total_loss / max(1, n)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, label_len, pred_len):
    model.eval()
    total_loss = 0.0
    n = 0

    for x_enc, y_full, x_mark_enc, y_mark_dec in loader:
        x_enc = x_enc.to(device).float()
        y_full = y_full.to(device).float()
        x_mark_enc = x_mark_enc.to(device).float()
        y_mark_dec = y_mark_dec.to(device).float()

        x_dec = torch.zeros_like(y_full).to(device)
        x_dec[:, :label_len, :] = y_full[:, :label_len, :]

        y_target = y_full[:, -pred_len:, :]
        y_pred = model(x_enc, x_mark_enc, x_dec, y_mark_dec)

        loss = criterion(y_pred, y_target)

        bsz = x_enc.size(0)
        total_loss += loss.item() * bsz
        n += bsz

    return total_loss / max(1, n)


def train_itransformer(config_path: str):
    configs = load_config(config_path)
    device = get_device(configs.device)

    os.makedirs(configs.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(configs.checkpoint_dir, configs.checkpoint_name)

    train_loader, val_loader, test_loader, scaler = build_forecast_dataloaders(configs)
    model, _ = build_itransformer(config_path, device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=configs.learning_rate,
        weight_decay=configs.weight_decay,
    )

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    print("Start training iTransformer...")
    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    for epoch in range(1, configs.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            label_len=configs.label_len,
            pred_len=configs.pred_len,
        )

        val_loss = validate_one_epoch(
            model, val_loader, criterion, device,
            label_len=configs.label_len,
            pred_len=configs.pred_len,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                    "config": vars(configs),
                },
                checkpoint_path,
            )
            print(f"✅ Saved best checkpoint: {checkpoint_path}")

    # save training history
    history_path = os.path.join(configs.checkpoint_dir, "itransformer_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("Training finished.")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoint saved at: {checkpoint_path}")
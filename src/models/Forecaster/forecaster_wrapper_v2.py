"""
forecaster_wrapper_v2.py
------------------------
Generic forecaster wrapper — supports iTransformer, GRU, DLinear, TimesNet.
Does NOT touch the original forecaster_wrapper.py.

Usage in config JSON:
    "model_type": "GRU"   # or "DLinear", "TimesNet", "iTransformer"
"""

import os
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(model_type: str, configs, device: torch.device):
    mt = model_type.lower()
    if mt == "itransformer":
        from src.models.Forecaster.iTransformer import Model
    elif mt == "gru":
        from src.models.Forecaster.GRU import Model
    elif mt == "dlinear":
        from src.models.Forecaster.DLinear import Model
    elif mt == "patchtst":
        from src.models.Forecaster.PatchTST import Model
    elif mt == "timesnet":
        from src.models.Forecaster.TimesNet import Model
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. "
                         f"Choose from: iTransformer, GRU, DLinear, TimesNet")

    model = Model(configs).float().to(device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Decoder input helper (used by iTransformer / seq2seq-style models)
# ─────────────────────────────────────────────────────────────────────────────

def _build_decoder_input(batch_y, label_len, pred_len, device):
    dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
    return dec_inp


# ─────────────────────────────────────────────────────────────────────────────
# Generic wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ForecasterWrapperV2:
    """
    Drop-in replacement for ForecasterWrapper that supports multiple architectures.

    The config JSON must include:
        "model_type"      : "GRU" | "DLinear" | "TimesNet" | "iTransformer"
        "checkpoint_dir"  : path to checkpoint folder
        "checkpoint_name" : checkpoint filename (.pth)
        "seq_len"         : look-back window
        "pred_len"        : forecast horizon
        "label_len"       : decoder label length (only used by iTransformer)
    """

    def __init__(self, cfg, device: torch.device):
        self.cfg    = cfg
        self.device = device

        model_type  = getattr(cfg, "model_type", "iTransformer")
        self.model  = _load_model(model_type, cfg, device)

        ckpt_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Support both raw state_dict and wrapped checkpoints
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()

        print(f"[Forecaster-v2] {model_type} loaded ✓  "
              f"seq={cfg.seq_len}  pred={cfg.pred_len}")

    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, x, x_mark, y_mark=None):
        """
        Full multivariate prediction.
        x      : (B, seq_len, n_features)
        x_mark : (B, seq_len, time_features)
        returns: (B, pred_len, n_features)
        """
        B = x.shape[0]
        model_type = getattr(self.cfg, "model_type", "iTransformer").lower()

        if model_type == "itransformer":
            # iTransformer needs decoder input
            batch_y_dummy = torch.zeros(
                B, self.cfg.label_len + self.cfg.pred_len, x.shape[-1]
            ).float().to(self.device)
            dec_inp = _build_decoder_input(
                batch_y_dummy, self.cfg.label_len, self.cfg.pred_len, self.device
            )
            if y_mark is None:
                y_mark = torch.zeros(
                    B, self.cfg.label_len + self.cfg.pred_len, x_mark.shape[-1]
                ).float().to(self.device)
            output = self.model(x, x_mark, dec_inp, y_mark)
        else:
            # GRU / DLinear / TimesNet — simple forward
            output = self.model(x, x_mark)

        if isinstance(output, tuple):
            output = output[0]

        return output[:, -self.cfg.pred_len:, :]   # (B, pred_len, n_features)

    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_ot(self, x, x_mark, y_mark=None):
        """Predict only the OT (last) feature."""
        return self.predict(x, x_mark, y_mark)[:, :, -1:]   # (B, pred_len, 1)

    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_from_ot(self, x_ot, x_full, x_mark):
        """
        Replace OT channel in x_full with x_ot, then predict OT.
        Used by the RL trainer to evaluate counterfactual inputs.
        """
        x_cf_full = x_full.clone()
        x_cf_full[:, :, -1:] = x_ot
        return self.predict_ot(x_cf_full, x_mark)

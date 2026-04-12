import torch

from src.models.Forecaster.iTransformer import Model


def build_itransformer(configs, device):
    model = Model(configs).float().to(device)
    return model, configs


# ─────────────────────────────────────────────────────────────
# AJOUTE CE BLOC À LA FIN DE src/models/forecaster_wrapper.py
# ─────────────────────────────────────────────────────────────

import os
import torch


def build_decoder_input(batch_y, label_len, pred_len, device):
    dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
    dec_inp = torch.cat(
        [batch_y[:, :label_len, :], dec_inp], dim=1
    ).float().to(device)
    return dec_inp


class ForecasterWrapper:
   

    def __init__(self, cfg, device: torch.device):
        self.cfg    = cfg
        self.device = device

        self.model, _ = build_itransformer(cfg, device)
        ckpt_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)
        self.model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=False)
        )
        self.model.eval()
        print(f"[Forecaster] Loaded ✓  seq={cfg.seq_len}  pred={cfg.pred_len}")

    @torch.no_grad()
    def predict(self, x, x_mark, y_mark=None):
        B = x.shape[0]

        batch_y_dummy = torch.zeros(
            B, self.cfg.label_len + self.cfg.pred_len, x.shape[-1]
        ).float().to(self.device)

        dec_inp = build_decoder_input(
            batch_y   = batch_y_dummy,
            label_len = self.cfg.label_len,
            pred_len  = self.cfg.pred_len,
            device    = self.device,
        )

        if y_mark is None:
            y_mark = torch.zeros(
                B, self.cfg.label_len + self.cfg.pred_len,
                x_mark.shape[-1]
            ).float().to(self.device)

        output = self.model(x, x_mark, dec_inp, y_mark)
        if isinstance(output, tuple):
            output = output[0]

        return output[:, -self.cfg.pred_len:, :]    # (B, 96, 7)

    @torch.no_grad()
    def predict_ot(self, x, x_mark, y_mark=None):
        return self.predict(x, x_mark, y_mark)[:, :, -1:]

    @torch.no_grad()
    def predict_from_ot(self, x_ot, x_full, x_mark):
    
        x_cf_full = x_full.clone()
        x_cf_full[:, :, -1:] = x_ot
        return self.predict_ot(x_cf_full, x_mark)
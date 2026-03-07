import json
import torch
from types import SimpleNamespace

from src.models.iTransformer import Model as ITransformerModel


def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    return SimpleNamespace(**cfg_dict)


def build_itransformer(config_path: str, device: torch.device):
    configs = load_config(config_path)
    model = ITransformerModel(configs).to(device)
    return model, configs


def load_itransformer_checkpoint(model, checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    return model
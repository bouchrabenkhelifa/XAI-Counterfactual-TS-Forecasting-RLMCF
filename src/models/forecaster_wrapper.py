import torch

from src.models.iTransformer import Model


def build_itransformer(configs, device):
    model = Model(configs).float().to(device)
    return model, configs
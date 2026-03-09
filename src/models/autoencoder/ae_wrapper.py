import os
import torch

from src.models.autoencoder.Encoder import MLPEncoder
from src.models.autoencoder.Decoder import MLPDecoder
from src.models.autoencoder.ae import AutoEncoder


def build_autoencoder(configs, device):
    encoder = MLPEncoder(
        input_dim=configs.input_dim,
        hidden_dim=configs.hidden_dim,
        latent_dim=configs.latent_dim,
    )

    decoder = MLPDecoder(
        latent_dim=configs.latent_dim,
        hidden_dim=configs.hidden_dim,
        output_dim=configs.input_dim,
        seq_len=configs.seq_len,
        enc_in=configs.enc_in,
    )

    model = AutoEncoder(encoder=encoder, decoder=decoder).to(device)
    return model, configs


def load_autoencoder_checkpoint(model, checkpoint_path: str, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model
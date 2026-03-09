import torch

from src.models.autoencoder.ae import AutoEncoder
from src.models.autoencoder.conv_encoder import Conv1DEncoder
from src.models.autoencoder.conv_decoder import Conv1DDecoder


def build_conv_autoencoder(configs, device):
    encoder = Conv1DEncoder(
        enc_in=configs.enc_in,
        seq_len=configs.seq_len,
        hidden_dim=configs.hidden_dim,
        latent_dim=configs.latent_dim,
    )

    decoder = Conv1DDecoder(
        enc_in=configs.enc_in,
        seq_len=configs.seq_len,
        hidden_dim=configs.hidden_dim,
        latent_dim=configs.latent_dim,
        reduced_len=encoder.reduced_len,
    )

    model = AutoEncoder(encoder=encoder, decoder=decoder).to(device)
    return model, configs


def load_conv_autoencoder_checkpoint(model, checkpoint_path: str, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model
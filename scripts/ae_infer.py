import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.autoencoder_inference import run_reconstruction_and_plot


if __name__ == "__main__":

    CONFIG_PATH = "configs/models/ae/conv_ae_etth1_96.json"

    run_reconstruction_and_plot(
        config_path=CONFIG_PATH,
        batch_index=0,
        feature_idx=-1,
        use_inverse=True,
        save_name="conv_ae_etth1_96.png",
    )
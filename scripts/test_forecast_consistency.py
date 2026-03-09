import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.forecast_consistency import run_forecast_consistency


if __name__ == "__main__":
    FORECASTER_CONFIG = "configs/models/itransformer/etth1_96_96.json"
    AE_CONFIG = "configs/models/ae/conv_ae_etth1_96.json"

    run_forecast_consistency(
        forecaster_config_path=FORECASTER_CONFIG,
        ae_config_path=AE_CONFIG,
        batch_index=0,
        feature_idx=-1,
        use_inverse=True,
        save_name="forecast_consistency_etth1_96_96.png",
    )
from src.training.train_itransformer import train_itransformer

if __name__ == "__main__":
    config_path = "configs/itransformer.json"
    train_itransformer(config_path)
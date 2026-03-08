import json
from types import SimpleNamespace


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return SimpleNamespace(**cfg)
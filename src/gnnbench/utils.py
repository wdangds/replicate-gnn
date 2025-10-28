import os, random, json, time
import numpy as np
import torch

def seed_everything(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(pref: str="auto"):
    if pref == "cpu": return torch.device("cpu")
    if pref == "cuda": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok = True)

def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

        
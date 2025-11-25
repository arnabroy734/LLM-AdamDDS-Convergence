# utils.py
import os, random, numpy as np, torch

def set_seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

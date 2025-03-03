import torch
import numpy as np


def breif_dict(output: dict, prefix=""):
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            print(f"{prefix}{k}: tensor, {list(v.shape)}")
        elif isinstance(v, np.ndarray):
            print(f"{prefix}{k}: array, {list(v.size)}")
        elif isinstance(v, (str, int, float, list, tuple)):
            print(f"{prefix}{k}: {type(v).__name__}, {v}")
        elif v is None:
            print(f"{prefix}{k}: None")
        else:
            breif_dict(v, f"{k}.")
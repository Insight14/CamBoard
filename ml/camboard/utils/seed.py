"""Reproducible seed utilities across Python, NumPy, and PyTorch."""

import os
import random
import numpy as np
import torch


def fix_seed(seed: int = 42) -> None:
    """Fix all RNG seeds for deterministic and reproducible runs.

    Args:
        seed: Integer seed value (default 42).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS deterministic settings if supported
        pass


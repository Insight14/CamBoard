"""Dataset loader for isolated sign vocabulary (WLASL / ASL Citizen) split by signer."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

WLASL_50_SIGNS = [
    "hello", "thank_you", "please", "sorry", "yes", "no", "help", "more",
    "want", "eat", "drink", "water", "food", "home", "work", "school",
    "book", "car", "go", "come", "stop", "good", "bad", "love",
    "friend", "family", "mother", "father", "child", "name", "what", "where",
    "when", "who", "how", "today", "tomorrow", "yesterday", "morning", "night",
    "hot", "cold", "big", "small", "again", "different", "same", "time",
    "now", "finish"
]
SIGN_LABEL_MAP = {name: i for i, name in enumerate(WLASL_50_SIGNS)}


class SignSequenceDataset(Dataset):
    """Dataset delivering fixed-length landmark sequence windows (seq_len, 132)."""

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        max_seq_len: int = 64,
        input_dim: int = 132,
        augment: bool = False,
        augment_config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.input_dim = input_dim
        self.augment = augment
        self.augment_config = augment_config or {}
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path, label_idx = self.samples[idx]
        seq = np.load(str(file_path))  # shape: (T, input_dim)

        # Pad or truncate to max_seq_len
        t_len = len(seq)
        if t_len > self.max_seq_len:
            # Random or center crop
            start = 0 if not self.augment else self.rng.randint(0, t_len - self.max_seq_len + 1)
            seq = seq[start : start + self.max_seq_len]
        elif t_len < self.max_seq_len:
            pad = np.zeros((self.max_seq_len - t_len, self.input_dim), dtype=np.float32)
            seq = np.vstack([seq, pad])

        if self.augment:
            # Spatial jitter
            jitter = self.rng.normal(0, 0.005, size=seq.shape).astype(np.float32)
            seq = seq + jitter

        return torch.from_numpy(seq).float(), torch.tensor(label_idx, dtype=torch.long)


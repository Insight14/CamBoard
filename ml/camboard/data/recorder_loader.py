"""Dataset loader for in-app recorder collected sessions, split strictly by signer/session."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from camboard.data.augment_landmarks import augment_landmarks
from camboard.data.landmark_utils import extract_features


class RecorderLandmarkDataset(Dataset):
    """PyTorch Dataset loading landmark arrays collected via CamBoard in-app recorder.

    Expected directory structure:
        data_dir/
            signer_01/
                pinch_001.npy
                palm_001.npy
                ...
            signer_02/
                ...
    Each .npy file contains a (21, 3) landmark array or sequence.
    """

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        label_map: Dict[str, int],
        augment: bool = False,
        augment_config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        self.samples = samples
        self.label_map = label_map
        self.augment = augment
        self.augment_config = augment_config or {}
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path, label_idx = self.samples[idx]
        raw_landmarks = np.load(str(file_path))  # shape: (21, 3)

        if self.augment:
            raw_landmarks = augment_landmarks(
                raw_landmarks, config=self.augment_config, rng=self.rng
            )

        feat = extract_features(raw_landmarks)  # shape: (63,)
        return torch.from_numpy(feat).float(), torch.tensor(label_idx, dtype=torch.long)


def load_recorder_splits(
    data_dir: str | Path,
    class_names: List[str],
    val_signer_fraction: float = 0.15,
    test_signer_fraction: float = 0.15,
    augment_train: bool = True,
    augment_config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Tuple[RecorderLandmarkDataset, RecorderLandmarkDataset, RecorderLandmarkDataset]:
    """Discover files, partition strictly by signer/session directory, and return (train, val, test) datasets.

    Strict zero-leakage guarantee: every signer appears in exactly one partition.
    """
    root = Path(data_dir)
    label_map = {name: i for i, name in enumerate(class_names)}

    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    # Discover signer/session subdirectories
    signer_dirs = sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")])
    if not signer_dirs:
        raise ValueError(f"No signer directories found under {root}")

    rng = np.random.RandomState(seed)
    shuffled_signers = list(signer_dirs)
    rng.shuffle(shuffled_signers)

    n_total = len(shuffled_signers)
    n_test = max(1, int(round(n_total * test_signer_fraction)))
    n_val = max(1, int(round(n_total * val_signer_fraction)))
    n_train = max(1, n_total - n_test - n_val)

    test_signers = set(shuffled_signers[:n_test])
    val_signers = set(shuffled_signers[n_test : n_test + n_val])
    train_signers = set(shuffled_signers[n_test + n_val :])

    def collect_samples(target_signers: set) -> List[Tuple[Path, int]]:
        samples = []
        for s_dir in target_signers:
            for npy_file in s_dir.glob("*.npy"):
                stem = npy_file.stem
                # Class name is typically the prefix before '_' (e.g. pinch_001.npy)
                matched_label = None
                for name in class_names:
                    if stem.startswith(name):
                        matched_label = label_map[name]
                        break
                if matched_label is not None:
                    samples.append((npy_file, matched_label))
        return samples

    train_samples = collect_samples(train_signers)
    val_samples = collect_samples(val_signers)
    test_samples = collect_samples(test_signers)

    train_ds = RecorderLandmarkDataset(
        train_samples, label_map, augment=augment_train, augment_config=augment_config, seed=seed
    )
    val_ds = RecorderLandmarkDataset(val_samples, label_map, augment=False, seed=seed + 1)
    test_ds = RecorderLandmarkDataset(test_samples, label_map, augment=False, seed=seed + 2)

    return train_ds, val_ds, test_ds


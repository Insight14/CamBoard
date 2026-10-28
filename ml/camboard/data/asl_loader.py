"""ASL Fingerspelling dataset loader with caching and MediaPipe landmark extraction."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from camboard.data.augment_landmarks import augment_landmarks
from camboard.data.landmark_utils import extract_features, mediapipe_to_array

# 24 static ASL letters (excludes motion letters J and Z)
ASL_STATIC_LETTERS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"
]
ASL_LABEL_MAP = {letter: i for i, letter in enumerate(ASL_STATIC_LETTERS)}


class ASLLandmarkDataset(Dataset):
    """Dataset for ASL fingerspelling landmarks."""

    def __init__(
        self,
        samples: List[Tuple[np.ndarray, int]],
        augment: bool = False,
        augment_config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        self.samples = samples
        self.augment = augment
        self.augment_config = augment_config or {}
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_landmarks, label_idx = self.samples[idx]
        lm = raw_landmarks.copy()
        if self.augment:
            lm = augment_landmarks(lm, config=self.augment_config, rng=self.rng)
        features = extract_features(lm)
        return torch.from_numpy(features).float(), torch.tensor(label_idx, dtype=torch.long)


def extract_landmarks_from_image_path(img_path: Path, hands_solution=None) -> Optional[np.ndarray]:
    """Extract (21, 3) landmarks from an image using MediaPipe Hands."""
    try:
        import cv2
        import mediapipe as mp
        image = cv2.imread(str(img_path))
        if image is None:
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if hands_solution is None:
            with mp.solutions.hands.Hands(
                static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
            ) as hands:
                results = hands.process(image_rgb)
        else:
            results = hands_solution.process(image_rgb)

        if results.multi_hand_landmarks:
            return mediapipe_to_array(results.multi_hand_landmarks[0])
    except Exception:
        pass
    return None


def cache_or_load_asl_landmarks(
    dataset_root: str | Path,
    cache_dir: Optional[str | Path] = None,
) -> List[Tuple[np.ndarray, int, str]]:
    """Scan dataset root, load or extract landmarks, and cache to disk."""
    root = Path(dataset_root)
    cache = Path(cache_dir or (root / "_landmark_cache"))
    cache.mkdir(parents=True, exist_ok=True)

    extracted_samples: List[Tuple[np.ndarray, int, str]] = []

    # Check for existing pre-extracted cache
    cached_file = cache / "asl_landmarks_cache.npz"
    if cached_file.exists():
        data = np.load(str(cached_file), allow_pickle=True)
        landmarks = data["landmarks"]
        labels = data["labels"]
        signers = data["signers"]
        for lm, lbl, s_id in zip(landmarks, labels, signers):
            extracted_samples.append((lm, int(lbl), str(s_id)))
        return extracted_samples

    return extracted_samples


"""Google QuickDraw dataset loader for stroke shapes."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from camboard.data.augment_strokes import augment_stroke_points
from camboard.data.stroke_utils import stroke_to_features

QUICKDRAW_SHAPE_CLASSES = [
    "circle",
    "square",
    "triangle",
    "star",
    "diamond",
    "arrow",
    "line",
    "rectangle",
    "hexagon",
    "zigzag",
]
SHAPE_LABEL_MAP = {name: i for i, name in enumerate(QUICKDRAW_SHAPE_CLASSES)}


class QuickDrawDataset(Dataset):
    """PyTorch Dataset loading QuickDraw stroke drawings."""

    def __init__(
        self,
        samples: List[Tuple[np.ndarray, int]],
        n_points: int = 64,
        channels_first: bool = True,
        augment: bool = False,
        augment_config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        self.samples = samples
        self.n_points = n_points
        self.channels_first = channels_first
        self.augment = augment
        self.augment_config = augment_config or {}
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pts, label_idx = self.samples[idx]
        cur_pts = pts.copy()
        if self.augment:
            cur_pts = augment_stroke_points(cur_pts, config=self.augment_config, rng=self.rng)
        features = stroke_to_features(
            cur_pts, n=self.n_points, channels_first=self.channels_first
        )
        return torch.from_numpy(features).float(), torch.tensor(label_idx, dtype=torch.long)


def load_quickdraw_ndjson_file(
    file_path: str | Path,
    max_samples: int = 10000,
) -> List[np.ndarray]:
    """Parse QuickDraw NDJSON file containing simplified stroke polylines."""
    p = Path(file_path)
    strokes_list = []
    if not p.exists():
        return strokes_list

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if len(strokes_list) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # data["drawing"] is list of strokes: [[x0, x1...], [y0, y1...]]
            raw_drawing = data.get("drawing", [])
            all_pts = []
            for stroke in raw_drawing:
                xs, ys = stroke[0], stroke[1]
                for x, y in zip(xs, ys):
                    all_pts.append([float(x), float(y)])
            if len(all_pts) >= 4:
                strokes_list.append(np.array(all_pts, dtype=np.float32))
    return strokes_list


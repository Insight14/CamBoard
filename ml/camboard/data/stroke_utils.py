"""Stroke resampling, bounding box normalization, and $Q recognizer algorithm."""

from typing import List, Optional, Tuple
import numpy as np
from scipy.interpolate import interp1d


from PIL import Image, ImageDraw


def render_strokes_to_canvas(
    strokes: list,
    width: int = 512,
    height: int = 512,
    stroke_width: int = 3,
    padding: int = 32,
) -> np.ndarray:
    """Render 2D strokes onto normalized binary bitmap canvas (1, H, W) for math model input."""
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)

    for stroke in strokes:
        pts = np.asarray(stroke, dtype=np.float32)
        if len(pts) >= 2:
            xy_tuples = [(float(p[0]), float(p[1])) for p in pts]
            draw.line(xy_tuples, fill=0, width=stroke_width)

    arr = np.array(img, dtype=np.float32) / 255.0
    inverted = 1.0 - arr
    return np.expand_dims(inverted, axis=0)


def resample_stroke_points(points: np.ndarray, n: int = 64) -> np.ndarray:
    """Resample arbitrary (T, 2) stroke polyline to exactly n equidistant points.

    Uses cumulative arc-length parameterization with linear interpolation.
    """
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return np.zeros((n, 2), dtype=np.float32)
    if len(pts) == 1:
        return np.repeat(pts, n, axis=0)

    # Compute Euclidean step distances along the polyline
    deltas = np.diff(pts, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    cum_dist = np.insert(np.cumsum(distances), 0, 0.0)
    total_length = cum_dist[-1]

    if total_length < 1e-6:
        return np.repeat(pts[:1], n, axis=0)

    # Remove duplicate points along arc-length
    unique_mask = np.insert(distances > 1e-7, 0, True)
    pts = pts[unique_mask]
    cum_dist = cum_dist[unique_mask]

    if len(cum_dist) == 1:
        return np.repeat(pts[:1], n, axis=0)

    interp_fn = interp1d(cum_dist, pts, axis=0, kind="linear")
    uniform_s = np.linspace(0.0, cum_dist[-1], n)
    return interp_fn(uniform_s).astype(np.float32)


def normalize_stroke(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Normalize stroke coordinates to [0, 1] x [0, 1] while preserving aspect ratio.

    Returns:
        normalized_points: (n, 2) scaled and centered in unit square.
        aspect_ratio: width / height ratio.
    """
    pts = np.asarray(points, dtype=np.float32)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    box_size = max_xy - min_xy
    width, height = float(box_size[0]), float(box_size[1])

    max_dim = max(width, height, 1e-6)
    aspect_ratio = width / max(height, 1e-6)

    # Center within unit box
    centered = pts - min_xy
    offset = (max_dim - box_size) / 2.0
    normalized = (centered + offset) / max_dim
    return normalized.astype(np.float32), aspect_ratio


def stroke_to_features(
    points: np.ndarray,
    n: int = 64,
    channels_first: bool = False,
) -> np.ndarray:
    """Convert stroke polyline to (n, 3) feature array with (dx, dy, pen_state).

    pen_state is 1.0 for in-stroke drawing.
    If channels_first is True, returns (3, n) for 1D-CNN.
    """
    resampled = resample_stroke_points(points, n=n)
    normalized, _ = normalize_stroke(resampled)

    # Compute deltas
    deltas = np.zeros_like(normalized)
    deltas[1:] = normalized[1:] - normalized[:-1]
    pen_state = np.ones((n, 1), dtype=np.float32)

    features = np.hstack([deltas, pen_state]).astype(np.float32)  # (n, 3)
    if channels_first:
        return features.T  # (3, n)
    return features


def multi_stroke_to_features(
    strokes: List[np.ndarray],
    n_total: int = 64,
    channels_first: bool = False,
) -> np.ndarray:
    """Convert multi-stroke drawing (list of polylines) to unified (n_total, 3) features.

    pen_state is 0.0 at pen-lift transitions.
    """
    if not strokes:
        res = np.zeros((n_total, 3), dtype=np.float32)
        return res.T if channels_first else res

    total_pts = sum(max(1, len(s)) for s in strokes)
    allocated_points = []
    accum = 0
    for i, s in enumerate(strokes):
        n_pts = max(4, int(round((len(s) / max(1, total_pts)) * n_total)))
        if i == len(strokes) - 1:
            n_pts = max(4, n_total - accum)
        allocated_points.append(n_pts)
        accum += n_pts

    all_feat_list = []
    for s, n_pts in zip(strokes, allocated_points):
        resampled = resample_stroke_points(s, n=n_pts)
        norm_s, _ = normalize_stroke(resampled)
        d = np.zeros_like(norm_s)
        d[1:] = norm_s[1:] - norm_s[:-1]
        p_state = np.ones((n_pts, 1), dtype=np.float32)
        p_state[0] = 0.0  # pen-down marker
        all_feat_list.append(np.hstack([d, p_state]))

    merged = np.vstack(all_feat_list)
    if len(merged) > n_total:
        merged = merged[:n_total]
    elif len(merged) < n_total:
        pad = np.zeros((n_total - len(merged), 3), dtype=np.float32)
        merged = np.vstack([merged, pad])

    return merged.T if channels_first else merged


class DollarQRecognizer:
    """$Q Super-Quick Gesture Recognizer implementation for baseline comparison.

    Implements Point-Cloud distance matching against template stroke point sets.
    """

    def __init__(self, n_points: int = 32):
        self.n_points = n_points
        self.templates: List[Tuple[str, np.ndarray]] = []

    def add_template(self, name: str, points: np.ndarray) -> None:
        """Add named template after resampling and normalization."""
        resampled = resample_stroke_points(points, n=self.n_points)
        norm, _ = normalize_stroke(resampled)
        self.templates.append((name, norm))

    def recognize(self, points: np.ndarray) -> Tuple[str, float]:
        """Classify candidate stroke points by minimum point-cloud distance."""
        if not self.templates:
            return "none", 0.0

        resampled = resample_stroke_points(points, n=self.n_points)
        candidate, _ = normalize_stroke(resampled)

        best_score = float("inf")
        best_name = "unknown"

        for name, tmpl in self.templates:
            dist = float(np.mean(np.linalg.norm(candidate - tmpl, axis=1)))
            if dist < best_score:
                best_score = dist
                best_name = name

        # Convert distance to score in [0, 1]
        confidence = float(np.exp(-best_score * 3.0))
        return best_name, confidence

    def evaluate(self, test_samples: List[Tuple[np.ndarray, str]]) -> float:
        """Evaluate top-1 classification accuracy across test set."""
        if not test_samples:
            return 0.0
        correct = 0
        for pts, true_label in test_samples:
            pred_label, _ = self.recognize(pts)
            if pred_label == true_label:
                correct += 1
        return correct / len(test_samples)

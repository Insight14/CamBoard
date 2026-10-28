"""Stroke data augmentations: rotation, scale, shear, jitter, speed, dropout, mirror."""

from typing import Any, Dict, Optional, Tuple
import numpy as np


def rotate_stroke_2d(points: np.ndarray, max_angle_deg: float = 20.0, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Rotate stroke coordinates about its center."""
    if rng is None:
        rng = np.random.RandomState()
    angle = np.radians(rng.uniform(-max_angle_deg, max_angle_deg))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot_m = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    center = points.mean(axis=0)
    centered = points - center
    rotated = centered @ rot_m.T
    return rotated + center


def shear_stroke_2d(points: np.ndarray, max_shear: float = 0.2, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Apply affine shear (slant) to simulate handwriting variations."""
    if rng is None:
        rng = np.random.RandomState()
    shear_factor = rng.uniform(-max_shear, max_shear)
    shear_m = np.array([[1.0, shear_factor], [0.0, 1.0]], dtype=np.float32)
    center = points.mean(axis=0)
    return ((points - center) @ shear_m.T) + center


def add_point_jitter_stroke(points: np.ndarray, std: float = 0.005, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Add small Gaussian jitter to stroke coordinates."""
    if rng is None:
        rng = np.random.RandomState()
    noise = rng.normal(0.0, std, size=points.shape).astype(np.float32)
    return points + noise


def stroke_dropout(points: np.ndarray, drop_prob: float = 0.05, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Randomly drop intermediate points to simulate uneven camera frame capture."""
    if rng is None:
        rng = np.random.RandomState()
    if len(points) <= 4:
        return points
    mask = rng.rand(len(points)) > drop_prob
    # Ensure endpoints are preserved
    mask[0] = True
    mask[-1] = True
    return points[mask]


def mirror_stroke_horizontal(points: np.ndarray) -> np.ndarray:
    """Mirror stroke horizontally across its vertical centroid."""
    mirrored = points.copy()
    center_x = float(np.mean(mirrored[:, 0]))
    mirrored[:, 0] = 2 * center_x - mirrored[:, 0]
    return mirrored


def time_warp_stroke(points: np.ndarray, warp_range: Tuple[float, float] = (0.8, 1.2), rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Simulate non-linear drawing speed variation along the path."""
    if rng is None:
        rng = np.random.RandomState()
    n = len(points)
    if n <= 4:
        return points

    # Distort progression with cubic polynomial
    s = np.linspace(0, 1, n)
    factor = rng.uniform(warp_range[0], warp_range[1])
    warped_s = s ** factor
    warped_s = (warped_s - warped_s[0]) / (warped_s[-1] - warped_s[0] + 1e-6)

    warped_idx = (warped_s * (n - 1)).astype(np.int32)
    return points[np.clip(warped_idx, 0, n - 1)]


def augment_stroke_points(
    points: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Run full augmentation pipeline on raw stroke coordinates before feature extraction."""
    if rng is None:
        rng = np.random.RandomState()
    cfg = config or {}
    out = points.copy()

    # Rotation
    if rng.rand() < float(cfg.get("rotate_p", 0.5)):
        out = rotate_stroke_2d(out, max_angle_deg=float(cfg.get("max_angle_deg", 20.0)), rng=rng)

    # Shear
    if rng.rand() < float(cfg.get("shear_p", 0.3)):
        out = shear_stroke_2d(out, max_shear=float(cfg.get("max_shear", 0.2)), rng=rng)

    # Point jitter
    if rng.rand() < float(cfg.get("jitter_p", 0.5)):
        out = add_point_jitter_stroke(out, std=float(cfg.get("jitter_std", 0.005)), rng=rng)

    # Time warp / speed variation
    if rng.rand() < float(cfg.get("time_warp_p", 0.5)):
        out = time_warp_stroke(out, warp_range=tuple(cfg.get("time_warp_range", (0.8, 1.2))), rng=rng)

    # Stroke dropout
    if rng.rand() < float(cfg.get("dropout_p", 0.3)):
        out = stroke_dropout(out, drop_prob=float(cfg.get("dropout_prob", 0.05)), rng=rng)

    # Mirroring
    if rng.rand() < float(cfg.get("mirror_p", 0.2)):
        out = mirror_stroke_horizontal(out)

    return out.astype(np.float32)


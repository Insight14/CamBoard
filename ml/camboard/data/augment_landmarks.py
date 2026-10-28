"""Data augmentation for 3D hand landmark representations."""

from typing import Any, Dict, Optional, Tuple
import numpy as np


def rotate_landmarks_xy(
    landmarks: np.ndarray,
    max_angle_deg: float = 15.0,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Random rotation in the image (x-y) plane by angle in [-max_angle_deg, +max_angle_deg]."""
    if rng is None:
        rng = np.random.RandomState()
    angle = np.radians(rng.uniform(-max_angle_deg, max_angle_deg))
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    rot_matrix = np.array([
        [cos_a, -sin_a, 0.0],
        [sin_a,  cos_a, 0.0],
        [0.0,    0.0,   1.0],
    ], dtype=np.float32)

    return (landmarks @ rot_matrix.T).astype(np.float32)


def scale_landmarks(
    landmarks: np.ndarray,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Random uniform scaling of landmark coordinates."""
    if rng is None:
        rng = np.random.RandomState()
    factor = rng.uniform(scale_range[0], scale_range[1])
    return (landmarks * factor).astype(np.float32)


def add_gaussian_noise(
    landmarks: np.ndarray,
    std: float = 0.01,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Add zero-mean Gaussian jitter to coordinates."""
    if rng is None:
        rng = np.random.RandomState()
    noise = rng.normal(0.0, std, size=landmarks.shape).astype(np.float32)
    return landmarks + noise


def mirror_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Mirror hand horizontally (simulate left/right hand swap)."""
    mirrored = landmarks.copy()
    mirrored[:, 0] = -mirrored[:, 0]
    return mirrored


def augment_landmarks(
    landmarks: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Apply stochastic augmentation pipeline to (21, 3) landmarks."""
    if rng is None:
        rng = np.random.RandomState()
    cfg = config or {}

    out = landmarks.copy()

    # Rotation
    rot_p = float(cfg.get("rotation_p", 0.5))
    max_angle = float(cfg.get("max_angle_deg", 15.0))
    if rng.rand() < rot_p:
        out = rotate_landmarks_xy(out, max_angle_deg=max_angle, rng=rng)

    # Scaling
    scale_p = float(cfg.get("scale_p", 0.5))
    scale_range = tuple(cfg.get("scale_range", (0.9, 1.1)))
    if rng.rand() < scale_p:
        out = scale_landmarks(out, scale_range=scale_range, rng=rng)

    # Gaussian noise
    noise_p = float(cfg.get("noise_p", 0.5))
    noise_std = float(cfg.get("noise_std", 0.01))
    if rng.rand() < noise_p:
        out = add_gaussian_noise(out, std=noise_std, rng=rng)

    # Left/Right hand mirroring
    mirror_p = float(cfg.get("mirror_p", 0.5))
    if rng.rand() < mirror_p:
        out = mirror_landmarks(out)

    return out.astype(np.float32)


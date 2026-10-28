"""Synthetic data generators for hand gestures, strokes, shapes, and characters.

Enables reproducible local training, fixture creation, and wobble modeling.
"""

import math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def generate_synthetic_hand_landmarks(
    gesture_name: str,
    n_samples: int = 50,
    wobble: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """Generate (n_samples, 21, 3) realistic synthetic 3D hand landmarks for a specific gesture."""
    rng = np.random.RandomState(seed)
    base_landmarks = np.zeros((21, 3), dtype=np.float32)

    # Base palm geometry
    base_landmarks[0] = [0.0, 0.0, 0.0]  # wrist
    base_landmarks[1] = [-0.2, 0.1, 0.0]  # thumb cmc
    base_landmarks[2] = [-0.3, 0.25, 0.0]
    base_landmarks[3] = [-0.35, 0.35, 0.0]
    base_landmarks[4] = [-0.4, 0.45, 0.0]  # thumb tip

    base_landmarks[5] = [-0.15, 0.4, 0.0]  # index mcp
    base_landmarks[6] = [-0.15, 0.6, 0.0]
    base_landmarks[7] = [-0.15, 0.75, 0.0]
    base_landmarks[8] = [-0.15, 0.9, 0.0]  # index tip

    base_landmarks[9] = [0.0, 0.42, 0.0]  # middle mcp
    base_landmarks[10] = [0.0, 0.65, 0.0]
    base_landmarks[11] = [0.0, 0.82, 0.0]
    base_landmarks[12] = [0.0, 0.98, 0.0]  # middle tip

    base_landmarks[13] = [0.15, 0.38, 0.0]  # ring mcp
    base_landmarks[14] = [0.15, 0.58, 0.0]
    base_landmarks[15] = [0.15, 0.73, 0.0]
    base_landmarks[16] = [0.15, 0.88, 0.0]  # ring tip

    base_landmarks[17] = [0.28, 0.32, 0.0]  # pinky mcp
    base_landmarks[18] = [0.30, 0.48, 0.0]
    base_landmarks[19] = [0.32, 0.60, 0.0]
    base_landmarks[20] = [0.34, 0.72, 0.0]  # pinky tip

    # Gesture-specific anatomical modifications
    if gesture_name == "pinch":
        # Thumb tip and index tip touch
        base_landmarks[4] = [-0.15, 0.65, 0.0]
        base_landmarks[8] = [-0.16, 0.65, 0.0]
    elif gesture_name == "fist":
        # Curl all fingers down toward palm
        for tip in [8, 12, 16, 20]:
            base_landmarks[tip, 1] *= 0.45
            base_landmarks[tip, 2] -= 0.15
    elif gesture_name == "point":
        # Index extended, others curled
        for tip in [12, 16, 20]:
            base_landmarks[tip, 1] *= 0.45
    elif gesture_name == "thumbs_up":
        # Fist with thumb pointing upward
        for tip in [8, 12, 16, 20]:
            base_landmarks[tip, 1] *= 0.45
        base_landmarks[4] = [-0.1, 0.8, 0.1]
    elif gesture_name == "palm_out":
        # Splayed fingers forward
        base_landmarks[:, 2] += 0.05

    samples = []
    for _ in range(n_samples):
        noise = rng.normal(0, wobble, size=base_landmarks.shape).astype(np.float32)
        scale = rng.uniform(0.92, 1.08)
        samples.append((base_landmarks + noise) * scale)

    return np.array(samples, dtype=np.float32)


def generate_synthetic_shape_stroke(
    shape_name: str,
    n_points: int = 100,
    wobble_sigma: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """Generate (n_points, 2) stroke polyline of geometric shapes with human-like drawing wobble."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, n_points)

    if shape_name == "circle":
        x = np.cos(t)
        y = np.sin(t)
    elif shape_name == "square":
        # 4 corners
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=np.float32)
        idx = np.linspace(0, 4, n_points)
        x = np.interp(idx, np.arange(5), corners[:, 0])
        y = np.interp(idx, np.arange(5), corners[:, 1])
    elif shape_name == "triangle":
        corners = np.array([[0, 1], [0.866, -0.5], [-0.866, -0.5], [0, 1]], dtype=np.float32)
        idx = np.linspace(0, 3, n_points)
        x = np.interp(idx, np.arange(4), corners[:, 0])
        y = np.interp(idx, np.arange(4), corners[:, 1])
    elif shape_name == "star":
        angles = np.linspace(0, 2 * np.pi, 11)
        r = np.array([1.0, 0.4] * 5 + [1.0])
        corners_x = r * np.cos(angles - np.pi / 2)
        corners_y = r * np.sin(angles - np.pi / 2)
        idx = np.linspace(0, 10, n_points)
        x = np.interp(idx, np.arange(11), corners_x)
        y = np.interp(idx, np.arange(11), corners_y)
    elif shape_name == "diamond":
        corners = np.array([[0, 1], [0.8, 0], [0, -1], [-0.8, 0], [0, 1]], dtype=np.float32)
        idx = np.linspace(0, 4, n_points)
        x = np.interp(idx, np.arange(5), corners[:, 0])
        y = np.interp(idx, np.arange(5), corners[:, 1])
    elif shape_name == "arrow":
        corners = np.array([[-1, 0], [0.5, 0], [0.5, 0.4], [1.0, 0], [0.5, -0.4], [0.5, 0]], dtype=np.float32)
        idx = np.linspace(0, 5, n_points)
        x = np.interp(idx, np.arange(6), corners[:, 0])
        y = np.interp(idx, np.arange(6), corners[:, 1])
    elif shape_name == "line":
        x = np.linspace(-1, 1, n_points)
        y = np.zeros(n_points)
    elif shape_name == "rectangle":
        corners = np.array([[-1.2, -0.6], [1.2, -0.6], [1.2, 0.6], [-1.2, 0.6], [-1.2, -0.6]], dtype=np.float32)
        idx = np.linspace(0, 4, n_points)
        x = np.interp(idx, np.arange(5), corners[:, 0])
        y = np.interp(idx, np.arange(5), corners[:, 1])
    elif shape_name == "hexagon":
        angles = np.linspace(0, 2 * np.pi, 7)
        corners_x = np.cos(angles)
        corners_y = np.sin(angles)
        idx = np.linspace(0, 6, n_points)
        x = np.interp(idx, np.arange(7), corners_x)
        y = np.interp(idx, np.arange(7), corners_y)
    elif shape_name == "zigzag":
        corners = np.array([[-1, -0.5], [-0.5, 0.5], [0, -0.5], [0.5, 0.5], [1, -0.5]], dtype=np.float32)
        idx = np.linspace(0, 4, n_points)
        x = np.interp(idx, np.arange(5), corners[:, 0])
        y = np.interp(idx, np.arange(5), corners[:, 1])
    else:
        x = np.cos(t)
        y = np.sin(t)

    pts = np.column_stack([x, y])
    # Add human tremor / wobble
    tremor = rng.normal(0, wobble_sigma, size=pts.shape).astype(np.float32)
    return pts + tremor


def generate_synthetic_character_stroke(
    char: str,
    n_points: int = 64,
    wobble_sigma: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """Generate (n_points, 2) single-stroke representation of alphanumeric characters."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, n_points)

    # Simplified single-stroke templates
    if char in ["O", "0", "o"]:
        ang = t * 2 * np.pi
        pts = np.column_stack([0.6 * np.cos(ang), np.sin(ang)])
    elif char in ["I", "1", "l"]:
        pts = np.column_stack([np.zeros(n_points), 2 * t - 1.0])
    elif char in ["C", "c"]:
        ang = (1 - t) * 1.5 * np.pi + 0.25 * np.pi
        pts = np.column_stack([np.cos(ang), np.sin(ang)])
    elif char in ["S", "s"]:
        ang = t * 2 * np.pi
        pts = np.column_stack([np.sin(ang), np.cos(ang * 0.5)])
    elif char in ["V", "v"]:
        corners = np.array([[-0.8, 1.0], [0.0, -1.0], [0.8, 1.0]], dtype=np.float32)
        idx = np.linspace(0, 2, n_points)
        x = np.interp(idx, np.arange(3), corners[:, 0])
        y = np.interp(idx, np.arange(3), corners[:, 1])
        pts = np.column_stack([x, y])
    elif char in ["Z", "z"]:
        corners = np.array([[-0.8, 1.0], [0.8, 1.0], [-0.8, -1.0], [0.8, -1.0]], dtype=np.float32)
        idx = np.linspace(0, 3, n_points)
        x = np.interp(idx, np.arange(4), corners[:, 0])
        y = np.interp(idx, np.arange(4), corners[:, 1])
        pts = np.column_stack([x, y])
    else:
        # Default smooth curve
        pts = np.column_stack([np.sin(t * np.pi * 2), np.cos(t * np.pi)])

    tremor = rng.normal(0, wobble_sigma, size=pts.shape).astype(np.float32)
    return (pts + tremor).astype(np.float32)


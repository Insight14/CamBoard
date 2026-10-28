"""MediaPipe 21-landmark normalization, palm-scaling, and velocity extraction."""

from typing import Optional, Tuple, Union
import numpy as np

# Landmark indices defined by MediaPipe Hands
WRIST_IDX = 0
THUMB_CMC = 1
INDEX_FINGER_MCP = 5
MIDDLE_FINGER_MCP = 9
RING_FINGER_MCP = 13
PINKY_MCP = 17

PALM_INDICES = [INDEX_FINGER_MCP, MIDDLE_FINGER_MCP, RING_FINGER_MCP, PINKY_MCP]


def extract_features(
    landmarks: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize raw (21, 3) MediaPipe hand landmarks to 63-dim feature vector.

    Steps:
      1. Translate wrist (landmark 0) to origin (0, 0, 0).
      2. Compute palm size as Euclidean distance from wrist to MCP joints (landmarks 5, 9, 13, 17).
      3. Scale all coordinates by palm size to achieve camera-distance invariance.
      4. Flatten to 63-dimensional 1D array.

    Args:
        landmarks: Array of shape (21, 3).
        eps: Small epsilon to guard against division by zero.

    Returns:
        Flattened normalized features of shape (63,).
    """
    coords = np.asarray(landmarks, dtype=np.float32)
    assert coords.shape == (21, 3), f"Expected shape (21, 3), got {coords.shape}"

    # Step 1: Wrist-relative translation
    wrist = coords[WRIST_IDX:WRIST_IDX + 1, :]  # (1, 3)
    rel_coords = coords - wrist

    # Step 2: Palm scale calculation (average distance from wrist to MCP knuckles)
    mcp_coords = rel_coords[PALM_INDICES, :2]  # use 2D planar distance for scale
    distances = np.linalg.norm(mcp_coords, axis=1)
    palm_scale = float(np.mean(distances))
    if palm_scale < eps:
        palm_scale = 1.0

    # Step 3: Scale invariance
    normalized = rel_coords / palm_scale

    # Step 4: Flatten to 63 features
    return normalized.flatten()


def extract_velocity_features(
    curr_landmarks: np.ndarray,
    prev_landmarks: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Extract 63-dim inter-frame velocity features from consecutive landmark frames."""
    curr_norm = extract_features(curr_landmarks, eps=eps)
    prev_norm = extract_features(prev_landmarks, eps=eps)
    return curr_norm - prev_norm


def extract_full_features(
    curr_landmarks: np.ndarray,
    prev_landmarks: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Extract features: 63 static features or 126 features (63 static + 63 velocity)."""
    static_feat = extract_features(curr_landmarks, eps=eps)
    if prev_landmarks is None:
        return static_feat
    vel_feat = extract_velocity_features(curr_landmarks, prev_landmarks, eps=eps)
    return np.concatenate([static_feat, vel_feat], axis=0)


def mediapipe_to_array(hand_landmarks) -> np.ndarray:
    """Convert MediaPipe NormalizedLandmarkList protocol buffer to (21, 3) NumPy array."""
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


"""MLP classifier for hand-landmark gestures and ASL fingerspelling."""

from typing import Any, Dict
import torch
import torch.nn as nn
from camboard.training.onnx_export import register_model


class LandmarkMLP(nn.Module):
    """Multi-Layer Perceptron for 21-landmark hand feature classification.

    Architecture:
        Input (63 or 126 with velocity)
        -> Linear(128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.2)
        -> Linear(64)  -> BatchNorm1d(64)  -> ReLU -> Dropout(0.2)
        -> Linear(num_classes)
    """

    def __init__(self, input_dim: int = 63, num_classes: int = 8, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handles 1D (single vector) or 2D (batched) tensors
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


def build_gesture_mlp(num_classes: int = 8, with_velocity: bool = False, dropout: float = 0.2) -> LandmarkMLP:
    """Build Gesture MLP: 8 classes (pinch, palm, fist, point, swipe-start, thumbs-up, palm-out, none)."""
    input_dim = 126 if with_velocity else 63
    return LandmarkMLP(input_dim=input_dim, num_classes=num_classes, dropout=dropout)


def build_fingerspelling_mlp(num_classes: int = 24, dropout: float = 0.2) -> LandmarkMLP:
    """Build ASL static fingerspelling MLP: 24 letters (A-Y excluding J and Z)."""
    return LandmarkMLP(input_dim=63, num_classes=num_classes, dropout=dropout)


@register_model("gesture")
def _create_gesture(cfg: Dict[str, Any]) -> nn.Module:
    return build_gesture_mlp(
        num_classes=int(cfg.get("num_classes", 8)),
        with_velocity=bool(cfg.get("with_velocity", False)),
        dropout=float(cfg.get("dropout", 0.2)),
    )


@register_model("fingerspelling")
def _create_fingerspelling(cfg: Dict[str, Any]) -> nn.Module:
    return build_fingerspelling_mlp(
        num_classes=int(cfg.get("num_classes", 24)),
        dropout=float(cfg.get("dropout", 0.2)),
    )


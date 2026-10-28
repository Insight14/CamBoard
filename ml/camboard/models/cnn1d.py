"""1D-CNN architecture for stroke shape recognition."""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from camboard.training.onnx_export import register_model


class ResidualBlock1D(nn.Module):
    """Residual convolutional block for 1D stroke sequences."""

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)


class CNN1DClassifier(nn.Module):
    """1D Convolutional Neural Network for stroke recognition.

    Input shape: (batch_size, input_dim=3, seq_len=64) [channels-first]
    Features: (dx, dy, pen_state)
    Parameter count: ~0.5M - 1.5M parameters
    """

    def __init__(
        self,
        input_dim: int = 3,
        num_classes: int = 10,
        channels: Optional[List[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256]

        self.input_conv = nn.Sequential(
            nn.Conv1d(input_dim, channels[0], kernel_size=5, padding=2),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
        )

        blocks = []
        in_ch = channels[0]
        for out_ch in channels:
            if out_ch != in_ch:
                blocks.append(
                    nn.Sequential(
                        nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, stride=2),
                        nn.BatchNorm1d(out_ch),
                        nn.ReLU(),
                    )
                )
            blocks.append(ResidualBlock1D(out_ch, dropout=dropout * 0.5))
            in_ch = out_ch

        self.backbone = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 3, 64)
        feat = self.input_conv(x)
        feat = self.backbone(feat)
        feat = self.pool(feat)
        return self.head(feat)


@register_model("stroke_shapes")
def _create_stroke_shapes(cfg: Dict[str, Any]) -> nn.Module:
    channels = cfg.get("channels", [64, 128, 256])
    if hasattr(channels, "__iter__"):
        channels = [int(c) for c in channels]
    return CNN1DClassifier(
        input_dim=int(cfg.get("input_dim", 3)),
        num_classes=int(cfg.get("num_classes", 10)),
        channels=channels,
        dropout=float(cfg.get("dropout", 0.2)),
    )


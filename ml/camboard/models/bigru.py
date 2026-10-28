"""Bidirectional GRU with auxiliary orientation head for stroke characters."""

from typing import Any, Dict, Tuple, Union
import torch
import torch.nn as nn
from camboard.training.onnx_export import register_model


class BiGRUClassifier(nn.Module):
    """2-layer Bidirectional GRU for air-drawn and online handwriting characters.

    Multi-task architecture:
    - Primary Head: Character classification (A-Z, a-z, 0-9, symbols)
    - Auxiliary Head: Orientation prediction (normal, mirrored, rotated, mirrored+rotated)
      enabling robust canonical character mapping even under inverted/slanted input.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 62,
        num_orientations: int = 4,
        dropout: float = 0.2,
        orientation_head: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.orientation_head = orientation_head

        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        feat_dim = hidden_dim * 2

        self.char_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

        if orientation_head:
            self.orient_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feat_dim, 64),
                nn.ReLU(),
                nn.Linear(64, num_orientations),
            )
        else:
            self.orient_head = None

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: (batch, seq_len=64, input_dim=3)
        out, _ = self.bigru(x)  # (batch, seq_len, hidden_dim * 2)
        summary = torch.mean(out, dim=1)  # mean pooling over sequence

        char_logits = self.char_head(summary)
        if self.training and self.orientation_head and self.orient_head is not None:
            orient_logits = self.orient_head(summary)
            return char_logits, orient_logits

        # In eval/export mode, return primary character logits
        return char_logits


@register_model("stroke_chars")
def _create_stroke_chars(cfg: Dict[str, Any]) -> nn.Module:
    return BiGRUClassifier(
        input_dim=int(cfg.get("input_dim", 3)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        num_layers=int(cfg.get("num_layers", 2)),
        num_classes=int(cfg.get("num_classes", 62)),
        dropout=float(cfg.get("dropout", 0.2)),
        orientation_head=bool(cfg.get("orientation_head", True)),
    )


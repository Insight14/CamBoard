"""2-layer GRU sequence classifier for motion letters (J, Z) and gesture dynamics."""

from typing import Any, Dict
import torch
import torch.nn as nn
from camboard.training.onnx_export import register_model


class GRUSequenceClassifier(nn.Module):
    """2-layer Gated Recurrent Unit for temporal landmark sequences.

    Input shape: (batch_size, seq_len=30, input_dim=126)
    where input_dim is 63 wrist-relative landmarks + 63 inter-frame deltas.
    """

    def __init__(
        self,
        input_dim: int = 126,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 26,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        _, h_n = self.gru(x)  # h_n: (num_layers, batch, hidden_dim)
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        return self.classifier(last_hidden)


@register_model("motion_letters")
def _create_motion_letters(cfg: Dict[str, Any]) -> nn.Module:
    return GRUSequenceClassifier(
        input_dim=int(cfg.get("input_dim", 126)),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        num_layers=int(cfg.get("num_layers", 2)),
        num_classes=int(cfg.get("num_classes", 26)),
        dropout=float(cfg.get("dropout", 0.2)),
    )


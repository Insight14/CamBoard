"""Temporal Transformer for isolated sign vocabulary (50-200 signs)."""

import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from camboard.training.onnx_export import register_model


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence tokens."""

    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]  # type: ignore
        return self.dropout(x)


class SignVocabTransformer(nn.Module):
    """Compact Temporal Transformer for ASL vocabulary recognition (< 3M params).

    Input: (batch_size, seq_len=64, input_dim=132)
      - 2 hands * 21 landmarks * 3 coords = 126
      - + 6 upper-body keypoints (left/right shoulders, elbows, wrists)
    Outputs:
      - Class logits: (batch_size, num_classes=50)
      - Confidence score: (batch_size, 1) in [0, 1] for thresholded abstention
    """

    def __init__(
        self,
        input_dim: int = 132,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 50,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes

        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len + 1, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Classification and confidence estimation heads
        self.classifier = nn.Linear(d_model, num_classes)
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: (batch, seq_len, input_dim)
        b, t, _ = x.shape
        tokens = self.input_proj(x)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (b, t+1, d_model)
        tokens = self.pos_encoder(tokens)

        if padding_mask is not None:
            # Prepend False for CLS token
            cls_mask = torch.zeros((b, 1), dtype=torch.bool, device=x.device)
            full_mask = torch.cat([cls_mask, padding_mask], dim=1)
        else:
            full_mask = None

        encoded = self.transformer(tokens, src_key_padding_mask=full_mask)
        encoded = self.norm(encoded)
        cls_rep = encoded[:, 0, :]  # aggregated CLS token representation

        logits = self.classifier(cls_rep)
        conf = self.confidence_head(cls_rep)

        # In training, return logits and confidence
        if self.training:
            return logits, conf
        # In export / evaluation, can return both or logits
        return logits, conf


@register_model("sign_vocab")
def _create_sign_vocab(cfg: Dict[str, Any]) -> nn.Module:
    return SignVocabTransformer(
        input_dim=int(cfg.get("input_dim", 132)),
        d_model=int(cfg.get("d_model", 128)),
        num_heads=int(cfg.get("num_heads", 4)),
        num_layers=int(cfg.get("num_layers", 2)),
        num_classes=int(cfg.get("num_classes", 50)),
        dropout=float(cfg.get("dropout", 0.1)),
        max_seq_len=int(cfg.get("max_seq_len", 64)),
    )


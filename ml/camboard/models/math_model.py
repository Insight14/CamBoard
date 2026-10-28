"""Math recognition model wrappers (TAMER / BTTR style encoder-decoder architecture)."""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from camboard.training.onnx_export import register_model


class MathEncoder(nn.Module):
    """Visual encoder for rendered stroke images (e.g. 1 x 512 x 512).

    Produces spatial feature map / token sequence for the LaTeX decoder.
    """

    def __init__(self, in_channels: int = 1, d_model: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),  # 256x256
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128x128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32
            nn.Conv2d(128, d_model, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),  # 8x8 = 64 tokens
        )
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 512, 512)
        feat = self.conv(x)  # (batch, d_model, 8, 8)
        b, c, h, w = feat.shape
        tokens = feat.view(b, c, h * w).permute(0, 2, 1)  # (batch, 64, d_model)
        return self.proj(tokens)


class MathDecoder(nn.Module):
    """Autoregressive decoder predicting LaTeX symbol token IDs."""

    def __init__(self, vocab_size: int = 110, d_model: int = 256, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.GRU(d_model, d_model, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        encoder_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # tgt_tokens: (batch, tgt_seq_len)
        emb = self.embed(tgt_tokens)
        out, _ = self.rnn(emb)
        return self.head(out)


class FullMathModel(nn.Module):
    """Composite model encapsulating encoder + symbol prediction for direct export."""

    def __init__(self, in_channels: int = 1, vocab_size: int = 110, d_model: int = 256):
        super().__init__()
        self.encoder = MathEncoder(in_channels=in_channels, d_model=d_model)
        self.symbol_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 512, 512)
        tokens = self.encoder(x)  # (batch, 64, d_model)
        # pooled symbol logits for fast online inference
        pooled = tokens.permute(0, 2, 1)  # (batch, d_model, 64)
        return self.symbol_head(pooled)


@register_model("math")
def _create_math(cfg: Dict[str, Any]) -> nn.Module:
    return FullMathModel(
        in_channels=int(cfg.get("in_channels", 1)),
        vocab_size=int(cfg.get("vocab_size", 110)),
        d_model=int(cfg.get("d_model", 256)),
    )


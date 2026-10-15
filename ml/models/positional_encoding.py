"""
Positional Encoding
--------------------
Classic sinusoidal positional encoding (Vaswani et al., 2017).

PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

The encoding is added to the token-embedding tensor and is NOT learned.
A learned-embedding variant could be substituted without changing the
encoder/decoder call sites.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as a non-learnable buffer.

    Args:
        d_model:  Embedding / model dimension.
        max_len:  Maximum sequence length to pre-compute.
        dropout:  Applied after adding the positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build (max_len, d_model) table once; never updated by backprop
        pe = torch.zeros(max_len, d_model)                    # (L, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10_000.0) / d_model)
        )                                                       # (D/2,)

        pe[:, 0::2] = torch.sin(position * div_term)          # even dims
        pe[:, 1::2] = torch.cos(position * div_term)          # odd dims
        pe = pe.unsqueeze(0)                                   # (1, L, D) — batch-broadcastable

        # Register as a buffer so it moves with .to(device) but is not a param
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) token embeddings.

        Returns:
            Tensor of same shape with positional encoding added.
        """
        # self.pe is (1, max_len, d_model); slice to actual seq_len
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

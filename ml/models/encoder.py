"""
SentriEncoder
-------------
Custom Transformer encoder built from `torch.nn` primitives.

Pipeline:
    input_ids  ──► TokenEmbedding ──► PositionalEncoding
                ──► N × EncoderLayer (MultiheadSelfAttn + FFN + LN + Residual)
                ──► encoder_output  (batch, src_len, d_model)

No pretrained weights are loaded.  The encoder is initialised randomly
and trained from scratch on SentriMail data.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ml.models.positional_encoding import PositionalEncoding
from ml.models.transformer_config import TransformerConfig


class SentriEncoder(nn.Module):
    """Transformer encoder stack.

    Args:
        config: :class:`TransformerConfig` instance.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        # ── Token embedding ──────────────────────────────────────────────────
        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
        )

        # ── Positional encoding ──────────────────────────────────────────────
        self.pos_encoding = PositionalEncoding(
            d_model=config.d_model,
            max_len=config.max_src_len + 2,   # small safety margin
            dropout=config.dropout,
        )

        # ── Encoder layers ───────────────────────────────────────────────────
        # Each layer contains:
        #   1. Multi-head self-attention
        #   2. Add & Norm
        #   3. Position-wise feed-forward network (two linear layers + ReLU)
        #   4. Add & Norm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,   # (batch, seq, feature) layout throughout
            norm_first=False,   # Post-LN (original Vaswani convention)
        )
        self.encoder_stack = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_encoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )

        # Weight initialisation (Xavier uniform on linear, normal on embeddings)
        self._init_weights()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=self.config.d_model ** -0.5)
        if self.token_embedding.padding_idx is not None:
            nn.init.zeros_(self.token_embedding.weight[self.token_embedding.padding_idx])

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,         # (batch, src_len)
        attention_mask: torch.Tensor,    # (batch, src_len)  1=real token, 0=pad
    ) -> torch.Tensor:
        """
        Encode a batch of tokenised complaints.

        Args:
            input_ids:      Integer token ids, shape (B, S).
            attention_mask: Padding mask from the tokenizer, shape (B, S).
                            1 = attend, 0 = masked (pad).

        Returns:
            Encoder hidden states, shape (B, S, d_model).
        """
        # Convert HuggingFace attention_mask convention (1=attend, 0=pad) to
        # PyTorch TransformerEncoder src_key_padding_mask convention (True=ignore).
        src_key_padding_mask = attention_mask == 0   # (B, S)  True where padded

        # Embedding + positional encoding, scaled by sqrt(d_model)
        scale = self.config.d_model ** 0.5
        x = self.token_embedding(input_ids) * scale  # (B, S, D)
        x = self.pos_encoding(x)                      # (B, S, D)

        # Encoder stack (multi-head self-attention × N)
        encoder_output = self.encoder_stack(
            src=x,
            src_key_padding_mask=src_key_padding_mask,
        )  # (B, S, D)

        return encoder_output

    def masked_mean_pool(
        self,
        encoder_output: torch.Tensor,   # (B, S, D)
        attention_mask: torch.Tensor,   # (B, S)
    ) -> torch.Tensor:
        """Masked mean pooling over non-padding positions.

        Returns a single vector per sequence: (B, D).
        Used as input to classification heads.
        """
        # attention_mask: 1=real token, 0=pad  →  expand to (B, S, D)
        mask = attention_mask.unsqueeze(-1).float()       # (B, S, 1)
        summed = (encoder_output * mask).sum(dim=1)       # (B, D)
        lengths = mask.sum(dim=1).clamp(min=1e-9)         # (B, 1)
        return summed / lengths                            # (B, D)

"""
SentriDecoder
-------------
Custom autoregressive Transformer decoder built from `torch.nn` primitives.

Each decoder layer contains:
  1. Masked multi-head causal self-attention   (prevents looking at future tokens)
  2. Add & LayerNorm
  3. Encoder–decoder cross-attention           (attends to encoder hidden states)
  4. Add & LayerNorm
  5. Position-wise feed-forward network
  6. Add & LayerNorm

A linear projection layer at the top maps hidden states → vocabulary logits.

Training:  teacher-forcing — target tokens are shifted right and fed as
           decoder input; only non-padding positions contribute to loss.

Inference: autoregressive — generate one token at a time, appending to the
           running sequence until EOS is produced or max_tgt_len is reached.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ml.models.positional_encoding import PositionalEncoding
from ml.models.transformer_config import TransformerConfig


class SentriDecoder(nn.Module):
    """Transformer decoder stack with a vocabulary projection head.

    Args:
        config: :class:`TransformerConfig` instance.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        # ── Token embedding (decoder side) ────────────────────────────────────
        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
        )

        # ── Positional encoding ───────────────────────────────────────────────
        self.pos_encoding = PositionalEncoding(
            d_model=config.d_model,
            max_len=config.max_tgt_len + 2,
            dropout=config.dropout,
        )

        # ── Decoder layers ────────────────────────────────────────────────────
        # torch.nn.TransformerDecoderLayer provides:
        #   - self-attention  (causal mask supplied in forward)
        #   - cross-attention (memory = encoder output)
        #   - FFN, Add & Norm at each sub-layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder_stack = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=config.num_decoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )

        # ── Vocabulary projection ─────────────────────────────────────────────
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share the token embedding matrix with the projection
        # layer so that embeddings and logits stay in the same space.
        self.output_projection.weight = self.token_embedding.weight

        self._init_weights()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=self.config.d_model ** -0.5)
        if self.token_embedding.padding_idx is not None:
            nn.init.zeros_(self.token_embedding.weight[self.token_embedding.padding_idx])

    # ── Causal mask helper ────────────────────────────────────────────────────

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular boolean mask that prevents each position from
        attending to later positions.

        PyTorch's convention: True = masked (ignored), False = attend.

        Shape: (seq_len, seq_len)
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    # ── Teacher-forced forward (training / fine-tuning) ───────────────────────

    def forward(
        self,
        tgt_ids: torch.Tensor,            # (B, T) — right-shifted target tokens
        encoder_output: torch.Tensor,     # (B, S, D)
        src_key_padding_mask: torch.Tensor,  # (B, S)  True=pad (PyTorch convention)
        tgt_key_padding_mask: Optional[torch.Tensor] = None,  # (B, T) True=pad
    ) -> torch.Tensor:
        """
        Teacher-forced decoder forward pass.

        Args:
            tgt_ids:                Right-shifted target token IDs, (B, T).
            encoder_output:         Encoder hidden states, (B, S, D).
            src_key_padding_mask:   True where encoder positions are padding.
            tgt_key_padding_mask:   True where decoder positions are padding.

        Returns:
            Logits over vocabulary, shape (B, T, vocab_size).
        """
        T = tgt_ids.size(1)
        device = tgt_ids.device

        # Causal self-attention mask (prevent future look-ahead)
        causal_mask = self._causal_mask(T, device)  # (T, T)

        # Embedding + positional encoding
        scale = self.config.d_model ** 0.5
        x = self.token_embedding(tgt_ids) * scale   # (B, T, D)
        x = self.pos_encoding(x)                     # (B, T, D)

        # Decoder stack:
        #   self-attention uses causal_mask (tgt_mask)
        #   cross-attention uses src_key_padding_mask (memory_key_padding_mask)
        hidden = self.decoder_stack(
            tgt=x,
            memory=encoder_output,
            tgt_mask=causal_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # (B, T, D)

        logits = self.output_projection(hidden)  # (B, T, vocab_size)
        return logits

    # ── Autoregressive generation (inference) ─────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        encoder_output: torch.Tensor,       # (B, S, D)
        src_key_padding_mask: torch.Tensor,  # (B, S)  True=pad
        bos_token_id: int,
        eos_token_id: int,
        max_new_tokens: int,
        pad_token_id: int,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding.

        Generates one token at a time.  Stops when every sequence in the
        batch has produced EOS or ``max_new_tokens`` is reached.

        Args:
            encoder_output:         (B, S, D) from the encoder.
            src_key_padding_mask:   (B, S) padding mask for cross-attention.
            bos_token_id:           Token id to start generation with.
            eos_token_id:           Token id that signals end of sequence.
            max_new_tokens:         Hard limit on generated length.
            pad_token_id:           Filled in for sequences that finished early.

        Returns:
            Generated token ids, shape (B, max_new_tokens).
        """
        B = encoder_output.size(0)
        device = encoder_output.device

        # Running token sequence — starts with BOS
        generated = torch.full((B, 1), fill_value=bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            T = generated.size(1)
            causal_mask = self._causal_mask(T, device)  # (T, T)

            scale = self.config.d_model ** 0.5
            x = self.token_embedding(generated) * scale
            x = self.pos_encoding(x)

            hidden = self.decoder_stack(
                tgt=x,
                memory=encoder_output,
                tgt_mask=causal_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )  # (B, T, D)

            # Greedy: take argmax over vocab at last position
            next_logits = self.output_projection(hidden[:, -1, :])  # (B, vocab_size)
            next_token = next_logits.argmax(dim=-1, keepdim=True)    # (B, 1)

            # Replace tokens in already-finished sequences with PAD
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.tensor(pad_token_id, device=device),
                next_token,
            )

            generated = torch.cat([generated, next_token], dim=1)  # (B, T+1)

            # Mark sequences that just produced EOS
            finished = finished | (next_token.squeeze(1) == eos_token_id)
            if finished.all():
                break

        # Return only the new tokens (strip leading BOS)
        return generated[:, 1:]

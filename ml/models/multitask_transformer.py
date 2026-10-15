"""
SentriTransformer — Multi-task Encoder–Decoder Transformer
-----------------------------------------------------------
Full architecture:

                      ┌────────────────────────┐
  complaint text ───► │   SentriEncoder         │ ──► encoder_output (B, S, D)
                      └────────────────────────┘
                                │
              ┌─────────────────┼──────────────────────┐
              │                 │                       │
    ┌─────────▼──────┐  ┌───────▼──────┐   ┌──────────▼──────────┐
    │ ClassifHeads   │  │ ClassifHeads │   │   SentriDecoder      │
    │ sentiment (3)  │  │ emotion (7)  │   │  (cross-attention)   │
    │ category (7)   │  │ issue (6)    │   │  root_cause / resp.  │
    └────────────────┘  └──────────────┘   └─────────────────────┘

Training objective:
    total_loss = w_sent * CE_sent
               + w_emot * CE_emot
               + w_cat  * CE_cat
               + w_iss  * CE_iss
               + w_gen  * CE_gen   (decoder token-level, ignores PAD)

Only tasks that have non-null labels in the batch contribute to the loss.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.encoder import SentriEncoder
from ml.models.decoder import SentriDecoder
from ml.models.transformer_config import TransformerConfig


class SentriTransformer(nn.Module):
    """Encoder–decoder Transformer with multi-task classification heads.

    Args:
        config: :class:`TransformerConfig` instance.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        # ── Encoder ───────────────────────────────────────────────────────────
        self.encoder = SentriEncoder(config)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.decoder = SentriDecoder(config)

        # ── Classification heads (pooled encoder → logits) ────────────────────
        self.sentiment_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.num_sentiment_classes),
        )
        self.emotion_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.num_emotion_classes),
        )
        self.category_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.num_category_classes),
        )
        self.issue_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.num_issue_classes),
        )

        self._init_heads()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_heads(self) -> None:
        for module in [self.sentiment_head, self.emotion_head,
                       self.category_head, self.issue_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    # ── Forward (training / fine-tuning) ──────────────────────────────────────

    def forward(
        self,
        # Encoder inputs
        src_input_ids: torch.Tensor,       # (B, S)
        src_attention_mask: torch.Tensor,  # (B, S)
        # Decoder inputs (teacher-forced; None → skip generation loss)
        tgt_input_ids: Optional[torch.Tensor] = None,   # (B, T)  right-shifted
        tgt_attention_mask: Optional[torch.Tensor] = None,  # (B, T)
        tgt_labels: Optional[torch.Tensor] = None,       # (B, T)  token-level targets
        # Classification targets (−1 = label missing, skip that task)
        sentiment_labels: Optional[torch.Tensor] = None,  # (B,)
        emotion_labels: Optional[torch.Tensor] = None,    # (B,)
        category_labels: Optional[torch.Tensor] = None,   # (B,)
        issue_labels: Optional[torch.Tensor] = None,      # (B,)
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with logits and loss components.

        Keys always present: ``sentiment_logits``, ``emotion_logits``,
        ``category_logits``, ``issue_logits``.

        Keys present when labels are given: ``total_loss``,
        ``sentiment_loss``, ``emotion_loss``, ``category_loss``,
        ``issue_loss``, and optionally ``generation_loss``.
        """
        # 1. Encode the complaint
        encoder_output = self.encoder(src_input_ids, src_attention_mask)  # (B, S, D)

        # 2. Masked mean pooling → classification representation
        pooled = self.encoder.masked_mean_pool(encoder_output, src_attention_mask)  # (B, D)

        # 3. Classification logits
        sentiment_logits = self.sentiment_head(pooled)  # (B, num_sent)
        emotion_logits = self.emotion_head(pooled)       # (B, num_emot)
        category_logits = self.category_head(pooled)     # (B, num_cat)
        issue_logits = self.issue_head(pooled)            # (B, num_iss)

        result: Dict[str, torch.Tensor] = {
            "sentiment_logits": sentiment_logits,
            "emotion_logits": emotion_logits,
            "category_logits": category_logits,
            "issue_logits": issue_logits,
        }

        # 4. Decoder (teacher-forced generation)
        if tgt_input_ids is not None:
            src_pad_mask = src_attention_mask == 0        # (B, S)  True=pad
            tgt_pad_mask = (
                (tgt_attention_mask == 0) if tgt_attention_mask is not None else None
            )
            gen_logits = self.decoder(
                tgt_ids=tgt_input_ids,
                encoder_output=encoder_output,
                src_key_padding_mask=src_pad_mask,
                tgt_key_padding_mask=tgt_pad_mask,
            )  # (B, T, vocab_size)
            result["generation_logits"] = gen_logits

        # 5. Loss computation
        any_label = any(
            lbl is not None
            for lbl in [sentiment_labels, emotion_labels, category_labels, issue_labels]
        )
        has_gen_target = tgt_labels is not None and "generation_logits" in result

        if any_label or has_gen_target:
            total_loss = torch.tensor(0.0, device=src_input_ids.device, requires_grad=True)
            cfg = self.config

            def _cls_loss(logits: torch.Tensor, labels: torch.Tensor, weight: float):
                """Cross-entropy, skipping samples with label == -1."""
                valid = labels >= 0
                if not valid.any():
                    return None
                return weight * F.cross_entropy(logits[valid], labels[valid])

            for logits, labels, weight, key in [
                (sentiment_logits, sentiment_labels, cfg.sentiment_loss_weight, "sentiment_loss"),
                (emotion_logits,   emotion_labels,   cfg.emotion_loss_weight,   "emotion_loss"),
                (category_logits,  category_labels,  cfg.category_loss_weight,  "category_loss"),
                (issue_logits,     issue_labels,     cfg.issue_loss_weight,     "issue_loss"),
            ]:
                if labels is not None:
                    loss = _cls_loss(logits, labels, weight)
                    if loss is not None:
                        result[key] = loss
                        total_loss = total_loss + loss

            if has_gen_target:
                gen_logits = result["generation_logits"]   # (B, T, V)
                B, T, V = gen_logits.shape
                # Flatten and compute CE ignoring pad positions (label=-100)
                gen_loss = F.cross_entropy(
                    gen_logits.reshape(B * T, V),
                    tgt_labels.reshape(B * T),
                    ignore_index=-100,
                )
                gen_loss = cfg.generation_loss_weight * gen_loss
                result["generation_loss"] = gen_loss
                total_loss = total_loss + gen_loss

            result["total_loss"] = total_loss

        return result

    # ── Inference helpers ─────────────────────────────────────────────────────

    @torch.no_grad()
    def classify(
        self,
        src_input_ids: torch.Tensor,
        src_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run only encoder + classification heads (no generation).

        Returns logits and softmax probabilities for all tasks.
        """
        self.eval()
        encoder_output = self.encoder(src_input_ids, src_attention_mask)
        pooled = self.encoder.masked_mean_pool(encoder_output, src_attention_mask)

        return {
            "sentiment_logits": self.sentiment_head(pooled),
            "emotion_logits":   self.emotion_head(pooled),
            "category_logits":  self.category_head(pooled),
            "issue_logits":     self.issue_head(pooled),
            # Probabilities
            "sentiment_probs":  F.softmax(self.sentiment_head(pooled), dim=-1),
            "emotion_probs":    F.softmax(self.emotion_head(pooled), dim=-1),
            "category_probs":   F.softmax(self.category_head(pooled), dim=-1),
            "issue_probs":      F.softmax(self.issue_head(pooled), dim=-1),
            "encoder_output":   encoder_output,
        }

    @torch.no_grad()
    def generate_text(
        self,
        src_input_ids: torch.Tensor,
        src_attention_mask: torch.Tensor,
        max_new_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """Greedy-decode a text output conditioned on the encoder.

        Returns generated token ids (B, max_new_tokens).
        """
        self.eval()
        cfg = self.config
        max_new_tokens = max_new_tokens or cfg.max_tgt_len
        encoder_output = self.encoder(src_input_ids, src_attention_mask)
        src_pad_mask = src_attention_mask == 0

        return self.decoder.generate(
            encoder_output=encoder_output,
            src_key_padding_mask=src_pad_mask,
            bos_token_id=cfg.bos_token_id,
            eos_token_id=cfg.eos_token_id,
            max_new_tokens=max_new_tokens,
            pad_token_id=cfg.pad_token_id,
        )

    # ── Parameter counting ────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

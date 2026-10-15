"""
SentriCollator — dynamic padding collate function
--------------------------------------------------
Handles variable-length sequences within a batch:

- Source sequences: padded to the longest in the batch.
- Target sequences: padded to the longest in the batch.
  tgt_labels padding positions are set to -100 (ignored by CE loss).

This avoids allocating memory for fixed maximum lengths on every batch.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence


class SentriCollator:
    """Callable collator for use as ``collate_fn`` in a DataLoader.

    Args:
        pad_id:    Token id to use when padding source / target sequences.
        label_pad: Padding value for tgt_labels (-100 → ignored by CE loss).
    """

    def __init__(self, pad_id: int = 0, label_pad: int = -100) -> None:
        self.pad_id = pad_id
        self.label_pad = label_pad

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: List of feature dicts from ComplaintDataset.__getitem__.

        Returns:
            Dict with tensors ready for the model forward pass.
        """
        # ── Source sequences ─────────────────────────────────────────────────
        src_ids   = [torch.tensor(item["src_input_ids"],      dtype=torch.long) for item in batch]
        src_masks = [torch.tensor(item["src_attention_mask"], dtype=torch.long) for item in batch]

        src_input_ids      = self._pad(src_ids,   self.pad_id)       # (B, S)
        src_attention_mask = self._pad(src_masks, 0)                 # (B, S)

        # ── Classification labels ─────────────────────────────────────────────
        sentiment_labels = torch.tensor([item["sentiment_label"] for item in batch], dtype=torch.long)
        emotion_labels   = torch.tensor([item["emotion_label"]   for item in batch], dtype=torch.long)
        category_labels  = torch.tensor([item["category_label"]  for item in batch], dtype=torch.long)
        issue_labels     = torch.tensor([item["issue_label"]     for item in batch], dtype=torch.long)

        # ── Target (generation) sequences ─────────────────────────────────────
        has_tgt = any(item["tgt_input_ids"] for item in batch)

        tgt_input_ids: Optional[torch.Tensor] = None
        tgt_attention_mask: Optional[torch.Tensor] = None
        tgt_labels: Optional[torch.Tensor] = None

        if has_tgt:
            raw_tgt_ids   = []
            raw_tgt_masks = []
            raw_tgt_lbls  = []

            for item in batch:
                ids = item["tgt_input_ids"]
                if ids:
                    raw_tgt_ids.append(torch.tensor(ids, dtype=torch.long))
                    raw_tgt_masks.append(torch.tensor(item["tgt_attention_mask"], dtype=torch.long))
                    raw_tgt_lbls.append(torch.tensor(item["tgt_labels"], dtype=torch.long))
                else:
                    # Sample has no generation target → pad with a single PAD token
                    raw_tgt_ids.append(torch.tensor([self.pad_id], dtype=torch.long))
                    raw_tgt_masks.append(torch.tensor([0], dtype=torch.long))
                    raw_tgt_lbls.append(torch.tensor([self.label_pad], dtype=torch.long))

            tgt_input_ids      = self._pad(raw_tgt_ids,   self.pad_id)      # (B, T)
            tgt_attention_mask = self._pad(raw_tgt_masks, 0)                # (B, T)
            # Labels: pad with -100 so CE loss ignores those positions
            tgt_labels = self._pad(raw_tgt_lbls, self.label_pad)            # (B, T)

        out: Dict[str, torch.Tensor] = {
            "src_input_ids":      src_input_ids,
            "src_attention_mask": src_attention_mask,
            "sentiment_labels":   sentiment_labels,
            "emotion_labels":     emotion_labels,
            "category_labels":    category_labels,
            "issue_labels":       issue_labels,
        }
        if tgt_input_ids is not None:
            out["tgt_input_ids"]      = tgt_input_ids
            out["tgt_attention_mask"] = tgt_attention_mask
            out["tgt_labels"]         = tgt_labels

        return out

    @staticmethod
    def _pad(sequences: List[torch.Tensor], pad_value: int) -> torch.Tensor:
        """Right-pad a list of 1-D tensors to the same length."""
        return pad_sequence(sequences, batch_first=True, padding_value=pad_value)

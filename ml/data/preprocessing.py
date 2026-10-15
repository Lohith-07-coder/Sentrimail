"""
Preprocessing pipeline
-----------------------
Converts raw :class:`~ml.dataset.schema.UnifiedSample` objects into
tokenised tensors ready for the custom Transformer.

Key responsibilities:
- Load a HuggingFace tokenizer (token IDs only; no pretrained weights used
  inside the neural network layers themselves).
- Convert string labels to integer indices using the config label maps.
- Produce train / validation / test splits with a fixed random seed.
- Fit nothing on validation or test data (no leakage).

Label encoding:
- Missing labels (empty string / None) are encoded as -1.
  The training loop skips -1-labelled samples for that task.
- For generation targets (root_cause, response), missing text results in
  tgt_labels filled with -100 so nn.CrossEntropyLoss ignores those tokens.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ml.dataset.schema import UnifiedSample
from ml.models.transformer_config import TransformerConfig

logger = logging.getLogger(__name__)

# ── Lazy tokenizer cache ───────────────────────────────────────────────────────
_tokenizer_cache: Dict[str, object] = {}


def get_tokenizer(tokenizer_name: str):
    """Load and cache a HuggingFace tokenizer by name."""
    global _tokenizer_cache
    if tokenizer_name not in _tokenizer_cache:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_name)
            _tokenizer_cache[tokenizer_name] = tok
            logger.info("Tokenizer loaded: %s  vocab_size=%d", tokenizer_name, tok.vocab_size)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load tokenizer '{tokenizer_name}'. "
                "Install transformers and ensure internet access for first download."
            ) from exc
    return _tokenizer_cache[tokenizer_name]


# ── Label → int helpers ────────────────────────────────────────────────────────

def encode_label(value: Optional[str], inv_map: Dict[str, int]) -> int:
    """Return class index or -1 when the label is missing / unknown."""
    if not value:
        return -1
    v = value.strip()
    return inv_map.get(v, inv_map.get(v.lower(), -1))


def tokenize_text(
    text: str,
    tokenizer,
    max_length: int,
) -> Dict[str, List[int]]:
    """Tokenise a single text and return input_ids + attention_mask."""
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,         # dynamic padding done in collator
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def encode_generation_target(
    text: str,
    tokenizer,
    max_length: int,
    bos_id: int,
    eos_id: int,
    pad_id: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Encode a generation target into decoder input and label sequences.

    Returns:
        tgt_input_ids:    [BOS] + tokens           (right-shifted decoder input)
        tgt_attention_mask: 1 for real, 0 for pad
        tgt_labels:       tokens + [EOS], -100 for pad (for CE loss)

    If ``text`` is empty, returns empty lists (caller should skip gen loss).
    """
    if not text.strip():
        return [], [], []

    encoded = tokenizer(
        text,
        max_length=max_length - 1,   # reserve 1 slot for BOS / EOS
        truncation=True,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    token_ids = encoded["input_ids"]

    tgt_input_ids = [bos_id] + token_ids                     # [BOS, t1, t2, ...]
    tgt_labels = token_ids + [eos_id]                         # [t1, t2, ..., EOS]
    tgt_attention_mask = [1] * len(tgt_input_ids)

    return tgt_input_ids, tgt_attention_mask, tgt_labels


# ── Train / val / test split ───────────────────────────────────────────────────

def split_samples(
    samples: List[UnifiedSample],
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> Tuple[List[UnifiedSample], List[UnifiedSample], List[UnifiedSample]]:
    """Reproducible stratified split (stratified by category).

    Returns (train, val, test) lists.
    """
    random.seed(seed)
    shuffled = list(samples)
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = shuffled[:n_train]
    val   = shuffled[n_train: n_train + n_val]
    test  = shuffled[n_train + n_val:]

    logger.info(
        "Split: %d train / %d val / %d test  (seed=%d)",
        len(train), len(val), len(test), seed,
    )
    return train, val, test


# ── Full preprocessing pipeline ────────────────────────────────────────────────

def preprocess_samples(
    samples: List[UnifiedSample],
    config: TransformerConfig,
) -> List[Dict]:
    """Convert UnifiedSample list into tokenised feature dicts.

    Each returned dict contains:
        src_input_ids, src_attention_mask,
        sentiment_label, emotion_label, category_label, issue_label,
        tgt_input_ids, tgt_attention_mask, tgt_labels  (may be empty lists)

    Missing classification labels are encoded as -1.
    Missing generation labels produce empty tgt_* lists.
    """
    tokenizer = get_tokenizer(config.tokenizer_name)

    inv_sent = config.inv_sentiment_map
    inv_emot = config.inv_emotion_map
    inv_cat  = config.inv_category_map
    inv_iss  = config.inv_issue_map

    processed = []
    skipped = 0

    for sample in samples:
        if not sample.text.strip():
            skipped += 1
            continue

        # ── Source encoding ────────────────────────────────────────────────
        src = tokenize_text(sample.text, tokenizer, config.max_src_len)

        # ── Classification labels ──────────────────────────────────────────
        sent_lbl  = encode_label(sample.sentiment, inv_sent)
        emot_lbl  = encode_label(sample.emotion,   inv_emot)
        cat_lbl   = encode_label(sample.category,  inv_cat)
        iss_lbl   = encode_label(sample.issue,     inv_iss)

        # ── Generation targets (root_cause preferred; fall back to response)
        gen_text = (sample.root_cause or sample.response or "").strip()
        tgt_inp, tgt_mask, tgt_lbl = encode_generation_target(
            text=gen_text,
            tokenizer=tokenizer,
            max_length=config.max_tgt_len,
            bos_id=config.bos_token_id,
            eos_id=config.eos_token_id,
            pad_id=config.pad_token_id,
        )

        processed.append({
            "id": sample.id,
            "src_input_ids":      src["input_ids"],
            "src_attention_mask": src["attention_mask"],
            "sentiment_label":    sent_lbl,
            "emotion_label":      emot_lbl,
            "category_label":     cat_lbl,
            "issue_label":        iss_lbl,
            "tgt_input_ids":      tgt_inp,
            "tgt_attention_mask": tgt_mask,
            "tgt_labels":         tgt_lbl,
        })

    if skipped:
        logger.warning("Skipped %d empty samples during preprocessing.", skipped)
    logger.info("Preprocessed %d samples.", len(processed))
    return processed

"""
SentriMail Dataset Preprocessing Pipeline
-----------------------------------------
Cleans text, masks PII (emails, phones, credit cards, SSNs), deduplicates records,
and creates a stratified 80/10/10 train/val/test split.
"""

import json
import re
import logging
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

from ml.dataset.schema import UnifiedSample

logger = logging.getLogger(__name__)

# PII Regex Patterns
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def clean_and_mask_pii(text: str) -> str:
    """Removes control characters, normalizes whitespace, and masks sensitive PII."""
    if not text:
        return ""
    
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", text).strip()
    
    # Mask PII
    cleaned = EMAIL_PATTERN.sub("[EMAIL_MASKED]", cleaned)
    cleaned = PHONE_PATTERN.sub("[PHONE_MASKED]", cleaned)
    cleaned = CREDIT_CARD_PATTERN.sub("[CARD_MASKED]", cleaned)
    cleaned = SSN_PATTERN.sub("[SSN_MASKED]", cleaned)
    
    return cleaned


def deduplicate_samples(samples: List[UnifiedSample]) -> List[UnifiedSample]:
    """Removes duplicate text entries keeping the first occurrence."""
    seen_texts = set()
    deduped: List[UnifiedSample] = []
    
    for s in samples:
        norm_text = clean_and_mask_pii(s.text).lower()
        if norm_text not in seen_texts and len(norm_text) > 5:
            seen_texts.add(norm_text)
            s.text = clean_and_mask_pii(s.text)
            deduped.append(s)
            
    logger.info("Deduplication: %d -> %d samples", len(samples), len(deduped))
    return deduped


def create_stratified_split(
    samples: List[UnifiedSample],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42
) -> Tuple[List[UnifiedSample], List[UnifiedSample], List[UnifiedSample]]:
    """
    Creates stratified 80/10/10 train/val/test splits based on composite target tags.
    """
    random.seed(seed)
    
    # Group by category / message_type
    buckets: Dict[str, List[UnifiedSample]] = {}
    for s in samples:
        key = f"{s.message_type}_{s.category}"
        buckets.setdefault(key, []).append(s)
        
    train_set, val_set, test_set = [], [], []
    
    for key, items in buckets.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_set.extend(items[:n_train])
        val_set.extend(items[n_train:n_train + n_val])
        test_set.extend(items[n_train + n_val:])
        
    logger.info("Stratified Split Complete: Train=%d, Val=%d, Test=%d", len(train_set), len(val_set), len(test_set))
    return train_set, val_set, test_set


def preprocess_pipeline(raw_samples: List[UnifiedSample], output_dir: Path) -> Tuple[Path, Path, Path]:
    """Runs end-to-end cleaning, PII masking, deduping, splitting, and JSON saving."""
    deduped = deduplicate_samples(raw_samples)
    train, val, test = create_stratified_split(deduped)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.json"
    val_path = output_dir / "val.json"
    test_path = output_dir / "test.json"
    
    train_path.write_text(json.dumps([s.to_dict() for s in train], indent=2), encoding="utf-8")
    val_path.write_text(json.dumps([s.to_dict() for s in val], indent=2), encoding="utf-8")
    test_path.write_text(json.dumps([s.to_dict() for s in test], indent=2), encoding="utf-8")
    
    logger.info("Saved processed splits to %s", output_dir)
    return train_path, val_path, test_path

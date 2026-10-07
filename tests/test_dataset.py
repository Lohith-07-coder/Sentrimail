import pytest
from pathlib import Path
import pandas as pd
from ml.dataset.loaders import (
    load_cfpb_complaints,
    load_bitext_support_dataset,
    load_banking77_dataset,
    load_goemotions_dataset,
    load_sms_jigsaw_dataset,
    load_massive_dataset
)
from ml.dataset.preprocess import (
    clean_and_mask_pii,
    deduplicate_samples,
    create_stratified_split,
    preprocess_pipeline
)
from ml.dataset.schema import UnifiedSample

def test_load_cfpb_validation(tmp_path: Path):
    df = pd.DataFrame({"Wrong Column": ["text1", "text2"]})
    df.to_csv(tmp_path / "cfpb_complaints.csv", index=False)
    samples = load_cfpb_complaints(tmp_path)
    assert len(samples) == 0

    df_correct = pd.DataFrame({"Consumer complaint narrative": ["text1", "text2"]})
    df_correct.to_csv(tmp_path / "cfpb_complaints.csv", index=False)
    samples_correct = load_cfpb_complaints(tmp_path)
    assert len(samples_correct) == 2

def test_load_massive_validation(tmp_path: Path):
    df = pd.DataFrame({"Wrong Column": ["text1", "text2"]})
    df.to_csv(tmp_path / "massive_multilingual.csv", index=False)
    samples = load_massive_dataset(tmp_path)
    assert len(samples) == 0

    df_correct = pd.DataFrame({"utt": ["text1", "text2"], "locale": ["en", "fr"]})
    df_correct.to_csv(tmp_path / "massive_multilingual.csv", index=False)
    samples_correct = load_massive_dataset(tmp_path)
    assert len(samples_correct) == 2

def test_clean_and_mask_pii():
    text = "Contact me at name@example.com or 555-123-4567. My card is 1234-5678-9012-3456 and SSN is 123-45-6789."
    cleaned = clean_and_mask_pii(text)
    assert "name@example.com" not in cleaned
    assert "555-123-4567" not in cleaned
    assert "1234-5678-9012-3456" not in cleaned
    assert "123-45-6789" not in cleaned
    assert "[EMAIL_MASKED]" in cleaned
    assert "[PHONE_MASKED]" in cleaned
    assert "[CARD_MASKED]" in cleaned
    assert "[SSN_MASKED]" in cleaned

def test_deduplicate_samples():
    samples = [
        UnifiedSample(id="1", text="Duplicate text", language="en", message_type="query"),
        UnifiedSample(id="2", text="Duplicate text", language="en", message_type="query"),
        UnifiedSample(id="3", text="Unique text", language="en", message_type="query"),
    ]
    deduped = deduplicate_samples(samples)
    assert len(deduped) == 2
    assert {s.id for s in deduped} == {"1", "3"}

def test_create_stratified_split():
    samples = []
    for i in range(100):
        samples.append(UnifiedSample(
            id=str(i),
            text=f"Text {i}",
            language="en",
            message_type="query" if i < 50 else "complaint",
            category="billing" if i % 2 == 0 else "technical"
        ))
    
    train, val, test = create_stratified_split(samples, 0.8, 0.1, 0.1, seed=42)
    assert len(train) == 80
    assert len(val) == 8
    assert len(test) == 12
    
    # Check no leakage
    train_ids = {s.id for s in train}
    val_ids = {s.id for s in val}
    test_ids = {s.id for s in test}
    
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0

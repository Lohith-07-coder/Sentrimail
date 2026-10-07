"""
SentriMail Dataset Loaders Module
---------------------------------
Modular data loaders for multi-task model training.
Supports:
- Local Kaggle Complaint-Response dataset (with dynamic column discovery)
- CFPB Financial Complaints
- Bitext Customer Support
- Banking77 Intent Classification
- GoEmotions Multiclass Sentiment/Emotion
- SMS Spam & Jigsaw Toxic Comment datasets
- MASSIVE Multilingual Intent dataset
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from ml.dataset.schema import UnifiedSample

logger = logging.getLogger(__name__)


def load_kaggle_complaint_dataset(data_dir: Path) -> List[UnifiedSample]:
    """
    Inspects and loads Kaggle customer complaint-response dataset files (.csv or .json)
    with dynamic column detection.
    """
    samples: List[UnifiedSample] = []
    possible_paths = list(data_dir.glob("*kaggle*")) + list(data_dir.glob("*.csv")) + list(data_dir.glob("*.json"))
    
    for path in possible_paths:
        if path.name.endswith(".csv"):
            try:
                df = pd.read_csv(path)
                logger.info("Inspecting Kaggle CSV columns in %s: %s", path.name, list(df.columns))
                
                # Column resolution heuristic
                text_col = next((c for c in df.columns if c.lower() in ["text", "complaint", "description", "issue_description", "narrative"]), None)
                resp_col = next((c for c in df.columns if c.lower() in ["response", "reply", "resolution", "admin_response", "answer"]), None)
                cat_col = next((c for c in df.columns if c.lower() in ["category", "dept", "department", "topic"]), None)
                sent_col = next((c for c in df.columns if c.lower() in ["sentiment", "sentiment_label"]), None)
                
                if not text_col:
                    continue

                for idx, row in df.iterrows():
                    txt = str(row[text_col]).strip() if pd.notna(row[text_col]) else ""
                    if not txt:
                        continue
                    
                    resp = str(row[resp_col]).strip() if resp_col and pd.notna(row[resp_col]) else ""
                    cat = str(row[cat_col]).lower().strip() if cat_col and pd.notna(row[cat_col]) else "other"
                    sent = str(row[sent_col]).upper().strip() if sent_col and pd.notna(row[sent_col]) else "NEUTRAL"
                    
                    samples.append(UnifiedSample(
                        id=f"kaggle_{path.stem}_{idx}",
                        text=txt,
                        language="en",
                        message_type="complaint",
                        category=cat if cat in ["billing", "technical", "delivery", "customer_service", "product", "refund"] else "other",
                        issue="general",
                        sentiment=sent if sent in ["POSITIVE", "NEUTRAL", "NEGATIVE"] else "NEUTRAL",
                        emotion="neutral",
                        response=resp
                    ))
            except Exception as e:
                logger.warning("Error loading Kaggle CSV %s: %s", path, e)
                
    return samples


def load_cfpb_complaints(data_dir: Path) -> List[UnifiedSample]:
    """Load CFPB Consumer Financial Complaints data if present."""
    samples: List[UnifiedSample] = []
    cfpb_file = data_dir / "cfpb_complaints.csv"
    if not cfpb_file.exists():
        return samples

    try:
        df = pd.read_csv(cfpb_file)
        for idx, row in df.iterrows():
            narrative = str(row.get("Consumer complaint narrative", "")).strip()
            if not narrative or narrative == "nan":
                continue
            product = str(row.get("Product", "")).lower()
            cat = "billing" if "credit" in product or "bank" in product or "mortgage" in product else "other"
            samples.append(UnifiedSample(
                id=f"cfpb_{idx}",
                text=narrative,
                language="en",
                message_type="complaint",
                category=cat,
                issue="billing" if cat == "billing" else "general",
                sentiment="NEGATIVE",
                emotion="anger",
            ))
    except Exception as e:
        logger.warning("Failed to load CFPB data: %s", e)
    return samples


def load_bitext_support_dataset(data_dir: Path) -> List[UnifiedSample]:
    """Load Bitext Customer Support Dataset if present."""
    samples: List[UnifiedSample] = []
    bitext_file = data_dir / "bitext_customer_support.csv"
    if not bitext_file.exists():
        return samples

    try:
        df = pd.read_csv(bitext_file)
        for idx, row in df.iterrows():
            uttr = str(row.get("instruction", row.get("utterance", ""))).strip()
            resp = str(row.get("response", "")).strip()
            if not uttr:
                continue
            samples.append(UnifiedSample(
                id=f"bitext_{idx}",
                text=uttr,
                language="en",
                message_type="query",
                category="customer_service",
                issue="support",
                sentiment="NEUTRAL",
                response=resp,
            ))
    except Exception as e:
        logger.warning("Failed to load Bitext dataset: %s", e)
    return samples


def load_banking77_dataset(data_dir: Path) -> List[UnifiedSample]:
    """Load Banking77 Intent Dataset if present."""
    samples: List[UnifiedSample] = []
    b77_file = data_dir / "banking77.csv"
    if not b77_file.exists():
        return samples

    try:
        df = pd.read_csv(b77_file)
        for idx, row in df.iterrows():
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            samples.append(UnifiedSample(
                id=f"b77_{idx}",
                text=text,
                language="en",
                message_type="query",
                category="billing",
                issue="billing",
                sentiment="NEUTRAL",
            ))
    except Exception as e:
        logger.warning("Failed to load Banking77 dataset: %s", e)
    return samples


def load_goemotions_dataset(data_dir: Path) -> List[UnifiedSample]:
    """Load GoEmotions Dataset for fine emotion mapping if present."""
    samples: List[UnifiedSample] = []
    file_path = data_dir / "goemotions.csv"
    if not file_path.exists():
        return samples

    try:
        df = pd.read_csv(file_path)
        emotion_map = {
            "anger": "anger", "fear": "fear", "sadness": "sadness",
            "disgust": "disgust", "surprise": "surprise", "joy": "joy"
        }
        for idx, row in df.iterrows():
            text = str(row.get("text", "")).strip()
            em = str(row.get("emotion", "neutral")).lower()
            if not text:
                continue
            samples.append(UnifiedSample(
                id=f"goemotions_{idx}",
                text=text,
                language="en",
                message_type="complaint" if em in ["anger", "disgust"] else "query",
                emotion=emotion_map.get(em, "neutral"),
                sentiment="NEGATIVE" if em in ["anger", "disgust", "fear", "sadness"] else "NEUTRAL",
            ))
    except Exception as e:
        logger.warning("Failed to load GoEmotions dataset: %s", e)
    return samples


def load_sms_jigsaw_dataset(data_dir: Path) -> List[UnifiedSample]:
    """Load SMS Spam & Jigsaw Toxic Comment Datasets for spam/abuse detection."""
    samples: List[UnifiedSample] = []
    spam_file = data_dir / "spam_abuse.csv"
    if not spam_file.exists():
        return samples

    try:
        df = pd.read_csv(spam_file)
        for idx, row in df.iterrows():
            text = str(row.get("text", "")).strip()
            is_spam = int(row.get("label", 0)) == 1
            if not text:
                continue
            samples.append(UnifiedSample(
                id=f"spam_{idx}",
                text=text,
                language="en",
                message_type="spam_abuse" if is_spam else "query",
                category="other",
                sentiment="NEGATIVE" if is_spam else "NEUTRAL",
            ))
    except Exception as e:
        logger.warning("Failed to load Spam/Abuse dataset: %s", e)
    return samples


def load_massive_dataset(data_dir: Path) -> List[UnifiedSample]:
    """Load Amazon Massive Multilingual Dataset if present."""
    samples: List[UnifiedSample] = []
    file_path = data_dir / "massive_multilingual.json"
    if not file_path.exists():
        return samples

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        for idx, item in enumerate(data):
            text = item.get("utt", "").strip()
            lang = item.get("locale", "en")[:2]
            if not text:
                continue
            samples.append(UnifiedSample(
                id=f"massive_{idx}",
                text=text,
                language=lang,
                message_type="query",
            ))
    except Exception as e:
        logger.warning("Failed to load MASSIVE dataset: %s", e)
    return samples


def load_all_datasets(data_dir: Path, flags: Dict[str, bool]) -> List[UnifiedSample]:
    """
    Consolidates data across configured sources into a single list of UnifiedSample objects.
    """
    all_samples: List[UnifiedSample] = []

    if flags.get("use_kaggle", True):
        kaggle = load_kaggle_complaint_dataset(data_dir)
        logger.info("Loaded %d Kaggle samples", len(kaggle))
        all_samples.extend(kaggle)

    if flags.get("use_cfpb", True):
        cfpb = load_cfpb_complaints(data_dir)
        all_samples.extend(cfpb)

    if flags.get("use_bitext", True):
        bitext = load_bitext_support_dataset(data_dir)
        all_samples.extend(bitext)

    if flags.get("use_banking77", True):
        b77 = load_banking77_dataset(data_dir)
        all_samples.extend(b77)

    if flags.get("use_goemotions", True):
        goem = load_goemotions_dataset(data_dir)
        all_samples.extend(goem)

    if flags.get("use_sms_jigsaw", True):
        spam = load_sms_jigsaw_dataset(data_dir)
        all_samples.extend(spam)

    if flags.get("use_massive", True):
        massive = load_massive_dataset(data_dir)
        all_samples.extend(massive)

    logger.info("Total consolidated multi-task samples: %d", len(all_samples))
    return all_samples

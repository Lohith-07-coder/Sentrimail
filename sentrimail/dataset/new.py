"""
Train a lightweight response model from local dataset files.

Usage:
    python sentrimail/dataset/new.py
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
APP_DATA_DIR = ROOT / "data"
MODEL_PATH = APP_DATA_DIR / "response_model.json"

TEXT_KEYS = [
    "description",
    "complaint",
    "complaint_text",
    "customer_complaint",
    "issue",
    "message",
    "text",
    "input",
    "prompt",
]
RESPONSE_KEYS = [
    "admin_response",
    "response",
    "reply",
    "resolution",
    "target",
    "output",
    "ai_suggested_response",
    "suggested_response",
]

_token_pattern = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> List[str]:
    return [tok for tok in _token_pattern.findall(text.lower()) if len(tok) > 1]


def _get_first_non_empty(row: Dict[str, object], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _load_json_records(path: Path) -> List[Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _load_jsonl_records(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                records.append(row)
    return records


def _load_csv_records(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _sanitize_response_template(response: str) -> str:
    lines = response.strip().splitlines()
    if not lines:
        return response.strip()

    first_line = lines[0].strip()
    if first_line.lower().startswith("dear "):
        lines[0] = "Dear {username},"
    return "\n".join(lines).strip()


def _vectorize(text: str, idf: Dict[str, float]) -> Dict[str, float]:
    tokens = _tokenize(text)
    if not tokens:
        return {}

    tf_counts = Counter(tokens)
    total = float(sum(tf_counts.values()))
    vec: Dict[str, float] = {}
    for token, count in tf_counts.items():
        if token not in idf:
            continue
        vec[token] = (count / total) * idf[token]

    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm > 0:
        vec = {k: v / norm for k, v in vec.items()}
    return vec


def _build_model(records: List[Dict[str, object]]) -> Dict[str, object]:
    docs: List[List[str]] = []
    samples: List[Dict[str, object]] = []
    unique_responses = set()

    for row in records:
        complaint = _get_first_non_empty(row, TEXT_KEYS)
        response = _get_first_non_empty(row, RESPONSE_KEYS)
        if not complaint or not response:
            continue

        tokens = _tokenize(complaint)
        if not tokens:
            continue

        docs.append(tokens)
        cleaned_response = _sanitize_response_template(response)
        unique_responses.add(cleaned_response)
        samples.append(
            {
                "text": complaint,
                "response": cleaned_response,
                "category": str(row.get("category", "")).strip().lower(),
                "priority": str(row.get("priority", "")).strip().upper(),
            }
        )

    if not samples:
        raise ValueError("No valid (complaint, response) pairs were found in dataset files.")

    doc_count = len(docs)
    df_counts = Counter()
    for tokens in docs:
        df_counts.update(set(tokens))

    idf = {
        token: math.log((1.0 + doc_count) / (1.0 + df)) + 1.0
        for token, df in df_counts.items()
    }

    for sample in samples:
        sample["vector"] = _vectorize(str(sample["text"]), idf)

    return {
        "model_type": "tfidf-nearest-neighbor",
        "num_samples": len(samples),
        "unique_response_count": len(unique_responses),
        "idf": idf,
        "samples": samples,
    }


def _load_training_records() -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    include_patterns = ("*.csv", "*.json", "*.jsonl")

    for pattern in include_patterns:
        for path in sorted(DATASET_DIR.glob(pattern)):
            if path.name == "response_model.json" or path.name == "new.py":
                continue
            if path.suffix.lower() == ".csv":
                records.extend(_load_csv_records(path))
            elif path.suffix.lower() == ".json":
                records.extend(_load_json_records(path))
            elif path.suffix.lower() == ".jsonl":
                records.extend(_load_jsonl_records(path))

    fallback = APP_DATA_DIR / "complaints.json"
    if fallback.exists():
        records.extend(_load_json_records(fallback))

    return records


def train() -> None:
    records = _load_training_records()
    model = _build_model(records)

    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"Trained model with {model['num_samples']} samples")
    print(f"Saved: {MODEL_PATH}")


if __name__ == "__main__":
    train()

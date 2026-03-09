"""
SentriMail AI Engine
--------------------
Hybrid analysis pipeline:
- Sentiment + emotion (transformer when available, otherwise rule-based)
- Priority scoring
- Auto-resolvable decision for LOW/simple complaints
- Admin suggested response for non-auto cases
"""

import logging
import json
import math
import re
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_sentiment_pipeline = None
_emotion_pipeline = None
_models_loaded = False
_use_transformers = True
_response_model = None
_response_model_loaded = False


NEGATIVE_KEYWORDS = [
    "angry", "furious", "terrible", "horrible", "awful", "disgusting", "unacceptable",
    "worst", "broken", "failed", "fraud", "scam", "useless", "incompetent", "lied",
    "cheated", "stolen", "damaged", "defective", "ruined", "lawsuit", "legal",
    "never", "outraged", "disgusted", "appalled", "ridiculous", "pathetic",
    "disaster", "catastrophic", "urgent", "immediately", "asap", "emergency",
    "error", "issue", "problem", "corrupted", "failing", "failure", "down",
    "unable", "cannot", "can't", "stuck", "crash", "crashing", "data loss",
]

POSITIVE_KEYWORDS = [
    "good", "great", "fine", "okay", "thanks", "appreciate", "help", "resolved",
]

EMOTION_PATTERNS = {
    "anger": ["angry", "furious", "outraged", "mad", "infuriated", "livid", "rage"],
    "fear": ["scared", "afraid", "worried", "anxious", "terrified", "concern", "panic", "risk", "failing", "corrupted", "data loss"],
    "sadness": ["sad", "disappointed", "upset", "devastated", "miserable", "heartbroken"],
    "disgust": ["disgusting", "disgusted", "revolting", "nasty", "appalled", "appalling"],
    "surprise": ["shocked", "unbelievable", "incredible", "unexpected", "sudden"],
    "joy": ["happy", "pleased", "delighted", "glad", "satisfied", "grateful"],
    "neutral": [],
}

CATEGORY_ROOT_CAUSES = {
    "billing": "Likely caused by unexpected charges, a billing mismatch, or unclear invoicing.",
    "technical": "Likely caused by service instability, system defects, or reliability gaps.",
    "delivery": "Likely caused by delivery delays or logistics breakdowns.",
    "customer_service": "Likely caused by support quality issues such as delays or unclear communication.",
    "product": "Likely caused by product quality, mismatch, or performance issues.",
    "refund": "Likely caused by refund delay, policy confusion, or unresolved payment reversal.",
    "other": "Likely caused by cross-functional service/process issues that need deeper triage.",
}

URGENT_TRIGGERS = {
    "urgent", "immediately", "asap", "emergency", "legal action", "lawsuit",
    "police", "critical", "danger", "injury", "health", "fraud", "stolen",
    "server down", "production down", "data loss", "corrupted", "outage",
}

MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "response_model.json"
_token_pattern = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return [tok for tok in _token_pattern.findall(text.lower()) if len(tok) > 1]


def _dot(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def _vectorize_for_inference(text: str, idf: Dict[str, float]) -> Dict[str, float]:
    tokens = _tokenize(text)
    if not tokens:
        return {}

    counts: Dict[str, int] = {}
    for token in tokens:
        if token in idf:
            counts[token] = counts.get(token, 0) + 1

    total = sum(counts.values()) or 1
    vec = {tok: (cnt / total) * idf[tok] for tok, cnt in counts.items()}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm > 0:
        vec = {k: v / norm for k, v in vec.items()}
    return vec


def _load_response_model() -> None:
    global _response_model, _response_model_loaded
    if _response_model_loaded:
        return

    _response_model_loaded = True
    if not MODEL_PATH.exists():
        logger.info("No response model found at %s", MODEL_PATH)
        return

    try:
        _response_model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        logger.info(
            "Response model loaded: %s samples",
            _response_model.get("num_samples", "?"),
        )
    except Exception as exc:
        _response_model = None
        logger.warning("Failed loading response model: %s", exc)


def _predict_response_from_dataset(
    text: str,
    username: str,
    category: str,
    priority: str,
) -> str:
    _load_response_model()
    if not _response_model:
        return ""

    idf = _response_model.get("idf", {})
    samples = _response_model.get("samples", [])
    if not idf or not samples:
        return ""

    query_vec = _vectorize_for_inference(text, idf)
    if not query_vec:
        return ""

    best_score = -1.0
    best_response = ""
    category_lower = (category or "").lower()
    priority_upper = (priority or "").upper()

    for sample in samples:
        sample_vec = sample.get("vector", {})
        if not sample_vec:
            continue

        score = _dot(query_vec, sample_vec)
        if sample.get("category") == category_lower:
            score += 0.03
        if sample.get("priority") == priority_upper:
            score += 0.02

        if score > best_score:
            best_score = score
            best_response = str(sample.get("response", "")).strip()

    # Avoid using weakly similar examples.
    if best_score < 0.12 or not best_response:
        return ""

    if "{username}" in best_response:
        return best_response.format(username=username)
    return best_response


def _is_generic_dataset_response(text: str) -> bool:
    t = (text or "").lower()
    if not t:
        return True
    generic_markers = [
        "we have logged your feedback",
        "address it in due course",
        "if your situation changes",
        "best regards",
    ]
    hits = sum(1 for marker in generic_markers if marker in t)
    return hits >= 2


def _load_models() -> None:
    global _sentiment_pipeline, _emotion_pipeline, _models_loaded, _use_transformers
    if _models_loaded:
        return

    try:
        from transformers import pipeline

        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
        _emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            truncation=True,
            max_length=512,
        )
        _use_transformers = True
        logger.info("Transformer models loaded.")
    except Exception as exc:
        _use_transformers = False
        logger.warning("Transformer load failed; using rule-based fallback: %s", exc)
    finally:
        _models_loaded = True


def _rule_based_sentiment(text: str) -> Dict[str, Any]:
    text_lower = text.lower()
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
    pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)

    if neg_count > pos_count:
        score = min(0.5 + (neg_count * 0.08), 0.99)
        return {"label": "NEGATIVE", "score": round(score, 3)}
    if pos_count > neg_count:
        return {"label": "POSITIVE", "score": round(0.5 + (pos_count * 0.1), 3)}
    return {"label": "NEUTRAL", "score": 0.55}


def _rule_based_emotion(text: str) -> Dict[str, Any]:
    text_lower = text.lower()
    scores = {emotion: sum(1 for kw in kws if kw in text_lower) for emotion, kws in EMOTION_PATTERNS.items()}

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        best = "neutral"

    total = sum(scores.values()) or 1
    confidence = round(min((scores.get(best, 1) / total) + 0.3, 0.99), 3)
    return {"label": best, "score": confidence}


def _compute_priority(sentiment: Dict[str, Any], emotion: Dict[str, Any], text: str) -> Dict[str, Any]:
    sentiment_label = sentiment["label"]
    sentiment_score = sentiment["score"]
    emotion_label = emotion["label"].lower()
    emotion_score = emotion["score"]

    high_intensity_emotions = {"anger", "fear", "disgust"}
    urgency_boost = any(kw in text.lower() for kw in URGENT_TRIGGERS)

    score = 0
    if sentiment_label == "NEGATIVE":
        score += int(sentiment_score * 40)
    elif sentiment_label == "NEUTRAL":
        score += 10

    if emotion_label in high_intensity_emotions:
        score += int(emotion_score * 40)
    elif emotion_label in {"sadness", "surprise"}:
        score += int(emotion_score * 20)

    if urgency_boost:
        score += 20

    if len(text.split()) > 100:
        score += 5

    if score >= 75:
        priority = "CRITICAL"
        priority_color = "#ef4444"
        priority_desc = "Immediate attention required."
    elif score >= 50:
        priority = "HIGH"
        priority_color = "#f97316"
        priority_desc = "Elevated concern; prioritize human response."
    elif score >= 25:
        priority = "MEDIUM"
        priority_color = "#eab308"
        priority_desc = "Moderate concern; review and respond."
    else:
        priority = "LOW"
        priority_color = "#22c55e"
        priority_desc = "Low urgency; can be auto-handled if clearly actionable."

    return {
        "priority": priority,
        "priority_color": priority_color,
        "priority_score": score,
        "priority_description": priority_desc,
    }


def _generate_root_cause(category: str, emotion_label: str) -> str:
    base = CATEGORY_ROOT_CAUSES.get((category or "other").lower(), CATEGORY_ROOT_CAUSES["other"])
    if emotion_label == "fear":
        return base + " Customer tone indicates anxiety and requires clear reassurance."
    if emotion_label in {"anger", "disgust"}:
        return base + " Emotional intensity suggests trust recovery is important."
    if emotion_label == "sadness":
        return base + " Empathetic response is recommended."
    return base


def _is_auto_resolvable(priority: str, text: str, sentiment: Dict[str, Any]) -> bool:
    if priority != "LOW":
        return False

    text_lower = text.lower()
    if any(kw in text_lower for kw in URGENT_TRIGGERS):
        return False

    # If explicitly high-risk/legal/safety terms exist, keep human in loop.
    hard_blockers = ["refund", "chargeback", "legal", "fraud", "threat", "injury", "security breach", "data leak"]
    if any(term in text_lower for term in hard_blockers):
        return False

    # Very low-confidence negative sentiment should still go to admin.
    if sentiment.get("label") == "NEGATIVE" and sentiment.get("score", 0) > 0.75:
        return False

    return True


def _generate_auto_user_response(username: str) -> str:
    return (
        f"Dear {username},\n\n"
        "Thanks for reporting this. We have automatically logged your issue and started basic remediation checks. "
        "If the issue continues, please reply with additional details and our support team will take over.\n\n"
        "Best regards,\nSentriMail Support"
    )


def _generate_admin_suggestion(priority: str, username: str) -> str:
    if priority == "CRITICAL":
        return (
            f"Dear {username},\n\n"
            "We sincerely apologize. Your complaint has been escalated to our critical-response queue and is being "
            "handled immediately. A senior specialist will contact you shortly with a concrete resolution plan.\n\n"
            "Regards,\nSentriMail Resolution Team"
        )
    if priority == "HIGH":
        return (
            f"Dear {username},\n\n"
            "Thank you for reporting this issue. We have prioritized your complaint and assigned it to a specialist. "
            "You will receive an update soon after investigation.\n\n"
            "Regards,\nSentriMail Support"
        )
    return (
        f"Dear {username},\n\n"
        "Thanks for sharing the details. We are reviewing your complaint and will provide a full response shortly.\n\n"
        "Regards,\nSentriMail Support"
    )


def analyze_complaint(
    text: str,
    category: str = "other",
    username: str = "Customer",
    complaint_id: str = "N/A",
) -> Dict[str, Any]:
    _load_models()

    if _use_transformers and _sentiment_pipeline:
        try:
            raw = _sentiment_pipeline(text[:512])[0]
            sentiment = {"label": raw["label"], "score": round(raw["score"], 3)}
        except Exception:
            sentiment = _rule_based_sentiment(text)
    else:
        sentiment = _rule_based_sentiment(text)

    if _use_transformers and _emotion_pipeline:
        try:
            raw = _emotion_pipeline(text[:512])[0]
            emotion = {"label": raw["label"], "score": round(raw["score"], 3)}
        except Exception:
            emotion = _rule_based_emotion(text)
    else:
        emotion = _rule_based_emotion(text)

    priority_data = _compute_priority(sentiment, emotion, text)
    auto_resolvable = _is_auto_resolvable(priority_data["priority"], text, sentiment)

    dataset_suggestion = _predict_response_from_dataset(
        text=text,
        username=username,
        category=category,
        priority=priority_data["priority"],
    )
    if dataset_suggestion and priority_data["priority"] in {"HIGH", "CRITICAL"} and _is_generic_dataset_response(dataset_suggestion):
        dataset_suggestion = ""

    admin_suggestion = dataset_suggestion or _generate_admin_suggestion(priority_data["priority"], username)
    auto_response = (dataset_suggestion or _generate_auto_user_response(username)) if auto_resolvable else ""

    return {
        "sentiment_label": sentiment["label"],
        "sentiment_score": sentiment["score"],
        "emotion_label": emotion["label"].capitalize(),
        "emotion_score": emotion["score"],
        **priority_data,
        "root_cause_summary": _generate_root_cause(category, emotion["label"].lower()),
        "auto_resolvable": auto_resolvable,
        "auto_resolution_reason": "Low priority and safe to auto-handle." if auto_resolvable else "Requires admin review.",
        "user_auto_response": auto_response,
        "admin_suggested_response": admin_suggestion,
        # Backward-compatible key consumed by existing templates.
        "ai_suggested_response": admin_suggestion if not auto_resolvable else auto_response,
        "model_used": "transformer" if _use_transformers else "rule-based",
        "response_source": "dataset" if dataset_suggestion else "template",
        "reference_id": complaint_id,
    }

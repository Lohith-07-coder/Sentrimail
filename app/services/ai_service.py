"""
SentriMail AI Service
---------------------
Hybrid analysis pipeline:
- Sentiment + emotion (transformer when available, otherwise rule-based)
- Priority scoring
- Auto-resolvable decision for LOW/simple complaints
- Admin suggested response generation
"""

import logging
import json
import math
import re
from pathlib import Path
from typing import Any, Dict

try:
    from langdetect import detect
except ImportError:
    def detect(text): return "en"

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None

from app.core.priority import calculate_priority

logger = logging.getLogger(__name__)

_sentiment_pipeline = None
_emotion_pipeline = None
_models_loaded = False
_use_transformers = True
_response_model = None
_response_model_loaded = False
_generative_pipeline = None


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

MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "response_model.json"
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
    num_samples = int(_response_model.get("num_samples", 0) or 0)
    unique_response_count = int(_response_model.get("unique_response_count", 0) or 0)
    if not idf or not samples:
        return ""
    if num_samples < 10 or unique_response_count <= 1:
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


def _infer_issue_type(text: str, category: str) -> str:
    t = (text or "").lower()
    c = (category or "").lower()
    if c in {"billing", "refund"} or any(k in t for k in ["refund", "charged", "billing", "payment", "invoice"]):
        return "billing"
    if any(k in t for k in ["login", "log in", "sign in", "password", "otp", "account locked"]):
        return "auth"
    if c == "technical" or any(k in t for k in ["server", "internet", "network", "pipeline", "error", "crash", "bug", "down"]):
        return "technical"
    if c == "delivery" or any(k in t for k in ["delivery", "shipment", "parcel", "courier", "late order"]):
        return "delivery"
    if c == "customer_service" or any(k in t for k in ["support", "agent", "rude", "no response"]):
        return "support"
    return "general"


def infer_category_and_issue(text: str, category: str = "other") -> tuple[str, str, list[str]]:
    """Infer a transparent keyword category only when no upstream category exists."""
    supplied_category = (category or "other").lower()
    if supplied_category != "other":
        return supplied_category, _infer_issue_type(text, supplied_category), []

    lower = (text or "").lower()
    # Refund/chargeback requests have a distinct operational workflow and take
    # precedence when they co-occur with general billing words.
    refund_terms = [term for term in ("refund", "chargeback", "reversal") if term in lower]
    if refund_terms:
        return "refund", _infer_issue_type(text, "refund"), refund_terms
    category_terms = {
        "billing": ["charged", "charge", "payment", "invoice", "billing", "subscription"],
        "technical": ["crash", "error", "bug", "outage", "server", "down", "data loss"],
        "delivery": ["delivery", "package", "parcel", "shipment", "courier", "tracking"],
        "customer_service": ["support", "agent", "representative", "no response", "rude"],
        "product": ["defective", "broken", "product", "device", "quality"],
    }
    matches = {
        name: [term for term in terms if term in lower]
        for name, terms in category_terms.items()
    }
    best_category, terms = max(matches.items(), key=lambda item: len(item[1]))
    if not terms:
        return "other", "general", []
    return best_category, _infer_issue_type(text, best_category), terms


def _infer_intent(category: str, issue: str, text: str) -> dict[str, Any]:
    text_lower = text.lower()
    if "refund" in text_lower or "chargeback" in text_lower:
        return {"label": "request_refund", "confidence": 0.85}
    if "cancel" in text_lower and ("subscription" in text_lower or "account" in text_lower):
        return {"label": "cancel_subscription", "confidence": 0.8}
    if issue == "auth":
        return {"label": "recover_account", "confidence": 0.8}
    if category == "technical":
        return {"label": "report_technical_issue", "confidence": 0.75}
    if category == "delivery":
        return {"label": "track_package", "confidence": 0.75}
    if category == "billing":
        return {"label": "dispute_charge", "confidence": 0.75}
    return {"label": "general_inquiry", "confidence": 0.5}

def _extract_entities(text: str) -> list[dict[str, str]]:
    entities = []
    for match in re.finditer(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text):
        entities.append({"type": "email", "value": match.group(0)})
    for match in re.finditer(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", text):
        entities.append({"type": "phone", "value": match.group(0)})
    for match in re.finditer(r"(?i)\b(?:order|ticket|ref|reference|complaint)[\s#:]+([A-Z0-9-]+)\b", text):
        entities.append({"type": "reference_id", "value": match.group(1)})
    for match in re.finditer(r"\$\s?\d+(?:\.\d{2})?", text):
        entities.append({"type": "money", "value": match.group(0)})
    return entities

def _detect_and_translate(text: str) -> tuple[str, str]:
    if not text.strip():
        return "en", text
    try:
        lang = detect(text)
    except Exception:
        lang = "en"
    if lang != "en" and GoogleTranslator is not None:
        try:
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            return lang, translated
        except Exception:
            return lang, text
    return "en", text

def _load_models() -> None:
    global _sentiment_pipeline, _emotion_pipeline, _generative_pipeline, _models_loaded, _use_transformers
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
        try:
            _generative_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
        except:
            _generative_pipeline = None

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
    from app.core.priority import detect_urgency
    if any(detect_urgency(text).values()):
        return False

    hard_blockers = ["refund", "chargeback", "legal", "fraud", "threat", "injury", "security breach", "data leak"]
    if any(term in text_lower for term in hard_blockers):
        return False

    if sentiment.get("label") == "NEGATIVE" and sentiment.get("score", 0) > 0.75:
        return False

    return True


def _generate_local_llm_response(text: str, category: str, language: str, username: str) -> str:
    if not _generative_pipeline:
        return ""
    prompt = (
        "System message:\n"
        "You are a professional customer support AI. You represent an organization that takes every complaint seriously. "
        "Always respond with empathy and give a specific, actionable solution. Never give vague or generic replies.\n\n"
        "User message:\n"
        f"Category: {category}. Language: {language}. Complaint text: {text}. "
        f"Write a helpful, specific reply to the user. Keep it to 3 to 4 sentences."
    )
    try:
        res = _generative_pipeline(prompt, max_length=150, do_sample=True, top_p=0.95)[0]['generated_text']
        res = res.strip()
        if "{username}" in res:
            res = res.format(username=username)
        return res
    except:
        return ""

def _get_category_fallback(category: str, issue: str, text: str, username: str) -> str:
    cat_lower = category.lower()
    iss_lower = issue.lower()
    txt_lower = text.lower()
    
    if "card" in cat_lower or "card" in txt_lower:
        return (
            f"Dear {username},\n\n"
            "You can locate your card through your account/card-management section, or track its delivery status "
            "via the provided tracking link. Let us know if it hasn't arrived within the expected timeframe.\n\n"
            "Regards,\nSentriMail Support"
        )
    if "charge" in txt_lower or "bill" in txt_lower or "refund" in txt_lower or cat_lower in {"billing", "refund"} or iss_lower == "billing":
        return (
            f"Dear {username},\n\n"
            "We're sorry about the billing concern you raised. Please provide the transaction details "
            "so we can verify the charge and assist with any necessary refund or adjustment process.\n\n"
            "Regards,\nSentriMail Support"
        )
    if "hack" in txt_lower or "unauthorized" in txt_lower or "secure" in txt_lower or iss_lower == "auth" or "account" in txt_lower:
        return (
            f"Dear {username},\n\n"
            "We're sorry to hear about the account access or security issue. Please secure your account immediately "
            "and contact support so we can help protect the account and investigate the incident.\n\n"
            "Regards,\nSentriMail Support"
        )
    if cat_lower == "delivery" or iss_lower == "delivery" or "status" in txt_lower:
        return (
            f"Dear {username},\n\n"
            "We apologize for the issue with your delivery. Please ensure your shipping address is correct, "
            "and we will check the tracking status with our courier partners to resolve this quickly.\n\n"
            "Regards,\nSentriMail Support"
        )
    if cat_lower == "technical" or iss_lower == "technical":
        return (
            f"Dear {username},\n\n"
            "We apologize for the technical difficulties you're experiencing. Our engineering team has been notified "
            "and we are investigating the system logs to identify and resolve the root cause.\n\n"
            "Regards,\nSentriMail Support"
        )
    return ""


def analyze_complaint(
    text: str,
    category: str = "other",
    username: str = "Customer",
    complaint_id: str = "N/A",
    history: dict[str, int] | None = None,
    created_at: str | None = None,
    status: str | None = None,
) -> Dict[str, Any]:
    _load_models()

    language, translated_text = _detect_and_translate(text)
    analysis_text = translated_text if translated_text else text

    if _use_transformers and _sentiment_pipeline:
        try:
            raw = _sentiment_pipeline(analysis_text[:512])[0]
            sentiment = {"label": raw["label"], "score": round(raw["score"], 3)}
        except Exception:
            sentiment = _rule_based_sentiment(analysis_text)
    else:
        sentiment = _rule_based_sentiment(analysis_text)

    if _use_transformers and _emotion_pipeline:
        try:
            raw = _emotion_pipeline(analysis_text[:512])[0]
            emotion = {"label": raw["label"], "score": round(raw["score"], 3)}
        except Exception:
            emotion = _rule_based_emotion(analysis_text)
    else:
        emotion = _rule_based_emotion(analysis_text)

    inferred_category, issue, category_terms = infer_category_and_issue(analysis_text, category)
    intent_data = _infer_intent(inferred_category, issue, analysis_text)
    entities = _extract_entities(analysis_text)

    priority_data = calculate_priority(
        sentiment_label=sentiment["label"], sentiment_score=sentiment["score"],
        emotion_label=emotion["label"], emotion_score=emotion["score"], text=analysis_text,
        category=inferred_category, issue=issue, history=history,
        created_at=created_at, status=status,
    )
    priority_colors = {
        "CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#eab308", "LOW": "#22c55e",
    }
    priority_data["priority_color"] = priority_colors[priority_data["priority"]]
    auto_resolvable = _is_auto_resolvable(priority_data["priority"], analysis_text, sentiment)

    from app.services.response_intelligence import generate_grounded_response
    
    grounded_res = generate_grounded_response(
        text=analysis_text,
        username=username,
        category=inferred_category,
        priority=priority_data["priority"],
        language=language
    )
    
    dataset_suggestion = grounded_res["response"]
    response_source = grounded_res.get("source", "none")
    response_confidence = grounded_res.get("confidence", 0.0)

    if dataset_suggestion and _is_generic_dataset_response(dataset_suggestion):
        dataset_suggestion = ""
        response_source = "none"
        response_confidence = 0.0

    final_response = dataset_suggestion

    if not final_response:
        # Step 3: try local LLM generation
        generated = _generate_local_llm_response(analysis_text, inferred_category, language, username)
        if generated:
            final_response = generated
            response_source = "local_llm"
            response_confidence = 0.6
        else:
            # Step 4: use category/intent-specific fallback
            final_response = _get_category_fallback(inferred_category, issue, analysis_text, username)
            if final_response:
                response_source = "category_template"
                response_confidence = 0.8
            else:
                # Step 5: generic acknowledgement fallback
                final_response = (
                    f"Dear {username},\n\n"
                    "Thanks for sharing the details. We are reviewing your complaint and will provide a full response shortly.\n\n"
                    "Regards,\nSentriMail Support"
                )
                response_source = "generic_fallback"
                response_confidence = 1.0

    human_review_required = grounded_res.get("human_review_required", True)
    if response_source in ["local_llm", "category_template", "generic_fallback", "none"]:
        human_review_required = True

    auto_resolvable = _is_auto_resolvable(priority_data["priority"], analysis_text, sentiment)
    if human_review_required:
        auto_resolvable = False

    admin_suggestion = final_response
    auto_response = final_response if auto_resolvable else ""

    return {
        "language": language,
        "translated_text": translated_text if language != "en" else "",
        "sentiment_label": sentiment["label"],
        "sentiment_score": sentiment["score"],
        "emotion_label": emotion["label"].capitalize(),
        "emotion_score": emotion["score"],
        **priority_data,
        "urgency": priority_data.get("urgency_indicators", []),
        "category": inferred_category,
        "issue": issue,
        "intent": intent_data["label"],
        "intent_confidence": intent_data["confidence"],
        "entities": entities,
        "category_indicators": category_terms,
        "root_cause_summary": _generate_root_cause(inferred_category, emotion["label"].lower()),
        "auto_resolvable": auto_resolvable,
        "auto_resolution_reason": "Low priority and safe to auto-handle." if auto_resolvable else ("Low retrieval confidence requires review." if grounded_res.get("human_review_required") else "Requires admin review."),
        "user_auto_response": auto_response,
        "admin_suggested_response": admin_suggestion,
        "ai_suggested_response": admin_suggestion,
        "model_used": "transformer" if _use_transformers else "rule-based",
        "response_source": response_source,
        "response_confidence": response_confidence,
        "reference_id": complaint_id,
    }

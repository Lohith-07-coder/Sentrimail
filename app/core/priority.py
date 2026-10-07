"""Deterministic, explainable complaint-priority scoring.

This module deliberately uses rules rather than a learned priority model: the
application has no labelled historical severity outcomes suitable for training.
Every score returned here is the sum of the listed feature contributions.

Priority Bands
--------------
    CRITICAL : score >= 75  OR  any emergency keyword present
    HIGH     : 45 <= score < 75
    MEDIUM   : 20 <= score < 45
    LOW      : score < 20
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


# ── Keyword sets ────────────────────────────────────────────────────────────

EMERGENCY_TERMS = {
    "legal action", "lawsuit", "police", "danger", "injury", "fraud",
    "stolen", "data loss", "data leak", "security breach", "outage",
    "server down", "production down", "violence", "assault", "harassment",
    "abuse", "threat", "court",
}
URGENCY_TERMS = {
    "urgent", "immediately", "asap", "emergency", "critical", "today",
    "blocked", "cannot access", "unable to access",
}

# ── Priority thresholds (exported for tests & callers) ──────────────────────

PRIORITY_THRESHOLDS = {
    "CRITICAL": 75,
    "HIGH": 45,
    "MEDIUM": 20,
    "LOW": 0,
}

PRIORITY_DESCRIPTIONS = {
    "CRITICAL": "Immediate attention required.",
    "HIGH": "Elevated concern; prioritize human response.",
    "MEDIUM": "Moderate concern; review and respond.",
    "LOW": "Low urgency; suitable for standard handling.",
}

# ── Feature scoring tables (exported for transparency) ──────────────────────

SENTIMENT_SCORES = {
    # (label, min_score_threshold) → contribution
    ("NEGATIVE", 0.80): 20,
    ("NEGATIVE", 0.60): 14,
    ("NEGATIVE", 0.00): 8,
    ("POSITIVE", 0.00): -4,
}

EMOTION_SCORES = {
    "anger":    {"high": 18, "low": 12, "threshold": 0.70},
    "fear":     {"high": 18, "low": 12, "threshold": 0.70},
    "disgust":  {"high": 14, "low": 14, "threshold": 0.00},
    "sadness":  {"high": 10, "low": 7,  "threshold": 0.70},
    "surprise": {"high": 5,  "low": 3,  "threshold": 0.70},
}

SEVERE_ISSUES = {"fraud", "security", "data_loss", "outage"}
ELEVATED_CATEGORIES = {"refund", "billing", "technical"}


# ── Helpers ─────────────────────────────────────────────────────────────────

def detect_urgency(text: str) -> dict[str, list[str]]:
    """Return the actual urgency and emergency terms present in the text."""
    lower = (text or "").lower()
    return {
        "emergency": sorted(term for term in EMERGENCY_TERMS if term in lower),
        "urgency": sorted(term for term in URGENCY_TERMS if term in lower),
    }


def _add(contributions: dict[str, int], reasons: list[str], key: str, value: int, reason: str) -> None:
    """Record a non-zero feature contribution and its human-readable reason."""
    if value:
        contributions[key] = value
        reasons.append(reason)


def _sentiment_contribution(label: str, score: float) -> tuple[int, str]:
    """Sentiment → score contribution using tiered thresholds."""
    sentiment = (label or "NEUTRAL").upper()
    if sentiment == "NEGATIVE":
        if score >= 0.80:
            return 20, "Strong negative sentiment"
        elif score >= 0.60:
            return 14, "Moderate negative sentiment"
        else:
            return 8, "Weak negative sentiment"
    elif sentiment == "POSITIVE":
        return -4, "Positive sentiment lowers urgency"
    return 0, ""


def _emotion_contribution(label: str, score: float) -> tuple[int, str]:
    """Emotion → score contribution with per-emotion thresholds."""
    emotion = (label or "neutral").lower()
    cfg = EMOTION_SCORES.get(emotion)
    if not cfg:
        return 0, ""
    value = cfg["high"] if score >= cfg["threshold"] else cfg["low"]
    return value, f"{emotion.capitalize()} detected"


def _keyword_contribution(text: str) -> tuple[int, str, dict[str, list[str]]]:
    """Emergency/urgency keyword scan."""
    indicators = detect_urgency(text)
    if indicators["emergency"]:
        return 60, f"Emergency indicator(s): {', '.join(indicators['emergency'])}", indicators
    if indicators["urgency"]:
        return 18, f"Urgency indicator(s): {', '.join(indicators['urgency'])}", indicators
    return 0, "", indicators


def _issue_category_contribution(category: str, issue: str) -> tuple[int, str]:
    """Category and issue severity scoring."""
    if issue in SEVERE_ISSUES:
        return 15, f"High-impact issue type: {issue}"
    if category in ELEVATED_CATEGORIES:
        return 5, f"Complaint category: {category}"
    return 0, ""


def _history_contributions(history: dict[str, int] | None) -> list[tuple[str, int, str]]:
    """Repeated-complaint and open-history scoring."""
    history = history or {}
    results = []
    previous = max(0, int(history.get("previous_complaints", 0)))
    unresolved = max(0, int(history.get("unresolved_complaints", 0)))
    if previous >= 3:
        results.append(("history", 7, f"Repeated customer contact ({previous} previous complaints)"))
    if unresolved >= 2:
        results.append(("open_history", 5, f"Multiple unresolved complaints ({unresolved})"))
    return results


def _sla_contribution(created_at: str | None, status: str | None) -> tuple[int, str | None]:
    """Score actual age only for unresolved complaints with parseable timestamps."""
    if not created_at or status not in {"pending", "pending_admin", "in_progress"}:
        return 0, None
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - created.astimezone(timezone.utc)).total_seconds() / 3600
    except (TypeError, ValueError):
        return 0, None
    if age_hours >= 48:
        return 25, "Complaint is past the 48-hour escalation threshold"
    if age_hours >= 24:
        return 15, "Complaint is past the 24-hour review threshold"
    return 0, None


def _length_contribution(text: str) -> tuple[int, str]:
    """Long complaints signal higher effort / more complex issues."""
    word_count = len((text or "").split())
    if word_count > 100:
        return 5, f"Detailed complaint ({word_count} words)"
    return 0, ""


def _classify_priority(score: int, has_emergency: bool) -> tuple[str, str]:
    """Map a numeric score to a priority band."""
    if has_emergency or score >= PRIORITY_THRESHOLDS["CRITICAL"]:
        return "CRITICAL", PRIORITY_DESCRIPTIONS["CRITICAL"]
    if score >= PRIORITY_THRESHOLDS["HIGH"]:
        return "HIGH", PRIORITY_DESCRIPTIONS["HIGH"]
    if score >= PRIORITY_THRESHOLDS["MEDIUM"]:
        return "MEDIUM", PRIORITY_DESCRIPTIONS["MEDIUM"]
    return "LOW", PRIORITY_DESCRIPTIONS["LOW"]


# ── Main entry point ───────────────────────────────────────────────────────

def calculate_priority(
    *,
    sentiment_label: str,
    sentiment_score: float,
    emotion_label: str,
    emotion_score: float,
    text: str,
    category: str = "other",
    issue: str = "general",
    history: dict[str, int] | None = None,
    created_at: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """Calculate an inspectable score and severity from available inputs only.

    Every score is reproducible: it equals sum(feature_contributions.values()).
    Every priority has reasons: a non-empty list of human-readable strings.
    """
    contributions: dict[str, int] = {}
    reasons: list[str] = []
    category = (category or "other").lower()
    issue = (issue or "general").lower()

    # 1. Sentiment
    s_val, s_reason = _sentiment_contribution(sentiment_label, sentiment_score)
    _add(contributions, reasons, "sentiment", s_val, s_reason)

    # 2. Emotion
    e_val, e_reason = _emotion_contribution(emotion_label, emotion_score)
    _add(contributions, reasons, "emotion", e_val, e_reason)

    # 3. Emergency / urgency keywords
    k_val, k_reason, indicators = _keyword_contribution(text)
    if indicators["emergency"]:
        _add(contributions, reasons, "emergency", k_val, k_reason)
    elif indicators["urgency"]:
        _add(contributions, reasons, "urgency", k_val, k_reason)

    # 4. Issue severity / category
    ic_val, ic_reason = _issue_category_contribution(category, issue)
    _add(contributions, reasons, "issue_severity" if issue in SEVERE_ISSUES else "category", ic_val, ic_reason)

    # 5. Complaint history
    for key, val, reason in _history_contributions(history):
        _add(contributions, reasons, key, val, reason)

    # 6. SLA elapsed time
    sla_value, sla_reason = _sla_contribution(created_at, status)
    if sla_reason:
        _add(contributions, reasons, "sla", sla_value, sla_reason)

    # 7. Complaint length
    l_val, l_reason = _length_contribution(text)
    _add(contributions, reasons, "length", l_val, l_reason)

    # ── Score capping ───────────────────────────────────────────────────────
    raw_score = sum(contributions.values())
    if raw_score > 100:
        _add(contributions, reasons, "score_cap", 100 - raw_score, "Score capped at 100")
    elif raw_score < 0:
        _add(contributions, reasons, "score_floor", -raw_score, "Score floored at 0")

    score = max(0, min(100, sum(contributions.values())))

    # ── Classification ──────────────────────────────────────────────────────
    has_emergency = bool(indicators["emergency"])
    priority, description = _classify_priority(score, has_emergency)

    return {
        "priority": priority,
        "priority_score": score,
        "priority_description": description,
        "priority_reasons": reasons or ["No elevated priority signals detected"],
        "feature_contributions": contributions,
        "urgency_indicators": indicators["urgency"],
        "emergency_indicators": indicators["emergency"],
        "thresholds": PRIORITY_THRESHOLDS,
    }

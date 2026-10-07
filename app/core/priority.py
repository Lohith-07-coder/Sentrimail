"""Deterministic, explainable complaint-priority scoring.

This module deliberately uses rules rather than a learned priority model: the
application has no labelled historical severity outcomes suitable for training.
Every score returned here is the sum of the listed feature contributions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


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


def detect_urgency(text: str) -> dict[str, list[str]]:
    """Return the actual urgency and emergency terms present in the text."""
    lower = (text or "").lower()
    return {
        "emergency": sorted(term for term in EMERGENCY_TERMS if term in lower),
        "urgency": sorted(term for term in URGENCY_TERMS if term in lower),
    }


def _add(contributions: dict[str, int], reasons: list[str], key: str, value: int, reason: str) -> None:
    if value:
        contributions[key] = value
        reasons.append(reason)


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
    """Calculate an inspectable score and severity from available inputs only."""
    contributions: dict[str, int] = {}
    reasons: list[str] = []
    sentiment = (sentiment_label or "NEUTRAL").upper()
    emotion = (emotion_label or "neutral").lower()
    category = (category or "other").lower()
    issue = (issue or "general").lower()

    if sentiment == "NEGATIVE":
        value = 20 if sentiment_score >= 0.80 else 14 if sentiment_score >= 0.60 else 8
        _add(contributions, reasons, "sentiment", value, "Negative customer sentiment")
    elif sentiment == "POSITIVE":
        _add(contributions, reasons, "sentiment", -4, "Positive sentiment lowers urgency")

    if emotion in {"anger", "fear"}:
        value = 18 if emotion_score >= 0.70 else 12
        _add(contributions, reasons, "emotion", value, f"{emotion.capitalize()} detected")
    elif emotion == "disgust":
        _add(contributions, reasons, "emotion", 14, "Disgust detected")
    elif emotion == "sadness":
        _add(contributions, reasons, "emotion", 7, "Sadness detected")

    indicators = detect_urgency(text)
    if indicators["emergency"]:
        _add(
            contributions, reasons, "emergency", 60,
            f"Emergency indicator(s): {', '.join(indicators['emergency'])}",
        )
    elif indicators["urgency"]:
        _add(
            contributions, reasons, "urgency", 18,
            f"Urgency indicator(s): {', '.join(indicators['urgency'])}",
        )

    severe_issues = {"fraud", "security", "data_loss", "outage"}
    elevated_categories = {"refund", "billing", "technical"}
    if issue in severe_issues:
        _add(contributions, reasons, "issue_severity", 15, f"High-impact issue type: {issue}")
    elif category in elevated_categories:
        _add(contributions, reasons, "category", 5, f"Complaint category: {category}")

    history = history or {}
    previous = max(0, int(history.get("previous_complaints", 0)))
    unresolved = max(0, int(history.get("unresolved_complaints", 0)))
    if previous >= 3:
        _add(contributions, reasons, "history", 7, f"Repeated customer contact ({previous} previous complaints)")
    if unresolved >= 2:
        _add(contributions, reasons, "open_history", 5, f"Multiple unresolved complaints ({unresolved})")

    sla_value, sla_reason = _sla_contribution(created_at, status)
    if sla_reason:
        _add(contributions, reasons, "sla", sla_value, sla_reason)

    raw_score = sum(contributions.values())
    if raw_score > 100:
        _add(contributions, reasons, "score_cap", 100 - raw_score, "Score capped at 100")
    elif raw_score < 0:
        _add(contributions, reasons, "score_floor", -raw_score, "Score floored at 0")
    score = sum(contributions.values())
    if indicators["emergency"] or score >= 75:
        priority = "CRITICAL"
        description = "Immediate attention required."
    elif score >= 45:
        priority = "HIGH"
        description = "Elevated concern; prioritize human response."
    elif score >= 20:
        priority = "MEDIUM"
        description = "Moderate concern; review and respond."
    else:
        priority = "LOW"
        description = "Low urgency; suitable for standard handling."

    return {
        "priority": priority,
        "priority_score": score,
        "priority_description": description,
        "priority_reasons": reasons or ["No elevated priority signals detected"],
        "feature_contributions": contributions,
        "urgency_indicators": indicators["urgency"],
        "emergency_indicators": indicators["emergency"],
    }

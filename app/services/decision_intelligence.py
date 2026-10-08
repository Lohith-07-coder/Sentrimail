"""Phase 6 — Decision Intelligence (recommendation-only).

This module implements a *rule-based baseline* recommendation engine.
It does NOT claim to be an offline-RL or bandit policy — we currently
lack the historical (state, action, reward) transition data required
for that.  The architecture is designed so that once real human-feedback
data accumulates via `feedback_store`, a learned policy can replace or
augment the rules below without changing the external API.

Action space
------------
REPLY                 — send the AI-suggested response
ESCALATE              — forward to a senior / specialist queue
ASSIGN                — route to an appropriate team
REQUEST_INFORMATION   — ask the customer for missing details
INCREASE_PRIORITY     — bump severity band by one level
CLOSE                 — mark as resolved / no further action needed
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Action space ────────────────────────────────────────────────────────────

ACTIONS = [
    "REPLY",
    "ESCALATE",
    "ASSIGN",
    "REQUEST_INFORMATION",
    "INCREASE_PRIORITY",
    "CLOSE",
]

# ── State builder ───────────────────────────────────────────────────────────

def build_state(
    *,
    category: str = "other",
    issue: str = "general",
    sentiment_label: str = "NEUTRAL",
    sentiment_score: float = 0.5,
    emotion_label: str = "neutral",
    emotion_score: float = 0.5,
    priority: str = "LOW",
    priority_score: int = 0,
    urgency_indicators: Optional[List[str]] = None,
    emergency_indicators: Optional[List[str]] = None,
    sla_hours: Optional[float] = None,
    status: str = "pending",
    policy_decision: str = "ABSTAIN",
    policy_confidence: float = 0.0,
    auto_resolvable: bool = False,
    previous_complaints: int = 0,
    unresolved_complaints: int = 0,
    response_confidence: float = 0.0,
) -> Dict[str, Any]:
    """Build a normalised state dict from SentriMail analysis outputs.

    Every field is explicitly named so the schema is self-documenting and
    can later be serialised to a training dataset.
    """
    return {
        "category": (category or "other").lower(),
        "issue": (issue or "general").lower(),
        "sentiment_label": (sentiment_label or "NEUTRAL").upper(),
        "sentiment_score": float(sentiment_score),
        "emotion_label": (emotion_label or "neutral").lower(),
        "emotion_score": float(emotion_score),
        "priority": (priority or "LOW").upper(),
        "priority_score": int(priority_score),
        "urgency_indicators": urgency_indicators or [],
        "emergency_indicators": emergency_indicators or [],
        "sla_hours": sla_hours,
        "status": (status or "pending").lower(),
        "policy_decision": (policy_decision or "ABSTAIN").upper(),
        "policy_confidence": float(policy_confidence),
        "auto_resolvable": bool(auto_resolvable),
        "previous_complaints": int(previous_complaints),
        "unresolved_complaints": int(unresolved_complaints),
        "response_confidence": float(response_confidence),
    }


# ── Baseline rule-based recommender ────────────────────────────────────────

def recommend_action(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return a recommended action, confidence, and reasoning.

    This is a deterministic heuristic baseline.  It is *not* a learned
    policy — see module docstring.
    """
    priority = state.get("priority", "LOW")
    sentiment = state.get("sentiment_label", "NEUTRAL")
    emotion = state.get("emotion_label", "neutral")
    emergency = state.get("emergency_indicators", [])
    urgency = state.get("urgency_indicators", [])
    sla_hours = state.get("sla_hours")
    status = state.get("status", "pending")
    auto_resolvable = state.get("auto_resolvable", False)
    policy_decision = state.get("policy_decision", "ABSTAIN")
    response_confidence = state.get("response_confidence", 0.0)
    previous = state.get("previous_complaints", 0)
    unresolved = state.get("unresolved_complaints", 0)
    category = state.get("category", "other")

    reasons: List[str] = []

    # ── Rule 1: already resolved → CLOSE ────────────────────────────────
    if status in ("resolved", "closed", "auto_replied"):
        return _result("CLOSE", 0.90, ["Complaint already resolved or auto-replied."], state)

    # ── Rule 2: emergency keywords → ESCALATE immediately ───────────────
    if emergency:
        reasons.append(f"Emergency indicators detected: {', '.join(emergency)}")
        return _result("ESCALATE", 0.95, reasons, state)

    # ── Rule 3: CRITICAL priority → ESCALATE ────────────────────────────
    if priority == "CRITICAL":
        reasons.append("Priority is CRITICAL.")
        if sla_hours is not None and sla_hours >= 24:
            reasons.append(f"SLA breach: {sla_hours:.0f}h elapsed.")
        return _result("ESCALATE", 0.92, reasons, state)

    # ── Rule 4: HIGH + repeat offender → ESCALATE ───────────────────────
    if priority == "HIGH" and (previous >= 3 or unresolved >= 2):
        reasons.append("HIGH priority with complaint history.")
        return _result("ESCALATE", 0.85, reasons, state)

    # ── Rule 5: SLA breach on any unresolved → INCREASE_PRIORITY ────────
    if sla_hours is not None and sla_hours >= 48 and priority != "CRITICAL":
        reasons.append(f"48-hour SLA breach ({sla_hours:.0f}h).")
        return _result("INCREASE_PRIORITY", 0.88, reasons, state)

    if sla_hours is not None and sla_hours >= 24 and priority in ("LOW", "MEDIUM"):
        reasons.append(f"24-hour SLA breach ({sla_hours:.0f}h).")
        return _result("INCREASE_PRIORITY", 0.80, reasons, state)

    # ── Rule 6: LOW + auto-resolvable + good confidence → REPLY ─────────
    if priority == "LOW" and auto_resolvable and response_confidence >= 0.25:
        reasons.append("Low priority, auto-resolvable, adequate response confidence.")
        return _result("REPLY", 0.85, reasons, state)

    # ── Rule 7: needs more info (vague complaint) ───────────────────────
    if (priority in ("LOW", "MEDIUM")
            and category == "other"
            and state.get("issue") == "general"
            and sentiment == "NEUTRAL"):
        reasons.append("Vague complaint with no clear category; request more details.")
        return _result("REQUEST_INFORMATION", 0.75, reasons, state)

    # ── Rule 8: HIGH priority → ASSIGN to specialist ────────────────────
    if priority == "HIGH":
        team = _suggest_team(category)
        reasons.append(f"HIGH priority; route to {team} team.")
        return _result("ASSIGN", 0.80, reasons, state, metadata={"suggested_team": team})

    # ── Rule 9: MEDIUM with policy match → REPLY ────────────────────────
    if priority == "MEDIUM" and policy_decision == "POLICY_FOUND" and response_confidence >= 0.20:
        reasons.append("MEDIUM priority with matching policy context.")
        return _result("REPLY", 0.72, reasons, state)

    # ── Rule 10: MEDIUM general → ASSIGN ────────────────────────────────
    if priority == "MEDIUM":
        team = _suggest_team(category)
        reasons.append(f"MEDIUM priority; route to {team} team for review.")
        return _result("ASSIGN", 0.65, reasons, state, metadata={"suggested_team": team})

    # ── Fallback: REPLY with low confidence ─────────────────────────────
    reasons.append("No strong signal; defaulting to REPLY with human review recommended.")
    return _result("REPLY", 0.40, reasons, state)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _suggest_team(category: str) -> str:
    """Map category to a team name for ASSIGN recommendations."""
    mapping = {
        "billing": "billing",
        "refund": "billing",
        "technical": "engineering",
        "delivery": "logistics",
        "customer_service": "support",
        "product": "product",
    }
    return mapping.get(category, "general_support")


def _result(
    action: str,
    confidence: float,
    reasons: List[str],
    state: Dict[str, Any],
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the standard recommendation envelope."""
    return {
        "recommended_action": action,
        "confidence": round(confidence, 3),
        "reasons": reasons,
        "human_review_required": confidence < 0.70,
        "state_snapshot": state,
        "metadata": metadata or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

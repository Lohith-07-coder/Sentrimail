"""Phase 6 — Feedback Store.

Persists every (state, recommendation, human_action, outcome) transition
to a local JSON-lines file so that:

1. We can audit what the system recommended vs. what the human did.
2. We accumulate the dataset needed to later train an offline-RL or
   contextual-bandit policy.

Schema (one JSON object per line)
---------------------------------
{
    "id":                   str,    # unique transition id
    "complaint_id":         str,    # SentriMail complaint reference
    "timestamp":            str,    # ISO-8601 when recommendation was made
    "state":                dict,   # full state snapshot (see decision_intelligence.build_state)
    "recommended_action":   str,    # what the engine suggested
    "recommendation_confidence": float,
    "human_action":         str | null,   # what the admin actually did (null until recorded)
    "recommendation_accepted": bool | null,
    "outcome":              str | null,   # e.g. "resolved", "re-escalated", "customer_satisfied"
    "outcome_timestamp":    str | null,
    "is_synthetic":         bool,   # True if generated for bootstrap / testing
}

IMPORTANT: All synthetic rows are explicitly labelled ``is_synthetic=True``
so that any future learner can distinguish real feedback from bootstrap data.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

FEEDBACK_PATH = Path(__file__).resolve().parents[2] / "data" / "feedback_log.jsonl"


# ── Write helpers ───────────────────────────────────────────────────────────

def record_recommendation(
    complaint_id: str,
    state: Dict[str, Any],
    recommended_action: str,
    confidence: float,
    *,
    is_synthetic: bool = False,
) -> str:
    """Persist a new recommendation row; returns its transition id."""
    transition_id = str(uuid.uuid4())
    row = {
        "id": transition_id,
        "complaint_id": complaint_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "recommended_action": recommended_action,
        "recommendation_confidence": round(confidence, 4),
        "human_action": None,
        "recommendation_accepted": None,
        "outcome": None,
        "outcome_timestamp": None,
        "is_synthetic": is_synthetic,
    }
    _append(row)
    logger.info(
        "Recorded recommendation %s for complaint %s: %s (%.2f)",
        transition_id, complaint_id, recommended_action, confidence,
    )
    return transition_id


def record_human_action(
    transition_id: str,
    human_action: str,
) -> bool:
    """Update a transition row with the action the human actually took."""
    rows = _load_all()
    updated = False
    for row in rows:
        if row.get("id") == transition_id:
            row["human_action"] = human_action
            row["recommendation_accepted"] = (
                human_action == row.get("recommended_action")
            )
            updated = True
            break
    if updated:
        _write_all(rows)
    return updated


def record_outcome(
    transition_id: str,
    outcome: str,
) -> bool:
    """Attach a terminal outcome label (e.g. 'resolved', 'customer_satisfied')."""
    rows = _load_all()
    updated = False
    for row in rows:
        if row.get("id") == transition_id:
            row["outcome"] = outcome
            row["outcome_timestamp"] = datetime.now(timezone.utc).isoformat()
            updated = True
            break
    if updated:
        _write_all(rows)
    return updated


def get_transition(transition_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single transition by id."""
    for row in _load_all():
        if row.get("id") == transition_id:
            return row
    return None


def list_transitions(
    complaint_id: Optional[str] = None,
    only_real: bool = False,
) -> List[Dict[str, Any]]:
    """Return transitions, optionally filtered."""
    rows = _load_all()
    if complaint_id:
        rows = [r for r in rows if r.get("complaint_id") == complaint_id]
    if only_real:
        rows = [r for r in rows if not r.get("is_synthetic")]
    return rows


def acceptance_rate() -> Dict[str, Any]:
    """Quick stats on how often recommendations were accepted."""
    rows = [r for r in _load_all() if r.get("human_action") is not None]
    if not rows:
        return {"total": 0, "accepted": 0, "corrected": 0, "rate": 0.0}
    accepted = sum(1 for r in rows if r.get("recommendation_accepted"))
    return {
        "total": len(rows),
        "accepted": accepted,
        "corrected": len(rows) - accepted,
        "rate": round(accepted / len(rows), 3),
    }


# ── I/O primitives ─────────────────────────────────────────────────────────

def _append(row: Dict[str, Any]) -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _load_all() -> List[Dict[str, Any]]:
    if not FEEDBACK_PATH.exists():
        return []
    rows = []
    with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _write_all(rows: List[Dict[str, Any]]) -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def clear_store() -> None:
    """Remove all feedback data (useful in tests)."""
    if FEEDBACK_PATH.exists():
        FEEDBACK_PATH.unlink()

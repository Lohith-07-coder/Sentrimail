"""Boundary and explanation tests for the deterministic Phase 2 priority engine."""

from datetime import datetime, timedelta, timezone

from app.core.priority import calculate_priority
from app.services.ai_service import infer_category_and_issue


def _score(**overrides):
    payload = {
        "sentiment_label": "NEUTRAL",
        "sentiment_score": 0.55,
        "emotion_label": "neutral",
        "emotion_score": 0.55,
        "text": "I would like an update.",
        "category": "other",
        "issue": "general",
    }
    payload.update(overrides)
    return calculate_priority(**payload)


def test_obvious_emergency_is_critical_and_explained():
    result = _score(
        sentiment_label="NEGATIVE", sentiment_score=0.95,
        emotion_label="fear", emotion_score=0.90,
        text="This is fraud and I need urgent help immediately.",
        category="billing", issue="fraud",
    )
    assert result["priority"] == "CRITICAL"
    assert result["feature_contributions"]["emergency"] == 60
    assert "fraud" in result["emergency_indicators"]
    assert result["priority_score"] == sum(result["feature_contributions"].values())


def test_high_priority_complaint_has_named_contributions():
    result = _score(
        sentiment_label="NEGATIVE", sentiment_score=0.90,
        emotion_label="anger", emotion_score=0.90,
        text="My billing problem is urgent and needs attention today.",
        category="billing", issue="billing",
    )
    assert result["priority"] == "HIGH"
    assert result["feature_contributions"] == {
        "sentiment": 20, "emotion": 18, "urgency": 18, "category": 5,
    }
    assert any("Negative" in reason for reason in result["priority_reasons"])


def test_normal_and_low_priority_boundaries():
    normal = _score(sentiment_label="NEGATIVE", sentiment_score=0.9)
    low = _score(sentiment_label="POSITIVE", sentiment_score=0.9, text="Thank you.")
    assert normal["priority"] == "MEDIUM"
    assert normal["priority_score"] == 20
    assert low["priority"] == "LOW"
    assert low["priority_score"] == 0


def test_sla_breach_and_repeated_history_use_real_available_features():
    old = (datetime.now(timezone.utc) - timedelta(hours=49)).isoformat()
    breached = _score(
        sentiment_label="POSITIVE", sentiment_score=0.9,
        created_at=old, status="pending_admin",
    )
    repeated = _score(
        sentiment_label="NEGATIVE", sentiment_score=0.9,
        history={"previous_complaints": 3, "unresolved_complaints": 2},
    )
    assert breached["priority"] == "MEDIUM"
    assert breached["feature_contributions"]["sla"] == 25
    assert repeated["priority"] == "MEDIUM"
    assert repeated["feature_contributions"]["history"] == 7
    assert repeated["feature_contributions"]["open_history"] == 5


def test_category_and_issue_are_inferred_only_from_matching_text():
    category, issue, indicators = infer_category_and_issue(
        "I was charged twice and need a refund for my subscription."
    )
    unknown = infer_category_and_issue("Could you please take a look at this?")
    assert (category, issue) == ("refund", "billing")
    assert "refund" in indicators
    assert unknown == ("other", "general", [])

"""Phase 3 — Explainable Priority & SLA tests.

Covers:
  - Boundary conditions at every threshold (LOW/MEDIUM/HIGH/CRITICAL)
  - Emergency keyword override
  - SLA escalation at 24h and 48h
  - Repeated-complaint history
  - Emotion tiers (anger, fear, disgust, sadness, surprise)
  - Sentiment tiers (high/medium/low negative, positive reduction)
  - Length contribution for detailed complaints
  - Score reproducibility (score == sum of contributions)
  - Every result has reasons
  - Feature contributions are returned
  - Valid negative-sentiment billing complaints are NOT incorrectly LOW
  - Thresholds are included in output
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.core.priority import (
    PRIORITY_THRESHOLDS,
    calculate_priority,
    detect_urgency,
    _sentiment_contribution,
    _emotion_contribution,
    _keyword_contribution,
    _issue_category_contribution,
    _history_contributions,
    _sla_contribution,
    _length_contribution,
    _classify_priority,
)
from app.services.ai_service import infer_category_and_issue


# ── Helper ──────────────────────────────────────────────────────────────────

def _score(**overrides):
    """Build a priority result with sensible defaults; override any field."""
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


# ── Reproducibility ─────────────────────────────────────────────────────────

class TestReproducibility:
    """Every score must equal sum(feature_contributions) and be deterministic."""

    def test_score_equals_sum_of_contributions(self):
        result = _score(
            sentiment_label="NEGATIVE", sentiment_score=0.92,
            emotion_label="anger", emotion_score=0.85,
            text="Urgent billing fraud! I demand a refund immediately.",
            category="billing", issue="fraud",
        )
        assert result["priority_score"] == sum(result["feature_contributions"].values())

    def test_identical_inputs_produce_identical_scores(self):
        kwargs = dict(
            sentiment_label="NEGATIVE", sentiment_score=0.70,
            emotion_label="sadness", emotion_score=0.60,
            text="My order was never delivered.",
            category="delivery", issue="general",
        )
        a = _score(**kwargs)
        b = _score(**kwargs)
        assert a["priority_score"] == b["priority_score"]
        assert a["priority"] == b["priority"]
        assert a["feature_contributions"] == b["feature_contributions"]


# ── Every result has reasons ────────────────────────────────────────────────

class TestReasonsPresent:

    def test_critical_has_reasons(self):
        r = _score(text="This is fraud and I will sue!", sentiment_label="NEGATIVE", sentiment_score=0.9)
        assert len(r["priority_reasons"]) > 0

    def test_high_has_reasons(self):
        r = _score(sentiment_label="NEGATIVE", sentiment_score=0.9, emotion_label="anger", emotion_score=0.9,
                   text="Urgent billing issue", category="billing")
        assert len(r["priority_reasons"]) > 0

    def test_low_has_reasons(self):
        r = _score()
        assert r["priority_reasons"] == ["No elevated priority signals detected"]

    def test_feature_contributions_dict_present(self):
        r = _score()
        assert isinstance(r["feature_contributions"], dict)

    def test_thresholds_included_in_output(self):
        r = _score()
        assert "thresholds" in r
        assert r["thresholds"] == PRIORITY_THRESHOLDS


# ── Emergency keyword override ──────────────────────────────────────────────

class TestEmergencyOverride:

    def test_fraud_is_critical(self):
        r = _score(text="This is fraud!", sentiment_label="NEGATIVE", sentiment_score=0.9)
        assert r["priority"] == "CRITICAL"
        assert "fraud" in r["emergency_indicators"]

    def test_lawsuit_is_critical(self):
        r = _score(text="I will file a lawsuit against you.", sentiment_label="NEGATIVE", sentiment_score=0.8)
        assert r["priority"] == "CRITICAL"
        assert "lawsuit" in r["emergency_indicators"]

    def test_data_loss_is_critical(self):
        r = _score(text="There has been a data loss event.", sentiment_label="NEGATIVE", sentiment_score=0.7)
        assert r["priority"] == "CRITICAL"

    def test_security_breach_is_critical(self):
        r = _score(text="We discovered a security breach.", sentiment_label="NEGATIVE", sentiment_score=0.9)
        assert r["priority"] == "CRITICAL"

    def test_emergency_always_contributes_60(self):
        r = _score(text="fraud detected", sentiment_label="NEUTRAL", sentiment_score=0.5)
        assert r["feature_contributions"]["emergency"] == 60

    def test_emergency_overrides_even_positive_sentiment(self):
        r = _score(text="Thanks, but there was fraud.", sentiment_label="POSITIVE", sentiment_score=0.9)
        assert r["priority"] == "CRITICAL"


# ── Urgency keywords (not emergency) ───────────────────────────────────────

class TestUrgencyKeywords:

    def test_urgent_adds_18(self):
        r = _score(text="This is urgent please help.", sentiment_label="NEUTRAL", sentiment_score=0.5)
        assert r["feature_contributions"].get("urgency") == 18

    def test_immediately_adds_18(self):
        r = _score(text="Fix this immediately.", sentiment_label="NEUTRAL", sentiment_score=0.5)
        assert r["feature_contributions"].get("urgency") == 18

    def test_urgency_shows_in_indicators(self):
        r = _score(text="I am blocked, cannot access my account. This is urgent.")
        assert "urgent" in r["urgency_indicators"]
        assert "blocked" in r["urgency_indicators"]


# ── Sentiment tiers ─────────────────────────────────────────────────────────

class TestSentimentTiers:

    def test_strong_negative_contributes_20(self):
        val, _ = _sentiment_contribution("NEGATIVE", 0.92)
        assert val == 20

    def test_moderate_negative_contributes_14(self):
        val, _ = _sentiment_contribution("NEGATIVE", 0.65)
        assert val == 14

    def test_weak_negative_contributes_8(self):
        val, _ = _sentiment_contribution("NEGATIVE", 0.50)
        assert val == 8

    def test_positive_subtracts_4(self):
        val, _ = _sentiment_contribution("POSITIVE", 0.9)
        assert val == -4

    def test_neutral_contributes_0(self):
        val, _ = _sentiment_contribution("NEUTRAL", 0.5)
        assert val == 0


# ── Emotion tiers ───────────────────────────────────────────────────────────

class TestEmotionTiers:

    def test_anger_high(self):
        val, reason = _emotion_contribution("anger", 0.85)
        assert val == 18
        assert "Anger" in reason

    def test_anger_low(self):
        val, _ = _emotion_contribution("anger", 0.50)
        assert val == 12

    def test_fear_high(self):
        val, _ = _emotion_contribution("fear", 0.80)
        assert val == 18

    def test_fear_low(self):
        val, _ = _emotion_contribution("fear", 0.55)
        assert val == 12

    def test_disgust_always_14(self):
        val, _ = _emotion_contribution("disgust", 0.3)
        assert val == 14

    def test_sadness_high(self):
        val, _ = _emotion_contribution("sadness", 0.80)
        assert val == 10

    def test_sadness_low(self):
        val, _ = _emotion_contribution("sadness", 0.50)
        assert val == 7

    def test_surprise_high(self):
        val, _ = _emotion_contribution("surprise", 0.80)
        assert val == 5

    def test_neutral_contributes_0(self):
        val, _ = _emotion_contribution("neutral", 0.9)
        assert val == 0

    def test_joy_contributes_0(self):
        val, _ = _emotion_contribution("joy", 0.9)
        assert val == 0


# ── Boundary conditions ────────────────────────────────────────────────────

class TestBoundaryConditions:

    def test_score_0_is_low(self):
        r = _score()
        assert r["priority"] == "LOW"
        assert r["priority_score"] < PRIORITY_THRESHOLDS["MEDIUM"]

    def test_score_at_medium_boundary(self):
        # NEGATIVE sentiment 0.9 → 20pts → exactly at MEDIUM boundary
        r = _score(sentiment_label="NEGATIVE", sentiment_score=0.9)
        assert r["priority_score"] == 20
        assert r["priority"] == "MEDIUM"

    def test_score_just_below_medium(self):
        # NEGATIVE sentiment 0.65 → 14pts, category other → 0 = 14 → LOW
        r = _score(sentiment_label="NEGATIVE", sentiment_score=0.65)
        assert r["priority_score"] < 20
        assert r["priority"] == "LOW"

    def test_score_at_high_boundary(self):
        # NEGATIVE 0.9 (20) + anger high (18) + urgency (18) = 56 → HIGH
        r = _score(
            sentiment_label="NEGATIVE", sentiment_score=0.9,
            emotion_label="anger", emotion_score=0.9,
            text="This is urgent!"
        )
        assert r["priority"] == "HIGH"
        assert r["priority_score"] >= PRIORITY_THRESHOLDS["HIGH"]

    def test_score_at_critical_boundary_no_emergency(self):
        # Needs score >= 75 without emergency
        r = _score(
            sentiment_label="NEGATIVE", sentiment_score=0.95,
            emotion_label="anger", emotion_score=0.95,
            text="This is urgent and blocked! Fix it today!",
            category="billing", issue="fraud",
            history={"previous_complaints": 5, "unresolved_complaints": 3},
        )
        assert r["priority"] == "CRITICAL"
        assert r["priority_score"] >= 75

    def test_score_capped_at_100(self):
        r = _score(
            sentiment_label="NEGATIVE", sentiment_score=0.99,
            emotion_label="anger", emotion_score=0.99,
            text="FRAUD! LAWSUIT! URGENT! IMMEDIATELY! This is an emergency and data loss!",
            category="billing", issue="fraud",
            history={"previous_complaints": 10, "unresolved_complaints": 5},
        )
        assert r["priority_score"] <= 100

    def test_score_floored_at_0(self):
        r = _score(sentiment_label="POSITIVE", sentiment_score=0.99, text="Thank you for everything.")
        assert r["priority_score"] >= 0


# ── SLA escalation ──────────────────────────────────────────────────────────

class TestSLAEscalation:

    def test_no_sla_for_resolved(self):
        old = (datetime.now(timezone.utc) - timedelta(hours=49)).isoformat()
        val, reason = _sla_contribution(old, "resolved")
        assert val == 0
        assert reason is None

    def test_no_sla_without_timestamp(self):
        val, reason = _sla_contribution(None, "pending")
        assert val == 0

    def test_24h_sla_breach(self):
        ts = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        val, reason = _sla_contribution(ts, "pending")
        assert val == 15
        assert "24-hour" in reason

    def test_48h_sla_breach(self):
        ts = (datetime.now(timezone.utc) - timedelta(hours=49)).isoformat()
        val, reason = _sla_contribution(ts, "pending_admin")
        assert val == 25
        assert "48-hour" in reason

    def test_sla_escalation_raises_priority(self):
        old = (datetime.now(timezone.utc) - timedelta(hours=49)).isoformat()
        r_no_sla = _score(sentiment_label="NEGATIVE", sentiment_score=0.65)
        r_with_sla = _score(
            sentiment_label="NEGATIVE", sentiment_score=0.65,
            created_at=old, status="pending",
        )
        assert r_with_sla["priority_score"] > r_no_sla["priority_score"]
        assert r_with_sla["feature_contributions"]["sla"] == 25

    def test_fresh_complaint_no_sla(self):
        ts = datetime.now(timezone.utc).isoformat()
        r = _score(created_at=ts, status="pending")
        assert "sla" not in r["feature_contributions"]

    def test_sla_with_pending_admin_status(self):
        ts = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        val, _ = _sla_contribution(ts, "pending_admin")
        assert val == 15

    def test_sla_with_in_progress_status(self):
        ts = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        val, _ = _sla_contribution(ts, "in_progress")
        assert val == 15


# ── Repeated complaints ────────────────────────────────────────────────────

class TestRepeatedComplaints:

    def test_3_previous_adds_7(self):
        contribs = _history_contributions({"previous_complaints": 3, "unresolved_complaints": 0})
        assert len(contribs) == 1
        assert contribs[0] == ("history", 7, "Repeated customer contact (3 previous complaints)")

    def test_2_unresolved_adds_5(self):
        contribs = _history_contributions({"previous_complaints": 0, "unresolved_complaints": 2})
        assert len(contribs) == 1
        assert contribs[0][1] == 5

    def test_both_history_signals(self):
        contribs = _history_contributions({"previous_complaints": 5, "unresolved_complaints": 3})
        assert len(contribs) == 2
        total = sum(c[1] for c in contribs)
        assert total == 12

    def test_no_history(self):
        contribs = _history_contributions(None)
        assert contribs == []

    def test_history_raises_priority(self):
        r_no_hist = _score(sentiment_label="NEGATIVE", sentiment_score=0.9)
        r_with_hist = _score(
            sentiment_label="NEGATIVE", sentiment_score=0.9,
            history={"previous_complaints": 5, "unresolved_complaints": 3},
        )
        assert r_with_hist["priority_score"] > r_no_hist["priority_score"]


# ── Issue & category scoring ───────────────────────────────────────────────

class TestIssueCategoryScoring:

    def test_fraud_issue_adds_15(self):
        val, _ = _issue_category_contribution("billing", "fraud")
        assert val == 15

    def test_billing_category_adds_5(self):
        val, _ = _issue_category_contribution("billing", "general")
        assert val == 5

    def test_technical_category_adds_5(self):
        val, _ = _issue_category_contribution("technical", "general")
        assert val == 5

    def test_other_adds_0(self):
        val, _ = _issue_category_contribution("other", "general")
        assert val == 0


# ── Length contribution ─────────────────────────────────────────────────────

class TestLengthContribution:

    def test_short_text_no_contribution(self):
        val, _ = _length_contribution("A short complaint.")
        assert val == 0

    def test_long_text_adds_5(self):
        long_text = " ".join(["word"] * 101)
        val, reason = _length_contribution(long_text)
        assert val == 5
        assert "101 words" in reason


# ── Fix: valid complaints incorrectly classified as LOW ─────────────────────

class TestValidComplaintsNotLow:

    def test_negative_billing_complaint_is_not_low(self):
        """A billing complaint with clear NEGATIVE sentiment should be MEDIUM+."""
        r = _score(
            sentiment_label="NEGATIVE", sentiment_score=0.85,
            emotion_label="anger", emotion_score=0.6,
            text="I was charged twice for my subscription and no one is helping.",
            category="billing",
        )
        assert r["priority"] != "LOW", (
            f"Valid billing complaint got LOW with score {r['priority_score']}. "
            f"Contributions: {r['feature_contributions']}"
        )

    def test_negative_technical_complaint_is_not_low(self):
        """A technical complaint with negative sentiment and fear should be MEDIUM+."""
        r = _score(
            sentiment_label="NEGATIVE", sentiment_score=0.75,
            emotion_label="fear", emotion_score=0.65,
            text="The app keeps crashing and I am worried about data loss.",
            category="technical",
        )
        assert r["priority"] != "LOW"

    def test_negative_delivery_complaint_is_not_low(self):
        r = _score(
            sentiment_label="NEGATIVE", sentiment_score=0.70,
            emotion_label="sadness", emotion_score=0.60,
            text="My package has been lost and I am very upset about it.",
            category="delivery",
        )
        assert r["priority"] != "LOW"


# ── Category inference (preserved from Phase 2) ────────────────────────────

class TestCategoryInference:

    def test_refund_detected(self):
        cat, issue, indicators = infer_category_and_issue(
            "I was charged twice and need a refund for my subscription."
        )
        assert (cat, issue) == ("refund", "billing")
        assert "refund" in indicators

    def test_unknown_text(self):
        assert infer_category_and_issue("Could you please take a look at this?") == ("other", "general", [])


# ── detect_urgency ──────────────────────────────────────────────────────────

class TestDetectUrgency:

    def test_detects_fraud(self):
        r = detect_urgency("This is fraud!")
        assert "fraud" in r["emergency"]

    def test_detects_urgent(self):
        r = detect_urgency("This is urgent please")
        assert "urgent" in r["urgency"]

    def test_empty_text(self):
        r = detect_urgency("")
        assert r["emergency"] == []
        assert r["urgency"] == []

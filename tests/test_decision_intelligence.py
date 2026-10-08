"""Phase 6 — Decision Intelligence & Human Feedback tests.

Covers:
  - State builder produces all required fields
  - Every action in ACTIONS can be recommended
  - Emergency / CRITICAL → ESCALATE
  - SLA breach → INCREASE_PRIORITY
  - LOW + auto-resolvable → REPLY
  - Vague complaint → REQUEST_INFORMATION
  - HIGH → ASSIGN with team metadata
  - Resolved → CLOSE
  - Confidence thresholds trigger human_review_required
  - Feedback store: record, update, acceptance rate
  - Synthetic flag is preserved
  - Dataset schema integrity
"""

import pytest

from app.services.decision_intelligence import (
    ACTIONS,
    build_state,
    recommend_action,
)
from app.services.feedback_store import (
    record_recommendation,
    record_human_action,
    record_outcome,
    get_transition,
    list_transitions,
    acceptance_rate,
    clear_store,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_feedback_log():
    """Wipe the feedback log before and after each test."""
    clear_store()
    yield
    clear_store()


def _state(**overrides):
    return build_state(**overrides)


# ── State builder ───────────────────────────────────────────────────────────

class TestBuildState:

    def test_returns_all_required_keys(self):
        s = _state()
        required = {
            "category", "issue", "sentiment_label", "sentiment_score",
            "emotion_label", "emotion_score", "priority", "priority_score",
            "urgency_indicators", "emergency_indicators", "sla_hours",
            "status", "policy_decision", "policy_confidence",
            "auto_resolvable", "previous_complaints", "unresolved_complaints",
            "response_confidence",
        }
        assert required.issubset(set(s.keys()))

    def test_normalises_labels(self):
        s = _state(priority="low", sentiment_label="negative", emotion_label="Anger")
        assert s["priority"] == "LOW"
        assert s["sentiment_label"] == "NEGATIVE"
        assert s["emotion_label"] == "anger"


# ── Recommendation rules ───────────────────────────────────────────────────

class TestRecommendAction:

    def test_emergency_escalates(self):
        s = _state(emergency_indicators=["fraud"])
        r = recommend_action(s)
        assert r["recommended_action"] == "ESCALATE"
        assert r["confidence"] >= 0.90

    def test_critical_priority_escalates(self):
        s = _state(priority="CRITICAL", priority_score=80)
        r = recommend_action(s)
        assert r["recommended_action"] == "ESCALATE"

    def test_critical_with_sla_mentions_breach(self):
        s = _state(priority="CRITICAL", sla_hours=30)
        r = recommend_action(s)
        assert r["recommended_action"] == "ESCALATE"
        assert any("SLA" in reason for reason in r["reasons"])

    def test_high_repeat_offender_escalates(self):
        s = _state(priority="HIGH", previous_complaints=5)
        r = recommend_action(s)
        assert r["recommended_action"] == "ESCALATE"

    def test_48h_sla_breach_increases_priority(self):
        s = _state(priority="HIGH", sla_hours=50, status="pending")
        r = recommend_action(s)
        assert r["recommended_action"] == "INCREASE_PRIORITY"

    def test_24h_sla_breach_low_increases_priority(self):
        s = _state(priority="LOW", sla_hours=26, status="pending")
        r = recommend_action(s)
        assert r["recommended_action"] == "INCREASE_PRIORITY"

    def test_low_auto_resolvable_replies(self):
        s = _state(priority="LOW", auto_resolvable=True, response_confidence=0.5)
        r = recommend_action(s)
        assert r["recommended_action"] == "REPLY"

    def test_vague_requests_information(self):
        s = _state(
            priority="LOW", category="other", issue="general",
            sentiment_label="NEUTRAL",
        )
        r = recommend_action(s)
        assert r["recommended_action"] == "REQUEST_INFORMATION"

    def test_high_priority_assigns(self):
        s = _state(priority="HIGH", category="billing")
        r = recommend_action(s)
        assert r["recommended_action"] == "ASSIGN"
        assert r["metadata"]["suggested_team"] == "billing"

    def test_high_technical_assigns_engineering(self):
        s = _state(priority="HIGH", category="technical")
        r = recommend_action(s)
        assert r["metadata"]["suggested_team"] == "engineering"

    def test_medium_with_policy_replies(self):
        s = _state(
            priority="MEDIUM", policy_decision="POLICY_FOUND",
            response_confidence=0.35, category="billing",
        )
        r = recommend_action(s)
        assert r["recommended_action"] == "REPLY"

    def test_medium_general_assigns(self):
        s = _state(priority="MEDIUM", category="delivery")
        r = recommend_action(s)
        assert r["recommended_action"] == "ASSIGN"
        assert r["metadata"]["suggested_team"] == "logistics"

    def test_resolved_closes(self):
        s = _state(status="resolved")
        r = recommend_action(s)
        assert r["recommended_action"] == "CLOSE"

    def test_auto_replied_closes(self):
        s = _state(status="auto_replied")
        r = recommend_action(s)
        assert r["recommended_action"] == "CLOSE"

    def test_low_confidence_triggers_human_review(self):
        s = _state(priority="LOW", auto_resolvable=False, category="other", issue="general",
                    sentiment_label="POSITIVE")
        r = recommend_action(s)
        # Fallback path has confidence 0.40, which is < 0.70
        assert r["human_review_required"] is True

    def test_high_confidence_does_not_require_review(self):
        s = _state(emergency_indicators=["fraud"])
        r = recommend_action(s)
        assert r["human_review_required"] is False

    def test_result_contains_state_snapshot(self):
        s = _state(priority="LOW")
        r = recommend_action(s)
        assert r["state_snapshot"] == s

    def test_result_contains_timestamp(self):
        s = _state()
        r = recommend_action(s)
        assert "timestamp" in r

    def test_all_actions_reachable(self):
        """Every action in the action space should be reachable by at least one state."""
        triggered = set()
        states = [
            _state(status="resolved"),                                                      # CLOSE
            _state(emergency_indicators=["fraud"]),                                         # ESCALATE
            _state(priority="HIGH", category="billing"),                                    # ASSIGN
            _state(priority="LOW", category="other", issue="general", sentiment_label="NEUTRAL"),  # REQUEST_INFORMATION
            _state(priority="LOW", sla_hours=26),                                           # INCREASE_PRIORITY
            _state(priority="LOW", auto_resolvable=True, response_confidence=0.5),          # REPLY
        ]
        for s in states:
            r = recommend_action(s)
            triggered.add(r["recommended_action"])
        assert triggered == set(ACTIONS), f"Unreachable actions: {set(ACTIONS) - triggered}"


# ── Feedback store ──────────────────────────────────────────────────────────

class TestFeedbackStore:

    def test_record_and_retrieve(self):
        s = _state(priority="HIGH")
        tid = record_recommendation("C-001", s, "ESCALATE", 0.92)
        row = get_transition(tid)
        assert row is not None
        assert row["complaint_id"] == "C-001"
        assert row["recommended_action"] == "ESCALATE"
        assert row["human_action"] is None

    def test_record_human_action(self):
        s = _state(priority="HIGH")
        tid = record_recommendation("C-002", s, "ESCALATE", 0.92)
        record_human_action(tid, "ASSIGN")
        row = get_transition(tid)
        assert row["human_action"] == "ASSIGN"
        assert row["recommendation_accepted"] is False

    def test_accepted_recommendation(self):
        s = _state(priority="HIGH")
        tid = record_recommendation("C-003", s, "ESCALATE", 0.92)
        record_human_action(tid, "ESCALATE")
        row = get_transition(tid)
        assert row["recommendation_accepted"] is True

    def test_record_outcome(self):
        s = _state(priority="LOW")
        tid = record_recommendation("C-004", s, "REPLY", 0.85)
        record_human_action(tid, "REPLY")
        record_outcome(tid, "resolved")
        row = get_transition(tid)
        assert row["outcome"] == "resolved"
        assert row["outcome_timestamp"] is not None

    def test_acceptance_rate(self):
        s = _state()
        t1 = record_recommendation("C-010", s, "REPLY", 0.80)
        t2 = record_recommendation("C-011", s, "ESCALATE", 0.90)
        t3 = record_recommendation("C-012", s, "ASSIGN", 0.75)
        record_human_action(t1, "REPLY")      # accepted
        record_human_action(t2, "ESCALATE")   # accepted
        record_human_action(t3, "REPLY")      # corrected
        stats = acceptance_rate()
        assert stats["total"] == 3
        assert stats["accepted"] == 2
        assert stats["corrected"] == 1
        assert abs(stats["rate"] - 0.667) < 0.01

    def test_synthetic_flag_preserved(self):
        s = _state()
        tid = record_recommendation("SYN-001", s, "REPLY", 0.50, is_synthetic=True)
        row = get_transition(tid)
        assert row["is_synthetic"] is True

    def test_list_only_real(self):
        s = _state()
        record_recommendation("REAL-001", s, "REPLY", 0.80, is_synthetic=False)
        record_recommendation("SYN-002", s, "REPLY", 0.50, is_synthetic=True)
        real = list_transitions(only_real=True)
        assert len(real) == 1
        assert real[0]["complaint_id"] == "REAL-001"

    def test_list_by_complaint(self):
        s = _state()
        record_recommendation("C-020", s, "REPLY", 0.80)
        record_recommendation("C-021", s, "ESCALATE", 0.90)
        rows = list_transitions(complaint_id="C-020")
        assert len(rows) == 1


# ── Dataset schema contract ────────────────────────────────────────────────

class TestDatasetSchema:
    """Every feedback row must contain the fields needed to later train
    an offline-RL or contextual-bandit model."""

    REQUIRED_FIELDS = {
        "id", "complaint_id", "timestamp", "state",
        "recommended_action", "recommendation_confidence",
        "human_action", "recommendation_accepted",
        "outcome", "outcome_timestamp", "is_synthetic",
    }

    def test_schema_completeness(self):
        s = _state(priority="HIGH")
        tid = record_recommendation("SCHEMA-001", s, "ESCALATE", 0.90)
        row = get_transition(tid)
        missing = self.REQUIRED_FIELDS - set(row.keys())
        assert not missing, f"Missing fields in feedback schema: {missing}"

    def test_state_within_row_has_all_fields(self):
        s = _state()
        tid = record_recommendation("SCHEMA-002", s, "REPLY", 0.80)
        row = get_transition(tid)
        state_keys = set(row["state"].keys())
        expected = set(s.keys())
        assert expected.issubset(state_keys)

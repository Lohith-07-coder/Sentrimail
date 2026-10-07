"""
Phase 2 — Complaint Intelligence Tests
---------------------------------------
Validates the enriched analysis pipeline:
  - Output schema (new fields: language, translated_text, intent, entities, urgency)
  - Entity extraction (email, phone, money, reference_id)
  - Intent inference (keyword-driven heuristic)
  - Language detection fallback
  - Backward compatibility: all original /api/analyze fields still present
  - Confidence values present where supported
  - No forced classifications on ambiguous input
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.services.ai_service import (
    analyze_complaint,
    _extract_entities,
    _infer_intent,
    _detect_and_translate,
    _rule_based_sentiment,
    _rule_based_emotion,
    infer_category_and_issue,
)


# ── Output schema contract ─────────────────────────────────────────────────

# These are the keys that /api/analyze returned BEFORE Phase 2.
# They must remain present so no downstream consumer breaks.
LEGACY_KEYS = {
    "sentiment_label", "sentiment_score",
    "emotion_label", "emotion_score",
    "priority", "priority_score", "priority_description",
    "priority_color",
    "category", "issue", "category_indicators",
    "root_cause_summary",
    "auto_resolvable", "auto_resolution_reason",
    "user_auto_response",
    "admin_suggested_response", "ai_suggested_response",
    "model_used", "response_source", "reference_id",
}

# New fields added by Phase 2.
PHASE2_KEYS = {
    "language", "translated_text",
    "intent", "intent_confidence",
    "entities", "urgency",
}


class TestAnalyzeComplaintSchema:
    """Ensure every call to analyze_complaint returns a superset of the legacy schema."""

    def test_legacy_keys_present(self):
        result = analyze_complaint("My order was never delivered.", category="delivery")
        missing = LEGACY_KEYS - set(result.keys())
        assert not missing, f"Missing legacy keys: {missing}"

    def test_phase2_keys_present(self):
        result = analyze_complaint("My order was never delivered.", category="delivery")
        missing = PHASE2_KEYS - set(result.keys())
        assert not missing, f"Missing Phase 2 keys: {missing}"

    def test_language_defaults_to_en_for_english_text(self):
        result = analyze_complaint("I need help with my account.")
        assert result["language"] == "en"
        assert result["translated_text"] == ""

    def test_sentiment_has_confidence(self):
        result = analyze_complaint("This is absolutely terrible service!")
        assert isinstance(result["sentiment_score"], float)
        assert 0.0 <= result["sentiment_score"] <= 1.0

    def test_emotion_has_confidence(self):
        result = analyze_complaint("I am furious about this!")
        assert isinstance(result["emotion_score"], float)
        assert 0.0 <= result["emotion_score"] <= 1.0

    def test_intent_has_confidence(self):
        result = analyze_complaint("I want a refund for order #123")
        assert isinstance(result["intent_confidence"], float)
        assert 0.0 <= result["intent_confidence"] <= 1.0

    def test_entities_is_list(self):
        result = analyze_complaint("Contact me at user@example.com")
        assert isinstance(result["entities"], list)

    def test_urgency_is_list(self):
        result = analyze_complaint("This is urgent, fix it immediately!")
        assert isinstance(result["urgency"], list)


# ── Entity extraction ──────────────────────────────────────────────────────

class TestEntityExtraction:

    def test_extracts_email(self):
        entities = _extract_entities("Please reach me at test@sentrimail.com for updates.")
        emails = [e for e in entities if e["type"] == "email"]
        assert len(emails) == 1
        assert emails[0]["value"] == "test@sentrimail.com"

    def test_extracts_phone(self):
        entities = _extract_entities("Call me at 555-123-4567 or (800) 555-0199.")
        phones = [e for e in entities if e["type"] == "phone"]
        assert len(phones) == 2

    def test_extracts_money(self):
        entities = _extract_entities("I was charged $499.99 instead of $199.")
        money = [e for e in entities if e["type"] == "money"]
        assert len(money) == 2
        values = {m["value"] for m in money}
        assert "$499.99" in values
        assert "$199" in values

    def test_extracts_reference_id(self):
        entities = _extract_entities("My order #ABC-1234 was lost in transit.")
        refs = [e for e in entities if e["type"] == "reference_id"]
        assert len(refs) == 1
        assert refs[0]["value"] == "ABC-1234"

    def test_extracts_multiple_entity_types(self):
        text = "Order #REF-999 charged $50. Email me at a@b.com or call 555-000-1111."
        entities = _extract_entities(text)
        types = {e["type"] for e in entities}
        assert types == {"reference_id", "money", "email", "phone"}

    def test_no_entities_in_plain_text(self):
        entities = _extract_entities("Everything is fine, no issues at all.")
        assert entities == []


# ── Intent inference ────────────────────────────────────────────────────────

class TestIntentInference:

    def test_refund_intent(self):
        result = _infer_intent("billing", "billing", "I want a refund for the duplicate charge.")
        assert result["label"] == "request_refund"
        assert result["confidence"] >= 0.8

    def test_cancel_subscription_intent(self):
        result = _infer_intent("billing", "billing", "I need to cancel my subscription immediately.")
        assert result["label"] == "cancel_subscription"

    def test_auth_intent(self):
        result = _infer_intent("technical", "auth", "I can't log in to my account.")
        assert result["label"] == "recover_account"

    def test_technical_intent(self):
        result = _infer_intent("technical", "technical", "The app keeps crashing on startup.")
        assert result["label"] == "report_technical_issue"

    def test_delivery_intent(self):
        result = _infer_intent("delivery", "delivery", "Where is my package?")
        assert result["label"] == "track_package"

    def test_billing_intent(self):
        result = _infer_intent("billing", "billing", "I was overcharged on my last invoice.")
        assert result["label"] == "dispute_charge"

    def test_general_fallback_intent(self):
        result = _infer_intent("other", "general", "I have a question about your service.")
        assert result["label"] == "general_inquiry"
        assert result["confidence"] == 0.5

    def test_confidence_never_exceeds_one(self):
        for cat in ("billing", "technical", "delivery", "other"):
            result = _infer_intent(cat, "general", "some text")
            assert 0.0 <= result["confidence"] <= 1.0


# ── Language detection ──────────────────────────────────────────────────────

class TestLanguageDetection:

    def test_english_detected(self):
        lang, translated = _detect_and_translate("My server crashed and lost all data.")
        assert lang == "en"
        assert translated == "My server crashed and lost all data."

    def test_empty_text_defaults_to_en(self):
        lang, translated = _detect_and_translate("   ")
        assert lang == "en"


# ── Rule-based fallback consistency ─────────────────────────────────────────

class TestRuleBasedFallbacks:

    def test_negative_sentiment_on_angry_text(self):
        result = _rule_based_sentiment("This is terrible and disgusting service, I am furious!")
        assert result["label"] == "NEGATIVE"
        assert result["score"] > 0.5

    def test_positive_sentiment_on_happy_text(self):
        result = _rule_based_sentiment("Thanks for the great help, I appreciate it!")
        assert result["label"] == "POSITIVE"

    def test_neutral_sentiment_on_bland_text(self):
        result = _rule_based_sentiment("I placed an order yesterday.")
        assert result["label"] == "NEUTRAL"

    def test_anger_emotion_detected(self):
        result = _rule_based_emotion("I am angry and furious about this!")
        assert result["label"] == "anger"

    def test_fear_emotion_detected(self):
        result = _rule_based_emotion("I am terrified about potential data loss and corruption.")
        assert result["label"] == "fear"

    def test_neutral_emotion_when_no_patterns(self):
        result = _rule_based_emotion("I placed an order yesterday.")
        assert result["label"] == "neutral"


# ── Category & issue inference ──────────────────────────────────────────────

class TestCategoryInference:

    def test_refund_detected(self):
        cat, issue, terms = infer_category_and_issue("I need a refund for the overcharge.")
        assert cat == "refund"
        assert "refund" in terms

    def test_billing_detected(self):
        cat, issue, terms = infer_category_and_issue("I was charged twice on my invoice.")
        assert cat == "billing"

    def test_technical_detected(self):
        cat, issue, terms = infer_category_and_issue("The server is down and showing errors.")
        assert cat == "technical"

    def test_delivery_detected(self):
        cat, issue, terms = infer_category_and_issue("My package was not delivered on time.")
        assert cat == "delivery"

    def test_explicit_category_honored(self):
        cat, issue, _ = infer_category_and_issue("Something happened.", category="product")
        assert cat == "product"

    def test_ambiguous_text_returns_other(self):
        cat, issue, terms = infer_category_and_issue("Hello, I have a question.")
        assert cat == "other"
        assert terms == []


# ── Urgency field in output ─────────────────────────────────────────────────

class TestUrgencyOutput:

    def test_urgency_populated_for_urgent_text(self):
        result = analyze_complaint("This is urgent! Fix it immediately, I am blocked!")
        assert isinstance(result["urgency"], list)
        assert len(result["urgency"]) > 0

    def test_urgency_empty_for_calm_text(self):
        result = analyze_complaint("Just checking in on my order status.")
        assert result["urgency"] == []


# ── End-to-end integration ──────────────────────────────────────────────────

class TestEndToEndAnalysis:

    def test_full_pipeline_billing_refund(self):
        result = analyze_complaint(
            "I was charged $499 on order #REF-8812. Contact me at alice@example.com. I want a refund immediately!",
            category="billing",
            username="alice",
        )
        assert result["language"] == "en"
        assert result["category"] == "billing"
        assert result["intent"] == "request_refund"
        assert result["intent_confidence"] >= 0.8
        # Entities
        types = {e["type"] for e in result["entities"]}
        assert "email" in types
        assert "money" in types
        assert "reference_id" in types
        # Urgency
        assert "immediately" in result["urgency"]
        # Confidence fields
        assert isinstance(result["sentiment_score"], float)
        assert isinstance(result["emotion_score"], float)

    def test_full_pipeline_technical_crash(self):
        result = analyze_complaint(
            "App crashes every time I open settings. This is a critical bug causing data loss!",
            category="technical",
        )
        assert result["category"] == "technical"
        assert result["intent"] == "report_technical_issue"
        assert result["priority"] in {"HIGH", "CRITICAL"}

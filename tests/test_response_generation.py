import pytest
from app.services.ai_service import analyze_complaint

def test_card_location_response():
    res = analyze_complaint(text="How do I locate my card?", category="card_arrival", username="alice")
    assert res["response_source"] in ["category_template", "local_llm", "tfidf"]
    assert "locate your card" in res["ai_suggested_response"].lower() or "track its delivery" in res["ai_suggested_response"].lower()
    assert "reviewing your complaint" not in res["ai_suggested_response"].lower()

def test_duplicate_charge_response():
    res = analyze_complaint(text="I was charged twice for the same transaction.", category="billing", username="bob")
    assert res["response_source"] in ["category_template", "local_llm", "tfidf"]
    assert "billing concern" in res["ai_suggested_response"].lower() or "verify the charge" in res["ai_suggested_response"].lower()
    assert "reviewing your complaint" not in res["ai_suggested_response"].lower()

def test_hacked_account_response():
    res = analyze_complaint(text="My account was hacked.", category="other", username="charlie")
    assert res["response_source"] in ["category_template", "local_llm", "tfidf"]
    assert "secure your account" in res["ai_suggested_response"].lower() or "unauthorized" in res["ai_suggested_response"].lower() or "access" in res["ai_suggested_response"].lower()
    assert "reviewing your complaint" not in res["ai_suggested_response"].lower()
    assert res["auto_resolvable"] is False  # High/critical

def test_refund_status_response():
    res = analyze_complaint(text="I want to know the status of my refund.", category="refund", username="diana")
    assert res["response_source"] in ["category_template", "local_llm", "tfidf"]
    assert "billing concern" in res["ai_suggested_response"].lower() or "refund" in res["ai_suggested_response"].lower()
    assert "reviewing your complaint" not in res["ai_suggested_response"].lower()

def test_unknown_complaint_response():
    res = analyze_complaint(text="The sky is blue today.", category="other", username="eve")
    # This might fall back to generic if local_llm is off
    assert res["ai_suggested_response"] != ""
    # Should use local_llm or generic fallback
    assert res["response_source"] in ["local_llm", "generic_fallback"]

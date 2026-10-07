import pytest
from app.services.response_intelligence import generate_grounded_response, _retrieve_candidates

def test_retrieval_relevance():
    candidates = _retrieve_candidates("I cannot log into my account", "technical", "LOW")
    assert len(candidates) > 0
    assert candidates[0]['score'] > 0

def test_response_grounding():
    resp = generate_grounded_response("I have an internet issue", "user1", "technical", "LOW", "en")
    assert resp["confidence"] > 0
    assert resp["source"] in ["semantic", "tfidf", "template"]

def test_fallback_behavior():
    resp = generate_grounded_response("Something completely random that has no match anywhere", "user1", "other", "LOW", "en")
    if resp["confidence"] < 0.2:
        assert resp["human_review_required"] == True

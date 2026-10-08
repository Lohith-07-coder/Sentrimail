import pytest
import app.services.policy_intelligence as pi
from app.services.policy_intelligence import ingest_policies, retrieve_policy, get_policy_decision

def test_ingestion():
    ingest_policies()
    assert len(pi._policy_store) > 0
    assert any(p["source"] == "return_policy.md" for p in pi._policy_store)
    assert "text" in pi._policy_store[0]

def test_retrieval():
    ingest_policies()
    results = retrieve_policy("Can I get a refund for software?")
    assert len(results) > 0
    assert "evidence" in results[0]
    assert "score" in results[0]

def test_policy_decision():
    ingest_policies()
    
    # High confidence match
    res = get_policy_decision("What is the return window for damaged items?")
    assert res["decision"] == "POLICY_FOUND"
    assert "evidence" in res
    assert len(res["evidence"]) > 0
    
    # Low confidence match (abstention)
    res2 = get_policy_decision("Do you cover alien abductions?")
    assert res2["decision"] == "ABSTAIN"
    assert res2["confidence"] < 0.1 # Using 0.1 threshold in code

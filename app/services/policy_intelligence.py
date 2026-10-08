import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any
from numpy import dot
from numpy.linalg import norm
import math

logger = logging.getLogger(__name__)

POLICIES_DIR = Path(__file__).resolve().parents[2] / "data" / "policies"

_semantic_model = None
_policy_store = []
_embeddings = []
_idf = {}

def _tokenize(text: str) -> List[str]:
    return [tok for tok in re.findall(r"[a-z0-9']+", text.lower()) if len(tok) > 1]

def _load_model():
    global _semantic_model
    if _semantic_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Could not load semantic model for policies: {e}")

def _cosine_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b)) if norm(a) > 0 and norm(b) > 0 else 0.0

def ingest_policies():
    global _policy_store, _embeddings, _idf
    _policy_store = []
    _embeddings = []
    
    if not POLICIES_DIR.exists():
        POLICIES_DIR.mkdir(parents=True, exist_ok=True)
        
    for file_path in POLICIES_DIR.glob("**/*"):
        if file_path.is_file() and file_path.suffix in [".txt", ".md"]:
            content = file_path.read_text(encoding="utf-8")
            chunks = [c.strip() for c in re.split(r'\n+', content) if len(c.strip()) > 10]
            for i, chunk in enumerate(chunks):
                _policy_store.append({
                    "source": file_path.name,
                    "section_id": i,
                    "text": chunk
                })
    
    _load_model()
    texts = [p["text"] for p in _policy_store]
    
    if _semantic_model and _policy_store:
        try:
            _embeddings = _semantic_model.encode(texts)
        except Exception:
            pass

    # Build TF-IDF as fallback
    doc_counts = {}
    for text in texts:
        tokens = set(_tokenize(text))
        for token in tokens:
            doc_counts[token] = doc_counts.get(token, 0) + 1
            
    n_docs = len(texts)
    _idf = {token: math.log(n_docs / count) for token, count in doc_counts.items()}

def _vectorize(text: str) -> Dict[str, float]:
    tokens = _tokenize(text)
    if not tokens:
        return {}
    counts = {}
    for t in tokens:
        if t in _idf:
            counts[t] = counts.get(t, 0) + 1
    total = sum(counts.values()) or 1
    vec = {k: (c/total)*_idf[k] for k, c in counts.items()}
    norm_val = math.sqrt(sum(v*v for v in vec.values()))
    if norm_val > 0:
        vec = {k: v/norm_val for k, v in vec.items()}
    return vec

def retrieve_policy(query: str, top_k: int = 2) -> List[Dict[str, Any]]:
    if not _policy_store:
        ingest_policies()
        
    if not _policy_store:
        return []
        
    scored = []
    
    if len(_embeddings) == len(_policy_store):
        # Use Semantic
        try:
            query_emb = _semantic_model.encode(query)
            for i, emb in enumerate(_embeddings):
                score = float(_cosine_sim(query_emb, emb))
                scored.append((score, _policy_store[i]))
        except Exception:
            pass

    if not scored:
        # Fallback to TF-IDF
        query_vec = _vectorize(query)
        for doc in _policy_store:
            doc_vec = _vectorize(doc["text"])
            score = sum(query_vec.get(k, 0)*v for k, v in doc_vec.items())
            scored.append((score, doc))
        
    scored.sort(key=lambda x: x[0], reverse=True)
    
    results = []
    for score, doc in scored[:top_k]:
        results.append({
            "score": score,
            "evidence": doc["text"],
            "source": doc["source"],
            "section_id": doc["section_id"]
        })
    return results

def get_policy_decision(query: str, confidence_threshold: float = 0.1) -> Dict[str, Any]:
    candidates = retrieve_policy(query)
    
    if not candidates:
        return {
            "decision": "ABSTAIN",
            "reason": "No policy documents found.",
            "confidence": 0.0,
            "evidence": []
        }
        
    best = candidates[0]
    
    if best["score"] < confidence_threshold:
        return {
            "decision": "ABSTAIN",
            "reason": "Insufficient evidence to make a policy claim.",
            "confidence": best["score"],
            "evidence": candidates
        }
        
    return {
        "decision": "POLICY_FOUND",
        "reason": "Found relevant policy.",
        "confidence": best["score"],
        "evidence": candidates,
        "grounded_context": best["evidence"]
    }

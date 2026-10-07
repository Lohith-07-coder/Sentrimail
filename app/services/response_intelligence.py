import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
from numpy import dot
from numpy.linalg import norm
from app.services.ai_service import _response_model, _load_response_model, _vectorize_for_inference, _generative_pipeline, _is_generic_dataset_response

logger = logging.getLogger(__name__)

_semantic_model = None

def _load_semantic():
    global _semantic_model
    if _semantic_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Could not load semantic model: {e}")

def _cosine_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b)) if norm(a) > 0 and norm(b) > 0 else 0.0

def _retrieve_candidates(text: str, category: str, priority: str) -> List[Dict[str, Any]]:
    _load_response_model()
    from app.services.ai_service import _response_model as r_model
    if not r_model:
        return []
        
    idf = r_model.get("idf", {})
    samples = r_model.get("samples", [])
    if not idf or not samples:
        return []

    query_vec = _vectorize_for_inference(text, idf)
    
    _load_semantic()
    semantic_query = None
    if _semantic_model:
        semantic_query = _semantic_model.encode(text)

    candidates = []
    category_lower = (category or "").lower()
    priority_upper = (priority or "").upper()
    
    for sample in samples:
        # TF-IDF
        sample_vec = sample.get("vector", {})
        tfidf_score = sum(query_vec.get(k, 0)*v for k, v in sample_vec.items()) if query_vec else 0.0
        
        # Semantic
        semantic_score = 0.0
        if semantic_query is not None:
            sample_text = sample.get("text", "")
            sample_emb = _semantic_model.encode(sample_text)
            semantic_score = float(_cosine_sim(semantic_query, sample_emb))
            
        # Boosts
        if sample.get("category") == category_lower:
            tfidf_score += 0.03
            semantic_score += 0.03
        if sample.get("priority") == priority_upper:
            tfidf_score += 0.02
            semantic_score += 0.02
            
        best_score = max(tfidf_score, semantic_score)
        source = "semantic" if semantic_score > tfidf_score else "tfidf"
        
        if best_score > 0.12:
            candidates.append({
                "response": sample.get("response", ""),
                "score": best_score,
                "source": source
            })
            
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates

def generate_grounded_response(
    text: str,
    username: str,
    category: str,
    priority: str,
    language: str = "en"
) -> Dict[str, Any]:
    
    candidates = _retrieve_candidates(text, category, priority)
    
    if not candidates:
        return {
            "response": "",
            "confidence": 0.0,
            "source": "none",
            "human_review_required": True
        }
        
    best = candidates[0]
    confidence = best["score"]
    source = best["source"]
    
    if confidence < 0.25:
        return {
            "response": best["response"].format(username=username) if "{username}" in best["response"] else best["response"],
            "confidence": confidence,
            "source": source,
            "human_review_required": True
        }
        
    context_response = best["response"]
    from app.services.ai_service import _generative_pipeline as g_pipe
    
    if g_pipe:
        prompt = (
            "System message:\n"
            "You are a professional customer support AI. Base your response strictly on the provided Context Policy. "
            "Do not hallucinate policy claims or external facts.\n\n"
            f"Context Policy:\n{context_response}\n\n"
            "User message:\n"
            f"Complaint: {text}\n"
            f"Write a helpful, specific reply in {language} language. Keep it to 3 to 4 sentences."
        )
        try:
            res = g_pipe(prompt, max_length=150, do_sample=True, top_p=0.95)[0]['generated_text']
            return {
                "response": res.strip().format(username=username) if "{username}" in res else res.strip(),
                "confidence": confidence,
                "source": source,
                "human_review_required": False
            }
        except:
            pass
            
    return {
        "response": context_response.format(username=username) if "{username}" in context_response else context_response,
        "confidence": confidence,
        "source": source,
        "human_review_required": False
    }

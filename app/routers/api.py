"""
REST API Router
---------------
Programmatic API endpoints for complaint listing, dashboard stats, AI text analysis, and audio transcription.
"""

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.services.auth_service import auth_service
from app.repositories.complaint_repository import complaint_repository
from app.services.ai_service import analyze_complaint
from app.services.complaint_service import calculate_dashboard_stats
from app.services.transcription_service import transcribe_audio_file
from app.schemas.complaint import ComplaintAnalysisRequest

router = APIRouter(prefix="/api", tags=["REST API"])


@router.get("/complaints")
def api_complaints(request: Request):
    user = auth_service.get_current_user(request)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return complaint_repository.list_all()


@router.get("/dashboard-stats")
def dashboard_stats(request: Request):
    user = auth_service.get_current_user(request)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return calculate_dashboard_stats()


@router.post("/analyze")
async def api_analyze(request: Request, analysis_request: ComplaintAnalysisRequest):
    user = auth_service.get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return analyze_complaint(analysis_request.text)


@router.post("/transcribe")
async def api_transcribe(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    text = await transcribe_audio_file(audio_bytes)
    return {"text": text}

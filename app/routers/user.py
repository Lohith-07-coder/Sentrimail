"""
User Portal Router
-------------------
Handles user dashboard, complaint submission, tracking, and detail views.
"""

from typing import Annotated
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import os

from app.services.auth_service import auth_service
from app.repositories.complaint_repository import complaint_repository
from app.services.complaint_service import process_complaint_submission
from app.services.email_service import send_resolution_email
from app.schemas.complaint import ComplaintCreateRequest, ComplaintTrackingRequest

router = APIRouter(tags=["User Portal"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@router.get("/user/dashboard", response_class=HTMLResponse)
def user_dashboard(request: Request):
    user = auth_service.get_current_user(request)
    if not user or user["role"] != "user":
        return RedirectResponse("/login", status_code=302)
    complaints = complaint_repository.list_for_user(user["username"])
    return templates.TemplateResponse("user_dashboard.html", {
        "request": request,
        "user": user,
        "complaints": complaints,
        "total": len(complaints),
        "pending": len([c for c in complaints if c.get("status") in ["pending", "pending_admin"]]),
        "resolved": len([c for c in complaints if c.get("status") in ["resolved", "auto_replied"]]),
    })


@router.get("/user/submit", response_class=HTMLResponse)
def submit_page(request: Request):
    user = auth_service.get_current_user(request)
    if not user or user["role"] != "user":
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse("submit_complaint.html", {"request": request, "user": user})


@router.post("/user/submit")
def submit_complaint(
    request: Request,
    background_tasks: BackgroundTasks,
    complaint_request: Annotated[ComplaintCreateRequest, Depends(ComplaintCreateRequest.as_form)],
):
    user = auth_service.get_current_user(request)
    if not user or user["role"] != "user":
        return RedirectResponse("/login", status_code=302)

    complaint, analysis, ai_reply_text, is_low = process_complaint_submission(
        username=user["username"],
        email=user.get("email", ""),
        title=complaint_request.title,
        description=complaint_request.description,
        language=complaint_request.language,
    )

    if is_low and ai_reply_text:
        background_tasks.add_task(send_resolution_email, complaint, ai_reply_text)

    return templates.TemplateResponse(
        "submit_complaint.html",
        {
            "request": request,
            "user": user,
            "success": True,
            "analysis": analysis,
            "title": complaint_request.title,
            "complaint": complaint,
            "auto_resolved": analysis.get("auto_resolvable", False),
        },
    )


@router.get("/track", response_class=HTMLResponse)
def track_page_get(request: Request):
    return templates.TemplateResponse("track.html", {"request": request})


@router.post("/track", response_class=HTMLResponse)
def track_page_post(request: Request, tracking: Annotated[ComplaintTrackingRequest, Depends(ComplaintTrackingRequest.as_form)]):
    complaint = complaint_repository.find_by_code(tracking.complaint_code)
    if not complaint:
        return templates.TemplateResponse("track.html", {"request": request, "error": "Complaint code not found."})

    return templates.TemplateResponse("track.html", {"request": request, "complaint": complaint})


@router.get("/user/complaint/{complaint_id}", response_class=HTMLResponse)
def user_complaint_detail(request: Request, complaint_id: str):
    user = auth_service.get_current_user(request)
    if not user or user["role"] != "user":
        return RedirectResponse("/login", status_code=302)

    complaint = complaint_repository.find_by_id(complaint_id)
    if not complaint or complaint.get("username") != user["username"]:
        raise HTTPException(status_code=404, detail="Complaint not found")

    return templates.TemplateResponse(
        "user_complaint_detail.html",
        {
            "request": request,
            "user": user,
            "complaint": complaint,
        },
    )

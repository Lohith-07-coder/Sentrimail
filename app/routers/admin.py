"""
Admin Portal Router
--------------------
Handles admin dashboard, complaint review, status and response updates, user management, and CSV export.
"""

import csv
import io
import os
from typing import Annotated
from deep_translator import GoogleTranslator
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from app.services.auth_service import auth_service
from app.repositories.complaint_repository import complaint_repository
from app.services.complaint_service import merge_or_backfill_analysis
from app.services.email_service import send_resolution_email
from app.schemas.admin import ComplaintResponseUpdateRequest, ComplaintStatusUpdateRequest

router = APIRouter(tags=["Admin Portal"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@router.get("/admin/dashboard", response_class=HTMLResponse)
def admin_dashboard(request: Request):
    user = auth_service.get_current_user(request)
    if not user or user.get("role") != "admin":
        return RedirectResponse("/login", status_code=302)

    complaints = complaint_repository.list_all()

    total = len(complaints)
    critical = len([c for c in complaints if c.get("priority") == "CRITICAL"])
    high = len([c for c in complaints if c.get("priority") == "HIGH"])
    medium = len([c for c in complaints if c.get("priority") == "MEDIUM"])
    low = len([c for c in complaints if c.get("priority") == "LOW"])
    pending = len([c for c in complaints if c.get("status") in ["pending", "pending_admin"]])

    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

    normalized_complaints = []
    for c in complaints:
        normalized_complaints.append({
            "id": c.get("id"),
            "title": c.get("title", "Untitled"),
            "priority": c.get("priority", "LOW"),
            "status": c.get("status", "pending"),
            "created_at": c.get("created_at", "N/A"),
            "description": c.get("description", "")
        })

    complaints_sorted = sorted(
        normalized_complaints,
        key=lambda x: priority_order.get(x["priority"], 3)
    )

    return templates.TemplateResponse(
        "admin_dashboard.html",
        {
            "request": request,
            "user": user,
            "complaints": complaints_sorted,
            "stats": {
                "total": total,
                "critical": critical,
                "high": high,
                "medium": medium,
                "low": low,
                "pending": pending
            }
        }
    )


@router.get("/admin/complaint/{complaint_id}", response_class=HTMLResponse)
def complaint_detail(request: Request, complaint_id: str):
    user = auth_service.get_current_user(request)

    if not user or user.get("role") != "admin":
        return RedirectResponse("/login", status_code=302)

    complaint = complaint_repository.find_by_id(complaint_id)

    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")

    complaint = merge_or_backfill_analysis(complaint)

    complaint_data = {
        "id": complaint.get("id"),
        "title": complaint.get("title", "Untitled"),
        "priority": complaint.get("priority", "LOW"),
        "status": complaint.get("status", "pending"),
        "created_at": complaint.get("created_at", ""),
        "updated_at": complaint.get("updated_at") or complaint.get("created_at", ""),
        "description": complaint.get("description", ""),
        "username": complaint.get("username", "Unknown"),
        "email": complaint.get("email", ""),
        "category": complaint.get("category", "other"),
        "root_cause_summary": complaint.get("root_cause_summary", "Not available."),
        "admin_response": complaint.get("admin_response", ""),
        "admin_suggested_response": complaint.get(
            "admin_suggested_response",
            complaint.get("ai_suggested_response", ""),
        ),
        "ai_suggested_response": complaint.get("ai_suggested_response", ""),
        "model_used": complaint.get("model_used", "unknown"),
        "sentiment_label": complaint.get("sentiment_label", "NEUTRAL"),
        "sentiment_score": complaint.get("sentiment_score", 0),
        "emotion_label": complaint.get("emotion_label", "neutral"),
        "emotion_score": complaint.get("emotion_score", 0),
        "priority_score": complaint.get("priority_score", 0),
        "priority_description": complaint.get("priority_description", ""),
    }

    return templates.TemplateResponse(
        "complaint_detail.html",
        {
            "request": request,
            "user": user,
            "complaint": complaint_data
        }
    )


@router.post("/admin/complaint/{complaint_id}/status")
def update_status(request: Request, complaint_id: str, status_update: Annotated[ComplaintStatusUpdateRequest, Depends(ComplaintStatusUpdateRequest.as_form)]):
    user = auth_service.get_current_user(request)
    if not user or user.get("role") != "admin":
        return RedirectResponse("/login", status_code=302)

    complaint_repository.update_status(complaint_id, status_update.status)
    return RedirectResponse(f"/admin/complaint/{complaint_id}", status_code=302)


@router.post("/admin/complaint/{complaint_id}/response")
def update_response(
    request: Request,
    complaint_id: str,
    background_tasks: BackgroundTasks,
    response_update: Annotated[ComplaintResponseUpdateRequest, Depends(ComplaintResponseUpdateRequest.as_form)],
):
    user = auth_service.get_current_user(request)
    if not user or user.get("role") != "admin":
        return RedirectResponse("/login", status_code=302)

    complaint = complaint_repository.find_by_id(complaint_id)
    if complaint:
        original_language = complaint.get("original_language", "en")
        if original_language and original_language != "en":
            try:
                response_update.response = GoogleTranslator(source='en', target=original_language).translate(response_update.response)
            except Exception as e:
                print(f"Translation failed: {e}")

        if response_update.status == "resolved":
            background_tasks.add_task(send_resolution_email, complaint, response_update.response)

    complaint_repository.update_response(complaint_id, response_update.response, response_update.status)
    return RedirectResponse(f"/admin/complaint/{complaint_id}", status_code=302)


@router.get("/admin/users", response_class=HTMLResponse)
def admin_users(request: Request):
    user = auth_service.get_current_user(request)
    if not user or user.get("role") != "admin":
        return RedirectResponse("/login", status_code=302)
    users = auth_service.get_all_users()
    return templates.TemplateResponse("admin_users.html", {"request": request, "user": user, "users": users})


@router.get("/admin/export")
def export_complaints_csv(request: Request):
    user = auth_service.get_current_user(request)
    if not user or user.get("role") != "admin":
        return RedirectResponse("/login", status_code=302)

    complaints = complaint_repository.list_all_raw()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "id", "complaint_code", "original_language", "original_text", "translated_text",
        "category", "priority", "sentiment", "keyword_escalated", "status", "created_at", "resolved_at"
    ])

    for c in complaints:
        writer.writerow([
            c.get("id", ""),
            c.get("complaint_code", ""),
            c.get("original_language", "en"),
            c.get("original_text", ""),
            c.get("description", ""),
            c.get("category", ""),
            c.get("priority", "LOW"),
            c.get("sentiment_label", "NEUTRAL"),
            str(c.get("keyword_escalated", False)),
            c.get("status", ""),
            c.get("created_at", ""),
            (c.get("updated_at", "") if c.get("status") == "resolved" else "")
        ])

    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=complaints_export.csv"}
    )

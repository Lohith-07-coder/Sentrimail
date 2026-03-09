from fastapi import FastAPI, Request, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys
from datetime import datetime
from app.mongodb import db, init_mongodb


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.auth import (
    authenticate_user,
    create_session,
    ensure_default_users,
    get_all_users,
    get_current_user,
    logout_user,
    register_user,
)
from app.storage import (
    get_all_complaints,
    get_user_complaints,
    save_complaint,
    get_complaint_by_id,
    update_complaint_response,
)
from app.ai_engine import analyze_complaint

app = FastAPI(title="SentriMail", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


def _merge_or_backfill_analysis(complaint: dict) -> dict:
    """
    Backfill missing/stale AI fields for older complaint records.
    This keeps admin pages useful even when records were saved before AI fields existed.
    """
    description = complaint.get("description", "") or ""
    category = complaint.get("category", "other") or "other"
    username = complaint.get("username", "Customer") or "Customer"
    complaint_id = complaint.get("id", "N/A") or "N/A"

    computed = analyze_complaint(
        description,
        category=category,
        username=username,
        complaint_id=complaint_id,
    )

    model_used = complaint.get("model_used")
    existing_suggestion = complaint.get("admin_suggested_response") or complaint.get("ai_suggested_response")
    has_priority_score = isinstance(complaint.get("priority_score"), (int, float))
    has_sentiment_score = isinstance(complaint.get("sentiment_score"), (int, float))
    has_emotion_score = isinstance(complaint.get("emotion_score"), (int, float))

    force_refresh = model_used in (None, "", "unknown")

    needs_backfill = (
        model_used in (None, "", "unknown")
        or not existing_suggestion
        or not has_priority_score
        or not has_sentiment_score
        or not has_emotion_score
        or not complaint.get("root_cause_summary")
    )

    if not needs_backfill:
        return complaint

    merged = dict(complaint)
    merged["priority"] = computed.get("priority", "LOW") if force_refresh else complaint.get("priority", computed.get("priority", "LOW"))
    merged["priority_score"] = computed.get("priority_score", 0) if force_refresh else complaint.get("priority_score", computed.get("priority_score", 0))
    merged["priority_description"] = computed.get("priority_description", "") if force_refresh else (complaint.get("priority_description") or computed.get("priority_description", ""))
    merged["sentiment_label"] = computed.get("sentiment_label", "NEUTRAL") if force_refresh else (complaint.get("sentiment_label") or computed.get("sentiment_label", "NEUTRAL"))
    merged["sentiment_score"] = computed.get("sentiment_score", 0) if force_refresh else complaint.get("sentiment_score", computed.get("sentiment_score", 0))
    merged["emotion_label"] = computed.get("emotion_label", "Neutral") if force_refresh else (complaint.get("emotion_label") or computed.get("emotion_label", "Neutral"))
    merged["emotion_score"] = computed.get("emotion_score", 0) if force_refresh else complaint.get("emotion_score", computed.get("emotion_score", 0))
    merged["root_cause_summary"] = complaint.get("root_cause_summary") or computed.get("root_cause_summary", "")
    merged["admin_suggested_response"] = (
        complaint.get("admin_suggested_response")
        or complaint.get("ai_suggested_response")
        or computed.get("admin_suggested_response", "")
    )
    merged["ai_suggested_response"] = (
        complaint.get("ai_suggested_response")
        or complaint.get("admin_suggested_response")
        or computed.get("ai_suggested_response", "")
    )
    merged["model_used"] = computed.get("model_used", "rule-based")

    # Persist backfilled AI fields so future renders are consistent.
    db.complaints.update_one(
        {"id": complaint_id},
        {
            "$set": {
                "priority": merged.get("priority", "LOW"),
                "priority_score": merged.get("priority_score", 0),
                "priority_description": merged.get("priority_description", ""),
                "sentiment_label": merged.get("sentiment_label", "NEUTRAL"),
                "sentiment_score": merged.get("sentiment_score", 0),
                "emotion_label": merged.get("emotion_label", "Neutral"),
                "emotion_score": merged.get("emotion_score", 0),
                "root_cause_summary": merged.get("root_cause_summary", ""),
                "admin_suggested_response": merged.get("admin_suggested_response", ""),
                "ai_suggested_response": merged.get("ai_suggested_response", ""),
                "model_used": merged.get("model_used", "rule-based"),
                "updated_at": datetime.utcnow().isoformat(),
            }
        },
    )

    return merged


@app.on_event("startup")
async def startup_event():
    init_mongodb()
    ensure_default_users()


# ─── Routes ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    try:
        user = get_current_user(request)
    except Exception:
        user = None

    if user:
        if user.get("role") == "admin":
            return RedirectResponse("/admin/dashboard", status_code=302)
        return RedirectResponse("/user/dashboard", status_code=302)

    return RedirectResponse("/login", status_code=302)

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = None):
    user = get_current_user(request)
    if user:
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": error})


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):

    user = authenticate_user(username, password)

    if not user:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Invalid credentials. Please try again."
        })

    # 🔹 Save login event
    db.login_logs.insert_one({
        "username": username,
        "role": user.get("role"),
        "login_time": datetime.utcnow().isoformat()
    })

    response = RedirectResponse("/", status_code=302)
    create_session(response, user)

    return response
@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, error: str = None, success: str = None):
    return templates.TemplateResponse("register.html", {"request": request, "error": error, "success": success})


@app.post("/register")
async def register(request: Request, username: str = Form(...), password: str = Form(...), email: str = Form(...)):
    result = register_user(username, password, email)
    if not result["success"]:
        return templates.TemplateResponse("register.html", {"request": request, "error": result["message"]})
    return templates.TemplateResponse("register.html", {"request": request, "success": "Account created! You can now log in."})


@app.get("/logout")
async def logout(request: Request):
    response = RedirectResponse("/login", status_code=302)
    logout_user(response)
    return response


# ─── User Routes ────────────────────────────────────────────────

@app.get("/user/dashboard", response_class=HTMLResponse)
async def user_dashboard(request: Request):
    user = get_current_user(request)
    if not user or user["role"] != "user":
        return RedirectResponse("/login", status_code=302)
    complaints = get_user_complaints(user["username"])
    return templates.TemplateResponse("user_dashboard.html", {
        "request": request,
        "user": user,
        "complaints": complaints,
        "total": len(complaints),
        "pending": len([c for c in complaints if c["status"] == "pending"]),
        "resolved": len([c for c in complaints if c["status"] == "resolved"]),
    })


@app.get("/user/submit", response_class=HTMLResponse)
async def submit_page(request: Request):
    user = get_current_user(request)
    if not user or user["role"] != "user":
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse("submit_complaint.html", {"request": request, "user": user})


@app.post("/user/submit")
async def submit_complaint(
    request: Request,
    title: str = Form(...),
    description: str = Form(...)
):
    user = get_current_user(request)
    if not user or user["role"] != "user":
        return RedirectResponse("/login", status_code=302)

    # AI Analysis
    analysis = analyze_complaint(description, category="other", username=user["username"])

    auto_resolvable = bool(analysis.get("auto_resolvable", False))

    complaint_data = {
        "title": title,
        "category": "other",
        "description": description,
        "username": user["username"],
        "email": user.get("email", ""),
        **analysis,
    }

    # Auto-resolve only when AI marks it safe/resolvable.
    if auto_resolvable:
        complaint_data["status"] = "resolved"
        complaint_data["admin_response"] = analysis.get("user_auto_response", "")
    else:
        complaint_data["status"] = "pending"

    complaint = save_complaint(complaint_data)

    return templates.TemplateResponse(
        "submit_complaint.html",
        {
            "request": request,
            "user": user,
            "success": True,
            "analysis": analysis,
            "title": title,
            "complaint": complaint,
            "auto_resolved": auto_resolvable,
        },
    )


# ─── Admin Routes ────────────────────────────────────────────────

@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    user = get_current_user(request)
    if not user or user["role"] != "admin":
        return RedirectResponse("/login", status_code=302)

    complaints = get_all_complaints()
    
    # Stats
    total = len(complaints)
    critical = len([c for c in complaints if c.get("priority") == "CRITICAL"])
    high = len([c for c in complaints if c.get("priority") == "HIGH"])
    medium = len([c for c in complaints if c.get("priority") == "MEDIUM"])
    low = len([c for c in complaints if c.get("priority") == "LOW"])
    pending = len([c for c in complaints if c.get("status") == "pending"])

    # Sort by priority weight
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    
    # Normalize complaints so template never crashes
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
    
    # Sort complaints safely
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

@app.get("/admin/complaint/{complaint_id}", response_class=HTMLResponse)
async def complaint_detail(request: Request, complaint_id: str):
    user = get_current_user(request)

    if not user or user.get("role") != "admin":
        return RedirectResponse("/login", status_code=302)

    complaint = get_complaint_by_id(complaint_id)

    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")

    complaint = _merge_or_backfill_analysis(complaint)

    # Normalize complaint so template rendering remains safe for older/incomplete records.
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

@app.post("/admin/complaint/{complaint_id}/status")
async def update_status(request: Request, complaint_id: str, status: str = Form(...)):
    user = get_current_user(request)
    if not user or user["role"] != "admin":
        return RedirectResponse("/login", status_code=302)

    from app.storage import update_complaint_status

    update_complaint_status(complaint_id, status)
    return RedirectResponse(f"/admin/complaint/{complaint_id}", status_code=302)


@app.post("/admin/complaint/{complaint_id}/response")
async def update_response(
    request: Request,
    complaint_id: str,
    response: str = Form(...),
    status: str = Form("resolved"),
):
    user = get_current_user(request)
    if not user or user["role"] != "admin":
        return RedirectResponse("/login", status_code=302)

    update_complaint_response(complaint_id, response, status)
    return RedirectResponse(f"/admin/complaint/{complaint_id}", status_code=302)


@app.get("/admin/users", response_class=HTMLResponse)
async def admin_users(request: Request):
    user = get_current_user(request)
    if not user or user["role"] != "admin":
        return RedirectResponse("/login", status_code=302)
    users = get_all_users()
    return templates.TemplateResponse("admin_users.html", {"request": request, "user": user, "users": users})


@app.get("/user/complaint/{complaint_id}", response_class=HTMLResponse)
async def user_complaint_detail(request: Request, complaint_id: str):
    user = get_current_user(request)
    if not user or user["role"] != "user":
        return RedirectResponse("/login", status_code=302)

    complaint = get_complaint_by_id(complaint_id)
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


# ─── API Endpoints ────────────────────────────────────────────────

@app.get("/api/complaints")
async def api_complaints(request: Request):
    user = get_current_user(request)
    if not user or user["role"] != "admin":
        raise HTTPException(status_code=403)
    return get_all_complaints()


@app.post("/api/analyze")
async def api_analyze(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401)
    body = await request.json()
    text = body.get("text", "")
    return analyze_complaint(text)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

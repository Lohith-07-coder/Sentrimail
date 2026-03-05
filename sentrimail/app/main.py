from fastapi import FastAPI, Request, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys
from datetime import datetime
from app.firebase_config import db


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.auth import authenticate_user, create_session, get_current_user, logout_user, get_all_users, register_user
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
    db.collection("login_logs").add({
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

    is_low_priority = analysis.get("priority") == "LOW"

    complaint_data = {
        "title": title,
        "category": "other",
        "description": description,
        "username": user["username"],
        "email": user.get("email", ""),
        **analysis,
    }

    # If LOW priority, let AI auto-respond and mark as resolved
    if is_low_priority:
        complaint_data["status"] = "resolved"
        complaint_data["admin_response"] = analysis.get("ai_suggested_response", "")

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
            "auto_resolved": is_low_priority,
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
    if not user or user["role"] != "admin":
        return RedirectResponse("/login", status_code=302)

    complaint = get_complaint_by_id(complaint_id)
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")

    return templates.TemplateResponse("complaint_detail.html", {
        "request": request,
        "user": user,
        "complaint": complaint
    })


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

from fastapi import FastAPI, Request, Form, Depends, HTTPException, status, File, UploadFile, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys
from datetime import datetime
import langdetect
from deep_translator import GoogleTranslator
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import timedelta
import csv
import io
import smtplib
from email.mime.text import MIMEText
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


def escalate_complaints():
    now = datetime.utcnow()
    # Find MEDIUM > 24 hours old
    threshold_24 = (now - timedelta(hours=24)).isoformat()
    db.complaints.update_many(
        {"status": "pending_admin", "priority": "MEDIUM", "created_at": {"$lt": threshold_24}},
        {"$set": {"priority": "HIGH"}}
    )
    # Find HIGH > 48 hours old
    threshold_48 = (now - timedelta(hours=48)).isoformat()
    db.complaints.update_many(
        {"status": "pending_admin", "priority": "HIGH", "created_at": {"$lt": threshold_48}},
        {"$set": {"priority": "CRITICAL"}}
    )

scheduler = BackgroundScheduler()
scheduler.add_job(escalate_complaints, 'interval', hours=1)

@app.on_event("startup")
async def startup_event():
    init_mongodb()
    ensure_default_users()
    scheduler.start()


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

@app.get("/track", response_class=HTMLResponse)
async def track_page_get(request: Request):
    return templates.TemplateResponse("track.html", {"request": request})

@app.post("/track", response_class=HTMLResponse)
async def track_page_post(request: Request, complaint_code: str = Form(...)):
    complaint = db.complaints.find_one({"complaint_code": complaint_code}, {"_id": 0})
    if not complaint:
        return templates.TemplateResponse("track.html", {"request": request, "error": "Complaint code not found."})
        
    return templates.TemplateResponse("track.html", {"request": request, "complaint": complaint})


def send_resolution_email(complaint_data: dict, reply_text: dict):
    email = complaint_data.get("email")
    if not email:
        return
        
    code = complaint_data.get("complaint_code", "N/A")
    orig_lang = complaint_data.get("original_language", "en")
    desc = complaint_data.get("description", "N/A")
    
    subject = f"Your complaint {code} has been resolved — SentriMail"
    body = f"Complaint Summary:\n{desc}\n\nOur Reply:\n{reply_text}"
    
    if orig_lang and orig_lang != "en":
        try:
            subject = GoogleTranslator(source='en', target=orig_lang).translate(subject)
            body = GoogleTranslator(source='en', target=orig_lang).translate(body)
        except:
            pass

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = os.environ.get("MAIL_DEFAULT_SENDER", "noreply@sentrimail.com")
    msg["To"] = email

    try:
        server = smtplib.SMTP(
            os.environ.get("MAIL_SERVER", "localhost"),
            int(os.environ.get("MAIL_PORT", 587))
        )
        if os.environ.get("MAIL_USE_TLS") == "True":
            server.starttls()
            server.login(
                os.environ.get("MAIL_USERNAME", ""),
                os.environ.get("MAIL_PASSWORD", "")
            )
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print(f"Email failed to send: {e}")


@app.post("/user/submit")
async def submit_complaint(
    request: Request,
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    description: str = Form(...),
    language: str = Form(None)
):
    user = get_current_user(request)
    if not user or user["role"] != "user":
        return RedirectResponse("/login", status_code=302)

    original_text = description
    detected_lang = language

    if not detected_lang:
        try:
            detected_lang = langdetect.detect(original_text)
        except:
            detected_lang = "en"
    
    translated_text = original_text
    if detected_lang and detected_lang != "en":
        try:
            translated_text = GoogleTranslator(source='auto', target='en').translate(original_text)
        except Exception as e:
            print(f"Translation failed: {e}")

    # AI Analysis
    analysis = analyze_complaint(translated_text, category="other", username=user["username"])

    priority = analysis.get("priority", "LOW").upper()

    keywords = ["legal", "police", "harassment", "abuse", "threat", "court", "violence", "urgent", "emergency", "lawsuit", "assault"]
    text_lower = translated_text.lower()
    keyword_escalated = any(kw in text_lower for kw in keywords)

    if keyword_escalated:
        priority = "CRITICAL"
        analysis["priority"] = "CRITICAL"

    complaint_data = {
        "title": title,
        "category": "other",
        "description": translated_text,
        "original_text": original_text,
        "original_language": detected_lang,
        "keyword_escalated": keyword_escalated,
        "username": user["username"],
        "email": user.get("email", ""),
        **analysis,
    }

    if priority == "LOW":
        complaint_data["status"] = "auto_replied"
        ai_reply_text = analysis.get("ai_suggested_response", "Thank you for your feedback.")
        if detected_lang and detected_lang != "en":
            try:
                ai_reply_text = GoogleTranslator(source='en', target=detected_lang).translate(ai_reply_text)
            except:
                pass
        complaint_data["admin_response"] = ai_reply_text
    else:
        complaint_data["status"] = "pending_admin"

    complaint = save_complaint(complaint_data)

    if priority == "LOW":
        db.replies.insert_one({
            "complaint_id": complaint["id"],
            "reply_text": ai_reply_text,
            "is_ai_reply": True,
            "replied_at": datetime.utcnow().isoformat(),
            "replied_by": "AI"
        })
        background_tasks.add_task(send_resolution_email, complaint_data, ai_reply_text)

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
    background_tasks: BackgroundTasks,
    response: str = Form(...),
    status: str = Form("resolved"),
):
    user = get_current_user(request)
    if not user or user["role"] != "admin":
        return RedirectResponse("/login", status_code=302)

    complaint = get_complaint_by_id(complaint_id)
    if complaint:
        original_language = complaint.get("original_language", "en")
        if original_language and original_language != "en":
            try:
                response = GoogleTranslator(source='en', target=original_language).translate(response)
            except Exception as e:
                print(f"Translation failed: {e}")
        
        if status == "resolved":
            background_tasks.add_task(send_resolution_email, complaint, response)

    update_complaint_response(complaint_id, response, status)
    return RedirectResponse(f"/admin/complaint/{complaint_id}", status_code=302)


@app.get("/admin/users", response_class=HTMLResponse)
async def admin_users(request: Request):
    user = get_current_user(request)
    if not user or user["role"] != "admin":
        return RedirectResponse("/login", status_code=302)
    users = get_all_users()
    return templates.TemplateResponse("admin_users.html", {"request": request, "user": user, "users": users})


@app.get("/admin/export")
async def export_complaints_csv(request: Request):
    user = get_current_user(request)
    if not user or user["role"] != "admin":
        return RedirectResponse("/login", status_code=302)

    complaints = list(db.complaints.find({}).sort("created_at", -1))
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Columns requested: id, complaint_code, original_language, original_text, translated_text, category, priority, sentiment, keyword_escalated, status, created_at, resolved_at
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
            c.get("description", ""),  # translated_text maps to description
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


@app.get("/api/dashboard-stats")
async def dashboard_stats(request: Request):
    user = get_current_user(request)
    if not user or user["role"] != "admin":
        raise HTTPException(status_code=403)

    complaints = list(db.complaints.find({}))
    
    # Priority counts
    priority_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    # Category counts
    category_counts = {}
    # Sentiment by category
    sentiment_by_category = {}
    
    pending = 0
    resolved_today = 0
    total_hours = 0
    resolved_count = 0
    
    now = datetime.utcnow()
    daily_counts_map = {}

    for c in complaints:
        category = c.get("category", "other").lower()
        priority = c.get("priority", "LOW").lower()
        sentiment = c.get("sentiment_label", "NEUTRAL").lower()
        status = c.get("status", "pending")
        created_at_raw = c.get("created_at", "")
        updated_at_raw = c.get("updated_at", created_at_raw)
        
        category_counts[category] = category_counts.get(category, 0) + 1
        
        if priority in priority_counts:
            priority_counts[priority] += 1
            
        if category not in sentiment_by_category:
            sentiment_by_category[category] = {"positive": 0, "neutral": 0, "negative": 0}
        if sentiment in sentiment_by_category[category]:
            sentiment_by_category[category][sentiment] += 1
            
        if status in ["pending", "pending_admin"]:
            pending += 1
            
        if created_at_raw:
            date_str = created_at_raw[:10]
            daily_counts_map[date_str] = daily_counts_map.get(date_str, 0) + 1
            
        if status == "resolved" and updated_at_raw:
            try:
                date_updated = datetime.fromisoformat(updated_at_raw)
                date_created = datetime.fromisoformat(created_at_raw)
                if date_updated.date() == now.date():
                    resolved_today += 1
                diff = (date_updated - date_created).total_seconds() / 3600.0
                if diff > 0:
                    total_hours += diff
                    resolved_count += 1
            except:
                pass
                
    # Sort last 30 days
    daily_counts = [{"date": k, "count": v} for k, v in daily_counts_map.items()]
    daily_counts.sort(key=lambda x: x["date"])
    daily_counts = daily_counts[-30:]
    
    avg_response_hours = round(total_hours / resolved_count, 1) if resolved_count > 0 else 0
    
    return {
        "category_counts": category_counts,
        "priority_counts": priority_counts,
        "daily_counts": daily_counts,
        "sentiment_by_category": sentiment_by_category,
        "stats": {
            "total": len(complaints),
            "pending_admin": pending,
            "resolved_today": resolved_today,
            "avg_response_hours": avg_response_hours
        }
    }


@app.post("/api/analyze")
async def api_analyze(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401)
    body = await request.json()
    text = body.get("text", "")
    return analyze_complaint(text)


import whisper
import tempfile

@app.post("/api/transcribe")
async def api_transcribe(audio: UploadFile = File(...)):
    # Save the file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        model = whisper.load_model("base")
        result = model.transcribe(tmp_path)
        text = result["text"]
    except Exception as e:
        print(f"Whisper transcription failed: {e}")
        text = ""
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {"text": text.strip()}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

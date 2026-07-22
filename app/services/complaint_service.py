"""
SentriMail Complaint Service
----------------------------
Business logic for complaints lifecycle, analysis backfill, threshold escalation,
submission processing, and analytics aggregation.
"""

from datetime import datetime, timedelta
import langdetect
from deep_translator import GoogleTranslator

from app.repositories.complaint_repository import complaint_repository
from app.services.ai_service import analyze_complaint


def merge_or_backfill_analysis(complaint: dict) -> dict:
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

    # Persist backfilled AI fields
    complaint_repository.update_analysis(
        complaint_id,
        {
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
        },
    )

    return merged


def escalate_complaints() -> None:
    now = datetime.utcnow()
    threshold_24 = (now - timedelta(hours=24)).isoformat()
    threshold_48 = (now - timedelta(hours=48)).isoformat()
    complaint_repository.escalate_by_thresholds(threshold_24, threshold_48)


def process_complaint_submission(username: str, email: str, title: str, description: str, language: str | None = None) -> tuple[dict, dict, str | None, bool]:
    original_text = description
    detected_lang = language

    if not detected_lang:
        try:
            detected_lang = langdetect.detect(original_text)
        except Exception:
            detected_lang = "en"

    translated_text = original_text
    if detected_lang and detected_lang != "en":
        try:
            translated_text = GoogleTranslator(source='auto', target='en').translate(original_text)
        except Exception as e:
            print(f"Translation failed: {e}")

    analysis = analyze_complaint(translated_text, category="other", username=username)
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
        "username": username,
        "email": email,
        **analysis,
    }

    ai_reply_text = None
    if priority == "LOW":
        complaint_data["status"] = "auto_replied"
        ai_reply_text = analysis.get("ai_suggested_response", "Thank you for your feedback.")
        if detected_lang and detected_lang != "en":
            try:
                ai_reply_text = GoogleTranslator(source='en', target=detected_lang).translate(ai_reply_text)
            except Exception:
                pass
        complaint_data["admin_response"] = ai_reply_text
    else:
        complaint_data["status"] = "pending_admin"

    complaint = complaint_repository.create(complaint_data)

    if priority == "LOW" and ai_reply_text:
        complaint_repository.add_ai_reply(complaint["id"], ai_reply_text)

    return complaint, analysis, ai_reply_text, priority == "LOW"


def calculate_dashboard_stats() -> dict:
    complaints = complaint_repository.list_all_raw()

    priority_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    category_counts = {}
    sentiment_by_category = {}

    pending = 0
    resolved_today = 0
    total_hours = 0.0
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
            except Exception:
                pass

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

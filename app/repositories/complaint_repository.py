"""MongoDB persistence operations for complaints and their replies."""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from app.core.database import db


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ComplaintRepository:
    """Provide the only persistence boundary for complaint-related collections."""

    def list_all(self) -> list[dict[str, Any]]:
        return list(db.complaints.find({}, {"_id": 0}).sort("created_at", -1))

    def list_all_raw(self) -> list[dict[str, Any]]:
        return list(db.complaints.find({}).sort("created_at", -1))

    def list_for_user(self, username: str) -> list[dict[str, Any]]:
        return list(
            db.complaints.find({"username": username}, {"_id": 0}).sort(
                "created_at", -1
            )
        )

    def find_by_id(self, complaint_id: str) -> Optional[dict[str, Any]]:
        return db.complaints.find_one({"id": complaint_id}, {"_id": 0})

    def find_by_code(self, complaint_code: str) -> Optional[dict[str, Any]]:
        return db.complaints.find_one({"complaint_code": complaint_code}, {"_id": 0})

    def create(self, data: dict[str, Any]) -> dict[str, Any]:
        current_year = datetime.now().year
        prefix = f"SENT-{current_year}-"
        matches = list(
            db.complaints.find({"complaint_code": {"$regex": f"^{prefix}"}}).sort(
                "created_at", -1
            )
        )
        last_code = matches[0].get("complaint_code") if matches else None
        sequence = int(last_code.split("-")[-1]) + 1 if last_code else 1

        complaint = dict(data)
        complaint["id"] = str(uuid.uuid4())
        complaint["complaint_code"] = f"{prefix}{sequence:04d}"
        complaint["status"] = complaint.get("status", "pending")
        complaint["created_at"] = complaint.get("created_at", _now_iso())
        complaint["updated_at"] = complaint.get("updated_at", complaint["created_at"])
        db.complaints.insert_one(complaint)
        return complaint

    def update_status(self, complaint_id: str, status: str) -> bool:
        result = db.complaints.update_one(
            {"id": complaint_id},
            {"$set": {"status": status, "updated_at": _now_iso()}},
        )
        return result.matched_count > 0

    def update_response(self, complaint_id: str, response: str, status: str) -> bool:
        result = db.complaints.update_one(
            {"id": complaint_id},
            {
                "$set": {
                    "admin_response": response,
                    "status": status,
                    "updated_at": _now_iso(),
                }
            },
        )
        return result.matched_count > 0

    def update_analysis(self, complaint_id: str, fields: dict[str, Any]) -> None:
        db.complaints.update_one({"id": complaint_id}, {"$set": fields})

    def escalate_by_thresholds(self, medium_threshold: str, high_threshold: str) -> None:
        db.complaints.update_many(
            {"status": "pending_admin", "priority": "MEDIUM", "created_at": {"$lt": medium_threshold}},
            {"$set": {"priority": "HIGH"}},
        )
        db.complaints.update_many(
            {"status": "pending_admin", "priority": "HIGH", "created_at": {"$lt": high_threshold}},
            {"$set": {"priority": "CRITICAL"}},
        )

    def add_ai_reply(self, complaint_id: str, reply_text: str) -> None:
        db.replies.insert_one(
            {
                "complaint_id": complaint_id,
                "reply_text": reply_text,
                "is_ai_reply": True,
                "replied_at": datetime.utcnow().isoformat(),
                "replied_by": "AI",
            }
        )


complaint_repository = ComplaintRepository()

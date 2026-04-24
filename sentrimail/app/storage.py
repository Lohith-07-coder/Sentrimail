import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from app.mongodb import db


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_all_complaints() -> List[Dict]:
    complaints = db.complaints.find({}, {"_id": 0}).sort("created_at", -1)
    return list(complaints)


def get_user_complaints(username: str) -> List[Dict]:
    complaints = db.complaints.find(
        {"username": username},
        {"_id": 0},
    ).sort("created_at", -1)
    return list(complaints)


def get_complaint_by_id(cid: str) -> Optional[Dict]:
    return db.complaints.find_one({"id": cid}, {"_id": 0})


def save_complaint(data: Dict) -> Dict:
    current_year = datetime.now().year
    
    # Prefix mapping to query max
    prefix = f"SENT-{current_year}-"
    last_complaint = db.complaints.find({"complaint_code": {"$regex": f"^{prefix}"}}).sort("created_at", -1).limit(1)
    last_complaint = list(last_complaint)
    
    if last_complaint and last_complaint[0].get("complaint_code"):
        last_code = last_complaint[0]["complaint_code"]
        seq = int(last_code.split("-")[-1]) + 1
    else:
        seq = 1
        
    complaint_code = f"{prefix}{seq:04d}"

    complaint = dict(data)
    complaint["id"] = str(uuid.uuid4())
    complaint["complaint_code"] = complaint_code
    complaint["status"] = complaint.get("status", "pending")
    complaint["created_at"] = complaint.get("created_at", _now_iso())
    complaint["updated_at"] = complaint.get("updated_at", complaint["created_at"])

    db.complaints.insert_one(complaint)
    return complaint


def update_complaint_status(complaint_id: str, status: str) -> bool:
    result = db.complaints.update_one(
        {"id": complaint_id},
        {"$set": {"status": status, "updated_at": _now_iso()}},
    )
    return result.matched_count > 0


def update_complaint_response(
    complaint_id: str,
    response: str,
    status: str = "resolved",
) -> bool:
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

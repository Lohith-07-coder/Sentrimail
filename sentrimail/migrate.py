import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from app.mongodb import db, init_mongodb

def migrate():
    print("Starting database migration for SentriMail...")
    init_mongodb()
    
    complaints = list(db.complaints.find({}))
    updated = 0
    
    for c in complaints:
        update_fields = {}
        
        # original_text defaults to description if missing
        if "original_text" not in c:
            update_fields["original_text"] = c.get("description", "")
        
        # translated_text maps similarly back
        if "translated_text" not in c:
            update_fields["translated_text"] = c.get("description", "")
            
        # language defaults to en
        if "original_language" not in c:
            update_fields["original_language"] = "en"
            
        if "keyword_escalated" not in c:
            update_fields["keyword_escalated"] = False
            
        if "complaint_code" not in c:
            # We'll just generate one
            created_at = c.get("created_at", "2024-01-01T00:00:00")
            year = created_at[:4] if created_at else "2024"
            update_fields["complaint_code"] = f"SENT-{year}-{c['_id']}"
            
        if "priority" not in c:
            update_fields["priority"] = "LOW"
            
        if "status" not in c:
            update_fields["status"] = "pending"
            
        if update_fields:
            db.complaints.update_one({"_id": c["_id"]}, {"$set": update_fields})
            updated += 1
            
    print(f"Migration completed. Checked {len(complaints)} existing complaints. Updated {updated}.")
    
    # Check replies collection
    # In MongoDB collections are lazy created but we can do a dummy find to ensure logging
    _ = list(db.replies.find().limit(1))
    print("Replies collection initialized.")
    

if __name__ == "__main__":
    migrate()

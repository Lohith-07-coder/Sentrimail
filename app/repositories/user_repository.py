"""MongoDB persistence operations for users and login audit records."""

from datetime import datetime
from typing import Any, Optional

from app.core.database import db


class UserRepository:
    """Provide the only persistence boundary for user-related collections."""

    def find_by_username(self, username: str) -> Optional[dict[str, Any]]:
        return db.users.find_one({"username": username})

    def username_exists(self, username: str) -> bool:
        return db.users.find_one({"username": username}, {"_id": 1}) is not None

    def create(self, user: dict[str, Any]) -> None:
        db.users.insert_one(user)

    def create_if_absent(self, user: dict[str, Any]) -> None:
        db.users.update_one(
            {"username": user["username"]},
            {"$setOnInsert": user},
            upsert=True,
        )

    def update_password(self, username: str, password_hash: str) -> None:
        db.users.update_one(
            {"username": username},
            {"$set": {"password": password_hash}},
        )

    def list_without_passwords(self) -> list[dict[str, Any]]:
        return list(db.users.find({}, {"_id": 0, "password": 0}))

    def record_login(self, username: str, role: str) -> None:
        db.login_logs.insert_one(
            {
                "username": username,
                "role": role,
                "login_time": datetime.utcnow().isoformat(),
            }
        )


user_repository = UserRepository()

"""
SentriMail Authentication
--------------------------
Session-cookie based auth with MongoDB persistence.
"""

import hashlib
import secrets
from typing import Dict, Optional

from fastapi import Request
from fastapi.responses import Response

from app.mongodb import db

SESSIONS: Dict[str, dict] = {}
SESSION_COOKIE = "sentrimail_session"


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def ensure_default_users() -> None:
    default_users = [
        {
            "username": "admin",
            "password": _hash_password("admin123"),
            "email": "admin@sentrimail.io",
            "role": "admin",
        },
        {
            "username": "alice",
            "password": _hash_password("alice123"),
            "email": "alice@example.com",
            "role": "user",
        },
        {
            "username": "bob",
            "password": _hash_password("bob123"),
            "email": "bob@example.com",
            "role": "user",
        },
    ]

    for user in default_users:
        db.users.update_one(
            {"username": user["username"]},
            {"$setOnInsert": user},
            upsert=True,
        )


def authenticate_user(username: str, password: str):
    user = db.users.find_one({"username": username})
    if not user:
        return None

    input_hash = _hash_password(password)
    stored_password = user.get("password", "")

    # Migration-safe check for older plaintext passwords already in DB.
    if stored_password not in {input_hash, password}:
        return None

    if stored_password == password:
        db.users.update_one(
            {"username": username},
            {"$set": {"password": input_hash}},
        )

    return {
        "username": user.get("username"),
        "email": user.get("email", ""),
        "role": user.get("role", "user"),
    }


def register_user(username: str, password: str, email: str):
    existing = db.users.find_one({"username": username}, {"_id": 1})
    if existing:
        return {"success": False, "message": "Username already exists"}

    db.users.insert_one(
        {
            "username": username,
            "password": _hash_password(password),
            "email": email,
            "role": "user",
        }
    )
    return {"success": True}


def create_session(response: Response, user: dict):
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = user
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,
        max_age=86400,
        samesite="lax",
    )


def get_current_user(request: Request) -> Optional[dict]:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    return SESSIONS.get(token)


def logout_user(response: Response):
    response.delete_cookie(SESSION_COOKIE)


def get_all_users() -> list:
    users = db.users.find({}, {"_id": 0, "password": 0})
    return list(users)

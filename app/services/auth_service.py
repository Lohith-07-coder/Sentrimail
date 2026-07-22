"""Authentication and user-management business logic."""

import hashlib
from typing import Any, Optional, TypedDict

from fastapi import Request
from fastapi.responses import Response

from app.repositories.user_repository import user_repository
from app.core.config import get_settings
from app.core.security import (
    TokenVerificationError,
    create_access_token,
    create_refresh_token,
    verify_token,
)


class AuthenticatedUser(TypedDict):
    """The safe user data exposed to application routes and templates."""

    username: str
    email: str
    role: str


class RegistrationResult(TypedDict, total=False):
    """Outcome returned when a user registration is attempted."""

    success: bool
    message: str


class AuthService:
    """Encapsulate authentication, registration, and JWT cookie operations."""

    @staticmethod
    def _hash_password(password: str) -> str:
        """Hash a password using the legacy-compatible digest currently in use."""
        return hashlib.sha256(password.encode()).hexdigest()

    def ensure_default_users(self) -> None:
        """Create the development users only when they do not already exist."""
        default_users = (
            ("admin", "admin123", "admin@sentrimail.io", "admin"),
            ("alice", "alice123", "alice@example.com", "user"),
            ("bob", "bob123", "bob@example.com", "user"),
        )
        for username, password, email, role in default_users:
            user_repository.create_if_absent(
                {
                    "username": username,
                    "password": self._hash_password(password),
                    "email": email,
                    "role": role,
                }
            )

    def authenticate_user(self, username: str, password: str) -> Optional[AuthenticatedUser]:
        """Verify credentials and return only safe user fields on success."""
        user: Optional[dict[str, Any]] = user_repository.find_by_username(username)
        if not user:
            return None

        password_hash = self._hash_password(password)
        stored_password = user.get("password", "")
        if stored_password not in {password_hash, password}:
            return None

        # Preserve existing accounts while upgrading legacy plaintext records on login.
        if stored_password == password:
            user_repository.update_password(username, password_hash)

        return {
            "username": user.get("username", ""),
            "email": user.get("email", ""),
            "role": user.get("role", "user"),
        }

    def register_user(self, username: str, password: str, email: str) -> RegistrationResult:
        """Create a user unless the username already exists."""
        if user_repository.username_exists(username):
            return {"success": False, "message": "Username already exists"}

        user_repository.create(
            {
                "username": username,
                "password": self._hash_password(password),
                "email": email,
                "role": "user",
            }
        )
        return {"success": True}

    def create_session(self, response: Response, user: AuthenticatedUser) -> None:
        """Issue JWT cookies while preserving the established login route contract."""
        settings = get_settings()
        response.set_cookie(
            key=settings.session_cookie_name,
            value=create_access_token(user),
            httponly=True,
            secure=settings.secure_cookies,
            max_age=settings.access_token_expire_minutes * 60,
            samesite=settings.cookie_samesite,
        )
        response.set_cookie(
            key=settings.refresh_cookie_name,
            value=create_refresh_token(user),
            httponly=True,
            secure=settings.secure_cookies,
            max_age=settings.refresh_token_expire_days * 86_400,
            samesite=settings.cookie_samesite,
        )

    def get_current_user(self, request: Request) -> Optional[AuthenticatedUser]:
        """Resolve the user from a valid access-token cookie."""
        token = request.cookies.get(get_settings().session_cookie_name)
        if not token:
            return None
        try:
            payload = verify_token(token, expected_type="access")
        except (TokenVerificationError, RuntimeError):
            return None
        return {
            "username": payload["sub"],
            "email": payload.get("email", ""),
            "role": payload.get("role", "user"),
        }

    def refresh_session(self, request: Request, response: Response) -> bool:
        """Issue a fresh token pair when presented with a valid refresh token."""
        settings = get_settings()
        token = request.cookies.get(settings.refresh_cookie_name)
        if not token:
            return False
        try:
            payload = verify_token(token, expected_type="refresh")
        except (TokenVerificationError, RuntimeError):
            return False
        self.create_session(
            response,
            {
                "username": payload["sub"],
                "email": payload.get("email", ""),
                "role": payload.get("role", "user"),
            },
        )
        return True

    def logout_user(self, response: Response) -> None:
        """Clear both JWT cookies from the browser."""
        settings = get_settings()
        response.delete_cookie(settings.session_cookie_name, samesite=settings.cookie_samesite)
        response.delete_cookie(settings.refresh_cookie_name, samesite=settings.cookie_samesite)

    def get_all_users(self) -> list[dict[str, Any]]:
        """Return users without password hashes for the administration page."""
        return user_repository.list_without_passwords()


auth_service = AuthService()

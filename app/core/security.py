"""JWT creation and verification utilities."""

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

import jwt
from jwt import InvalidTokenError

from app.core.config import Settings, get_settings


class TokenVerificationError(Exception):
    """Raised when a token is absent, invalid, expired, or used for the wrong purpose."""


def _signing_key(settings: Settings) -> str:
    """Return configured signing key or a safe fallback for local development."""
    key = settings.jwt_secret_key or settings.secret_key
    if not key or not key.strip():
        return "sentrimail-dev-fallback-secret-key-32bytes"
    return key


def create_token(
    user: Mapping[str, str], token_type: str, expires_delta: timedelta
) -> str:
    """Create a signed access or refresh token containing safe user claims."""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": user["username"],
        "email": user.get("email", ""),
        "role": user.get("role", "user"),
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
    }
    return jwt.encode(payload, _signing_key(settings), algorithm=settings.jwt_algorithm)


def create_access_token(user: Mapping[str, str]) -> str:
    """Create a short-lived access token."""
    settings = get_settings()
    return create_token(
        user,
        token_type="access",
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes),
    )


def create_refresh_token(user: Mapping[str, str]) -> str:
    """Create a long-lived token used only to obtain a fresh access token."""
    settings = get_settings()
    return create_token(
        user,
        token_type="refresh",
        expires_delta=timedelta(days=settings.refresh_token_expire_days),
    )


def verify_token(token: str, expected_type: str) -> dict[str, Any]:
    """Verify a token signature, expiration, required claims, and token purpose."""
    settings = get_settings()
    try:
        payload = jwt.decode(token, _signing_key(settings), algorithms=[settings.jwt_algorithm])
    except InvalidTokenError as exc:
        raise TokenVerificationError("Invalid or expired token") from exc

    if payload.get("type") != expected_type or not isinstance(payload.get("sub"), str):
        raise TokenVerificationError("Invalid token claims")
    return payload

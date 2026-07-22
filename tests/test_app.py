"""
Automated Integration & Structure Test Suite for SentriMail
"""

import sys
import os
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.main import app
from app.services.ai_service import analyze_complaint
from app.services.complaint_service import calculate_dashboard_stats


def test_app_imports():
    """Verify that all core modules, services, and routers import cleanly."""
    from app.core.database import db, init_mongodb
    from app.services.auth_service import auth_service
    from app.services.email_service import send_resolution_email
    from app.services.transcription_service import transcribe_audio_file
    from app.routers import admin_router, api_router, auth_router, user_router

    assert db is not None
    assert auth_service is not None


def test_ai_service_analysis():
    """Test AI complaint analysis fallback and output structure."""
    result = analyze_complaint("System crashed during checkout", category="technical")
    assert "sentiment_label" in result
    assert "emotion_label" in result
    assert "priority" in result
    assert "root_cause_summary" in result


def test_public_routes():
    """Verify public endpoints return expected HTTP status codes."""
    client = TestClient(app)

    # Root redirects to /login when unauthenticated
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/login"

    # Login page renders HTML
    response = client.get("/login")
    assert response.status_code == 200
    assert "SentriMail" in response.text or "login" in response.text.lower()

    # Register page renders HTML
    response = client.get("/register")
    assert response.status_code == 200

    # Track page renders HTML
    response = client.get("/track")
    assert response.status_code == 200

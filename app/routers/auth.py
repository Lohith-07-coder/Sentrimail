"""
Authentication & Navigation Router
------------------------------------
Handles root redirects, login, registration, logout, and JWT refresh endpoints.
"""

from typing import Annotated
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.services.auth_service import auth_service
from app.repositories.user_repository import user_repository
from app.schemas.user import LoginRequest, RegisterRequest

router = APIRouter(tags=["Authentication"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=PROJECT_ROOT / "templates")


@router.get("/", response_class=HTMLResponse)
def root(request: Request):
    try:
        user = auth_service.get_current_user(request)
    except Exception:
        user = None

    if user:
        if user.get("role") == "admin":
            return RedirectResponse("/admin/dashboard", status_code=302)
        return RedirectResponse("/user/dashboard", status_code=302)

    return RedirectResponse("/login", status_code=302)


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request, error: str = None):
    user = auth_service.get_current_user(request)
    if user:
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": error})


@router.post("/login")
def login(request: Request, credentials: Annotated[LoginRequest, Depends(LoginRequest.as_form)]):
    user = auth_service.authenticate_user(credentials.username, credentials.password)

    if not user:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Invalid credentials. Please try again."
        })

    user_repository.record_login(credentials.username, user.get("role", "user"))

    response = RedirectResponse("/", status_code=302)
    auth_service.create_session(response, user)

    return response


@router.get("/register", response_class=HTMLResponse)
def register_page(request: Request, error: str = None, success: str = None):
    return templates.TemplateResponse("register.html", {"request": request, "error": error, "success": success})


@router.post("/register")
def register(request: Request, registration: Annotated[RegisterRequest, Depends(RegisterRequest.as_form)]):
    result = auth_service.register_user(registration.username, registration.password, registration.email)
    if not result["success"]:
        return templates.TemplateResponse("register.html", {"request": request, "error": result["message"]})
    return templates.TemplateResponse("register.html", {"request": request, "success": "Account created! You can now log in."})


@router.get("/logout")
def logout(request: Request):
    response = RedirectResponse("/login", status_code=302)
    auth_service.logout_user(response)
    return response


@router.post("/api/auth/refresh")
def refresh_authentication(request: Request):
    """Refresh JWT cookies without changing the existing browser login flow."""
    response = JSONResponse({"success": True})
    if not auth_service.refresh_session(request, response):
        return JSONResponse({"success": False}, status_code=401)
    return response

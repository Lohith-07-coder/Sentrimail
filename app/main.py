"""
SentriMail — FastAPI Application Entry Point
--------------------------------------------
Assembles settings, logging, database initialization, background scheduler,
and modular routers.
"""

import os
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.core.database import init_mongodb
from app.core.logging import configure_logging
from app.routers import admin_router, api_router, auth_router, user_router
from app.services.ai_service import _load_models, _load_response_model
from app.services.auth_service import auth_service
from app.services.complaint_service import escalate_complaints

# Configure application logging
configure_logging(get_settings())

# Initialize FastAPI app instance
app = FastAPI(title="SentriMail", version="1.0.0")

# Register CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "static"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Mount modular routers
app.include_router(auth_router)
app.include_router(user_router)
app.include_router(admin_router)
app.include_router(api_router)

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

templates = Jinja2Templates(directory=PROJECT_ROOT / "templates")


@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    
    titles = {
        404: "Page Not Found",
        403: "Access Denied",
        500: "Internal Server Error"
    }
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "user": auth_service.get_current_user(request),
            "status_code": exc.status_code,
            "title": titles.get(exc.status_code, "Error"),
            "detail": exc.detail or "An unexpected error occurred."
        },
        status_code=exc.status_code
    )


# Configure background escalation scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(escalate_complaints, 'interval', hours=1)


@app.on_event("startup")
async def startup_event():
    init_mongodb()
    auth_service.ensure_default_users()
    scheduler.start()
    _load_models()
    _load_response_model()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


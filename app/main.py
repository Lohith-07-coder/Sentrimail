"""
SentriMail — FastAPI Application Entry Point
--------------------------------------------
Assembles settings, logging, database initialization, background scheduler,
and modular routers.
"""

import os
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Mount modular routers
app.include_router(auth_router)
app.include_router(user_router)
app.include_router(admin_router)
app.include_router(api_router)

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

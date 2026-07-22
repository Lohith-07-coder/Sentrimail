from app.routers.admin import router as admin_router
from app.routers.api import router as api_router
from app.routers.auth import router as auth_router
from app.routers.user import router as user_router

__all__ = ["admin_router", "api_router", "auth_router", "user_router"]

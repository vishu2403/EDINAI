"""Route exports for modular backend."""

from .admin_portal_routes import router as admin_portal_router
from .auth_routes import router as auth_router
from ..contact.routes import router as contact_router
from .dashboard_routes import router as dashboard_router
from .chapter_material_routes import router as chapter_material_router
from .lecture_routes import router as lecture_router
from .public_lecture_routes import router as public_lecture_router
from .registration_routes import router as registration_router
from .student_portal_routes import router as student_portal_router
from .student_management_routes import router as student_management_router
from .student_routes import router as student_router
from .system_routes import router as system_router
from .teacher_routes import router as teacher_router
from .user_routes import router as user_router

from .super_admin_routes import router as super_admin_router

__all__ = [
    "admin_portal_router",
    "auth_router",
    "contact_router",
    "dashboard_router",
    "chapter_material_router",
    "lecture_router",
    "public_lecture_router",
    "registration_router",
    "student_portal_router",
    "super_admin_router",
    "system_router",
    "student_management_router",
    "student_router",
    "teacher_router",
    "user_router",
]

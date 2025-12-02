from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

# =======================
# REQUEST SCHEMAS
# =======================
class CreateLectureRequest(BaseModel):
    text: str
    language: Optional[str] = None
    duration: Optional[int] = None
    style: Optional[str] = None
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AskQuestionRequest(BaseModel):
    lecture_id: str
    question: str
    answer_type: Optional[str] = None
    is_edit_command: Optional[bool] = False
    context_override: Optional[str] = None

# =======================
# RESPONSE SCHEMAS
# =======================
class LectureResponse(BaseModel):
    lecture_id: str
    title: str
    language: Optional[str]
    style: Optional[str]
    requested_duration: Optional[int]
    estimated_duration: Optional[int]
    total_slides: Optional[int]
    slides: Optional[List[Dict[str, Any]]]
    context: Optional[str]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    fallback_used: Optional[bool]
    source_text: Optional[str]
    metadata: Optional[Dict[str, Any]] = None


class LectureSummaryResponse(BaseModel):
    lecture_id: str
    title: str
    language: Optional[str]
    created_at: Optional[datetime]
    lecture_url: Optional[str] = None
    std: Optional[str] = None


class AnswerResponse(BaseModel):
    answer: Optional[str]
    edited_content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    status: str
    message: str


class GenerationStatus(BaseModel):
    progress: int
    status: str
    details: Optional[Dict[str, Any]] = None


class LectureDashboardResponse(BaseModel):
    total_lectures: int
    recent_lectures: List[LectureSummaryResponse]
    metadata: Optional[Dict[str, Any]] = None

class LectureUpdate(BaseModel):
    title: Optional[str] = None
    text: Optional[str] = None
    language: Optional[str] = None
    duration: Optional[int] = None
    style: Optional[str] = None
    metadata: Optional[dict] = None


# Backwards compatibility alias for legacy imports
LectureCreate = CreateLectureRequest
# =======================
# EXPORTS
# =======================
__all__ = [
    "CreateLectureRequest",
    "LectureCreate",
    "AskQuestionRequest",
    "LectureResponse",
    "LectureSummaryResponse",
    "AnswerResponse",
    "ErrorResponse",
    "GenerationStatus",
    "LectureDashboardResponse",
    "LectureUpdate",
]
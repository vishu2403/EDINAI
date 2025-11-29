# app/schemas/chapter_material_schema.py

from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator


class TopicExtractRequest(BaseModel):
    material_ids: List[int]


class LectureGenerationRequest(BaseModel):
    material_id: int
    selected_topic_indices: List[int]
    language: Optional[str] = Field(default=None)
    duration: Optional[int] = Field(default=None, ge=5, le=180)
    style: Optional[str] = Field(default="storytelling")
    title: Optional[str] = Field(default=None)
    topics_override: Optional[List[Dict[str, Any]]] = Field(default=None)
    source_material_ids: Optional[List[int]] = Field(default=None)

    @validator("selected_topic_indices")
    def ensure_topics_selected(cls, value: List[int]) -> List[int]:
        if not value:
            raise ValueError("At least one topic must be selected")
        return value


class LectureChatRequest(BaseModel):
    question: str
    answer_type: Optional[str] = Field(default=None)
    is_edit_command: bool = Field(default=False)
    context_override: Optional[str] = Field(default=None)


class ManualTopicCreate(BaseModel):
    title: str
    summary: Optional[str] = None
    subtopics: Optional[List[Dict[str, str]]] = None


class SubtopicCreate(BaseModel):
    title: Optional[str] = None
    narration: Optional[str] = None


class AssistantSuggestRequest(BaseModel):
    user_query: str = Field(..., min_length=1)
    temperature: Optional[float] = Field(default=0.2, ge=0, le=1)
    plan_label: Optional[str] = Field(default=None)


class AssistantAddTopicsRequest(BaseModel):
    selected_suggestions: List[Dict[str, str]] = Field(..., min_items=1)


class ChapterMaterialCreate(BaseModel):
    """Schema for creating a chapter material"""
    std: str
    subject: str
    sem: Optional[str] = ""
    board: Optional[str] = ""
    chapter_number: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "std": "10",
                "subject": "Mathematics",
                "sem": "1",
                "board": "CBSE",
                "chapter_number": "1"
            }
        }


class ChapterMaterialResponse(BaseModel):
    """Schema for chapter material response"""
    id: int
    admin_id: int
    std: str
    subject: str
    sem: Optional[str]
    board: Optional[str]
    chapter_number: str
    file_name: str
    file_path: str
    file_size: int
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 1,
                "admin_id": 1,
                "std": "10",
                "subject": "Mathematics",
                "sem": "1",
                "board": "CBSE",
                "chapter_number": "1",
                "file_name": "chapter1.pdf",
                "file_path": "/uploads/chapter_materials/admin_1/chapter1.pdf",
                "file_size": 1024000,
                "created_at": "2025-01-15T10:30:00",
                "updated_at": "2025-01-15T10:30:00"
            }
        }


class ResponseBase(BaseModel):
    """Base response schema"""
    status: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": True,
                "message": "Operation successful",
                "data": {}
            }
        }
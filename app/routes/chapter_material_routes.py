# app/routes/chapter_material_routes.py

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status, Query, Request, Body
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, or_

from app.config import get_settings
from app.database import get_db
from app.models import ChapterMaterial, LectureGen
from app.schemas.admin_schema import WorkType
from app.schemas.chapter_material_schema import (
    TopicExtractRequest,
    LectureGenerationRequest,
    LectureConfigRequest,
    LectureChatRequest,
    LectureLookupRequest,
    ManualTopicCreate,
    AssistantSuggestRequest,
    AssistantAddTopicsRequest,
    MultipleChapterSelectionRequest,
    CreateMergedLectureRequest,
    TopicSelection,
    ResponseBase,
    ChapterOverviewUpdate,
)
from app.repository import auth_repository, registration_repository
from app.repository.chapter_material_repository import (
    create_chapter_material,
    get_chapter_material,
    list_chapter_materials,
    list_recent_chapter_materials,
    update_chapter_material_db,
    delete_chapter_material_db,
    get_dashboard_stats,
    get_chapter_overview_data,
    _load_topics_path,
    load_material_topics,
    persist_material_topics,
    save_extracted_topics_files,
    read_topics_file_if_exists,
    append_manual_topic_to_file,
    add_assistant_topics_to_file,
    topic_to_text,
    read_pdf_context_for_material,
    LANGUAGE_OUTPUT_RULES,
    SUPPORTED_LANGUAGES,
    DURATION_OPTIONS,
    persist_assistant_suggestions,
    get_cached_suggestions_by_ids,
    get_topics_by_ids,
    get_chapter_filter_options,
    list_chapters_for_selection,
    update_chapter_overview_fields,
)
from app.utils.file_handler import (
    save_uploaded_file,
    ALLOWED_PDF_EXTENSIONS,
    ALLOWED_PDF_TYPES,
    UPLOAD_DIR,
)
from app.services.lecture_service import LectureService
from app.utils.dependencies import admin_required, get_current_user
from groq import Groq

from fastapi.responses import FileResponse, JSONResponse
import os
from pathlib import Path
from datetime import datetime
router = APIRouter(prefix="/chapter-materials", tags=["Chapter Materials"])
def get_lecture_service(db: Session = Depends(get_db)) -> LectureService:
    return LectureService(db=db)

logger = logging.getLogger(__name__)

# Local constants (kept same)
PDF_MAX_SIZE = 15 * 1024 * 1024  # 15MB
DEFAULT_MIN_DURATION = 5
DEFAULT_MAX_DURATION = 180
MAX_ASSISTANT_SUGGESTIONS = 10
DEFAULT_LANGUAGE_CODE = "eng"

MERGED_LECTURES_DIR = Path("./storage/merged_lectures")

PLAN_SUGGESTION_LIMITS = {
    "20k": 2,
    "50k": 5,
    "100k": 8,
}


# -------------------------
# Chapter filters
# -------------------------


@router.get("/chapters/filters", response_model=ResponseBase)
async def get_chapter_filters(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ResponseBase:
    admin_id = _resolve_admin_id(current_user)
    filters = get_chapter_filter_options(db, admin_id=admin_id)
    return ResponseBase(status=True, message="Chapter filters fetched", data=filters)


@router.get("/chapters", response_model=ResponseBase)
async def list_chapters_endpoint(
    std: str = Query(..., description="Class/standard"),
    subject: str = Query(..., description="Subject"),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ResponseBase:
    admin_id = _resolve_admin_id(current_user)
    chapters = list_chapters_for_selection(
    	db,
        admin_id=admin_id,
        std=std,
        subject=subject,
    )
    return ResponseBase(
        status=True,
        message="Chapters fetched successfully",
        data={"chapters": chapters},
    )


# -------------------------
# Helper to resolve admin_id from current_user
# -------------------------
def _resolve_member_admin_id(current_user: dict) -> Optional[int]:
    user_obj = current_user.get("user_obj")
    if user_obj is None:
        return current_user.get("admin_id")
    if isinstance(user_obj, dict):
        return user_obj.get("admin_id") or current_user.get("admin_id")
    return getattr(user_obj, "admin_id", None) or current_user.get("admin_id")


def _resolve_admin_id(current_user: dict) -> int:
    if current_user["role"] == "admin":
        return current_user["id"]
    if current_user["role"] == "member":
        if current_user.get("work_type") != WorkType.CHAPTER.value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access restricted to chapter management members",
            )
        resolved = _resolve_member_admin_id(current_user)
        if resolved is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Member admin not found")
        return resolved
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")


def _normalize_plan_label(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    value = value.strip().lower()
    if value in PLAN_SUGGESTION_LIMITS:
        return value
    return None


def _ensure_lecture_config_access(current_user: dict) -> None:
    if current_user["role"] == "admin":
        return
    if current_user["role"] == "member":
        if current_user.get("work_type") != WorkType.CHAPTER.value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only chapter members can access lecture configuration",
            )
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unsupported role")


def _build_lecture_config_response(
    *,
    requested_language: Optional[str],
    requested_duration: Optional[int],
) -> Dict[str, Any]:
    settings = get_settings()
    default_language = settings.dict().get("default_language") or getattr(settings, "default_language", None)
    language_value = default_language or SUPPORTED_LANGUAGES[0]["value"]

    if requested_language:
        normalized_language = next(
            (
                option["value"]
                for option in SUPPORTED_LANGUAGES
                if option["value"].lower() == requested_language.strip().lower()
            ),
            None,
        )
        if normalized_language:
            language_value = normalized_language

    configured_default_duration = (
        getattr(settings, "default_lecture_duration", None)
        or settings.dict().get("default_lecture_duration")
        or DURATION_OPTIONS[0]
    )
    selected_duration = configured_default_duration
    if requested_duration and requested_duration in DURATION_OPTIONS:
        selected_duration = requested_duration

    return {
        "default_duration": configured_default_duration,
        "selected_duration": selected_duration,
        "min_duration": DEFAULT_MIN_DURATION,
        "max_duration": DEFAULT_MAX_DURATION,
        "default_language": language_value,
        "selected_language": language_value,
        "languages": SUPPORTED_LANGUAGES,
        "durations": DURATION_OPTIONS,
        "requested_language": requested_language,
        "requested_duration": requested_duration,
    }


def _normalize_requested_language(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip().lower()
    for option in SUPPORTED_LANGUAGES:
        if option["value"].strip().lower() == normalized:
            return option["value"]
    return None


def _normalize_requested_duration(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return value if value in DURATION_OPTIONS else None


def _save_merged_lecture_payload(lecture_id: str, payload: Dict[str, Any]) -> None:
    MERGED_LECTURES_DIR.mkdir(parents=True, exist_ok=True)
    file_path = MERGED_LECTURES_DIR / f"{lecture_id}.json"
    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _load_merged_lecture_payload(lecture_id: str) -> Optional[Dict[str, Any]]:
    file_path = MERGED_LECTURES_DIR / f"{lecture_id}.json"
    if not file_path.exists():
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to load merged lecture %s: %s", lecture_id, exc)
        return None


def _prepare_generation_from_material(
    *,
    request: LectureGenerationRequest,
    current_user: dict,
    db: Session,
    settings: Any,
) -> Dict[str, Any]:
    material = get_chapter_material(db, request.material_id)
    if not material:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")

    if current_user["role"] == "admin":
        if material.admin_id != current_user["id"]:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    elif current_user["role"] == "member":
        if current_user.get("work_type") != WorkType.CHAPTER.value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only chapter members can generate lectures",
            )
        member_admin_id = _resolve_member_admin_id(current_user)
        if material.admin_id != member_admin_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unsupported role")

    topics_path = _load_topics_path(material)
    if not os.path.exists(topics_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Topics not found. Please extract topics before generating a lecture.",
        )

    with open(topics_path, "r", encoding="utf-8") as fh:
        topics_payload = json.load(fh)
    topics = topics_payload.get("topics", [])
    if not topics:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No topics available for this material",
        )

    index_by_id: Dict[str, Dict[str, Any]] = {}
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        topic_id = topic.get("topic_id")
        suggestion_id = topic.get("suggestion_topic_id")
        if topic_id is not None:
            index_by_id[str(topic_id)] = topic
        if suggestion_id is not None:
            index_by_id[str(suggestion_id)] = topic

    selected_topics: List[Dict[str, Any]] = []
    missing_topic_ids: List[str] = []
    for tid in request.selected_topic_ids or []:
        entry = index_by_id.get(str(tid))
        if entry:
            selected_topics.append(entry)
        else:
            missing_topic_ids.append(str(tid))

    if missing_topic_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Topics not found for IDs: {', '.join(missing_topic_ids)}",
        )

    if not selected_topics:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Selected topics are invalid",
        )

    aggregate_text_parts = [topic_to_text(topic) for topic in selected_topics]
    aggregate_text = "\n\n".join(part for part in aggregate_text_parts if part)
    if not aggregate_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to extract content from selected topics",
        )

    topics_language_label = topics_payload.get("language_label") if topics_payload else None
    language = topics_language_label or "English"
    override_language = _normalize_requested_language(request.language)
    if override_language:
        language = override_language

    duration = settings.default_lecture_duration
    override_duration = _normalize_requested_duration(request.duration)
    if override_duration is not None:
        duration = override_duration
    title = f"{material.subject} Lecture" if material.subject else "Generated Lecture"

    metadata = {
        "material_id": material.id,
        "material_subject": material.subject,
        "selected_topic_ids": request.selected_topic_ids,
        "topics_source_file": topics_path,
        "language_label": topics_payload.get("language_label") if topics_payload else None,
        "language_code": topics_payload.get("language_code") if topics_payload else None,
        "std": material.std,
        "subject": material.subject,
        "requested_language": override_language,
        "requested_duration": override_duration,
    }

    std_value = material.std or "general"
    subject_value = material.subject or "subject"
    std_slug = std_value.replace(" ", "_").lower()
    subject_slug = subject_value.replace(" ", "_").lower()

    log_context = {
        "material_id": material.id,
        "std": material.std or "N/A",
        "subject": material.subject or "N/A",
        "board": material.board or "N/A",
        "sem": material.sem or "N/A",
        "selected_topics_count": len(selected_topics),
        "materials_count": 1,
        "merged_lecture_id": None,
    }

    return {
        "aggregate_text": aggregate_text,
        "language": language,
        "duration": duration,
        "title": title,
        "metadata": metadata,
        "std_slug": std_slug,
        "subject_slug": subject_slug,
        "log_context": log_context,
        "material_snapshot": {
            "id": material.id,
            "admin_id": material.admin_id,
            "std": material.std,
            "subject": material.subject,
            "board": material.board,
            "sem": material.sem,
            "chapter_number": material.chapter_number,
            "file_name": material.file_name,
        },
        "requested_language": override_language,
        "requested_duration": override_duration,
    }


def _prepare_generation_from_merged(
    *,
    request: LectureGenerationRequest,
    current_user: dict,
    db: Session,
    settings: Any,
) -> Dict[str, Any]:
    lecture_id = request.merged_lecture_id or ""
    merged_payload = _load_merged_lecture_payload(lecture_id)
    if not merged_payload:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Merged lecture not found")

    creator_admin_id = merged_payload.get("admin_id")

    response_payload = (merged_payload.get("response") or {}).get("data") or {}
    merged_topics = response_payload.get("merged_topics") or []

    aggregated_topics: List[Dict[str, Any]] = []
    candidate_material_ids: List[Any] = []
    for block in merged_topics:
        material_id_value = block.get("material_id")
        if material_id_value is not None:
            candidate_material_ids.append(material_id_value)
        for topic in block.get("topics") or []:
            if isinstance(topic, dict):
                aggregated_topics.append(topic)

    primary_material = None
    inferred_admin_id = None
    for candidate in candidate_material_ids:
        try:
            candidate_id = int(candidate)
        except (TypeError, ValueError):
            continue
        material = get_chapter_material(db, candidate_id)
        if material and material.admin_id == creator_admin_id:
            primary_material = material
            break
        if material and not inferred_admin_id:
            inferred_admin_id = material.admin_id
            if primary_material is None:
                primary_material = material

    if creator_admin_id is None:
        creator_admin_id = inferred_admin_id

    if creator_admin_id is None:
        try:
            creator_admin_id = _resolve_admin_id(current_user)
        except HTTPException:
            creator_admin_id = _resolve_member_admin_id(current_user)

    if creator_admin_id is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Merged lecture metadata missing admin reference",
        )

    if current_user["role"] == "admin":
        if current_user["id"] != creator_admin_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    elif current_user["role"] == "member":
        if current_user.get("work_type") != WorkType.CHAPTER.value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only chapter members can generate lectures",
            )
        member_admin_id = _resolve_member_admin_id(current_user)
        if member_admin_id != creator_admin_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unsupported role")

    if not aggregated_topics:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Merged lecture does not contain topics to generate from",
        )

    aggregate_text_parts = [topic_to_text(topic) for topic in aggregated_topics]
    aggregate_text = "\n\n".join(part for part in aggregate_text_parts if part).strip()
    if not aggregate_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to extract content from merged lecture topics",
        )

    language = "English"
    override_language = _normalize_requested_language(request.language)
    if override_language:
        language = override_language

    duration = settings.default_lecture_duration
    override_duration = _normalize_requested_duration(request.duration)
    if override_duration is not None:
        duration = override_duration
    title = response_payload.get("title") or f"Merged Lecture {lecture_id}"

    if primary_material:
        std_value = primary_material.std or "general"
        subject_value = primary_material.subject or "subject"
    else:
        std_value = "merged"
        subject_value = "combined"

    std_slug = std_value.replace(" ", "_").lower()
    subject_slug = subject_value.replace(" ", "_").lower()

    metadata: Dict[str, Any] = {
        "source": "merged_topics",
        "merged_lecture_id": lecture_id,
        "materials_count": response_payload.get("materials_count"),
        "topics_count": response_payload.get("topics_count"),
        "selected_material_ids": response_payload.get("selected_materials"),
    }

    if primary_material:
        metadata.update(
            {
                "primary_material_id": primary_material.id,
                "primary_material_subject": primary_material.subject,
                "primary_material_std": primary_material.std,
            }
        )

    log_context = {
        "material_id": primary_material.id if primary_material else None,
        "std": primary_material.std if primary_material else "Merged",
        "subject": primary_material.subject if primary_material else "Merged Topics",
        "board": primary_material.board if primary_material else "Mixed",
        "sem": primary_material.sem if primary_material else "Mixed",
        "selected_topics_count": response_payload.get("topics_count") or len(aggregated_topics),
        "materials_count": response_payload.get("materials_count") or len(candidate_material_ids),
        "merged_lecture_id": lecture_id,
    }

    material_snapshot = {
        "id": primary_material.id if primary_material else None,
        "admin_id": primary_material.admin_id if primary_material else creator_admin_id,
        "std": primary_material.std if primary_material else log_context["std"],
        "subject": primary_material.subject if primary_material else log_context["subject"],
        "board": primary_material.board if primary_material else log_context["board"],
        "sem": primary_material.sem if primary_material else log_context["sem"],
        "chapter_number": getattr(primary_material, "chapter_number", None) if primary_material else None,
        "file_name": getattr(primary_material, "file_name", None) if primary_material else None,
    }

    return {
        "aggregate_text": aggregate_text,
        "language": language,
        "duration": duration,
        "title": title,
        "metadata": metadata,
        "std_slug": std_slug,
        "subject_slug": subject_slug,
        "log_context": log_context,
        "material_snapshot": material_snapshot,
        "requested_language": override_language,
        "requested_duration": override_duration,
    }


def _resolve_plan_label_for_admin(admin_id: int, current_user: dict, override: Optional[str] = None) -> Optional[str]:
    plan_label = _normalize_plan_label(override)
    if plan_label:
        return plan_label

    if current_user.get("role") == "admin" and current_user.get("id") == admin_id:
        current_package = current_user.get("package") or current_user.get("user_obj", {}).get("package_plan")
        normalized = _normalize_plan_label(current_package)
        if normalized:
            return normalized

    admin_record = auth_repository.get_admin_by_id(admin_id)
    if admin_record:
        normalized = _normalize_plan_label(admin_record.get("package"))
        if normalized:
            return normalized

    reg_admin = registration_repository.get_admin_by_id(admin_id)
    if reg_admin:
        normalized = _normalize_plan_label(reg_admin.get("package_plan") or reg_admin.get("package"))
        if normalized:
            return normalized

    return None


# -------------------------
# Generated lectures listing
# -------------------------
@router.get("/chapter_lectures", response_model=ResponseBase)
async def list_generated_lectures(
    std: Optional[str] = Query(default=None, description="Filter by class/standard"),
    subject: Optional[str] = Query(default=None, description="Filter by subject"),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from pathlib import Path
    admin_id = _resolve_admin_id_for_lecture_access(current_user)

    query = (
        db.query(
            LectureGen,
            ChapterMaterial.chapter_number.label("chapter_name"),
            ChapterMaterial.subject.label("material_subject"),
            ChapterMaterial.std.label("material_std"),
            ChapterMaterial.sem.label("material_sem"),
            ChapterMaterial.board.label("material_board"),
        )
        .filter(LectureGen.admin_id == admin_id)
        .join(ChapterMaterial, ChapterMaterial.id == LectureGen.material_id, isouter=True)
    )

    if std:
        query = query.filter(LectureGen.std == std)
    if subject:
        query = query.filter(LectureGen.subject == subject)

    records = query.order_by(desc(LectureGen.created_at)).all()

    storage_base = Path("./storage/chapter_lectures")
    items: List[Dict[str, Any]] = []

    for (
        record,
        chapter_name,
        material_subject,
        material_std,
        material_sem,
        material_board,
    ) in records:
        # Get topics for this material if they exist
        topics_data = []
        extracted_chapter_title = None
        try:
            from app.repository.chapter_material_repository import read_topics_file_if_exists
            payload, topics_list = read_topics_file_if_exists(record.admin_id, record.material_id)
            if topics_list:
                # Take only first few topics as preview
                topics_data = topics_list[:5]  # Limit to 5 topics for overview
            # Get chapter title from extracted topics data
            if payload and payload.get("chapter_title"):
                extracted_chapter_title = payload.get("chapter_title")
        except Exception:
            topics_data = []
        
        # Use extracted chapter title if available, otherwise fallback to chapter_name
        chapter_title = extracted_chapter_title or chapter_name
        
        # Calculate lecture size if lecture exists
        lecture_size = 0
        video_info = None
        if record.lecture_link:
            try:
                import os
                storage_base = Path("./storage/chapter_lectures")
                lecture_json_path = storage_base / record.lecture_uid / "lecture.json"
                if lecture_json_path.exists():
                    lecture_size = lecture_json_path.stat().st_size
                    # For video info, we'll assume video exists alongside JSON
                    video_path = storage_base / record.lecture_uid / "lecture.mp4"
                    if video_path.exists():
                        video_size = video_path.stat().st_size
                        video_info = {
                            "size": video_size,
                            "path": str(video_path)
                        }
            except Exception:
                pass
        
        # Format file size
        def formatFileSize(bytes):
            """Format file size in human readable format"""
            if not bytes:
                return '41.1 KB'
            
            mb = bytes / (1024 * 1024)
            if mb < 0.5:
                kb = bytes / 1024
                return f"{kb:.1f} KB"
            
            return f"{mb:.2f} MB"
        
        subject_value = record.subject or material_subject or "Lecture"
        
        items.append(
            {
                "subject": subject_value,
                "chapter": chapter_title,
                "topics": topics_data,
                "size": formatFileSize(lecture_size) if lecture_size else "41.1 KB",
                "video": video_info
            }
        )

    return ResponseBase(
        status=True,
        message="Lectures fetched successfully",
        data={"items": items, "total": len(items)},
    )


@router.post("/public_lecture/start_new_lecture", response_model=ResponseBase)
async def lookup_chapter_lecture(
    request: LectureLookupRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    admin_id = _resolve_admin_id(current_user)

    std_value = request.std.strip().lower()
    subject_value = request.subject.strip().lower()
    chapter_value = request.chapter_title.strip().lower()

    material = (
        db.query(ChapterMaterial)
        .filter(ChapterMaterial.admin_id == admin_id)
        .filter(func.lower(ChapterMaterial.std) == std_value)
        .filter(func.lower(ChapterMaterial.subject) == subject_value)
        .filter(
            or_(
                func.lower(ChapterMaterial.chapter_number) == chapter_value,
                func.lower(func.coalesce(ChapterMaterial.chapter_title, "")) == chapter_value,
            )
        )
        .order_by(desc(ChapterMaterial.updated_at))
        .first()
    )

    if not material:
        return ResponseBase(
            status=True,
            message="No lecture found for the provided information",
            data={"has_lecture": False},
        )

    lecture_record = (
        db.query(LectureGen)
        .filter(LectureGen.admin_id == admin_id, LectureGen.material_id == material.id)
        .order_by(desc(LectureGen.created_at))
        .first()
    )

    material_summary = {
        "id": material.id,
        "std": material.std,
        "subject": material.subject,
        "chapter_number": material.chapter_number,
        "chapter_title": material.chapter_title,
        "sem": material.sem,
        "board": material.board,
    }

    if not lecture_record:
        return ResponseBase(
            status=True,
            message="No lecture exists for this chapter",
            data={"has_lecture": False, "material": material_summary},
        )

    topics_preview: List[Dict[str, Any]] = []
    topics_count = 0
    chapters_count = 1
    try:
        _, topics_list = read_topics_file_if_exists(material.admin_id, material.id)
        if topics_list:
            topics_count = len(topics_list)
            topics_preview = [
                {
                    "title": topic.get("title"),
                    "summary": topic.get("summary"),
                    "subtopics": topic.get("subtopics", []),
                }
                for topic in topics_list[:5]
                if isinstance(topic, dict)
            ]
    except Exception as exc:
        logger.warning("Failed to load topics for material %s: %s", material.id, exc)

    lecture_info = {
        "has_lecture": True,
        "lecture_uid": lecture_record.lecture_uid,
        "lecture_title": lecture_record.lecture_title,
        "lecture_link": lecture_record.lecture_link,
        "std": lecture_record.std or material.std,
        "subject": lecture_record.subject or material.subject,
        "sem": lecture_record.sem or material.sem,
        "board": lecture_record.board or material.board,
        "created_at": lecture_record.created_at.isoformat() if lecture_record.created_at else None,
        "chapters_count": chapters_count,
        "topics_count": topics_count,
        "topics_preview": topics_preview,
        "material": material_summary,
    }

    return ResponseBase(
        status=True,
        message="Lecture found",
        data=lecture_info,
    )


def _resolve_admin_id_for_lecture_access(current_user: dict) -> int:
    if current_user["role"] == "admin":
        return current_user["id"]
    if current_user["role"] == "member":
        allowed = {WorkType.CHAPTER.value, WorkType.LECTURE.value}
        if current_user.get("work_type") not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access restricted to chapter or lecture members",
            )
        resolved = _resolve_member_admin_id(current_user)
        if resolved is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Member admin not found")
        return resolved
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")


# -------------------------
# Dashboard endpoint
# -------------------------
@router.get("/dashboard")
async def get_chapter_dashboard(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        admin_id = _resolve_admin_id(current_user)
        stats = get_dashboard_stats(db, admin_id)
        chapter_overview = get_chapter_overview_data(db, admin_id)
        return {
            "status": True,
            "message": "Dashboard data retrieved successfully",
            "data": {
                "chapter_metrics": stats,
                "chapter_overview": chapter_overview
            },
        }
    except Exception as e:
        logger.exception("Error fetching dashboard data")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# -------------------------
# Upload / CRUD endpoints
# -------------------------
@router.post("/upload")
async def upload_chapter_material(
    std: str = Form(...),
    subject: str = Form(...),
    sem: str = Form(default=""),
    board: str = Form(default=""),
    chapter_number: str = Form(...),
    pdf_file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if pdf_file.content_type not in ALLOWED_PDF_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed.",
        )

    admin_id = _resolve_admin_id(current_user)

    file_info = await save_uploaded_file(
        pdf_file,
        subfolder=f"chapter_materials/admin_{admin_id}",
        allowed_extensions=ALLOWED_PDF_EXTENSIONS,
        allowed_types=ALLOWED_PDF_TYPES,
        max_size=PDF_MAX_SIZE,
    )

    chapter_material = create_chapter_material(
        db,
        admin_id=admin_id,
        std=std,
        subject=subject,
        sem=sem,
        board=board,
        chapter_number=chapter_number,
        file_info=file_info,
    )

    return {
        "status": True,
        "message": "Chapter material uploaded successfully",
        "data": {"material": chapter_material.to_dict() if hasattr(chapter_material, "to_dict") else chapter_material.__dict__},
    }


@router.post("/chapter-suggestion")
async def list_chapter_materials_post(
    request_data: dict = Body(...),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        admin_id = _resolve_admin_id(current_user)
        
        # Extract parameters from body
        std = request_data.get("std")
        subject = request_data.get("subject")
        # Validate parameters
        if std is not None and not str(std).strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Standard cannot be empty")
        if subject is not None and not str(subject).strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Subject cannot be empty")
        
        # Get materials with filtering
        materials = list_chapter_materials(db, admin_id, str(std).strip() if std else None, str(subject).strip() if subject else None)
        
        # Serialize materials with clean structure
        serialized = []
        for material in materials:
            try:
                # Clean data structure - only include necessary fields
                clean_material = {
                    "id": material.id,
                    "std": material.std,
                    "subject": material.subject,
                    "sem": material.sem,
                    "board": material.board,
                    "chapter_number": material.chapter_number,
                    "file_name": material.file_name,
                    "file_size": material.file_size,
                    "file_path": material.file_path,
                    "created_at": material.created_at.isoformat() if hasattr(material.created_at, 'isoformat') else str(material.created_at),
                    "updated_at": material.updated_at.isoformat() if hasattr(material.updated_at, 'isoformat') else str(material.updated_at)
                }
                serialized.append(clean_material)
            except Exception as e:
                logger.warning(f"Error serializing material {material.id}: {str(e)}")
                continue
        
        logger.info(f"Returning {len(serialized)} materials for std='{std}', subject='{subject}'")
        
        return {
            "status": True, 
            "message": "Chapter materials retrieved successfully", 
            "data": {
                "materials": serialized,
                "total": len(serialized),
                "filters_applied": {
                    "std": std,
                    "subject": subject
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in list_chapter_materials_post: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve chapter materials")


@router.get("/recent")
async def list_recent_chapter_materials_route(
    limit: int = 5,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if limit <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Limit must be a positive integer")
    admin_id = _resolve_admin_id(current_user)
    materials = list_recent_chapter_materials(db, admin_id, limit)
    serialized = [m.to_dict() if hasattr(m, "to_dict") else m.__dict__ for m in materials]
    return {"status": True, "message": "Recent chapter materials retrieved successfully", "data": {"materials": serialized}}


@router.put("/{material_id}")
async def update_chapter_material_route(
    material_id: int,
    std: str = Form(...),
    subject: str = Form(...),
    sem: str = Form(default=""),
    board: str = Form(default=""),
    chapter_number: str = Form(...),
    pdf_file: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    admin_id = _resolve_admin_id(current_user)

    chapter_material = get_chapter_material(db, material_id)
    if not chapter_material or chapter_material.admin_id != admin_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")

    new_file_info = None
    if pdf_file is not None:
        if pdf_file.content_type and pdf_file.content_type not in ALLOWED_PDF_TYPES:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are allowed.")

        new_file_info = await save_uploaded_file(
            pdf_file,
            subfolder=f"chapter_materials/admin_{admin_id}",
            allowed_extensions=ALLOWED_PDF_EXTENSIONS,
            allowed_types=ALLOWED_PDF_TYPES,
            max_size=PDF_MAX_SIZE,
        )

    updated = update_chapter_material_db(
        db,
        chapter_material,
        std=std,
        subject=subject,
        sem=sem,
        board=board,
        chapter_number=chapter_number,
        new_file_info=new_file_info,
    )

    return {"status": True, "message": "Chapter material updated successfully", "data": {"material": updated.to_dict() if hasattr(updated, "to_dict") else updated.__dict__}}


@router.patch("/{material_id}")
async def patch_chapter_overview_route(
    material_id: int,
    payload: ChapterOverviewUpdate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    admin_id = _resolve_admin_id(current_user)
    chapter_material = get_chapter_material(db, material_id)
    if not chapter_material or chapter_material.admin_id != admin_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")

    update_kwargs = payload.model_dump(exclude_unset=True)
    if not update_kwargs:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided to update")

    updated = update_chapter_overview_fields(db, chapter_material=chapter_material, **update_kwargs)

    return {
        "status": True,
        "message": "Chapter overview updated successfully",
        "data": {"material": updated.to_dict() if hasattr(updated, "to_dict") else updated.__dict__},
    }


@router.delete("/{material_id}")
async def delete_chapter_material_route(
    material_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    admin_id = _resolve_admin_id(current_user)
    chapter_material = get_chapter_material(db, material_id)
    if not chapter_material or chapter_material.admin_id != admin_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")

    delete_chapter_material_db(db, chapter_material)
    return {"status": True, "message": "Chapter material deleted successfully"}


# -------------------------
# Topic extraction endpoint (uses the user's topic_extractor)
# -------------------------
@router.post("/extract-topics")
async def extract_topics_from_materials(
    request_data: TopicExtractRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    material_ids = request_data.material_ids
    
    # Validate input
    if not material_ids or len(material_ids) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="material_ids is required and cannot be empty")
    
    if len(material_ids) > 5:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot process more than 5 materials at once")

    try:
        from app.utils.topic_extractor import extract_topics_from_pdf

        logger.info(f"Extracting topics for {len(material_ids)} materials by user {current_user.get('email')}")

        topics_by_material: List[Dict[str, Any]] = []

        for material_id in material_ids:
            entry: Dict[str, Any] = {
                "material_id": material_id,
                "chapter_title": "",
                "topics": [],
            }
            try:
                # Validate material exists and user has access
                material = get_chapter_material(db, material_id)
                if not material:
                    logger.warning(f"Material {material_id} not found")
                    topics_by_material.append(entry)
                    continue

                # Permission check
                if current_user["role"] == "admin":
                    if material.admin_id != current_user["id"]:
                        logger.warning(f"Access denied for material {material_id}")
                        topics_by_material.append(entry)
                        continue

                elif current_user["role"] == "member":
                    member_admin_id = _resolve_member_admin_id(current_user)
                    if material.admin_id != member_admin_id:
                        logger.warning(f"Access denied for material {material_id}")
                        topics_by_material.append(entry)
                        continue

                # Check file exists
                if not os.path.exists(material.file_path):
                    logger.warning(f"File not found for material {material_id}: {material.file_path}")
                    topics_by_material.append(entry)
                    continue

                # Extract topics
                extraction = extract_topics_from_pdf(Path(material.file_path))

                # Save to files
                txt_path, json_path = save_extracted_topics_files(material.admin_id, material_id, extraction)

                # Return only topics with a single chapter title value
                topics = extraction.get("topics", [])
                chapter_title = (
                    extraction.get("chapter_title")
                    or (extraction.get("chapter_titles") or [None])[0]
                    or material.chapter_number
                    or ""
                )

                entry.update({
                    "chapter_title": chapter_title,
                    "topics": topics,
                })
                topics_by_material.append(entry)
                
                logger.info(f"Successfully extracted {len(topics)} topics for material {material_id}")

            except Exception as e:
                logger.error(f"Error extracting topics for material {material_id}: {str(e)}")
                topics_by_material.append(entry)
                continue

        return {
            "status": True, 
            "message": "Topics extracted successfully", 
            "data": {
                "topics": topics_by_material,
                "total_materials": len(material_ids),
                "successful_extractions": sum(1 for item in topics_by_material if item.get("topics"))
            }
        }
        
    except Exception as exc:
        logger.exception("Unexpected error in extract_topics_from_materials")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to extract topics")




@router.get("/recent-topics")
async def list_recent_material_topics(
    limit: int = 10,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if limit <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Limit must be a positive integer")
    admin_id = _resolve_admin_id(current_user)
    materials = list_recent_chapter_materials(db, admin_id, limit)

    results = []
    for material in materials:
        payload, topics_list = read_topics_file_if_exists(material.admin_id, material.id)
        results.append({"material": material.to_dict() if hasattr(material, "to_dict") else material.__dict__, "topics": topics_list, "topics_metadata": payload})

    return {"status": True, "message": "Recent topics retrieved successfully", "data": {"items": results}}


@router.get("/{material_id}/topics")
async def get_material_topics_route(
    material_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    material = get_chapter_material(db, material_id)
    if not material:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")

    if current_user["role"] == "admin":
        if material.admin_id != current_user["id"]:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    elif current_user["role"] == "member":
        if current_user.get("work_type") != WorkType.CHAPTER.value:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access restricted to chapter management members")
        member_admin_id = _resolve_member_admin_id(current_user)
        if material.admin_id != member_admin_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unsupported role")

    payload, topics_list = read_topics_file_if_exists(material.admin_id, material.id)
    if not topics_list:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Topics not found. Please extract topics first.")

    sanitized_topics: List[Dict[str, Any]] = []
    for topic in topics_list:
        if not isinstance(topic, dict):
            continue
        topic_copy = dict(topic)
        if topic_copy.get("is_assistant_generated"):
            suggestion_value = topic_copy.get("suggestion_topic_id") or topic_copy.get("topic_id")
            if suggestion_value is not None:
                topic_copy["suggestion_topic_id"] = str(suggestion_value)
            topic_copy.pop("topic_id", None)
        else:
            topic_id_value = topic_copy.get("topic_id")
            if topic_id_value is not None:
                topic_copy["topic_id"] = str(topic_id_value)
            topic_copy.pop("suggestion_topic_id", None)
        sanitized_topics.append(topic_copy)

    return {
        "status": True,
        "message": "Topics fetched successfully",
        "data": {
            "material_id": material.id,
            "topics_count": len(sanitized_topics),
            "topic_id": sanitized_topics,
        },
    }


@router.post("/{material_id}/topics")
async def add_manual_topic_route(
    material_id: int,
    topic: ManualTopicCreate,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not topic.title.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Topic title is required")

    material = get_chapter_material(db, material_id)
    if not material:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")

    # Authorization checks
    if current_user["role"] == "admin":
        if material.admin_id != current_user["id"]:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    elif current_user["role"] == "member":
        if current_user.get("work_type") != WorkType.CHAPTER.value:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access restricted to chapter management members")
        member_admin_id = _resolve_member_admin_id(current_user)
        if material.admin_id != member_admin_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unsupported role")

    normalized_subtopics = []
    if topic.subtopics:
        for subtopic in topic.subtopics:
            if not subtopic.get("title") and not subtopic.get("narration"):
                continue
            normalized_subtopics.append({"title": (subtopic.get("title") or "").strip(), "narration": (subtopic.get("narration") or "").strip()})

    new_topic = {"title": topic.title.strip(), "summary": (topic.summary or "").strip(), "subtopics": normalized_subtopics, "is_manual": True}

    added = append_manual_topic_to_file(material.admin_id, material, new_topic)

    return {"status": True, "message": "Topic added successfully", "data": added}


# -------------------------
# Assistant suggest topics (calls Groq)
# -------------------------
@router.post("/{material_id}/assistant-suggest-topics")
async def assistant_suggest_topics(
    material_id: int,
    request: AssistantSuggestRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    material = get_chapter_material(db, material_id)
    if not material:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")

    # Authorization
    if current_user["role"] == "admin":
        admin_id = current_user["id"]
        if material.admin_id != admin_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    elif current_user["role"] == "member":
        if current_user.get("work_type") != WorkType.CHAPTER.value:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access restricted to chapter management members")
        admin_id = current_user["admin_id"]
        if material.admin_id != admin_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    topics_path = os.path.join(UPLOAD_DIR, f"chapter_materials/admin_{admin_id}", f"extracted_topics_{material.id}.json")
    if not os.path.exists(topics_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Topics not found. Please extract topics first.")

    try:
        with open(topics_path, "r", encoding="utf-8") as fh:
            topics_blob = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to read topics file") from exc

    existing_topics = topics_blob.get("topics", [])
    topics_text = topics_blob.get("topics_text", "")
    excerpt = topics_blob.get("excerpt", "")

    pdf_context_text = ""
    try:
        pdf_context_text = read_pdf_context_for_material(material.file_path) or excerpt or topics_text
    except Exception:
        pdf_context_text = excerpt or topics_text

    if not pdf_context_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="PDF content unavailable")

    condensed_topics = []
    for index, topic in enumerate(existing_topics, start=1):
        if isinstance(topic, dict):
            title = str(topic.get("title", f"Topic {index}"))
            summary = str(topic.get("summary", "")).strip()
            condensed_topics.append(f"{index}. {title}{' — ' + summary if summary else ''}")
        else:
            condensed_topics.append(f"{index}. {topic}")

    topics_summary = "\n".join(condensed_topics)[:4_000]

    context_payload = (
        "# Existing Topics\n"
        f"{topics_summary if topics_summary else 'No topics extracted yet.'}\n\n"
        "# PDF Content\n"
        f"{pdf_context_text}"
    )

    user_query = request.user_query
    temperature = 0.2  # Fixed temperature
    plan_label = _resolve_plan_label_for_admin(admin_id, current_user, request.plan_label)
    plan_limit = PLAN_SUGGESTION_LIMITS.get(plan_label) if plan_label else None

    addition_text = "Only suggest genuinely grounded subtopics."
    limit_text = (
        f"You must return no more than {MAX_ASSISTANT_SUGGESTIONS} suggested subtopics per response. {addition_text}"
    )

    material_language_code = topics_blob.get("language_code") or DEFAULT_LANGUAGE_CODE
    language_rule = LANGUAGE_OUTPUT_RULES.get(material_language_code, LANGUAGE_OUTPUT_RULES[DEFAULT_LANGUAGE_CODE])
    language_instruction = language_rule["instruction"]
    language_label = language_rule["label"]

    system_prompt = (
        "You are an AI assistant for educational content. "
        "Analyze the provided PDF content and existing topics. "
        "Suggest NEW, relevant subtopics that are missing but present in the PDF. "
        f"{language_instruction}. "
        f"Return ONLY valid JSON with this structure: "
        '{"suggestions": [{"title": "...", "summary": "...", "supporting_quote": "..."}]}'
        f"\nMaximum {MAX_ASSISTANT_SUGGESTIONS} suggestions. "
        f"{limit_text}"
    )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="GROQ_API_KEY not configured")

    client = Groq(api_key=api_key)
    model = "openai/gpt-oss-120b"

    chat_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": context_payload},
        {"role": "user", "content": user_query},
    ]

    try:
        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=chat_messages,
            temperature=temperature,
            max_completion_tokens=4000,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Assistant API failed: {exc}") from exc

    if not completion.choices:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="No response from assistant")

    raw_reply = (completion.choices[0].message.content or "").strip()
    suggestions = []
    reply_text = raw_reply

    try:
        parsed = json.loads(raw_reply)
        if isinstance(parsed, dict):
            parsed_suggestions = parsed.get("suggestions", [])
            for idx, item in enumerate(parsed_suggestions, start=1):
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip()
                summary = str(item.get("summary", "")).strip()
                quote = str(item.get("supporting_quote", "")).strip()
                if title and quote:
                    suggestions.append({
                        "suggestion_id": str(item.get("suggestion_id") or item.get("id") or idx),
                        "title": title,
                        "summary": summary,
                        "supporting_quote": quote[:240],
                    })
            if len(suggestions) > MAX_ASSISTANT_SUGGESTIONS:
                suggestions = suggestions[:MAX_ASSISTANT_SUGGESTIONS]
            if suggestions:
                reply_lines = ["Here are topic suggestions based on your PDF:"]
                for idx, suggestion in enumerate(suggestions, start=1):
                    summary_part = f" — {suggestion['summary']}" if suggestion['summary'] else ""
                    reply_lines.append(f"{idx}. {suggestion['title']}{summary_part}")
                reply_text = "\n".join(reply_lines)
            else:
                reply_text = "No additional grounded subtopics were found in the supplied PDF excerpt."
    except json.JSONDecodeError:
        logger.warning("Assistant did not return valid JSON")
        suggestions = []
        reply_text = "I couldn't generate structured suggestions. Please try rephrasing your query."

    if suggestions:
        persist_assistant_suggestions(admin_id, material.id, suggestions)

    return {
        "status": True,
        "message": "Suggestions generated",
        "data": {
            "suggestions": suggestions,
            "reply": reply_text,
            "plan_label": plan_label,
            "plan_limit": plan_limit,
            "max_suggestions": MAX_ASSISTANT_SUGGESTIONS,
            "language_code": material_language_code,
            "language_label": language_label,
            "existing_topics_count": len(existing_topics),
        },
    }


@router.post("/{material_id}/assistant-add-topics")
async def assistant_add_topics_route(
    material_id: int,
    request: AssistantAddTopicsRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    material = get_chapter_material(db, material_id)
    if not material:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")

    if current_user["role"] == "admin":
        admin_id = current_user["id"]
        if material.admin_id != admin_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    elif current_user["role"] == "member":
        if current_user.get("work_type") != WorkType.CHAPTER.value:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access restricted to chapter management members")
        member_admin_id = _resolve_member_admin_id(current_user)
        admin_id = member_admin_id or current_user.get("admin_id")
        if material.admin_id != admin_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unsupported role")

    selected_suggestions = list(request.selected_suggestions or [])
    suggestion_ids = request.suggestion_ids or []

    if suggestion_ids:
        resolved, missing = get_cached_suggestions_by_ids(material.admin_id, material.id, suggestion_ids)
        if missing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Suggestions not found for IDs: {', '.join(missing)}",
            )
        selected_suggestions.extend(resolved)

    if not selected_suggestions:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No suggestions selected to add")

    result = add_assistant_topics_to_file(material.admin_id, material, selected_suggestions)
    sanitized_added: List[Dict[str, Any]] = []
    for topic in result.get("added_topics", []) or []:
        if not isinstance(topic, dict):
            continue
        topic_copy = dict(topic)
        topic_copy["suggestion_topic_id"] = str(topic_copy.get("topic_id"))
        topic_copy.pop("topic_id", None)
        sanitized_added.append(topic_copy)
    result["added_topics"] = sanitized_added

    return {
        "status": True,
        "message": f"Added {len(result.get('added_topics', []))} topics from assistant suggestions",
        "data": result,
    }


# -------------------------
# Lecture generation endpoints (calls lecture_service)
# -------------------------
@router.post("/chapter_lecture/config")
async def post_lecture_generation_config(
    payload: LectureConfigRequest = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    _ensure_lecture_config_access(current_user)

    config_response = _build_lecture_config_response(
        requested_language=payload.language,
        requested_duration=payload.duration,
    )

    return {
        "status": True,
        "message": "Lecture configuration fetched successfully",
        "data": config_response,
    }

@router.get("/chapter_lectures/{std}/{subject}/{lecture_id}")
async def get_lecture_json(
    std: str,
    subject: str,
    lecture_id: str,
):
    """
    PUBLIC endpoint to serve lecture JSON file.
    URL: /chapter-materials/chapter_lectures/{std}/{subject}/{lecture_id}.json
    Example: /chapter-materials/chapter_lectures/9/science/4172d9c9c0e6.json
    """
    # Remove .json extension if present
    lecture_id_clean = lecture_id.replace(".json", "")
    
    # Build the file path
    storage_base = Path("./storage/chapter_lectures")
    lecture_path = storage_base / lecture_id_clean / "lecture.json"
    
    # Log for debugging
    logger.info(f"🔍 Searching for lecture: {lecture_id_clean}")
    logger.info(f"📂 Path: {lecture_path}")
    logger.info(f"✅ Exists: {lecture_path.exists()}")
    
    if not lecture_path.exists():
        logger.error(f"❌ Lecture file not found at: {lecture_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "Lecture file not found",
                "lecture_id": lecture_id_clean,
                "expected_path": str(lecture_path),
                "message": "The lecture JSON file does not exist. Please generate the lecture first."
            }
        )
    
    try:
        with open(lecture_path, "r", encoding="utf-8") as f:
            lecture_data = json.load(f)
        
        # Add metadata
        lecture_data["std"] = std.replace("_", " ").title()
        lecture_data["subject"] = subject.replace("_", " ").title()
        lecture_data["accessed_at"] = datetime.now().isoformat()
        lecture_data["file_path"] = str(lecture_path)
        
        logger.info(f"✅ Successfully serving lecture: {lecture_id_clean}")
        
        return JSONResponse(
            content=lecture_data,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            }
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in file {lecture_path}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Invalid JSON format in lecture file: {str(e)}"
        )
    except Exception as e:
        logger.error(f"❌ Error reading lecture file {lecture_path}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading lecture file: {str(e)}"
        )


# Modified generate_lecture_from_topics endpoint
@router.post("/chapter_lecture/generate")
async def generate_lecture_from_topics(
    request: LectureGenerationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate lecture content from either material topics or merged lecture payloads."""

    settings = get_settings()
    style = request.style or "storytelling"

    if request.merged_lecture_id:
        context_payload = _prepare_generation_from_merged(
            request=request,
            current_user=current_user,
            db=db,
            settings=settings,
        )
    else:
        context_payload = _prepare_generation_from_material(
            request=request,
            current_user=current_user,
            db=db,
            settings=settings,
        )

    lecture_record = await lecture_service.create_lecture_from_text(
        text=context_payload["aggregate_text"],
        language=context_payload["language"],
        duration=context_payload["duration"],
        style=style,
        title=context_payload["title"],
        metadata=context_payload["metadata"],
    )

    # ============================================================================
    # URL GENERATION AND TERMINAL PRINTING
    # ============================================================================
    
    # Get lecture ID from the generated lecture
    lecture_id = lecture_record.get("lecture_id", "")

    # Generate JSON URL with class and subject
    std_slug = context_payload["std_slug"]
    subject_slug = context_payload["subject_slug"]
    
    # Use chapter-materials prefix for the URL
    lecture_json_url = f"/chapter-materials/chapter_lectures/{std_slug}/{subject_slug}/{lecture_id}.json"

    # Add URL to lecture_record
    lecture_record["lecture_json_url"] = lecture_json_url

    # Print detailed information to terminal
    print(f"\n{'='*60}")
    print(f"📚 LECTURE GENERATED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Lecture ID: {lecture_id}")
    log_context = context_payload["log_context"]
    print(f"Material ID: {log_context['material_id'] or 'N/A'}")
    print(f"Class (STD): {log_context['std']}")
    print(f"Subject: {log_context['subject']}")
    print(f"Board: {log_context['board']}")
    print(f"Semester: {log_context['sem']}")
    print(f"Title: {context_payload['title']}")
    print(f"Language: {context_payload['language']}")
    print(f"Duration: {context_payload['duration']} minutes")
    print(f"Style: {style}")
    print(f"Selected Topics: {log_context['selected_topics_count']}")
    print(f"Total Slides: {lecture_record.get('total_slides', 'N/A')}")
    print(f"Fallback Used: {lecture_record.get('fallback_used', False)}")
    print(f"")
    print(f"📄 JSON URL: {lecture_json_url}")
    print(f"🌐 Full URL: http://localhost:3020{lecture_json_url}")
    if lecture_record.get("lecture_path"):
        print(f"📂 File Path: {lecture_record.get('lecture_path')}")
    print(f"{'='*60}\n")

    # ============================================================================
    # SAVE TO DATABASE WITH URL
    # ============================================================================
    
    from app.models.chapter_material import LectureGen

    material_snapshot = context_payload.get("material_snapshot") or {}
    log_material_id = material_snapshot.get("id") or 0
    log_admin_id = current_user.get("id") if current_user.get("role") == "admin" else current_user.get("admin_id")
    lecture_uid = str(lecture_id) if lecture_id else None

    if lecture_uid is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lecture identifier missing; please retry generation",
        )

    try:
        db_lecture = LectureGen(
            admin_id=log_admin_id,
            material_id=log_material_id,
            lecture_uid=lecture_uid,
            lecture_title=context_payload["title"],
            lecture_link=lecture_json_url,
            subject=log_context["subject"],
            std=log_context["std"],
            sem=log_context["sem"],
            board=log_context["board"],
        )
        
        db.add(db_lecture)
        db.commit()
        db.refresh(db_lecture)
        
        # Print database confirmation
        print(f"\n{'='*60}")
        print(f"💾 DATABASE RECORD SAVED")
        print(f"{'='*60}")
        print(f"DB Record ID: {db_lecture.id}")
        print(f"Lecture UID: {lecture_uid}")
        print(f"Material ID: {material_snapshot.get('id')}")
        print(f"Admin ID: {material_snapshot.get('admin_id') or log_admin_id}")
        print(f"Class (STD): {material_snapshot.get('std')}")
        print(f"Subject: {material_snapshot.get('subject')}")
        print(f"Board: {material_snapshot.get('board')}")
        print(f"Semester: {material_snapshot.get('sem')}")
        print(f"Chapter: {material_snapshot.get('chapter_number')}")
        print(f"Stored JSON URL: {lecture_json_url}")
        print(f"Database Table: lecture_gen")
        print(f"{'='*60}\n")
        
        lecture_record["db_record_id"] = db_lecture.id
        lecture_record["db_saved"] = True
        
    except Exception as e:
        logger.error(f"Failed to save lecture to database: {e}")
        lecture_record["db_saved"] = False
        lecture_record["db_error"] = str(e)

    return {
        "status": True,
        "message": "Lecture generated successfully",
        "data": {
            "lecture": {
                **lecture_record,
                "lecture_json_url": lecture_json_url,
                "db_record_id": lecture_record.get("db_record_id"),
                "db_saved": lecture_record.get("db_saved", False),
                "material_info": material_snapshot,
                "selected_topic_ids": request.selected_topic_ids,
            }
        }
    }


@router.post("/chapter_lecture/generate/{lecture_id}")
async def generate_lecture_from_path(
    lecture_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
    lecture_service: LectureService = Depends(get_lecture_service),
):
    request_payload = LectureGenerationRequest.model_validate({"lecture_id": lecture_id})
    return await generate_lecture_from_topics(
        request=request_payload,
        current_user=current_user,
        db=db,
    )

@router.post("/chapter_lecture/{lecture_id}/chat")
async def chat_about_lecture(
    lecture_id: str,
    request: LectureChatRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    lecture_service: LectureService = Depends(get_lecture_service),
):
    try:
        answer = await lecture_service.answer_question(
            lecture_id=lecture_id,
            question=request.question,
            answer_type=request.answer_type,
            is_edit_command=request.is_edit_command,
            context_override=request.context_override,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lecture not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to process chat request") from exc

    if isinstance(answer, dict):
        payload = answer
    else:
        payload = {"answer": answer}

    return {"status": True, "message": "Response generated", "data": payload}


# -------------------------
# -------------------------

@router.post("/create_merged_chapter_lecture")
async def create_merged_lecture(
    request: CreateMergedLectureRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        # Validate input
        if not request:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Lecture data is required")

        title = (request.title or "").strip()
        if not title:
            title = f"Merged Lecture {datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        materials = request.materials or []
        topic_selections = request.topic_selections or []
        selected_topics_payload = request.selected_topics or {}

        if not materials and not topic_selections and not selected_topics_payload:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Either materials or topics must be provided")

        # Validate materials if provided
        if materials:
            for material in materials:
                if not isinstance(material, dict) or "id" not in material:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid material format")

        # Resolve topics coming in via explicit payload (legacy support)
        combined_topics: Dict[str, List[Dict[str, Any]]] = {}
        selection_summary: Dict[str, Dict[str, Any]] = {}
        for material_key, topics in selected_topics_payload.items():
            if not isinstance(topics, list):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Topics for material {material_key} must be an array")
            filtered = []
            for topic in topics:
                if not isinstance(topic, dict) or "title" not in topic:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid topic format")
                filtered.append(topic)
            if filtered:
                combined_topics[str(material_key)] = filtered

        # Resolve topics via topic selections (material + indices)
        if topic_selections:
            for selection in topic_selections:
                if not isinstance(selection, TopicSelection):
                    continue

                material = get_chapter_material(db, selection.material_id)
                if not material:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Material {selection.material_id} not found")

                # Authorization checks
                if current_user["role"] == "admin":
                    if material.admin_id != current_user["id"]:
                        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
                elif current_user["role"] == "member":
                    member_admin_id = _resolve_member_admin_id(current_user)
                    if material.admin_id != member_admin_id:
                        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
                else:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unsupported role")

                payload, topics_list = read_topics_file_if_exists(material.admin_id, material.id)
                if not topics_list:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No topics found for material {material.id}")

                key = str(material.id)
                resolved_topics: List[Dict[str, Any]] = []
                selection_summary.setdefault(key, {"selected_indices": [], "selected_topic_ids": []})

                if selection.topic_indices:
                    for index in selection.topic_indices:
                        if index < 0 or index >= len(topics_list):
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Topic index {index} out of range for material {material.id}"
                            )
                        resolved_topics.append(topics_list[index])
                    selection_summary[key]["selected_indices"].extend(selection.topic_indices)

                selection_topic_ids: List[str] = []
                if selection.topic_ids:
                    selection_topic_ids.extend(selection.topic_ids)
                if selection.suggestion_topic_ids:
                    selection_topic_ids.extend(selection.suggestion_topic_ids)

                if selection_topic_ids:
                    resolved_by_ids, missing_ids = get_topics_by_ids(material.admin_id, material.id, selection_topic_ids)
                    if missing_ids:
                        raise HTTPException(
                            status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Topics not found for IDs: {', '.join(missing_ids)}",
                        )
                    resolved_topics.extend(resolved_by_ids)
                    selection_summary[key]["selected_topic_ids"].extend(selection_topic_ids)

                combined_topics.setdefault(key, []).extend(resolved_topics)

        total_topics = sum(len(topics) for topics in combined_topics.values())
        if total_topics == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one topic must be selected")

        # Generate unique lecture ID
        import uuid
        lecture_id = f"merged_lecture_{uuid.uuid4().hex[:8]}"
        
        # Log the creation
        logger.info(
            f"Creating merged lecture '{title}' with {len(materials) or len(combined_topics)} materials "
            f"Creating merged lecture '{title}' with {len(materials) or len(combined_topics.keys())} materials "
            f"and {total_topics} topics by user {current_user.get('email')}"
        )
        
        # TODO: Implement actual lecture creation logic
        # For now, return success response
        
        # Count materials and topics properly
        materials_count = len(materials) if materials else len(combined_topics)
        topics_count = total_topics

        merged_topics_response: List[Dict[str, Any]] = []
        for material_key, topics in combined_topics.items():
            try:
                material_id_value: Any = int(material_key)
            except ValueError:
                material_id_value = material_key
            sanitized_topics: List[Dict[str, Any]] = []
            for topic in topics:
                if not isinstance(topic, dict):
                    continue
                topic_copy = dict(topic)
                if topic_copy.get("is_assistant_generated"):
                    suggestion_value = topic_copy.get("suggestion_topic_id") or topic_copy.get("topic_id")
                    if suggestion_value is not None:
                        topic_copy["suggestion_topic_id"] = str(suggestion_value)
                    topic_copy.pop("topic_id", None)
                else:
                    topic_id_value = topic_copy.get("topic_id")
                    if topic_id_value is not None:
                        topic_copy["topic_id"] = str(topic_id_value)
                    topic_copy.pop("suggestion_topic_id", None)
                sanitized_topics.append(topic_copy)
            merged_topics_response.append(
                {
                    "material_id": material_id_value,
                    "topics_count": len(topics),
                    "topics": sanitized_topics,
                    "selection": selection_summary.get(material_key),
                }
            )

        response_payload = {
            "status": True,
            "message": "Merged lecture created successfully",
            "data": {
                "lecture_id": lecture_id,
                "title": title,
                "materials_count": materials_count,
                "topics_count": topics_count,
                "created_at": datetime.utcnow().isoformat(),
                "selected_materials": list(combined_topics.keys()) if combined_topics else [m.get("id") for m in materials],
                "merged_topics": merged_topics_response,
            },
        }

        creator_admin_id = None
        try:
            creator_admin_id = _resolve_admin_id(current_user)
        except HTTPException:
            creator_admin_id = _resolve_member_admin_id(current_user)

        stored_payload = {
            "admin_id": creator_admin_id,
            "created_by": current_user.get("id"),
            "created_by_role": current_user.get("role"),
            "response": response_payload,
        }
        try:
            _save_merged_lecture_payload(lecture_id, stored_payload)
        except Exception as exc:
            logger.warning("Failed to cache merged lecture %s: %s", lecture_id, exc)

        return response_payload
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating merged lecture: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create merged lecture")


@router.get("/merged-topics/{lecture_id}")
async def get_merged_lecture(
    lecture_id: str,
    current_user: dict = Depends(get_current_user),
):
    payload = _load_merged_lecture_payload(lecture_id)
    if not payload or "response" not in payload:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Merged lecture not found")

    try:
        viewer_admin_id = _resolve_admin_id(current_user)
    except HTTPException:
        # For lecture viewers (non chapter roles) fall back to member admin resolution
        viewer_admin_id = _resolve_member_admin_id(current_user)

    stored_admin_id = payload.get("admin_id")
    if stored_admin_id and viewer_admin_id != stored_admin_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied for this lecture")

    response = payload["response"]
    if not isinstance(response, dict):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Stored lecture payload is invalid")

    return response


# -------------------------
# Multiple Chapter Selection endpoints (POST only)
# -------------------------
@router.post("/select-multiple-chapters")
async def select_multiple_chapters(
    request: MultipleChapterSelectionRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        admin_id = _resolve_admin_id(current_user)
        
        # Extract selected IDs from Pydantic model
        selected_ids = request.selected_ids
        
        logger.info(f"Select multiple chapters - selected_ids: {selected_ids}")
        
        # Validate parameters
        if not selected_ids:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No chapters selected")
        
        # Get all materials for admin
        materials = list_chapter_materials(db, admin_id, None, None)
        
        # Filter by selected IDs
        selected_materials = [m for m in materials if str(m.id) in [str(id) for id in selected_ids]]
        
        # Debug logging
        logger.info(f"Requested IDs: {selected_ids}")
        logger.info(f"Found IDs: {[m.id for m in selected_materials]}")
        logger.info(f"Total materials in DB: {len(materials)}")
        
        return {
            "status": True, 
            "message": f"Successfully selected {len(selected_materials)} chapters", 
            "data": {
                "total_chapters": len(selected_materials)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in select_multiple_chapters: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve selected chapters")


# @router.get("/download-generated-pdf/{filename}")
# async def download_generated_pdf(
#     filename: str,
#     current_user: dict = Depends(get_current_user)
# ):
#     try:
#         pdf_path = os.path.join("generated_pdfs", filename)
#         if not os.path.exists(pdf_path):
#             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generated PDF not found")
#         if not filename.lower().endswith('.pdf'):
#             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type")
#         logger.info(f"User {current_user.get('email')} downloading PDF: {filename}")
#         return FileResponse(path=pdf_path, filename=filename, media_type='application/pdf')
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error downloading PDF {filename}: {str(e)}")
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to download PDF")

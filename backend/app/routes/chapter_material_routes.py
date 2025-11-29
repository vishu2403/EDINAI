# app/routes/chapter_material_routes.py

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.config import get_settings
from app.database import get_db
from app.models import ChapterMaterial, LectureGen
from app.schemas.admin_schema import WorkType
from app.schemas.chapter_material_schema import (
    TopicExtractRequest,
    LectureGenerationRequest,
    LectureChatRequest,
    ManualTopicCreate,
    AssistantSuggestRequest,
    AssistantAddTopicsRequest,
    ResponseBase,
)
from app.repository import (
    admin_management_repository,
    registration_repository,
    lecture_credit_repository,
)
from app.plan_limits import (
    PLAN_SUGGESTION_LIMITS,
    PLAN_DURATION_LIMITS,
    PLAN_CREDIT_LIMITS,
    normalize_plan_label,
)
from app.repository.chapter_material_repository import (
    create_chapter_material,
    get_chapter_material,
    list_chapter_materials,
    list_recent_chapter_materials,
    update_chapter_material_db,
    delete_chapter_material_db,
    get_dashboard_stats,
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
    persist_language_metadata,
    read_language_metadata,
)
from app.utils.file_handler import (
    save_uploaded_file,
    ALLOWED_PDF_EXTENSIONS,
    ALLOWED_PDF_TYPES,
    UPLOAD_DIR,
)
from app.services.lecture_service import LectureService
from app.utils.dependencies import admin_required, get_current_user
from app.utils.topic_extractor import detect_pdf_language
from groq import Groq

from fastapi.responses import FileResponse, JSONResponse
import os
from pathlib import Path
from datetime import datetime
router = APIRouter(prefix="/chapter-materials", tags=["Chapter Materials"])
lecture_service = LectureService()

logger = logging.getLogger(__name__)

# Local constants (kept same)
PDF_MAX_SIZE = 15 * 1024 * 1024  # 15MB
DEFAULT_MIN_DURATION = 5
DEFAULT_MAX_DURATION = 180
MAX_ASSISTANT_SUGGESTIONS = 10
DEFAULT_LANGUAGE_CODE = "eng"

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


# -------------------------
# Helper to resolve plan label
# -------------------------
def _resolve_plan_label(current_user: dict, admin_id: int) -> Optional[str]:
    def _normalized(value: Optional[str]) -> Optional[str]:
        return normalize_plan_label(value)

    direct_package = _normalized(current_user.get("package"))
    if direct_package:
        return direct_package

    user_obj = current_user.get("user_obj")
    if isinstance(user_obj, dict):
        from_obj = _normalized(user_obj.get("package_plan") or user_obj.get("package"))
        if from_obj:
            return from_obj

    registered_admin = registration_repository.get_admin_by_id(admin_id)
    if registered_admin:
        from_registration = _normalized(registered_admin.get("package_plan"))
        if from_registration:
            return from_registration

    legacy_admin = admin_management_repository.get_admin_by_id(admin_id)
    if legacy_admin:
        from_legacy = _normalized(legacy_admin.get("package"))
        if from_legacy:
            return from_legacy

    return None


# -------------------------
# Generated lectures listing
# -------------------------
@router.get("/lectures", response_model=ResponseBase)
async def list_generated_lectures(
    std: Optional[str] = Query(default=None, description="Filter by class/standard"),
    subject: Optional[str] = Query(default=None, description="Filter by subject"),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
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

    storage_base = Path("./storage/lectures")
    items: List[Dict[str, Any]] = []

    for (
        record,
        chapter_name,
        material_subject,
        material_std,
        material_sem,
        material_board,
    ) in records:
        std_value = record.std or material_std or "General"
        subject_value = record.subject or material_subject or "Lecture"
        sem_value = record.sem or material_sem
        board_value = record.board or material_board

        std_slug = std_value.replace(" ", "_").lower()
        subject_slug = subject_value.replace(" ", "_").lower()

        relative_url = record.lecture_link or f"/chapter-materials/lectures/{std_slug}/{subject_slug}/{record.lecture_uid}.json"
        storage_path = storage_base / record.lecture_uid / "lecture.json"

        items.append(
            {
                "id": record.id,
                "lecture_uid": record.lecture_uid,
                "lecture_title": record.lecture_title,
                "lecture_link": relative_url,
                "lecture_json_url": relative_url,
                "std": std_value,
                "subject": subject_value,
                "sem": sem_value,
                "board": board_value,
                "chapter_name": chapter_name,
                "material_id": record.material_id,
                "storage_path": str(storage_path),
                "created_at": record.created_at.isoformat() if record.created_at else None,
                "updated_at": record.updated_at.isoformat() if record.updated_at else None,
            }
        )

    return ResponseBase(
        status=True,
        message="Lectures fetched successfully",
        data={"items": items, "total": len(items)},
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
        return {
            "status": True,
            "message": "Dashboard data retrieved successfully",
            "data": {"chapter_metrics": stats},
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


@router.get("/")
async def list_chapter_materials_route(
    std: Optional[str] = None,
    subject: Optional[str] = None,
    sem: Optional[str] = None,
    board: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    admin_id = _resolve_admin_id(current_user)
    materials = list_chapter_materials(db, admin_id, std, subject, sem, board)
    serialized = [m.to_dict() if hasattr(m, "to_dict") else m.__dict__ for m in materials]
    return {"status": True, "message": "Chapter materials retrieved successfully", "data": {"materials": serialized}}


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
    if not material_ids or len(material_ids) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="material_ids is required and cannot be empty")

    try:
        from app.utils.topic_extractor import extract_topics_from_pdf

        logger.info(f"Extracting topics for {len(material_ids)} materials")

        MAX_BATCH = 3
        if len(material_ids) > MAX_BATCH:
            logger.warning(f"Limiting materials to {MAX_BATCH}")
            material_ids = material_ids[:MAX_BATCH]

        topics_by_material = {}
        flattened_topics = []

        for material_id in material_ids:
            material = get_chapter_material(db, material_id)
            if not material:
                topics_by_material[material_id] = {"success": False, "error": "Material not found", "topics": []}
                continue

            # Permission check
            if current_user["role"] == "admin":
                if material.admin_id != current_user["id"]:
                    topics_by_material[material_id] = {"success": False, "error": "Not allowed for this material", "topics": []}
                    continue

            if current_user["role"] == "member":
                member_admin_id = _resolve_member_admin_id(current_user)
                if material.admin_id != member_admin_id:
                    topics_by_material[material_id] = {"success": False, "error": "Not allowed for this material", "topics": []}
                    continue

            if not os.path.exists(material.file_path):
                topics_by_material[material_id] = {"success": False, "error": "File not found", "topics": []}
                continue

            try:
                extraction = extract_topics_from_pdf(Path(material.file_path))
                chapter_title = extraction.get("chapter_title") or material.chapter_number
                extraction.setdefault("chapter_title", chapter_title)
                extraction["material_id"] = material_id

                # Save to files using repository helper
                txt_path, json_path = save_extracted_topics_files(material.admin_id, material_id, extraction)
                extraction["topics_file"] = json_path
                extraction["topics_text_file"] = txt_path

                topics_by_material[material_id] = extraction

                if extraction.get("success"):
                    flattened_topics.extend(extraction.get("topics", []))

            except Exception as e:
                logger.exception(f"Error extracting topics for {material_id}")
                topics_by_material[material_id] = {"success": False, "error": str(e), "topics": []}

        return {"status": True, "message": "Topics extracted successfully", "data": {"topics": topics_by_material, "flattened_topics": flattened_topics}}
    except Exception as exc:
        logger.exception("Unexpected error in extract_topics_from_materials")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.get("/{material_id}/topics")
async def get_material_topics(
    material_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    material = get_chapter_material(db, material_id)
    if not material:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")

    # Authorization checks (preserve original logic)
    if current_user["role"] == "admin":
        if material.admin_id != current_user["id"]:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    elif current_user["role"] == "member":
        if current_user.get("work_type") != WorkType.CHAPTER.value:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access restricted to chapter management members")
        member_admin_id = _resolve_member_admin_id(current_user)
        if material.admin_id != (member_admin_id or current_user.get("admin_id")):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unsupported role")

    payload, topics_list = read_topics_file_if_exists(material.admin_id, material.id)
    return {"status": True, "message": "Material topics fetched successfully" if topics_list else "No topics found", "data": {"material": material.to_dict() if hasattr(material, "to_dict") else material.__dict__, "topics": topics_list, "topics_metadata": payload}}


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
        if material.admin_id != (member_admin_id or current_user.get("admin_id")):
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
    temperature = request.temperature if request.temperature is not None else 0.2
    plan_label = _resolve_plan_label(current_user, admin_id)
    plan_limit = PLAN_SUGGESTION_LIMITS.get(plan_label)
    addition_text = (
        f"The user may only add {plan_limit} assistant-generated subtopics under their current plan, so prioritize the highest-impact ideas."
        if plan_limit is not None
        else "Only suggest genuinely grounded subtopics."
    )
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
            for item in parsed_suggestions:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip()
                summary = str(item.get("summary", "")).strip()
                quote = str(item.get("supporting_quote", "")).strip()
                if title and quote:
                    suggestions.append({"title": title, "summary": summary, "supporting_quote": quote[:240]})
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

    return {
        "status": True,
        "message": "Suggestions generated",
        "data": {
            "suggestions": suggestions,
            "reply": reply_text,
            "plan_limit": plan_limit,
            "plan_label": plan_label,
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
        if material.admin_id != current_user["id"]:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    elif current_user["role"] == "member":
        if current_user.get("work_type") != WorkType.CHAPTER.value:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access restricted to chapter management members")
        member_admin_id = _resolve_member_admin_id(current_user)
        if material.admin_id != (member_admin_id or current_user.get("admin_id")):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unsupported role")

    selected_suggestions = request.selected_suggestions
    if not selected_suggestions:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No suggestions selected to add")

    plan_label = _resolve_plan_label(current_user, material.admin_id)
    plan_limit = PLAN_SUGGESTION_LIMITS.get(plan_label)

    existing_payload = read_topics_file_if_exists(material.admin_id, material.id)[0] or {}
    existing_topics = existing_payload.get("topics", [])
    assistant_generated = [topic for topic in existing_topics if isinstance(topic, dict) and topic.get("is_assistant_generated")]

    if plan_limit is not None and len(assistant_generated) + len(selected_suggestions) > plan_limit:
        remaining = max(plan_limit - len(assistant_generated), 0)
        detail = (
            f"Plan {plan_label.upper()} allows {plan_limit} assistant topics. "
            f"You already have {len(assistant_generated)}. "
            f"You can add {remaining} more." if remaining > 0 else "You have reached the assistant topic limit for this plan."
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

    result = add_assistant_topics_to_file(material.admin_id, material, selected_suggestions)
    result.setdefault("plan_limit", plan_limit)
    result.setdefault("plan_label", plan_label)
    return {"status": True, "message": f"Added {len(result.get('added_topics', []))} topics from assistant suggestions", "data": result}


# -------------------------
# Lecture generation endpoints (calls lecture_service)
# -------------------------
@router.get("/lecture/config")
async def get_lecture_generation_config(
    language: Optional[str] = Query(default=None),
    duration: Optional[int] = Query(default=None, ge=DEFAULT_MIN_DURATION, le=DEFAULT_MAX_DURATION),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    if current_user["role"] == "admin":
        admin_id = current_user["id"]
    elif current_user["role"] == "member":
        if current_user.get("work_type") != WorkType.CHAPTER.value:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only chapter members can access lecture configuration")
        admin_id = _resolve_member_admin_id(current_user)
        if admin_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Member admin not found")
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unsupported role")

    plan_label = _resolve_plan_label(current_user, admin_id)
    plan_max_duration = PLAN_DURATION_LIMITS.get(plan_label, DURATION_OPTIONS[-1])

    credit_usage = lecture_credit_repository.get_usage(admin_id)
    credits_total = PLAN_CREDIT_LIMITS.get(plan_label)
    credits_used = credit_usage["credits_used"]
    credits_remaining = (
        max(credits_total - credits_used, 0) if credits_total is not None else None
    )

    allowed_durations = [value for value in DURATION_OPTIONS if value <= plan_max_duration]
    if not allowed_durations:
        allowed_durations = [DURATION_OPTIONS[0]]

    settings = get_settings()
    default_language = settings.dict().get("default_language") or getattr(settings, "default_language", None)
    language_value = default_language or SUPPORTED_LANGUAGES[0]["value"]

    if language:
        normalized_language = next((option["value"] for option in SUPPORTED_LANGUAGES if option["value"].lower() == language.strip().lower()), None)
        if normalized_language:
            language_value = normalized_language

    configured_default_duration = getattr(settings, "default_lecture_duration", None) or settings.dict().get("default_lecture_duration")
    if configured_default_duration is None:
        configured_default_duration = DURATION_OPTIONS[0]
    default_duration = min(configured_default_duration, plan_max_duration)

    selected_duration = default_duration if default_duration in allowed_durations else allowed_durations[-1]
    if duration is not None:
        if duration not in allowed_durations:
            plan_name = (plan_label or "current").upper()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Plan {plan_name} allows a maximum lecture duration of {plan_max_duration} minutes",
            )
        selected_duration = duration

    return {
        "status": True,
        "message": "Lecture configuration fetched",
        "data": {
            "default_duration": default_duration,
            "selected_duration": selected_duration,
            "min_duration": DEFAULT_MIN_DURATION,
            "max_duration": plan_max_duration,
            "default_language": language_value,
            "selected_language": language_value,
            "languages": SUPPORTED_LANGUAGES,
            "durations": allowed_durations,
            "requested_language": language,
            "requested_duration": duration,
            "plan_label": plan_label,
            "plan_max_duration": plan_max_duration,
            "lecture_credits": {
                "plan_label": plan_label,
                "total": credits_total,
                "used": credits_used,
                "remaining": credits_remaining,
                "overflow_attempts": credit_usage["overflow_attempts"],
            },
        },
    }




@router.get("/lectures/{std}/{subject}/{lecture_id}")
async def get_lecture_json(
    std: str,
    subject: str,
    lecture_id: str,
):
    """
    PUBLIC endpoint to serve lecture JSON file.
    URL: /chapter-materials/lectures/{std}/{subject}/{lecture_id}.json
    Example: /chapter-materials/lectures/9/science/4172d9c9c0e6.json
    """
    # Remove .json extension if present
    lecture_id_clean = lecture_id.replace(".json", "")
    
    # Build the file path
    storage_base = Path("./storage/lectures")
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
@router.post("/lecture/generate")
async def generate_lecture_from_topics(
    request: LectureGenerationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Generate lecture from selected topics with automatic URL generation.
    """
    material = get_chapter_material(db, request.material_id)
    if not material:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Material not found")

    # Authorization checks
    if current_user["role"] == "admin":
        if material.admin_id != current_user["id"]:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    elif current_user["role"] == "member":
        if current_user.get("work_type") != WorkType.CHAPTER.value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only chapter members can generate lectures"
            )
        member_admin_id = _resolve_member_admin_id(current_user)
        if material.admin_id != member_admin_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unsupported role")

    # Load topics
    topics_payload = {}
    topics_path = None

    if request.topics_override:
        topics = request.topics_override
        if not isinstance(topics, list) or not topics:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Topics override payload is empty"
            )
    else:
        topics_path = _load_topics_path(material)
        if not os.path.exists(topics_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Topics not found. Please extract topics before generating a lecture."
            )
        with open(topics_path, "r", encoding="utf-8") as fh:
            topics_payload = json.load(fh)
        topics = topics_payload.get("topics", [])
        if not topics:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No topics available for this material"
            )

    # Select valid topics
    valid_indices = []
    selected_topics = []
    for index in request.selected_topic_indices:
        if 0 <= index < len(topics):
            valid_indices.append(index)
            selected_topics.append(topics[index])

    if not selected_topics:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Selected topics are invalid"
        )

    # Aggregate text from selected topics
    aggregate_text_parts = [topic_to_text(topic) for topic in selected_topics]
    aggregate_text = "\n\n".join(part for part in aggregate_text_parts if part)

    if not aggregate_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to extract content from selected topics"
        )

    # Prepare lecture generation parameters
    topics_language_label = topics_payload.get("language_label") if topics_payload else None
    language = request.language or topics_language_label or "English"
    plan_label = _resolve_plan_label(current_user, material.admin_id)
    plan_max_duration = PLAN_DURATION_LIMITS.get(plan_label, DURATION_OPTIONS[-1])

    credit_usage = lecture_credit_repository.get_usage(material.admin_id)
    plan_credit_limit = PLAN_CREDIT_LIMITS.get(plan_label)
    if plan_credit_limit is not None and credit_usage["credits_used"] >= plan_credit_limit:
        lecture_credit_repository.upsert_usage(material.admin_id, overflow_delta=1)
        plan_name = (plan_label or "current").upper()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Lecture credit limit reached. Plan {plan_name} allows {plan_credit_limit} lectures "
                f"and {credit_usage['credits_used']} have already been generated."
            ),
        )

    configured_default_duration = get_settings().default_lecture_duration
    fallback_default_duration = DURATION_OPTIONS[-1]
    default_duration = configured_default_duration or fallback_default_duration
    duration = request.duration or default_duration

    if duration > plan_max_duration:
        plan_name = (plan_label or "current").upper()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Plan {plan_name} allows a maximum lecture duration of {plan_max_duration} minutes",
        )

    style = request.style or "storytelling"
    title = request.title or f"{material.subject} Lecture"
    source_material_ids = request.source_material_ids or [material.id]

    # Generate lecture using lecture service
    lecture_record = await lecture_service.create_lecture_from_text(
        text=aggregate_text,
        language=language,
        duration=duration,
        style=style,
        title=title,
        metadata={
            "material_id": material.id,
            "material_subject": material.subject,
            "selected_topic_indices": valid_indices,
            "topics_source_file": topics_path,
            "language_label": topics_payload.get("language_label") if topics_payload else None,
            "language_code": topics_payload.get("language_code") if topics_payload else None,
            "topics_override": bool(request.topics_override),
            "source_material_ids": source_material_ids,
            "std": material.std,
            "subject": material.subject,
        },
    )

    # ============================================================================
    # URL GENERATION AND TERMINAL PRINTING
    # ============================================================================
    
    # Get lecture ID from the generated lecture
    lecture_id = lecture_record.get("lecture_id", "")

    # Generate JSON URL with class and subject
    std_slug = material.std.replace(" ", "_").lower() if material.std else "general"
    subject_slug = material.subject.replace(" ", "_").lower() if material.subject else "subject"
    
    # Use chapter-materials prefix for the URL
    lecture_json_url = f"/chapter-materials/lectures/{std_slug}/{subject_slug}/{lecture_id}.json"

    # Add URL to lecture_record
    lecture_record["lecture_json_url"] = lecture_json_url

    # Print detailed information to terminal
    print(f"\n{'='*60}")
    print(f"📚 LECTURE GENERATED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Lecture ID: {lecture_id}")
    print(f"Material ID: {material.id}")
    print(f"Class (STD): {material.std or 'N/A'}")
    print(f"Subject: {material.subject or 'N/A'}")
    print(f"Board: {material.board or 'N/A'}")
    print(f"Semester: {material.sem or 'N/A'}")
    print(f"Chapter: {material.chapter_number or 'N/A'}")
    print(f"Title: {title}")
    print(f"Language: {language}")
    print(f"Duration: {duration} minutes")
    print(f"Style: {style}")
    print(f"Selected Topics: {len(selected_topics)}")
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
    from uuid import uuid4
    
    lecture_uid = lecture_id or uuid4().hex[:12]
    
    latest_usage = credit_usage

    try:
        db_lecture = LectureGen(
            admin_id=material.admin_id,
            material_id=material.id,
            lecture_uid=lecture_uid,
            lecture_title=title,
            lecture_link=lecture_json_url,  # Store JSON URL
            subject=material.subject,
            std=material.std,
            sem=material.sem,
            board=material.board,
        )
        
        db.add(db_lecture)
        db.commit()
        db.refresh(db_lecture)

        lecture_credit_repository.upsert_usage(material.admin_id, credits_delta=1)
        latest_usage = lecture_credit_repository.get_usage(material.admin_id)

        # Print database confirmation
        print(f"\n{'='*60}")
        print(f"💾 DATABASE RECORD SAVED")
        print(f"{'='*60}")
        print(f"DB Record ID: {db_lecture.id}")
        print(f"Lecture UID: {lecture_uid}")
        print(f"Material ID: {material.id}")
        print(f"Admin ID: {material.admin_id}")
        print(f"Class (STD): {material.std}")
        print(f"Subject: {material.subject}")
        print(f"Board: {material.board}")
        print(f"Semester: {material.sem}")
        print(f"Chapter: {material.chapter_number}")
        print(f"Stored JSON URL: {lecture_json_url}")
        print(f"Database Table: lecture_gen")
        print(f"{'='*60}\n")
        
        lecture_record["db_record_id"] = db_lecture.id
        lecture_record["db_saved"] = True
        
    except Exception as e:
        logger.error(f"Failed to save lecture to database: {e}")
        lecture_record["db_saved"] = False
        lecture_record["db_error"] = str(e)

    credits_remaining_after = (
        max(plan_credit_limit - latest_usage["credits_used"], 0) if plan_credit_limit is not None else None
    )

    credits_payload = {
        "plan_label": plan_label,
        "total": plan_credit_limit,
        "used": latest_usage["credits_used"],
        "remaining": credits_remaining_after,
        "overflow_attempts": latest_usage["overflow_attempts"],
    }

    return {
        "status": True,
        "message": "Lecture generated successfully",
        "data": {
            "lecture": {
                **lecture_record,
                "lecture_json_url": lecture_json_url,
                "db_record_id": lecture_record.get("db_record_id"),
                "db_saved": lecture_record.get("db_saved", False),
                "material_info": {
                    "id": material.id,
                    "admin_id": material.admin_id,
                    "std": material.std,
                    "subject": material.subject,
                    "board": material.board,
                    "sem": material.sem,
                    "chapter_number": material.chapter_number,
                    "file_name": material.file_name,
                },
                "selected_topic_indices": valid_indices,
                "source_material_ids": source_material_ids,
            },
            "lecture_credits": credits_payload,
        }
    }

@router.post("/lecture/{lecture_id}/chat")
async def chat_about_lecture(
    lecture_id: str,
    request: LectureChatRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
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
# Utilities: merged lecture & download generated pdf
# -------------------------
@router.post("/create-merged-lecture")
async def create_merged_lecture(
    lecture_data: dict,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        return {"status": True, "message": "Merged lecture created successfully", "data": {"lecture_id": "temp_lecture_001", "title": lecture_data.get("title", "Merged Lecture"), "materials_count": len(lecture_data.get("materials", [])), "topics_count": sum(len(topics) for topics in lecture_data.get("selectedTopics", {}).values())}}
    except Exception as e:
        logger.error(f"Error creating merged lecture: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create merged lecture")


@router.get("/download-generated-pdf/{filename}")
async def download_generated_pdf(
    filename: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        pdf_path = os.path.join("generated_pdfs", filename)
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generated PDF not found")
        if not filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type")
        logger.info(f"User {current_user.get('email')} downloading PDF: {filename}")
        return FileResponse(path=pdf_path, filename=filename, media_type='application/pdf')
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading PDF {filename}: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to download PDF")

# app/repositories/chapter_material_repository.py

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models import ChapterMaterial
from app.schemas.admin_schema import WorkType
from app.utils.file_handler import (
    save_uploaded_file,
    delete_file,
    ALLOWED_PDF_EXTENSIONS,
    ALLOWED_PDF_TYPES,
    UPLOAD_DIR,
)
from app.config import get_settings
from app.postgres import get_pg_cursor

logger = logging.getLogger(__name__)

# Constants (keep same as original)
PDF_MAX_SIZE = 15 * 1024 * 1024  # 15MB
DEFAULT_MIN_DURATION = 5
DEFAULT_MAX_DURATION = 180
MAX_ASSISTANT_SUGGESTIONS = 10
DEFAULT_LANGUAGE_CODE = "eng"

LANGUAGE_OUTPUT_RULES: Dict[str, Dict[str, str]] = {
    "eng": {
        "label": "English",
        "instruction": "The PDF language is English. Unless the user explicitly asks for another language, write every suggestion title, summary, supporting_quote, and topic_response explanation in English.",
    },
    "hin": {
        "label": "Hindi (हिंदी)",
        "instruction": "The PDF language is Hindi (हिंदी). Unless the user explicitly asks for another language, write every suggestion title, summary, supporting_quote, and topic_response explanation in Hindi (हिंदी).",
    },
    "guj": {
        "label": "Gujarati (ગુજરાતી)",
        "instruction": "The PDF language is Gujarati (ગુજરાતी). Unless the user explicitly asks for another language, write every suggestion title, summary, supporting_quote, and topic_response explanation in Gujarati (ગુજરાતી).",
    },
}

SUPPORTED_LANGUAGES = [
    {"value": "English", "label": "English"},
    {"value": "Hindi", "label": "हिंदी / Hindi"},
    {"value": "Gujarati", "label": "ગુજરાતી / Gujarati"},
]

DURATION_OPTIONS = [30, 45, 60]


# -------------------------
# Simple DB operations
# -------------------------

def create_chapter_material(
    db: Session,
    *,
    admin_id: int,
    std: str,
    subject: str,
    sem: str,
    board: str,
    chapter_number: str,
    file_info: Dict[str, Any],
) -> ChapterMaterial:
    chapter_material = ChapterMaterial(
        admin_id=admin_id,
        std=std.strip(),
        sem=sem.strip() if sem else "",
        board=board.strip() if board else "",
        subject=subject.strip(),
        chapter_number=chapter_number.strip(),
        file_name=file_info["filename"],
        file_path=file_info["file_path"],
        file_size=file_info["file_size"],
    )

    db.add(chapter_material)
    db.commit()
    db.refresh(chapter_material)
    return chapter_material


def get_chapter_material(db: Session, material_id: int) -> Optional[ChapterMaterial]:
    return db.query(ChapterMaterial).filter(ChapterMaterial.id == material_id).first()


def list_chapter_materials(
    db: Session,
    admin_id: int,
    std: Optional[str] = None,
    subject: Optional[str] = None,
    sem: Optional[str] = None,
    board: Optional[str] = None,
) -> List[ChapterMaterial]:
    query = db.query(ChapterMaterial).filter(ChapterMaterial.admin_id == admin_id)
    if std:
        query = query.filter(ChapterMaterial.std == std.strip())
    if subject:
        query = query.filter(ChapterMaterial.subject == subject.strip())
    if sem:
        query = query.filter(ChapterMaterial.sem == sem.strip())
    if board:
        query = query.filter(ChapterMaterial.board == board.strip())
    return query.order_by(ChapterMaterial.created_at.desc()).all()


def list_materials(
    admin_id: int,
    *,
    std: Optional[str] = None,
    subject: Optional[str] = None,
    sem: Optional[str] = None,
    board: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return chapter materials for student portal screens using psycopg."""

    filters = ["admin_id = %(admin_id)s"]
    params: Dict[str, Any] = {"admin_id": admin_id}

    def _add_filter(column: str, key: str, value: Optional[str]) -> None:
        if value:
            params[key] = value.strip()
            filters.append(f"{column} = %({key})s")

    _add_filter("std", "std", std)
    _add_filter("subject", "subject", subject)
    _add_filter("sem", "sem", sem)
    _add_filter("board", "board", board)

    where_clause = " AND ".join(filters)
    query = f"""
        SELECT
            id,
            admin_id,
            std,
            subject,
            sem,
            board,
            chapter_number,
            file_name,
            file_path,
            file_size,
            created_at,
            updated_at
        FROM chapter_materials
        WHERE {where_clause}
        ORDER BY created_at DESC
    """

    with get_pg_cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def list_recent_chapter_materials(db: Session, admin_id: int, limit: int = 5) -> List[ChapterMaterial]:
    return (
        db.query(ChapterMaterial)
        .filter(ChapterMaterial.admin_id == admin_id)
        .order_by(ChapterMaterial.created_at.desc())
        .limit(limit)
        .all()
    )


def update_chapter_material_db(
    db: Session,
    chapter_material: ChapterMaterial,
    *,
    std: str,
    subject: str,
    sem: str,
    board: str,
    chapter_number: str,
    new_file_info: Optional[Dict[str, Any]] = None,
):
    chapter_material.std = std.strip()
    chapter_material.subject = subject.strip()
    chapter_material.sem = sem.strip() if sem else ""
    chapter_material.board = board.strip() if board else ""
    chapter_material.chapter_number = chapter_number.strip()

    if new_file_info is not None:
        old_file_path = chapter_material.file_path
        chapter_material.file_name = new_file_info["filename"]
        chapter_material.file_path = new_file_info["file_path"]
        chapter_material.file_size = new_file_info["file_size"]
        # delete old file after commit (caller should handle delete_file)
        try:
            delete_file(old_file_path)
        except Exception:
            logger.warning("Failed to delete old file: %s", old_file_path)

    db.commit()
    db.refresh(chapter_material)
    return chapter_material


def delete_chapter_material_db(db: Session, chapter_material: ChapterMaterial):
    file_path = chapter_material.file_path
    db.delete(chapter_material)
    db.commit()
    try:
        if file_path:
            delete_file(file_path)
    except Exception:
        logger.warning("Failed to delete material file: %s", file_path)
    delete_language_metadata(chapter_material.admin_id, chapter_material.id)


def get_dashboard_stats(db: Session, admin_id: int) -> Dict[str, int]:
    total_materials = db.query(func.count(ChapterMaterial.id)).filter(
        ChapterMaterial.admin_id == admin_id
    ).scalar()

    unique_subjects = db.query(func.count(func.distinct(ChapterMaterial.subject))).filter(
        ChapterMaterial.admin_id == admin_id
    ).scalar()

    unique_classes = db.query(func.count(func.distinct(ChapterMaterial.std))).filter(
        ChapterMaterial.admin_id == admin_id
    ).scalar()

    return {
        "total_materials": total_materials or 0,
        "unique_subjects": unique_subjects or 0,
        "unique_classes": unique_classes or 0,
    }


# -------------------------
# Topic file helpers & persistence
# -------------------------

def _load_topics_path(material: ChapterMaterial) -> str:
    topics_folder = os.path.join(UPLOAD_DIR, f"chapter_materials/admin_{material.admin_id}")
    return os.path.join(topics_folder, f"extracted_topics_{material.id}.json")


def _language_metadata_path(admin_id: int, material_id: int) -> Path:
    return Path(UPLOAD_DIR) / f"chapter_materials/admin_{admin_id}" / f"language_metadata_{material_id}.json"


def persist_language_metadata(admin_id: int, material_id: int, metadata: Dict[str, Any]) -> None:
    metadata_path = _language_metadata_path(admin_id, material_id)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.warning("Failed to persist language metadata for material %s: %s", material_id, exc)


def read_language_metadata(admin_id: int, material_id: int) -> Optional[Dict[str, Any]]:
    metadata_path = _language_metadata_path(admin_id, material_id)
    if not metadata_path.exists():
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read language metadata for material %s: %s", material_id, exc)
        return None


def delete_language_metadata(admin_id: int, material_id: int) -> None:
    metadata_path = _language_metadata_path(admin_id, material_id)
    try:
        if metadata_path.exists():
            metadata_path.unlink()
    except OSError as exc:
        logger.warning("Failed to delete language metadata for material %s: %s", material_id, exc)


def load_material_topics(admin_id: int, material_id: int) -> Dict[str, Any]:
    topics_path = os.path.join(UPLOAD_DIR, f"chapter_materials/admin_{admin_id}", f"extracted_topics_{material_id}.json")
    if not os.path.exists(topics_path):
        raise FileNotFoundError("Topics not found")
    with open(topics_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def persist_material_topics(admin_id: int, material_id: int, payload: Dict[str, Any]) -> None:
    admin_folder = f"chapter_materials/admin_{admin_id}"
    topics_dir = os.path.join(UPLOAD_DIR, admin_folder)
    os.makedirs(topics_dir, exist_ok=True)

    topics_json_path = os.path.join(topics_dir, f"extracted_topics_{material_id}.json")
    try:
        with open(topics_json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.warning("Failed to persist topics for material %s: %s", material_id, exc)


def save_extracted_topics_files(admin_id: int, material_id: int, extraction: Dict[str, Any]) -> Tuple[str, str]:
    admin_dir = os.path.join(UPLOAD_DIR, f"chapter_materials/admin_{admin_id}")
    os.makedirs(admin_dir, exist_ok=True)

    txt_path = os.path.join(admin_dir, f"extracted_topics_{material_id}.txt")
    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write(extraction.get("topics_text", ""))

    json_path = os.path.join(admin_dir, f"extracted_topics_{material_id}.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(extraction, jf, indent=2, ensure_ascii=False)

    return txt_path, json_path


def read_topics_file_if_exists(admin_id: int, material_id: int) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    topics_file = Path(UPLOAD_DIR) / f"chapter_materials/admin_{admin_id}" / f"extracted_topics_{material_id}.json"
    if topics_file.exists():
        try:
            with open(topics_file, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
                topics_list = payload.get("topics", [])
                return payload, topics_list
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading topics file for material {material_id}: {e}")
            return None, []
    return None, []


# -------------------------
# Utilities for text conversion (topic -> text)
# -------------------------

def _safe_join(parts: List[str]) -> str:
    return "\n".join(filter(None, parts))


def topic_to_text(topic: Dict[str, Any]) -> str:
    lines: List[str] = []

    title = topic.get("title")
    if title:
        lines.append(f"Topic: {title}")

    summary = topic.get("summary")
    if summary:
        lines.append(f"Summary: {summary}")

    if topic.get("content"):
        lines.append(f"Content: {topic['content']}")

    subtopics = topic.get("subtopics", [])
    if isinstance(subtopics, list) and subtopics:
        sub_lines: List[str] = []
        for subtopic in subtopics:
            if isinstance(subtopic, dict):
                stitle = subtopic.get("title")
                snarration = subtopic.get("narration")
                part = _safe_join([
                    f"Subtopic: {stitle}" if stitle else None,
                    snarration,
                ])
                if part:
                    sub_lines.append(part)
            else:
                sub_lines.append(f"Subtopic: {subtopic}")
        if sub_lines:
            lines.append("\n".join(sub_lines))

    return "\n".join(lines)


# -------------------------
# Manual topic append & assistant topics add
# -------------------------

def append_manual_topic_to_file(admin_id: int, material: ChapterMaterial, topic_payload: Dict[str, Any]) -> Dict[str, Any]:
    admin_folder = f"chapter_materials/admin_{material.admin_id}"
    topics_dir = os.path.join(UPLOAD_DIR, admin_folder)
    os.makedirs(topics_dir, exist_ok=True)
    topics_path = os.path.join(topics_dir, f"extracted_topics_{material.id}.json")

    topics_payload: dict = {
        "material_id": material.id,
        "language_code": None,
        "language_label": None,
        "topics": [],
        "headings": [],
        "excerpt": "",
        "topics_text": "",
        "chapter_title": material.chapter_number,
    }

    if os.path.exists(topics_path):
        try:
            with open(topics_path, "r", encoding="utf-8") as tf:
                topics_payload = json.load(tf)
        except json.JSONDecodeError:
            pass

    topics_list = topics_payload.setdefault("topics", [])
    topics_list.append(topic_payload)

    with open(topics_path, "w", encoding="utf-8") as tf:
        json.dump(topics_payload, tf, ensure_ascii=False, indent=2)

    topics_payload.setdefault("chapter_title", material.chapter_number)
    return {"topic": topic_payload, "topics": topics_list, "chapter_title": topics_payload.get("chapter_title")}


def add_assistant_topics_to_file(admin_id: int, material: ChapterMaterial, selected_suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
    admin_folder = f"chapter_materials/admin_{material.admin_id}"
    topics_dir = os.path.join(UPLOAD_DIR, admin_folder)
    os.makedirs(topics_dir, exist_ok=True)
    topics_path = os.path.join(topics_dir, f"extracted_topics_{material.id}.json")

    topics_payload: dict = {
        "material_id": material.id,
        "language_code": None,
        "language_label": None,
        "topics": [],
        "headings": [],
        "excerpt": "",
        "topics_text": "",
        "chapter_title": material.chapter_number,
    }

    if os.path.exists(topics_path):
        try:
            with open(topics_path, "r", encoding="utf-8") as tf:
                topics_payload = json.load(tf)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse topics file for material {material.id}")

    existing_topics = topics_payload.get("topics", [])
    existing_titles = {
        str(topic.get("title", "")).strip().lower()
        for topic in existing_topics
        if isinstance(topic, dict)
    }

    added_topics = []
    skipped_duplicates = []

    for suggestion in selected_suggestions:
        if not isinstance(suggestion, dict):
            continue
        title = str(suggestion.get("title", "")).strip()
        if not title:
            continue
        normalized_title = title.lower()
        if normalized_title in existing_titles:
            skipped_duplicates.append(title)
            continue
        new_topic = {
            "title": title,
            "summary": str(suggestion.get("summary", "")).strip(),
            "supporting_quote": str(suggestion.get("supporting_quote", "")).strip(),
            "is_assistant_generated": True,
            "subtopics": [],
        }
        existing_topics.append(new_topic)
        added_topics.append(new_topic)
        existing_titles.add(normalized_title)

    topics_payload["topics"] = existing_topics

    try:
        with open(topics_path, "w", encoding="utf-8") as tf:
            json.dump(topics_payload, tf, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save topics for material {material.id}: {e}")
        raise

    return {
        "added_topics": added_topics,
        "skipped_duplicates": skipped_duplicates,
        "total_topics": len(existing_topics),
        "chapter_title": topics_payload.get("chapter_title"),
    }


# -------------------------
# Read a PDF snippet for assistant context (uses topic_extractor.read_pdf if available)
# -------------------------

def read_pdf_context_for_material(material_file_path: str, max_chars: int = 12_000) -> str:
    try:
        from app.utils.topic_extractor import read_pdf
        return read_pdf(Path(material_file_path))[:max_chars]
    except Exception:
        logger.debug("read_pdf not available or failed; returning empty string")
        return ""

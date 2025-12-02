# app/repositories/chapter_material_repository.py

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models import ChapterMaterial, LectureGen
from app.schemas.admin_schema import WorkType
from app.utils.file_handler import (
    save_uploaded_file,
    delete_file,
    ALLOWED_PDF_EXTENSIONS,
    ALLOWED_PDF_TYPES,
    UPLOAD_DIR,
)
from app.config import get_settings
from app.database import SessionLocal

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
    chapter_title: Optional[str] = None,
    file_info: Dict[str, Any],
) -> ChapterMaterial:
    normalized_title = (chapter_title or chapter_number).strip()
    chapter_material = ChapterMaterial(
        admin_id=admin_id,
        std=std.strip(),
        sem=sem.strip() if sem else "",
        board=board.strip() if board else "",
        subject=subject.strip(),
        chapter_number=chapter_number.strip(),
        chapter_title=normalized_title,
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
) -> List[ChapterMaterial]:
    query = db.query(ChapterMaterial).filter(ChapterMaterial.admin_id == admin_id)
    
    # Require both std and subject for filtering
    if std and subject:
        # Case-insensitive filtering for std
        std_clean = std.strip().lower()
        query = query.filter(func.lower(ChapterMaterial.std) == std_clean)
        
        # Fuzzy matching for subject
        subject_clean = subject.strip().lower()
        
        # Try exact match first
        exact_results = query.filter(func.lower(ChapterMaterial.subject) == subject_clean).all()
        
        if exact_results:
            query = query.filter(func.lower(ChapterMaterial.subject) == subject_clean)
        else:
            # Try partial/contains match
            contains_results = query.filter(func.lower(ChapterMaterial.subject).contains(subject_clean)).all()
            
            if contains_results:
                query = query.filter(func.lower(ChapterMaterial.subject).contains(subject_clean))
            else:
                # Try common variations and spelling corrections
                variations = get_subject_variations(subject_clean)
                
                if variations:
                    query = query.filter(func.lower(ChapterMaterial.subject).in_(variations))
                else:
                    # No matches found, return empty
                    return []
    
    elif std:
        # Only std provided
        std_clean = std.strip().lower()
        query = query.filter(func.lower(ChapterMaterial.std) == std_clean)
    
    elif subject:
        # Only subject provided
        subject_clean = subject.strip().lower()
        
        # Try exact match first
        exact_results = query.filter(func.lower(ChapterMaterial.subject) == subject_clean).all()
        
        if exact_results:
            query = query.filter(func.lower(ChapterMaterial.subject) == subject_clean)
        else:
            # Try partial/contains match
            contains_results = query.filter(func.lower(ChapterMaterial.subject).contains(subject_clean)).all()
            
            if contains_results:
                query = query.filter(func.lower(ChapterMaterial.subject).contains(subject_clean))
            else:
                # Try common variations and spelling corrections
                variations = get_subject_variations(subject_clean)
                
                if variations:
                    query = query.filter(func.lower(ChapterMaterial.subject).in_(variations))
                else:
                    # No matches found, return empty
                    return []
    
    return query.order_by(ChapterMaterial.created_at.desc()).all()


def list_materials(
    admin_id: int,
    *,
    std: Optional[str] = None,
    subject: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Compatibility helper returning serialized chapter materials.

    This preserves the old repository interface relied upon by the student
    portal service while reusing the SQLAlchemy-powered list_chapter_materials.
    """

    db = SessionLocal()
    try:
        materials = list_chapter_materials(db, admin_id, std=std, subject=subject)
        return [material.to_dict() for material in materials]
    finally:
        db.close()


def get_chapter_filter_options(
    db: Session,
    *,
    admin_id: int,
) -> Dict[str, Any]:
    rows = (
        db.query(
            ChapterMaterial.std.label("std"),
            ChapterMaterial.subject.label("subject"),
            ChapterMaterial.chapter_number.label("chapter"),
        )
        .filter(ChapterMaterial.admin_id == admin_id)
        .all()
    )

    classes: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        std_value = (row.std or "").strip()
        subject_value = (row.subject or "").strip()
        chapter_value = (row.chapter or "").strip()
        if not std_value or not subject_value:
            continue

        std_slug = std_value.lower().replace(" ", "_")
        std_entry = classes.setdefault(
            std_slug,
            {
                "label": std_value,
                "value": std_value,
                "slug": std_slug,
                "subjects": {},
            },
        )

        subject_slug = subject_value.lower().replace(" ", "_")
        subject_entry = std_entry["subjects"].setdefault(
            subject_slug,
            {
                "label": subject_value,
                "value": subject_value,
                "slug": subject_slug,
                "chapters": set(),
            },
        )

        if chapter_value:
            subject_entry["chapters"].add(chapter_value)

    response: List[Dict[str, Any]] = []
    for std_entry in classes.values():
        subjects: List[Dict[str, Any]] = []
        for subject_entry in std_entry["subjects"].values():
            subjects.append(
                {
                    "label": subject_entry["label"],
                    "value": subject_entry["value"],
                    "slug": subject_entry["slug"],
                    "chapters": sorted(subject_entry["chapters"]),
                }
            )
        response.append(
            {
                "label": std_entry["label"],
                "value": std_entry["value"],
                "slug": std_entry["slug"],
                "subjects": sorted(subjects, key=lambda s: s["label"].lower()),
            }
        )

    response.sort(key=lambda entry: entry["label"].lower())
    return {"classes": response}


def list_chapters_for_selection(
    db: Session,
    *,
    admin_id: int,
    std: str,
    subject: str,
) -> List[str]:
    query = (
        db.query(ChapterMaterial.chapter_number)
        .filter(ChapterMaterial.admin_id == admin_id)
        .filter(func.lower(ChapterMaterial.std) == std.strip().lower())
        .filter(func.lower(ChapterMaterial.subject) == subject.strip().lower())
    )
    chapters = sorted({(row.chapter_number or "").strip() for row in query.all() if row.chapter_number})
    return chapters


def get_subject_variations(subject: str) -> List[str]:
    """Generate common subject variations and spelling corrections"""
    variations = [subject]
    
    # Common subject mappings
    subject_mappings = {
        'science': ['science', 'sci', 'scince', 'sceince', 'scienc'],
        'math': ['math', 'maths', 'mathematics', 'mathmatic'],
        'english': ['english', 'eng', 'engish', 'englis'],
        'hindi': ['hindi', 'hindi', 'hindi'],
        'social science': ['social science', 'socialscience', 'social', 'sst', 'social studies'],
        'physics': ['physics', 'physic', 'phy'],
        'chemistry': ['chemistry', 'chem', 'chemical'],
        'biology': ['biology', 'bio', 'bilogy'],
        'computer': ['computer', 'comp', 'computer science', 'cs', 'it'],
        'history': ['history', 'hist', 'histry'],
        'geography': ['geography', 'geo', 'geography'],
        'economics': ['economics', 'eco', 'economics', 'econ'],
    }
    
    # Find matching variations
    for key, values in subject_mappings.items():
        if subject in values or key in subject or any(v in subject for v in values):
            variations.extend(values)
            variations.append(key)
            break
    
    # Remove duplicates
    return list(set(variations))


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
    chapter_title: Optional[str] = None,
    new_file_info: Optional[Dict[str, Any]] = None,
):
    chapter_material.std = std.strip()
    chapter_material.subject = subject.strip()
    chapter_material.sem = sem.strip() if sem else ""
    chapter_material.board = board.strip() if board else ""
    chapter_material.chapter_number = chapter_number.strip()
    if chapter_title is not None:
        normalized_title = chapter_title.strip()
        chapter_material.chapter_title = normalized_title or chapter_material.chapter_number
    elif not chapter_material.chapter_title:
        chapter_material.chapter_title = chapter_material.chapter_number

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


def formatFileSize(bytes):
    """Format file size in human readable format"""
    if not bytes:
        return '—'
    
    mb = bytes / (1024 * 1024)
    if mb < 0.5:
        kb = bytes / 1024
        return f"{kb:.1f} KB"
    
    return f"{mb:.2f} MB"


def get_chapter_overview_data(db: Session, admin_id: int) -> List[Dict[str, Any]]:
    """
    Get chapter overview data with lecture information including:
    - subject, chapter title, topics, lecture size, and video info
    """
    from app.models import LectureGen
    from sqlalchemy import desc
    
    # Query chapter materials with their lectures
    query = (
        db.query(
            ChapterMaterial,
            LectureGen.lecture_title,
            LectureGen.lecture_link,
            LectureGen.lecture_uid,
            LectureGen.created_at.label('lecture_created_at')
        )
        .filter(ChapterMaterial.admin_id == admin_id)
        .outerjoin(LectureGen, LectureGen.material_id == ChapterMaterial.id)
        .order_by(desc(ChapterMaterial.created_at))
    )
    
    results = []
    for material, lecture_title, lecture_link, lecture_uid, lecture_created_at in query:
        # Get topics for this material if they exist
        topics_data = []
        extracted_chapter_title = None
        try:
            payload, topics_list = read_topics_file_if_exists(material.admin_id, material.id)
            if topics_list:
                # Take only first few topics as preview
                topics_data = topics_list[:5]  # Limit to 5 topics for overview
            # Get chapter title from extracted topics data
            if payload and payload.get("chapter_title"):
                extracted_chapter_title = payload.get("chapter_title")
        except Exception:
            topics_data = []
        
        # Use extracted chapter title if available, otherwise fallback to material chapter_number
        chapter_title = extracted_chapter_title or material.chapter_number
        
        # Calculate lecture size if lecture exists
        lecture_size = 0
        video_info = None
        if lecture_link:
            try:
                import os
                from pathlib import Path
                storage_base = Path("./storage/lectures")
                lecture_json_path = storage_base / lecture_uid / "lecture.json"
                if lecture_json_path.exists():
                    lecture_size = lecture_json_path.stat().st_size
                    # For video info, we'll assume video exists alongside JSON
                    video_path = storage_base / lecture_uid / "lecture.mp4"
                    if video_path.exists():
                        video_size = video_path.stat().st_size
                        video_info = {
                            "size": video_size,
                            "path": str(video_path)
                        }
            except Exception:
                pass
        
        chapter_data = {
            "material_id": material.id,
            "subject": material.subject,
            "chapter": chapter_title,
            "topics": topics_data,
            "size": formatFileSize(lecture_size) if lecture_size else "41.1 KB",
            "video": video_info
        }
        
        results.append(chapter_data)
    
    return results


def update_chapter_overview_fields(
    db: Session,
    *,
    chapter_material: ChapterMaterial,
    std: Optional[str] = None,
    subject: Optional[str] = None,
    sem: Optional[str] = None,
    board: Optional[str] = None,
    chapter_number: Optional[str] = None,
    chapter_title: Optional[str] = None,
) -> ChapterMaterial:
    updated = False

    if std is not None:
        chapter_material.std = std.strip()
        updated = True
    if subject is not None:
        chapter_material.subject = subject.strip()
        updated = True
    if sem is not None:
        chapter_material.sem = sem.strip()
        updated = True
    if board is not None:
        chapter_material.board = board.strip()
        updated = True
    if chapter_number is not None:
        chapter_material.chapter_number = chapter_number.strip()
        updated = True
    if chapter_title is not None:
        normalized_title = chapter_title.strip()
        chapter_material.chapter_title = normalized_title or chapter_material.chapter_number
        updated = True

    if updated:
        db.add(chapter_material)
        db.commit()
        db.refresh(chapter_material)

    return chapter_material


# -------------------------
# Topic file helpers & persistence
# -------------------------

def _load_topics_path(material: ChapterMaterial) -> str:
    topics_folder = os.path.join(UPLOAD_DIR, f"chapter_materials/admin_{material.admin_id}")
    return os.path.join(topics_folder, f"extracted_topics_{material.id}.json")


def load_material_topics(admin_id: int, material_id: int) -> Dict[str, Any]:
    topics_path = os.path.join(UPLOAD_DIR, f"chapter_materials/admin_{admin_id}", f"extracted_topics_{material_id}.json")
    if not os.path.exists(topics_path):
        raise FileNotFoundError("Topics not found")
    with open(topics_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    _ensure_topic_ids_for_payload(admin_id, material_id, payload)
    return payload


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

    topics = extraction.get("topics") or []
    if isinstance(topics, list):
        _assign_topic_ids(topics)

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
                _ensure_topic_ids_for_payload(admin_id, material_id, payload)
                topics_list = payload.get("topics", [])
                return payload, topics_list
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading topics file for material {material_id}: {e}")
            return None, []
    return None, []


def _assign_topic_ids(topics: List[Dict[str, Any]]) -> bool:
    updated = False
    for idx, topic in enumerate(topics, start=1):
        if not isinstance(topic, dict):
            continue
        expected = str(idx)
        if topic.get("topic_id") != expected:
            topic["topic_id"] = expected
            updated = True
    return updated


def _ensure_topic_ids_for_payload(admin_id: int, material_id: int, payload: Dict[str, Any]) -> None:
    topics = payload.get("topics")
    if isinstance(topics, list) and _assign_topic_ids(topics):
        persist_material_topics(admin_id, material_id, payload)


def _assistant_suggestions_path(admin_id: int, material_id: int) -> str:
    return os.path.join(
        UPLOAD_DIR,
        f"chapter_materials/admin_{admin_id}",
        f"assistant_suggestions_{material_id}.json",
    )


def persist_assistant_suggestions(admin_id: int, material_id: int, suggestions: List[Dict[str, Any]]) -> None:
    path = _assistant_suggestions_path(admin_id, material_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "admin_id": admin_id,
        "material_id": material_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "suggestions": suggestions,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def load_assistant_suggestions(admin_id: int, material_id: int) -> Dict[str, Any]:
    path = _assistant_suggestions_path(admin_id, material_id)
    if not os.path.exists(path):
        return {"suggestions": []}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read assistant suggestions cache for material %s: %s", material_id, exc)
        return {"suggestions": []}


def get_cached_suggestions_by_ids(
    admin_id: int,
    material_id: int,
    suggestion_ids: List[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    cache = load_assistant_suggestions(admin_id, material_id)
    suggestions = cache.get("suggestions", []) or []
    index = {str(item.get("suggestion_id")): item for item in suggestions if item.get("suggestion_id")}
    resolved: List[Dict[str, Any]] = []
    missing: List[str] = []
    for sid in suggestion_ids:
        entry = index.get(str(sid))
        if entry:
            resolved.append(entry)
        else:
            missing.append(str(sid))
    return resolved, missing


def get_topics_by_ids(
    admin_id: int,
    material_id: int,
    topic_ids: List[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    payload, topics_list = read_topics_file_if_exists(admin_id, material_id)
    topics_list = topics_list or []
    index = {
        str(topic.get("topic_id")): topic
        for topic in topics_list
        if isinstance(topic, dict) and topic.get("topic_id")
    }
    resolved: List[Dict[str, Any]] = []
    missing: List[str] = []
    for tid in topic_ids:
        entry = index.get(str(tid))
        if entry:
            resolved.append(entry)
        else:
            missing.append(str(tid))
    return resolved, missing


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
    _assign_topic_ids(topics_list)
    if not topic_payload.get("topic_id"):
        topic_payload["topic_id"] = str(len([t for t in topics_list if isinstance(t, dict)]) + 1)
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
    _assign_topic_ids(existing_topics)
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
        new_topic["topic_id"] = str(len([t for t in existing_topics if isinstance(t, dict)]) + 1)
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

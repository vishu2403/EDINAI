"""
Lecture Repository - Data Storage and Retrieval Layer
Handles all database operations for lecture storage
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import cast, desc, func
from sqlalchemy.orm import Session
from sqlalchemy.sql.sqltypes import Integer

from app.models.chapter_material import LectureGen


def _slugify(value: Any) -> str:
    """Convert metadata values to slug format for comparisons."""
    return str(value or "").strip().lower().replace(" ", "_")


def _sort_key(value: str) -> Tuple[int, str]:
    """Sort numerically when possible, otherwise lexicographically."""
    try:
        return (0, f"{int(value):02d}")
    except (ValueError, TypeError):
        return (1, (value or "").lower())


def _clone_record(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of the payload to prevent accidental mutations."""
    return json.loads(json.dumps(payload or {}))


class LectureRepository:
    """Repository for managing lecture persistence via the database."""

    def __init__(self, db: Session) -> None:
        self._db = db

    async def _generate_lecture_id(self) -> str:
        numeric_max = (
            self._db.query(func.max(cast(LectureGen.lecture_uid, Integer)))
            .filter(LectureGen.lecture_uid.isnot(None))
            .filter(LectureGen.lecture_uid.op("~")(r"^\d+$"))
            .scalar()
        )
        next_id = (numeric_max or 0) + 1
        return str(next_id)

    @staticmethod
    def _default_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return metadata.copy() if metadata else {}

    @staticmethod
    def _metadata_value(metadata: Dict[str, Any], *keys: str, default: Optional[str] = None) -> Optional[str]:
        for key in keys:
            if key in metadata and metadata[key]:
                return metadata[key]
        return default

    def _build_default_url(self, record: Dict[str, Any]) -> Optional[str]:
        metadata = record.get("metadata") or {}
        std_value = self._metadata_value(metadata, "std", "class", default="general")
        subject_value = self._metadata_value(metadata, "subject", default="lecture")
        lecture_id = record.get("lecture_id")
        if not lecture_id:
            return None
        std_slug = _slugify(std_value)
        subject_slug = _slugify(subject_value)
        return f"/lectures/{std_slug}/{subject_slug}/{lecture_id}.json"

    def _db_summary(self, row: LectureGen, record: Dict[str, Any]) -> Dict[str, Any]:
        metadata = record.get("metadata") or {}
        std_value = metadata.get("std") or row.std or metadata.get("class") or "general"
        subject_value = metadata.get("subject") or row.subject or "lecture"
        division_value = metadata.get("division") or metadata.get("section")
        std_slug = _slugify(std_value)
        subject_slug = _slugify(subject_value)
        division_slug = _slugify(division_value) if division_value else None
        lecture_id = record.get("lecture_id") or row.lecture_uid
        default_url = self._build_default_url(record)

        return {
            "lecture_id": lecture_id,
            "title": record.get("title") or row.lecture_title,
            "language": record.get("language"),
            "total_slides": record.get("total_slides"),
            "estimated_duration": record.get("estimated_duration"),
            "created_at": record.get("created_at") or row.created_at.isoformat() if row.created_at else None,
            "fallback_used": record.get("fallback_used", False),
            "lecture_url": record.get("lecture_url") or row.lecture_link or default_url,
            "std": std_value,
            "subject": subject_value,
            "division": division_value,
            "std_slug": std_slug,
            "subject_slug": subject_slug,
            "division_slug": division_slug,
        }

    def _persist_record(self, row: LectureGen, record: Dict[str, Any]) -> Dict[str, Any]:
        cloned = _clone_record(record)
        metadata = cloned.get("metadata") or {}

        row.lecture_title = cloned.get("title") or row.lecture_title
        row.lecture_link = cloned.get("lecture_url") or row.lecture_link
        row.std = metadata.get("std") or metadata.get("class") or row.std
        row.subject = metadata.get("subject") or row.subject
        row.sem = metadata.get("sem") or row.sem
        row.board = metadata.get("board") or row.board
        row.lecture_data = cloned

        self._db.add(row)
        self._db.commit()
        self._db.refresh(row)

        cloned.setdefault("lecture_id", row.lecture_uid)
        cloned.setdefault("metadata", metadata)
        cloned.setdefault("lecture_url", row.lecture_link)
        cloned["db_record_id"] = row.id
        return cloned

    async def create_lecture(
        self,
        *,
        title: str,
        language: str,
        style: str,
        duration: int,
        slides: List[Dict[str, Any]],
        context: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        fallback_used: bool = False,
        admin_id: Optional[int] = None,
        material_id: Optional[int] = None,
        std: Optional[str] = None,
        subject: Optional[str] = None,
        sem: Optional[str] = None,
        board: Optional[str] = None,
        lecture_uid: Optional[str] = None,
        lecture_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        metadata = self._default_metadata(metadata)
        lecture_id = lecture_uid or await self._generate_lecture_id()
        created_at = datetime.utcnow().isoformat()

        record: Dict[str, Any] = {
            "lecture_id": lecture_id,
            "title": title,
            "language": language,
            "style": style,
            "requested_duration": duration,
            "estimated_duration": len(slides) * 3,
            "total_slides": len(slides),
            "slides": slides,
            "context": context,
            "created_at": created_at,
            "updated_at": created_at,
            "fallback_used": fallback_used,
            "source_text": text,
            "metadata": metadata,
            "play_count": 0,
            "last_played_at": None,
        }

        record["lecture_url"] = lecture_url or self._build_default_url(record)

        admin_value = admin_id or metadata.get("admin_id") or 0
        material_value = material_id or metadata.get("material_id") or 0
        subject_value = subject or metadata.get("subject")
        std_value = std or metadata.get("std") or metadata.get("class")
        sem_value = sem or metadata.get("sem")
        board_value = board or metadata.get("board")

        existing_row = (
            self._db.query(LectureGen)
            .filter(LectureGen.lecture_uid == lecture_id)
            .first()
        )

        if existing_row:
            db_row = existing_row
            db_row.admin_id = admin_value or db_row.admin_id
            db_row.material_id = material_value or db_row.material_id
        else:
            db_row = LectureGen(
                admin_id=admin_value,
                material_id=material_value,
                lecture_uid=lecture_id,
                lecture_title=title,
                lecture_link=record["lecture_url"],
                subject=subject_value,
                std=std_value,
                sem=sem_value,
                board=board_value,
            )

        return self._persist_record(db_row, record)

    async def get_lecture(self, lecture_id: str) -> Dict[str, Any]:
        row = (
            self._db.query(LectureGen)
            .filter(LectureGen.lecture_uid == str(lecture_id))
            .first()
        )
        if not row or not row.lecture_data:
            raise FileNotFoundError(f"Lecture {lecture_id} not found")

        record = _clone_record(row.lecture_data)
        record.setdefault("lecture_id", row.lecture_uid)
        record.setdefault("metadata", {})
        record.setdefault("lecture_url", row.lecture_link)
        record["db_record_id"] = row.id
        return record

    async def update_lecture(self, lecture_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        row = (
            self._db.query(LectureGen)
            .filter(LectureGen.lecture_uid == str(lecture_id))
            .first()
        )
        if not row or not row.lecture_data:
            raise FileNotFoundError(f"Lecture {lecture_id} not found")

        record = _clone_record(row.lecture_data)
        record.update(updates)
        record["updated_at"] = datetime.utcnow().isoformat()

        metadata = record.get("metadata") or {}
        row.lecture_title = record.get("title") or row.lecture_title
        row.lecture_link = record.get("lecture_url") or row.lecture_link
        row.std = metadata.get("std") or metadata.get("class") or row.std
        row.subject = metadata.get("subject") or row.subject
        row.sem = metadata.get("sem") or row.sem
        row.board = metadata.get("board") or row.board
        row.lecture_data = record

        self._db.add(row)
        self._db.commit()
        self._db.refresh(row)

        record["db_record_id"] = row.id
        return record
    async def delete_lectures_by_metadata(
            self,
            *,
            std: str,
            subject: str,
            division: Optional[str] = None,
            lecture_id: Optional[str] = None,
        ) -> List[Dict[str, Any]]:
            """Delete lectures matching metadata filters and return details of deletions."""
            std_filter = _slugify(std)
            subject_filter = _slugify(subject)
            division_filter = _slugify(division) if division else None
            matched: List[Dict[str, Any]] = []
            for lecture_dir in self._storage_dir.iterdir():
                if not lecture_dir.is_dir():
                    continue
                json_path = lecture_dir / "lecture.json"
                if not json_path.is_file():
                    continue
                try:
                    record = await self._load_json(json_path)
                except Exception as exc:
                    print(f":warning: Error loading lecture {lecture_dir.name} for deletion: {exc}")
                    continue
                metadata = record.get("metadata") or {}
                std_value = metadata.get("std") or metadata.get("class") or "general"
                subject_value = metadata.get("subject") or "lecture"
                division_value = metadata.get("division") or metadata.get("section")
                std_slug = _slugify(std_value)
                subject_slug = _slugify(subject_value)
                division_slug = _slugify(division_value) if division_value else None
                if std_slug != std_filter or subject_slug != subject_filter:
                    continue
                if division_filter and division_slug != division_filter:
                    continue
                lecture_id_value = record.get("lecture_id") or lecture_dir.name
                if lecture_id and lecture_id_value != lecture_id:
                    continue
                matched.append(
                    {
                        "lecture_id": lecture_id_value,
                        "title": record.get("title"),
                        "std": str(std_value),
                        "subject": str(subject_value),
                        "division": str(division_value) if division_value else None,
                        "std_slug": std_slug,
                        "subject_slug": subject_slug,
                        "division_slug": division_slug,
                    }
                )
            deleted: List[Dict[str, Any]] = []
            for entry in matched:
                lecture_id_value = entry["lecture_id"]
                try:
                    if await self.delete_lecture(lecture_id_value):
                        deleted.append(entry)
                except Exception as exc:  # pragma: no cover - defensive guard
                    print(f":warning: Error deleting lecture {lecture_id_value}: {exc}")
            return deleted

    async def update_slide(
        self,
        lecture_id: str,
        slide_number: int,
        slide_updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        record = await self.get_lecture(lecture_id)
        slides = record.get("slides") or []

        for slide in slides:
            if slide.get("number") == slide_number:
                slide.update(slide_updates)
                break

        if "narration" in slide_updates:
            record["context"] = "\n\n".join(
                slide.get("narration", "") for slide in slides if slide.get("narration")
            )

        record["slides"] = slides
        return await self.update_lecture(lecture_id, record)

    async def delete_lecture(self, lecture_id: str) -> bool:
        row = (
            self._db.query(LectureGen)
            .filter(LectureGen.lecture_uid == str(lecture_id))
            .first()
        )
        if not row:
            return False

        self._db.delete(row)
        self._db.commit()
        return True

    async def list_lectures(
        self,
        *,
        language: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        std: Optional[str] = None,
        subject: Optional[str] = None,
        division: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        rows = (
            self._db.query(LectureGen)
            .order_by(desc(LectureGen.created_at))
            .all()
        )

        std_filter = _slugify(std) if std else None
        subject_filter = _slugify(subject) if subject else None
        division_filter = _slugify(division) if division else None
        lang_filter = (language or "").lower() if language else None

        summaries: List[Dict[str, Any]] = []
        for row in rows:
            if not row.lecture_data:
                continue
            record = _clone_record(row.lecture_data)
            metadata = record.get("metadata") or {}

            if lang_filter and (record.get("language") or "").lower() != lang_filter:
                continue

            std_value = metadata.get("std") or metadata.get("class") or row.std or "general"
            subject_value = metadata.get("subject") or row.subject or "lecture"
            division_value = metadata.get("division") or metadata.get("section")

            if std_filter and _slugify(std_value) != std_filter:
                continue
            if subject_filter and _slugify(subject_value) != subject_filter:
                continue
            if division_filter and _slugify(division_value) != division_filter:
                continue

            summaries.append(self._db_summary(row, record))

        return summaries[offset : offset + limit]

    async def list_played_lectures(self) -> List[Dict[str, Any]]:
        rows = (
            self._db.query(LectureGen)
            .order_by(desc(LectureGen.updated_at))
            .all()
        )

        played: List[Dict[str, Any]] = []
        for row in rows:
            record = row.lecture_data or {}
            play_count = int(record.get("play_count") or 0)
            if play_count <= 0:
                continue
            played.append(
                {
                    "lecture_id": row.lecture_uid,
                    "title": record.get("title") or row.lecture_title,
                    "language": record.get("language"),
                    "play_count": play_count,
                    "last_played_at": record.get("last_played_at"),
                    "lecture_url": record.get("lecture_url") or row.lecture_link,
                }
            )

        played.sort(key=lambda item: item.get("last_played_at") or "", reverse=True)
        return played

    async def get_class_subject_filters(self) -> Dict[str, Any]:
        rows = (
            self._db.query(LectureGen.std, LectureGen.subject)
            .filter(LectureGen.std.isnot(None), LectureGen.subject.isnot(None))
            .distinct()
            .all()
        )

        class_map: Dict[str, Dict[str, Any]] = {}
        for std_value, subject_value in rows:
            std_value = (std_value or "").strip()
            subject_value = (subject_value or "").strip()
            if not std_value or not subject_value:
                continue
            entry = class_map.setdefault(
                std_value,
                {"std": std_value, "subjects": set()},
            )
            entry["subjects"].add(subject_value)

        normalized_classes = []
        for entry in sorted(class_map.values(), key=lambda item: _sort_key(item["std"])):
            normalized_classes.append(
                {
                    "std": entry["std"],
                    "subject": sorted(entry["subjects"]),
                }
            )

        return {
            "classes": normalized_classes,
        }

    async def lecture_exists(self, lecture_id: str) -> bool:
        return (
            self._db.query(LectureGen)
            .filter(LectureGen.lecture_uid == str(lecture_id))
            .count()
            > 0
        )

    async def record_play(self, lecture_id: str) -> Dict[str, Any]:
        record = await self.get_lecture(lecture_id)
        play_count = int(record.get("play_count") or 0) + 1
        timestamp = datetime.utcnow().isoformat()

        record.update(
            {
                "play_count": play_count,
                "last_played_at": timestamp,
            }
        )
        return await self.update_lecture(lecture_id, record)

    async def get_lecture_stats(self) -> Dict[str, Any]:
        rows = self._db.query(LectureGen).all()
        stats = {
            "total_lectures": 0,
            "by_language": {},
            "fallback_count": 0,
            "total_slides": 0,
        }

        for row in rows:
            record = row.lecture_data or {}
            stats["total_lectures"] += 1
            language = record.get("language", "Unknown")
            stats["by_language"][language] = stats["by_language"].get(language, 0) + 1
            if record.get("fallback_used"):
                stats["fallback_count"] += 1
            stats["total_slides"] += record.get("total_slides", 0) or 0

        return stats

    async def get_source_text(self, lecture_id: str) -> str:
        record = await self.get_lecture(lecture_id)
        source_text = record.get("source_text")
        if not source_text:
            raise FileNotFoundError(f"Source text not found for lecture {lecture_id}")
        return source_text
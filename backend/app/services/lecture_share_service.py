"""Service for sharing lectures with students."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.chapter_material import ChapterMaterial, LectureGen
from app.repository import student_portal_repository
from app.repository import lecture_share_repository
from app.schemas.lecture_schema import LectureShareRequest
from app.repository.lecture_repository import _clone_record, _slugify, get_lecture, LectureRepository
from app.postgres import get_pg_cursor

logger = logging.getLogger(__name__)


class LectureShareService:
    """Coordinate lecture sharing to students within a class."""

    def __init__(self, db: Session) -> None:
        self._db = db

    @staticmethod
    def _clean_subject(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        cleaned = str(value).strip()
        if not cleaned:
            return None
        if cleaned.lower() in {"lecture", "general", "subject"}:
            return None
        return cleaned

    @staticmethod
    def _format_title(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        cleaned = str(value).strip()
        if not cleaned:
            return None
        if cleaned.islower() or cleaned.isupper():
            return cleaned.title()
        if "" in cleaned and cleaned.replace("", "").isalnum():
            return cleaned.replace("_", " ").title()
        return cleaned

    @staticmethod
    def _extract_subject(payload: Dict[str, any], depth: int = 0) -> Optional[str]:
        if depth > 5 or not isinstance(payload, dict):
            return None

        subject_keys = {
            "subject",
            "subject_name",
            "subject_title",
            "subjectlabel",
            "subjectlabeltext",
            "subject_slug",
            "subjectslug",
        }

        for key, value in payload.items():
            lowered = key.lower()
            if lowered in subject_keys and isinstance(value, str):
                candidate = value.strip()
                if lowered in {"subject_slug", "subjectslug"}:
                    candidate = candidate.replace("_", " ")
                candidate = LectureShareService._clean_subject(candidate)
                if candidate:
                    return candidate

        for value in payload.values():
            candidate: Optional[str] = None
            if isinstance(value, dict):
                candidate = LectureShareService._extract_subject(value, depth + 1)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        candidate = LectureShareService._extract_subject(item, depth + 1)
                        if candidate:
                            break
            if candidate:
                return candidate

        return None

    @staticmethod
    def _extract_title(payload: Dict[str, any], depth: int = 0) -> Optional[str]:
        if depth > 5 or not isinstance(payload, dict):
            return None

        title_keys = {
            "title",
            "lecture_title",
            "chapter_title",
            "topic_title",
            "heading",
        }

        for key, value in payload.items():
            lowered = key.lower()
            if lowered in title_keys and isinstance(value, str):
                candidate = LectureShareService._format_title(value)
                if candidate:
                    return candidate

        for value in payload.values():
            candidate: Optional[str] = None
            if isinstance(value, dict):
                candidate = LectureShareService._extract_title(value, depth + 1)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        candidate = LectureShareService._extract_title(item, depth + 1)
                        if candidate:
                            break
            if candidate:
                return candidate

        return None

    def _resolve_subject(
        self,
        *,
        request_subject: Optional[str],
        row: LectureGen,
        record: Dict[str, any],
    ) -> Optional[str]:
        if request_subject:
            return request_subject

        record_subject = self._clean_subject(record.get("subject"))
        if record_subject:
            return record_subject

        metadata = record.get("metadata") or {}

        # Direct metadata keys
        for key in ("subject", "subject_name", "lesson_subject", "subjectTitle"):
            value = metadata.get(key)
            candidate = self._clean_subject(value if isinstance(value, str) else None)
            if candidate:
                return candidate

        # Nested metadata containers that often include subject information
        for container_key in (
            "material_info",
            "material_snapshot",
            "log_context",
            "context",
            "context_payload",
            "chapter_overview",
            "lecture_metadata",
        ):
            nested = metadata.get(container_key)
            if isinstance(nested, dict):
                candidate = self._extract_subject(nested)
                if candidate:
                    return candidate

        # Generic recursive search
        candidate = self._extract_subject(metadata)
        if candidate:
            return candidate

        # Top-level record fields
        for container_key in (
            "material_info",
            "material_snapshot",
            "log_context",
            "context",
        ):
            nested = record.get(container_key)
            if isinstance(nested, dict):
                candidate = self._extract_subject(nested)
                if candidate:
                    return candidate

        subject_value = self._clean_subject(row.get("subject") if isinstance(row, dict) else getattr(row, "subject", None))
        if subject_value:
            return subject_value

        material_id = row.get("material_id") if isinstance(row, dict) else getattr(row, "material_id", None)
        if material_id:
            with get_pg_cursor() as cur:
                cur.execute("SELECT subject FROM chapter_material WHERE id = %(id)s", {"id": material_id})
                material = cur.fetchone()
            if material:
                candidate = self._clean_subject(material.get("subject"))
                if candidate:
                    return candidate

        return None

    def _resolve_std(
        self,
        *,
        request_std: str,
        row_std: Optional[str],
        record: Dict[str, any],
    ) -> str:
        if request_std:
            return request_std
        metadata = record.get("metadata") or {}
        return row_std or metadata.get("std") or metadata.get("class") or "general"

    def _build_lecture_url(
        self,
        *,
        record: Dict[str, any],
        row: LectureGen,
        resolved_std: str,
        resolved_subject: Optional[str],
    ) -> Optional[str]:
        if record.get("lecture_url"):
            return record["lecture_url"]

        lecture_id = record.get("lecture_id") or row.get("lecture_uid") if isinstance(row, dict) else row.lecture_uid
        if not lecture_id:
            return None

        std_slug = _slugify(resolved_std or "general")
        subject_slug = _slugify(resolved_subject or "lecture")
        return f"/lectures/{std_slug}/{subject_slug}/{lecture_id}.json"

    @staticmethod
    def _subject_from_url(lecture_url: Optional[str]) -> Optional[str]:
        if not lecture_url:
            return None
        parts = [segment for segment in lecture_url.strip("/").split("/") if segment]
        if len(parts) < 3:
            return None
        # Format expected: lectures/<std>/<subject>/<file>
        if parts[0] != "lectures":
            return None
        subject_slug = parts[2]
        # convert slug back to title case
        subject_slug = subject_slug.replace("_", " ").strip()
        if subject_slug.lower() in {"lecture", "general"}:
            return None
        return subject_slug.title() if subject_slug else None

    def _resolve_title(
        self,
        *,
        row: LectureGen,
        record: Dict[str, any],
    ) -> Optional[str]:
        metadata = record.get("metadata") or {}

        for key in ("title", "lecture_title", "chapter_title", "topic_title"):
            value = metadata.get(key)
            candidate = self._format_title(value if isinstance(value, str) else None)
            if candidate:
                return candidate

        for container_key in (
            "material_info",
            "material_snapshot",
            "log_context",
            "context",
            "lecture_metadata",
        ):
            nested = metadata.get(container_key)
            if isinstance(nested, dict):
                candidate = self._extract_title(nested)
                if candidate:
                    return candidate

        candidate = self._extract_title(metadata)
        if candidate:
            return candidate

        for key in ("title", "lecture_title", "chapter_title", "topic_title"):
            value = record.get(key)
            candidate = self._format_title(value if isinstance(value, str) else None)
            if candidate:
                return candidate

        for container_key in (
            "material_info",
            "material_snapshot",
            "log_context",
            "context",
        ):
            nested = record.get(container_key)
            if isinstance(nested, dict):
                candidate = self._extract_title(nested)
                if candidate:
                    return candidate

        candidate = self._format_title(row.get("lecture_title") if isinstance(row, dict) else getattr(row, "lecture_title", None))
        if candidate:
            return candidate

        return None

    async def share_lecture(
        self,
        *,
        lecture_id: str,
        payload: LectureShareRequest,
        shared_by: Optional[str],
        admin_id: Optional[int],
    ) -> Dict[str, any]:
        # Fetch lecture row from database
        with get_pg_cursor() as cur:
            cur.execute("SELECT * FROM lecture_gen WHERE lecture_uid = %(lecture_uid)s", {"lecture_uid": lecture_id})
            lecture_row = cur.fetchone()

        if not lecture_row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lecture not found")

        scope_admin_id = admin_id or lecture_row.get("admin_id")
        if not scope_admin_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to determine school context for lecture",
            )

        record = _clone_record(lecture_row.get("lecture_data") or {})
        record.setdefault("lecture_id", lecture_row.get("lecture_uid"))

        metadata = record.get("metadata")
        if metadata is None:
            metadata = {}
            record["metadata"] = metadata

        if lecture_row.get("admin_id") != scope_admin_id:
            lecture_row["admin_id"] = scope_admin_id

        if lecture_row.get("subject"):
            record.setdefault("subject", lecture_row["subject"])
            metadata.setdefault("subject", lecture_row["subject"])
        if lecture_row.get("std"):
            metadata.setdefault("std", lecture_row["std"])
        if lecture_row.get("sem"):
            metadata.setdefault("sem", lecture_row["sem"])
        if lecture_row.get("board"):
            metadata.setdefault("board", lecture_row["board"])

        if lecture_row.get("lecture_link") and not record.get("lecture_url"):
            record["lecture_url"] = lecture_row["lecture_link"]

        resolved_subject = self._resolve_subject(
            request_subject=None,
            row=lecture_row,
            record=record,
        )
        resolved_std = self._resolve_std(
            request_std=payload.std,
            row_std=lecture_row.get("std"),
            record=record,
        )

        classmates = student_portal_repository.list_students_for_class(
            admin_id=scope_admin_id,
            std=resolved_std,
            division=None,
        )
        if not classmates:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No students found for the provided class/division",
            )

        enrollments: List[str] = []
        for student in classmates or []:
            enrollment = student.get("enrollment_number")
            if not enrollment:
                continue
            enrollments.append(enrollment)

        if not enrollments:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No students found with valid enrollment numbers",
            )

            logger.info(
                "Lecture %s shared with class %s but no student enrollments were found; marking as shared without notifications",
                lecture_row.get("lecture_uid"),
                resolved_std,
            )

        lecture_url = self._build_lecture_url(
            record=record,
            row=lecture_row,
            resolved_std=resolved_std,
            resolved_subject=resolved_subject,
        )
        if lecture_url:
            prefix, sep, filename = lecture_url.rpartition("/")
            if sep:
                name, dot, ext = filename.partition('.')
                if dot:
                    lecture_url = f"{prefix}/{lecture_id}.{ext}"
                else:
                    lecture_url = f"{prefix}/{lecture_id}"

        metadata = record.get("metadata") or {}
        subject_display = (
            resolved_subject
            or metadata.get("subject")
            or metadata.get("lesson")
            or lecture_row.get("subject")
        )
        if not subject_display:
            subject_display = self._subject_from_url(lecture_url)
        if not subject_display:
            subject_display = self._extract_subject(record)
        subject_display = self._clean_subject(subject_display)

        resolved_subject = subject_display or resolved_subject or lecture_row.get("subject") or None

        # lecture_share_repository.record_shares(
        #     lecture_id=lecture_id,
        #     std=resolved_std,
        #     subject=resolved_subject or None,
        #     division=None,
        #     shared_by=shared_by,
        #     share_message=None,
        #     enrollments=enrollments,
        # )

        if enrollments:
            lecture_share_repository.record_shares(
                lecture_id=lecture_row.get("lecture_uid"),
                std=resolved_std,
                subject=resolved_subject or None,
                division=None,
                shared_by=shared_by,
                share_message=None,
                enrollments=enrollments,
            )

        # Update lecture_shared flag in database
        with get_pg_cursor() as cur:
            cur.execute(
                "UPDATE lecture_gen SET lecture_shared = TRUE WHERE lecture_uid = %(lecture_uid)s RETURNING *",
                {"lecture_uid": lecture_row.get("lecture_uid")}
            )
            updated_row = cur.fetchone()
        if updated_row:
            lecture_row = updated_row

        return {
            "lecture_id": lecture_row.get("lecture_uid"), 
            "lecture_url": lecture_url,
            "subject": subject_display or None,
            "lecture_shared": bool(lecture_row.get("lecture_shared")),
        }

    async def list_shared_lectures(
        self,
        *,
        admin_id: Optional[int],
        std: Optional[str] = None,
        subject: Optional[str] = None,
    ) -> List[Dict[str, any]]:
        if not admin_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to determine school context for lecture",
            )

        records = lecture_share_repository.list_shared_lectures(
            admin_id=admin_id,
            std=std,
            subject=subject,
        )

        return records

    async def delete_shared_lecture(
        self,
        *,
        lecture_id: str,
        admin_id: Optional[int],
    ) -> Dict[str, any]:
        if not admin_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to determine school context for lecture",
            )

        # Fetch lecture row from database
        with get_pg_cursor() as cur:
            cur.execute("SELECT * FROM lecture_gen WHERE lecture_uid = %(lecture_uid)s", {"lecture_uid": lecture_id})
            lecture_row = cur.fetchone()

        if not lecture_row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lecture not found")

        row_admin_id = lecture_row.get("admin_id")
        if row_admin_id and row_admin_id != admin_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Lecture does not belong to the current admin",
            )

        removed_records = lecture_share_repository.delete_shares_for_lecture(lecture_id=lecture_row.get("lecture_uid"))

        lecture_was_shared = bool(lecture_row.get("lecture_shared"))
        if lecture_was_shared or removed_records:
            # Update lecture_shared flag in database
            with get_pg_cursor() as cur:
                cur.execute(
                    "UPDATE lecture_gen SET lecture_shared = FALSE WHERE lecture_uid = %(lecture_uid)s RETURNING *",
                    {"lecture_uid": lecture_row.get("lecture_uid")}
                )
                updated_row = cur.fetchone()
            if updated_row:
                lecture_row = updated_row

        return {
            "lecture_id": lecture_row.get("lecture_uid"),
            "removed_records": removed_records,
            "lecture_shared": bool(lecture_row.get("lecture_shared")),
        }
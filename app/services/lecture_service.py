"""High-level service that orchestrates lecture generation and storage."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.config import get_settings
from app.repository.lecture_repository import LectureRepository
from app.services.lecture_generation_service import GroqService


class LectureService:
    """Provide a cohesive API for lecture CRUD and AI interactions."""

    def __init__(
        self,
        *,
        db: Session,
        groq_api_key: Optional[str] = None,
    ) -> None:
        settings = get_settings()

        inferred_api_key = (
            groq_api_key
            or getattr(settings, "groq_api_key", None)
            or settings.dict().get("GROQ_API_KEY")
        )

        self._repository = LectureRepository(db)
        self._generator = GroqService(api_key=inferred_api_key or "")

    @property
    def repository(self) -> LectureRepository:
        return self._repository

    @property
    def generator(self) -> GroqService:
        return self._generator

    async def create_lecture_from_text(
        self,
        *,
        text: str,
        language: str,
        duration: int,
        style: str,
        title: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self._generator.configured:
            raise RuntimeError("Groq service is not configured")

        lecture_payload = await self._generator.generate_lecture_content(
            text=text,
            language=language,
            duration=duration,
            style=style,
        )

        slides: List[Dict[str, Any]] = lecture_payload.get("slides", [])  # type: ignore[assignment]
        if not slides:
            raise RuntimeError("Lecture generation produced no slides")

        context = "\n\n".join(
            filter(None, (slide.get("narration", "") for slide in slides))
        )

        return await self._repository.create_lecture(
            title=title,
            language=language,
            style=style,
            duration=duration,
            slides=slides,
            context=context,
            text=text,
            metadata=metadata,
            fallback_used=lecture_payload.get("fallback_used", False),
        )

    async def answer_question(
        self,
        *,
        lecture_id: str,
        question: str,
        answer_type: Optional[str] = None,
        is_edit_command: bool = False,
        context_override: Optional[str] = None,
    ) -> Any:
        record = await self._repository.get_lecture(lecture_id)
        context = context_override or record.get("context", "")
        language = record.get("language", "English")

        return await self._generator.answer_question(
            question=question,
            context=context,
            language=language,
            answer_type=answer_type,
            is_edit_command=is_edit_command,
        )

    async def get_lecture(self, lecture_id: str) -> Dict[str, Any]:
        return await self._repository.get_lecture(lecture_id)

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
        return await self._repository.list_lectures(
            language=language,
            limit=limit,
            offset=offset,
            std=std,
            subject=subject,
            division=division,
        )

    async def delete_lecture(self, lecture_id: str) -> bool:
        return await self._repository.delete_lecture(lecture_id)

    async def get_class_subject_filters(self) -> Dict[str, Any]:
        return await self._repository.get_class_subject_filters()
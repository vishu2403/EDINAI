"""Utilities for generating lecture audio via Edge TTS."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import edge_tts

logger = logging.getLogger(__name__)


class EdgeTTSService:
    """Lightweight wrapper around the edge-tts client."""

    _CHUNK_CHAR_LIMIT = 4200

    def __init__(self, storage_root: str = "./storage/chapter_lectures") -> None:
        self._storage_root = Path(storage_root)
        self._storage_root.mkdir(parents=True, exist_ok=True)

    async def synthesize_text(
        self,
        *,
        lecture_id: str,
        text: str,
        language: str,
        filename: str,
        subfolder: str | None = None,
    ) -> Optional[Path]:
        """Generate an MP3 file for the provided text chunk."""
        normalized_text = (text or "").strip()
        if not normalized_text:
            logger.info(
                "Skipping TTS for lecture %s (%s) because text is empty.",
                lecture_id,
                filename,
            )
            return None

        target_path = self._build_audio_path(lecture_id, filename, subfolder=subfolder)
        voice = self._voice_for_language(language)

        try:
            with open(target_path, "wb") as audio_file:
                for chunk_text in self._chunk_text(normalized_text):
                    communicator = edge_tts.Communicate(chunk_text, voice=voice)
                    async for chunk in communicator.stream():
                        if chunk["type"] == "audio":
                            audio_file.write(chunk["data"])
            logger.info("Generated lecture audio at %s", target_path)
            return target_path
        except Exception as exc:  # pragma: no cover - network call
            logger.error(
                "Failed to synthesize audio for lecture %s (%s): %s",
                lecture_id,
                filename,
                exc,
            )
            try:
                if target_path.exists():
                    target_path.unlink(missing_ok=True)
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            return None

    def _build_audio_path(self, lecture_id: str, filename: str, *, subfolder: str | None) -> Path:
        lecture_dir = self._storage_root / str(lecture_id)
        if subfolder:
            lecture_dir = lecture_dir / subfolder
        lecture_dir.mkdir(parents=True, exist_ok=True)
        return lecture_dir / filename

    @staticmethod
    def _voice_for_language(language: str) -> str:
        mapping = {
            "English": "en-US-JennyNeural",
            "Hindi": "hi-IN-SwaraNeural",
            "Gujarati": "gu-IN-DhwaniNeural",
        }
        return mapping.get(language, "en-US-JennyNeural")

    def _chunk_text(self, text: str) -> Iterator[str]:
        if len(text) <= self._CHUNK_CHAR_LIMIT:
            yield text
            return

        start = 0
        while start < len(text):
            end = min(start + self._CHUNK_CHAR_LIMIT, len(text))
            yield text[start:end]
            start = end
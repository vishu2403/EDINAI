"""Utilities for generating lecture audio via Google Text-to-Speech (gTTS)."""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Iterator, Optional

from gtts import gTTS

logger = logging.getLogger(__name__)


class GTTSService:
    """Production-level wrapper around the gTTS client with retry logic and caching."""

    _CHUNK_CHAR_LIMIT = 100  # gTTS has lower character limits per request
    _MAX_RETRIES = 3
    _RETRY_DELAY = 1  # seconds
    _REQUEST_TIMEOUT = 30  # seconds

    def __init__(self, storage_root: str = "./storage/chapter_lectures") -> None:
        self._storage_root = Path(storage_root)
        self._storage_root.mkdir(parents=True, exist_ok=True)
        self._language_cache = self._build_language_cache()

    def _build_language_cache(self) -> dict[str, str]:
        """Build a cache of supported languages."""
        return {
            "en": "English",
            "hi": "Hindi",
            "gu": "Gujarati",
        }

    async def synthesize_text(
        self,
        *,
        lecture_id: str,
        text: str,
        language: str,
        filename: str,
        subfolder: str | None = None,
    ) -> Optional[Path]:
        """Generate an MP3 file for the provided text chunk with retry logic."""
        normalized_text = (text or "").strip()
        if not normalized_text:
            logger.info(
                "Skipping TTS for lecture %s (%s) because text is empty.",
                lecture_id,
                filename,
            )
            return None

        target_path = self._build_audio_path(lecture_id, filename, subfolder=subfolder)
        lang_code = self._language_to_code(language)

        # If file already exists, return it
        if target_path.exists():
            logger.info("Audio file already exists at %s", target_path)
            return target_path

        try:
            # Run gTTS synthesis in thread pool to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._synthesize_with_retry,
                normalized_text,
                lang_code,
                target_path,
            )
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

    def _synthesize_with_retry(self, text: str, lang_code: str, target_path: Path) -> None:
        """Synthesize text with retry logic for network resilience."""
        last_exception = None

        for attempt in range(self._MAX_RETRIES):
            try:
                self._synthesize_chunks(text, lang_code, target_path)
                logger.debug(f"Successfully synthesized audio on attempt {attempt + 1}")
                return
            except Exception as exc:
                last_exception = exc
                logger.warning(
                    f"Attempt {attempt + 1}/{self._MAX_RETRIES} failed for TTS synthesis: {exc}"
                )
                if attempt < self._MAX_RETRIES - 1:
                    time.sleep(self._RETRY_DELAY * (attempt + 1))  # Exponential backoff

        raise last_exception

    def _synthesize_chunks(self, text: str, lang_code: str, target_path: Path) -> None:
        """Synthesize text in chunks and write to file."""
        with open(target_path, "wb") as audio_file:
            for chunk_text in self._chunk_text(text):
                try:
                    tts = gTTS(text=chunk_text, lang=lang_code, slow=False, timeout=self._REQUEST_TIMEOUT)
                    tts.write_to_fp(audio_file)
                except Exception as exc:
                    logger.error(f"Failed to synthesize chunk: {exc}")
                    raise

    def _build_audio_path(self, lecture_id: str, filename: str, *, subfolder: str | None) -> Path:
        lecture_dir = self._storage_root / str(lecture_id)
        if subfolder:
            lecture_dir = lecture_dir / subfolder
        lecture_dir.mkdir(parents=True, exist_ok=True)
        return lecture_dir / filename

    def _language_to_code(self, language: str) -> str:
        """Convert language name to gTTS language code."""
        mapping = {
            "English": "en",
            "Hindi": "hi",
            "Gujarati": "gu",
        }
        lang_code = mapping.get(language, "en")
        logger.debug(f"Language '{language}' mapped to code '{lang_code}'")
        return lang_code

    def _chunk_text(self, text: str) -> Iterator[str]:
        """Split text into chunks respecting gTTS character limits."""
        if len(text) <= self._CHUNK_CHAR_LIMIT:
            yield text
            return

        # Split by sentences to maintain readability
        sentences = self._split_sentences(text)
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self._CHUNK_CHAR_LIMIT:
                current_chunk += sentence
            else:
                if current_chunk:
                    yield current_chunk
                current_chunk = sentence

        if current_chunk:
            yield current_chunk

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences intelligently."""
        import re

        # Split on common sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


# Backward compatibility alias
EdgeTTSService = GTTSService
"""Rule-based AI content analyzer fallback (legacy support)."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List

from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

logger = logging.getLogger(__name__)


@dataclass
class TopicSection:
    title: str
    content: str
    start_position: int
    end_position: int
    subtopics: List[str]
    summary: str


class AIContentAnalyzer:
    """Provides rule-based content analysis after removing AI topic generation."""

    def __init__(self) -> None:
        self.api_available = False
        self.client = None
        self.model = None
        self.initialization_error = "AI topic generation disabled; using rule-based fallback only."

    def analyze_and_split_content(self, content: str) -> List[TopicSection]:
        if not content:
            return []
        return self._fallback_topic_extraction(content)

    def _detect_language(self, content: str) -> str:
        sample = content[:4000] if content else ""
        if not sample.strip():
            return "unknown"
        try:
            return detect(sample)
        except (LangDetectException, Exception):
            return "unknown"

    def _extract_keywords(self, content: str) -> List[str]:
        words = re.findall(r"\b\w+\b", content.lower())
        stop_words = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
            "a",
            "an",
        }
        word_count: Dict[str, int] = {}
        for word in words:
            if len(word) > 3 and word not in stop_words:
                word_count[word] = word_count.get(word, 0) + 1
        sorted_words = sorted(word_count.items(), key=lambda item: item[1], reverse=True)
        return [word for word, _ in sorted_words[:5]]

    def _fallback_topic_extraction(self, content: str) -> List[TopicSection]:
        logger.info("Using fallback topic extraction method")
        try:
            sections: List[TopicSection] = []
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            if len(paragraphs) <= 1:
                paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
            if len(paragraphs) <= 2:
                sentences = [s.strip() for s in content.split(".") if s.strip()]
                if len(sentences) > 3:
                    paragraphs = [". ".join(sentences[i : i + 3]) + "." for i in range(0, len(sentences), 3)]
                else:
                    return [
                        TopicSection(
                            title="Complete Content",
                            content=content,
                            start_position=0,
                            end_position=len(content),
                            subtopics=self._extract_keywords(content),
                            summary=content[:200] + "..." if len(content) > 200 else content,
                        )
                    ]

            current_section: Optional[str] = None
            current_content: List[str] = []
            start_pos = 0
            for paragraph in paragraphs:
                is_header = (
                    len(paragraph) < 100
                    and len(paragraph.split()) <= 10
                    and (
                        paragraph.isupper()
                        or paragraph.istitle()
                        or (not paragraph.endswith(".") and len(paragraph.split()) <= 5)
                        or any(
                            keyword in paragraph.lower()
                            for keyword in ["chapter", "section", "part", "introduction", "conclusion", "summary"]
                        )
                    )
                )

                if is_header and current_section is not None:
                    section_content = "\n\n".join(current_content)
                    if section_content.strip():
                        sections.append(
                            TopicSection(
                                title=current_section,
                                content=section_content,
                                start_position=start_pos,
                                end_position=start_pos + len(section_content),
                                subtopics=self._extract_keywords(section_content),
                                summary=section_content[:150] + "..." if len(section_content) > 150 else section_content,
                            )
                        )
                    current_section = paragraph
                    current_content = []
                    start_pos += len(section_content) + 4
                elif is_header and current_section is None:
                    current_section = paragraph
                    current_content = []
                else:
                    if current_section is None:
                        current_section = "Introduction"
                    current_content.append(paragraph)

            if current_section and current_content:
                section_content = "\n\n".join(current_content)
                sections.append(
                    TopicSection(
                        title=current_section,
                        content=section_content,
                        start_position=start_pos,
                        end_position=start_pos + len(section_content),
                        subtopics=self._extract_keywords(section_content),
                        summary=section_content[:150] + "..." if len(section_content) > 150 else section_content,
                    )
                )

            if not sections:
                sections = [
                    TopicSection(
                        title="Complete Content",
                        content=content,
                        start_position=0,
                        end_position=len(content),
                        subtopics=self._extract_keywords(content),
                        summary=content[:200] + "..." if len(content) > 200 else content,
                    )
                ]
            return sections
        except Exception as exc:  # pragma: no cover - logging only
            logger.error("Error in fallback topic extraction: %s", exc)
            return [
                TopicSection(
                    title="Complete Content",
                    content=content,
                    start_position=0,
                    end_position=len(content),
                    subtopics=[],
                    summary="Topic extraction failed, showing complete content.",
                )
            ]


def analyze_pdf_content(content: str) -> Dict[str, object]:
    logger.info("Starting PDF content analysis. Content length: %s", len(content))
    if not content or not content.strip():
        return {
            "success": False,
            "error": "No content provided for analysis",
            "sections": [],
            "total_sections": 0,
            "total_word_count": 0,
            "total_char_count": 0,
        }

    analyzer = AIContentAnalyzer()
    detected_language = analyzer._detect_language(content)
    sections = analyzer.analyze_and_split_content(content)

    if not sections:
        return {
            "success": False,
            "detected_language": detected_language,
            "error": "No sections could be extracted from content",
            "sections": [
                {
                    "title": "Complete Content",
                    "content": content,
                    "summary": "Could not split content into sections.",
                    "subtopics": [],
                    "word_count": len(content.split()),
                    "char_count": len(content),
                }
            ],
            "total_sections": 1,
            "total_word_count": len(content.split()),
            "total_char_count": len(content),
        }

    return {
        "success": True,
        "detected_language": detected_language,
        "total_sections": len(sections),
        "sections": [
            {
                "title": section.title,
                "content": section.content,
                "summary": section.summary,
                "subtopics": section.subtopics,
                "word_count": len(section.content.split()),
                "char_count": len(section.content),
            }
            for section in sections
        ],
        "total_word_count": len(content.split()),
        "total_char_count": len(content),
    }

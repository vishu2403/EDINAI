"""System configuration and AI status routes."""
from __future__ import annotations

import logging
import os
import time
from typing import Dict

from fastapi import APIRouter, Depends

from ..schemas import ResponseBase
from ..utils.ai_service import AIContentAnalyzer, analyze_pdf_content
from ..utils.dependencies import admin_required

router = APIRouter(prefix="/system", tags=["System"])
logger = logging.getLogger(__name__)


@router.get("/ai-status", response_model=ResponseBase)
async def check_ai_status(current_user: dict = Depends(admin_required)) -> ResponseBase:
    env_status = {
        "groq_api_key_set": bool(os.getenv("GROQ_API_KEY")),
        "groq_model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    }

    analyzer = AIContentAnalyzer()
    ai_status = {
        "api_available": analyzer.api_available,
        "initialization_error": analyzer.initialization_error,
        "model": analyzer.model,
    }

    test_result: Dict[str, object] | None = None
    if analyzer.api_available:
        try:
            sample = "Mathematics covers numbers, shapes, and operations."
            sections = analyzer.analyze_and_split_content(sample)
            test_result = {"test_successful": True, "sections_found": len(sections)}
        except Exception as exc:  # pragma: no cover - debug only
            test_result = {"test_successful": False, "error": str(exc)}

    overall_status = env_status["groq_api_key_set"] and ai_status["api_available"]
    return ResponseBase(
        status=overall_status,
        message="AI service is operational" if overall_status else "AI service has issues",
        data={"environment": env_status, "ai_service": ai_status, "test_result": test_result},
    )


@router.get("/configuration-help", response_model=ResponseBase)
async def get_configuration_help(current_user: dict = Depends(admin_required)) -> ResponseBase:
    return ResponseBase(
        status=True,
        message="AI Service Configuration Help",
        data={
            "steps": [
                {
                    "step": 1,
                    "title": "Get GROQ API Key",
                    "description": "Sign up at https://console.groq.com/ and create an API key",
                },
                {
                    "step": 2,
                    "title": "Set Environment Variable",
                    "description": "Set GROQ_API_KEY via shell or .env",
                },
                {
                    "step": 3,
                    "title": "Restart Application",
                    "description": "Restart the backend to load new env vars",
                },
                {
                    "step": 4,
                    "title": "Test Configuration",
                    "description": "Call GET /system/ai-status",
                },
            ],
        },
    )


@router.post("/test-topic-extraction", response_model=ResponseBase)
async def test_topic_extraction(
    request_data: dict,
    current_user: dict = Depends(admin_required),
) -> ResponseBase:
    sample = request_data.get(
        "content",
        "Mathematics introduces basic operations like addition, subtraction, multiplication, and division.",
    )
    logger.info("Testing topic extraction with %s characters", len(sample))
    start_time = time.time()
    result = analyze_pdf_content(sample)
    processing_time = time.time() - start_time

    return ResponseBase(
        status=result.get("success", False),
        message="Topic extraction test completed" if result.get("success") else "Test failed",
        data={"processing_time_seconds": round(processing_time, 2), "analysis_result": result},
    )

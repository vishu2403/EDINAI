"""
Lecture Routes - FastAPI API Endpoints
Handles HTTP requests for lecture operations
"""
import os
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, status, Depends, Query
from fastapi.responses import JSONResponse

from app.schemas.lecture_schema import (
    CreateLectureRequest,
    AskQuestionRequest,
    LectureResponse,
    LectureSummaryResponse,
    AnswerResponse,
    ErrorResponse,
    GenerationStatus,
)
from app.repository.lecture_repository import LectureRepository
from app.services.lecture_generation_service import GroqService
load_dotenv()

# ============================================================================
# ROUTER SETUP
# ============================================================================

router = APIRouter(
    prefix="/api/lectures",
    tags=["lectures"],
    responses={
        404: {"model": ErrorResponse, "description": "Lecture not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    }
)


# ============================================================================
# DEPENDENCIES
# ============================================================================

# These should be configured based on your app settings
STORAGE_PATH = Path("./storage/lectures")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_repository() -> LectureRepository:
    """Dependency to get repository instance."""
    return LectureRepository(storage_path=STORAGE_PATH)

def get_groq_service() -> GroqService:
    """Dependency to get Groq service instance."""
    return GroqService(api_key=GROQ_API_KEY)


# ============================================================================
# LECTURE CRUD ENDPOINTS
# ============================================================================

@router.post(
    "/create",
    response_model=LectureResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new lecture from text",
    description="Generate a complete lecture with slides from source text using AI"
)
async def create_lecture(
    request: CreateLectureRequest,
    repository: LectureRepository = Depends(get_repository),
    groq_service: GroqService = Depends(get_groq_service),
) -> LectureResponse:
    """
    Create a new lecture from text content.
    
    - **text**: Source text content (minimum 50 characters)
    - **language**: English, Hindi, or Gujarati
    - **duration**: Requested duration in minutes (10-120)
    - **style**: Teaching style (default: storytelling)
    - **title**: Lecture title
    - **metadata**: Optional additional metadata
    """
    try:
        # Generate lecture content using AI
        lecture_data = await groq_service.generate_lecture_content(
            text=request.text,
            language=request.language,
            duration=request.duration,
            style=request.style,
        )
        
        slides = lecture_data.get("slides", [])
        if not slides:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate lecture slides"
            )
        
        # Build context from narrations
        context = "\n\n".join(
            slide.get("narration", "")
            for slide in slides
            if slide.get("narration")
        )
        
        # Save to repository
        record = await repository.create_lecture(
            title=request.title,
            language=request.language,
            style=request.style,
            duration=request.duration,
            slides=slides,
            context=context,
            text=request.text,
            metadata=request.metadata,
            fallback_used=lecture_data.get("fallback_used", False),
        )
        
        # Generate JSON file URL with class and subject
        lecture_id = record.get("lecture_id", "")
        metadata = request.metadata or {}
        
        # Get class and subject from metadata
        std = metadata.get("std") or metadata.get("class") or "general"
        subject = metadata.get("subject") or "lecture"
        
        # Create URL slug
        std_slug = std.replace(" ", "_").lower()
        subject_slug = subject.replace(" ", "_").lower()
        
        # JSON file URL format: /lectures/{class}/{subject}/{lecture_id}.json
        lecture_json_url = f"/lectures/{std_slug}/{subject_slug}/{lecture_id}.json"
        
        # Add URL to record
        record["lecture_url"] = lecture_json_url
        
        # Print to terminal
        print(f"\n{'='*60}")
        print(f"📚 LECTURE GENERATED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Lecture ID: {lecture_id}")
        print(f"Class: {std}")
        print(f"Subject: {subject}")
        print(f"Title: {request.title}")
        print(f"Language: {request.language}")
        print(f"JSON URL: {lecture_json_url}")
        print(f"Full Path: https://yourdomain.com{lecture_json_url}")
        print(f"Total Slides: {len(slides)}")
        print(f"{'='*60}\n")
        
        return LectureResponse(**record)
        
    except Exception as e:
        print(f"❌ Error creating lecture: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create lecture: {str(e)}"
        )


@router.get(
    "/{lecture_id}",
    response_model=LectureResponse,
    summary="Get lecture by ID",
    description="Retrieve complete lecture data including all slides"
)
async def get_lecture(
    lecture_id: str,
    repository: LectureRepository = Depends(get_repository),
) -> LectureResponse:
    """
    Retrieve a lecture by its unique ID.
    
    - **lecture_id**: Unique lecture identifier
    """
    try:
        record = await repository.get_lecture(lecture_id)
        return LectureResponse(**record)
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lecture {lecture_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving lecture: {str(e)}"
        )


@router.get(
    "/",
    response_model=List[LectureSummaryResponse],
    summary="List all lectures",
    description="Get a list of all lectures with optional filtering"
)
async def list_lectures(
    language: Optional[str] = Query(None, description="Filter by language"),
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    repository: LectureRepository = Depends(get_repository),
) -> List[LectureSummaryResponse]:
    """
    List all lectures with pagination and filtering.
    
    - **language**: Optional filter by language (English, Hindi, Gujarati)
    - **limit**: Maximum number of results (1-500)
    - **offset**: Number of results to skip for pagination
    """
    try:
        lectures = await repository.list_lectures(
            language=language,
            limit=limit,
            offset=offset,
        )
        return [LectureSummaryResponse(**lecture) for lecture in lectures]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing lectures: {str(e)}"
        )


@router.delete(
    "/{lecture_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a lecture",
    description="Permanently delete a lecture and all its associated files"
)
async def delete_lecture(
    lecture_id: str,
    repository: LectureRepository = Depends(get_repository),
) -> None:
    """
    Delete a lecture permanently.
    
    - **lecture_id**: Lecture to delete
    """
    try:
        deleted = await repository.delete_lecture(lecture_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Lecture {lecture_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting lecture: {str(e)}"
        )


# ============================================================================
# QUESTION & ANSWER ENDPOINTS
# ============================================================================

@router.post(
    "/ask",
    response_model=AnswerResponse,
    summary="Ask a question about a lecture",
    description="Get AI-powered answers to questions about lecture content"
)
async def ask_question(
    request: AskQuestionRequest,
    repository: LectureRepository = Depends(get_repository),
    groq_service: GroqService = Depends(get_groq_service),
) -> AnswerResponse:
    """
    Ask a question about a lecture or edit slide content.
    
    - **lecture_id**: Lecture to query
    - **question**: Question text or edit command
    - **answer_type**: Response format (text or json)
    - **is_edit_command**: Whether this is an edit command
    - **context_override**: Optional context override
    """
    try:
        # Get lecture for context
        record = await repository.get_lecture(request.lecture_id)
        
        context = request.context_override or record.get("context", "")
        language = record.get("language", "English")
        
        # Get answer from AI
        response = await groq_service.answer_question(
            question=request.question,
            context=context,
            language=language,
            answer_type=request.answer_type,
            is_edit_command=request.is_edit_command,
        )
        
        # Handle different response types
        if isinstance(response, dict):
            return AnswerResponse(**response)
        else:
            return AnswerResponse(answer=str(response))
            
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lecture {request.lecture_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing question: {str(e)}"
        )


# ============================================================================
# SLIDE MANAGEMENT ENDPOINTS
# ============================================================================

@router.get(
    "/{lecture_id}/slides/{slide_number}",
    summary="Get a specific slide",
    description="Retrieve details of a specific slide from a lecture"
)
async def get_slide(
    lecture_id: str,
    slide_number: int,
    repository: LectureRepository = Depends(get_repository),
) -> Dict[str, Any]:
    """
    Get a specific slide from a lecture.
    
    - **lecture_id**: Lecture containing the slide
    - **slide_number**: Slide number (1-indexed)
    """
    try:
        record = await repository.get_lecture(lecture_id)
        slides = record.get("slides", [])
        
        for slide in slides:
            if slide.get("number") == slide_number:
                return slide
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Slide {slide_number} not found in lecture {lecture_id}"
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lecture {lecture_id} not found"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving slide: {str(e)}"
        )


@router.patch(
    "/{lecture_id}/slides/{slide_number}",
    summary="Update a slide",
    description="Update specific fields of a slide"
)
async def update_slide(
    lecture_id: str,
    slide_number: int,
    updates: Dict[str, Any],
    repository: LectureRepository = Depends(get_repository),
) -> Dict[str, Any]:
    """
    Update a slide's content.
    
    - **lecture_id**: Lecture containing the slide
    - **slide_number**: Slide number to update
    - **updates**: Dictionary of fields to update
    """
    try:
        record = await repository.update_slide(
            lecture_id=lecture_id,
            slide_number=slide_number,
            slide_updates=updates,
        )
        
        # Return updated slide
        for slide in record.get("slides", []):
            if slide.get("number") == slide_number:
                return slide
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Slide {slide_number} not found"
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lecture {lecture_id} not found"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating slide: {str(e)}"
        )


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@router.get(
    "/stats/summary",
    summary="Get lecture statistics",
    description="Get overall statistics about stored lectures"
)
async def get_stats(
    repository: LectureRepository = Depends(get_repository),
) -> Dict[str, Any]:
    """Get statistics about all lectures."""
    try:
        stats = await repository.get_lecture_stats()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting stats: {str(e)}"
        )


@router.get(
    "/{lecture_id}/source",
    summary="Get source text",
    description="Retrieve the original source text used to generate the lecture"
)
async def get_source_text(
    lecture_id: str,
    repository: LectureRepository = Depends(get_repository),
) -> Dict[str, str]:
    """
    Get the original source text for a lecture.
    
    - **lecture_id**: Lecture ID
    """
    try:
        text = await repository.get_source_text(lecture_id)
        return {"lecture_id": lecture_id, "source_text": text}
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source text not found for lecture {lecture_id}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving source text: {str(e)}"
        )


@router.post(
    "/{lecture_id}/regenerate",
    response_model=LectureResponse,
    summary="Regenerate lecture content",
    description="Regenerate a lecture using the same source text but with new AI generation"
)
async def regenerate_lecture(
    lecture_id: str,
    language: Optional[str] = None,
    duration: Optional[int] = None,
    repository: LectureRepository = Depends(get_repository),
    groq_service: GroqService = Depends(get_groq_service),
) -> LectureResponse:
    """
    Regenerate a lecture with new content.
    
    - **lecture_id**: Lecture to regenerate
    - **language**: Optional new language (uses original if not provided)
    - **duration**: Optional new duration (uses original if not provided)
    """
    try:
        # Get original lecture
        original = await repository.get_lecture(lecture_id)
        source_text = await repository.get_source_text(lecture_id)
        
        # Use provided params or fall back to originals
        new_language = language or original.get("language", "English")
        new_duration = duration or original.get("requested_duration", 30)
        
        # Generate new content
        lecture_data = await groq_service.generate_lecture_content(
            text=source_text,
            language=new_language,
            duration=new_duration,
            style=original.get("style", "storytelling"),
        )
        
        slides = lecture_data.get("slides", [])
        context = "\n\n".join(
            slide.get("narration", "")
            for slide in slides
            if slide.get("narration")
        )
        
        # Update lecture
        updates = {
            "language": new_language,
            "requested_duration": new_duration,
            "estimated_duration": lecture_data.get("estimated_duration"),
            "slides": slides,
            "context": context,
            "total_slides": len(slides),
            "fallback_used": lecture_data.get("fallback_used", False),
        }
        
        record = await repository.update_lecture(lecture_id, updates)
        
        # Generate JSON URL again after regeneration
        metadata = original.get("metadata", {})
        std = metadata.get("std") or metadata.get("class") or "general"
        subject = metadata.get("subject") or "lecture"
        
        std_slug = std.replace(" ", "_").lower()
        subject_slug = subject.replace(" ", "_").lower()
        
        # JSON file URL format
        lecture_json_url = f"/lectures/{std_slug}/{subject_slug}/{lecture_id}.json"
        
        record["lecture_url"] = lecture_json_url
        
        # Print to terminal
        print(f"\n{'='*60}")
        print(f"🔄 LECTURE REGENERATED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Lecture ID: {lecture_id}")
        print(f"Class: {std}")
        print(f"Subject: {subject}")
        print(f"New Language: {new_language}")
        print(f"JSON URL: {lecture_json_url}")
        print(f"Full Path: https://yourdomain.com{lecture_json_url}")
        print(f"{'='*60}\n")
        
        return LectureResponse(**record)
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lecture {lecture_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error regenerating lecture: {str(e)}"
        )


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get(
    "/health",
    tags=["health"],
    summary="Health check",
    description="Check if the lecture service is running"
)
async def health_check(
    groq_service: GroqService = Depends(get_groq_service),
) -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "lecture_generation",
        "groq_configured": groq_service.configured,
    }
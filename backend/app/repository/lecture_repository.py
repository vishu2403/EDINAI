"""
Lecture Repository - Data Storage and Retrieval Layer
Handles all file I/O operations for lecture storage
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from uuid import uuid4


class LectureRepository:
    """Repository for managing lecture persistence."""
    
    def __init__(self, storage_path: Path) -> None:
        """
        Initialize repository with storage path.
        
        Args:
            storage_path: Base directory for storing lectures
        """
        self._storage_dir = storage_path
        self._storage_dir.mkdir(parents=True, exist_ok=True)
    
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
    ) -> Dict[str, Any]:
        """
        Create and persist a new lecture.
        
        Args:
            title: Lecture title
            language: Lecture language (English, Hindi, Gujarati)
            style: Teaching style
            duration: Requested duration in minutes
            slides: List of slide dictionaries
            context: Combined narration context
            text: Source text content
            metadata: Additional metadata
            fallback_used: Whether fallback content was used
            
        Returns:
            Complete lecture record with lecture_id
        """
        lecture_id = uuid4().hex[:12]
        lecture_dir = self._storage_dir / lecture_id
        lecture_dir.mkdir(parents=True, exist_ok=True)
        
        # Save source text
        source_text_path = lecture_dir / "source.txt"
        await asyncio.to_thread(
            lambda: source_text_path.write_text(text, encoding="utf-8")
        )
        
        # Build lecture record
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
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "fallback_used": fallback_used,
            "source_text": str(source_text_path),
            "metadata": metadata or {},
        }
        
        # Save lecture JSON
        json_path = lecture_dir / "lecture.json"
        await self._save_json(json_path, record)
        
        print(f"📝 Lecture saved: {lecture_id}")
        return record
    
    async def get_lecture(self, lecture_id: str) -> Dict[str, Any]:
        """
        Retrieve lecture by ID.
        
        Args:
            lecture_id: Unique lecture identifier
            
        Returns:
            Complete lecture record
            
        Raises:
            FileNotFoundError: If lecture doesn't exist
        """
        lecture_dir = self._storage_dir / lecture_id
        json_path = lecture_dir / "lecture.json"
        
        if not json_path.is_file():
            raise FileNotFoundError(f"Lecture {lecture_id} not found")
        
        return await self._load_json(json_path)
    
    async def update_lecture(
        self, 
        lecture_id: str, 
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update lecture data.
        
        Args:
            lecture_id: Lecture to update
            updates: Dictionary of fields to update
            
        Returns:
            Updated lecture record
        """
        record = await self.get_lecture(lecture_id)
        
        # Update fields
        record.update(updates)
        record["updated_at"] = datetime.utcnow().isoformat()
        
        # Save updated record
        lecture_dir = self._storage_dir / lecture_id
        json_path = lecture_dir / "lecture.json"
        await self._save_json(json_path, record)
        
        print(f"🔄 Lecture updated: {lecture_id}")
        return record
    
    async def update_slide(
        self,
        lecture_id: str,
        slide_number: int,
        slide_updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update a specific slide in a lecture.
        
        Args:
            lecture_id: Lecture containing the slide
            slide_number: Slide number to update (1-indexed)
            slide_updates: Fields to update in the slide
            
        Returns:
            Updated lecture record
        """
        record = await self.get_lecture(lecture_id)
        slides = record.get("slides", [])
        
        # Find and update slide
        for slide in slides:
            if slide.get("number") == slide_number:
                slide.update(slide_updates)
                break
        
        # Update context if narration changed
        if "narration" in slide_updates:
            context = "\n\n".join(
                slide.get("narration", "") 
                for slide in slides 
                if slide.get("narration")
            )
            record["context"] = context
        
        record["updated_at"] = datetime.utcnow().isoformat()
        
        # Save
        lecture_dir = self._storage_dir / lecture_id
        json_path = lecture_dir / "lecture.json"
        await self._save_json(json_path, record)
        
        print(f"🔄 Slide {slide_number} updated in lecture {lecture_id}")
        return record
    
    async def delete_lecture(self, lecture_id: str) -> bool:
        """
        Delete a lecture and all its files.
        
        Args:
            lecture_id: Lecture to delete
            
        Returns:
            True if deleted successfully
        """
        lecture_dir = self._storage_dir / lecture_id
        
        if not lecture_dir.exists():
            return False
        
        # Delete all files in directory
        def _delete_dir():
            import shutil
            shutil.rmtree(lecture_dir)
        
        await asyncio.to_thread(_delete_dir)
        print(f"🗑️ Lecture deleted: {lecture_id}")
        return True
    
    async def list_lectures(
        self,
        *,
        language: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List all lectures with optional filtering.
        
        Args:
            language: Filter by language
            limit: Maximum number of results
            offset: Skip this many results
            
        Returns:
            List of lecture summary records
        """
        all_lectures = []
        
        # Iterate through all lecture directories
        for lecture_dir in self._storage_dir.iterdir():
            if not lecture_dir.is_dir():
                continue
            
            json_path = lecture_dir / "lecture.json"
            if not json_path.is_file():
                continue
            
            try:
                record = await self._load_json(json_path)
                
                # Apply language filter
                if language and record.get("language") != language:
                    continue
                
                # Create summary
                summary = {
                    "lecture_id": record.get("lecture_id"),
                    "title": record.get("title"),
                    "language": record.get("language"),
                    "total_slides": record.get("total_slides"),
                    "estimated_duration": record.get("estimated_duration"),
                    "created_at": record.get("created_at"),
                    "fallback_used": record.get("fallback_used", False),
                }
                all_lectures.append(summary)
                
            except Exception as e:
                print(f"⚠️ Error loading lecture {lecture_dir.name}: {e}")
                continue
        
        # Sort by creation date (newest first)
        all_lectures.sort(
            key=lambda x: x.get("created_at", ""), 
            reverse=True
        )
        
        # Apply pagination
        return all_lectures[offset:offset + limit]
    
    async def lecture_exists(self, lecture_id: str) -> bool:
        """
        Check if lecture exists.
        
        Args:
            lecture_id: Lecture to check
            
        Returns:
            True if lecture exists
        """
        lecture_dir = self._storage_dir / lecture_id
        json_path = lecture_dir / "lecture.json"
        return json_path.is_file()
    
    async def get_lecture_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored lectures.
        
        Returns:
            Dictionary with stats (total count, by language, etc.)
        """
        stats = {
            "total_lectures": 0,
            "by_language": {},
            "fallback_count": 0,
            "total_slides": 0,
        }
        
        for lecture_dir in self._storage_dir.iterdir():
            if not lecture_dir.is_dir():
                continue
            
            json_path = lecture_dir / "lecture.json"
            if not json_path.is_file():
                continue
            
            try:
                record = await self._load_json(json_path)
                stats["total_lectures"] += 1
                
                language = record.get("language", "Unknown")
                stats["by_language"][language] = stats["by_language"].get(language, 0) + 1
                
                if record.get("fallback_used"):
                    stats["fallback_count"] += 1
                
                stats["total_slides"] += record.get("total_slides", 0)
                
            except Exception:
                continue
        
        return stats
    
    async def get_source_text(self, lecture_id: str) -> str:
        """
        Get original source text for a lecture.
        
        Args:
            lecture_id: Lecture ID
            
        Returns:
            Source text content
        """
        lecture_dir = self._storage_dir / lecture_id
        source_path = lecture_dir / "source.txt"
        
        if not source_path.is_file():
            raise FileNotFoundError(f"Source text not found for lecture {lecture_id}")
        
        return await asyncio.to_thread(
            lambda: source_path.read_text(encoding="utf-8")
        )
    
    async def _save_json(self, path: Path, data: Dict[str, Any]) -> None:
        """Save JSON data to file."""
        def _write():
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        
        await asyncio.to_thread(_write)
    
    async def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON data from file."""
        def _read():
            return json.loads(path.read_text(encoding="utf-8"))
        
        return await asyncio.to_thread(_read)
    
    def get_lecture_path(self, lecture_id: str) -> Path:
        """Get the directory path for a lecture."""
        return self._storage_dir / lecture_id
    
    async def backup_lecture(self, lecture_id: str, backup_dir: Path) -> Path:
        """
        Create a backup of a lecture.
        
        Args:
            lecture_id: Lecture to backup
            backup_dir: Directory to store backup
            
        Returns:
            Path to backup file
        """
        import shutil
        
        lecture_dir = self._storage_dir / lecture_id
        if not lecture_dir.exists():
            raise FileNotFoundError(f"Lecture {lecture_id} not found")
        
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{lecture_id}_{timestamp}"
        
        def _backup():
            shutil.copytree(lecture_dir, backup_path)
        
        await asyncio.to_thread(_backup)
        print(f"💾 Backup created: {backup_path}")
        return backup_path
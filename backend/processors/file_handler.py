"""
File Handler — Per-user file storage with text extraction.

Handles file uploads, text extraction (PDF, TXT, MD, images via OCR),
and search across stored files.

Storage: data/users/{profile_id}/files/
Max size: 10MB per file
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
SUPPORTED_TYPES = {
    '.txt': 'text',
    '.md': 'text',
    '.csv': 'text',
    '.json': 'text',
    '.pdf': 'pdf',
    '.png': 'image',
    '.jpg': 'image',
    '.jpeg': 'image',
    '.bmp': 'image',
    '.webp': 'image',
}


class FileHandler:
    """Per-user file storage with text extraction."""

    def __init__(self, config: dict, db=None):
        self.base_dir = Path(config.get("system", {}).get("data_dir", "data")) / "users"
        self.db = db

    def _user_dir(self, profile_id: int) -> Path:
        """Get or create user's file directory."""
        d = self.base_dir / str(profile_id) / "files"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_file(self, profile_id: int, filename: str, data: bytes, persona: str = '') -> Optional[Dict[str, Any]]:
        """Save an uploaded file and extract text content.

        Args:
            profile_id: User's profile ID
            filename: Original filename
            data: File bytes

        Returns:
            File metadata dict or None on failure
        """
        if not isinstance(data, (bytes, bytearray)):
            logger.error("save_file: data must be bytes")
            return None

        if len(data) > MAX_FILE_SIZE:
            logger.warning(f"File too large: {len(data)} bytes (max {MAX_FILE_SIZE})")
            return None

        if not filename or not filename.strip():
            logger.warning("save_file: empty filename")
            return None

        # Sanitize filename: strip directory components, then scrub chars
        basename = os.path.basename(filename)
        safe_name = re.sub(r'[^\w\-.]', '_', basename)
        safe_name = safe_name.lstrip('.')  # prevent hidden files like .env
        if not safe_name:
            safe_name = 'upload'
        suffix = Path(safe_name).suffix.lower()

        if suffix not in SUPPORTED_TYPES:
            logger.warning(f"Unsupported file type: {suffix}")
            return None

        file_type = SUPPORTED_TYPES[suffix]
        user_dir = self._user_dir(profile_id)

        # Avoid overwrites
        target = user_dir / safe_name
        counter = 1
        while target.exists():
            stem = Path(safe_name).stem
            target = user_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        # Save file
        target.write_bytes(data)
        logger.info(f"Saved file: {target} ({len(data)} bytes)")

        # Extract text
        content_text = self._extract_text(target, file_type, data)

        # Store metadata in DB
        file_meta = {
            'profile_id': profile_id,
            'filename': target.name,
            'file_type': file_type,
            'content_text': content_text or '',
            'uploaded_at': datetime.now().isoformat(),
            'file_path': str(target),
            'size_bytes': len(data),
        }

        file_meta['persona'] = persona

        if self.db:
            try:
                cursor = self.db.conn.cursor()
                cursor.execute(
                    """INSERT INTO user_files
                       (profile_id, filename, file_type, content_text, uploaded_at, file_path, persona)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (profile_id, target.name, file_type,
                     content_text or '', file_meta['uploaded_at'], str(target), persona)
                )
                self.db._commit()
                file_meta['id'] = cursor.lastrowid
            except Exception as e:
                logger.error(f"Failed to save file metadata: {e}")

        return file_meta

    def _extract_text(self, path: Path, file_type: str, data: bytes = None) -> Optional[str]:
        """Extract text content from a file."""
        try:
            if file_type == 'text':
                return path.read_text(encoding='utf-8', errors='replace')[:50000]

            elif file_type == 'pdf':
                return self._extract_pdf_text(path)

            elif file_type == 'image':
                # Use OCR processor if available
                try:
                    from processors.ocr import OCRProcessor
                    ocr = OCRProcessor({})
                    if ocr.is_available():
                        return ocr.extract_text(str(path))
                except ImportError:
                    pass
                logger.info("OCR not available for image text extraction")
                return None

        except Exception as e:
            logger.error(f"Text extraction failed for {path}: {e}")
        return None

    def _extract_pdf_text(self, path: Path) -> Optional[str]:
        """Extract text from PDF using pdftotext or fallback."""
        try:
            import subprocess
            result = subprocess.run(
                ['pdftotext', '-layout', str(path), '-'],
                capture_output=True, timeout=30
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout.decode('utf-8', errors='replace')[:50000]
        except FileNotFoundError:
            logger.debug("pdftotext not installed, trying PyMuPDF fallback")
        except subprocess.TimeoutExpired:
            logger.warning(f"pdftotext timed out for {path}")
        except Exception as e:
            logger.warning(f"pdftotext failed for {path}: {e}")

        # Fallback: try PyMuPDF if available
        try:
            import fitz
            doc = fitz.open(str(path))
            text = ""
            for page in doc:
                text += page.get_text()
                if len(text) > 50000:
                    break
            doc.close()
            return text[:50000] if text else None
        except ImportError:
            pass

        logger.warning(f"No PDF reader available for {path}")
        return None

    def list_files(self, profile_id: int, persona: str = '') -> List[Dict[str, Any]]:
        """List files for a user, optionally filtered by persona."""
        if not self.db:
            return []
        try:
            cursor = self.db.conn.cursor()
            if persona:
                cursor.execute(
                    """SELECT id, filename, file_type, uploaded_at, persona,
                              LENGTH(content_text) as text_length
                       FROM user_files WHERE profile_id = ? AND persona = ?
                       ORDER BY uploaded_at DESC""",
                    (profile_id, persona)
                )
            else:
                cursor.execute(
                    """SELECT id, filename, file_type, uploaded_at, persona,
                              LENGTH(content_text) as text_length
                       FROM user_files WHERE profile_id = ?
                       ORDER BY uploaded_at DESC""",
                    (profile_id,)
                )
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []

    def search_files(self, profile_id: int, query: str, limit: int = 10, persona: str = '') -> List[Dict[str, Any]]:
        """Search across user's uploaded files, optionally filtered by persona."""
        if not self.db:
            return []
        try:
            cursor = self.db.conn.cursor()
            if persona:
                cursor.execute(
                    """SELECT id, filename, file_type, uploaded_at, persona,
                              SUBSTR(content_text, MAX(1, INSTR(LOWER(content_text), LOWER(?)) - 100), 300) as snippet
                       FROM user_files
                       WHERE profile_id = ? AND persona = ? AND content_text LIKE ?
                       ORDER BY uploaded_at DESC LIMIT ?""",
                    (query, profile_id, persona, f"%{query}%", limit)
                )
            else:
                cursor.execute(
                    """SELECT id, filename, file_type, uploaded_at, persona,
                              SUBSTR(content_text, MAX(1, INSTR(LOWER(content_text), LOWER(?)) - 100), 300) as snippet
                       FROM user_files
                       WHERE profile_id = ? AND content_text LIKE ?
                       ORDER BY uploaded_at DESC LIMIT ?""",
                    (query, profile_id, f"%{query}%", limit)
                )
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"File search failed: {e}")
            return []

    def get_file_content(self, profile_id: int, file_id: int) -> Optional[str]:
        """Get the extracted text content of a file."""
        if not self.db:
            return None
        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                "SELECT content_text FROM user_files WHERE id = ? AND profile_id = ?",
                (file_id, profile_id)
            )
            row = cursor.fetchone()
            return row['content_text'] if row else None
        except Exception as e:
            logger.error(f"Failed to get file content: {e}")
            return None

    def delete_file(self, profile_id: int, file_id: int) -> bool:
        """Delete a user's file."""
        if not self.db:
            return False
        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                "SELECT file_path FROM user_files WHERE id = ? AND profile_id = ?",
                (file_id, profile_id)
            )
            row = cursor.fetchone()
            if not row:
                return False

            # Delete from filesystem — verify path is within user directory
            path = Path(row['file_path']).resolve()
            user_dir = self._user_dir(profile_id).resolve()
            if not str(path).startswith(str(user_dir)):
                logger.error(f"Path traversal blocked on delete: {path} outside {user_dir}")
                return False
            if path.exists():
                path.unlink()

            # Delete from DB
            cursor.execute("DELETE FROM user_files WHERE id = ? AND profile_id = ?",
                         (file_id, profile_id))
            self.db._commit()
            return True
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            return False

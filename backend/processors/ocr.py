"""
OCR Pipeline — PaddleOCR-VL via Ollama

Extracts text from images and PDFs using PaddleOCR-VL 0.9B model.
~1GB VRAM, loaded on demand.

Usage:
    ocr = OCRProcessor(config)
    text = ocr.extract_text('/path/to/image.png')
"""

import logging
import base64
import requests
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

OCR_MODEL = "MedAIBase/PaddleOCR-VL:0.9b"


class OCRProcessor:
    """Extract text from images using PaddleOCR-VL via Ollama."""

    def __init__(self, config: dict):
        self.host = config.get("scoring", {}).get("ollama_host", "http://localhost:11434")
        self._session = requests.Session()

    MAX_FILE_SIZE = 50_000_000  # 50MB

    def is_available(self) -> bool:
        """Check if the OCR model is loaded in Ollama."""
        try:
            r = self._session.get(f"{self.host}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m.get("name", "") for m in r.json().get("models", [])]
                return any(OCR_MODEL.split(":")[0].lower() in m.lower() for m in models)
        except Exception as e:
            logger.warning(f"OCR availability check failed: {e}")
        return False

    def extract_text(self, file_path: str) -> Optional[str]:
        """Extract text from an image file.

        Args:
            file_path: Path to image file (PNG, JPG, JPEG, BMP, WEBP)

        Returns:
            Extracted text or None on failure
        """
        path = Path(file_path).resolve()

        # Path traversal guard: reject paths outside expected directories
        if '..' in str(path):
            logger.error(f"OCR: path traversal rejected: {file_path}")
            return None

        if not path.exists():
            logger.error(f"OCR: file not found: {file_path}")
            return None

        suffix = path.suffix.lower()
        if suffix not in ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif'):
            logger.warning(f"OCR: unsupported format {suffix}")
            return None

        try:
            # Guard against oversized files (50MB limit)
            if path.stat().st_size > self.MAX_FILE_SIZE:
                logger.warning(f"OCR: file too large ({path.stat().st_size} bytes)")
                return None

            # Read and base64 encode the image
            img_data = path.read_bytes()
            b64_image = base64.b64encode(img_data).decode('utf-8')

            # Call Ollama with image
            resp = self._session.post(
                f"{self.host}/api/chat",
                json={
                    "model": OCR_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Extract all text from this image. Return only the extracted text, nothing else.",
                            "images": [b64_image],
                        }
                    ],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 2000},
                },
                timeout=60,
            )

            if resp.status_code != 200:
                logger.error(f"OCR: Ollama returned {resp.status_code}")
                return None

            text = resp.json().get("message", {}).get("content", "").strip()
            if text:
                logger.info(f"OCR: extracted {len(text)} chars from {path.name}")
                return text
            return None

        except requests.RequestException as e:
            logger.error(f"OCR request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return None

    def extract_text_from_bytes(self, data: bytes, filename: str = "image") -> Optional[str]:
        """Extract text from raw image bytes."""
        if not data:
            logger.warning("OCR: empty data provided")
            return None
        if len(data) > self.MAX_FILE_SIZE:
            logger.warning(f"OCR: data too large ({len(data)} bytes, max {self.MAX_FILE_SIZE})")
            return None

        b64_image = base64.b64encode(data).decode('utf-8')
        try:
            resp = self._session.post(
                f"{self.host}/api/chat",
                json={
                    "model": OCR_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Extract all text from this image. Return only the extracted text.",
                            "images": [b64_image],
                        }
                    ],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 2000},
                },
                timeout=60,
            )
            if resp.status_code == 200:
                text = resp.json().get("message", {}).get("content", "").strip()
                if text:
                    logger.info(f"OCR: extracted {len(text)} chars from {filename}")
                    return text
                return None
            else:
                logger.error(f"OCR: Ollama returned {resp.status_code} for {filename}")
                return None
        except requests.RequestException as e:
            logger.error(f"OCR request failed for {filename}: {e}")
            return None
        except Exception as e:
            logger.error(f"OCR error for {filename}: {e}")
        return None

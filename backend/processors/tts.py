"""
TTS Pipeline — Piper TTS (CPU-only, zero VRAM)

Text-to-speech via Piper for agent response audio.

Usage:
    tts = TTSProcessor()
    audio_bytes = tts.synthesize("Hello world")
"""

import logging
import subprocess
import shutil
from typing import Optional

logger = logging.getLogger(__name__)

# Check if piper is available
HAS_PIPER = shutil.which("piper") is not None


class TTSProcessor:
    """Text-to-speech via Piper CLI."""

    def __init__(self, voice: str = "en_US-lessac-medium"):
        self.voice = voice
        self._available = HAS_PIPER

    def is_available(self) -> bool:
        """Check if Piper TTS is installed and working."""
        return self._available

    def synthesize(self, text: str, output_path: str = None) -> Optional[bytes]:
        """Convert text to speech.

        Args:
            text: Text to synthesize
            output_path: Optional path to save WAV file. If None, returns bytes.

        Returns:
            WAV audio bytes, or None on failure
        """
        if not self._available:
            logger.warning("TTS: Piper not installed")
            return None

        if not text or len(text) > 5000:
            logger.warning(f"TTS: text too {'long' if text else 'short'} ({len(text) if text else 0} chars)")
            return None

        try:
            cmd = [
                "piper",
                "--model", self.voice,
                "--output-raw",
            ]

            result = subprocess.run(
                cmd,
                input=text.encode('utf-8'),
                capture_output=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.error(f"TTS: piper failed: {result.stderr.decode()[:200]}")
                return None

            audio_data = result.stdout
            if not audio_data:
                logger.warning("TTS: piper returned empty output")
                return None

            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(audio_data)
                logger.info(f"TTS: saved {len(audio_data)} bytes to {output_path}")

            return audio_data

        except subprocess.TimeoutExpired:
            logger.error("TTS: piper timed out")
            return None
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

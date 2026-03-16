"""Speech-to-Text processor using faster-whisper (CPU-only)."""

import os
import time
import tempfile
import logging
import threading

logger = logging.getLogger(__name__)

ALLOWED_AUDIO_MIME_PREFIXES = ('audio/', 'video/webm', 'application/ogg', 'application/octet-stream')


class STTProcessor:
    """Singleton processor — model loads once, reuses for all requests."""

    _model = None
    _model_lock = threading.Lock()

    MODEL_NAME = "large-v3-turbo"
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"

    MAX_AUDIO_SECONDS = 120
    MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10MB

    @classmethod
    def _ensure_model(cls):
        """Lazy-load the whisper model on first use (thread-safe)."""
        if cls._model is not None:
            return cls._model

        with cls._model_lock:
            # Double-check after acquiring lock
            if cls._model is not None:
                return cls._model

            try:
                logger.info(f"Loading faster-whisper model: {cls.MODEL_NAME} (first use)")
                from faster_whisper import WhisperModel
                cls._model = WhisperModel(
                    cls.MODEL_NAME,
                    device=cls.DEVICE,
                    compute_type=cls.COMPUTE_TYPE
                )
                logger.info("faster-whisper model loaded successfully")
                return cls._model
            except Exception as e:
                logger.error(f"Failed to load whisper model: {e}")
                raise RuntimeError(f"Failed to load whisper model: {e}")

    @classmethod
    def transcribe(cls, audio_bytes, language_hint=None):
        """Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio data (WAV, WebM, OGG format)
            language_hint: Optional ISO language code ('en', 'ar', 'ja').
                           If None, auto-detects language.

        Returns:
            dict with text, language, language_probability, duration_seconds,
            processing_seconds
        """
        if not isinstance(audio_bytes, (bytes, bytearray)):
            raise ValueError("audio_bytes must be bytes")

        if len(audio_bytes) > cls.MAX_AUDIO_BYTES:
            raise ValueError(f"Audio too large: {len(audio_bytes)} bytes (max {cls.MAX_AUDIO_BYTES})")

        if len(audio_bytes) < 100:
            raise ValueError("Audio too short or empty")

        model = cls._ensure_model()

        # Save to temp file — faster-whisper reads from file path.
        # faster-whisper uses PyAV internally, so it can read WebM/OGG/WAV directly.
        suffix = '.webm'
        tmp_path = None
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            start_time = time.time()

            transcribe_kwargs = {
                "beam_size": 5,
                "best_of": 5,
                "vad_filter": True,
                "vad_parameters": {"min_silence_duration_ms": 500},
            }

            if language_hint:
                transcribe_kwargs["language"] = language_hint

            segments, info = model.transcribe(tmp_path, **transcribe_kwargs)

            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())

            full_text = ' '.join(text_parts).strip()
            processing_time = time.time() - start_time

            return {
                "text": full_text,
                "language": info.language,
                "language_probability": round(info.language_probability, 3),
                "duration_seconds": round(info.duration, 1),
                "processing_seconds": round(processing_time, 1),
            }

        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError as e:
                    logger.warning(f"STT: failed to clean up temp file {tmp_path}: {e}")

    @classmethod
    def is_available(cls):
        """Check if STT is available (faster-whisper installed)."""
        try:
            from faster_whisper import WhisperModel  # noqa: F401
        except ImportError:
            return False, "faster-whisper not installed"
        return True, "OK"

    @classmethod
    def preload(cls):
        """Optionally call at server startup to pre-load the model (~1.6GB RAM)."""
        try:
            cls._ensure_model()
            return True
        except Exception as e:
            logger.error(f"STT preload failed: {e}")
            return False

"""Dual-engine TTS: Kokoro-82M (multi-language) + XTTS-v2 (Arabic).

Both engines run on GPU by default, fall back to CPU if GPU unavailable.
Piper kept as last-resort fallback if neither engine loads.

VRAM budget:
  Kokoro: ~500MB on GPU
  XTTS-v2: ~1.8GB on GPU
  Total: ~2.3GB (fits comfortably alongside Ollama's ~9.5GB on 24GB VRAM)
"""

import os
import io
import time
import tempfile
import logging
import subprocess
import re
import shutil
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# VOICE REGISTRY
# ═══════════════════════════════════════════════════════════

KOKORO_VOICES = {
    # American Female
    "af_heart": {"name": "Heart", "gender": "female", "accent": "american", "lang": "en"},
    "af_alloy": {"name": "Alloy", "gender": "female", "accent": "american", "lang": "en"},
    "af_aoede": {"name": "Aoede", "gender": "female", "accent": "american", "lang": "en"},
    "af_bella": {"name": "Bella", "gender": "female", "accent": "american", "lang": "en"},
    "af_jessica": {"name": "Jessica", "gender": "female", "accent": "american", "lang": "en"},
    "af_nicole": {"name": "Nicole", "gender": "female", "accent": "american", "lang": "en"},
    "af_nova": {"name": "Nova", "gender": "female", "accent": "american", "lang": "en"},
    "af_river": {"name": "River", "gender": "female", "accent": "american", "lang": "en"},
    "af_sarah": {"name": "Sarah", "gender": "female", "accent": "american", "lang": "en"},
    "af_sky": {"name": "Sky", "gender": "female", "accent": "american", "lang": "en"},
    # American Male
    "am_adam": {"name": "Adam", "gender": "male", "accent": "american", "lang": "en"},
    "am_echo": {"name": "Echo", "gender": "male", "accent": "american", "lang": "en"},
    "am_eric": {"name": "Eric", "gender": "male", "accent": "american", "lang": "en"},
    "am_fenrir": {"name": "Fenrir", "gender": "male", "accent": "american", "lang": "en"},
    "am_liam": {"name": "Liam", "gender": "male", "accent": "american", "lang": "en"},
    "am_michael": {"name": "Michael", "gender": "male", "accent": "american", "lang": "en"},
    "am_onyx": {"name": "Onyx", "gender": "male", "accent": "american", "lang": "en"},
    "am_puck": {"name": "Puck", "gender": "male", "accent": "american", "lang": "en"},
    # British Female
    "bf_alice": {"name": "Alice", "gender": "female", "accent": "british", "lang": "en"},
    "bf_emma": {"name": "Emma", "gender": "female", "accent": "british", "lang": "en"},
    "bf_isabella": {"name": "Isabella", "gender": "female", "accent": "british", "lang": "en"},
    "bf_lily": {"name": "Lily", "gender": "female", "accent": "british", "lang": "en"},
    # British Male
    "bm_daniel": {"name": "Daniel", "gender": "male", "accent": "british", "lang": "en"},
    "bm_fable": {"name": "Fable", "gender": "male", "accent": "british", "lang": "en"},
    "bm_george": {"name": "George", "gender": "male", "accent": "british", "lang": "en"},
    "bm_lewis": {"name": "Lewis", "gender": "male", "accent": "british", "lang": "en"},
    # Japanese
    "jf_alpha": {"name": "Alpha", "gender": "female", "accent": "japanese", "lang": "ja"},
    "jf_gongitsune": {"name": "Gongitsune", "gender": "female", "accent": "japanese", "lang": "ja"},
    "jm_kumo": {"name": "Kumo", "gender": "male", "accent": "japanese", "lang": "ja"},
    # Chinese
    "zf_xiaobei": {"name": "Xiaobei", "gender": "female", "accent": "chinese", "lang": "zh"},
    "zf_xiaoni": {"name": "Xiaoni", "gender": "female", "accent": "chinese", "lang": "zh"},
    "zm_yunjian": {"name": "Yunjian", "gender": "male", "accent": "chinese", "lang": "zh"},
    # French
    "ff_siwis": {"name": "Siwis", "gender": "female", "accent": "french", "lang": "fr"},
    # Korean
    "kf_sarah": {"name": "Sarah", "gender": "female", "accent": "korean", "lang": "ko"},
    # Hindi
    "hf_alpha": {"name": "Alpha", "gender": "female", "accent": "hindi", "lang": "hi"},
    "hm_omega": {"name": "Omega", "gender": "male", "accent": "hindi", "lang": "hi"},
    # Italian
    "if_sara": {"name": "Sara", "gender": "female", "accent": "italian", "lang": "it"},
    "im_nicola": {"name": "Nicola", "gender": "male", "accent": "italian", "lang": "it"},
    # Portuguese (Brazilian)
    "pf_dora": {"name": "Dora", "gender": "female", "accent": "portuguese", "lang": "pt"},
    "pm_alex": {"name": "Alex", "gender": "male", "accent": "portuguese", "lang": "pt"},
}

XTTS_VOICES = {
    "ar_male": {"name": "Arabic Male", "gender": "male", "accent": "arabic", "lang": "ar"},
    "ar_female": {"name": "Arabic Female", "gender": "female", "accent": "arabic", "lang": "ar"},
}

# Language code → Kokoro lang_code mapping
KOKORO_LANG_CODES = {
    "en": "a",  # American English (default)
    "en-us": "a",
    "en-gb": "b",  # British English
    "ja": "j",
    "zh": "z",
    "fr": "f",
    "ko": "k",
    "hi": "h",
    "it": "i",
    "pt": "p",
}

# Languages supported by each engine
KOKORO_LANGUAGES = {"en", "ja", "zh", "fr", "ko", "hi", "it", "pt"}
XTTS_LANGUAGES = {"ar"}

# Default voices per persona (overridable via config.yaml tts.persona_voices)
_DEFAULT_PERSONA_VOICES = {
    "intelligence": "af_heart",
    "market": "am_michael",
    "scholarly": "ar_male",
    "gaming": "am_fenrir",
    "anime": "jf_alpha",
    "tcg": "am_echo",
}

# Mutable — updated by load_persona_voices_from_config()
PERSONA_DEFAULT_VOICES = dict(_DEFAULT_PERSONA_VOICES)


def load_persona_voices_from_config(config: dict):
    """Override persona default voices from config.yaml tts.persona_voices section."""
    tts_cfg = config.get("tts", {})
    overrides = tts_cfg.get("persona_voices", {})
    for persona, voice_id in overrides.items():
        if voice_id in KOKORO_VOICES or voice_id in XTTS_VOICES:
            PERSONA_DEFAULT_VOICES[persona] = voice_id
            logger.info(f"TTS persona voice override: {persona} → {voice_id}")


# ═══════════════════════════════════════════════════════════
# SYNTHESIS TIMEOUT HELPER
# ═══════════════════════════════════════════════════════════

def _run_with_timeout(fn, timeout_secs):
    """Run fn() in a thread with a timeout. Returns result or raises TimeoutError."""
    result = [None]
    error = [None]

    def _wrapper():
        try:
            result[0] = fn()
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_wrapper, daemon=True)
    t.start()
    t.join(timeout=timeout_secs)

    if t.is_alive():
        raise TimeoutError(f"TTS synthesis timed out after {timeout_secs}s")
    if error[0]:
        raise error[0]
    return result[0]


# ═══════════════════════════════════════════════════════════
# ENGINE CLASSES
# ═══════════════════════════════════════════════════════════

class KokoroEngine:
    """Kokoro-82M TTS engine — 54 voices, 8 languages, GPU-accelerated."""

    MAX_PIPELINES = 3  # LRU cap — evict oldest if exceeded
    _pipelines = OrderedDict()  # lang_code → KPipeline (LRU order)
    TIMEOUT = 15  # seconds

    @classmethod
    def _get_pipeline(cls, lang_code='a'):
        """Get or create a Kokoro pipeline for the given language (LRU cached, max 3)."""
        if lang_code in cls._pipelines:
            cls._pipelines.move_to_end(lang_code)
            return cls._pipelines[lang_code]

        try:
            from kokoro import KPipeline
            logger.info(f"Loading Kokoro pipeline for lang_code='{lang_code}' (first use — downloads ~300MB)")

            # Evict LRU pipeline if at capacity
            if len(cls._pipelines) >= cls.MAX_PIPELINES:
                evicted_key, _ = cls._pipelines.popitem(last=False)
                logger.info(f"Evicted Kokoro pipeline for lang_code='{evicted_key}' (LRU, max {cls.MAX_PIPELINES})")

            pipeline = KPipeline(lang_code=lang_code)
            cls._pipelines[lang_code] = pipeline
            logger.info(f"Kokoro pipeline loaded for lang_code='{lang_code}' ({len(cls._pipelines)}/{cls.MAX_PIPELINES} active)")
            return pipeline
        except Exception as e:
            logger.error(f"Failed to load Kokoro pipeline: {e}")
            return None

    @classmethod
    def synthesize(cls, text, voice='af_heart', speed=1.0):
        """Generate speech from text with timeout.

        Returns: (audio_bytes, sample_rate) or (None, None) on failure
        """
        voice_info = KOKORO_VOICES.get(voice, {})
        lang = voice_info.get('lang', 'en')

        if voice.startswith('b'):
            lang_code = 'b'
        else:
            lang_code = KOKORO_LANG_CODES.get(lang, 'a')

        pipeline = cls._get_pipeline(lang_code)
        if not pipeline:
            return None, None

        def _do_synthesis():
            import soundfile as sf
            import numpy as np

            start = time.time()
            audio_segments = []
            generator = pipeline(text, voice=voice, speed=speed)
            for i, (gs, ps, audio) in enumerate(generator):
                audio_segments.append(audio)

            if not audio_segments:
                return None, None

            full_audio = np.concatenate(audio_segments)
            buf = io.BytesIO()
            sf.write(buf, full_audio, 24000, format='WAV')
            buf.seek(0)

            elapsed = time.time() - start
            logger.info(f"Kokoro synthesized {len(text)} chars → {len(full_audio)/24000:.1f}s audio in {elapsed:.2f}s (voice={voice})")
            return buf.read(), 24000

        try:
            return _run_with_timeout(_do_synthesis, cls.TIMEOUT)
        except TimeoutError:
            logger.error(f"Kokoro synthesis timed out after {cls.TIMEOUT}s for {len(text)} chars")
            return None, None
        except Exception as e:
            logger.error(f"Kokoro synthesis failed: {e}", exc_info=True)
            return None, None

    @classmethod
    def is_available(cls):
        try:
            from kokoro import KPipeline
            return True, "Kokoro-82M available"
        except ImportError:
            return False, "Kokoro not installed (pip install kokoro>=0.9.2)"


class XTTSEngine:
    """XTTS-v2 TTS engine — Arabic language support, GPU-accelerated."""

    _model = None
    _loading = False
    _gpu_failed = False  # Track GPU→CPU fallback
    TIMEOUT = 30  # seconds

    @classmethod
    def _ensure_model(cls):
        """Lazy-load the XTTS-v2 model."""
        if cls._model is not None:
            return cls._model

        if cls._loading:
            for _ in range(120):
                time.sleep(1)
                if cls._model is not None:
                    return cls._model
            raise RuntimeError("XTTS model loading timed out")

        cls._loading = True
        try:
            from TTS.api import TTS
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading XTTS-v2 on {device} (first use — downloads ~1.8GB)")

            try:
                cls._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            except Exception as gpu_err:
                if device == "cuda":
                    logger.warning(f"XTTS-v2 GPU inference failed — falling back to CPU. Arabic TTS will be slower (~1-2s). Error: {gpu_err}")
                    cls._gpu_failed = True
                    cls._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
                else:
                    raise

            logger.info(f"XTTS-v2 loaded on {'CPU (GPU failed)' if cls._gpu_failed else device}")
            cls._loading = False
            return cls._model
        except Exception as e:
            cls._loading = False
            logger.error(f"Failed to load XTTS-v2: {e}")
            raise

    @classmethod
    def _get_speaker_wav(cls, voice_id):
        """Get the reference audio file for a voice."""
        voices_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'tts_voices')
        os.makedirs(voices_dir, exist_ok=True)

        voice_path = os.path.join(voices_dir, f"{voice_id}.wav")

        if os.path.exists(voice_path):
            return voice_path

        try:
            pitch = '70' if 'female' in voice_id else '40'
            subprocess.run([
                'espeak-ng', '-v', 'ar', '-s', '130', '-p', pitch,
                '-w', voice_path,
                '\u0628\u0633\u0645 \u0627\u0644\u0644\u0647 \u0627\u0644\u0631\u062d\u0645\u0646 \u0627\u0644\u0631\u062d\u064a\u0645'
            ], capture_output=True, timeout=10)

            if os.path.exists(voice_path):
                return voice_path
        except Exception as e:
            logger.warning(f"Could not generate default speaker wav: {e}")

        return None

    @classmethod
    def synthesize(cls, text, voice='ar_male', speed=1.0):
        """Generate Arabic speech with timeout.

        Returns: (audio_bytes, sample_rate) or (None, None) on failure
        """
        def _do_synthesis():
            model = cls._ensure_model()

            speaker_wav = cls._get_speaker_wav(voice)
            if not speaker_wav:
                logger.error(f"No speaker reference wav found for voice: {voice}")
                return None, None

            start = time.time()

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name

            try:
                model.tts_to_file(
                    text=text,
                    file_path=tmp_path,
                    speaker_wav=speaker_wav,
                    language="ar",
                    split_sentences=True,
                )

                with open(tmp_path, 'rb') as f:
                    audio_bytes = f.read()

                import wave
                with wave.open(tmp_path, 'rb') as wf:
                    sample_rate = wf.getframerate()

                os.unlink(tmp_path)

                elapsed = time.time() - start
                logger.info(f"XTTS-v2 synthesized {len(text)} chars → audio in {elapsed:.2f}s (voice={voice}, lang=ar)")
                return audio_bytes, sample_rate
            except Exception:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                raise

        try:
            return _run_with_timeout(_do_synthesis, cls.TIMEOUT)
        except TimeoutError:
            logger.error(f"XTTS-v2 synthesis timed out after {cls.TIMEOUT}s for {len(text)} chars")
            return None, None
        except Exception as e:
            logger.error(f"XTTS-v2 synthesis failed: {e}", exc_info=True)
            return None, None

    @classmethod
    def is_available(cls):
        try:
            from TTS.api import TTS
            return True, "XTTS-v2 available"
        except ImportError:
            return False, "Coqui TTS not installed (pip install TTS) — requires Python <3.12"


class PiperFallback:
    """Piper TTS fallback — used only if both Kokoro and XTTS-v2 fail."""

    PIPER_BIN = shutil.which("piper") or os.path.expanduser("~/.local/bin/piper")
    VOICE_DIR = os.path.expanduser("~/.local/share/piper_voices")

    @classmethod
    def synthesize(cls, text, voice=None, speed=1.0):
        """Fallback synthesis via Piper."""
        if not os.path.exists(cls.PIPER_BIN):
            return None, None

        onnx_files = []
        if os.path.isdir(cls.VOICE_DIR):
            for f in os.listdir(cls.VOICE_DIR):
                if f.endswith('.onnx'):
                    onnx_files.append(os.path.join(cls.VOICE_DIR, f))

        if not onnx_files:
            return None, None

        model_path = onnx_files[0]

        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name

            result = subprocess.run(
                [cls.PIPER_BIN, '--model', model_path, '--output_file', tmp_path],
                input=text.encode('utf-8'),
                capture_output=True,
                timeout=30,
            )

            if result.returncode == 0 and os.path.exists(tmp_path):
                with open(tmp_path, 'rb') as f:
                    audio_bytes = f.read()
                os.unlink(tmp_path)
                return audio_bytes, 22050

            os.unlink(tmp_path)
            return None, None

        except Exception as e:
            logger.warning(f"Piper fallback failed: {e}")
            return None, None

    @classmethod
    def is_available(cls):
        if os.path.exists(cls.PIPER_BIN):
            return True, "Piper available (fallback)"
        return False, "Piper not installed"


# ═══════════════════════════════════════════════════════════
# MAIN TTS PROCESSOR — LANGUAGE ROUTING
# ═══════════════════════════════════════════════════════════

class TTSProcessor:
    """Unified TTS processor with automatic language-based engine routing.

    Routing logic:
      ar          → XTTS-v2 (only engine with Arabic)
      en, ja, zh, fr, ko, hi, it, pt → Kokoro-82M (best quality + speed)
      anything else → Kokoro with English fallback
      all fail    → Piper (last resort)
    """

    MAX_TEXT_LENGTH = 5000

    @classmethod
    def synthesize(cls, text, voice=None, language=None, speed=1.0):
        """Synthesize speech with automatic engine selection.

        Returns dict with audio bytes and metadata, or error dict.
        """
        start_time = time.time()

        if len(text) > cls.MAX_TEXT_LENGTH:
            text = text[:cls.MAX_TEXT_LENGTH]

        text = cls._clean_text_for_tts(text)

        if not text.strip():
            return {"error": "Nothing to speak — message is only code/formatting"}

        # Determine language
        if not language:
            if voice:
                language = cls._language_from_voice(voice)
            else:
                language = cls._detect_language(text)

        # Determine voice
        if not voice:
            voice = cls._default_voice_for_language(language)

        # Route to engine
        engine_name = None
        audio_bytes = None
        sample_rate = None

        if language == 'ar' or voice in XTTS_VOICES:
            xtts_available, _ = XTTSEngine.is_available()
            if xtts_available:
                audio_bytes, sample_rate = XTTSEngine.synthesize(text, voice=voice, speed=speed)
                engine_name = 'xtts'

        if audio_bytes is None and voice in KOKORO_VOICES:
            kokoro_available, _ = KokoroEngine.is_available()
            if kokoro_available:
                audio_bytes, sample_rate = KokoroEngine.synthesize(text, voice=voice, speed=speed)
                engine_name = 'kokoro'

        if audio_bytes is None:
            kokoro_available, _ = KokoroEngine.is_available()
            if kokoro_available:
                audio_bytes, sample_rate = KokoroEngine.synthesize(text, voice='af_heart', speed=speed)
                voice = 'af_heart'
                engine_name = 'kokoro'

        if audio_bytes is None:
            piper_available, _ = PiperFallback.is_available()
            if piper_available:
                audio_bytes, sample_rate = PiperFallback.synthesize(text, speed=speed)
                engine_name = 'piper'
                voice = 'piper_default'

        if audio_bytes is None:
            return {"error": "No TTS engine available. Install Kokoro (pip install kokoro) or Coqui TTS (pip install TTS)."}

        elapsed = time.time() - start_time
        duration = len(audio_bytes) / (sample_rate * 2) if sample_rate else 0

        return {
            "audio": audio_bytes,
            "sample_rate": sample_rate,
            "engine": engine_name,
            "voice": voice,
            "language": language,
            "duration_seconds": round(duration, 1),
            "processing_seconds": round(elapsed, 2),
        }

    @classmethod
    def get_available_voices(cls):
        """Return all available voices grouped by language and engine."""
        voices = {"kokoro": {}, "xtts": {}, "piper": {}}

        kokoro_available, _ = KokoroEngine.is_available()
        if kokoro_available:
            for voice_id, info in KOKORO_VOICES.items():
                lang = info['lang']
                if lang not in voices['kokoro']:
                    voices['kokoro'][lang] = []
                voices['kokoro'][lang].append({
                    "id": voice_id,
                    "name": info['name'],
                    "gender": info['gender'],
                    "accent": info['accent'],
                })

        xtts_available, _ = XTTSEngine.is_available()
        if xtts_available:
            for voice_id, info in XTTS_VOICES.items():
                lang = info['lang']
                if lang not in voices['xtts']:
                    voices['xtts'][lang] = []
                voices['xtts'][lang].append({
                    "id": voice_id,
                    "name": info['name'],
                    "gender": info['gender'],
                    "accent": info['accent'],
                })

            custom_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'tts_voices')
            if os.path.isdir(custom_dir):
                for f in os.listdir(custom_dir):
                    if f.endswith('.wav') and f.replace('.wav', '') not in XTTS_VOICES:
                        voice_id = f.replace('.wav', '')
                        if 'ar' not in voices['xtts']:
                            voices['xtts']['ar'] = []
                        voices['xtts']['ar'].append({
                            "id": voice_id,
                            "name": voice_id.replace('_', ' ').title(),
                            "gender": "unknown",
                            "accent": "custom",
                        })

        piper_available, _ = PiperFallback.is_available()
        if piper_available:
            voices['piper']['en'] = [{"id": "piper_default", "name": "Piper Default", "gender": "neutral", "accent": "american"}]

        return voices

    @classmethod
    def get_engine_status(cls):
        """Return status of all TTS engines."""
        kokoro_ok, kokoro_msg = KokoroEngine.is_available()
        xtts_ok, xtts_msg = XTTSEngine.is_available()
        piper_ok, piper_msg = PiperFallback.is_available()

        gpu_available = False
        gpu_name = None
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
        except ImportError:
            pass

        xtts_info = {"available": xtts_ok, "message": xtts_msg, "gpu": gpu_available and not XTTSEngine._gpu_failed}
        if XTTSEngine._gpu_failed:
            xtts_info["note"] = "Running on CPU (GPU failed)"

        return {
            "kokoro": {"available": kokoro_ok, "message": kokoro_msg, "gpu": gpu_available},
            "xtts": xtts_info,
            "piper": {"available": piper_ok, "message": piper_msg, "gpu": False},
            "gpu": {"available": gpu_available, "name": gpu_name},
            "supported_languages": {
                "kokoro": sorted(KOKORO_LANGUAGES),
                "xtts": sorted(XTTS_LANGUAGES),
            }
        }

    @classmethod
    def _clean_text_for_tts(cls, text):
        """Strip markdown formatting, code blocks, and other non-speech content."""
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        text = re.sub(r'#{1,6}\s*', '', text)
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)
        text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'---+', '', text)
        text = re.sub(r'\|[^\n]+\|', '', text)
        text = re.sub(r'[\U0001f5e1\ufe0f\U0001f4cd\U0001f3ad\U0001f4ac\u2694\ufe0f\U0001f4cb\U0001f52e\U0001f4dc\U0001f3c6\u26a1\U0001f3af\U0001f4bc\U0001f9e0\u26a0\ufe0f\U0001f30d\U0001f50a\U0001f465\U0001f3b2]\s*', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'  +', ' ', text)
        text = text.strip()
        return text

    @classmethod
    def _language_from_voice(cls, voice):
        """Determine language from voice ID."""
        if voice in KOKORO_VOICES:
            return KOKORO_VOICES[voice]['lang']
        if voice in XTTS_VOICES:
            return XTTS_VOICES[voice]['lang']
        return 'en'

    @classmethod
    def _detect_language(cls, text):
        """Simple language detection from text content."""
        if re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', text):
            return 'ar'
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
            return 'ja'
        if re.search(r'[\u4E00-\u9FFF]', text) and not re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
            return 'zh'
        if re.search(r'[\uAC00-\uD7AF\u1100-\u11FF]', text):
            return 'ko'
        if re.search(r'[\u0900-\u097F]', text):
            return 'hi'
        return 'en'

    @classmethod
    def _default_voice_for_language(cls, language):
        """Pick a default voice for a language."""
        defaults = {
            'en': 'af_heart',
            'ar': 'ar_male',
            'ja': 'jf_alpha',
            'zh': 'zf_xiaobei',
            'fr': 'ff_siwis',
            'ko': 'kf_sarah',
            'hi': 'hf_alpha',
            'it': 'if_sara',
            'pt': 'pf_dora',
        }
        return defaults.get(language, 'af_heart')

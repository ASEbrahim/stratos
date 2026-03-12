"""Dual-engine TTS: Kokoro-82M (8 languages, GPU) + Edge-TTS (Arabic, cloud).

Kokoro: 54 voices, en/ja/zh/fr/ko/hi/it/pt, GPU-accelerated (~500MB VRAM)
Edge-TTS: 26 Arabic dialect voices via Microsoft Neural TTS (cloud, zero VRAM)
"""

import os
import io
import time
import logging
import re
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

# Edge-TTS voices (Microsoft Neural, cloud-based, all Arabic dialects)
EDGE_TTS_VOICES = {
    # Saudi
    "ar-SA-HamedNeural": {"name": "Hamed (Saudi)", "gender": "male", "accent": "saudi", "lang": "ar"},
    "ar-SA-ZariyahNeural": {"name": "Zariyah (Saudi)", "gender": "female", "accent": "saudi", "lang": "ar"},
    # Kuwaiti
    "ar-KW-FahedNeural": {"name": "Fahed (Kuwaiti)", "gender": "male", "accent": "kuwaiti", "lang": "ar"},
    "ar-KW-NouraNeural": {"name": "Noura (Kuwaiti)", "gender": "female", "accent": "kuwaiti", "lang": "ar"},
    # Egyptian
    "ar-EG-ShakirNeural": {"name": "Shakir (Egyptian)", "gender": "male", "accent": "egyptian", "lang": "ar"},
    "ar-EG-SalmaNeural": {"name": "Salma (Egyptian)", "gender": "female", "accent": "egyptian", "lang": "ar"},
    # Emirati
    "ar-AE-HamdanNeural": {"name": "Hamdan (Emirati)", "gender": "male", "accent": "emirati", "lang": "ar"},
    "ar-AE-FatimaNeural": {"name": "Fatima (Emirati)", "gender": "female", "accent": "emirati", "lang": "ar"},
    # Iraqi
    "ar-IQ-BasselNeural": {"name": "Bassel (Iraqi)", "gender": "male", "accent": "iraqi", "lang": "ar"},
    "ar-IQ-RanaNeural": {"name": "Rana (Iraqi)", "gender": "female", "accent": "iraqi", "lang": "ar"},
    # Jordanian
    "ar-JO-TaimNeural": {"name": "Taim (Jordanian)", "gender": "male", "accent": "jordanian", "lang": "ar"},
    "ar-JO-SanaNeural": {"name": "Sana (Jordanian)", "gender": "female", "accent": "jordanian", "lang": "ar"},
    # Lebanese
    "ar-LB-RamiNeural": {"name": "Rami (Lebanese)", "gender": "male", "accent": "lebanese", "lang": "ar"},
    "ar-LB-LaylaNeural": {"name": "Layla (Lebanese)", "gender": "female", "accent": "lebanese", "lang": "ar"},
    # Syrian
    "ar-SY-LaithNeural": {"name": "Laith (Syrian)", "gender": "male", "accent": "syrian", "lang": "ar"},
    "ar-SY-AmanyNeural": {"name": "Amany (Syrian)", "gender": "female", "accent": "syrian", "lang": "ar"},
    # Moroccan
    "ar-MA-JamalNeural": {"name": "Jamal (Moroccan)", "gender": "male", "accent": "moroccan", "lang": "ar"},
    "ar-MA-MounaNeural": {"name": "Mouna (Moroccan)", "gender": "female", "accent": "moroccan", "lang": "ar"},
    # Bahraini
    "ar-BH-AliNeural": {"name": "Ali (Bahraini)", "gender": "male", "accent": "bahraini", "lang": "ar"},
    "ar-BH-LailaNeural": {"name": "Laila (Bahraini)", "gender": "female", "accent": "bahraini", "lang": "ar"},
    # Qatari
    "ar-QA-MoazNeural": {"name": "Moaz (Qatari)", "gender": "male", "accent": "qatari", "lang": "ar"},
    "ar-QA-AmalNeural": {"name": "Amal (Qatari)", "gender": "female", "accent": "qatari", "lang": "ar"},
    # Algerian
    "ar-DZ-IsmaelNeural": {"name": "Ismael (Algerian)", "gender": "male", "accent": "algerian", "lang": "ar"},
    "ar-DZ-AminaNeural": {"name": "Amina (Algerian)", "gender": "female", "accent": "algerian", "lang": "ar"},
}

# Language code → Kokoro lang_code mapping
KOKORO_LANG_CODES = {
    "en": "a",
    "en-us": "a",
    "en-gb": "b",
    "ja": "j",
    "zh": "z",
    "fr": "f",
    "ko": "k",
    "hi": "h",
    "it": "i",
    "pt": "p",
}

KOKORO_LANGUAGES = {"en", "ja", "zh", "fr", "ko", "hi", "it", "pt"}
EDGE_LANGUAGES = {"ar"}

# Default voices per persona (overridable via config.yaml tts.persona_voices)
_DEFAULT_PERSONA_VOICES = {
    "intelligence": "af_heart",
    "market": "am_michael",
    "scholarly": "ar-KW-FahedNeural",
    "gaming": "am_fenrir",
    "anime": "jf_alpha",
    "tcg": "am_echo",
}

PERSONA_DEFAULT_VOICES = dict(_DEFAULT_PERSONA_VOICES)


def load_persona_voices_from_config(config: dict):
    """Override persona default voices from config.yaml tts.persona_voices section."""
    tts_cfg = config.get("tts", {})
    overrides = tts_cfg.get("persona_voices", {})
    for persona, voice_id in overrides.items():
        if voice_id in KOKORO_VOICES or voice_id in EDGE_TTS_VOICES or voice_id.startswith('ar-'):
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
    """Kokoro-82M TTS engine — 54 voices, 8 languages, CPU (faster than ROCm for this model)."""

    MAX_PIPELINES = 3
    _pipelines = OrderedDict()
    TIMEOUT = 15

    @classmethod
    def _get_pipeline(cls, lang_code='a'):
        """Get or create a Kokoro pipeline for the given language (LRU cached, max 3)."""
        if lang_code in cls._pipelines:
            cls._pipelines.move_to_end(lang_code)
            return cls._pipelines[lang_code]

        try:
            from kokoro import KPipeline
            logger.info(f"Loading Kokoro pipeline for lang_code='{lang_code}'")

            if len(cls._pipelines) >= cls.MAX_PIPELINES:
                evicted_key, _ = cls._pipelines.popitem(last=False)
                logger.info(f"Evicted Kokoro pipeline '{evicted_key}' (LRU, max {cls.MAX_PIPELINES})")

            pipeline = KPipeline(lang_code=lang_code, device='cpu')
            cls._pipelines[lang_code] = pipeline
            logger.info(f"Kokoro pipeline loaded on CPU: '{lang_code}' ({len(cls._pipelines)}/{cls.MAX_PIPELINES} active)")
            return pipeline
        except Exception as e:
            logger.error(f"Failed to load Kokoro pipeline: {e}")
            return None

    @classmethod
    def synthesize(cls, text, voice='af_heart', speed=1.0):
        """Generate speech from text with timeout."""
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
            for i, (gs, ps, audio) in enumerate(pipeline(text, voice=voice, speed=speed)):
                audio_segments.append(audio)

            if not audio_segments:
                return None, None

            full_audio = np.concatenate(audio_segments)
            buf = io.BytesIO()
            sf.write(buf, full_audio, 24000, format='WAV')
            buf.seek(0)

            elapsed = time.time() - start
            logger.info(f"Kokoro: {len(text)} chars → {len(full_audio)/24000:.1f}s audio in {elapsed:.2f}s (voice={voice})")
            return buf.read(), 24000

        try:
            return _run_with_timeout(_do_synthesis, cls.TIMEOUT)
        except TimeoutError:
            logger.error(f"Kokoro timed out after {cls.TIMEOUT}s for {len(text)} chars")
            return None, None
        except Exception as e:
            logger.error(f"Kokoro synthesis failed: {e}", exc_info=True)
            return None, None

    @classmethod
    def is_available(cls):
        try:
            from kokoro import KPipeline
            return True, "Kokoro-82M available (54 voices, 8 languages, CPU)"
        except ImportError:
            return False, "Kokoro not installed (pip install kokoro>=0.9.2)"


class EdgeTTSEngine:
    """Edge-TTS engine — Microsoft Neural voices for Arabic (cloud-based, zero VRAM)."""

    TIMEOUT = 15

    @classmethod
    def synthesize(cls, text, voice='ar-KW-FahedNeural', speed=1.0):
        """Generate Arabic speech via Microsoft Edge TTS. Returns MP3 bytes."""
        import asyncio

        def _do_synthesis():
            import edge_tts

            async def _generate():
                rate_str = f"+{int((speed - 1) * 100)}%" if speed >= 1 else f"{int((speed - 1) * 100)}%"
                comm = edge_tts.Communicate(text, voice, rate=rate_str)
                buf = io.BytesIO()
                async for chunk in comm.stream():
                    if chunk['type'] == 'audio':
                        buf.write(chunk['data'])
                return buf.getvalue()

            start = time.time()
            try:
                loop = asyncio.new_event_loop()
                mp3_data = loop.run_until_complete(_generate())
                loop.close()
            except Exception as e:
                logger.error(f"Edge-TTS stream failed: {e}")
                return None, None

            if not mp3_data:
                return None, None

            elapsed = time.time() - start
            logger.info(f"Edge-TTS: {len(text)} chars in {elapsed:.2f}s (voice={voice})")
            return mp3_data, 24000

        try:
            return _run_with_timeout(_do_synthesis, cls.TIMEOUT)
        except TimeoutError:
            logger.error(f"Edge-TTS timed out after {cls.TIMEOUT}s for {len(text)} chars")
            return None, None
        except Exception as e:
            logger.error(f"Edge-TTS synthesis failed: {e}", exc_info=True)
            return None, None

    @classmethod
    def is_available(cls):
        try:
            import edge_tts
            return True, "Edge-TTS available (26 Arabic voices)"
        except ImportError:
            return False, "edge-tts not installed (pip install edge-tts)"


# ═══════════════════════════════════════════════════════════
# MAIN TTS PROCESSOR — LANGUAGE ROUTING
# ═══════════════════════════════════════════════════════════

class TTSProcessor:
    """Unified TTS processor with automatic language-based engine routing.

    Routing:
      ar → Edge-TTS (26 Arabic dialect voices)
      en, ja, zh, fr, ko, hi, it, pt → Kokoro-82M (GPU)
      anything else → Kokoro with English fallback
    """

    MAX_TEXT_LENGTH = 5000

    @classmethod
    def synthesize(cls, text, voice=None, language=None, speed=1.0):
        """Synthesize speech with automatic engine selection."""
        start_time = time.time()

        if len(text) > cls.MAX_TEXT_LENGTH:
            text = text[:cls.MAX_TEXT_LENGTH]

        text = cls._clean_text_for_tts(text)

        if not text.strip():
            return {"error": "Nothing to speak — message is only code/formatting"}

        # Always detect the actual text language
        detected_lang = cls._detect_language(text)

        if not language:
            language = detected_lang

        # If user selected a voice in a different language than the text,
        # override with the correct language's default voice
        if voice:
            voice_lang = cls._language_from_voice(voice)
            if voice_lang != language:
                voice = cls._default_voice_for_language(language)
                logger.info(f"TTS: voice language mismatch ({voice_lang} vs text {language}), using default {voice}")

        if not voice:
            voice = cls._default_voice_for_language(language)

        engine_name = None
        audio_bytes = None
        sample_rate = None

        # Arabic → Edge-TTS
        if language == 'ar' or voice in EDGE_TTS_VOICES or voice.startswith('ar-'):
            edge_available, _ = EdgeTTSEngine.is_available()
            if edge_available:
                audio_bytes, sample_rate = EdgeTTSEngine.synthesize(text, voice=voice, speed=speed)
                engine_name = 'edge'

        # Kokoro languages
        if audio_bytes is None and voice in KOKORO_VOICES:
            kokoro_available, _ = KokoroEngine.is_available()
            if kokoro_available:
                audio_bytes, sample_rate = KokoroEngine.synthesize(text, voice=voice, speed=speed)
                engine_name = 'kokoro'

        # Fallback: Kokoro with default English voice
        if audio_bytes is None:
            kokoro_available, _ = KokoroEngine.is_available()
            if kokoro_available:
                audio_bytes, sample_rate = KokoroEngine.synthesize(text, voice='af_heart', speed=speed)
                voice = 'af_heart'
                engine_name = 'kokoro'

        if audio_bytes is None:
            return {"error": "No TTS engine available. Install Kokoro (pip install kokoro) and/or edge-tts (pip install edge-tts)."}

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
        """Return all available voices grouped by engine and language."""
        voices = {"kokoro": {}, "edge": {}}

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

        edge_available, _ = EdgeTTSEngine.is_available()
        if edge_available:
            for voice_id, info in EDGE_TTS_VOICES.items():
                lang = info['lang']
                if lang not in voices['edge']:
                    voices['edge'][lang] = []
                voices['edge'][lang].append({
                    "id": voice_id,
                    "name": info['name'],
                    "gender": info['gender'],
                    "accent": info['accent'],
                })

        return voices

    @classmethod
    def get_engine_status(cls):
        """Return status of all TTS engines."""
        kokoro_ok, kokoro_msg = KokoroEngine.is_available()
        edge_ok, edge_msg = EdgeTTSEngine.is_available()

        gpu_available = False
        gpu_name = None
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
        except ImportError:
            pass

        return {
            "kokoro": {"available": kokoro_ok, "message": kokoro_msg, "gpu": gpu_available},
            "edge": {"available": edge_ok, "message": edge_msg, "gpu": False},
            "gpu": {"available": gpu_available, "name": gpu_name},
            "supported_languages": {
                "kokoro": sorted(KOKORO_LANGUAGES),
                "edge": sorted(EDGE_LANGUAGES),
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
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'  +', ' ', text)
        text = text.strip()
        return text

    @classmethod
    def _language_from_voice(cls, voice):
        """Determine language from voice ID."""
        if voice in KOKORO_VOICES:
            return KOKORO_VOICES[voice]['lang']
        if voice in EDGE_TTS_VOICES or voice.startswith('ar-'):
            return 'ar'
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
            'ar': 'ar-KW-FahedNeural',
            'ja': 'jf_alpha',
            'zh': 'zf_xiaobei',
            'fr': 'ff_siwis',
            'ko': 'kf_sarah',
            'hi': 'hf_alpha',
            'it': 'if_sara',
            'pt': 'pf_dora',
        }
        return defaults.get(language, 'af_heart')

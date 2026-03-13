"""
YouTube Knowledge Extraction Pipeline — Transcript Acquisition & Channel Management

Three-tier transcript acquisition:
  Tier 1: youtube-transcript-api (free, instant, for captioned videos)
  Tier 2: Supadata API (hosted, handles no-caption videos, 100 free/month)
  Tier 3: faster-whisper (local CPU, offline, slowest, most reliable)

Usage:
    yt = YouTubeProcessor(config, db)
    transcript, method = yt.get_transcript("VIDEO_ID")
    videos = yt.fetch_channel_videos("CHANNEL_ID")
"""

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from xml.etree import ElementTree

import requests

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# CHANNEL MANAGEMENT
# ═══════════════════════════════════════════════════════════

def parse_youtube_input(input_str: str) -> Dict[str, str]:
    """Detect whether input is a video URL, channel URL, or handle.

    Returns dict with 'type' ('video', 'channel', or 'unknown') and 'id'.
    """
    input_str = input_str.strip()

    # Video URL: youtube.com/watch?v=XXXXX or youtu.be/XXXXX
    video_match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})', input_str)
    if video_match:
        return {'type': 'video', 'id': video_match.group(1)}

    # YouTube Shorts: youtube.com/shorts/XXXXX
    shorts_match = re.search(r'youtube\.com/shorts/([\w-]{11})', input_str)
    if shorts_match:
        return {'type': 'video', 'id': shorts_match.group(1)}

    # Everything else is treated as a channel input
    return {'type': 'channel', 'id': input_str}


def resolve_channel_id(channel_input: str) -> Optional[Dict[str, str]]:
    """Resolve a YouTube channel URL, @handle, or channel ID to a channel_id.

    Returns dict with 'channel_id' and 'channel_name', or None on failure.
    """
    channel_input = channel_input.strip()

    # Direct channel ID
    if re.match(r'^UC[\w-]{22}$', channel_input):
        return {'channel_id': channel_input, 'channel_name': ''}

    # Extract from URL patterns
    patterns = [
        r'youtube\.com/channel/(UC[\w-]{22})',           # /channel/UCxxxx
        r'youtube\.com/@([\w.-]+)',                        # /@handle
        r'youtube\.com/c/([\w.-]+)',                       # /c/name
        r'youtube\.com/user/([\w.-]+)',                    # /user/name
    ]

    identifier = None
    is_handle = False
    for pat in patterns:
        m = re.search(pat, channel_input)
        if m:
            identifier = m.group(1)
            if identifier.startswith('UC') and len(identifier) == 24:
                return {'channel_id': identifier, 'channel_name': ''}
            is_handle = True
            break

    if not identifier:
        # Bare @handle
        if channel_input.startswith('@'):
            identifier = channel_input[1:]
            is_handle = True
        else:
            identifier = channel_input
            is_handle = True

    if not is_handle:
        return None

    # Resolve handle to channel ID by fetching the channel page
    try:
        resp = requests.get(
            f'https://www.youtube.com/@{identifier}',
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=15,
            allow_redirects=True,
        )
        if resp.status_code != 200:
            logger.warning(f"YouTube channel resolve failed: HTTP {resp.status_code}")
            return None

        # Extract channel ID from page meta
        m = re.search(r'"channelId":"(UC[\w-]{22})"', resp.text)
        if not m:
            m = re.search(r'<meta\s+itemprop="channelId"\s+content="(UC[\w-]{22})"', resp.text)
        if not m:
            m = re.search(r'channel_id=(UC[\w-]{22})', resp.text)
        if not m:
            logger.warning(f"Could not extract channel ID from @{identifier}")
            return None

        channel_id = m.group(1)

        # Try to get channel name
        name_match = re.search(r'"name":"([^"]+)"', resp.text)
        channel_name = name_match.group(1) if name_match else identifier

        return {'channel_id': channel_id, 'channel_name': channel_name}

    except Exception as e:
        logger.error(f"Channel resolve error: {e}")
        return None


def fetch_channel_videos(channel_id: str, limit: int = 15) -> List[Dict[str, Any]]:
    """Fetch recent videos from a YouTube channel via RSS feed.

    Returns list of dicts with: video_id, title, published_at, link
    """
    feed_url = f'https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}'
    try:
        resp = requests.get(feed_url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        if resp.status_code != 200:
            logger.warning(f"YouTube RSS feed returned {resp.status_code} for {channel_id}")
            return []

        root = ElementTree.fromstring(resp.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom', 'yt': 'http://www.youtube.com/xml/schemas/2015'}

        videos = []
        for entry in root.findall('atom:entry', ns)[:limit]:
            video_id_el = entry.find('yt:videoId', ns)
            title_el = entry.find('atom:title', ns)
            published_el = entry.find('atom:published', ns)

            if video_id_el is None:
                continue

            videos.append({
                'video_id': video_id_el.text,
                'title': title_el.text if title_el is not None else '',
                'published_at': published_el.text if published_el is not None else '',
                'link': f'https://www.youtube.com/watch?v={video_id_el.text}',
            })

        return videos

    except Exception as e:
        logger.error(f"RSS feed error for {channel_id}: {e}")
        return []


# ═══════════════════════════════════════════════════════════
# TRANSCRIPT ACQUISITION (3-tier)
# ═══════════════════════════════════════════════════════════

def get_transcript(video_id: str, preferred_lang: str = 'ar',
                   supadata_key: str = '', whisper_model: str = 'large-v3-turbo') -> Tuple[str, str, str]:
    """Get transcript using best available method.

    Tries in order:
      Tier 1: youtube-transcript-api (free, instant)
      Tier 2: Supadata API (if key configured)
      Tier 3: faster-whisper on CPU (slow but reliable)

    Returns (transcript_text, method, detected_language) where method is
    'youtube-api'/'supadata'/'whisper-turbo'/'whisper-v3'.
    Raises RuntimeError if all tiers fail.
    """
    # Tier 1: youtube-transcript-api
    text, method, lang = _tier1_youtube_api(video_id, preferred_lang)
    if text:
        return text, method, lang

    # Tier 2: Supadata API
    if supadata_key:
        text, method, lang = _tier2_supadata(video_id, supadata_key, preferred_lang)
        if text:
            return text, method, lang

    # Tier 3: faster-whisper
    text, method, lang = _tier3_whisper(video_id, whisper_model)
    if text:
        return text, method, lang

    raise RuntimeError(f"All transcript tiers failed for video {video_id}")


def _tier1_youtube_api(video_id: str, preferred_lang: str = 'ar') -> Tuple[Optional[str], str, str]:
    """Tier 1: Free youtube-transcript-api library (v1.2.4+)."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        api = YouTubeTranscriptApi()

        # Try fetching transcript with language fallback chain
        # Each entry: (lang_codes, detected_lang_if_matched)
        _LANG_CHAIN = [
            (['en'], 'en'),
            (['en-US', 'en-GB'], 'en'),
            (['ar'], 'ar'),
            (['ja'], 'ja'),
            (['zh-CN', 'zh-Hans'], 'zh'),
            (['zh-TW', 'zh-Hant', 'zh-HK'], 'zh'),
            (['ko'], 'ko'),
            (['fr'], 'fr'),
            (['de'], 'de'),
            (['es'], 'es'),
            (['ru'], 'ru'),
            ([], None),  # Any available
        ]
        transcript = None
        detected_lang = 'en'
        for lang_list, lang_tag in _LANG_CHAIN:
            try:
                if lang_list:
                    transcript = api.fetch(video_id, languages=lang_list)
                else:
                    transcript = api.fetch(video_id)
                if transcript:
                    if lang_tag:
                        detected_lang = lang_tag
                    else:
                        detected_lang = getattr(transcript, 'language', None) or 'en'
                    break
            except Exception:
                continue

        if not transcript:
            return None, '', ''

        # Extract text from transcript entries (v1.2.4 returns objects with .text)
        parts = []
        for entry in transcript:
            text = getattr(entry, 'text', None) or (entry.get('text') if isinstance(entry, dict) else '')
            if text:
                parts.append(text)

        full_text = ' '.join(parts)
        if len(full_text.strip()) < 500:
            logger.debug(f"Tier 1 (youtube-api): too short ({len(full_text)} chars) for {video_id} — likely bad data")
            return None, '', ''

        logger.info(f"Tier 1 (youtube-api): got {len(full_text)} chars for {video_id} [lang={detected_lang}]")
        return full_text.strip(), 'youtube-api', detected_lang

    except ImportError:
        logger.warning("youtube-transcript-api not installed")
        return None, '', ''
    except Exception as e:
        logger.debug(f"Tier 1 failed for {video_id}: {e}")
        return None, '', ''


def _tier2_supadata(video_id: str, api_key: str, preferred_lang: str = 'en') -> Tuple[Optional[str], str, str]:
    """Tier 2: Supadata hosted API for no-caption videos."""
    try:
        detected_lang = 'en'
        # Try preferred lang first, then fallback to no lang preference
        for lang in [preferred_lang, 'en', 'ja', 'zh', 'ko', None]:
            params = {'url': f'https://youtube.com/watch?v={video_id}'}
            if lang:
                params['lang'] = lang
            resp = requests.get(
                'https://api.supadata.ai/v1/youtube/transcript',
                params=params,
                headers={'x-api-key': api_key},
                timeout=60,
            )
            if resp.status_code == 200:
                detected_lang = lang or 'en'
                break
        else:
            logger.debug(f"Tier 2 (Supadata) all langs failed for {video_id}")
            return None, '', ''

        if resp.status_code != 200:
            logger.debug(f"Tier 2 (Supadata) returned {resp.status_code} for {video_id}")
            return None, '', ''

        data = resp.json()
        # Supadata may return detected language
        detected_lang = data.get('lang', detected_lang) or detected_lang
        # Supadata returns content as array of segments or as text
        content = data.get('content', '')
        if isinstance(content, list):
            full_text = ' '.join(seg.get('text', '') for seg in content if seg.get('text'))
        elif isinstance(content, str):
            full_text = content
        else:
            return None, '', ''

        if len(full_text.strip()) < 500:
            logger.debug(f"Tier 2 (Supadata): too short ({len(full_text)} chars) for {video_id} — likely bad data")
            return None, '', ''

        logger.info(f"Tier 2 (Supadata): got {len(full_text)} chars for {video_id} [lang={detected_lang}]")
        return full_text.strip(), 'supadata', detected_lang

    except Exception as e:
        logger.debug(f"Tier 2 failed for {video_id}: {e}")
        return None, '', ''


def _tier3_whisper(video_id: str, model_name: str = 'large-v3-turbo') -> Tuple[Optional[str], str, str]:
    """Tier 3: Local faster-whisper transcription on CPU."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.warning("faster-whisper not installed — skipping Tier 3")
        return None, '', ''

    # Download audio via yt-dlp (no ffmpeg conversion — faster-whisper reads webm/opus natively)
    audio_path = None
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, f'{video_id}.%(ext)s')
            result = subprocess.run(
                [
                    'yt-dlp', '-f', 'bestaudio',
                    '--no-playlist',
                    '-o', audio_path,
                    f'https://www.youtube.com/watch?v={video_id}',
                ],
                capture_output=True, timeout=300,
            )
            if result.returncode != 0:
                logger.error(f"yt-dlp failed: {result.stderr.decode()[:300]}")
                return None, '', ''

            # Find the actual output file (yt-dlp may add extension)
            actual_files = list(Path(tmpdir).glob(f'{video_id}*'))
            if not actual_files:
                logger.error("yt-dlp produced no output file")
                return None, '', ''
            audio_path = str(actual_files[0])

            # Transcribe with faster-whisper
            logger.info(f"Tier 3: Transcribing {video_id} with {model_name} on CPU...")
            model = WhisperModel(model_name, device='cpu', compute_type='int8')
            segments, info = model.transcribe(audio_path, language=None)  # auto-detect language
            detected_lang = getattr(info, 'language', 'en') or 'en'

            full_text = ' '.join(segment.text for segment in segments)
            if len(full_text.strip()) < 50:
                # If turbo produced poor results, try full large-v3
                if model_name == 'large-v3-turbo':
                    logger.info("Turbo output too short, trying large-v3...")
                    model = WhisperModel('large-v3', device='cpu', compute_type='int8')
                    segments, info = model.transcribe(audio_path, language=None)
                    detected_lang = getattr(info, 'language', 'en') or 'en'
                    full_text = ' '.join(segment.text for segment in segments)
                    if len(full_text.strip()) >= 50:
                        method = 'whisper-v3'
                        logger.info(f"Tier 3 (whisper-v3): got {len(full_text)} chars for {video_id} [lang={detected_lang}]")
                        return full_text.strip(), method, detected_lang
                return None, '', ''

            method = f'whisper-{"turbo" if "turbo" in model_name else "v3"}'
            logger.info(f"Tier 3 ({method}): got {len(full_text)} chars for {video_id} [lang={detected_lang}]")
            return full_text.strip(), method, detected_lang

    except subprocess.TimeoutExpired:
        logger.error("yt-dlp timed out")
        return None, '', ''
    except Exception as e:
        logger.error(f"Tier 3 failed for {video_id}: {e}")
        return None, '', ''


# ═══════════════════════════════════════════════════════════
# YOUTUBE PROCESSOR CLASS
# ═══════════════════════════════════════════════════════════

class YouTubeProcessor:
    """Manages YouTube channels, videos, transcripts, and lens extraction."""

    def __init__(self, config: dict, db=None):
        self.config = config
        self.db = db
        self.supadata_key = os.environ.get('SUPADATA_API_KEY', '')
        self.whisper_model = config.get('youtube', {}).get('whisper_model', 'large-v3-turbo')
        self.ollama_host = config.get('scoring', {}).get('ollama_host', 'http://localhost:11434')
        self.inference_model = config.get('scoring', {}).get('inference_model', 'qwen3.5:9b')

    def add_channel(self, profile_id: int, channel_input: str,
                    lenses: List[str] = None) -> Optional[Dict[str, Any]]:
        """Add a YouTube channel to tracking.

        Args:
            profile_id: User's profile ID
            channel_input: URL, @handle, or channel ID
            lenses: List of lens names to apply (default: ['summary'])

        Returns:
            Channel info dict or None on failure
        """
        if not self.db:
            return None

        resolved = resolve_channel_id(channel_input)
        if not resolved:
            return None

        channel_id = resolved['channel_id']
        channel_name = resolved.get('channel_name', '')
        lenses = lenses or ['transcript']

        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                """INSERT OR IGNORE INTO youtube_channels
                   (profile_id, channel_id, channel_name, channel_url, lenses)
                   VALUES (?, ?, ?, ?, ?)""",
                (profile_id, channel_id, channel_name,
                 f'https://www.youtube.com/channel/{channel_id}',
                 json.dumps(lenses))
            )
            self.db._commit()
            if cursor.rowcount == 0:
                # Already exists
                cursor.execute(
                    "SELECT * FROM youtube_channels WHERE profile_id = ? AND channel_id = ?",
                    (profile_id, channel_id)
                )
                row = cursor.fetchone()
                return dict(row) if row else None

            return {
                'id': cursor.lastrowid,
                'profile_id': profile_id,
                'channel_id': channel_id,
                'channel_name': channel_name,
                'lenses': lenses,
            }
        except Exception as e:
            logger.error(f"Failed to add channel: {e}")
            return None

    def add_single_video(self, profile_id: int, video_id: str,
                         lenses: List[str] = None) -> Optional[Dict[str, Any]]:
        """Add a single YouTube video by video ID for processing.

        Creates a pseudo-channel '__singles__' to hold standalone videos,
        then inserts the video as pending.
        """
        if not self.db:
            return None
        lenses = lenses or ['transcript']

        try:
            cursor = self.db.conn.cursor()

            # Ensure a pseudo-channel exists for single videos
            cursor.execute(
                "SELECT id FROM youtube_channels WHERE profile_id = ? AND channel_id = '__singles__'",
                (profile_id,)
            )
            row = cursor.fetchone()
            if row:
                ch_db_id = row['id']
                # Update lenses to include any new ones
                cursor.execute(
                    "UPDATE youtube_channels SET lenses = ? WHERE id = ?",
                    (json.dumps(lenses), ch_db_id)
                )
            else:
                cursor.execute(
                    """INSERT INTO youtube_channels
                       (profile_id, channel_id, channel_name, channel_url, lenses)
                       VALUES (?, '__singles__', 'Single Videos', '', ?)""",
                    (profile_id, json.dumps(lenses))
                )
                ch_db_id = cursor.lastrowid

            # Fetch video title from YouTube page
            title = video_id
            try:
                resp = requests.get(
                    f'https://www.youtube.com/watch?v={video_id}',
                    headers={'User-Agent': 'Mozilla/5.0'},
                    timeout=10,
                )
                m = re.search(r'<title>(.+?)(?:\s*-\s*YouTube)?</title>', resp.text)
                if m:
                    title = m.group(1).strip()
            except Exception:
                pass

            # Insert video (ignore if already exists)
            cursor.execute(
                """INSERT OR IGNORE INTO youtube_videos
                   (channel_id, profile_id, video_id, title, published_at, status)
                   VALUES (?, ?, ?, ?, ?, 'pending')""",
                (ch_db_id, profile_id, video_id, title,
                 datetime.now().isoformat())
            )
            self.db._commit()

            if cursor.rowcount == 0:
                # Already exists — return existing
                cursor.execute(
                    "SELECT * FROM youtube_videos WHERE profile_id = ? AND video_id = ?",
                    (profile_id, video_id)
                )
                existing = cursor.fetchone()
                return dict(existing) if existing else None

            return {
                'id': cursor.lastrowid,
                'video_id': video_id,
                'title': title,
                'status': 'pending',
                'channel_id': ch_db_id,
            }
        except Exception as e:
            logger.error(f"Failed to add single video: {e}")
            return None

    def list_channels(self, profile_id: int) -> List[Dict[str, Any]]:
        """List all tracked channels for a profile."""
        if not self.db:
            return []
        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                """SELECT c.*, COUNT(v.id) as video_count,
                          SUM(CASE WHEN v.status = 'complete' THEN 1 ELSE 0 END) as completed_count
                   FROM youtube_channels c
                   LEFT JOIN youtube_videos v ON v.channel_id = c.id AND v.profile_id = c.profile_id
                   WHERE c.profile_id = ?
                   GROUP BY c.id
                   ORDER BY c.added_at DESC""",
                (profile_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to list channels: {e}")
            return []

    def remove_channel(self, profile_id: int, channel_db_id: int) -> bool:
        """Remove a tracked channel and its data."""
        if not self.db:
            return False
        try:
            cursor = self.db.conn.cursor()
            # Delete insights for this channel's videos
            cursor.execute(
                """DELETE FROM video_insights WHERE video_id IN
                   (SELECT id FROM youtube_videos WHERE channel_id = ? AND profile_id = ?)""",
                (channel_db_id, profile_id)
            )
            # Delete videos
            cursor.execute(
                "DELETE FROM youtube_videos WHERE channel_id = ? AND profile_id = ?",
                (channel_db_id, profile_id)
            )
            # Delete channel
            cursor.execute(
                "DELETE FROM youtube_channels WHERE id = ? AND profile_id = ?",
                (channel_db_id, profile_id)
            )
            self.db._commit()
            return True
        except Exception as e:
            logger.error(f"Failed to remove channel: {e}")
            return False

    def discover_new_videos(self, profile_id: int, channel_db_id: int = None) -> List[Dict[str, Any]]:
        """Fetch new videos from tracked channels and insert as pending.

        Args:
            profile_id: User's profile ID
            channel_db_id: Specific channel to check (None = all channels)

        Returns:
            List of newly discovered video dicts
        """
        if not self.db:
            return []

        cursor = self.db.conn.cursor()
        if channel_db_id:
            cursor.execute(
                "SELECT * FROM youtube_channels WHERE id = ? AND profile_id = ?",
                (channel_db_id, profile_id)
            )
        else:
            cursor.execute(
                "SELECT * FROM youtube_channels WHERE profile_id = ?",
                (profile_id,)
            )

        channels = [dict(r) for r in cursor.fetchall()]
        new_videos = []

        for ch in channels:
            videos = fetch_channel_videos(ch['channel_id'])
            for v in videos:
                try:
                    cursor.execute(
                        """INSERT OR IGNORE INTO youtube_videos
                           (channel_id, profile_id, video_id, title, published_at, status)
                           VALUES (?, ?, ?, ?, ?, 'pending')""",
                        (ch['id'], profile_id, v['video_id'], v['title'], v['published_at'])
                    )
                    if cursor.rowcount > 0:
                        new_videos.append({
                            'id': cursor.lastrowid,
                            'channel_id': ch['id'],
                            'channel_name': ch.get('channel_name', ''),
                            'video_id': v['video_id'],
                            'title': v['title'],
                            'published_at': v['published_at'],
                            'status': 'pending',
                        })
                except Exception:
                    continue

            # Update last_checked
            cursor.execute(
                "UPDATE youtube_channels SET last_checked = ? WHERE id = ?",
                (datetime.now().isoformat(), ch['id'])
            )

        self.db._commit()
        if new_videos:
            logger.info(f"Discovered {len(new_videos)} new videos for profile {profile_id}")
        return new_videos

    def get_transcript_for_video(self, video_id: str) -> Tuple[str, str, str]:
        """Get transcript for a video using the 3-tier system.

        Returns (transcript_text, method, detected_language).
        """
        return get_transcript(
            video_id,
            preferred_lang='ar',
            supadata_key=self.supadata_key,
            whisper_model=self.whisper_model,
        )

    def process_video(self, video_db_id: int, profile_id: int) -> bool:
        """Process a single video: transcribe + run lenses.

        Returns True on success, False on failure.
        """
        if not self.db:
            return False

        cursor = self.db.conn.cursor()
        cursor.execute(
            "SELECT v.*, c.lenses FROM youtube_videos v "
            "JOIN youtube_channels c ON v.channel_id = c.id "
            "WHERE v.id = ? AND v.profile_id = ?",
            (video_db_id, profile_id)
        )
        row = cursor.fetchone()
        if not row:
            return False

        video = dict(row)
        video_id = video['video_id']

        # Step 1: Transcribe
        self._update_status(video_db_id, 'transcribing')
        try:
            transcript, method, detected_lang = self.get_transcript_for_video(video_id)
        except RuntimeError as e:
            self._update_status(video_db_id, 'failed', str(e))
            return False

        # Save transcript + detected language
        cursor.execute(
            "UPDATE youtube_videos SET transcript_text = ?, transcript_method = ?, transcript_language = ? WHERE id = ?",
            (transcript, method, detected_lang, video_db_id)
        )
        self.db._commit()

        # Step 2: Run lenses (bilingual — original language + English)
        self._update_status(video_db_id, 'extracting')
        lenses = json.loads(video.get('lenses', '["summary"]'))

        # Special lens: 'transcript' — store raw transcript text, no LLM call
        if 'transcript' in lenses:
            lenses = [l for l in lenses if l != 'transcript']
            # Store transcript as-is in the detected language
            cursor.execute(
                """INSERT INTO video_insights
                   (video_id, profile_id, lens_name, content, language)
                   VALUES (?, ?, 'transcript', ?, ?)""",
                (video_db_id, profile_id,
                 json.dumps({"transcript": transcript}, ensure_ascii=False),
                 detected_lang)
            )
            # If non-English, also store under 'en' key (same text — user can read original)
            if detected_lang and detected_lang != 'en':
                cursor.execute(
                    """INSERT INTO video_insights
                       (video_id, profile_id, lens_name, content, language)
                       VALUES (?, ?, 'transcript', ?, 'en')""",
                    (video_db_id, profile_id,
                     json.dumps({"transcript": transcript, "note": f"Original language: {detected_lang}"}, ensure_ascii=False))
                )
            self.db._commit()

        for lens_name in lenses:
            try:
                from processors.lenses import extract_lens
                # Always extract in English
                insight_en = extract_lens(
                    transcript, lens_name, video.get('title', ''),
                    self.ollama_host, self.inference_model,
                    target_language='en',
                )
                if insight_en:
                    cursor.execute(
                        """INSERT INTO video_insights
                           (video_id, profile_id, lens_name, content, language)
                           VALUES (?, ?, ?, ?, 'en')""",
                        (video_db_id, profile_id, lens_name, json.dumps(insight_en, ensure_ascii=False))
                    )
                # If transcript is non-English, also extract in original language
                if detected_lang and detected_lang != 'en':
                    insight_orig = extract_lens(
                        transcript, lens_name, video.get('title', ''),
                        self.ollama_host, self.inference_model,
                        target_language=detected_lang,
                    )
                    if insight_orig:
                        cursor.execute(
                            """INSERT INTO video_insights
                               (video_id, profile_id, lens_name, content, language)
                               VALUES (?, ?, ?, ?, ?)""",
                            (video_db_id, profile_id, lens_name,
                             json.dumps(insight_orig, ensure_ascii=False), detected_lang)
                        )
            except Exception as e:
                logger.error(f"Lens '{lens_name}' failed for video {video_id}: {e}")

        self.db._commit()
        self._update_status(video_db_id, 'complete')
        return True

    def _update_status(self, video_db_id: int, status: str, error: str = None):
        """Update video processing status."""
        cursor = self.db.conn.cursor()
        if status in ('complete', 'failed'):
            cursor.execute(
                "UPDATE youtube_videos SET status = ?, error_message = ?, processed_at = ? WHERE id = ?",
                (status, error, datetime.now().isoformat(), video_db_id)
            )
        else:
            cursor.execute(
                "UPDATE youtube_videos SET status = ? WHERE id = ?",
                (status, video_db_id)
            )
        self.db._commit()

    def get_pending_videos(self, profile_id: int = None) -> List[Dict[str, Any]]:
        """Get all videos with pending status."""
        if not self.db:
            return []
        cursor = self.db.conn.cursor()
        if profile_id:
            cursor.execute(
                "SELECT * FROM youtube_videos WHERE status = 'pending' AND profile_id = ? ORDER BY published_at DESC",
                (profile_id,)
            )
        else:
            cursor.execute(
                "SELECT * FROM youtube_videos WHERE status = 'pending' ORDER BY published_at DESC"
            )
        return [dict(r) for r in cursor.fetchall()]

    def get_video_insights(self, video_db_id: int, profile_id: int,
                           language: str = 'en') -> List[Dict[str, Any]]:
        """Get insights for a video, optionally filtered by language.

        Args:
            language: 'en', 'ar', 'ja', or 'all' for all languages.
        """
        if not self.db:
            return []
        cursor = self.db.conn.cursor()
        if language == 'all':
            cursor.execute(
                "SELECT * FROM video_insights WHERE video_id = ? AND profile_id = ? ORDER BY lens_name, language",
                (video_db_id, profile_id)
            )
        else:
            cursor.execute(
                "SELECT * FROM video_insights WHERE video_id = ? AND profile_id = ? AND language = ? ORDER BY lens_name",
                (video_db_id, profile_id, language)
            )
        results = []
        for row in cursor.fetchall():
            d = dict(row)
            try:
                d['content'] = json.loads(d['content'])
            except (json.JSONDecodeError, TypeError):
                pass
            results.append(d)
        return results

    def search_insights(self, profile_id: int, query: str, lens_name: str = None,
                        limit: int = 20) -> List[Dict[str, Any]]:
        """Search across all extracted insights for a profile."""
        if not self.db:
            return []
        cursor = self.db.conn.cursor()
        if lens_name:
            cursor.execute(
                """SELECT i.*, v.title as video_title, v.video_id as yt_video_id,
                          c.channel_name
                   FROM video_insights i
                   JOIN youtube_videos v ON i.video_id = v.id
                   JOIN youtube_channels c ON v.channel_id = c.id
                   WHERE i.profile_id = ? AND i.lens_name = ? AND i.content LIKE ?
                   ORDER BY i.created_at DESC LIMIT ?""",
                (profile_id, lens_name, f'%{query}%', limit)
            )
        else:
            cursor.execute(
                """SELECT i.*, v.title as video_title, v.video_id as yt_video_id,
                          c.channel_name
                   FROM video_insights i
                   JOIN youtube_videos v ON i.video_id = v.id
                   JOIN youtube_channels c ON v.channel_id = c.id
                   WHERE i.profile_id = ? AND i.content LIKE ?
                   ORDER BY i.created_at DESC LIMIT ?""",
                (profile_id, f'%{query}%', limit)
            )
        results = []
        for row in cursor.fetchall():
            d = dict(row)
            try:
                d['content'] = json.loads(d['content'])
            except (json.JSONDecodeError, TypeError):
                pass
            results.append(d)
        return results

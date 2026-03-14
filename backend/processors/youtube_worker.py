"""
YouTube Background Worker — Processes video queue (transcription + lens extraction).

Runs as a daemon thread alongside the server. Processes one video at a time
(Whisper CPU uses all cores — don't parallelize).

Usage:
    from processors.youtube_worker import start_youtube_worker
    start_youtube_worker(strat, sse_manager)
"""

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

_worker_thread: Optional[threading.Thread] = None
_stop_flag = threading.Event()


def start_youtube_worker(strat, sse_manager=None):
    """Start the background video processing worker."""
    global _worker_thread

    if _worker_thread and _worker_thread.is_alive():
        logger.info("YouTube worker already running")
        return

    _stop_flag.clear()
    _worker_thread = threading.Thread(
        target=_worker_loop,
        args=(strat, sse_manager),
        daemon=True,
        name="youtube-worker",
    )
    _worker_thread.start()
    logger.info("YouTube background worker started")


def stop_youtube_worker():
    """Signal the worker to stop."""
    _stop_flag.set()


def _worker_loop(strat, sse_manager):
    """Main worker loop — polls for pending videos and processes them."""
    from processors.youtube import YouTubeProcessor

    _backoff = 30  # seconds between polls (increases on lock errors)

    # Worker uses its own DB path — never touches strat.db to avoid lock contention
    db_path = str(strat.db.db_path) if hasattr(strat, 'db') and strat.db and hasattr(strat.db, 'db_path') else None

    # Wait for server to fully initialize before processing
    _stop_flag.wait(15)
    if _stop_flag.is_set():
        return

    while not _stop_flag.is_set():
        try:
            if not db_path:
                time.sleep(30)
                continue

            # Use a short-lived cursor for the query — don't hold connection open during transcription
            with sqlite3.connect(db_path) as poll_conn:
                poll_conn.row_factory = sqlite3.Row
                poll_conn.execute("PRAGMA busy_timeout = 5000")
                poll_cursor = poll_conn.cursor()
                poll_cursor.execute(
                    """SELECT v.id, v.video_id, v.title, v.profile_id, v.channel_id,
                              c.lenses, c.channel_name
                       FROM youtube_videos v
                       JOIN youtube_channels c ON v.channel_id = c.id
                       WHERE v.status = 'pending'
                       ORDER BY v.published_at ASC
                       LIMIT 1"""
                )
                row = poll_cursor.fetchone()

            if not row:
                _stop_flag.wait(_backoff)
                continue

            video = dict(row)
            video_db_id = video['id']
            video_id = video['video_id']
            profile_id = video['profile_id']
            title = video.get('title', video_id)

            logger.info(f"YouTube worker: processing '{title}' ({video_id})")

            # Mark as transcribing (quick DB touch, then release)
            with sqlite3.connect(db_path) as status_conn:
                status_conn.execute("PRAGMA busy_timeout = 5000")
                status_conn.execute(
                    "UPDATE youtube_videos SET status = 'transcribing' WHERE id = ?",
                    (video_db_id,)
                )
                status_conn.commit()

            _notify(sse_manager, {
                'type': 'youtube_processing',
                'video_id': video_id,
                'title': title,
                'status': 'transcribing',
            })

            # Transcription — long operation, NO DB connection held
            # Use standalone get_transcript() — no DB needed for transcription itself
            from processors.youtube import get_transcript
            try:
                transcript, method, detected_lang = get_transcript(
                    video_id, preferred_lang='',
                    supadata_key=strat.config.get('search', {}).get('supadata_api_key', ''),
                )
            except RuntimeError as e:
                with sqlite3.connect(db_path) as err_conn:
                    err_conn.execute("PRAGMA busy_timeout = 5000")
                    err_conn.execute(
                        "UPDATE youtube_videos SET status = 'failed', error_message = ? WHERE id = ?",
                        (str(e)[:500], video_db_id)
                    )
                    err_conn.commit()
                _notify(sse_manager, {
                    'type': 'youtube_processing',
                    'video_id': video_id,
                    'title': title,
                    'status': 'failed',
                    'error': str(e),
                })
                logger.error(f"YouTube worker: transcription failed for {video_id}: {e}")
                continue

            # Save transcript — quick DB touch
            with sqlite3.connect(db_path) as save_conn:
                save_conn.execute("PRAGMA busy_timeout = 5000")
                save_conn.execute(
                    "UPDATE youtube_videos SET transcript_text = ?, transcript_method = ?, transcript_language = ? WHERE id = ?",
                    (transcript, method, detected_lang, video_db_id)
                )
                save_conn.commit()
            logger.info(f"YouTube worker: transcribed {video_id} via {method} ({len(transcript)} chars) [lang={detected_lang}]")

            # Store transcript lens + mark as transcribed — quick DB touch
            insights_count = 0
            with sqlite3.connect(db_path) as lens_conn:
                lens_conn.execute("PRAGMA busy_timeout = 5000")
                lens_conn.execute(
                    """INSERT INTO video_insights
                       (video_id, profile_id, lens_name, content, language)
                       VALUES (?, ?, 'transcript', ?, ?)""",
                    (video_db_id, profile_id,
                     json.dumps({"transcript": transcript}, ensure_ascii=False),
                     detected_lang)
                )
                insights_count += 1
                if detected_lang and detected_lang != 'en':
                    lens_conn.execute(
                        """INSERT INTO video_insights
                           (video_id, profile_id, lens_name, content, language)
                           VALUES (?, ?, 'transcript', ?, 'en')""",
                        (video_db_id, profile_id,
                         json.dumps({"transcript": "", "original_language": detected_lang,
                                     "translation_available": False,
                                     "note": f"Transcript is in {detected_lang}. View the {detected_lang} tab for content."}, ensure_ascii=False))
                    )
                    insights_count += 1
                lens_conn.execute(
                    "UPDATE youtube_videos SET status = 'transcribed' WHERE id = ?",
                    (video_db_id,)
                )
                lens_conn.commit()
            _notify(sse_manager, {
                'type': 'youtube_processing',
                'video_id': video_id,
                'title': title,
                'status': 'transcribed',
                'insights_count': insights_count,
                'transcript_method': method,
            })
            logger.info(f"YouTube worker: transcribed '{title}' — ready for on-demand lens extraction")

            _backoff = 30  # Reset backoff on success

        except Exception as e:
            if 'database is locked' in str(e):
                _backoff = min(_backoff * 2, 120)  # Exponential backoff up to 2 min
                logger.debug(f"YouTube worker: DB locked, backing off {_backoff}s")
            else:
                logger.error(f"YouTube worker error: {e}")
                _backoff = 30
            _stop_flag.wait(_backoff)

    logger.info("YouTube background worker stopped")


def _notify(sse_manager, event: dict):
    """Send SSE notification if manager available."""
    if sse_manager:
        try:
            event_type = event.get('type', 'youtube_processing')
            payload = {k: v for k, v in event.items() if k != 'type'}
            sse_manager.broadcast(event_type, payload)
        except Exception:
            pass

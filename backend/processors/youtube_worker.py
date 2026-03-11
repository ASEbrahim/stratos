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
    from processors.verification import verify_narrations_batch

    while not _stop_flag.is_set():
        try:
            db = strat.db
            if not db:
                time.sleep(30)
                continue

            cursor = db.conn.cursor()
            cursor.execute(
                """SELECT v.id, v.video_id, v.title, v.profile_id, v.channel_id,
                          c.lenses, c.channel_name
                   FROM youtube_videos v
                   JOIN youtube_channels c ON v.channel_id = c.id
                   WHERE v.status = 'pending'
                   ORDER BY v.published_at ASC
                   LIMIT 1"""
            )
            row = cursor.fetchone()

            if not row:
                _stop_flag.wait(30)  # Sleep 30s, but wake immediately if stopped
                continue

            video = dict(row)
            video_db_id = video['id']
            video_id = video['video_id']
            profile_id = video['profile_id']
            title = video.get('title', video_id)

            logger.info(f"YouTube worker: processing '{title}' ({video_id})")

            yt = YouTubeProcessor(strat.config, db=db)

            # Notify start
            _notify(sse_manager, {
                'type': 'youtube_processing',
                'video_id': video_id,
                'title': title,
                'status': 'started',
            })

            # Step 1: Transcribe
            yt._update_status(video_db_id, 'transcribing')
            _notify(sse_manager, {
                'type': 'youtube_processing',
                'video_id': video_id,
                'title': title,
                'status': 'transcribing',
            })

            try:
                transcript, method = yt.get_transcript_for_video(video_id)
            except RuntimeError as e:
                yt._update_status(video_db_id, 'failed', str(e))
                _notify(sse_manager, {
                    'type': 'youtube_processing',
                    'video_id': video_id,
                    'title': title,
                    'status': 'failed',
                    'error': str(e),
                })
                logger.error(f"YouTube worker: transcription failed for {video_id}: {e}")
                continue

            # Save transcript
            cursor.execute(
                "UPDATE youtube_videos SET transcript_text = ?, transcript_method = ? WHERE id = ?",
                (transcript, method, video_db_id)
            )
            db._commit()
            logger.info(f"YouTube worker: transcribed {video_id} via {method} ({len(transcript)} chars)")

            # Step 2: Run lenses
            yt._update_status(video_db_id, 'extracting')
            _notify(sse_manager, {
                'type': 'youtube_processing',
                'video_id': video_id,
                'title': title,
                'status': 'extracting',
            })

            lenses = json.loads(video.get('lenses', '["summary"]'))
            insights_count = 0

            for lens_name in lenses:
                if _stop_flag.is_set():
                    break
                try:
                    from processors.lenses import extract_lens
                    insight = extract_lens(
                        transcript, lens_name, title,
                        yt.ollama_host, yt.inference_model,
                    )
                    if insight:
                        cursor.execute(
                            """INSERT INTO video_insights
                               (video_id, profile_id, lens_name, content)
                               VALUES (?, ?, ?, ?)""",
                            (video_db_id, profile_id, lens_name,
                             json.dumps(insight, ensure_ascii=False))
                        )
                        insights_count += 1
                        logger.info(f"YouTube worker: lens '{lens_name}' extracted for {video_id}")
                except Exception as e:
                    logger.error(f"YouTube worker: lens '{lens_name}' failed: {e}")

            db._commit()

            # Step 3: Verify narrations if applicable
            if 'narrations' in lenses and not _stop_flag.is_set():
                try:
                    cursor.execute(
                        "SELECT id, content FROM video_insights WHERE video_id = ? AND lens_name = 'narrations'",
                        (video_db_id,)
                    )
                    narr_row = cursor.fetchone()
                    if narr_row:
                        narr_data = dict(narr_row)
                        narrations = json.loads(narr_data['content'])
                        if isinstance(narrations, list) and narrations:
                            verified = verify_narrations_batch(narrations, strat.config)
                            cursor.execute(
                                "UPDATE video_insights SET content = ? WHERE id = ?",
                                (json.dumps(verified, ensure_ascii=False), narr_data['id'])
                            )
                            db._commit()
                            logger.info(f"YouTube worker: verified {len(verified)} narrations for {video_id}")
                except Exception as e:
                    logger.error(f"YouTube worker: narration verification failed: {e}")

            # Mark complete
            yt._update_status(video_db_id, 'complete')
            _notify(sse_manager, {
                'type': 'youtube_processing',
                'video_id': video_id,
                'title': title,
                'status': 'complete',
                'insights_count': insights_count,
                'transcript_method': method,
            })
            logger.info(f"YouTube worker: completed '{title}' — {insights_count} insights extracted")

        except Exception as e:
            logger.error(f"YouTube worker error: {e}")
            time.sleep(10)  # Back off on error

    logger.info("YouTube background worker stopped")


def _notify(sse_manager, event: dict):
    """Send SSE notification if manager available."""
    if sse_manager:
        try:
            sse_manager.broadcast(event)
        except Exception:
            pass

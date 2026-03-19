"""
YouTube routes — channel CRUD, video listing, insights, processing.
Extracted from server.py (Sprint 5K Phase 1).
"""

import json
import logging
import threading
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger("STRAT_OS")


def _merge_lens_content(lens_name, existing, new_data):
    """Merge new lens extraction into existing content.

    - summary: replace entirely (re-summarize)
    - eloquence: append new terms only (deduplicate by term)
    - narrations: append new narrations only (deduplicate by narration_text)
    - history/spiritual/politics: append new items (deduplicate by primary key)
    """
    if lens_name == 'summary':
        return new_data  # Full replace

    # Array-based lenses — deduplicate and append
    old_items = existing if isinstance(existing, list) else []
    new_items = new_data if isinstance(new_data, list) else []

    if not new_items:
        return old_items or existing

    # Pick dedup key per lens
    dedup_keys = {
        'eloquence': ('term', 'word'),
        'narrations': ('narration_text', 'text', 'narration'),
        'history': ('event',),
        'spiritual': ('lesson',),
        'politics': ('topic',),
    }
    keys = dedup_keys.get(lens_name, ())

    def _get_key(item):
        for k in keys:
            v = item.get(k, '').strip().lower() if isinstance(item.get(k), str) else ''
            if v:
                return v
        return None

    existing_keys = set()
    for item in old_items:
        k = _get_key(item)
        if k:
            existing_keys.add(k)

    merged = list(old_items)
    for item in new_items:
        k = _get_key(item)
        if k and k not in existing_keys:
            merged.append(item)
            existing_keys.add(k)
        elif not k:
            merged.append(item)

    return merged


def _send_json(handler, data, status=200):
    """Send a JSON response with proper headers."""
    body = json.dumps(data).encode()
    handler.send_response(status)
    handler.send_header("Content-type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def handle_get(handler, strat, auth, path):
    """Handle GET requests for YouTube routes. Returns True if handled."""

    if not path.startswith("/api/youtube/"):
        return False

    from processors.youtube import YouTubeProcessor
    yt = YouTubeProcessor(strat.config, db=strat.db)
    parsed = urlparse(handler.path)
    path_parts = parsed.path.strip('/').split('/')

    if parsed.path == '/api/youtube/channels':
        channels = yt.list_channels(handler._profile_id)
        _send_json(handler, {"channels": channels})
        return True
    elif parsed.path == '/api/youtube/status':
        pending = yt.get_pending_videos(handler._profile_id)
        _send_json(handler, {"pending_count": len(pending), "pending": pending[:5]})
        return True
    elif len(path_parts) == 4 and path_parts[2] == 'videos':
        # /api/youtube/videos/:channel_id
        try:
            ch_id = int(path_parts[3])
            cursor = strat.db.conn.cursor()
            cursor.execute(
                "SELECT * FROM youtube_videos WHERE channel_id = ? AND profile_id = ? ORDER BY pinned DESC, published_at DESC",
                (ch_id, handler._profile_id)
            )
            _send_json(handler, {"videos": [dict(r) for r in cursor.fetchall()]})
        except (ValueError, IndexError):
            _send_json(handler, {"error": "Invalid channel ID"}, 400)
        return True
    elif len(path_parts) == 4 and path_parts[2] == 'insights':
        # /api/youtube/insights/:video_db_id?language=en|ar|ja|all
        try:
            vid_id = int(path_parts[3])
            qs = parse_qs(parsed.query)
            language = qs.get('language', ['en'])[0]
            insights = yt.get_video_insights(vid_id, handler._profile_id, language=language)

            # Get available languages and video title for this video
            cursor = strat.db.conn.cursor()
            cursor.execute(
                "SELECT DISTINCT language FROM video_insights WHERE video_id = ? AND profile_id = ?",
                (vid_id, handler._profile_id)
            )
            available_langs = [row['language'] for row in cursor.fetchall()]
            cursor.execute(
                "SELECT title, video_id, transcript_language, transcript_text FROM youtube_videos WHERE id = ? AND profile_id = ?",
                (vid_id, handler._profile_id)
            )
            vrow = cursor.fetchone()
            video_title = vrow['title'] if vrow else ''
            yt_video_id = vrow['video_id'] if vrow else ''
            transcript_lang = (vrow['transcript_language'] if vrow else 'en') or 'en'
            transcript_text = vrow['transcript_text'] if vrow else None

            # Enrich narration insights with resolved source URLs
            resolved_sources = {}
            try:
                cursor.execute(
                    "SELECT narration_hash, resolved_url, resolution_method, confidence "
                    "FROM narration_sources WHERE video_id = ? AND profile_id = ?",
                    (vid_id, handler._profile_id)
                )
                for srow in cursor.fetchall():
                    resolved_sources[srow['narration_hash']] = {
                        'url': srow['resolved_url'],
                        'method': srow['resolution_method'],
                        'confidence': srow['confidence'],
                    }
            except Exception as e:
                logger.debug(f"narration_sources query failed (table may not exist): {e}")

            # Attach resolved URLs to narration items
            if resolved_sources:
                import hashlib
                for ins in insights:
                    if ins.get('lens_name') != 'narrations':
                        continue
                    content = ins.get('content')
                    items = content if isinstance(content, list) else []
                    for item in items:
                        text = item.get('narration_text', '') or item.get('text', '') or item.get('narration', '')
                        if not text:
                            continue
                        nar_hash = hashlib.sha256(text.strip().lower().encode()).hexdigest()[:32]
                        if nar_hash in resolved_sources:
                            item['_resolved'] = resolved_sources[nar_hash]

            _send_json(handler, {
                "insights": insights,
                "video_title": video_title,
                "yt_video_id": yt_video_id,
                "available_languages": available_langs,
                "transcript_language": transcript_lang,
                "transcript_text": transcript_text,
                "current_language": language,
            })
        except (ValueError, IndexError):
            _send_json(handler, {"error": "Invalid video ID"}, 400)
        return True

    elif len(path_parts) == 4 and path_parts[2] == 'captions':
        # GET /api/youtube/captions/:video_db_id — fetch timed YouTube captions
        try:
            vid_id = int(path_parts[3])
            cursor = strat.db.conn.cursor()
            cursor.execute(
                "SELECT video_id, title, transcript_language FROM youtube_videos WHERE id = ? AND profile_id = ?",
                (vid_id, handler._profile_id)
            )
            row = cursor.fetchone()
            if not row:
                _send_json(handler, {"error": "Video not found"}, 404)
                return True

            yt_video_id = row[0]
            try:
                import requests as _req
                from youtube_transcript_api import YouTubeTranscriptApi
                _worker_base = strat.config.get("proxy", {}).get("cloudflare_worker", "")
                target_lang = parse_qs(parsed.query).get('lang', [row[2] or 'en'])[0]

                api = YouTubeTranscriptApi()
                transcript_list = api.list(yt_video_id)
                tracks = [{"language": t.language_code, "language_name": t.language, "is_generated": t.is_generated} for t in transcript_list]

                # Find best track
                preferred = [target_lang, 'en', 'ar', 'ja', 'ko', 'fr', 'de', 'es']
                found = None
                try:
                    found = transcript_list.find_transcript(preferred)
                except Exception:
                    for t in transcript_list:
                        found = t
                        break

                captions = []
                if found:
                    target_lang = found.language_code
                    try:
                        # Try direct fetch first
                        result = found.fetch()
                        for snippet in result:
                            captions.append({"start": round(snippet.start, 2), "duration": round(snippet.duration, 2), "text": snippet.text})
                    except Exception as _direct_err:
                        # IP blocked — proxy through CF Worker using the track's HTTP URL
                        if _worker_base and hasattr(found, '_http_client') and hasattr(found, '_url'):
                            try:
                                _timedtext_url = found._url
                                _proxy_url = f"{_worker_base}/captions?url={requests.utils.quote(_timedtext_url, safe='')}"
                                _proxy_resp = _req.get(_proxy_url, timeout=15)
                                if _proxy_resp.status_code == 200:
                                    _proxy_data = _proxy_resp.json()
                                    captions = _proxy_data.get('captions', [])
                            except Exception as _proxy_err:
                                logger.warning(f"CF proxy caption fetch failed: {_proxy_err}")

                _send_json(handler, {
                    "video_id": yt_video_id, "title": row[1], "language": target_lang,
                    "tracks": tracks, "captions": captions, "count": len(captions),
                })
            except Exception as e:
                logger.warning(f"Caption fetch failed for {yt_video_id}: {e}")
                _send_json(handler, {"error": f"No captions available: {e}", "tracks": []}, 404)
        except (ValueError, IndexError):
            _send_json(handler, {"error": "Invalid video ID"}, 400)
        return True

    elif len(path_parts) == 4 and path_parts[2] == 'export':
        # GET /api/youtube/export/:channel_db_id?format=json|md
        try:
            ch_id = int(path_parts[3])
            qs = parse_qs(parsed.query)
            fmt = qs.get('format', ['json'])[0]
            cursor = strat.db.conn.cursor()

            # Get channel info
            cursor.execute(
                "SELECT * FROM youtube_channels WHERE id = ? AND profile_id = ?",
                (ch_id, handler._profile_id)
            )
            ch_row = cursor.fetchone()
            if not ch_row:
                _send_json(handler, {"error": "Channel not found"}, 404)
                return True
            channel = dict(ch_row)

            # Get all videos
            cursor.execute(
                "SELECT id, video_id, title, status, transcript_text, transcript_language, published_at "
                "FROM youtube_videos WHERE channel_id = ? AND profile_id = ? ORDER BY published_at DESC",
                (ch_id, handler._profile_id)
            )
            videos = [dict(r) for r in cursor.fetchall()]

            # Get all insights for each video
            for video in videos:
                cursor.execute(
                    "SELECT lens_name, content, language FROM video_insights "
                    "WHERE video_id = ? AND profile_id = ? ORDER BY lens_name, language",
                    (video['id'], handler._profile_id)
                )
                video['insights'] = {}
                for row in cursor.fetchall():
                    key = f"{row['lens_name']}_{row['language']}" if row['language'] else row['lens_name']
                    try:
                        video['insights'][key] = json.loads(row['content'])
                    except (json.JSONDecodeError, TypeError):
                        video['insights'][key] = row['content']

            channel_name = channel.get('channel_name', 'channel')

            if fmt == 'md':
                # Markdown export
                lines = [f"# {channel_name}\n"]
                for v in videos:
                    yt_url = f"https://www.youtube.com/watch?v={v['video_id']}"
                    lines.append(f"\n## {v['title']}\n")
                    lines.append(f"- **URL**: {yt_url}")
                    lines.append(f"- **Status**: {v['status']}")
                    if v.get('published_at'):
                        lines.append(f"- **Published**: {v['published_at']}")
                    lines.append(f"- **Language**: {v.get('transcript_language', 'en')}\n")

                    if v.get('transcript_text'):
                        lines.append("### Transcript\n")
                        lines.append(v['transcript_text'][:50000] + "\n")

                    for lens_key, content in v.get('insights', {}).items():
                        lens_label = lens_key.replace('_', ' — ', 1).title()
                        lines.append(f"### {lens_label}\n")
                        if isinstance(content, dict):
                            if content.get('summary'):
                                lines.append(content['summary'] + "\n")
                            if content.get('key_takeaways'):
                                for i, t in enumerate(content['key_takeaways'], 1):
                                    text = t if isinstance(t, str) else (t.get('text', '') or t.get('point', ''))
                                    lines.append(f"{i}. {text}")
                                lines.append("")
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict):
                                    primary = (item.get('narration_text') or item.get('term') or
                                               item.get('event') or item.get('lesson') or
                                               item.get('topic') or item.get('text') or '')
                                    lines.append(f"- {primary}")
                                else:
                                    lines.append(f"- {item}")
                            lines.append("")
                        elif isinstance(content, str):
                            lines.append(content + "\n")

                body = '\n'.join(lines).encode('utf-8')
                handler.send_response(200)
                handler.send_header("Content-type", "text/markdown; charset=utf-8")
                handler.send_header("Content-Disposition",
                                    f'attachment; filename="{channel_name.replace(" ", "_")}_export.md"')
                handler.send_header("Access-Control-Allow-Origin", "*")
                handler.end_headers()
                handler.wfile.write(body)
            else:
                # JSON export
                export = {
                    "channel": channel_name,
                    "channel_id": channel.get('channel_id', ''),
                    "exported_at": __import__('datetime').datetime.now().isoformat(),
                    "video_count": len(videos),
                    "videos": [{
                        "title": v['title'],
                        "youtube_url": f"https://www.youtube.com/watch?v={v['video_id']}",
                        "youtube_id": v['video_id'],
                        "status": v['status'],
                        "published_at": v.get('published_at'),
                        "language": v.get('transcript_language', 'en'),
                        "transcript": v.get('transcript_text', ''),
                        "insights": v.get('insights', {}),
                    } for v in videos],
                }
                body = json.dumps(export, ensure_ascii=False, indent=2).encode('utf-8')
                handler.send_response(200)
                handler.send_header("Content-type", "application/json; charset=utf-8")
                handler.send_header("Content-Disposition",
                                    f'attachment; filename="{channel_name.replace(" ", "_")}_export.json"')
                handler.send_header("Access-Control-Allow-Origin", "*")
                handler.end_headers()
                handler.wfile.write(body)
        except (ValueError, IndexError):
            _send_json(handler, {"error": "Invalid channel ID"}, 400)
        return True

    return False


def handle_post(handler, strat, auth, path):
    """Handle POST requests for YouTube routes. Returns True if handled."""

    if not path.startswith("/api/youtube/"):
        return False

    from processors.youtube import YouTubeProcessor
    yt = YouTubeProcessor(strat.config, db=strat.db)
    _cl = int(handler.headers.get('Content-Length', 0))
    body = json.loads(handler.rfile.read(_cl).decode()) if _cl > 0 else {}

    if path == "/api/youtube/channels":
        channel_input = (body.get('channel') or body.get('channel_url') or '').strip()
        lenses = body.get('lenses', ['transcript'])
        if not channel_input:
            _send_json(handler, {"error": "No channel URL/handle provided"}, 400)
            return True

        # Detect video URL vs channel URL
        from processors.youtube import parse_youtube_input
        parsed = parse_youtube_input(channel_input)

        if parsed['type'] == 'video':
            result = yt.add_single_video(handler._profile_id, parsed['id'], lenses)
            if result:
                _send_json(handler, {"ok": True, "video": result, "type": "video"})
            else:
                _send_json(handler, {"error": "Could not add video"}, 400)
        else:
            result = yt.add_channel(handler._profile_id, channel_input, lenses)
            if result:
                _send_json(handler, {"ok": True, "channel": result})
            else:
                _send_json(handler, {"error": "Could not resolve channel"}, 400)
        return True

    path_parts = path.strip('/').split('/')
    if len(path_parts) == 4 and path_parts[2] == 'process':
        # POST /api/youtube/process/:channel_db_id
        # If reprocess=true, wipe existing transcripts and re-queue all videos
        try:
            ch_id = int(path_parts[3])
            reprocess = body.get('reprocess', False)

            if reprocess:
                import sqlite3 as _sq
                _db_path = str(strat.db.db_path)
                with _sq.connect(_db_path) as _rpc:
                    _rpc.execute("PRAGMA busy_timeout = 5000")
                    _rpc.execute(
                        """DELETE FROM video_insights WHERE video_id IN
                           (SELECT id FROM youtube_videos WHERE channel_id = ? AND profile_id = ?)""",
                        (ch_id, handler._profile_id)
                    )
                    _reset_cur = _rpc.execute(
                        """UPDATE youtube_videos SET status = 'pending', transcript_text = NULL,
                           transcript_method = NULL, transcript_language = NULL, error_message = NULL
                           WHERE channel_id = ? AND profile_id = ?""",
                        (ch_id, handler._profile_id)
                    )
                    reset_count = _reset_cur.rowcount
                    _rpc.commit()
                logger.info(f"Re-process channel {ch_id}: reset {reset_count} videos to pending")
                _send_json(handler, {"ok": True, "reset": reset_count, "new_videos": 0})
            else:
                new_videos = yt.discover_new_videos(handler._profile_id, ch_id)
                _send_json(handler, {"ok": True, "new_videos": len(new_videos), "videos": new_videos[:10]})
        except (ValueError, IndexError):
            _send_json(handler, {"error": "Invalid channel ID"}, 400)
        return True

    if len(path_parts) == 4 and path_parts[2] == 'extract-all':
        # POST /api/youtube/extract-all/:channel_db_id
        # Queues lens extraction for all transcribed videos in the channel
        try:
            ch_id = int(path_parts[3])
            cursor = strat.db.conn.cursor()
            # Get all transcribed/complete videos that have transcripts
            cursor.execute(
                """SELECT id, transcript_text, title, profile_id, transcript_language
                   FROM youtube_videos
                   WHERE channel_id = ? AND profile_id = ? AND transcript_text IS NOT NULL AND transcript_text != ''""",
                (ch_id, handler._profile_id)
            )
            videos = [dict(r) for r in cursor.fetchall()]
            if not videos:
                _send_json(handler, {"ok": True, "queued": 0, "message": "No transcribed videos found"})
                return True

            # Get channel lenses config
            cursor.execute(
                "SELECT lenses FROM youtube_channels WHERE id = ? AND profile_id = ?",
                (ch_id, handler._profile_id)
            )
            ch_row = cursor.fetchone()
            channel_lenses = ['summary', 'eloquence', 'narrations', 'history', 'spiritual', 'politics']
            if ch_row and ch_row['lenses']:
                try:
                    parsed = json.loads(ch_row['lenses'])
                    # Filter to only LLM lenses (not transcript)
                    channel_lenses = [l for l in parsed if l != 'transcript'] or channel_lenses
                except (json.JSONDecodeError, TypeError):
                    pass

            from processors.lenses import extract_lens, AVAILABLE_LENSES
            queued = 0

            def _extract_all_background():
                import sqlite3 as _sq
                nonlocal queued
                _profile = handler._profile_id
                _db_path = str(strat.db.db_path)
                # Use independent DB connection for thread safety
                bg_conn = _sq.connect(_db_path)
                bg_conn.execute("PRAGMA busy_timeout = 5000")
                bg_conn.row_factory = _sq.Row
                try:
                    for video in videos:
                        vid_id = video['id']
                        transcript = video['transcript_text']
                        title = video['title'] or ''
                        lang = video.get('transcript_language') or 'en'
                        # Check which lenses already exist for this video
                        cur = bg_conn.cursor()
                        cur.execute(
                            "SELECT DISTINCT lens_name FROM video_insights WHERE video_id = ? AND profile_id = ?",
                            (vid_id, _profile)
                        )
                        existing = {row['lens_name'] for row in cur.fetchall()}
                        for lens_name in channel_lenses:
                            if lens_name in existing or lens_name not in AVAILABLE_LENSES or lens_name == 'transcript':
                                continue
                            try:
                                result = extract_lens(
                                    transcript, lens_name, title,
                                    yt.ollama_host, yt.inference_model,
                                    target_language=lang if lang != 'en' else 'en',
                                )
                                if result:
                                    cur.execute(
                                        """INSERT INTO video_insights
                                           (video_id, profile_id, lens_name, content, language)
                                           VALUES (?, ?, ?, ?, ?)""",
                                        (vid_id, _profile, lens_name,
                                         json.dumps(result, ensure_ascii=False), lang if lang != 'en' else 'en')
                                    )
                                    bg_conn.commit()
                                    logger.info(f"Extract-all: '{lens_name}' for video {vid_id} ({title[:40]})")
                                    # Trigger source resolution for narrations
                                    if lens_name == 'narrations':
                                        try:
                                            from processors.source_resolver import resolve_sources_async
                                            narration_list = result if isinstance(result, list) else []
                                            if narration_list:
                                                resolve_sources_async(
                                                    vid_id, _profile,
                                                    narration_list, strat.db, strat.config,
                                                    sse_manager=strat.sse if hasattr(strat, 'sse') else None,
                                                )
                                        except Exception as e:
                                            logger.debug(f"Extract-all source resolution failed: {e}")
                            except Exception as e:
                                logger.error(f"Extract-all: lens '{lens_name}' failed for video {vid_id}: {e}")
                    # Broadcast completion
                    if hasattr(strat, 'sse') and strat.sse:
                        strat.sse.broadcast('extract_all_complete', {
                            'channel_id': ch_id,
                        })
                    logger.info(f"Extract-all complete for channel {ch_id}: processed {len(videos)} videos")
                finally:
                    bg_conn.close()

            # Count how many video-lens pairs will be extracted
            cursor_check = strat.db.conn.cursor()
            for video in videos:
                cursor_check.execute(
                    "SELECT DISTINCT lens_name FROM video_insights WHERE video_id = ? AND profile_id = ?",
                    (video['id'], handler._profile_id)
                )
                existing = {row['lens_name'] for row in cursor_check.fetchall()}
                for lens_name in channel_lenses:
                    if lens_name not in existing and lens_name in AVAILABLE_LENSES and lens_name != 'transcript':
                        queued += 1

            threading.Thread(target=_extract_all_background, daemon=True).start()
            _send_json(handler, {"ok": True, "queued": queued, "videos": len(videos)})
        except (ValueError, IndexError):
            _send_json(handler, {"error": "Invalid channel ID"}, 400)
        return True

    if path == "/api/youtube/extract-lens":
        video_id = body.get('video_id')
        lens_name = body.get('lens')
        language = body.get('language', 'en')
        mode = body.get('mode', 'new')  # 'new', 'replace', or 'merge'
        if not video_id or not lens_name:
            _send_json(handler, {"error": "video_id and lens required"}, 400)
            return True
        from processors.lenses import AVAILABLE_LENSES
        if lens_name not in AVAILABLE_LENSES or lens_name == 'transcript':
            _send_json(handler, {"error": f"Invalid lens: {lens_name}"}, 400)
            return True
        cursor = strat.db.conn.cursor()
        cursor.execute(
            "SELECT id, transcript_text, title, profile_id, transcript_language FROM youtube_videos WHERE id = ? AND profile_id = ?",
            (int(video_id), handler._profile_id)
        )
        vrow = cursor.fetchone()
        if not vrow:
            _send_json(handler, {"error": "Video not found. It may have been deleted."}, 404)
            return True
        if not vrow['transcript_text']:
            _send_json(handler, {"error": "No transcript available for this video. Transcribe it first by clicking the Transcribe button."}, 404)
            return True
        video = dict(vrow)

        # For merge/replace, load existing content
        existing_content = None
        if mode in ('replace', 'merge'):
            cursor.execute(
                "SELECT content FROM video_insights WHERE video_id = ? AND profile_id = ? AND lens_name = ? AND language = ?",
                (int(video_id), handler._profile_id, lens_name, language)
            )
            erow = cursor.fetchone()
            if erow:
                try:
                    existing_content = json.loads(erow['content'])
                except (json.JSONDecodeError, TypeError):
                    existing_content = None

        _send_json(handler, {"ok": True, "status": "extracting"})

        def _extract_in_background():
            import sqlite3 as _sq
            _db_path = str(strat.db.db_path)
            try:
                from processors.lenses import extract_lens
                insight = extract_lens(
                    video['transcript_text'], lens_name, video['title'],
                    yt.ollama_host, yt.inference_model,
                    target_language=language,
                )
                if insight:
                    final = insight
                    if mode == 'merge' and existing_content is not None:
                        final = _merge_lens_content(lens_name, existing_content, insight)
                    # Use independent DB connection for thread safety
                    with _sq.connect(_db_path) as bg_conn:
                        bg_conn.execute("PRAGMA busy_timeout = 5000")
                        cur = bg_conn.cursor()
                        if mode in ('replace', 'merge') and existing_content is not None:
                            cur.execute(
                                """UPDATE video_insights SET content = ?
                                   WHERE video_id = ? AND profile_id = ? AND lens_name = ? AND language = ?""",
                                (json.dumps(final, ensure_ascii=False),
                                 video['id'], video['profile_id'], lens_name, language)
                            )
                        else:
                            cur.execute(
                                """INSERT INTO video_insights
                                   (video_id, profile_id, lens_name, content, language)
                                   VALUES (?, ?, ?, ?, ?)""",
                                (video['id'], video['profile_id'], lens_name,
                                 json.dumps(final, ensure_ascii=False), language)
                            )
                        bg_conn.commit()
                    if hasattr(strat, 'sse') and strat.sse:
                        strat.sse.broadcast('lens_extracted', {
                            'video_id': video['id'],
                            'lens': lens_name,
                            'language': language,
                        })
                    logger.info(f"On-demand lens '{lens_name}' [{language}] {mode} for video {video['id']}")

                    # Trigger narration source resolution after narrations extraction
                    if lens_name == 'narrations' and final:
                        try:
                            from processors.source_resolver import resolve_sources_async
                            narration_list = final if isinstance(final, list) else []
                            if narration_list:
                                resolve_sources_async(
                                    video['id'], video['profile_id'],
                                    narration_list, strat.db, strat.config,
                                    sse_manager=strat.sse if hasattr(strat, 'sse') else None,
                                )
                        except Exception as e:
                            logger.debug(f"Narration source resolution trigger failed: {e}")
            except Exception as e:
                logger.error(f"On-demand lens extraction failed: {e}")

        threading.Thread(target=_extract_in_background, daemon=True).start()
        return True

    if path == "/api/youtube/cancel-transcribe":
        video_id = body.get('video_id')
        if not video_id:
            _send_json(handler, {"error": "video_id required"}, 400)
            return True
        try:
            import sqlite3 as _sq
            _db_path = str(strat.db.db_path)
            with _sq.connect(_db_path) as _cc:
                _cc.execute("PRAGMA busy_timeout = 5000")
                cur = _cc.execute(
                    "UPDATE youtube_videos SET status = 'pending', transcript_text = NULL, error_message = NULL "
                    "WHERE id = ? AND profile_id = ? AND status IN ('transcribing', 'extracting', 'processing', 'failed', 'low_quality')",
                    (int(video_id), handler._profile_id)
                )
                if cur.rowcount > 0:
                    _cc.execute(
                        "DELETE FROM video_insights WHERE video_id = ? AND profile_id = ?",
                        (int(video_id), handler._profile_id)
                    )
                    _cc.commit()
                    _send_json(handler, {"ok": True, "status": "cancelled"})
                else:
                    _send_json(handler, {"error": "Video not found or not in cancellable state"}, 404)
        except Exception as e:
            _send_json(handler, {"error": str(e)}, 500)
        return True

    if path == "/api/youtube/retranscribe":
        video_id = body.get('video_id')
        if not video_id:
            _send_json(handler, {"error": "video_id required"}, 400)
            return True
        # Use independent connection to avoid lock contention
        import sqlite3 as _sq
        _db_path = str(strat.db.db_path)
        try:
            with _sq.connect(_db_path) as _rc:
                _rc.row_factory = _sq.Row
                _rc.execute("PRAGMA busy_timeout = 5000")
                _rc.execute(
                    "SELECT id, video_id, title, profile_id FROM youtube_videos WHERE id = ? AND profile_id = ?",
                    (int(video_id), handler._profile_id)
                )
                vrow = _rc.execute(
                    "SELECT id, video_id, title, profile_id FROM youtube_videos WHERE id = ? AND profile_id = ?",
                    (int(video_id), handler._profile_id)
                ).fetchone()
                if not vrow:
                    _send_json(handler, {"error": "Video not found"}, 404)
                    return True
                video = dict(vrow)
                _rc.execute("UPDATE youtube_videos SET status = 'pending', transcript_text = NULL WHERE id = ?", (video['id'],))
                _rc.execute("DELETE FROM video_insights WHERE video_id = ? AND profile_id = ?", (video['id'], handler._profile_id))
                _rc.commit()
        except Exception as e:
            _send_json(handler, {"error": str(e)}, 500)
            return True
        _send_json(handler, {"ok": True, "status": "queued"})

        def _retranscribe():
            _db_path = str(strat.db.db_path)
            try:
                from processors.youtube import get_transcript
                text, method, lang = get_transcript(
                    video['video_id'],
                    preferred_lang='en',
                    supadata_key=strat.config.get('search', {}).get('supadata_api_key', ''),
                )
                # Use short-lived connection — don't hold strat.db open in daemon thread
                import sqlite3 as _sq
                with _sq.connect(_db_path) as _c:
                    _c.execute("PRAGMA busy_timeout = 5000")
                    _c.execute(
                        "UPDATE youtube_videos SET transcript_text = ?, transcript_language = ?, status = 'transcribed' WHERE id = ?",
                        (text, lang, video['id'])
                    )
                    _c.commit()
                logger.info(f"Re-transcribed video {video['id']} ({video['title'][:40]}) via {method}: {len(text)} chars")
                if hasattr(strat, 'sse') and strat.sse:
                    strat.sse.broadcast('youtube_processing', {
                        'video_id': video['video_id'],
                        'title': video['title'],
                        'status': 'transcribed',
                    })
            except Exception as e:
                logger.error(f"Re-transcribe failed for video {video['id']}: {e}")
                try:
                    import sqlite3 as _sq
                    with _sq.connect(_db_path) as _c:
                        _c.execute("PRAGMA busy_timeout = 5000")
                        _c.execute("UPDATE youtube_videos SET status = 'failed' WHERE id = ?", (video['id'],))
                        _c.commit()
                except Exception as e2:
                    logger.error(f"Failed to mark video as failed: {e2}")
                if hasattr(strat, 'sse') and strat.sse:
                    strat.sse.broadcast('youtube_processing', {
                        'video_id': video['video_id'],
                        'title': video['title'],
                        'status': 'failed',
                        'error': str(e)[:100],
                    })

        threading.Thread(target=_retranscribe, daemon=True).start()
        return True

    if path == "/api/youtube/pin-video":
        video_id = body.get('video_id')
        pinned = body.get('pinned', True)
        if not video_id:
            _send_json(handler, {"error": "Missing video_id"}, 400)
            return True
        cursor = strat.db.conn.cursor()
        cursor.execute(
            "UPDATE youtube_videos SET pinned = ? WHERE id = ? AND profile_id = ?",
            (1 if pinned else 0, video_id, handler._profile_id)
        )
        strat.db._commit()
        _send_json(handler, {"status": "ok", "pinned": bool(pinned)})
        return True

    if path == "/api/youtube/translate-transcript":
        video_id = body.get('video_id')
        target_lang = body.get('target_language', 'en')
        if not video_id:
            _send_json(handler, {"error": "Missing video_id"}, 400)
            return True

        cursor = strat.db.conn.cursor()
        cursor.execute(
            "SELECT transcript_text, transcript_language FROM youtube_videos WHERE id = ? AND profile_id = ?",
            (video_id, handler._profile_id)
        )
        row = cursor.fetchone()
        if not row or not row[0]:
            _send_json(handler, {"error": "No transcript found"}, 404)
            return True

        transcript_text = row[0]
        source_lang = row[1] or 'unknown'

        # Translation chain: Google Translate (free API) → Argos (offline) → Qwen LLM
        try:
            translated = None
            _method = 'google'
            _src = source_lang if source_lang != 'unknown' else 'auto'

            # --- Tier 1: Google Translate via deep-translator (free, reliable) ---
            try:
                from deep_translator import GoogleTranslator
                # deep-translator handles chunking for long texts internally
                _gt_src = _src if _src != 'unknown' else 'auto'
                translated = GoogleTranslator(source=_gt_src, target=target_lang).translate(transcript_text[:5000])
                if not translated or not translated.strip():
                    raise RuntimeError("Empty Google Translate result")
                logger.info(f"Translation via Google Translate: {_src}→{target_lang} ({len(transcript_text)} chars)")
            except Exception as gt_err:
                logger.warning(f"Google Translate failed ({gt_err}), trying Argos")
                translated = None

            # --- Tier 2: Argos Translate (offline, no network needed) ---
            if not translated:
                _method = 'argos'
                try:
                    import argostranslate.package
                    import argostranslate.translate

                    _argos_src = source_lang if source_lang != 'unknown' else 'en'
                    installed = argostranslate.translate.get_installed_languages()
                    src_lang_obj = next((l for l in installed if l.code == _argos_src), None)
                    tgt_lang_obj = next((l for l in installed if l.code == target_lang), None)

                    if not src_lang_obj or not tgt_lang_obj:
                        logger.info(f"Downloading Argos language package: {_argos_src} → {target_lang}")
                        argostranslate.package.update_package_index()
                        available = argostranslate.package.get_available_packages()
                        pkg = next((p for p in available if p.from_code == _argos_src and p.to_code == target_lang), None)
                        if pkg:
                            argostranslate.package.install_from_path(pkg.download())
                        else:
                            if _argos_src != 'en':
                                pkg_to_en = next((p for p in available if p.from_code == _argos_src and p.to_code == 'en'), None)
                                if pkg_to_en:
                                    argostranslate.package.install_from_path(pkg_to_en.download())
                            if target_lang != 'en':
                                pkg_from_en = next((p for p in available if p.from_code == 'en' and p.to_code == target_lang), None)
                                if pkg_from_en:
                                    argostranslate.package.install_from_path(pkg_from_en.download())
                        installed = argostranslate.translate.get_installed_languages()
                        src_lang_obj = next((l for l in installed if l.code == _argos_src), None)
                        tgt_lang_obj = next((l for l in installed if l.code == target_lang), None)

                    if src_lang_obj and tgt_lang_obj:
                        translation = src_lang_obj.get_translation(tgt_lang_obj)
                        if translation:
                            translated = translation.translate(transcript_text)
                        else:
                            en_obj = next((l for l in installed if l.code == 'en'), None)
                            if en_obj:
                                t1 = src_lang_obj.get_translation(en_obj)
                                t2 = en_obj.get_translation(tgt_lang_obj)
                                if t1 and t2:
                                    translated = t2.translate(t1.translate(transcript_text))
                    if not translated:
                        raise RuntimeError(f"No Argos path for {_argos_src}→{target_lang}")
                except Exception as argos_err:
                    logger.warning(f"Argos Translate failed ({argos_err}), falling back to LLM")
                    translated = None

            # --- Tier 3: Qwen LLM via Ollama (last resort, uses VRAM) ---
            if not translated:
                _method = 'ollama'
                scoring_cfg = strat.config.get('scoring', {})
                ollama_host = scoring_cfg.get('ollama_host', 'http://localhost:11434')
                model = scoring_cfg.get('inference_model', 'qwen3.5:9b')
                text_chunk = transcript_text[:4000]
                import requests as req
                resp = req.post(f"{ollama_host}/api/chat", json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": f"You are a translator. Translate the following text to {target_lang}. Output ONLY the translation, nothing else."},
                        {"role": "user", "content": text_chunk},
                    ],
                    "stream": False,
                    "think": False,
                    "options": {"num_predict": 4000, "temperature": 0.3},
                }, timeout=120)
                if resp.status_code == 200:
                    result = resp.json()
                    translated = result.get('message', {}).get('content', '')
                    import re
                    translated = re.sub(r'<think>.*?</think>', '', translated, flags=re.DOTALL).strip()

            if translated:
                # Save translation as a transcript insight in the target language
                cursor.execute(
                    "DELETE FROM video_insights WHERE video_id = ? AND profile_id = ? AND lens_name = 'transcript' AND language = ?",
                    (video_id, handler._profile_id, target_lang)
                )
                cursor.execute(
                    "INSERT INTO video_insights (video_id, profile_id, lens_name, content, language) VALUES (?, ?, 'transcript', ?, ?)",
                    (video_id, handler._profile_id, json.dumps({"transcript": translated}, ensure_ascii=False), target_lang)
                )
                strat.db._commit()

                _send_json(handler, {
                    "translated_text": translated,
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "chars_translated": len(transcript_text),
                    "total_chars": len(transcript_text),
                    "method": _method,
                })
            else:
                _send_json(handler, {"error": "Translation failed"}, 500)
        except Exception as e:
            logger.error(f"Translation error: {e}")
            _send_json(handler, {"error": str(e)}, 500)
        return True

    if len(path_parts) == 5 and path_parts[2] == 'channels' and path_parts[4] == 'lenses':
        # PUT-like: POST /api/youtube/channels/:id/lenses
        try:
            ch_id = int(path_parts[3])
            lenses = body.get('lenses', [])
            cursor = strat.db.conn.cursor()
            cursor.execute(
                "UPDATE youtube_channels SET lenses = ? WHERE id = ? AND profile_id = ?",
                (json.dumps(lenses), ch_id, handler._profile_id)
            )
            strat.db._commit()
            _send_json(handler, {"ok": True})
        except (ValueError, IndexError):
            _send_json(handler, {"error": "Invalid channel ID"}, 400)
        return True

    return False


def handle_delete(handler, strat, auth, path):
    """Handle DELETE requests for YouTube routes. Returns True if handled."""

    if not path.startswith("/api/youtube/channels/"):
        return False

    try:
        import sqlite3 as _sq
        ch_id = int(path.split("/")[-1])
        profile_id = handler._profile_id
        logger.info(f"DELETE channel {ch_id} for profile {profile_id}")
        # Use independent connection to avoid lock contention with worker
        _db_path = str(strat.db.db_path)
        with _sq.connect(_db_path) as _dc:
            _dc.execute("PRAGMA busy_timeout = 5000")
            _dc.execute(
                """DELETE FROM video_insights WHERE video_id IN
                   (SELECT id FROM youtube_videos WHERE channel_id = ? AND profile_id = ?)""",
                (ch_id, profile_id)
            )
            _dc.execute(
                "DELETE FROM youtube_videos WHERE channel_id = ? AND profile_id = ?",
                (ch_id, profile_id)
            )
            _dc.execute(
                "DELETE FROM youtube_channels WHERE id = ? AND profile_id = ?",
                (ch_id, profile_id)
            )
            _dc.commit()
        _send_json(handler, {"ok": True})
    except Exception as e:
        logger.error(f"Delete channel failed: {e}")
        _send_json(handler, {"error": str(e)}, 500)
    return True

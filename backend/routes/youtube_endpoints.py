"""
YouTube routes — channel CRUD, video listing, insights, processing.
Extracted from server.py (Sprint 5K Phase 1).
"""

import json
import logging
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger("STRAT_OS")


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
                "SELECT * FROM youtube_videos WHERE channel_id = ? AND profile_id = ? ORDER BY published_at DESC",
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
                "SELECT title, transcript_language FROM youtube_videos WHERE id = ? AND profile_id = ?",
                (vid_id, handler._profile_id)
            )
            vrow = cursor.fetchone()
            video_title = vrow['title'] if vrow else ''
            transcript_lang = (vrow['transcript_language'] if vrow else 'en') or 'en'

            _send_json(handler, {
                "insights": insights,
                "video_title": video_title,
                "available_languages": available_langs,
                "transcript_language": transcript_lang,
                "current_language": language,
            })
        except (ValueError, IndexError):
            _send_json(handler, {"error": "Invalid video ID"}, 400)
        return True

    return False


def handle_post(handler, strat, auth, path):
    """Handle POST requests for YouTube routes. Returns True if handled."""

    if not path.startswith("/api/youtube/"):
        return False

    from processors.youtube import YouTubeProcessor
    yt = YouTubeProcessor(strat.config, db=strat.db)
    body = json.loads(handler.rfile.read(int(handler.headers.get('Content-Length', 0))).decode()) if int(handler.headers.get('Content-Length', 0)) > 0 else {}

    if path == "/api/youtube/channels":
        channel_input = (body.get('channel') or body.get('channel_url') or '').strip()
        lenses = body.get('lenses', ['summary'])
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
        try:
            ch_id = int(path_parts[3])
            new_videos = yt.discover_new_videos(handler._profile_id, ch_id)
            _send_json(handler, {"ok": True, "new_videos": len(new_videos), "videos": new_videos[:10]})
        except (ValueError, IndexError):
            _send_json(handler, {"error": "Invalid channel ID"}, 400)
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
        ch_id = int(handler.path.split("/")[-1])
        from processors.youtube import YouTubeProcessor
        yt = YouTubeProcessor(strat.config, db=strat.db)
        if yt.remove_channel(handler._profile_id, ch_id):
            _send_json(handler, {"ok": True})
        else:
            _send_json(handler, {"error": "Channel not found"}, 404)
    except (ValueError, IndexError):
        _send_json(handler, {"error": "Invalid channel ID"}, 400)
    return True

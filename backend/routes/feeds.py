"""
Feed routes — extra RSS feeds, custom user feeds, feed catalog, RSS discovery.
Extracted from server.py (Sprint 5K Phase 1).
"""

import json
import logging
import re as _re
from datetime import datetime
from urllib.parse import urlparse, parse_qs, urljoin

import requests

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
    """Handle GET requests for feed routes. Returns True if handled."""

    if path in ("/api/finance-news", "/api/politics-news", "/api/jobs-news"):
        feed_type = "finance" if "finance" in path else "jobs" if "jobs" in path else "politics"
        handler.send_response(200)
        handler.send_header("Content-type", "application/json")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("Cache-Control", "no-cache, no-store")
        handler.end_headers()
        try:
            from fetchers.extra_feeds import fetch_extra_feeds
            # Use user-specific feed toggles from profile overlay
            _ef_config = dict(strat.config)
            if handler._profile_id:
                try:
                    _ef_row = strat.db.conn.execute(
                        "SELECT config_overlay FROM profiles WHERE id = ?",
                        (handler._profile_id,)
                    ).fetchone()
                    if _ef_row and _ef_row[0]:
                        _ef_overlay = json.loads(_ef_row[0])
                        for _ek in [f"extra_feeds_{feed_type}", f"custom_feeds_{feed_type}"]:
                            if _ek in _ef_overlay:
                                _ef_config[_ek] = _ef_overlay[_ek]
                except Exception:
                    pass
            items = fetch_extra_feeds(feed_type, config=_ef_config)
        except Exception as e:
            logger.error(f"Extra feeds ({feed_type}) error: {e}")
            items = []
        handler.wfile.write(json.dumps({"items": items, "fetched_at": datetime.now().isoformat()}).encode())
        return True

    # Custom user feeds (RSS from user-defined URLs)
    if path == "/api/custom-news":
        handler.send_response(200)
        handler.send_header("Content-type", "application/json")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("Cache-Control", "no-cache, no-store")
        handler.end_headers()
        items = []
        try:
            # Resolve per-user custom feeds from profile overlay
            _cf_feeds = strat.config.get("custom_feeds", [])
            if handler._profile_id:
                try:
                    _cf_row = strat.db.conn.execute(
                        "SELECT config_overlay FROM profiles WHERE id = ?",
                        (handler._profile_id,)
                    ).fetchone()
                    if _cf_row and _cf_row[0]:
                        _cf_overlay = json.loads(_cf_row[0])
                        if "custom_feeds" in _cf_overlay:
                            _cf_feeds = _cf_overlay["custom_feeds"]
                except Exception:
                    pass
            custom_feeds = _cf_feeds
            enabled_feeds = [f for f in custom_feeds if f.get("on", True)]
            if enabled_feeds:
                import feedparser
                _ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                _worker_base = strat.config.get("proxy", {}).get("cloudflare_worker", "")
                _blocked = strat.config.get("proxy", {}).get("blocked_domains", [])

                for feed_cfg in enabled_feeds:
                    try:
                        feed_url = feed_cfg["url"]
                        feed_domain = urlparse(feed_url).hostname or ""
                        _is_blocked = any(feed_domain == b or feed_domain.endswith("." + b) for b in _blocked)

                        # YouTube RSS fallback: if RSS returns 404/500, use yt-dlp
                        _yt_rss_match = _re.search(r'youtube\.com/feeds/videos\.xml\?channel_id=(UC[\w-]{22})', feed_url)
                        if _yt_rss_match:
                            try:
                                _yt_resp = requests.get(feed_url, timeout=8, headers={"User-Agent": _ua})
                                if _yt_resp.status_code == 200:
                                    feed = feedparser.parse(_yt_resp.content)
                                else:
                                    raise requests.RequestException(f"HTTP {_yt_resp.status_code}")
                            except Exception:
                                # yt-dlp fallback for YouTube feeds
                                import subprocess as _sp
                                try:
                                    _yt_cid = _yt_rss_match.group(1)
                                    _yt_result = _sp.run(
                                        ['yt-dlp', '--flat-playlist', '--no-download', '-J',
                                         '--playlist-end', '15', '--js-runtimes', 'node',
                                         f'https://www.youtube.com/channel/{_yt_cid}/videos'],
                                        capture_output=True, text=True, timeout=30
                                    )
                                    if _yt_result.returncode == 0:
                                        _yt_data = json.loads(_yt_result.stdout)
                                        _yt_ch_name = _yt_data.get('channel', feed_cfg.get('name', ''))
                                        for _yt_entry in (_yt_data.get('entries', []) or [])[:15]:
                                            items.append({
                                                "title": _yt_entry.get("title", ""),
                                                "url": _yt_entry.get("url", ""),
                                                "source": _yt_ch_name,
                                                "category": "custom",
                                                "summary": _yt_entry.get("description", "")[:200] if _yt_entry.get("description") else "",
                                                "timestamp": _yt_entry.get("upload_date", ""),
                                                "thumbnail": (_yt_entry.get("thumbnails", [{}])[-1] or {}).get("url", ""),
                                            })
                                except Exception as _yt_e:
                                    logger.debug(f"YouTube yt-dlp fallback failed for {feed_cfg.get('name','')}: {_yt_e}")
                                continue  # Skip normal feed processing

                        # Route blocked feeds through Cloudflare Worker
                        if _is_blocked and _worker_base:
                            try:
                                _proxy_url = f"{_worker_base}/feed?url={requests.utils.quote(feed_url, safe='')}"
                                _resp = requests.get(_proxy_url, headers={"User-Agent": _ua}, timeout=20)
                                feed = feedparser.parse(_resp.content)
                            except Exception as _pe:
                                logger.warning(f"CF proxy failed for {feed_cfg.get('name','')}: {_pe}")
                                feed = feedparser.parse("")
                        else:
                            # Direct fetch with headers
                            try:
                                _resp = requests.get(feed_url, headers={
                                    "User-Agent": _ua,
                                    "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
                                }, timeout=15, allow_redirects=True)
                                feed = feedparser.parse(_resp.content)
                            except requests.RequestException:
                                try:
                                    from curl_cffi import requests as cf_req
                                    _resp = cf_req.get(feed_url, impersonate="chrome", timeout=15)
                                    feed = feedparser.parse(_resp.content)
                                except Exception:
                                    feed = feedparser.parse(feed_url)

                        for entry in feed.entries[:15]:
                            pub = entry.get("published", entry.get("updated", ""))
                            link = entry.get("link", "")

                            # ── Extract thumbnail ──
                            thumb = ""
                            # 1. media:thumbnail or media:content
                            media_thumb = entry.get("media_thumbnail", [])
                            if media_thumb and isinstance(media_thumb, list):
                                thumb = media_thumb[0].get("url", "")
                            if not thumb:
                                media_content = entry.get("media_content", [])
                                if media_content and isinstance(media_content, list):
                                    for mc in media_content:
                                        if mc.get("medium") == "image" or "image" in mc.get("type", ""):
                                            thumb = mc.get("url", "")
                                            break
                            # 2. enclosure with image type
                            if not thumb:
                                enclosures = entry.get("enclosures", [])
                                if enclosures:
                                    for enc in enclosures:
                                        if "image" in enc.get("type", ""):
                                            thumb = enc.get("href", enc.get("url", ""))
                                            break
                            # 3. First <img> in summary/content HTML
                            full_image = ""
                            if not thumb:
                                content_html = entry.get("summary", "") or ""
                                if entry.get("content"):
                                    content_html = entry["content"][0].get("value", content_html)
                                img_match = _re.search(r'<img[^>]+src=["\']([^"\']+)["\']', content_html)
                                if img_match:
                                    thumb = img_match.group(1)
                            # 4. Construct sample + full URLs from known booru patterns
                            sample_image = ""
                            if thumb:
                                # Yande.re: assets.yande.re/data/preview/ab/cd/hash.jpg
                                #   sample: files.yande.re/sample/hash/yande.re+NNN+sample+tags.jpg (unreliable)
                                #   jpeg:   files.yande.re/jpeg/ab/cd/hash.jpg (reliable mid-res)
                                #   image:  files.yande.re/image/ab/cd/hash.png (full, might be png)
                                if 'yande.re' in thumb and '/preview/' in thumb:
                                    sample_image = thumb.replace('assets.yande.re/data/preview/', 'files.yande.re/jpeg/').replace('/preview/', '/jpeg/')
                                    full_image = thumb.replace('assets.yande.re/data/preview/', 'files.yande.re/image/').replace('/preview/', '/image/')
                                # Konachan: same structure as yande.re
                                elif 'konachan.' in thumb and '/preview/' in thumb:
                                    sample_image = thumb.replace('/preview/', '/jpeg/')
                                    full_image = thumb.replace('/preview/', '/image/')
                                # Danbooru: 360px thumbnail is the max free resolution
                                # /sample/ and /original/ require different filename formats or paid account
                                elif 'donmai.us' in thumb:
                                    pass  # Just use thumbnail as-is
                                # Gelbooru: img*.gelbooru.com/thumbnails/hash/thumbnail_file.jpg
                                elif 'gelbooru.' in thumb and '/thumbnails/' in thumb:
                                    sample_image = thumb.replace('/thumbnails/', '/samples/').replace('thumbnail_', 'sample_')
                                    full_image = thumb.replace('/thumbnails/', '/images/').replace('thumbnail_', '')
                                # Safebooru: same as gelbooru
                                elif 'safebooru.' in thumb and '/thumbnails/' in thumb:
                                    sample_image = thumb.replace('/thumbnails/', '/samples/').replace('thumbnail_', 'sample_')
                                    full_image = thumb.replace('/thumbnails/', '/images/').replace('thumbnail_', '')
                            # Also scan content HTML for any larger image URLs
                            if not full_image:
                                content_html = entry.get("summary", "") or ""
                                if entry.get("content"):
                                    content_html = entry["content"][0].get("value", content_html)
                                all_imgs = _re.findall(r'(?:src|href)=["\']([^"\']+\.(?:jpg|jpeg|png|webp))["\']', content_html)
                                for img_url in all_imgs:
                                    if '/sample/' in img_url or '/jpeg/' in img_url or '/image/' in img_url:
                                        full_image = img_url
                                        break

                            # ── Detect media type from URL ──
                            media_type = "article"
                            embed_id = ""
                            embed_type = ""

                            if any(d in link for d in ["youtube.com/watch", "youtu.be/", "youtube.com/shorts"]):
                                media_type = "video"
                                yt_match = _re.search(r'(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})', link)
                                if yt_match:
                                    embed_id = yt_match.group(1)
                                    embed_type = "youtube"
                                    if not thumb:
                                        thumb = f"https://img.youtube.com/vi/{embed_id}/mqdefault.jpg"
                            elif "twitch.tv/" in link:
                                media_type = "stream"
                                tw_match = _re.search(r'twitch\.tv/(?:videos/)?([a-zA-Z0-9_]+)', link)
                                if tw_match:
                                    embed_id = tw_match.group(1)
                                    embed_type = "twitch"
                            elif any(d in link for d in ["danbooru.", "yande.re", "gelbooru.", "konachan.", "safebooru."]):
                                media_type = "image"
                            elif any(d in link for d in ["mangadex.", "mangaplus.", "webtoons.", "mangakakalot.", "manganato."]):
                                media_type = "manga"
                            elif any(d in link for d in ["vimeo.com", "dailymotion.com"]):
                                media_type = "video"
                            elif link.lower().endswith((".mp4", ".webm", ".mov")):
                                media_type = "video"

                            # ── Proxy blocked thumbnails ──
                            if thumb:
                                thumb_domain = urlparse(thumb).hostname or ""
                                if any(thumb_domain == b or thumb_domain.endswith("." + b) for b in _blocked):
                                    thumb = f"/api/proxy?url={requests.utils.quote(thumb, safe='')}"
                            if sample_image:
                                si_domain = urlparse(sample_image).hostname or ""
                                if any(si_domain == b or si_domain.endswith("." + b) for b in _blocked):
                                    sample_image = f"/api/proxy?url={requests.utils.quote(sample_image, safe='')}"
                            if full_image:
                                fi_domain = urlparse(full_image).hostname or ""
                                if any(fi_domain == b or fi_domain.endswith("." + b) for b in _blocked):
                                    full_image = f"/api/proxy?url={requests.utils.quote(full_image, safe='')}"

                            # ── Build item ──
                            item = {
                                "title": entry.get("title", "No title"),
                                "url": link,
                                "summary": _re.sub(r'<[^>]+>', '', entry.get("summary", ""))[:300],
                                "source": feed_cfg.get("name", "Custom"),
                                "timestamp": pub,
                                "media_type": media_type,
                            }
                            if thumb:
                                item["thumbnail"] = thumb
                            if sample_image:
                                item["sample_image"] = sample_image
                            if full_image:
                                item["full_image"] = full_image
                            if embed_id:
                                item["embed_id"] = embed_id
                                item["embed_type"] = embed_type
                            items.append(item)
                    except Exception as e:
                        logger.warning(f"Custom feed error ({feed_cfg.get('name','')}): {e}")
                # Sort by timestamp descending
                items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                items = items[:80]  # Cap at 80 (more content for media view)
        except Exception as e:
            logger.error(f"Custom feeds error: {e}")
        handler.wfile.write(json.dumps({"items": items, "fetched_at": datetime.now().isoformat()}).encode())
        return True

    # Serve feed catalog (for Settings UI)
    if path in ("/api/feed-catalog/finance", "/api/feed-catalog/politics", "/api/feed-catalog/jobs"):
        feed_type = "finance" if "finance" in path else "jobs" if "jobs" in path else "politics"
        handler.send_response(200)
        handler.send_header("Content-type", "application/json")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        from fetchers.extra_feeds import get_catalog
        catalog = get_catalog(feed_type)
        # Merge with user toggles from config
        toggle_key = f"extra_feeds_{feed_type}"
        user_toggles = strat.config.get(toggle_key, {})
        if user_toggles:
            # Profile has saved preferences — apply them
            for item in catalog:
                if item["id"] in user_toggles:
                    item["on"] = user_toggles[item["id"]]
        else:
            # No saved preferences (fresh profile) — default all to off
            for item in catalog:
                item["on"] = False
        handler.wfile.write(json.dumps({"catalog": catalog, "type": feed_type}).encode())
        return True

    return False


def handle_post(handler, strat, auth, path):
    """Handle POST requests for feed routes. Returns True if handled."""

    # RSS feed auto-discovery from a regular URL
    if path == "/api/discover-rss":
        content_length = int(handler.headers.get('Content-Length', 0))
        post_data = handler.rfile.read(content_length) if content_length else b'{}'
        try:
            body = json.loads(post_data.decode('utf-8')) if post_data.strip() else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = {}
        url = body.get("url", "").strip()
        if not url:
            _send_json(handler, {"error": "URL required"}, 400)
            return True
        try:
            # YouTube shortcut: detect channel URLs and resolve to RSS
            _yt_match = _re.match(r'https?://(?:www\.)?youtube\.com/(?:@|channel/|c/|user/)([^/?&#]+)', url)
            if _yt_match:
                _yt_id_or_handle = _yt_match.group(1)
                # If it's already a channel ID (starts with UC), use directly
                if _yt_id_or_handle.startswith('UC') and len(_yt_id_or_handle) == 24:
                    _channel_id = _yt_id_or_handle
                else:
                    # Use YouTube's internal resolve API (bypasses consent pages)
                    try:
                        _yt_api = requests.post("https://www.youtube.com/youtubei/v1/navigation/resolve_url",
                            json={"url": url, "context": {"client": {"clientName": "WEB", "clientVersion": "2.20240101"}}},
                            headers={"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"},
                            timeout=10)
                        if _yt_api.status_code == 200:
                            import re as _re2
                            _uc_matches = _re2.findall(r'UC[a-zA-Z0-9_-]{22}', _yt_api.text)
                            _channel_id = _uc_matches[0] if _uc_matches else None
                        else:
                            _channel_id = None
                    except Exception:
                        _channel_id = None
                        _channel_id = None

                if _channel_id:
                    _yt_feed_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={_channel_id}"
                    # Verify the feed works
                    try:
                        import feedparser
                        _yt_feed_resp = requests.get(_yt_feed_url, timeout=8)
                        _yt_parsed = feedparser.parse(_yt_feed_resp.content)
                        _yt_title = _yt_parsed.feed.get("title", _yt_id_or_handle)
                        _send_json(handler, {"feeds": [{
                            "url": _yt_feed_url,
                            "title": _yt_title,
                            "type": "youtube",
                            "entries": len(_yt_parsed.entries)
                        }]})
                    except Exception:
                        _send_json(handler, {"feeds": [{
                            "url": _yt_feed_url,
                            "title": _yt_id_or_handle,
                            "type": "youtube",
                            "entries": 0
                        }]})
                    return True
            # Check if domain is blocked
            from urllib.parse import urlparse as _urlparse, quote_plus as _quote_plus
            _parsed_url = _urlparse(url)
            _domain = _parsed_url.netloc.lower().replace('www.', '')
            _blocked = strat.config.get("proxy", {}).get("blocked_domains", [])
            if any(_domain == bd or _domain.endswith('.' + bd) for bd in _blocked):
                _send_json(handler, {"error": "Domain is blocked"}, 403)
                return True

            _disc_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            # Route through CF Worker proxy if configured
            _cf_worker = strat.config.get("proxy", {}).get("cloudflare_worker", "")
            _fetch_url = f"{_cf_worker}?url={_quote_plus(url)}" if _cf_worker else url
            try:
                resp = requests.get(_fetch_url, timeout=10, headers=_disc_headers, allow_redirects=True)
            except requests.RequestException:
                try:
                    from curl_cffi import requests as cf_req
                    resp = cf_req.get(_fetch_url, impersonate="chrome", timeout=10)
                except Exception:
                    raise
            feeds = []
            ct = resp.headers.get('content-type', '').lower() if hasattr(resp.headers, 'get') else ''
            html_text = resp.text[:100000]

            # Strategy 0: URL already IS a feed (XML/RSS/Atom content-type or content starts with XML)
            if 'xml' in ct or 'rss' in ct or 'atom' in ct or html_text.lstrip().startswith('<?xml') or html_text.lstrip().startswith('<rss') or html_text.lstrip().startswith('<feed'):
                try:
                    import feedparser
                    parsed = feedparser.parse(resp.content)
                    if parsed.entries and len(parsed.entries) > 0:
                        feeds.append({"url": url, "title": parsed.feed.get("title", ""), "type": "direct_feed", "entries": len(parsed.entries)})
                except Exception:
                    pass

            # Strategy 1: Find all <link> tags and check for RSS/Atom
            for tag_match in _re.finditer(r'<link\b[^>]*/?>', html_text, _re.IGNORECASE):
                tag = tag_match.group(0)
                # Must have type with rss/atom/xml
                type_m = _re.search(r'type=["\']([^"\']+)["\']', tag, _re.IGNORECASE)
                href_m = _re.search(r'href=["\']([^"\']+)["\']', tag, _re.IGNORECASE)
                if not type_m or not href_m:
                    continue
                ftype = type_m.group(1).lower()
                if 'rss' not in ftype and 'atom' not in ftype and 'xml' not in ftype:
                    continue
                href = href_m.group(1)
                if not href.startswith('http'):
                    href = urljoin(url, href)
                title_m = _re.search(r'title=["\']([^"\']+)["\']', tag, _re.IGNORECASE)
                feeds.append({
                    "url": href,
                    "title": title_m.group(1) if title_m else "",
                    "type": ftype
                })

            # Strategy 2: Try feedparser directly (some sites serve RSS at their main URL)
            if not feeds:
                try:
                    import feedparser
                    parsed = feedparser.parse(resp.content)
                    if parsed.entries and len(parsed.entries) > 0:
                        feeds.append({"url": url, "title": parsed.feed.get("title", ""), "type": "direct"})
                except Exception:
                    pass

            # Strategy 3: Probe common RSS paths
            if not feeds:
                common_paths = ['/feed', '/rss', '/feed.xml', '/rss.xml', '/atom.xml',
                                '/index.xml', '/feeds/posts/default', '/rss/index.xml']
                for rss_path in common_paths:
                    try:
                        test_url = urljoin(url, rss_path)
                        r = requests.get(test_url, timeout=5, headers={
                            "User-Agent": "Mozilla/5.0"
                        }, allow_redirects=True)
                        ct = r.headers.get('content-type', '').lower()
                        if r.status_code == 200 and ('xml' in ct or 'rss' in ct or 'atom' in ct):
                            feeds.append({"url": test_url, "title": "", "type": ct.split(';')[0]})
                            break
                    except Exception:
                        continue
            _send_json(handler, {"feeds": feeds, "source_url": url})
        except Exception as e:
            logger.warning(f"RSS discovery error for {url}: {e}")
            _send_json(handler, {"feeds": [], "error": str(e)})
        return True

    return False

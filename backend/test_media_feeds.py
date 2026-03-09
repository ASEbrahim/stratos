#!/usr/bin/env python3
"""
StratOS Rich Media Feed — Diagnostics
Run from backend/: python3 test_media_feeds.py
"""

import requests
import yaml
import sys
from urllib.parse import urlparse, quote

# Load config
try:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("ERROR: Run this from the backend/ directory")
    sys.exit(1)

worker_url = config.get("proxy", {}).get("cloudflare_worker", "")
blocked = config.get("proxy", {}).get("blocked_domains", [])
custom_feeds = config.get("custom_feeds", [])

print("=" * 60)
print("StratOS Media Feed Diagnostics")
print("=" * 60)

# 1. Check CF Worker
print(f"\n--- Cloudflare Worker ---")
if not worker_url:
    print("  SKIP: No cloudflare_worker URL in config.yaml")
else:
    print(f"  URL: {worker_url}")
    try:
        r = requests.get(f"{worker_url}/health", timeout=10)
        if r.status_code == 200:
            print(f"  HEALTH: OK ({r.json()})")
        else:
            print(f"  HEALTH: FAIL (HTTP {r.status_code})")
    except Exception as e:
        print(f"  HEALTH: ERROR ({e})")

# 2. Test blocked domains through worker
print(f"\n--- Blocked Domain Proxy Tests ---")
print(f"  Blocked domains: {len(blocked)}")
test_urls = [
    ("Yande.re RSS", "https://yande.re/post/atom?tags=landscape&limit=3"),
    ("Danbooru RSS", "https://danbooru.donmai.us/posts.atom?tags=scenery&limit=3"),
    ("Danbooru Image", "https://cdn.donmai.us/360x360/4e/38/4e38abb61858dadb4bf34af4dd40496c.jpg"),
]
for name, url in test_urls:
    domain = urlparse(url).hostname
    is_blocked = any(domain == b or domain.endswith("." + b) for b in blocked)
    if not worker_url:
        print(f"  {name}: SKIP (no worker)")
        continue
    proxy_url = f"{worker_url}/proxy?url={quote(url, safe='')}"
    try:
        r = requests.get(proxy_url, timeout=15)
        ct = r.headers.get("Content-Type", "?")
        size = len(r.content)
        status = "OK" if r.status_code == 200 else f"HTTP {r.status_code}"
        print(f"  {name}: {status} (type={ct[:40]}, size={size}b, blocked={is_blocked})")
    except Exception as e:
        print(f"  {name}: FAIL ({e})")

# 3. Test direct RSS feeds (non-blocked)
print(f"\n--- Direct Feed Tests ---")
direct_test = [
    ("YouTube (MKBHD)", "https://www.youtube.com/feeds/videos.xml?channel_id=UCBcRF18a7Qf58cCRy5xuWwQ"),
    ("BBC News", "https://feeds.bbci.co.uk/news/rss.xml"),
    ("MangaDex", "https://mangadex.org/rss/latest"),
    ("Safebooru", "https://safebooru.org/index.php?page=atom&s=post&tags=landscape"),
]
for name, url in direct_test:
    domain = urlparse(url).hostname
    is_blocked = any(domain == b or domain.endswith("." + b) for b in blocked)
    try:
        if is_blocked and worker_url:
            fetch_url = f"{worker_url}/feed?url={quote(url, safe='')}"
            label = "via proxy"
        else:
            fetch_url = url
            label = "direct"
        r = requests.get(fetch_url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
        })
        import feedparser
        feed = feedparser.parse(r.content)
        entries = len(feed.entries)
        title = feed.feed.get("title", "?")[:40]
        has_media = sum(1 for e in feed.entries[:10] if e.get("media_thumbnail") or e.get("media_content"))
        print(f"  {name} ({label}): {entries} entries, title='{title}', media={has_media}")
        if entries > 0:
            e = feed.entries[0]
            link = e.get("link", "?")
            # Check media type detection
            mtype = "article"
            if "youtube.com" in link or "youtu.be" in link:
                mtype = "video"
            elif "twitch.tv" in link:
                mtype = "stream"
            elif any(d in link for d in ["danbooru.", "yande.re", "gelbooru."]):
                mtype = "image"
            elif any(d in link for d in ["mangadex.", "mangaplus."]):
                mtype = "manga"
            print(f"    → first: '{e.get('title','?')[:50]}' | type={mtype} | link={link[:60]}")
    except Exception as e:
        print(f"  {name}: FAIL ({type(e).__name__}: {e})")

# 4. Test local /api/proxy endpoint
print(f"\n--- Local /api/proxy Endpoint ---")
local_base = f"http://localhost:{config.get('system', {}).get('frontend_port', 8080)}"
test_proxy_url = "https://img.youtube.com/vi/dQw4w9WgXcQ/mqdefault.jpg"
try:
    r = requests.get(f"{local_base}/api/proxy?url={quote(test_proxy_url, safe='')}", timeout=10)
    ct = r.headers.get("Content-Type", "?")
    print(f"  YouTube thumb (non-blocked): HTTP {r.status_code}, type={ct}, size={len(r.content)}b")
except Exception as e:
    print(f"  YouTube thumb: FAIL ({e})")

if worker_url:
    blocked_img = "https://cdn.donmai.us/360x360/4e/38/4e38abb61858dadb4bf34af4dd40496c.jpg"
    try:
        r = requests.get(f"{local_base}/api/proxy?url={quote(blocked_img, safe='')}", timeout=15)
        ct = r.headers.get("Content-Type", "?")
        print(f"  Danbooru img (blocked): HTTP {r.status_code}, type={ct}, size={len(r.content)}b")
    except Exception as e:
        print(f"  Danbooru img: FAIL ({e})")

# 5. Custom feeds from config
print(f"\n--- Custom Feeds in Config ---")
if not custom_feeds:
    print("  No custom feeds configured yet")
else:
    for f in custom_feeds:
        status = "ON" if f.get("on", True) else "off"
        domain = urlparse(f.get("url", "")).hostname or "?"
        is_blocked = any(domain == b or domain.endswith("." + b) for b in blocked)
        proxy_note = " [BLOCKED → proxy]" if is_blocked else ""
        print(f"  [{status}] {f.get('name', '?')}: {f.get('url', '?')[:60]}{proxy_note}")

print(f"\n{'=' * 60}")
print("Done.")

#!/usr/bin/env python3
import requests, yaml, feedparser, sqlite3, json
from urllib.parse import urlparse, quote

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
worker = cfg.get("proxy", {}).get("cloudflare_worker", "")
blocked = cfg.get("proxy", {}).get("blocked_domains", [])

feeds = []
conn = sqlite3.connect("strat_os.db")
for pid, name, ov in conn.execute("SELECT id, name, config_overlay FROM profiles").fetchall():
    c = json.loads(ov) if ov else {}
    if c.get("custom_feeds"):
        feeds = c["custom_feeds"]
        print(f"Testing feeds from profile: {name}")
        break
conn.close()

if not feeds:
    print("No custom feeds found")
    exit()

ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
for f in feeds:
    url, name, on = f.get("url",""), f.get("name","?"), f.get("on",True)
    if not on:
        print(f"  [off] {name}"); continue
    domain = urlparse(url).hostname or ""
    is_blocked = any(domain == b or domain.endswith("." + b) for b in blocked)
    try:
        if is_blocked and worker:
            fetch_url = f"{worker}/feed?url={quote(url, safe='')}"
            label = "proxy"
        else:
            fetch_url = url
            label = "direct"
        r = requests.get(fetch_url, headers={"User-Agent": ua}, timeout=15)
        if r.status_code >= 400:
            print(f"  [FAIL] {name}: HTTP {r.status_code} ({label})"); continue
        feed = feedparser.parse(r.content)
        n = len(feed.entries)
        if n == 0:
            ct = r.headers.get("Content-Type","?")[:40]
            print(f"  [EMPTY] {name}: 0 entries, ct={ct} ({label})")
        else:
            print(f"  [OK]   {name}: {n} entries ({label})")
    except Exception as e:
        print(f"  [ERR]  {name}: {type(e).__name__}: {e}")

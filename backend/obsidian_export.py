#!/usr/bin/env python3
"""StratOS → Obsidian Vault Exporter

Exports StratOS intelligence data into a richly linked Obsidian vault
with Dataview dashboards and an interactive D3/Plotly HTML dashboard.

Usage:
    python3 obsidian_export.py                          # Full export (profile 8)
    python3 obsidian_export.py --vault ~/my_vault       # Custom path
    python3 obsidian_export.py --days 30                # Last 30 days only
    python3 obsidian_export.py --min-score 5.0          # Lower threshold
    python3 obsidian_export.py --profile 8              # Specific profile
    python3 obsidian_export.py --incremental            # Only new since last sync
    python3 obsidian_export.py --no-dashboard           # Skip HTML
    python3 obsidian_export.py --dashboard-only         # Only HTML
    python3 obsidian_export.py --all-profiles           # Export all profiles
    python3 obsidian_export.py -v                       # Verbose
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

# ── Constants ──────────────────────────────────────────────────────────────────

DB_PATH = Path(__file__).parent / "strat_os.db"
DEFAULT_VAULT = Path.home() / "stratos_vault"
DEFAULT_PROFILE = 8
DEFAULT_MIN_SCORE = 6.0
MAX_FILENAME_LEN = 150
SCORE_TIERS = {
    "critical": (9.0, 10.0),
    "high": (7.0, 8.99),
    "medium": (5.0, 6.99),
    "noise": (0.0, 4.99),
}


def score_tier(score: float) -> str:
    if score >= 9.0:
        return "critical"
    if score >= 7.0:
        return "high"
    if score >= 5.0:
        return "medium"
    return "noise"


def sanitize_filename(name: str) -> str:
    """Strip illegal chars, truncate, and normalize whitespace."""
    if not name:
        return "untitled"
    name = re.sub(r'[/\\"|?*<>:\n\r\t]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    name = name.rstrip('.')
    if len(name) > MAX_FILENAME_LEN:
        name = name[:MAX_FILENAME_LEN].rsplit(' ', 1)[0]
    return name or "untitled"


def extract_domain(url: str) -> str:
    try:
        d = urlparse(url).netloc
        if d.startswith("www."):
            d = d[4:]
        return d or "unknown"
    except Exception:
        return "unknown"


def iso_date(ts: str) -> str:
    """Extract YYYY-MM-DD from any ISO timestamp."""
    if not ts:
        return ""
    return ts[:10]


def escape_yaml(s: str) -> str:
    """Escape a string for YAML frontmatter value."""
    if not s:
        return '""'
    s = s.replace('\\', '\\\\').replace('"', '\\"')
    return f'"{s}"'


def escape_yaml_multiline(s: str) -> str:
    """Escape for single-line YAML, replacing newlines."""
    if not s:
        return '""'
    s = s.replace('\n', ' ').replace('\r', '')
    return escape_yaml(s)


# ── Data Loading ───────────────────────────────────────────────────────────────

class StratOSData:
    def __init__(self, db_path, profile_id, min_score, days, incremental_since=None):
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row
        self.profile_id = profile_id
        self.min_score = min_score
        self.days = days
        self.incremental_since = incremental_since

        self.articles = []
        self.all_articles = []  # including below min_score for stats
        self.entities_by_name = {}
        self.entity_mentions = defaultdict(list)  # entity_name -> [recorded_at]
        self.entity_cooccurrence = defaultdict(lambda: defaultdict(int))
        self.scans = []
        self.feedback = []
        self.shadow_scores = []
        self.video_insights = []
        self.youtube_channels = []
        self.youtube_videos = []
        self.briefings = []
        self.profiles = {}

    def load_all(self, verbose=False):
        cur = self.conn.cursor()

        # Build date filter
        date_filter = ""
        if self.days:
            cutoff = (datetime.now() - timedelta(days=self.days)).isoformat()
            date_filter = f" AND timestamp >= '{cutoff}'"

        incr_filter = ""
        if self.incremental_since:
            incr_filter = f" AND fetched_at > '{self.incremental_since}'"

        # 1. Articles
        query = f"""SELECT * FROM news_items
                    WHERE profile_id = ? {date_filter} {incr_filter}
                    ORDER BY score DESC, timestamp DESC"""
        cur.execute(query, (self.profile_id,))
        all_rows = [dict(r) for r in cur.fetchall()]
        self.all_articles = all_rows
        self.articles = [a for a in all_rows if (a['score'] or 0) >= self.min_score]
        if verbose:
            print(f"  Articles: {len(self.articles)} (of {len(all_rows)} total)")

        # 2. Entity mentions (global — no profile_id column)
        cur.execute("SELECT entity_name, mention_count, recorded_at FROM entity_mentions ORDER BY recorded_at")
        for row in cur.fetchall():
            r = dict(row)
            self.entity_mentions[r['entity_name']].append(r)
        if verbose:
            print(f"  Entity mention records: {sum(len(v) for v in self.entity_mentions.values())}")

        # 3. Entities table
        cur.execute("SELECT * FROM entities")
        for row in cur.fetchall():
            r = dict(row)
            self.entities_by_name[r['name']] = r

        # 4. Compute entity-article links from article titles/summaries
        # Build entity index from entity_mentions unique names
        all_entity_names = sorted(self.entity_mentions.keys(), key=lambda x: -len(x))

        # Map articles to entities via title/summary matching
        self.article_entities = defaultdict(list)  # article_id -> [entity_name]
        self.entity_articles = defaultdict(list)   # entity_name -> [article_id]

        article_texts = {}
        for a in self.articles:
            text = f"{a.get('title', '')} {a.get('summary', '')}".lower()
            article_texts[a['id']] = text

        # Only use top entities (by total mention count) to avoid noise
        entity_total_mentions = {}
        for name, mentions in self.entity_mentions.items():
            entity_total_mentions[name] = sum(m['mention_count'] for m in mentions)

        top_entities = sorted(entity_total_mentions.items(), key=lambda x: -x[1])[:200]
        # Filter out very short/generic entity names
        _skip = {
            'the', 'and', 'for', 'how', 'new', 'top', 'can', 'find', 'get', 'pdf',
            'jobs', 'apply', 'best', 'with', 'from', 'this', 'that', 'your', 'our',
            'their', 'what', 'which', 'search', 'careers', 'login', 'sign', 'view',
            'learn', 'read', 'latest', 'today', 'world', 'based', 'using', 'like',
            'also', 'more', 'here', 'free', 'open', 'next', 'help', 'about',
        }
        top_entities = [(n, c) for n, c in top_entities if len(n) > 3 and n.lower() not in _skip]

        for ename, _ in top_entities:
            pattern = ename.lower()
            for aid, text in article_texts.items():
                if pattern in text:
                    self.article_entities[aid].append(ename)
                    self.entity_articles[ename].append(aid)

        # Compute co-occurrence
        for aid, ents in self.article_entities.items():
            for i, e1 in enumerate(ents):
                for e2 in ents[i+1:]:
                    self.entity_cooccurrence[e1][e2] += 1
                    self.entity_cooccurrence[e2][e1] += 1

        if verbose:
            print(f"  Entities with article links: {len(self.entity_articles)}")

        # 5. Scan log
        cur.execute("SELECT * FROM scan_log WHERE profile_id = ? ORDER BY started_at DESC", (self.profile_id,))
        self.scans = [dict(r) for r in cur.fetchall()]
        if verbose:
            print(f"  Scan logs: {len(self.scans)}")

        # 6. Feedback
        cur.execute("SELECT * FROM user_feedback WHERE profile_id = ? ORDER BY created_at DESC", (self.profile_id,))
        self.feedback = [dict(r) for r in cur.fetchall()]
        if verbose:
            print(f"  Feedback entries: {len(self.feedback)}")

        # 7. Shadow scores
        cur.execute("SELECT * FROM shadow_scores WHERE profile_id = ?", (self.profile_id,))
        self.shadow_scores = [dict(r) for r in cur.fetchall()]

        # 8. Video insights
        cur.execute("SELECT * FROM video_insights WHERE profile_id = ?", (self.profile_id,))
        self.video_insights = [dict(r) for r in cur.fetchall()]
        if verbose:
            print(f"  Video insights: {len(self.video_insights)}")

        # 9. YouTube
        cur.execute("SELECT * FROM youtube_channels WHERE profile_id = ?", (self.profile_id,))
        self.youtube_channels = [dict(r) for r in cur.fetchall()]
        cur.execute("SELECT * FROM youtube_videos WHERE profile_id = ?", (self.profile_id,))
        self.youtube_videos = [dict(r) for r in cur.fetchall()]

        # 10. Briefings
        cur.execute("SELECT * FROM briefings WHERE profile_id = ? ORDER BY generated_at DESC", (self.profile_id,))
        self.briefings = [dict(r) for r in cur.fetchall()]

        # 11. Profiles
        cur.execute("SELECT * FROM profiles")
        for row in cur.fetchall():
            r = dict(row)
            self.profiles[r['id']] = r

        if verbose:
            print(f"  Data loading complete.")

    def close(self):
        self.conn.close()


# ── Vault Generator ───────────────────────────────────────────────────────────

class VaultGenerator:
    def __init__(self, data: StratOSData, vault_path: Path, verbose=False):
        self.data = data
        self.vault = vault_path
        self.verbose = verbose
        self.files_written = 0

    def log(self, msg):
        if self.verbose:
            print(f"  {msg}")

    def write_file(self, rel_path: str, content: str):
        fp = self.vault / rel_path
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding='utf-8')
        self.files_written += 1

    def generate_all(self):
        self.log("Phase 1: Articles...")
        self.gen_articles()
        self.log("Phase 2: Entities...")
        self.gen_entities()
        self.log("Phase 3: Categories...")
        self.gen_categories()
        self.log("Phase 4: Sources...")
        self.gen_sources()
        self.log("Phase 5: Feedback...")
        self.gen_feedback()
        self.log("Phase 6: Scholarly...")
        self.gen_scholarly()
        self.log("Phase 7: Scans & Daily rollups...")
        self.gen_scans_and_daily()
        self.log("Phase 8: Dashboards...")
        self.gen_dashboards()
        self.log("Generating SETUP.md...")
        self.gen_setup()

    # ── Phase 1: Articles ──────────────────────────────────────────────────

    def gen_articles(self):
        for a in self.data.articles:
            title = a.get('title', 'Untitled')
            fname = sanitize_filename(title)
            domain = extract_domain(a.get('url', ''))
            tier = score_tier(a.get('score', 0))
            date = iso_date(a.get('timestamp', ''))
            score = a.get('score', 0) or 0

            # Entity links
            ents = self.data.article_entities.get(a['id'], [])
            ent_links = '\n'.join(f"- [[entities/{sanitize_filename(e)}]]" for e in ents[:20])
            if not ent_links:
                ent_links = "_No entity matches_"

            summary = (a.get('summary') or '')[:500]
            score_reason = a.get('score_reason', '') or ''

            # Score highlight
            score_display = f"=={score}/10== — **{tier.title()}**"

            has_feedback = any(f['news_id'] == a['id'] for f in self.data.feedback)

            content = f"""---
title: {escape_yaml_multiline(title)}
score: {score}
score_tier: {tier}
category: {escape_yaml(a.get('category', ''))}
source: {escape_yaml(a.get('source', ''))}
domain: {escape_yaml(domain)}
url: {escape_yaml(a.get('url', ''))}
timestamp: {a.get('timestamp', '')}
date: {date}
profile_id: {a.get('profile_id', '')}
score_reason: {escape_yaml_multiline(score_reason)}
has_feedback: {str(has_feedback).lower()}
type: article
---

# {title}

| Field | Value |
|-------|-------|
| Score | {score_display} |
| Category | [[categories/{a.get('category', 'unknown')}]] |
| Source | [[sources/{domain}]] |
| Date | {date} |
| URL | [Open ↗]({a.get('url', '')}) |

## Score Reasoning
> {score_reason}

## Summary
{summary}

## Entities Mentioned
{ent_links}
"""
            self.write_file(f"articles/{fname}.md", content)

    # ── Phase 2: Entities ──────────────────────────────────────────────────

    def gen_entities(self):
        # Get entities that appear in our articles
        relevant_entities = set()
        for aid, ents in self.data.article_entities.items():
            relevant_entities.update(ents)

        # Only add standalone entity notes for high-mention entities
        # that aren't generic/noisy words
        _noise_entities = {
            'jobs', 'apply', 'find', 'employment', 'remote', 'indeed', 'bayt',
            'additionally', 'announcements', 'about', 'code', 'more', 'here',
            'best', 'free', 'available', 'open', 'click', 'next', 'help',
            'with', 'from', 'this', 'that', 'your', 'their', 'what', 'which',
            'search', 'careers', 'login', 'sign', 'view', 'learn', 'read',
            'latest', 'today', 'world', 'based', 'using', 'like', 'also',
        }
        for name, mentions in self.data.entity_mentions.items():
            total = sum(m['mention_count'] for m in mentions)
            if total >= 50 and len(name) > 4 and name.lower() not in _noise_entities:
                relevant_entities.add(name)

        for ename in relevant_entities:
            fname = sanitize_filename(ename)
            mentions = self.data.entity_mentions.get(ename, [])
            total_mentions = sum(m['mention_count'] for m in mentions)
            first_seen = mentions[0]['recorded_at'][:10] if mentions else ""
            last_seen = mentions[-1]['recorded_at'][:10] if mentions else ""

            linked_articles = self.data.entity_articles.get(ename, [])
            article_count = len(linked_articles)

            # Average score of linked articles
            scores = []
            for aid in linked_articles:
                for a in self.data.articles:
                    if a['id'] == aid:
                        scores.append(a.get('score', 0) or 0)
                        break
            avg_score = round(sum(scores) / len(scores), 1) if scores else 0

            # Entity category from entities table
            einfo = self.data.entities_by_name.get(ename, {})
            ecat = einfo.get('category', 'auto-discovered')

            # Co-occurring entities
            cooccur = self.data.entity_cooccurrence.get(ename, {})
            cooccur_sorted = sorted(cooccur.items(), key=lambda x: -x[1])[:15]
            cooccur_lines = '\n'.join(
                f"- [[entities/{sanitize_filename(e)}]] ({c} shared articles)"
                for e, c in cooccur_sorted
            )
            if not cooccur_lines:
                cooccur_lines = "_No co-occurring entities found_"

            content = f"""---
entity: {escape_yaml(ename)}
entity_category: {escape_yaml(ecat)}
mention_count: {total_mentions}
article_count: {article_count}
avg_score: {avg_score}
first_seen: {first_seen}
last_seen: {last_seen}
type: entity
---

# {ename}

| Metric | Value |
|--------|-------|
| Category | `{ecat}` |
| Total Mentions | {total_mentions} |
| Articles | {article_count} |
| First Seen | {first_seen} |
| Last Seen | {last_seen} |
| Avg Article Score | {avg_score} |

## Articles Mentioning {ename}

```dataview
TABLE score, date, category
FROM "articles"
WHERE contains(file.outlinks, this.file.link)
SORT score DESC
```

## Co-occurring Entities
{cooccur_lines}
"""
            self.write_file(f"entities/{fname}.md", content)

    # ── Phase 3: Categories ────────────────────────────────────────────────

    def gen_categories(self):
        cats = defaultdict(list)
        for a in self.data.all_articles:
            cat = a.get('category', 'unknown') or 'unknown'
            cats[cat].append(a)

        for cat, items in cats.items():
            scores = [i.get('score', 0) or 0 for i in items]
            avg = round(sum(scores) / len(scores), 1) if scores else 0
            high = sum(1 for s in scores if s >= 7.0)
            critical = sum(1 for s in scores if s >= 9.0)

            content = f"""---
category: {escape_yaml(cat)}
article_count: {len(items)}
avg_score: {avg}
high_count: {high}
critical_count: {critical}
type: category
---

# {cat}

## Live Stats
```dataview
TABLE WITHOUT ID
  length(rows) as "Total Articles",
  round(sum(rows.score) / length(rows), 1) as "Avg Score",
  min(rows.date) as "First Article",
  max(rows.date) as "Latest"
FROM "articles"
WHERE category = "{cat}"
GROUP BY true
```

## Top Articles
```dataview
TABLE score, date, domain
FROM "articles"
WHERE category = "{cat}" AND score >= 8.0
SORT score DESC
LIMIT 20
```

## Recent Articles (Last 7 Days)
```dataview
TABLE score, domain
FROM "articles"
WHERE category = "{cat}" AND date >= date(today) - dur(7 days)
SORT date DESC
```

## Top Sources for This Category
```dataview
TABLE WITHOUT ID
  domain as "Source",
  length(rows) as "Articles",
  round(sum(rows.score) / length(rows), 1) as "Avg Score"
FROM "articles"
WHERE category = "{cat}"
GROUP BY domain
SORT length(rows) DESC
LIMIT 10
```
"""
            self.write_file(f"categories/{cat}.md", content)

    # ── Phase 4: Sources ───────────────────────────────────────────────────

    def gen_sources(self):
        sources = defaultdict(list)
        for a in self.data.articles:
            domain = extract_domain(a.get('url', ''))
            sources[domain].append(a)

        for domain, items in sources.items():
            scores = [i.get('score', 0) or 0 for i in items]
            avg = round(sum(scores) / len(scores), 1) if scores else 0
            cats = list(set(i.get('category', '') for i in items))

            cats_yaml = json.dumps(cats)

            content = f"""---
domain: {escape_yaml(domain)}
article_count: {len(items)}
avg_score: {avg}
categories: {cats_yaml}
type: source
---

# {domain}

## Quality Metrics
```dataview
TABLE WITHOUT ID
  length(rows) as "Total Articles",
  round(sum(rows.score) / length(rows), 1) as "Avg Score",
  round(max(rows.score), 1) as "Best Score",
  round(min(rows.score), 1) as "Worst Score"
FROM "articles"
WHERE domain = "{domain}"
GROUP BY true
```

## Articles from {domain}
```dataview
TABLE score, category, date
FROM "articles"
WHERE domain = "{domain}"
SORT score DESC
```
"""
            self.write_file(f"sources/{sanitize_filename(domain)}.md", content)

    # ── Phase 5: Feedback ──────────────────────────────────────────────────

    def gen_feedback(self):
        for fb in self.data.feedback:
            title = fb.get('title', 'Unknown')
            fname = sanitize_filename(title)
            ai_score = fb.get('ai_score', 0) or 0
            user_score = fb.get('user_score')
            action = fb.get('action', '')

            delta = (user_score - ai_score) if user_score is not None else 0
            if delta > 0:
                delta_dir = "underscored"
                delta_display = f"+{delta:.1f}"
            elif delta < 0:
                delta_dir = "overscored"
                delta_display = f"{delta:.1f}"
            else:
                delta_dir = "agreed"
                delta_display = "0"

            article_link = f"[[articles/{fname}]]"
            note = fb.get('note', '') or ''

            content = f"""---
article_title: {escape_yaml_multiline(title)}
action: {action}
original_score: {ai_score}
user_score: {user_score if user_score is not None else 'null'}
delta: {delta:.1f}
delta_direction: {delta_dir}
timestamp: {iso_date(fb.get('created_at', ''))}
type: feedback
---

# Feedback — {title}

| Metric | Value |
|--------|-------|
| Action | {action.title()} |
| AI Score | {ai_score} |
| User Score | {user_score} |
| Delta | **{delta_display}** ({delta_dir.title()}) |
| Article | {article_link} |

## Note
> {note}
"""
            self.write_file(f"feedback/{fname}.md", content)

    # ── Phase 6: Scholarly / YouTube ───────────────────────────────────────

    def gen_scholarly(self):
        # Video insights
        videos_by_id = {v['id']: v for v in self.data.youtube_videos}
        channels_by_id = {c['id']: c for c in self.data.youtube_channels}

        for vi in self.data.video_insights:
            video = videos_by_id.get(vi.get('video_id'), {})
            vtitle = video.get('title', 'Unknown Video')
            lens = vi.get('lens_name', 'summary')
            fname = sanitize_filename(f"{vtitle} — {lens}")

            channel_id = video.get('channel_id')
            channel = channels_by_id.get(channel_id, {})
            channel_name = channel.get('channel_name', 'Unknown')

            insight_content = vi.get('content', '')
            try:
                parsed = json.loads(insight_content)
                insight_display = json.dumps(parsed, indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                insight_display = insight_content

            content = f"""---
video_title: {escape_yaml_multiline(vtitle)}
lens: {escape_yaml(lens)}
channel: {escape_yaml(channel_name)}
video_id: {escape_yaml(video.get('video_id', ''))}
processed_at: {vi.get('created_at', '')}
type: video_insight
---

# {vtitle} — {lens}

| Field | Value |
|-------|-------|
| Channel | [[scholarly/channels/{sanitize_filename(channel_name)}]] |
| Lens | `{lens}` |
| Processed | {vi.get('created_at', '')} |

## Insight
```json
{insight_display}
```
"""
            self.write_file(f"scholarly/{fname}.md", content)

        # Channel overviews
        for ch in self.data.youtube_channels:
            cname = ch.get('channel_name', 'Unknown')
            fname = sanitize_filename(cname)
            lenses = ch.get('lenses', '[]')
            try:
                lenses_list = json.loads(lenses) if isinstance(lenses, str) else lenses
            except (json.JSONDecodeError, TypeError):
                lenses_list = []

            video_count = sum(1 for v in self.data.youtube_videos if v.get('channel_id') == ch['id'])

            content = f"""---
channel_name: {escape_yaml(cname)}
channel_id: {escape_yaml(ch.get('channel_id', ''))}
video_count: {video_count}
lenses: {json.dumps(lenses_list)}
type: channel
---

# {cname}

## Videos
```dataview
TABLE lens, processed_at
FROM "scholarly"
WHERE channel = "{cname}"
SORT processed_at DESC
```
"""
            self.write_file(f"scholarly/channels/{fname}.md", content)

    # ── Phase 7: Scans & Daily Rollups ─────────────────────────────────────

    def gen_scans_and_daily(self):
        # Scan notes
        for s in self.data.scans:
            date = iso_date(s.get('started_at', ''))
            sid = s.get('id', 0)

            content = f"""---
scan_id: {sid}
date: {date}
elapsed_secs: {s.get('elapsed_secs', 0)}
items_fetched: {s.get('items_fetched', 0)}
items_scored: {s.get('items_scored', 0)}
critical: {s.get('critical', 0)}
high: {s.get('high', 0)}
medium: {s.get('medium', 0)}
noise: {s.get('noise', 0)}
type: scan
---

# Scan #{sid} — {date}

| Metric | Value |
|--------|-------|
| Started | {s.get('started_at', '')} |
| Duration | {s.get('elapsed_secs', 0):.1f}s |
| Fetched | {s.get('items_fetched', 0)} |
| Scored | {s.get('items_scored', 0)} |
| Critical | {s.get('critical', 0)} |
| High | {s.get('high', 0)} |
| Medium | {s.get('medium', 0)} |
| Noise | {s.get('noise', 0)} |
| Error | {s.get('error', 'None')} |
"""
            self.write_file(f"scans/scan-{sid}.md", content)

        # Daily rollups from articles
        daily = defaultdict(list)
        for a in self.data.all_articles:
            date = iso_date(a.get('timestamp', ''))
            if date:
                daily[date].append(a)

        for date, items in sorted(daily.items()):
            scores = [i.get('score', 0) or 0 for i in items]
            above_min = [s for s in scores if s >= self.data.min_score]

            # Top articles for the day
            top = sorted(items, key=lambda x: -(x.get('score', 0) or 0))[:10]
            top_lines = '\n'.join(
                f"| [[articles/{sanitize_filename(a.get('title', ''))}\\|{(a.get('title', '') or 'Untitled')[:60]}]] | {a.get('score', 0)} | {a.get('category', '')} |"
                for a in top if (a.get('score', 0) or 0) >= self.data.min_score
            )

            content = f"""---
date: {date}
total_articles: {len(items)}
relevant_articles: {len(above_min)}
avg_score: {round(sum(scores)/len(scores), 1) if scores else 0}
max_score: {max(scores) if scores else 0}
type: daily
---

# Daily Rollup — {date}

| Metric | Value |
|--------|-------|
| Total Articles | {len(items)} |
| Relevant (score >= {self.data.min_score}) | {len(above_min)} |
| Average Score | {round(sum(scores)/len(scores), 1) if scores else 0} |
| Best Score | {max(scores) if scores else 0} |

## Top Articles
| Article | Score | Category |
|---------|-------|----------|
{top_lines}

## All Articles This Day
```dataview
TABLE score, category, domain
FROM "articles"
WHERE date = date("{date}")
SORT score DESC
```
"""
            self.write_file(f"daily/{date}.md", content)

    # ── Phase 8: Dashboards ────────────────────────────────────────────────

    def gen_dashboards(self):
        self._gen_index()
        self._gen_dashboard_scores()
        self._gen_dashboard_sources()
        self._gen_dashboard_entities()
        self._gen_dashboard_feedback()
        self._gen_dashboard_coverage()
        self._gen_dashboard_timeline()

    def _gen_index(self):
        profile = self.data.profiles.get(self.data.profile_id, {})
        pname = profile.get('name', f'Profile {self.data.profile_id}')

        total = len(self.data.all_articles)
        relevant = len(self.data.articles)
        cats = len(set(a.get('category', '') for a in self.data.all_articles))

        content = f"""---
type: index
---

# StratOS Intelligence Vault
**Profile:** {pname} | **Articles:** {total} | **Relevant (>= {self.data.min_score}):** {relevant} | **Categories:** {cats}

---

## Quick Stats
```dataview
TABLE WITHOUT ID
  length(rows) as "Total",
  round(sum(rows.score) / length(rows), 1) as "Avg Score",
  length(filter(rows, (r) => r.score >= 9.0)) as "Critical",
  length(filter(rows, (r) => r.score >= 7.0 AND r.score < 9.0)) as "High",
  length(filter(rows, (r) => r.score >= 5.0 AND r.score < 7.0)) as "Medium"
FROM "articles"
GROUP BY true
```

## Dashboards
- [[_dashboard-scores|Score Analytics]]
- [[_dashboard-sources|Source Quality]]
- [[_dashboard-entities|Entity Intelligence]]
- [[_dashboard-feedback|Feedback & Disagreements]]
- [[_dashboard-coverage|Category Coverage]]
- [[_dashboard-timeline|Timeline Analysis]]
- [Interactive Dashboard](analytics/dashboard.html)

## Browse
- **Articles** — `articles/` ({relevant} notes)
- **Entities** — `entities/`
- **Categories** — `categories/` ({cats} categories)
- **Sources** — `sources/`
- **Daily Rollups** — `daily/`
- **Scans** — `scans/`
- **Scholarly** — `scholarly/`

## Recent High-Value Articles
```dataview
TABLE score, category, domain, date
FROM "articles"
WHERE score >= 8.0
SORT date DESC
LIMIT 15
```

## Latest Scans
```dataview
TABLE date, items_fetched, critical, high, medium, noise
FROM "scans"
SORT scan_id DESC
LIMIT 5
```
"""
        self.write_file("_index.md", content)

    def _gen_dashboard_scores(self):
        content = """---
type: dashboard
---

# Score Analytics Dashboard

## Score Distribution
```dataviewjs
const articles = dv.pages('"articles"');
const tiers = {critical: 0, high: 0, medium: 0, noise: 0};
articles.forEach(a => {
  if (a.score >= 9) tiers.critical++;
  else if (a.score >= 7) tiers.high++;
  else if (a.score >= 5) tiers.medium++;
  else tiers.noise++;
});
const max = Math.max(...Object.values(tiers));
let out = "| Tier | Count | Bar |\\n|------|-------|-----|\\n";
for (const [tier, count] of Object.entries(tiers)) {
  const bar = "█".repeat(Math.round(count / max * 30));
  out += `| ${tier} | ${count} | ${bar} |\\n`;
}
dv.paragraph(out);
```

## Score Trend (by Date)
```dataviewjs
const articles = dv.pages('"articles"');
const byDate = {};
articles.forEach(a => {
  const d = a.date;
  if (!d) return;
  const key = d.toString().slice(0, 10);
  if (!byDate[key]) byDate[key] = {sum: 0, count: 0};
  byDate[key].sum += a.score;
  byDate[key].count++;
});
const sorted = Object.entries(byDate).sort();
let out = "| Date | Avg Score | Articles |\\n|------|-----------|----------|\\n";
sorted.slice(-14).forEach(([date, {sum, count}]) => {
  out += `| ${date} | ${(sum/count).toFixed(1)} | ${count} |\\n`;
});
dv.paragraph(out);
```

## Best Scoring Days
```dataview
TABLE WITHOUT ID
  date as "Date",
  round(sum(rows.score) / length(rows), 1) as "Avg Score",
  length(rows) as "Articles",
  max(rows.score) as "Best"
FROM "articles"
GROUP BY date
SORT round(sum(rows.score) / length(rows), 1) DESC
LIMIT 10
```

## Top 20 Articles (All Time)
```dataview
TABLE score, category, domain, date
FROM "articles"
SORT score DESC
LIMIT 20
```
"""
        self.write_file("_dashboard-scores.md", content)

    def _gen_dashboard_sources(self):
        content = """---
type: dashboard
---

# Source Quality Dashboard

## Source Ranking
```dataview
TABLE WITHOUT ID
  domain as "Source",
  length(rows) as "Articles",
  round(sum(rows.score) / length(rows), 1) as "Avg Score",
  round(max(rows.score), 1) as "Best",
  round(min(rows.score), 1) as "Worst"
FROM "articles"
GROUP BY domain
SORT round(sum(rows.score) / length(rows), 1) DESC
LIMIT 30
```

## High-Value Sources (Avg >= 7.5)
```dataviewjs
const articles = dv.pages('"articles"');
const byDomain = {};
articles.forEach(a => {
  const d = a.domain || "unknown";
  if (!byDomain[d]) byDomain[d] = {scores: [], count: 0};
  byDomain[d].scores.push(a.score);
  byDomain[d].count++;
});
let out = "| Source | Articles | Avg Score |\\n|--------|----------|-----------|\\n";
Object.entries(byDomain)
  .map(([d, v]) => ({domain: d, avg: v.scores.reduce((a,b)=>a+b,0)/v.count, count: v.count}))
  .filter(x => x.avg >= 7.5 && x.count >= 2)
  .sort((a, b) => b.avg - a.avg)
  .forEach(x => { out += `| ${x.domain} | ${x.count} | ${x.avg.toFixed(1)} |\\n`; });
dv.paragraph(out);
```

## One-Hit Wonders (Single Article, Score >= 8)
```dataview
TABLE score, category, date
FROM "articles"
WHERE score >= 8.0
GROUP BY domain
FLATTEN length(rows) as cnt
WHERE cnt = 1
SORT score DESC
LIMIT 15
```
"""
        self.write_file("_dashboard-sources.md", content)

    def _gen_dashboard_entities(self):
        content = """---
type: dashboard
---

# Entity Intelligence Dashboard

## Most Connected Entities
```dataview
TABLE mention_count, article_count, avg_score, entity_category
FROM "entities"
SORT mention_count DESC
LIMIT 30
```

## Rising Entities (Recent & High Score)
```dataview
TABLE mention_count, article_count, avg_score, last_seen
FROM "entities"
WHERE avg_score >= 7.0
SORT last_seen DESC
LIMIT 20
```

## Entity Category Breakdown
```dataviewjs
const entities = dv.pages('"entities"');
const byCat = {};
entities.forEach(e => {
  const cat = e.entity_category || "uncategorized";
  if (!byCat[cat]) byCat[cat] = {count: 0, mentions: 0};
  byCat[cat].count++;
  byCat[cat].mentions += (e.mention_count || 0);
});
let out = "| Category | Entities | Total Mentions |\\n|----------|----------|----------------|\\n";
Object.entries(byCat)
  .sort((a, b) => b[1].mentions - a[1].mentions)
  .forEach(([cat, v]) => { out += `| ${cat} | ${v.count} | ${v.mentions} |\\n`; });
dv.paragraph(out);
```

## Entity Graph View
> Open **Graph View** (Ctrl/Cmd+G) and filter to `path:entities` to see the entity relationship network. Entities are linked via co-occurrence in articles.
"""
        self.write_file("_dashboard-entities.md", content)

    def _gen_dashboard_feedback(self):
        content = """---
type: dashboard
---

# Feedback & Disagreement Dashboard

## All Feedback
```dataview
TABLE original_score, user_score, delta, delta_direction, action, timestamp
FROM "feedback"
SORT timestamp DESC
```

## Underscored (AI too low)
```dataview
TABLE original_score, user_score, delta, article_title
FROM "feedback"
WHERE delta > 0
SORT delta DESC
```

## Overscored (AI too high)
```dataview
TABLE original_score, user_score, delta, article_title
FROM "feedback"
WHERE delta < 0
SORT delta ASC
```

## Feedback Stats
```dataviewjs
const fb = dv.pages('"feedback"');
const total = fb.length;
const under = fb.filter(f => f.delta > 0).length;
const over = fb.filter(f => f.delta < 0).length;
const agreed = fb.filter(f => f.delta === 0).length;
const avgDelta = fb.length > 0 ? (fb.map(f => Math.abs(f.delta)).reduce((a,b)=>a+b,0) / total).toFixed(1) : 0;
dv.paragraph(`**Total Feedback:** ${total} | **Underscored:** ${under} | **Overscored:** ${over} | **Agreed:** ${agreed} | **Avg |Delta|:** ${avgDelta}`);
```
"""
        self.write_file("_dashboard-feedback.md", content)

    def _gen_dashboard_coverage(self):
        content = """---
type: dashboard
---

# Category Coverage Dashboard

## Category Breakdown
```dataview
TABLE WITHOUT ID
  category as "Category",
  length(rows) as "Articles",
  round(sum(rows.score) / length(rows), 1) as "Avg Score",
  max(rows.date) as "Latest",
  length(filter(rows, (r) => r.score >= 9.0)) as "Critical",
  length(filter(rows, (r) => r.score >= 7.0 AND r.score < 9.0)) as "High"
FROM "articles"
GROUP BY category
SORT length(rows) DESC
```

## Coverage Gaps (No articles in 7+ days)
```dataviewjs
const articles = dv.pages('"articles"');
const byCat = {};
articles.forEach(a => {
  const cat = a.category || "unknown";
  const d = a.date ? a.date.toString().slice(0,10) : "";
  if (!byCat[cat] || d > byCat[cat]) byCat[cat] = d;
});
const today = new Date().toISOString().slice(0,10);
const gaps = Object.entries(byCat)
  .map(([cat, last]) => {
    const diff = Math.floor((new Date(today) - new Date(last)) / 86400000);
    return {cat, last, diff};
  })
  .filter(x => x.diff >= 7)
  .sort((a, b) => b.diff - a.diff);
if (gaps.length === 0) {
  dv.paragraph("All categories have recent coverage.");
} else {
  let out = "| Category | Last Article | Days Ago |\\n|----------|--------------|----------|\\n";
  gaps.forEach(g => { out += `| [[categories/${g.cat}]] | ${g.last} | ${g.diff} |\\n`; });
  dv.paragraph(out);
}
```

## Category Volume (ASCII Chart)
```dataviewjs
const articles = dv.pages('"articles"');
const byCat = {};
articles.forEach(a => {
  const cat = a.category || "unknown";
  byCat[cat] = (byCat[cat] || 0) + 1;
});
const sorted = Object.entries(byCat).sort((a, b) => b[1] - a[1]);
const max = sorted[0] ? sorted[0][1] : 1;
let out = "```\\n";
sorted.forEach(([cat, count]) => {
  const bar = "█".repeat(Math.round(count / max * 40));
  out += `${cat.padEnd(30)} ${bar} ${count}\\n`;
});
out += "```";
dv.paragraph(out);
```
"""
        self.write_file("_dashboard-coverage.md", content)

    def _gen_dashboard_timeline(self):
        content = """---
type: dashboard
---

# Timeline Analysis Dashboard

## Daily Volume (Last 30 Days)
```dataviewjs
const articles = dv.pages('"articles"');
const byDate = {};
articles.forEach(a => {
  const d = a.date ? a.date.toString().slice(0,10) : "";
  if (!d) return;
  byDate[d] = (byDate[d] || 0) + 1;
});
const sorted = Object.entries(byDate).sort().slice(-30);
const max = Math.max(...sorted.map(x => x[1]));
let out = "| Date | Articles | Volume |\\n|------|----------|--------|\\n";
sorted.forEach(([date, count]) => {
  const bar = "█".repeat(Math.round(count / max * 25));
  out += `| ${date} | ${count} | ${bar} |\\n`;
});
dv.paragraph(out);
```

## High-Value Articles Timeline
```dataview
TABLE score, category, domain
FROM "articles"
WHERE score >= 8.0
SORT date DESC
LIMIT 30
```

## Hourly Distribution of High-Value Articles
```dataviewjs
const articles = dv.pages('"articles"');
const byHour = new Array(24).fill(0);
articles.filter(a => a.score >= 7.0).forEach(a => {
  const ts = a.timestamp;
  if (!ts) return;
  const h = parseInt(ts.toString().slice(11, 13));
  if (!isNaN(h)) byHour[h]++;
});
const max = Math.max(...byHour);
let out = "| Hour | Count | Distribution |\\n|------|-------|--------------|\\n";
byHour.forEach((count, hour) => {
  const bar = max > 0 ? "█".repeat(Math.round(count / max * 20)) : "";
  out += `| ${hour.toString().padStart(2, '0')}:00 | ${count} | ${bar} |\\n`;
});
dv.paragraph(out);
```

## Weekly Summary
```dataview
TABLE WITHOUT ID
  dateformat(date, "yyyy-'W'WW") as "Week",
  length(rows) as "Articles",
  round(sum(rows.score) / length(rows), 1) as "Avg Score",
  length(filter(rows, (r) => r.score >= 8.0)) as "High Value"
FROM "articles"
GROUP BY dateformat(date, "yyyy-'W'WW")
SORT dateformat(date, "yyyy-'W'WW") DESC
LIMIT 12
```
"""
        self.write_file("_dashboard-timeline.md", content)

    # ── Setup ──────────────────────────────────────────────────────────────

    def gen_setup(self):
        content = """# StratOS Obsidian Vault — Setup Guide

## Quick Start
1. Open Obsidian
2. **Open Vault** → select this folder (`stratos_vault`)
3. Go to **Settings → Community Plugins → Turn on community plugins**
4. **Browse** → search **Dataview** → Install → Enable
5. In Dataview settings, **enable JavaScript queries** (for DataviewJS blocks)
6. (Optional) Install **Charts** plugin for additional visualizations

## Core Plugins to Enable
- **Graph View** — see entity relationships and article clusters
- **Backlinks** — see which articles reference each entity
- **Tags** — browse by frontmatter tags

## Graph View Tips
- Open with `Ctrl/Cmd + G`
- Filter by path: `path:entities` for entity network
- Filter by path: `path:articles` for article clusters
- Color groups by folder for visual distinction
- Increase node size for better readability

## Interactive Dashboard
Open `analytics/dashboard.html` in your browser for interactive D3/Plotly charts with:
- Entity relationship graph (force-directed)
- Score distribution & trends
- Source quality heatmap
- Category radar chart
- Timeline view

## Vault Structure
- `_index.md` — Start here! Master dashboard
- `_dashboard-*.md` — Analytics dashboards (Dataview-powered)
- `articles/` — One note per article (scored >= 6.0)
- `entities/` — One note per entity (auto-linked)
- `categories/` — Hub pages per category
- `sources/` — One note per source domain
- `daily/` — Daily rollup summaries
- `scans/` — Individual scan reports
- `feedback/` — User feedback on scoring
- `scholarly/` — YouTube video insights
- `analytics/` — Interactive HTML dashboard + chart data

## Re-export
```bash
cd ~/Downloads/StratOS/StratOS1/backend
python3 obsidian_export.py -v                    # Full re-export
python3 obsidian_export.py --incremental -v      # Only new items
```
"""
        self.write_file("SETUP.md", content)


# ── HTML Dashboard Generator ──────────────────────────────────────────────────

class DashboardGenerator:
    def __init__(self, data: StratOSData, vault_path: Path, verbose=False):
        self.data = data
        self.vault = vault_path
        self.verbose = verbose

    def generate(self):
        # Pre-compute all chart data
        chart_data = self._compute_chart_data()

        # Write JSON data files
        analytics_dir = self.vault / "analytics"
        analytics_dir.mkdir(parents=True, exist_ok=True)

        for key in ['score_distribution', 'source_quality', 'category_radar', 'timeline', 'entity_graph', 'feedback_scatter']:
            (analytics_dir / f"{key.replace('_', '-')}.json").write_text(
                json.dumps(chart_data[key], ensure_ascii=False, indent=2), encoding='utf-8'
            )

        # Generate HTML dashboard
        html = self._build_html(chart_data)
        (analytics_dir / "dashboard.html").write_text(html, encoding='utf-8')

        if self.verbose:
            print(f"  Dashboard generated: {analytics_dir / 'dashboard.html'}")

    def _compute_chart_data(self):
        data = {}

        # 1. Score distribution
        scores = [a.get('score', 0) or 0 for a in self.data.articles]
        hist = defaultdict(int)
        for s in scores:
            bucket = int(s)
            hist[bucket] += 1

        by_date = defaultdict(list)
        for a in self.data.articles:
            d = iso_date(a.get('timestamp', ''))
            if d:
                by_date[d].append(a.get('score', 0) or 0)

        score_trend = [
            {"date": d, "avg": round(sum(s)/len(s), 2), "count": len(s)}
            for d, s in sorted(by_date.items())
        ]

        data['score_distribution'] = {
            "histogram": dict(sorted(hist.items())),
            "trend": score_trend,
            "tiers": {
                "critical": sum(1 for s in scores if s >= 9),
                "high": sum(1 for s in scores if 7 <= s < 9),
                "medium": sum(1 for s in scores if 5 <= s < 7),
                "noise": sum(1 for s in scores if s < 5),
            }
        }

        # 2. Source quality
        by_source = defaultdict(list)
        for a in self.data.articles:
            domain = extract_domain(a.get('url', ''))
            by_source[domain].append({
                "score": a.get('score', 0) or 0,
                "category": a.get('category', ''),
            })

        source_data = []
        for domain, items in by_source.items():
            scs = [i['score'] for i in items]
            cats = list(set(i['category'] for i in items))
            source_data.append({
                "domain": domain,
                "count": len(items),
                "avg_score": round(sum(scs)/len(scs), 2),
                "categories": cats,
            })
        source_data.sort(key=lambda x: -x['count'])
        data['source_quality'] = source_data[:50]

        # 3. Category radar
        by_cat = defaultdict(list)
        for a in self.data.articles:
            cat = a.get('category', 'unknown') or 'unknown'
            by_cat[cat].append(a.get('score', 0) or 0)

        cat_data = []
        for cat, scs in by_cat.items():
            cat_data.append({
                "category": cat,
                "count": len(scs),
                "avg_score": round(sum(scs)/len(scs), 2),
                "composite": round(len(scs) * sum(scs)/len(scs), 1),
            })
        data['category_radar'] = sorted(cat_data, key=lambda x: -x['composite'])

        # 4. Timeline
        timeline = []
        for a in self.data.articles:
            timeline.append({
                "title": (a.get('title', '') or '')[:80],
                "score": a.get('score', 0) or 0,
                "date": iso_date(a.get('timestamp', '')),
                "category": a.get('category', ''),
                "domain": extract_domain(a.get('url', '')),
                "url": a.get('url', ''),
            })
        data['timeline'] = timeline

        # 5. Entity graph
        nodes = []
        edges = []
        entity_set = set()

        for ename, cooccur in self.data.entity_cooccurrence.items():
            entity_set.add(ename)
            for other, count in cooccur.items():
                entity_set.add(other)
                if ename < other:  # avoid duplicates
                    edges.append({"source": ename, "target": other, "weight": count})

        for ename in entity_set:
            mentions = self.data.entity_mentions.get(ename, [])
            total = sum(m['mention_count'] for m in mentions)
            einfo = self.data.entities_by_name.get(ename, {})
            nodes.append({
                "name": ename,
                "mentions": total,
                "category": einfo.get('category', 'auto'),
                "articles": len(self.data.entity_articles.get(ename, [])),
            })

        data['entity_graph'] = {"nodes": nodes, "edges": edges}

        # 6. Feedback scatter
        fb_data = []
        for fb in self.data.feedback:
            fb_data.append({
                "title": fb.get('title', ''),
                "ai_score": fb.get('ai_score', 0),
                "user_score": fb.get('user_score'),
                "action": fb.get('action', ''),
                "category": fb.get('category', ''),
            })
        data['feedback_scatter'] = fb_data

        return data

    def _build_html(self, chart_data):
        # Embed all data as JSON
        entity_graph_json = json.dumps(chart_data['entity_graph'], ensure_ascii=False)
        score_dist_json = json.dumps(chart_data['score_distribution'], ensure_ascii=False)
        source_quality_json = json.dumps(chart_data['source_quality'], ensure_ascii=False)
        category_radar_json = json.dumps(chart_data['category_radar'], ensure_ascii=False)
        timeline_json = json.dumps(chart_data['timeline'], ensure_ascii=False)
        feedback_json = json.dumps(chart_data['feedback_scatter'], ensure_ascii=False)

        profile = self.data.profiles.get(self.data.profile_id, {})
        pname = profile.get('name', f'Profile {self.data.profile_id}')

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>StratOS Analytics — {pname}</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root {{
  --bg: #0a0e17;
  --bg2: #111827;
  --bg3: #1a2332;
  --fg: #e2e8f0;
  --fg2: #94a3b8;
  --accent: #3b82f6;
  --critical: #ef4444;
  --high: #f97316;
  --medium: #eab308;
  --noise: #6b7280;
  --border: #1e293b;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--fg); display: flex; height: 100vh; overflow: hidden; }}
nav {{ width: 220px; min-width: 220px; background: var(--bg2); border-right: 1px solid var(--border); display: flex; flex-direction: column; padding: 20px 0; }}
nav h1 {{ font-size: 16px; padding: 0 20px 20px; color: var(--accent); border-bottom: 1px solid var(--border); margin-bottom: 10px; }}
nav h1 span {{ font-size: 11px; color: var(--fg2); display: block; margin-top: 4px; }}
nav button {{ display: block; width: 100%; text-align: left; padding: 12px 20px; background: none; border: none; color: var(--fg2); cursor: pointer; font-size: 13px; transition: all 0.15s; }}
nav button:hover {{ background: var(--bg3); color: var(--fg); }}
nav button.active {{ background: var(--bg3); color: var(--accent); border-left: 3px solid var(--accent); }}
main {{ flex: 1; display: flex; overflow: hidden; }}
.chart-area {{ flex: 1; padding: 24px; overflow-y: auto; }}
.chart-area h2 {{ font-size: 20px; margin-bottom: 16px; color: var(--fg); }}
.chart-area .subtitle {{ color: var(--fg2); font-size: 13px; margin-bottom: 20px; }}
.sidebar {{ width: 320px; min-width: 320px; background: var(--bg2); border-left: 1px solid var(--border); padding: 20px; overflow-y: auto; display: none; }}
.sidebar.visible {{ display: block; }}
.sidebar h3 {{ font-size: 14px; margin-bottom: 12px; color: var(--accent); }}
.sidebar .detail {{ margin-bottom: 8px; font-size: 13px; }}
.sidebar .detail label {{ color: var(--fg2); display: block; font-size: 11px; margin-bottom: 2px; }}
.card {{ background: var(--bg3); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 12px; }}
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 20px; }}
.stat {{ text-align: center; }}
.stat .value {{ font-size: 28px; font-weight: 700; }}
.stat .label {{ font-size: 11px; color: var(--fg2); margin-top: 4px; }}
.stat.critical .value {{ color: var(--critical); }}
.stat.high .value {{ color: var(--high); }}
.stat.medium .value {{ color: var(--medium); }}
#chart {{ width: 100%; min-height: 500px; }}
.plotly-chart {{ width: 100%; height: 550px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
th {{ color: var(--fg2); font-weight: 600; position: sticky; top: 0; background: var(--bg3); }}
tr:hover {{ background: rgba(59,130,246,0.05); }}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.filter-bar {{ display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }}
.filter-bar input, .filter-bar select {{ background: var(--bg); border: 1px solid var(--border); color: var(--fg); padding: 6px 10px; border-radius: 4px; font-size: 13px; }}
.tier-badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
.tier-critical {{ background: rgba(239,68,68,0.2); color: var(--critical); }}
.tier-high {{ background: rgba(249,115,22,0.2); color: var(--high); }}
.tier-medium {{ background: rgba(234,179,8,0.2); color: var(--medium); }}
.tier-noise {{ background: rgba(107,114,128,0.2); color: var(--noise); }}
.hidden {{ display: none !important; }}
</style>
</head>
<body>
<nav>
  <h1>StratOS Analytics<span>{pname}</span></h1>
  <button class="active" onclick="showView('entities')" id="nav-entities">Entity Graph</button>
  <button onclick="showView('scores')" id="nav-scores">Score Distribution</button>
  <button onclick="showView('sources')" id="nav-sources">Source Quality</button>
  <button onclick="showView('feedback')" id="nav-feedback">Feedback Scatter</button>
  <button onclick="showView('radar')" id="nav-radar">Category Radar</button>
  <button onclick="showView('timeline')" id="nav-timeline">Timeline</button>
</nav>
<main>
<div class="chart-area" id="content">
  <div id="view-entities">
    <h2>Entity Relationship Graph</h2>
    <p class="subtitle">Entities connected by co-occurrence in articles. Node size = mentions, edge thickness = shared articles.</p>
    <div class="filter-bar">
      <input type="text" id="entity-search" placeholder="Search entities..." oninput="filterEntityGraph()">
      <label style="color:var(--fg2);font-size:13px;display:flex;align-items:center;gap:4px;">
        Min co-occurrence: <input type="range" id="min-cooccur" min="1" max="10" value="1" oninput="filterEntityGraph()"><span id="cooccur-val">1</span>
      </label>
    </div>
    <div id="chart"></div>
  </div>
  <div id="view-scores" class="hidden">
    <h2>Score Distribution</h2>
    <p class="subtitle">Article score distribution and trends over time.</p>
    <div class="stats-grid" id="score-stats"></div>
    <div class="filter-bar">
      <button onclick="showScoreView('histogram')" id="btn-hist" style="background:var(--accent);color:white;border:none;padding:6px 14px;border-radius:4px;cursor:pointer;">Histogram</button>
      <button onclick="showScoreView('trend')" id="btn-trend" style="background:var(--bg);color:var(--fg2);border:1px solid var(--border);padding:6px 14px;border-radius:4px;cursor:pointer;">Trend</button>
    </div>
    <div id="score-chart" class="plotly-chart"></div>
  </div>
  <div id="view-sources" class="hidden">
    <h2>Source Quality Heatmap</h2>
    <p class="subtitle">Sources ranked by article count and average score.</p>
    <div class="filter-bar">
      <label style="color:var(--fg2);font-size:13px;display:flex;align-items:center;gap:4px;">
        Min articles: <input type="number" id="min-articles" value="2" min="1" max="50" onchange="renderSources()" style="width:60px;">
      </label>
    </div>
    <div id="source-chart" class="plotly-chart"></div>
  </div>
  <div id="view-feedback" class="hidden">
    <h2>Feedback Disagreement</h2>
    <p class="subtitle">AI score vs user score. Above diagonal = underscored, below = overscored.</p>
    <div id="feedback-chart" class="plotly-chart"></div>
  </div>
  <div id="view-radar" class="hidden">
    <h2>Category Radar</h2>
    <p class="subtitle">Category coverage by volume and quality.</p>
    <div class="filter-bar">
      <button onclick="renderRadar('volume')" id="btn-vol" style="background:var(--accent);color:white;border:none;padding:6px 14px;border-radius:4px;cursor:pointer;">Volume</button>
      <button onclick="renderRadar('quality')" id="btn-qual" style="background:var(--bg);color:var(--fg2);border:1px solid var(--border);padding:6px 14px;border-radius:4px;cursor:pointer;">Quality</button>
    </div>
    <div id="radar-chart" class="plotly-chart"></div>
  </div>
  <div id="view-timeline" class="hidden">
    <h2>Timeline</h2>
    <p class="subtitle">Articles plotted by date and score. Size = relative importance.</p>
    <div id="timeline-chart" class="plotly-chart"></div>
  </div>
</div>
<div class="sidebar" id="sidebar">
  <h3 id="sidebar-title">Details</h3>
  <div id="sidebar-content"></div>
</div>
</main>

<script>
// ── Data ──
const entityGraph = {entity_graph_json};
const scoreDist = {score_dist_json};
const sourceQuality = {source_quality_json};
const categoryRadar = {category_radar_json};
const timelineData = {timeline_json};
const feedbackData = {feedback_json};

const tierColors = {{critical:'#ef4444', high:'#f97316', medium:'#eab308', noise:'#6b7280'}};
function scoreTier(s) {{ return s >= 9 ? 'critical' : s >= 7 ? 'high' : s >= 5 ? 'medium' : 'noise'; }}

// ── Navigation ──
let currentView = 'entities';
function showView(view) {{
  document.querySelectorAll('[id^="view-"]').forEach(el => el.classList.add('hidden'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById('view-' + view).classList.remove('hidden');
  document.getElementById('nav-' + view).classList.add('active');
  document.getElementById('sidebar').classList.remove('visible');
  currentView = view;
  if (view === 'entities' && !window._entityGraphRendered) renderEntityGraph();
  if (view === 'scores') renderScores();
  if (view === 'sources') renderSources();
  if (view === 'feedback') renderFeedback();
  if (view === 'radar') renderRadar('volume');
  if (view === 'timeline') renderTimeline();
}}

// ── Sidebar ──
function showSidebar(title, html) {{
  document.getElementById('sidebar-title').textContent = title;
  document.getElementById('sidebar-content').innerHTML = html;
  document.getElementById('sidebar').classList.add('visible');
}}

// ── 1. Entity Graph (D3) ──
let simulation, svg, link, node, labels;
function renderEntityGraph() {{
  window._entityGraphRendered = true;
  const container = document.getElementById('chart');
  container.innerHTML = '';
  const width = container.clientWidth;
  const height = 550;

  const nodes = entityGraph.nodes.map(d => ({{...d}}));
  const edges = entityGraph.edges.filter(e => e.weight >= 1).map(d => ({{...d}}));

  if (nodes.length === 0) {{
    container.innerHTML = '<p style="color:var(--fg2);padding:40px;">No entity co-occurrence data to display.</p>';
    return;
  }}

  const nodeMap = {{}};
  nodes.forEach((n,i) => nodeMap[n.name] = i);
  const links = edges.filter(e => nodeMap[e.source] !== undefined && nodeMap[e.target] !== undefined)
    .map(e => ({{source: nodeMap[e.source], target: nodeMap[e.target], weight: e.weight}}));

  const mentionScale = d3.scaleSqrt().domain([0, d3.max(nodes, d=>d.mentions)||1]).range([4, 25]);
  const catColors = d3.scaleOrdinal(d3.schemeTableau10);

  svg = d3.select(container).append('svg').attr('width', width).attr('height', height);
  const g = svg.append('g');

  svg.call(d3.zoom().scaleExtent([0.2, 5]).on('zoom', e => g.attr('transform', e.transform)));

  link = g.selectAll('line').data(links).join('line')
    .attr('stroke', '#334155').attr('stroke-opacity', 0.5)
    .attr('stroke-width', d => Math.min(d.weight, 6));

  node = g.selectAll('circle').data(nodes).join('circle')
    .attr('r', d => mentionScale(d.mentions))
    .attr('fill', d => catColors(d.category))
    .attr('stroke', '#1e293b').attr('stroke-width', 1)
    .attr('cursor', 'pointer')
    .call(d3.drag().on('start', dragstart).on('drag', dragged).on('end', dragend))
    .on('click', (e, d) => {{
      showSidebar(d.name, `
        <div class="detail"><label>Category</label>${{d.category}}</div>
        <div class="detail"><label>Mentions</label>${{d.mentions}}</div>
        <div class="detail"><label>Articles</label>${{d.articles}}</div>
      `);
    }});

  node.append('title').text(d => `${{d.name}} (${{d.mentions}} mentions)`);

  labels = g.selectAll('text').data(nodes.filter(d => d.mentions > 5)).join('text')
    .text(d => d.name.length > 20 ? d.name.slice(0,18)+'…' : d.name)
    .attr('font-size', '10px').attr('fill', '#94a3b8').attr('text-anchor', 'middle')
    .attr('dy', d => -mentionScale(d.mentions) - 4);

  simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).distance(80))
    .force('charge', d3.forceManyBody().strength(-120))
    .force('center', d3.forceCenter(width/2, height/2))
    .force('collision', d3.forceCollide(d => mentionScale(d.mentions) + 3))
    .on('tick', () => {{
      link.attr('x1', d=>d.source.x).attr('y1', d=>d.source.y).attr('x2', d=>d.target.x).attr('y2', d=>d.target.y);
      node.attr('cx', d=>d.x).attr('cy', d=>d.y);
      labels.attr('x', d=>d.x).attr('y', d=>d.y);
    }});

  function dragstart(e, d) {{ if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }}
  function dragged(e, d) {{ d.fx = e.x; d.fy = e.y; }}
  function dragend(e, d) {{ if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}
}}

function filterEntityGraph() {{
  const q = document.getElementById('entity-search').value.toLowerCase();
  const minC = parseInt(document.getElementById('min-cooccur').value);
  document.getElementById('cooccur-val').textContent = minC;
  // Re-render with filter
  window._entityGraphRendered = false;
  const filtered = {{
    nodes: entityGraph.nodes.filter(n => !q || n.name.toLowerCase().includes(q)),
    edges: entityGraph.edges.filter(e => e.weight >= minC)
  }};
  const origGraph = entityGraph;
  Object.assign(entityGraph, filtered);
  renderEntityGraph();
  Object.assign(entityGraph, origGraph);
}}

// ── 2. Scores ──
let scoreViewMode = 'histogram';
function showScoreView(mode) {{
  scoreViewMode = mode;
  document.getElementById('btn-hist').style.background = mode === 'histogram' ? 'var(--accent)' : 'var(--bg)';
  document.getElementById('btn-hist').style.color = mode === 'histogram' ? 'white' : 'var(--fg2)';
  document.getElementById('btn-trend').style.background = mode === 'trend' ? 'var(--accent)' : 'var(--bg)';
  document.getElementById('btn-trend').style.color = mode === 'trend' ? 'white' : 'var(--fg2)';
  renderScores();
}}
function renderScores() {{
  const t = scoreDist.tiers;
  document.getElementById('score-stats').innerHTML = `
    <div class="card stat critical"><div class="value">${{t.critical}}</div><div class="label">Critical (9+)</div></div>
    <div class="card stat high"><div class="value">${{t.high}}</div><div class="label">High (7-8.9)</div></div>
    <div class="card stat medium"><div class="value">${{t.medium}}</div><div class="label">Medium (5-6.9)</div></div>
    <div class="card stat"><div class="value" style="color:var(--noise)">${{t.noise}}</div><div class="label">Noise (&lt;5)</div></div>
  `;
  if (scoreViewMode === 'histogram') {{
    const keys = Object.keys(scoreDist.histogram).map(Number).sort((a,b)=>a-b);
    const vals = keys.map(k => scoreDist.histogram[k]);
    const colors = keys.map(k => tierColors[scoreTier(k)]);
    Plotly.newPlot('score-chart', [{{x: keys, y: vals, type: 'bar', marker: {{color: colors}}}}],
      {{paper_bgcolor:'transparent', plot_bgcolor:'transparent', font:{{color:'#94a3b8'}},
        xaxis:{{title:'Score', gridcolor:'#1e293b'}}, yaxis:{{title:'Count', gridcolor:'#1e293b'}}, margin:{{t:20}}}},
      {{responsive:true}});
  }} else {{
    const dates = scoreDist.trend.map(d=>d.date);
    const avgs = scoreDist.trend.map(d=>d.avg);
    const counts = scoreDist.trend.map(d=>d.count);
    Plotly.newPlot('score-chart', [
      {{x:dates, y:avgs, type:'scatter', mode:'lines+markers', name:'Avg Score', line:{{color:'#3b82f6'}}}},
      {{x:dates, y:counts, type:'bar', name:'Article Count', yaxis:'y2', marker:{{color:'rgba(59,130,246,0.2)'}}}},
    ], {{paper_bgcolor:'transparent', plot_bgcolor:'transparent', font:{{color:'#94a3b8'}},
      xaxis:{{gridcolor:'#1e293b'}}, yaxis:{{title:'Avg Score', gridcolor:'#1e293b'}},
      yaxis2:{{title:'Count', overlaying:'y', side:'right', gridcolor:'#1e293b'}},
      margin:{{t:20}}, legend:{{x:0,y:1.1,orientation:'h'}}}}, {{responsive:true}});
  }}
}}

// ── 3. Sources ──
function renderSources() {{
  const minA = parseInt(document.getElementById('min-articles')?.value || 2);
  const filtered = sourceQuality.filter(s => s.count >= minA).slice(0, 30);
  const domains = filtered.map(s => s.domain);
  const counts = filtered.map(s => s.count);
  const avgs = filtered.map(s => s.avg_score);
  const colors = avgs.map(a => tierColors[scoreTier(a)]);

  Plotly.newPlot('source-chart', [{{
    x: avgs, y: domains, type: 'bar', orientation: 'h',
    marker: {{color: colors}},
    text: counts.map(c => c + ' articles'),
    hovertemplate: '%{{y}}<br>Avg: %{{x:.1f}}<br>%{{text}}<extra></extra>',
  }}], {{
    paper_bgcolor:'transparent', plot_bgcolor:'transparent', font:{{color:'#94a3b8'}},
    xaxis:{{title:'Average Score', gridcolor:'#1e293b', range:[0,10]}},
    yaxis:{{autorange:'reversed', tickfont:{{size:11}}}},
    margin:{{l:180, t:20}}, height: Math.max(400, filtered.length * 25),
  }}, {{responsive:true}});
}}

// ── 4. Feedback ──
function renderFeedback() {{
  if (feedbackData.length === 0) {{
    document.getElementById('feedback-chart').innerHTML = '<p style="color:var(--fg2);padding:40px;">No feedback data for this profile.</p>';
    return;
  }}
  const ai = feedbackData.map(f => f.ai_score);
  const user = feedbackData.map(f => f.user_score);
  const labels = feedbackData.map(f => f.title?.slice(0,50));
  const colors = feedbackData.map(f => tierColors[scoreTier(f.ai_score)]);

  Plotly.newPlot('feedback-chart', [
    {{x:[0,10], y:[0,10], mode:'lines', line:{{color:'#334155', dash:'dash'}}, showlegend:false}},
    {{x:ai, y:user, mode:'markers', marker:{{color:colors, size:10}}, text:labels,
      hovertemplate:'%{{text}}<br>AI: %{{x}}<br>User: %{{y}}<extra></extra>'}},
  ], {{
    paper_bgcolor:'transparent', plot_bgcolor:'transparent', font:{{color:'#94a3b8'}},
    xaxis:{{title:'AI Score', gridcolor:'#1e293b', range:[0,10]}},
    yaxis:{{title:'User Score', gridcolor:'#1e293b', range:[0,10]}},
    margin:{{t:20}}, annotations:[
      {{x:2,y:8,text:'Underscored',showarrow:false,font:{{color:'#6b7280',size:12}}}},
      {{x:8,y:2,text:'Overscored',showarrow:false,font:{{color:'#6b7280',size:12}}}},
    ],
  }}, {{responsive:true}});
}}

// ── 5. Radar ──
function renderRadar(mode) {{
  document.getElementById('btn-vol').style.background = mode === 'volume' ? 'var(--accent)' : 'var(--bg)';
  document.getElementById('btn-vol').style.color = mode === 'volume' ? 'white' : 'var(--fg2)';
  document.getElementById('btn-qual').style.background = mode === 'quality' ? 'var(--accent)' : 'var(--bg)';
  document.getElementById('btn-qual').style.color = mode === 'quality' ? 'white' : 'var(--fg2)';

  const top = categoryRadar.slice(0, 12);
  const cats = top.map(c => c.category);
  const vals = top.map(c => mode === 'volume' ? c.count : c.avg_score);

  Plotly.newPlot('radar-chart', [{{
    type: 'scatterpolar', r: [...vals, vals[0]], theta: [...cats, cats[0]],
    fill: 'toself', fillcolor: 'rgba(59,130,246,0.15)',
    line: {{color: '#3b82f6'}}, marker: {{size: 6}},
  }}], {{
    paper_bgcolor:'transparent', font:{{color:'#94a3b8'}},
    polar: {{bgcolor:'transparent', radialaxis:{{gridcolor:'#1e293b', linecolor:'#1e293b'}},
      angularaxis:{{gridcolor:'#1e293b', linecolor:'#1e293b'}}}},
    margin:{{t:40, b:40}},
  }}, {{responsive:true}});
}}

// ── 6. Timeline ──
function renderTimeline() {{
  const dates = timelineData.map(d=>d.date);
  const scores = timelineData.map(d=>d.score);
  const colors = scores.map(s => tierColors[scoreTier(s)]);
  const texts = timelineData.map(d=>d.title);
  const sizes = scores.map(s => 4 + s);

  Plotly.newPlot('timeline-chart', [{{
    x: dates, y: scores, mode: 'markers',
    marker: {{color: colors, size: sizes, opacity: 0.7}},
    text: texts,
    hovertemplate: '%{{text}}<br>Score: %{{y}}<br>Date: %{{x}}<extra></extra>',
  }}], {{
    paper_bgcolor:'transparent', plot_bgcolor:'transparent', font:{{color:'#94a3b8'}},
    xaxis:{{title:'Date', gridcolor:'#1e293b', type:'date'}},
    yaxis:{{title:'Score', gridcolor:'#1e293b', range:[0,10.5]}},
    margin:{{t:20}},
  }}, {{responsive:true}});
}}

// ── Init ──
renderEntityGraph();
</script>
</body>
</html>"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="StratOS → Obsidian Vault Exporter")
    parser.add_argument('--vault', type=str, default=str(DEFAULT_VAULT), help='Vault output path')
    parser.add_argument('--days', type=int, default=None, help='Only export last N days')
    parser.add_argument('--min-score', type=float, default=DEFAULT_MIN_SCORE, help='Minimum score threshold')
    parser.add_argument('--profile', type=int, default=DEFAULT_PROFILE, help='Profile ID')
    parser.add_argument('--all-profiles', action='store_true', help='Export all profiles')
    parser.add_argument('--incremental', action='store_true', help='Only new since last sync')
    parser.add_argument('--no-dashboard', action='store_true', help='Skip HTML dashboard')
    parser.add_argument('--dashboard-only', action='store_true', help='Only generate HTML dashboard')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    vault_path = Path(args.vault).expanduser()

    # Check incremental
    incr_since = None
    if args.incremental:
        sync_file = vault_path / ".last_sync"
        if sync_file.exists():
            incr_since = sync_file.read_text().strip()
            if args.verbose:
                print(f"Incremental mode: syncing since {incr_since}")
        else:
            if args.verbose:
                print("No .last_sync found, doing full export.")

    profiles_to_export = [args.profile]
    if args.all_profiles:
        # Discover all profiles with articles
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT profile_id FROM news_items WHERE profile_id > 0")
        profiles_to_export = [r[0] for r in cur.fetchall()]
        conn.close()
        if args.verbose:
            print(f"Exporting profiles: {profiles_to_export}")

    for pid in profiles_to_export:
        print(f"Loading data for profile {pid}...")
        data = StratOSData(DB_PATH, pid, args.min_score, args.days, incr_since)
        data.load_all(verbose=args.verbose)

        if args.all_profiles and len(profiles_to_export) > 1:
            pname = data.profiles.get(pid, {}).get('name', f'profile_{pid}')
            target_vault = vault_path / sanitize_filename(pname)
        else:
            target_vault = vault_path

        if not args.dashboard_only:
            print(f"Generating vault at {target_vault}...")
            gen = VaultGenerator(data, target_vault, verbose=args.verbose)
            gen.generate_all()
            print(f"  Vault: {gen.files_written} files written.")

        if not args.no_dashboard:
            print(f"Generating HTML dashboard...")
            dash = DashboardGenerator(data, target_vault, verbose=args.verbose)
            dash.generate()

        data.close()

    # Write .last_sync
    sync_ts = datetime.now().isoformat()
    (vault_path / ".last_sync").write_text(sync_ts)

    print(f"\nDone! Vault: {vault_path}")
    print(f"Open in Obsidian or view analytics/dashboard.html in browser.")


if __name__ == "__main__":
    main()

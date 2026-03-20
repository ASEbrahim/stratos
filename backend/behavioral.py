"""
Behavioral intelligence layer for StratOS.

Derives a user's behavioral profile from existing data (news_items, user_feedback,
scan_log) without any new tables or writes. Pure read-only computation.
"""

import logging
from datetime import datetime, timedelta

logger = logging.getLogger("behavioral")

# Hue labels and colors (Psycho-Pass spectrum)
HUE_LEVELS = [
    (80, "Clear",   "#00e5ff"),
    (60, "Stable",  "#4caf50"),
    (40, "Clouded", "#ffc107"),
    (20, "Turbid",  "#ff9800"),
    (0,  "Dark",    "#f44336"),
]


def compute_behavioral_profile(db, profile_id, days=30, config=None):
    """
    Compute a behavioral profile from existing DB tables.
    Returns a dict with category_engagement, source_quality, usage_patterns,
    trajectory, and alignment data.

    All queries filter by profile_id. Read-only — never writes.

    Args:
        config: Optional config dict for alignment computation (declared vs engaged).
                If None, alignment section is skipped.
    """
    result = {
        "category_engagement": {},
        "source_quality": {},
        "usage_patterns": {},
        "trajectory": "insufficient_data",
        "alignment": {"well_aligned": [], "over_declared": [], "under_declared": []},
        "feedback_count": 0,
        "article_count": 0,
    }

    try:
        cursor = db.conn.cursor()
    except Exception as e:
        logger.warning(f"behavioral: DB connection failed: {e}")
        return result

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()

    # --- Category engagement ---
    try:
        rows = cursor.execute(
            "SELECT category, COUNT(*), ROUND(AVG(score), 1) "
            "FROM news_items WHERE profile_id = ? AND fetched_at > ? "
            "GROUP BY category ORDER BY COUNT(*) DESC",
            (profile_id, cutoff)
        ).fetchall()

        cat_volumes = {}
        for cat, count, avg_score in rows:
            if not cat:
                continue
            cat_volumes[cat] = {"count": count, "avg_score": avg_score or 0}
            result["article_count"] += count

        # Feedback per category
        fb_rows = cursor.execute(
            "SELECT category, action, COUNT(*) FROM user_feedback "
            "WHERE profile_id = ? AND created_at > ? "
            "GROUP BY category, action",
            (profile_id, cutoff)
        ).fetchall()

        cat_fb = {}
        for cat, action, cnt in fb_rows:
            if not cat:
                continue
            if cat not in cat_fb:
                cat_fb[cat] = {}
            cat_fb[cat][action] = cnt
            result["feedback_count"] += cnt

        # Merge volumes + feedback
        for cat, vol in cat_volumes.items():
            fb = cat_fb.get(cat, {})
            total_fb = sum(fb.values()) or 1
            entry = {
                "count": vol["count"],
                "avg_score": vol["avg_score"],
                "clicks": fb.get("click", 0),
                "saves": fb.get("save", 0),
                "dismisses": fb.get("dismiss", 0),
                "rates": fb.get("rate", 0),
                "click_rate": round(fb.get("click", 0) / total_fb, 2) if sum(fb.values()) > 0 else 0,
                "save_rate": round(fb.get("save", 0) / total_fb, 2) if sum(fb.values()) > 0 else 0,
                "dismiss_rate": round(fb.get("dismiss", 0) / total_fb, 2) if sum(fb.values()) > 0 else 0,
            }
            result["category_engagement"][cat] = entry

    except Exception as e:
        logger.warning(f"behavioral: category engagement query failed: {e}")

    # --- Source quality ---
    try:
        rows = cursor.execute(
            "SELECT source, COUNT(*), ROUND(AVG(score), 1) "
            "FROM news_items WHERE profile_id = ? AND fetched_at > ? "
            "GROUP BY source HAVING COUNT(*) >= 3 "
            "ORDER BY COUNT(*) DESC LIMIT 20",
            (profile_id, cutoff)
        ).fetchall()

        for source, count, avg_score in rows:
            if not source:
                continue
            result["source_quality"][source] = {
                "count": count,
                "avg_score": avg_score or 0,
            }
    except Exception as e:
        logger.warning(f"behavioral: source quality query failed: {e}")

    # --- Usage patterns ---
    try:
        scan_rows = cursor.execute(
            "SELECT started_at FROM scan_log WHERE profile_id = ? AND started_at > ? "
            "ORDER BY started_at DESC",
            (profile_id, cutoff)
        ).fetchall()

        if scan_rows:
            # Hour distribution
            hours = {}
            for (ts,) in scan_rows:
                try:
                    h = datetime.fromisoformat(ts).hour
                    hours[h] = hours.get(h, 0) + 1
                except (ValueError, TypeError):
                    pass

            # Average interval
            intervals = []
            for i in range(len(scan_rows) - 1):
                try:
                    t1 = datetime.fromisoformat(scan_rows[i][0])
                    t2 = datetime.fromisoformat(scan_rows[i + 1][0])
                    intervals.append((t1 - t2).total_seconds() / 3600)
                except (ValueError, TypeError):
                    pass

            # Last scan age
            try:
                last_scan = datetime.fromisoformat(scan_rows[0][0])
                hours_since = (datetime.now() - last_scan).total_seconds() / 3600
            except (ValueError, TypeError):
                hours_since = 999

            # This week vs last week
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            two_weeks_ago = (datetime.now() - timedelta(days=14)).isoformat()
            this_week = sum(1 for (ts,) in scan_rows if ts > week_ago)
            last_week = sum(1 for (ts,) in scan_rows if two_weeks_ago < ts <= week_ago)

            result["usage_patterns"] = {
                "total_scans": len(scan_rows),
                "hour_distribution": hours,
                "avg_interval_hours": round(sum(intervals) / len(intervals), 1) if intervals else None,
                "hours_since_last_scan": round(hours_since, 1),
                "scans_this_week": this_week,
                "scans_last_week": last_week,
            }
        else:
            result["usage_patterns"] = {
                "total_scans": 0,
                "hour_distribution": {},
                "avg_interval_hours": None,
                "hours_since_last_scan": 999,
                "scans_this_week": 0,
                "scans_last_week": 0,
            }
    except Exception as e:
        logger.warning(f"behavioral: usage patterns query failed: {e}")

    # --- Trajectory ---
    try:
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        two_weeks_ago = (datetime.now() - timedelta(days=14)).isoformat()

        recent = cursor.execute(
            "SELECT COUNT(*) FROM user_feedback WHERE profile_id = ? AND created_at > ?",
            (profile_id, week_ago)
        ).fetchone()[0]

        previous = cursor.execute(
            "SELECT COUNT(*) FROM user_feedback WHERE profile_id = ? AND created_at > ? AND created_at <= ?",
            (profile_id, two_weeks_ago, week_ago)
        ).fetchone()[0]

        total = recent + previous
        if total < 3:
            result["trajectory"] = "insufficient_data"
        elif previous == 0:
            result["trajectory"] = "rising" if recent > 0 else "stable"
        else:
            ratio = recent / previous
            if ratio > 1.3:
                result["trajectory"] = "rising"
            elif ratio < 0.7:
                result["trajectory"] = "declining"
            else:
                result["trajectory"] = "stable"
    except Exception as e:
        logger.warning(f"behavioral: trajectory query failed: {e}")

    # --- Alignment (declared vs engaged) ---
    try:
        # Get declared categories from config
        declared = set()
        cfg = config

        if cfg:
            # Check dynamic_categories
            dyn = cfg.get("dynamic_categories", [])
            if isinstance(dyn, list):
                for d in dyn:
                    cid = d.get("id", "")
                    if cid:
                        declared.add(cid)
            # Check news.categories
            news_cats = cfg.get("news", {}).get("categories", {})
            if isinstance(news_cats, dict):
                declared.update(news_cats.keys())

        if declared and result["category_engagement"]:
            engaged = {k for k, v in result["category_engagement"].items()
                       if v.get("clicks", 0) + v.get("saves", 0) + v.get("rates", 0) > 0}
            all_cats = set(result["category_engagement"].keys())

            result["alignment"] = {
                "well_aligned": sorted(declared & engaged),
                "over_declared": sorted(declared - all_cats),
                "under_declared": sorted((engaged - declared))[:5],
            }
    except Exception as e:
        logger.warning(f"behavioral: alignment computation failed: {e}")

    return result


def compute_hue(behavioral_profile):
    """
    Compute Intelligence Hue (0-100) from a behavioral profile.
    Returns overall score, 5 dimension scores, label, color, and nudges.
    """
    bp = behavioral_profile
    article_count = bp.get("article_count", 0)

    # No data edge case
    if article_count == 0:
        return {
            "overall": 0,
            "label": "No data",
            "color": "#666",
            "provisional": True,
            "dimensions": {
                "freshness": 0, "diversity": 0, "coverage": 0,
                "signal_strength": 0, "engagement": 0,
            },
            "nudges": [{"type": "freshness", "message": "Run your first scan to initialize the hue system."}],
        }

    provisional = article_count < 10 or bp.get("usage_patterns", {}).get("total_scans", 0) < 2

    # --- Freshness (30%) ---
    hours_since = bp.get("usage_patterns", {}).get("hours_since_last_scan", 999)
    if hours_since <= 6:
        freshness = 100
    elif hours_since <= 12:
        freshness = 80
    elif hours_since <= 24:
        freshness = 60
    elif hours_since <= 48:
        freshness = 40
    elif hours_since <= 72:
        freshness = 20
    else:
        freshness = max(0, 10 - int(hours_since / 24))

    # --- Diversity (20%) ---
    cats = bp.get("category_engagement", {})
    sources = bp.get("source_quality", {})
    num_cats = len(cats)
    num_sources = len(sources)

    # Concentration check: no single category > 50% of volume
    total_vol = sum(c.get("count", 0) for c in cats.values()) or 1
    max_cat_share = max((c.get("count", 0) / total_vol for c in cats.values()), default=0)

    diversity = 0
    if num_cats >= 4:
        diversity += 30
    elif num_cats >= 2:
        diversity += 15
    if num_sources >= 5:
        diversity += 30
    elif num_sources >= 3:
        diversity += 15
    if max_cat_share <= 0.5:
        diversity += 40
    elif max_cat_share <= 0.7:
        diversity += 20
    diversity = min(100, diversity)

    # --- Coverage (20%) ---
    alignment = bp.get("alignment", {})
    well = len(alignment.get("well_aligned", []))
    over = len(alignment.get("over_declared", []))
    total_declared = well + over
    if total_declared > 0:
        coverage = min(100, int((well / total_declared) * 100))
    else:
        # No declared categories — give benefit of the doubt if articles exist
        coverage = 60 if article_count > 0 else 0

    # --- Signal Strength (15%) ---
    high_score_count = sum(1 for c in cats.values() if (c.get("avg_score", 0) or 0) >= 7)
    if num_cats > 0:
        signal_strength = min(100, int((high_score_count / num_cats) * 100))
    else:
        signal_strength = 0

    # --- Engagement (15%) ---
    fb_count = bp.get("feedback_count", 0)
    if fb_count >= 50:
        engagement = 100
    elif fb_count >= 20:
        engagement = 80
    elif fb_count >= 10:
        engagement = 60
    elif fb_count >= 3:
        engagement = 40
    elif fb_count >= 1:
        engagement = 20
    else:
        engagement = 0

    # --- Overall ---
    overall = int(
        freshness * 0.30 +
        diversity * 0.20 +
        coverage * 0.20 +
        signal_strength * 0.15 +
        engagement * 0.15
    )
    overall = max(0, min(100, overall))

    # Label and color
    label = "Dark"
    color = "#f44336"
    for threshold, lbl, clr in HUE_LEVELS:
        if overall >= threshold:
            label = lbl
            color = clr
            break

    # Nudges (up to 3, for dimensions below 40)
    dimensions = {
        "freshness": freshness,
        "diversity": diversity,
        "coverage": coverage,
        "signal_strength": signal_strength,
        "engagement": engagement,
    }

    nudge_templates = {
        "freshness": "Your feed is stale — run a scan?",
        "diversity": f"{int(max_cat_share * 100)}% of your feed is one category, across {num_sources} source{'s' if num_sources != 1 else ''}. Broaden your categories or add more sources.",
        "coverage": f"{over} declared categories have no recent articles. Check your settings.",
        "signal_strength": "Most signals scored below 7. Refine your keywords for sharper relevance.",
        "engagement": f"{article_count} signals delivered, {fb_count} interactions. Save or rate articles to sharpen your feed.",
    }

    nudges = []
    for dim, score in sorted(dimensions.items(), key=lambda x: x[1]):
        if score < 40 and len(nudges) < 3:
            nudges.append({"type": dim, "message": nudge_templates.get(dim, "")})

    return {
        "overall": overall,
        "label": label,
        "color": color,
        "provisional": provisional,
        "dimensions": dimensions,
        "nudges": nudges,
    }


def build_agent_behavioral_hint(db, profile_id, max_tokens=200, config=None):
    """
    Build a short behavioral context string for agent system prompt injection.
    Returns empty string on any failure. Never exceeds ~200 tokens.
    """
    try:
        bp = compute_behavioral_profile(db, profile_id, days=30, config=config)
        if bp["article_count"] == 0:
            return ""

        hue = compute_hue(bp)
        cats = bp.get("category_engagement", {})

        # Top 3 engaged categories (by click+save+rate count)
        ranked = sorted(cats.items(),
                        key=lambda x: x[1].get("clicks", 0) + x[1].get("saves", 0) + x[1].get("rates", 0),
                        reverse=True)
        top = [(k, v) for k, v in ranked if v.get("clicks", 0) + v.get("saves", 0) + v.get("rates", 0) > 0][:3]

        # Dismissed categories (dismiss_rate > 30%)
        dismissed = [k for k, v in cats.items() if v.get("dismiss_rate", 0) > 0.3][:2]

        if not top and not dismissed:
            return ""

        parts = [f"USER BEHAVIOR (last 30 days):"]
        if top:
            top_str = ", ".join(f"{k} ({v.get('click_rate', 0):.0%} click rate)" for k, v in top)
            parts.append(f"- Most engaged topics: {top_str}")
        if dismissed:
            parts.append(f"- Frequently dismissed: {', '.join(dismissed)}")

        # Alignment gaps
        align = bp.get("alignment", {})
        under = align.get("under_declared", [])[:2]
        if under:
            parts.append(f"- Emerging interests (not declared): {', '.join(under)}")

        parts.append(f"- Intelligence Hue: {hue['overall']}/100 ({hue['label']})")
        parts.append(f"- Engagement trend: {bp.get('trajectory', 'unknown').title()}")

        parts.append("")
        parts.append("Adapt: Lead with engaged topics. De-emphasize dismissed categories.")
        if hue["overall"] < 40:
            parts.append("Their hue is low — suggest scanning or refining settings.")

        return "\n".join(parts)
    except Exception as e:
        logger.debug(f"behavioral: agent hint failed: {e}")
        return ""


def build_briefing_behavioral_hint(db, profile_id, config=None):
    """
    Build a short (~80 token) behavioral hint for briefing prompt injection.
    Only returns content if 5+ feedback entries exist. Empty string on failure.
    """
    try:
        bp = compute_behavioral_profile(db, profile_id, days=30, config=config)
        if bp.get("feedback_count", 0) < 5:
            return ""

        cats = bp.get("category_engagement", {})
        ranked = sorted(cats.items(),
                        key=lambda x: x[1].get("clicks", 0) + x[1].get("saves", 0) + x[1].get("rates", 0),
                        reverse=True)
        top = [k for k, v in ranked if v.get("clicks", 0) + v.get("saves", 0) + v.get("rates", 0) > 0][:3]
        dismissed = [k for k, v in cats.items() if v.get("dismiss_rate", 0) > 0.3][:2]

        if not top:
            return ""

        parts = ["USER ENGAGEMENT CONTEXT:"]
        parts.append(f"User engages most with: {', '.join(top)}.")
        if dismissed:
            parts.append(f"Rarely interacts with: {', '.join(dismissed)}.")
        parts.append("Lead the briefing with the most engaged categories.")
        return "\n".join(parts)
    except Exception as e:
        logger.debug(f"behavioral: briefing hint failed: {e}")
        return ""

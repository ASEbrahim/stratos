"""
Profile Workspace — Export/import full profile data as ZIP.

Export includes:
  - profile.json (role, location, categories, tickers)
  - persona_contexts/ (system_context entries from DB)
  - youtube_channels.json (channel configs and lens mappings)
  - preference_signals.json
  - manifest.json (export metadata)
  - files/ (uploaded documents, optional)
  - insights/ (video insights, optional)

Import:
  - "replace" — overwrites everything in target profile
  - "merge" — adds new items, keeps existing
  - "context_only" — imports persona contexts only
"""

import io
import json
import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def export_profile(strat, profile_id: int, include_files: bool = True,
                   include_insights: bool = True) -> io.BytesIO:
    """Export a profile workspace as a ZIP file in memory."""
    buf = io.BytesIO()
    db = strat.db
    config = strat.config

    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. manifest.json
        manifest = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "profile_id": profile_id,
            "stratos_version": "2.2",
        }
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        # 2. profile.json — role, location, categories, tickers
        profile_data = {
            "profile": config.get("profile", {}),
            "dynamic_categories": config.get("dynamic_categories", []),
            "market": config.get("market", {}),
        }
        zf.writestr("profile.json", json.dumps(profile_data, indent=2))

        # 3. persona_contexts/ — from DB
        cursor = db.conn.cursor()
        cursor.execute(
            "SELECT persona_name, context_key, content, updated_at "
            "FROM persona_context WHERE profile_id = ?",
            (profile_id,)
        )
        contexts = [dict(r) for r in cursor.fetchall()]
        for ctx in contexts:
            path = f"persona_contexts/{ctx['persona_name']}/{ctx['context_key']}"
            zf.writestr(path, ctx['content'] or '')
        # Write a metadata index
        if contexts:
            zf.writestr("persona_contexts/_index.json", json.dumps(contexts, indent=2))

        # 4. youtube_channels.json
        try:
            cursor.execute(
                "SELECT channel_id, channel_name, channel_url, lenses "
                "FROM youtube_channels WHERE profile_id = ?",
                (profile_id,)
            )
            channels = [dict(r) for r in cursor.fetchall()]
            if channels:
                zf.writestr("youtube_channels.json", json.dumps(channels, indent=2))
        except Exception as e:
            logger.warning(f"Export: youtube channels skipped: {e}")

        # 5. preference_signals.json
        try:
            cursor.execute(
                "SELECT persona_source, signal_type, signal_key, signal_weight, auto_generated "
                "FROM user_preference_signals WHERE profile_id = ?",
                (profile_id,)
            )
            signals = [dict(r) for r in cursor.fetchall()]
            if signals:
                zf.writestr("preference_signals.json", json.dumps(signals, indent=2))
        except Exception as e:
            logger.warning(f"Export: preference signals skipped: {e}")

        # 6. insights/ — video insights (optional)
        if include_insights:
            try:
                cursor.execute(
                    """SELECT vi.lens_name, vi.content, vi.created_at,
                              yv.title, yv.video_id
                       FROM video_insights vi
                       JOIN youtube_videos yv ON vi.video_id = yv.id
                       WHERE vi.profile_id = ?""",
                    (profile_id,)
                )
                insights = [dict(r) for r in cursor.fetchall()]
                if insights:
                    zf.writestr("insights/all_insights.json", json.dumps(insights, indent=2))
            except Exception as e:
                logger.warning(f"Export: insights skipped: {e}")

        # 7. files/ — uploaded documents (optional)
        if include_files:
            try:
                cursor.execute(
                    "SELECT filename, file_type, content_text, file_path "
                    "FROM user_files WHERE profile_id = ?",
                    (profile_id,)
                )
                files = [dict(r) for r in cursor.fetchall()]
                for f in files:
                    file_path = Path(f.get('file_path', ''))
                    if file_path.exists():
                        zf.write(str(file_path), f"files/{f['filename']}")
                    elif f.get('content_text'):
                        zf.writestr(f"files/{f['filename']}", f['content_text'])
                # File metadata
                if files:
                    meta = [{"filename": f['filename'], "file_type": f['file_type']} for f in files]
                    zf.writestr("files/_index.json", json.dumps(meta, indent=2))
            except Exception as e:
                logger.warning(f"Export: files skipped: {e}")

        # 8. conversations/ — agent chat history
        try:
            cursor.execute(
                "SELECT id, persona, title, messages, is_active, created_at, updated_at "
                "FROM conversations WHERE profile_id = ? AND archived = 0",
                (profile_id,)
            )
            convs = [dict(r) for r in cursor.fetchall()]
            for conv in convs:
                conv['messages'] = json.loads(conv.get('messages', '[]'))
                zf.writestr(f"conversations/{conv['persona']}/conv_{conv['id']}.json",
                            json.dumps(conv, indent=2))
            if convs:
                zf.writestr("conversations/_index.json",
                            json.dumps([{"id": c['id'], "persona": c['persona'], "title": c['title'],
                                         "message_count": len(c['messages'])} for c in convs], indent=2))
        except Exception as e:
            logger.warning(f"Export: conversations skipped: {e}")

        # 9. scenarios/ — game scenarios
        try:
            cursor.execute(
                "SELECT id, name, persona, description, state_md, world_md, characters_json, "
                "genre, is_active, created_at, updated_at "
                "FROM scenarios WHERE profile_id = ?",
                (profile_id,)
            )
            scenarios = [dict(r) for r in cursor.fetchall()]
            for sc in scenarios:
                zf.writestr(f"scenarios/{sc['name']}.json", json.dumps(sc, indent=2))
            if scenarios:
                zf.writestr("scenarios/_index.json",
                            json.dumps([{"name": s['name'], "genre": s.get('genre', ''),
                                         "is_active": s['is_active']} for s in scenarios], indent=2))
        except Exception as e:
            logger.warning(f"Export: scenarios skipped: {e}")

    buf.seek(0)
    return buf


def import_profile(strat, profile_id: int, zip_data: bytes,
                   strategy: str = "replace") -> Dict[str, Any]:
    """
    Import a profile workspace from a ZIP file.

    strategy:
      - "replace" — overwrites everything in target profile
      - "merge" — adds new items, keeps existing
      - "context_only" — imports persona contexts only
    """
    db = strat.db
    cursor = db.conn.cursor()
    stats = {"contexts": 0, "channels": 0, "signals": 0, "files": 0, "insights": 0}

    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_data), 'r')
    except zipfile.BadZipFile:
        return {"error": "Invalid ZIP file"}

    try:
        return _do_import(strat, profile_id, zf, strategy, db, cursor, stats)
    finally:
        zf.close()


def _do_import(strat, profile_id, zf, strategy, db, cursor, stats):
    """Inner import logic — separated so ZipFile can be closed in finally."""
    # Verify manifest
    if "manifest.json" not in zf.namelist():
        return {"error": "Missing manifest.json — not a valid StratOS export"}

    manifest = json.loads(zf.read("manifest.json"))

    # 1. profile.json — config overlay (only for "replace" strategy)
    if strategy == "replace" and "profile.json" in zf.namelist():
        try:
            profile_data = json.loads(zf.read("profile.json"))
            # Update config in memory (caller should persist if needed)
            if profile_data.get("profile"):
                strat.config["profile"] = profile_data["profile"]
            if profile_data.get("dynamic_categories"):
                strat.config["dynamic_categories"] = profile_data["dynamic_categories"]
            if profile_data.get("market"):
                strat.config["market"] = profile_data["market"]
        except Exception as e:
            logger.warning(f"Import: profile.json failed: {e}")

    # 2. persona_contexts/
    if strategy != "context_only" or True:  # Always import contexts
        context_files = [n for n in zf.namelist()
                         if n.startswith("persona_contexts/") and not n.endswith("_index.json") and not n.endswith("/")]
        for cf in context_files:
            parts = cf.replace("persona_contexts/", "").split("/", 1)
            if len(parts) != 2:
                continue
            persona_name, context_key = parts
            content = zf.read(cf).decode('utf-8', errors='replace')

            if strategy == "merge":
                cursor.execute(
                    "INSERT OR IGNORE INTO persona_context (profile_id, persona_name, context_key, content, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (profile_id, persona_name, context_key, content, datetime.now().isoformat())
                )
            else:
                cursor.execute(
                    "INSERT INTO persona_context (profile_id, persona_name, context_key, content, updated_at) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(profile_id, persona_name, context_key) DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at",
                    (profile_id, persona_name, context_key, content, datetime.now().isoformat())
                )
            stats["contexts"] += 1

    if strategy == "context_only":
        db._commit()
        return {"ok": True, "strategy": strategy, "stats": stats, "manifest": manifest}

    # 3. youtube_channels.json
    if "youtube_channels.json" in zf.namelist():
        try:
            channels = json.loads(zf.read("youtube_channels.json"))
            for ch in channels:
                if strategy == "merge":
                    cursor.execute(
                        "INSERT OR IGNORE INTO youtube_channels (profile_id, channel_id, channel_name, channel_url, lenses) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (profile_id, ch['channel_id'], ch.get('channel_name'), ch.get('channel_url'), ch.get('lenses'))
                    )
                else:
                    cursor.execute(
                        "INSERT INTO youtube_channels (profile_id, channel_id, channel_name, channel_url, lenses) "
                        "VALUES (?, ?, ?, ?, ?) "
                        "ON CONFLICT(profile_id, channel_id) DO UPDATE SET channel_name = excluded.channel_name, "
                        "channel_url = excluded.channel_url, lenses = excluded.lenses",
                        (profile_id, ch['channel_id'], ch.get('channel_name'), ch.get('channel_url'), ch.get('lenses'))
                    )
                stats["channels"] += 1
        except Exception as e:
            logger.warning(f"Import: youtube channels failed: {e}")

    # 4. preference_signals.json
    if "preference_signals.json" in zf.namelist():
        try:
            signals = json.loads(zf.read("preference_signals.json"))
            for sig in signals:
                if strategy == "merge":
                    cursor.execute(
                        "INSERT OR IGNORE INTO user_preference_signals "
                        "(profile_id, persona_source, signal_type, signal_key, signal_weight, auto_generated) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (profile_id, sig['persona_source'], sig['signal_type'],
                         sig['signal_key'], sig.get('signal_weight', 1.0), sig.get('auto_generated', 0))
                    )
                else:
                    cursor.execute(
                        "INSERT INTO user_preference_signals "
                        "(profile_id, persona_source, signal_type, signal_key, signal_weight, auto_generated) "
                        "VALUES (?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(profile_id, persona_source, signal_type, signal_key) "
                        "DO UPDATE SET signal_weight = excluded.signal_weight",
                        (profile_id, sig['persona_source'], sig['signal_type'],
                         sig['signal_key'], sig.get('signal_weight', 1.0), sig.get('auto_generated', 0))
                    )
                stats["signals"] += 1
        except Exception as e:
            logger.warning(f"Import: preference signals failed: {e}")

    # 5. insights/
    if "insights/all_insights.json" in zf.namelist():
        try:
            insights = json.loads(zf.read("insights/all_insights.json"))
            # Insights are read-only export — skip import (they reference video IDs that may differ)
            stats["insights"] = len(insights)
            logger.info(f"Import: {len(insights)} insights in export (not imported — regenerate from videos)")
        except Exception as e:
            logger.warning(f"Import: insights read failed: {e}")

    # 6. files/ — skip for now (files reference filesystem paths)
    file_entries = [n for n in zf.namelist() if n.startswith("files/") and not n.endswith("_index.json") and not n.endswith("/")]
    stats["files"] = len(file_entries)
    if file_entries:
        logger.info(f"Import: {len(file_entries)} files in export (file import not yet supported)")

    # 7. conversations/
    stats["conversations"] = 0
    conv_files = [n for n in zf.namelist()
                  if n.startswith("conversations/") and n.endswith(".json") and "_index" not in n]
    for cf in conv_files:
        try:
            conv = json.loads(zf.read(cf))
            cursor.execute(
                "INSERT INTO conversations (profile_id, persona, title, messages, is_active) "
                "VALUES (?, ?, ?, ?, 0)",
                (profile_id, conv.get('persona', 'intelligence'),
                 conv.get('title', 'Imported'),
                 json.dumps(conv.get('messages', [])))
            )
            stats["conversations"] += 1
        except Exception as e:
            logger.warning(f"Import: conversation {cf} failed: {e}")

    # 8. scenarios/
    stats["scenarios"] = 0
    sc_files = [n for n in zf.namelist()
                if n.startswith("scenarios/") and n.endswith(".json") and "_index" not in n]
    for sf in sc_files:
        try:
            sc = json.loads(zf.read(sf))
            if strategy == "merge":
                cursor.execute(
                    "INSERT OR IGNORE INTO scenarios "
                    "(profile_id, persona, name, description, state_md, world_md, characters_json, genre, is_active) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
                    (profile_id, sc.get('persona', 'gaming'), sc.get('name', ''),
                     sc.get('description', ''), sc.get('state_md', ''),
                     sc.get('world_md', ''), sc.get('characters_json', '[]'),
                     sc.get('genre', ''))
                )
            else:
                cursor.execute(
                    "INSERT INTO scenarios "
                    "(profile_id, persona, name, description, state_md, world_md, characters_json, genre, is_active) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0) "
                    "ON CONFLICT(profile_id, name) DO UPDATE SET "
                    "state_md = excluded.state_md, world_md = excluded.world_md, "
                    "characters_json = excluded.characters_json",
                    (profile_id, sc.get('persona', 'gaming'), sc.get('name', ''),
                     sc.get('description', ''), sc.get('state_md', ''),
                     sc.get('world_md', ''), sc.get('characters_json', '[]'),
                     sc.get('genre', ''))
                )
            stats["scenarios"] += 1
        except Exception as e:
            logger.warning(f"Import: scenario {sf} failed: {e}")

    db._commit()
    return {"ok": True, "strategy": strategy, "stats": stats, "manifest": manifest}


def get_workspace_stats(strat, profile_id: int) -> Dict[str, Any]:
    """Get size breakdown of a profile's workspace data."""
    db = strat.db
    cursor = db.conn.cursor()
    stats = {}

    # Persona contexts count
    cursor.execute(
        "SELECT COUNT(*) FROM persona_context WHERE profile_id = ?",
        (profile_id,)
    )
    stats["persona_contexts"] = cursor.fetchone()[0]

    # YouTube channels/videos/insights
    try:
        cursor.execute("SELECT COUNT(*) FROM youtube_channels WHERE profile_id = ?", (profile_id,))
        stats["youtube_channels"] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM youtube_videos WHERE profile_id = ?", (profile_id,))
        stats["youtube_videos"] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM video_insights WHERE profile_id = ?", (profile_id,))
        stats["video_insights"] = cursor.fetchone()[0]
    except Exception as e:
        logger.debug(f"Workspace stats: youtube tables not available: {e}")
        stats["youtube_channels"] = 0
        stats["youtube_videos"] = 0
        stats["video_insights"] = 0

    # Preference signals
    try:
        cursor.execute("SELECT COUNT(*) FROM user_preference_signals WHERE profile_id = ?", (profile_id,))
        stats["preference_signals"] = cursor.fetchone()[0]
    except Exception as e:
        logger.debug(f"Workspace stats: preference_signals table not available: {e}")
        stats["preference_signals"] = 0

    # Uploaded files
    try:
        cursor.execute("SELECT COUNT(*) FROM user_files WHERE profile_id = ?", (profile_id,))
        stats["uploaded_files"] = cursor.fetchone()[0]
    except Exception as e:
        logger.debug(f"Workspace stats: user_files table not available: {e}")
        stats["uploaded_files"] = 0

    # News items & feedback
    cursor.execute("SELECT COUNT(*) FROM news_items WHERE profile_id = ?", (profile_id,))
    stats["news_items"] = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM user_feedback WHERE profile_id = ?", (profile_id,))
    stats["feedback_entries"] = cursor.fetchone()[0]

    # Context directory size on disk
    ctx_dir = Path(strat.config.get("system", {}).get("data_dir", "data")) / "users" / str(profile_id)
    if ctx_dir.exists():
        total_bytes = sum(f.stat().st_size for f in ctx_dir.rglob('*') if f.is_file())
        stats["disk_bytes"] = total_bytes
        stats["disk_human"] = f"{total_bytes / 1024:.1f} KB" if total_bytes < 1048576 else f"{total_bytes / 1048576:.1f} MB"
    else:
        stats["disk_bytes"] = 0
        stats["disk_human"] = "0 KB"

    return stats

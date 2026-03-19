"""
Character Card Community — CRUD, publishing, discovery, ratings, TavernCard V2 import.

Endpoints:
  POST   /api/cards                 — Create new card
  GET    /api/cards/<id>            — Get single card
  PUT    /api/cards/<id>            — Update card
  DELETE /api/cards/<id>            — Delete card (creator only)
  POST   /api/cards/<id>/publish    — Make card public
  GET    /api/cards/browse          — Browse published cards
  GET    /api/cards/trending        — Top cards
  GET    /api/cards/search          — Search by name/tags
  POST   /api/cards/<id>/rate       — Rate a card (1-5)
  POST   /api/cards/import/tavern   — Import TavernCard V2 from PNG
  GET    /api/cards/my              — User's own cards
"""

import base64
import json
import struct
import uuid
import logging
import threading
import requests as req

from routes.helpers import json_response, error_response, read_json_body

logger = logging.getLogger("character_cards")

OLLAMA_HOST = "http://localhost:11434"

MAX_CARD_NAME_LENGTH = 200
MAX_CARD_FIELD_LENGTH = 5000
MAX_TAVERN_IMPORT_BYTES = 20 * 1024 * 1024  # 20MB


# ═══════════════════════════════════════════════════════════
# TavernCard V2 Parser
# ═══════════════════════════════════════════════════════════

def parse_tavern_card_v2(png_bytes: bytes) -> dict | None:
    """Extract character data from a TavernCard V2 PNG file."""
    if png_bytes[:8] != b'\x89PNG\r\n\x1a\n':
        return None

    pos = 8
    while pos < len(png_bytes):
        if pos + 8 > len(png_bytes):
            break
        chunk_len = struct.unpack('>I', png_bytes[pos:pos+4])[0]
        chunk_type = png_bytes[pos+4:pos+8].decode('ascii', errors='ignore')
        chunk_data = png_bytes[pos+8:pos+8+chunk_len]

        if chunk_type == 'tEXt':
            null_pos = chunk_data.find(b'\x00')
            if null_pos >= 0:
                keyword = chunk_data[:null_pos].decode('ascii', errors='ignore')
                value = chunk_data[null_pos+1:]
                if keyword == 'chara':
                    try:
                        decoded = base64.b64decode(value)
                        return json.loads(decoded)
                    except Exception as e:
                        logger.warning(f"Failed to decode TavernCard chara data: {e}")

        if chunk_type == 'IEND':
            break

        pos += 12 + chunk_len

    return None


def tavern_card_to_stratos(card_data: dict) -> dict:
    """Map TavernCard V2 fields to StratOS character card format."""
    data = card_data.get("data", card_data)

    return {
        "name": data.get("name", "Imported Character"),
        "physical_description": "",
        "personality": data.get("personality", ""),
        "scenario": data.get("scenario", ""),
        "first_message": data.get("first_mes", ""),
        "example_dialogues": data.get("mes_example", ""),
        "genre_tags": json.dumps(data.get("tags", [])),
        "imported_from": "tavern_v2",
        "tavern_card_raw": json.dumps(card_data),
        "speech_pattern": "",
        "emotional_trigger": "",
        "defensive_mechanism": "",
        "vulnerability": "",
        "specific_detail": "",
    }


def _enrich_card_background(card_id: str, name: str, personality: str, scenario: str,
                            content_rating: str, db):
    """Background LLM call to auto-fill missing depth fields on a new card.

    Only fills fields the user left empty. Never overwrites user content.
    Runs in a background thread so card creation returns instantly.
    """
    from routes.rp_meta import call_haiku
    try:
        rating_note = "This is an NSFW character — mature themes are expected." if content_rating == "nsfw" else ""
        prompt = f"""Given this character, generate the missing depth fields as JSON.
Character: {name}
Personality: {personality}
Scenario: {scenario}
{rating_note}

Return ONLY valid JSON with these keys (1-2 sentences each, vivid and specific):
{{
  "emotional_trigger": "What specifically makes this character lose composure or react strongly",
  "defensive_mechanism": "How they protect themselves emotionally when threatened or vulnerable",
  "vulnerability": "Their hidden weakness or deepest fear that they try to hide",
  "specific_detail": "A small defining quirk, habit, or physical detail that makes them feel real"
}}"""

        # Try Haiku first (better creative quality)
        text = call_haiku(prompt, max_tokens=300)
        if text is None:
            # Fallback to Ollama
            r = req.post(f"{OLLAMA_HOST}/api/generate", json={
                "model": "qwen3.5:9b", "prompt": prompt, "stream": False,
                "options": {"temperature": 0.4, "num_predict": 300},
                "think": False,
            }, timeout=30)
            if r.status_code != 200:
                return
            text = r.json().get("response", "")

        import re
        # Extract JSON from response
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if not json_match:
            return
        data = json.loads(json_match.group())

        # Only update fields that are currently empty
        updates = {}
        for field in ["emotional_trigger", "defensive_mechanism", "vulnerability", "specific_detail"]:
            val = data.get(field, "").strip()
            if val and len(val) > 10:
                updates[field] = val[:500]

        if updates:
            updates["quality_elements_count"] = calculate_quality_elements({**updates})
            db.update_character_card(card_id, **updates)

        # ── Generate example dialogues if missing (anchors character voice) ──
        card_row = db.get_character_card(card_id)
        if card_row and not card_row.get('example_dialogues', '').strip():
            speech = card_row.get('speech_pattern', '') or ''
            dlg_prompt = f"""Write 2-3 short example dialogue exchanges for this character. Show how they talk, their personality through speech and actions.

Character: {name}
Personality: {personality}
Speech style: {speech[:200]}
{rating_note}

Format each exchange with <START> separator:
<START>
User: [a greeting or question]
{name}: [response showing personality through *actions* and "dialogue"]
<START>
User: [something that challenges or surprises the character]
{name}: [response showing how they handle pressure]

Write ONLY the exchanges. No explanation."""

            try:
                # Try Haiku first
                dlg_text = call_haiku(dlg_prompt, max_tokens=400)
                if dlg_text is None:
                    # Fallback to Ollama
                    r2 = req.post(f"{OLLAMA_HOST}/api/generate", json={
                        "model": "qwen3.5:9b", "prompt": dlg_prompt, "stream": False,
                        "options": {"temperature": 0.6, "num_predict": 400},
                        "think": False,
                    }, timeout=45)
                    if r2.status_code == 200:
                        dlg_text = r2.json().get("response", "").strip()
                    else:
                        dlg_text = None
                if dlg_text and len(dlg_text) > 50 and "<START>" in dlg_text:
                    db.update_character_card(card_id, example_dialogues=dlg_text[:2000])
                    logger.info(f"Auto-generated example dialogues for '{name}'")
            except Exception as e2:
                logger.warning(f"Example dialogue generation failed for '{name}': {e2}")
            logger.info(f"Auto-enriched card '{name}': filled {list(updates.keys())}")

    except Exception as e:
        logger.warning(f"Card auto-enrichment failed for '{name}' (non-fatal): {e}")


def calculate_quality_elements(card: dict) -> int:
    """Calculate character card quality element count (0-6)."""
    score = 0
    for field in ["physical_description", "speech_pattern", "emotional_trigger",
                   "defensive_mechanism", "vulnerability", "specific_detail"]:
        value = card.get(field, "").strip()
        if value and len(value) >= 20:
            score += 1
    return score


# ═══════════════════════════════════════════════════════════
# Route handlers
# ═══════════════════════════════════════════════════════════

def handle_post(handler, strat, auth, path) -> bool:
    if not path.startswith("/api/cards"):
        return False

    # PUT routes through do_POST in server.py — delegate to handle_put
    method = handler.command if hasattr(handler, 'command') else 'POST'
    if method == 'PUT':
        return handle_put(handler, strat, auth, path)

    profile_id = getattr(handler, '_profile_id', 0)
    db = strat.db

    # ── POST /api/cards (create) ──
    if path == "/api/cards":
        data = read_json_body(handler)
        name = data.get("name", "").strip()
        if not name:
            error_response(handler, "Name required", 400)
            return True
        if len(name) > MAX_CARD_NAME_LENGTH:
            error_response(handler, f"Name too long (max {MAX_CARD_NAME_LENGTH} chars)", 400)
            return True
        for field_name in ['physical_description', 'personality', 'scenario',
                           'first_message', 'example_dialogues', 'speech_pattern',
                           'emotional_trigger', 'defensive_mechanism', 'vulnerability',
                           'specific_detail']:
            if len(data.get(field_name, "")) > MAX_CARD_FIELD_LENGTH:
                error_response(handler, f"{field_name} too long (max {MAX_CARD_FIELD_LENGTH} chars)", 400)
                return True

        card_id = uuid.uuid4().hex[:16]
        quality = calculate_quality_elements(data)

        db.insert_character_card(
            card_id, profile_id, name,
            physical_description=data.get("physical_description", ""),
            speech_pattern=data.get("speech_pattern", ""),
            emotional_trigger=data.get("emotional_trigger", ""),
            defensive_mechanism=data.get("defensive_mechanism", ""),
            vulnerability=data.get("vulnerability", ""),
            specific_detail=data.get("specific_detail", ""),
            personality=data.get("personality", ""),
            scenario=data.get("scenario", ""),
            first_message=data.get("first_message", ""),
            example_dialogues=data.get("example_dialogues", ""),
            genre_tags=json.dumps(data.get("genre_tags", [])),
            content_rating=data.get("content_rating", "sfw"),
            quality_elements_count=quality,
            is_published=True,
        )

        # ── Auto-enrich missing depth fields in background ──
        depth_fields = ["emotional_trigger", "defensive_mechanism", "vulnerability", "specific_detail"]
        missing_depth = [f for f in depth_fields if not data.get(f, "").strip()]
        if missing_depth and data.get("personality", "").strip():
            threading.Thread(
                target=_enrich_card_background, daemon=True,
                args=(card_id, name, data.get("personality", ""),
                      data.get("scenario", ""), data.get("content_rating", "sfw"), db)
            ).start()

        json_response(handler, {"ok": True, "card_id": card_id, "quality_elements": quality})
        return True

    # ── POST /api/cards/import/tavern ──
    if path == "/api/cards/import/tavern":
        content_length = int(handler.headers.get('Content-Length', 0))
        if content_length == 0:
            error_response(handler, "No file data", 400)
            return True
        if content_length > MAX_TAVERN_IMPORT_BYTES:
            error_response(handler, f"File too large (max {MAX_TAVERN_IMPORT_BYTES // (1024*1024)}MB)", 400)
            return True

        png_bytes = handler.rfile.read(content_length)
        card_data = parse_tavern_card_v2(png_bytes)
        if not card_data:
            error_response(handler, "No TavernCard V2 data found in PNG", 400)
            return True

        stratos_fields = tavern_card_to_stratos(card_data)
        card_id = uuid.uuid4().hex[:16]

        db.insert_character_card(
            card_id, profile_id, stratos_fields["name"],
            **{k: v for k, v in stratos_fields.items() if k != "name"}
        )

        # Auto-enrich imported cards too
        if stratos_fields.get("personality", "").strip():
            threading.Thread(
                target=_enrich_card_background, daemon=True,
                args=(card_id, stratos_fields["name"], stratos_fields.get("personality", ""),
                      stratos_fields.get("scenario", ""), "sfw", db)
            ).start()

        json_response(handler, {
            "ok": True, "card_id": card_id,
            "name": stratos_fields["name"],
            "imported_from": "tavern_v2"
        })
        return True

    # ── POST /api/cards/<id>/publish ──
    if path.endswith("/publish"):
        card_id = path.split("/")[-2]
        card = db.get_character_card(card_id)
        if not card:
            error_response(handler, "Card not found", 404)
            return True
        if card['creator_profile_id'] != profile_id:
            error_response(handler, "Not your card", 403)
            return True
        db.publish_character_card(card_id)
        json_response(handler, {"ok": True})
        return True

    # ── POST /api/cards/<id>/rate ──
    if path.endswith("/rate"):
        data = read_json_body(handler)
        card_id = path.split("/")[-2]
        rating = data.get("rating", 0)
        if not (1 <= rating <= 5):
            error_response(handler, "Rating must be 1-5", 400)
            return True
        db.rate_character_card(card_id, profile_id, rating)
        json_response(handler, {"ok": True})
        return True

    # ── POST /api/cards/<id>/save ──
    if path.endswith("/save"):
        # Save someone else's card to your library (clone it)
        card_id = path.split("/")[-2]
        card = db.get_character_card(card_id)
        if not card:
            error_response(handler, "Card not found", 404)
            return True
        new_id = uuid.uuid4().hex[:16]
        fields = {k: card[k] for k in [
            'physical_description', 'speech_pattern', 'emotional_trigger',
            'defensive_mechanism', 'vulnerability', 'specific_detail',
            'personality', 'scenario', 'first_message', 'example_dialogues',
            'genre_tags', 'content_rating', 'quality_elements_count',
        ]}
        db.insert_character_card(new_id, profile_id, card['name'], **fields)
        json_response(handler, {"ok": True, "card_id": new_id})
        return True

    return False


def handle_get(handler, strat, auth, path) -> bool:
    if not path.startswith("/api/cards"):
        return False

    profile_id = getattr(handler, '_profile_id', 0)
    db = strat.db

    # ── GET /api/cards/my ──
    if path == "/api/cards/my":
        rows = db.conn.execute(
            "SELECT * FROM character_cards WHERE creator_profile_id = ? ORDER BY updated_at DESC",
            (profile_id,)
        ).fetchall()
        json_response(handler, {"cards": [dict(r) for r in rows]})
        return True

    # ── GET /api/cards/browse ──
    if path == "/api/cards/browse":
        from urllib.parse import parse_qs, urlparse
        qs = parse_qs(urlparse(handler.path).query)
        genre = qs.get("genre", [None])[0]
        sort = qs.get("sort", ["trending"])[0]
        limit = min(int(qs.get("limit", ["20"])[0]), 100)
        offset = max(int(qs.get("offset", ["0"])[0]), 0)
        cards = db.get_published_cards(genre=genre, sort=sort, limit=limit, offset=offset)
        json_response(handler, {"cards": cards})
        return True

    # ── GET /api/cards/trending ──
    if path == "/api/cards/trending":
        cards = db.get_published_cards(sort='trending', limit=10)
        json_response(handler, {"cards": cards})
        return True

    # ── GET /api/cards/search ──
    if path == "/api/cards/search":
        from urllib.parse import parse_qs, urlparse
        qs = parse_qs(urlparse(handler.path).query)
        q = qs.get("q", [""])[0]
        if not q:
            json_response(handler, {"cards": []})
            return True
        rows = db.conn.execute(
            "SELECT * FROM character_cards WHERE is_published = TRUE AND "
            "(name LIKE ? OR genre_tags LIKE ? OR personality LIKE ?) "
            "ORDER BY quality_elements_count DESC LIMIT 20",
            (f'%{q}%', f'%{q}%', f'%{q}%')
        ).fetchall()
        json_response(handler, {"cards": [dict(r) for r in rows]})
        return True

    # ── GET /api/cards/<id> ──
    parts = path.split("/")
    if len(parts) == 4 and parts[2] == "cards":
        card_id = parts[3]
        card = db.get_character_card(card_id)
        if not card:
            error_response(handler, "Card not found", 404)
            return True
        json_response(handler, card)
        return True

    return False


def handle_put(handler, strat, auth, path) -> bool:
    if not path.startswith("/api/cards/"):
        return False

    profile_id = getattr(handler, '_profile_id', 0)
    db = strat.db

    parts = path.split("/")
    if len(parts) == 4 and parts[2] == "cards":
        card_id = parts[3]
        card = db.get_character_card(card_id)
        if not card:
            error_response(handler, "Card not found", 404)
            return True
        if card['creator_profile_id'] != profile_id:
            error_response(handler, "Not your card", 403)
            return True

        data = read_json_body(handler)
        allowed = ['name', 'physical_description', 'speech_pattern', 'emotional_trigger',
                    'defensive_mechanism', 'vulnerability', 'specific_detail',
                    'personality', 'scenario', 'first_message', 'example_dialogues',
                    'genre_tags', 'content_rating', 'avatar_image_path']
        updates = {k: data[k] for k in allowed if k in data}
        if 'genre_tags' in updates and isinstance(updates['genre_tags'], list):
            updates['genre_tags'] = json.dumps(updates['genre_tags'])

        # Recalculate quality
        merged = {**card, **updates}
        updates['quality_elements_count'] = calculate_quality_elements(merged)

        db.update_character_card(card_id, **updates)
        json_response(handler, {"ok": True, "quality_elements": updates['quality_elements_count']})
        return True

    return False


def handle_delete(handler, strat, auth, path) -> bool:
    if not path.startswith("/api/cards/"):
        return False

    profile_id = getattr(handler, '_profile_id', 0)
    db = strat.db

    parts = path.split("/")
    if len(parts) == 4 and parts[2] == "cards":
        card_id = parts[3]
        card = db.get_character_card(card_id)
        if not card:
            error_response(handler, "Card not found", 404)
            return True
        if card['creator_profile_id'] != profile_id:
            error_response(handler, "Not your card", 403)
            return True
        db.conn.execute("DELETE FROM character_card_stats WHERE card_id = ?", (card_id,))
        db.conn.execute("DELETE FROM character_card_ratings WHERE card_id = ?", (card_id,))
        db.conn.execute("DELETE FROM character_cards WHERE id = ?", (card_id,))
        db.conn.commit()
        json_response(handler, {"ok": True, "deleted": card_id})
        return True

    return False

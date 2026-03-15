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

from routes.helpers import json_response, error_response, read_json_body

logger = logging.getLogger("character_cards")


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
                    except Exception:
                        pass

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

        json_response(handler, {"ok": True, "card_id": card_id, "quality_elements": quality})
        return True

    # ── POST /api/cards/import/tavern ──
    if path == "/api/cards/import/tavern":
        content_length = int(handler.headers.get('Content-Length', 0))
        if content_length == 0:
            error_response(handler, "No file data", 400)
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
        limit = int(qs.get("limit", ["20"])[0])
        offset = int(qs.get("offset", ["0"])[0])
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

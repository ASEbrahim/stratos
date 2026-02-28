"""
Email-based authentication routes for StratOS.

Handles registration, login, email verification, password reset,
profile management, and admin operations.
"""

import hashlib
import json
import logging
import secrets
import time
from datetime import datetime, timedelta

import user_data

logger = logging.getLogger("STRAT_OS")

try:
    import bcrypt
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False
    logger.warning("bcrypt not installed — using SHA-256 fallback (install bcrypt for production)")


def _hash_password(password: str) -> str:
    """Hash a password with bcrypt (or SHA-256 fallback)."""
    if HAS_BCRYPT:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    return "SHA256:" + hashlib.sha256(password.encode('utf-8')).hexdigest()


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored hash. Handles legacy PIN hashes."""
    if stored_hash.startswith("LEGACY:"):
        # Migrated PIN: SHA-256 hash
        sha256_hash = stored_hash[7:]
        return hashlib.sha256(password.encode('utf-8')).hexdigest() == sha256_hash
    if stored_hash.startswith("SHA256:"):
        # Fallback when bcrypt not installed
        return hashlib.sha256(password.encode('utf-8')).hexdigest() == stored_hash[7:]
    if HAS_BCRYPT:
        try:
            return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))
        except Exception:
            return False
    return False


def _upgrade_password(db, user_id: int, password: str, stored_hash: str):
    """Upgrade legacy password hash to bcrypt on successful login."""
    if stored_hash.startswith("LEGACY:") or stored_hash.startswith("SHA256:"):
        if HAS_BCRYPT:
            new_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            cursor = db.conn.cursor()
            cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id))
            db._commit()
            logger.info(f"Upgraded password hash for user {user_id} to bcrypt")


def _generate_token() -> str:
    """Generate a 64-char hex session token."""
    return secrets.token_hex(32)


def _generate_code() -> str:
    """Generate a 5-digit verification code."""
    return str(secrets.randbelow(90000) + 10000)


def _hash_code(code: str) -> str:
    """Hash a verification code for storage."""
    return hashlib.sha256(code.encode('utf-8')).hexdigest()


def handle_auth_routes(handler, method, path, data, db, strat, send_json, email_service=None):
    """Handle all /api/auth/* routes. Returns True if handled, False otherwise."""

    if path == "/api/auth/registration-status" and method == "GET":
        cursor = db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        row = cursor.fetchone()
        user_count = row[0] if row else 0
        has_smtp = email_service is not None and email_service.is_configured()
        send_json(handler, {
            "has_users": user_count > 0,
            "open_registration": True,  # Registration always open, gated by email verification
            "smtp_configured": has_smtp,
        })
        return True

    if path == "/api/auth/register" and method == "POST":
        email = (data.get("email") or "").strip().lower()
        password = data.get("password", "")
        display_name = (data.get("display_name") or "").strip()

        if not email or "@" not in email:
            send_json(handler, {"error": "Valid email is required"}, status=400)
            return True
        if not password or len(password) < 8:
            send_json(handler, {"error": "Password must be at least 8 characters"}, status=400)
            return True
        if not display_name or len(display_name) < 2:
            send_json(handler, {"error": "Username must be at least 2 characters"}, status=400)
            return True

        cursor = db.conn.cursor()

        # Check if email already exists as a verified user
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            send_json(handler, {"error": "Email already registered"}, status=409)
            return True

        # Store in pending_registrations (NOT users) until email is verified
        password_hash = _hash_password(password)
        code = _generate_code()
        verification_code_hash = _hash_code(code)
        verification_expires = (datetime.now() + timedelta(minutes=15)).isoformat()

        # First user will be admin
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        is_first = user_count == 0

        # Upsert into pending_registrations (allow re-registration with new code)
        cursor.execute("DELETE FROM pending_registrations WHERE email = ?", (email,))
        cursor.execute("""
            INSERT INTO pending_registrations (email, password_hash, display_name, is_admin,
                         verification_code_hash, verification_expires)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (email, password_hash, display_name, is_first,
              verification_code_hash, verification_expires))
        db._commit()

        # Send verification email (or log code if SMTP not configured)
        if email_service and email_service.is_configured():
            try:
                email_service.send_verification(email, code, display_name)
            except Exception as e:
                logger.warning(f"Failed to send verification email: {e}")
        else:
            logger.info(f"[EMAIL VERIFICATION] Code for {email}: {code} (SMTP not configured)")

        send_json(handler, {
            "status": "registered",
            "display_name": display_name,
            "needs_verification": True,
        }, status=201)
        return True

    if path == "/api/auth/verify" and method == "POST":
        email = (data.get("email") or "").strip().lower()
        code = (data.get("code") or "").strip()

        cursor = db.conn.cursor()
        # Look up pending registration (user doesn't exist in users table yet)
        cursor.execute(
            "SELECT id, password_hash, display_name, is_admin, verification_code_hash, verification_expires FROM pending_registrations WHERE email = ?",
            (email,)
        )
        row = cursor.fetchone()
        if not row:
            send_json(handler, {"error": "No pending registration for this email"}, status=404)
            return True

        pending_id, password_hash, display_name, is_admin, stored_hash, expires = row
        if expires and datetime.fromisoformat(expires) < datetime.now():
            send_json(handler, {"error": "Code expired. Request a new one."}, status=400)
            return True
        if _hash_code(code) != stored_hash:
            send_json(handler, {"error": "Invalid code"}, status=400)
            return True

        # Verification passed — NOW create the user in the users table
        cursor.execute("""
            INSERT INTO users (email, password_hash, display_name, is_admin, email_verified)
            VALUES (?, ?, ?, ?, TRUE)
        """, (email, password_hash, display_name, is_admin))
        user_id = cursor.lastrowid

        # Create per-user data directory
        user_data.ensure_dir(user_id)

        # Remove from pending
        cursor.execute("DELETE FROM pending_registrations WHERE id = ?", (pending_id,))

        # Create session — log the user in
        token = _generate_token()
        expires = (datetime.now() + timedelta(days=7)).isoformat()
        cursor.execute("""
            INSERT INTO sessions (token, user_id, expires_at)
            VALUES (?, ?, ?)
        """, (token, user_id, expires))
        db._commit()

        send_json(handler, {
            "status": "verified",
            "token": token,
            "display_name": display_name,
            "is_admin": bool(is_admin),
        })
        return True

    if path == "/api/auth/resend-verification" and method == "POST":
        email = (data.get("email") or "").strip().lower()
        cursor = db.conn.cursor()
        # Check pending_registrations (users table doesn't have unverified users anymore)
        cursor.execute("SELECT id, display_name FROM pending_registrations WHERE email = ?", (email,))
        row = cursor.fetchone()
        if not row:
            send_json(handler, {"error": "No pending registration for this email"}, status=404)
            return True
        pending_id, display_name = row
        code = _generate_code()
        cursor.execute("""
            UPDATE pending_registrations SET verification_code_hash = ?, verification_expires = ? WHERE id = ?
        """, (_hash_code(code), (datetime.now() + timedelta(minutes=15)).isoformat(), pending_id))
        db._commit()

        if email_service and email_service.is_configured():
            try:
                email_service.send_verification(email, code, display_name)
            except Exception as e:
                logger.warning(f"Failed to send verification email: {e}")
        else:
            logger.info(f"[EMAIL VERIFICATION] Resend code for {email}: {code} (SMTP not configured)")

        send_json(handler, {"status": "sent", "cooldown": 60})
        return True

    if path == "/api/auth/login" and method == "POST":
        identifier = (data.get("email") or "").strip()
        password = data.get("password", "")

        cursor = db.conn.cursor()
        # Support login by email or username (display_name)
        if "@" in identifier:
            cursor.execute("SELECT id, password_hash, display_name, is_admin, email_verified FROM users WHERE email = ?",
                           (identifier.lower(),))
        else:
            cursor.execute("SELECT id, password_hash, display_name, is_admin, email_verified FROM users WHERE LOWER(display_name) = ?",
                           (identifier.lower(),))
        row = cursor.fetchone()
        if not row:
            send_json(handler, {"error": "Invalid email or password"}, status=401)
            return True

        user_id, stored_hash, display_name, is_admin, email_verified = row
        if not _verify_password(password, stored_hash):
            send_json(handler, {"error": "Invalid email or password"}, status=401)
            return True

        # Block unverified users
        if not email_verified:
            send_json(handler, {"error": "Email not verified. Please check your inbox."}, status=403)
            return True

        # Upgrade legacy hash
        _upgrade_password(db, user_id, password, stored_hash)

        # Update last_login
        cursor.execute("UPDATE users SET last_login = ? WHERE id = ?",
                       (datetime.now().isoformat(), user_id))

        # Create session
        token = _generate_token()
        expires = (datetime.now() + timedelta(days=7)).isoformat()

        # Find default or last-active profile
        cursor.execute("""
            SELECT id, name FROM profiles WHERE user_id = ?
            ORDER BY is_default DESC, last_active DESC NULLS LAST
            LIMIT 1
        """, (user_id,))
        profile_row = cursor.fetchone()
        profile_id = profile_row[0] if profile_row else None

        cursor.execute("""
            INSERT INTO sessions (token, user_id, profile_id, expires_at, last_active)
            VALUES (?, ?, ?, ?, ?)
        """, (token, user_id, profile_id, expires, datetime.now().isoformat()))
        db._commit()

        # Get profile list
        cursor.execute("SELECT id, name, last_active FROM profiles WHERE user_id = ?", (user_id,))
        profiles = [{"id": r[0], "name": r[1], "last_active": r[2]} for r in cursor.fetchall()]

        send_json(handler, {
            "status": "authenticated",
            "token": token,
            "user_id": user_id,
            "display_name": display_name,
            "is_admin": bool(is_admin),
            "email_verified": bool(email_verified),
            "active_profile_id": profile_id,
            "profiles": profiles,
        })
        return True

    if path == "/api/auth/delete-account" and method == "POST":
        token = handler.headers.get("X-Auth-Token", "")
        user_id = _get_user_from_token(db, token)
        if not user_id:
            send_json(handler, {"error": "Not authenticated"}, status=401)
            return True

        password = data.get("password", "")
        cursor = db.conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if not row or not _verify_password(password, row[0]):
            send_json(handler, {"error": "Incorrect password"}, status=403)
            return True

        # Delete all user data (profiles cascade via FK, sessions too)
        cursor.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM profiles WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        db._commit()
        logger.info(f"User {user_id} deleted their account")
        send_json(handler, {"status": "deleted"})
        return True

    if path == "/api/auth/logout" and method == "POST":
        token = handler.headers.get("X-Auth-Token", "")
        if token:
            cursor = db.conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
            db._commit()
        send_json(handler, {"status": "logged_out"})
        return True

    if path == "/api/auth/check" and method == "GET":
        token = handler.headers.get("X-Auth-Token", "")
        if not token:
            send_json(handler, {"authenticated": False}, status=401)
            return True
        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT s.user_id, s.profile_id, s.expires_at, u.display_name, u.is_admin, u.email_verified
            FROM sessions s JOIN users u ON s.user_id = u.id
            WHERE s.token = ?
        """, (token,))
        row = cursor.fetchone()
        if not row or (row[2] and datetime.fromisoformat(row[2]) < datetime.now()):
            send_json(handler, {"authenticated": False}, status=401)
            return True

        user_id, profile_id, _, display_name, is_admin, email_verified = row

        # Sliding expiry
        new_expires = (datetime.now() + timedelta(days=7)).isoformat()
        cursor.execute("UPDATE sessions SET expires_at = ?, last_active = ? WHERE token = ?",
                       (new_expires, datetime.now().isoformat(), token))
        db._commit()

        # Get profiles
        cursor.execute("SELECT id, name, last_active FROM profiles WHERE user_id = ?", (user_id,))
        profiles = [{"id": r[0], "name": r[1], "last_active": r[2]} for r in cursor.fetchall()]

        send_json(handler, {
            "authenticated": True,
            "user_id": user_id,
            "display_name": display_name,
            "is_admin": bool(is_admin),
            "email_verified": bool(email_verified),
            "active_profile_id": profile_id,
            "profiles": profiles,
        })
        return True

    if path == "/api/auth/change-password" and method == "POST":
        token = handler.headers.get("X-Auth-Token", "")
        user_id = _get_user_from_token(db, token)
        if not user_id:
            send_json(handler, {"error": "Not authenticated"}, status=401)
            return True

        old_password = data.get("old_password", "")
        new_password = data.get("new_password", "")
        if not new_password or len(new_password) < 8:
            send_json(handler, {"error": "New password must be at least 8 characters"}, status=400)
            return True

        cursor = db.conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if not row or not _verify_password(old_password, row[0]):
            send_json(handler, {"error": "Current password is incorrect"}, status=400)
            return True

        new_hash = _hash_password(new_password)
        cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id))
        db._commit()
        send_json(handler, {"status": "password_changed"})
        return True

    if path == "/api/auth/forgot-password" and method == "POST":
        email = (data.get("email") or "").strip().lower()
        cursor = db.conn.cursor()
        cursor.execute("SELECT id, display_name FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        # Always return success to prevent email enumeration
        if not row or not email_service or not email_service.is_configured():
            send_json(handler, {"status": "sent"})
            return True

        user_id, display_name = row
        code = _generate_code()
        cursor.execute("""
            UPDATE users SET reset_code_hash = ?, reset_code_expires = ? WHERE id = ?
        """, (_hash_code(code), (datetime.now() + timedelta(minutes=15)).isoformat(), user_id))
        db._commit()
        try:
            email_service.send_reset_code(email, code, display_name)
        except Exception as e:
            logger.warning(f"Failed to send reset email: {e}")
        send_json(handler, {"status": "sent"})
        return True

    if path == "/api/auth/reset-password" and method == "POST":
        email = (data.get("email") or "").strip().lower()
        code = (data.get("code") or "").strip()
        new_password = data.get("new_password", "")

        if not new_password or len(new_password) < 8:
            send_json(handler, {"error": "Password must be at least 8 characters"}, status=400)
            return True

        cursor = db.conn.cursor()
        cursor.execute("SELECT id, reset_code_hash, reset_code_expires FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        if not row or not row[1]:
            send_json(handler, {"error": "No reset request found"}, status=400)
            return True

        user_id, stored_hash, expires = row
        if expires and datetime.fromisoformat(expires) < datetime.now():
            send_json(handler, {"error": "Code expired"}, status=400)
            return True
        if _hash_code(code) != stored_hash:
            send_json(handler, {"error": "Invalid code"}, status=400)
            return True

        new_hash = _hash_password(new_password)
        cursor.execute("""
            UPDATE users SET password_hash = ?, reset_code_hash = NULL, reset_code_expires = NULL WHERE id = ?
        """, (new_hash, user_id))
        # Invalidate all sessions for this user
        cursor.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        db._commit()
        send_json(handler, {"status": "password_reset"})
        return True

    # --- Profile management ---

    if path == "/api/profiles" and method == "GET":
        token = handler.headers.get("X-Auth-Token", "")
        user_id = _get_user_from_token(db, token)
        if not user_id:
            send_json(handler, {"error": "Not authenticated"}, status=401)
            return True

        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT id, name, config_overlay, is_default, last_active, created_at
            FROM profiles WHERE user_id = ?
            ORDER BY is_default DESC, last_active DESC NULLS LAST
        """, (user_id,))
        profiles = []
        for r in cursor.fetchall():
            overlay = json.loads(r[2]) if r[2] else {}
            profiles.append({
                "id": r[0], "name": r[1],
                "role": overlay.get("profile", {}).get("role", ""),
                "location": overlay.get("profile", {}).get("location", ""),
                "is_default": bool(r[3]),
                "last_active": r[4], "created_at": r[5],
            })
        send_json(handler, {"profiles": profiles})
        return True

    if path == "/api/profiles" and method == "POST":
        token = handler.headers.get("X-Auth-Token", "")
        user_id = _get_user_from_token(db, token)
        if not user_id:
            send_json(handler, {"error": "Not authenticated"}, status=401)
            return True

        name = (data.get("name") or "").strip()
        if not name or len(name) < 2:
            send_json(handler, {"error": "Profile name must be at least 2 characters"}, status=400)
            return True

        # Build clean config overlay from defaults
        config_overlay = {
            "profile": {
                "role": data.get("role", ""),
                "location": data.get("location", ""),
                "context": data.get("context", ""),
            },
            "market": {"tickers": []},
            "news": {"timelimit": "w"},
            "dynamic_categories": [],
        }

        cursor = db.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO profiles (user_id, name, config_overlay)
                VALUES (?, ?, ?)
            """, (user_id, name, json.dumps(config_overlay)))
            profile_id = cursor.lastrowid
            db._commit()
            send_json(handler, {"status": "created", "profile_id": profile_id, "name": name}, status=201)
        except Exception as e:
            if "UNIQUE" in str(e):
                send_json(handler, {"error": "Profile name already exists"}, status=409)
            else:
                send_json(handler, {"error": str(e)}, status=500)
        return True

    if path.startswith("/api/profiles/") and path.endswith("/activate") and method == "POST":
        token = handler.headers.get("X-Auth-Token", "")
        user_id = _get_user_from_token(db, token)
        if not user_id:
            send_json(handler, {"error": "Not authenticated"}, status=401)
            return True

        try:
            profile_id = int(path.split("/")[-2])
        except (ValueError, IndexError):
            send_json(handler, {"error": "Invalid profile ID"}, status=400)
            return True

        cursor = db.conn.cursor()
        cursor.execute("SELECT id, name FROM profiles WHERE id = ? AND user_id = ?", (profile_id, user_id))
        row = cursor.fetchone()
        if not row:
            send_json(handler, {"error": "Profile not found"}, status=404)
            return True

        # Update session's active profile
        cursor.execute("UPDATE sessions SET profile_id = ? WHERE token = ?", (profile_id, token))
        cursor.execute("UPDATE profiles SET last_active = ? WHERE id = ?",
                       (datetime.now().isoformat(), profile_id))
        db._commit()

        # Ensure per-user data directory exists
        user_data.ensure_dir(user_id)

        send_json(handler, {"status": "activated", "profile_id": profile_id, "name": row[1]})
        return True

    if path.startswith("/api/profiles/") and method == "DELETE":
        token = handler.headers.get("X-Auth-Token", "")
        user_id = _get_user_from_token(db, token)
        if not user_id:
            send_json(handler, {"error": "Not authenticated"}, status=401)
            return True

        try:
            profile_id = int(path.split("/")[-1])
        except (ValueError, IndexError):
            send_json(handler, {"error": "Invalid profile ID"}, status=400)
            return True

        cursor = db.conn.cursor()
        cursor.execute("SELECT id FROM profiles WHERE id = ? AND user_id = ?", (profile_id, user_id))
        if not cursor.fetchone():
            send_json(handler, {"error": "Profile not found"}, status=404)
            return True

        cursor.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
        db._commit()
        send_json(handler, {"status": "deleted"})
        return True

    # --- Admin routes ---

    if path == "/api/admin/users" and method == "GET":
        token = handler.headers.get("X-Auth-Token", "")
        user_id = _get_user_from_token(db, token)
        if not user_id or not _is_admin(db, user_id):
            send_json(handler, {"error": "Admin required"}, status=403)
            return True

        cursor = db.conn.cursor()
        cursor.execute("SELECT id, email, display_name, is_admin, email_verified, created_at, last_login FROM users")
        users = [{"id": r[0], "email": r[1], "display_name": r[2], "is_admin": bool(r[3]),
                  "email_verified": bool(r[4]), "created_at": r[5], "last_login": r[6]}
                 for r in cursor.fetchall()]
        send_json(handler, {"users": users})
        return True

    if path == "/api/admin/invite" and method == "POST":
        token = handler.headers.get("X-Auth-Token", "")
        user_id = _get_user_from_token(db, token)
        if not user_id or not _is_admin(db, user_id):
            send_json(handler, {"error": "Admin required"}, status=403)
            return True

        code = secrets.token_urlsafe(6)[:8].upper()
        expires = (datetime.now() + timedelta(days=7)).isoformat()
        cursor = db.conn.cursor()
        cursor.execute("INSERT INTO invite_codes (code, created_by, expires_at) VALUES (?, ?, ?)",
                       (code, user_id, expires))
        db._commit()
        send_json(handler, {"code": code, "expires_at": expires})
        return True

    if path == "/api/admin/reset-user" and method == "POST":
        token = handler.headers.get("X-Auth-Token", "")
        admin_id = _get_user_from_token(db, token)
        if not admin_id or not _is_admin(db, admin_id):
            send_json(handler, {"error": "Admin required"}, status=403)
            return True

        target_user_id = data.get("user_id")
        new_password = data.get("new_password", "")
        if not target_user_id or not new_password or len(new_password) < 8:
            send_json(handler, {"error": "user_id and new_password (8+ chars) required"}, status=400)
            return True

        new_hash = _hash_password(new_password)
        cursor = db.conn.cursor()
        cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, target_user_id))
        cursor.execute("DELETE FROM sessions WHERE user_id = ?", (target_user_id,))
        db._commit()
        send_json(handler, {"status": "password_reset"})
        return True

    return False


def _get_user_from_token(db, token: str):
    """Get user_id from a valid session token."""
    if not token:
        return None
    cursor = db.conn.cursor()
    cursor.execute("SELECT user_id, expires_at FROM sessions WHERE token = ?", (token,))
    row = cursor.fetchone()
    if not row:
        return None
    if row[1] and datetime.fromisoformat(row[1]) < datetime.now():
        cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
        db._commit()
        return None
    return row[0]


def _is_admin(db, user_id: int) -> bool:
    """Check if a user is admin."""
    cursor = db.conn.cursor()
    cursor.execute("SELECT is_admin FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    return bool(row and row[0])

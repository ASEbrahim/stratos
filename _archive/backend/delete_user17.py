#!/usr/bin/env python3
"""Delete all data for user_id=17 EXCEPT the auth.users row."""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent / "strat_os.db"
USER_ID = 17

def main(dry_run=True):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    # 1. Find profile IDs owned by this user
    cur.execute("SELECT id, name FROM profiles WHERE user_id = ?", (USER_ID,))
    profiles = cur.fetchall()
    profile_ids = [p[0] for p in profiles]

    if not profiles:
        print(f"No profiles found for user_id={USER_ID}. Nothing to delete.")
        return

    print(f"User {USER_ID} has profiles: {profiles}")
    placeholders = ",".join("?" * len(profile_ids))

    # Tables with profile_id FK
    profile_tables = ["news_items", "user_feedback", "briefings", "scan_log", "shadow_scores"]

    # Count and delete from profile-linked tables
    for table in profile_tables:
        cur.execute(f"SELECT COUNT(*) FROM {table} WHERE profile_id IN ({placeholders})", profile_ids)
        count = cur.fetchone()[0]
        print(f"  {table}: {count} rows to delete")
        if not dry_run:
            cur.execute(f"DELETE FROM {table} WHERE profile_id IN ({placeholders})", profile_ids)

    # Sessions (user_id FK)
    cur.execute("SELECT COUNT(*) FROM sessions WHERE user_id = ?", (USER_ID,))
    count = cur.fetchone()[0]
    print(f"  sessions: {count} rows to delete")
    if not dry_run:
        cur.execute("DELETE FROM sessions WHERE user_id = ?", (USER_ID,))

    # Invite codes (created_by or used_by)
    cur.execute("SELECT COUNT(*) FROM invite_codes WHERE created_by = ? OR used_by = ?", (USER_ID, USER_ID))
    count = cur.fetchone()[0]
    print(f"  invite_codes: {count} rows to delete")
    if not dry_run:
        cur.execute("DELETE FROM invite_codes WHERE created_by = ? OR used_by = ?", (USER_ID, USER_ID))

    # Profiles (user_id FK) — delete last since other tables reference profile_id
    cur.execute("SELECT COUNT(*) FROM profiles WHERE user_id = ?", (USER_ID,))
    count = cur.fetchone()[0]
    print(f"  profiles: {count} rows to delete")
    if not dry_run:
        cur.execute("DELETE FROM profiles WHERE user_id = ?", (USER_ID,))

    print(f"\n  users: SKIPPED (keeping auth row for user_id={USER_ID})")

    if dry_run:
        print("\n[DRY RUN] No changes made. Run with --execute to delete.")
        conn.close()
    else:
        conn.commit()
        conn.close()
        print("\nDone. All user_id=17 data deleted (users row preserved).")


if __name__ == "__main__":
    execute = "--execute" in sys.argv
    if execute:
        confirm = input("This will permanently delete all data for user_id=17 (except users row). Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(0)
    main(dry_run=not execute)

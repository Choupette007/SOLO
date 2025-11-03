#!/usr/bin/env python3
"""
One-shot migration: copy legacy 'json' or 'raw_json' columns into canonical 'data' column
on discovered_tokens if needed. Idempotent.

Usage:
  python scripts/migrate_discovered_json_to_data.py --db PATH_TO_DB
"""
import argparse
import sqlite3
import shutil
import os
import time

def backup_db(db_path: str) -> str:
    ts = time.strftime("%Y%m%dT%H%M%S")
    bak = f"{db_path}.{ts}.bak"
    shutil.copy2(db_path, bak)
    return bak

def run_migration(db_path: str) -> None:
    if not os.path.exists(db_path):
        raise SystemExit(f"DB not found: {db_path}")
    print("Backing up DB...")
    bak = backup_db(db_path)
    print("Backup created:", bak)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='discovered_tokens'")
    if not cur.fetchone():
        print("discovered_tokens table not present; nothing to do.")
        conn.close()
        return

    cols = [r[1] for r in cur.execute("PRAGMA table_info(discovered_tokens)").fetchall()]
    print("discovered_tokens columns:", cols)

    has_data = "data" in cols
    has_json = "json" in cols
    has_raw = "raw_json" in cols

    if not has_data:
        print("Adding 'data' column to discovered_tokens...")
        cur.execute("ALTER TABLE discovered_tokens ADD COLUMN data TEXT;")
        conn.commit()
        has_data = True

    source = None
    if has_json:
        source = "json"
    elif has_raw:
        source = "raw_json"

    if not source:
        print("No legacy 'json' or 'raw_json' column found. No copy necessary.")
        conn.close()
        return

    cur.execute(f"SELECT COUNT(*) FROM discovered_tokens WHERE ({source} IS NOT NULL AND {source} != '') AND (data IS NULL OR data = '')")
    to_copy = cur.fetchone()[0]
    print(f"Rows to copy from '{source}' -> 'data': {to_copy}")
    if to_copy == 0:
        print("Nothing to copy. Exiting.")
        conn.close()
        return

    batch = 500
    updated = 0
    while True:
        cur.execute(
            f"SELECT address, {source} FROM discovered_tokens WHERE ({source} IS NOT NULL AND {source} != '') AND (data IS NULL OR data = '') LIMIT ?",
            (batch,),
        )
        rows = cur.fetchall()
        if not rows:
            break
        for addr, src in rows:
            if src is None:
                continue
            cur.execute("UPDATE discovered_tokens SET data = ? WHERE address = ?", (src, addr))
            updated += 1
        conn.commit()
    print(f"Copied {updated} rows from '{source}' to 'data'.")
    conn.close()
    print("Migration complete. Backup at:", bak)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="Path to tokens.sqlite3")
    args = ap.parse_args()
    dbp = args.db or os.path.expanduser(os.path.join(os.getenv("LOCALAPPDATA", os.path.expanduser("~")), "SOLOTradingBot", "tokens.sqlite3"))
    run_migration(dbp)
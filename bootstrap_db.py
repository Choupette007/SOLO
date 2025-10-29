#!/usr/bin/env python3
import os, sqlite3
from pathlib import Path

def resolve_db_path():
    la = os.getenv("LOCALAPPDATA","")
    if la:
        p = Path(la)/"SOLOTradingBot"/"tokens.sqlite3"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return Path("tokens.sqlite3")

SCHEMA = [
    # Tracks currently eligible tokens for UI display
    """
    CREATE TABLE IF NOT EXISTS eligible_shortlist (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        address TEXT,
        symbol TEXT,
        name TEXT,
        volume_24h REAL,
        liquidity REAL,
        market_cap REAL,
        price REAL,
        price_change_1h REAL,
        price_change_6h REAL,
        price_change_24h REAL,
        categories TEXT,        -- JSON string (e.g., ["mid_cap","newly_launched"])
        score REAL,
        timestamp INTEGER
    );
    """,
    # Tracks buys still open
    """
    CREATE TABLE IF NOT EXISTS open_positions (
        address TEXT PRIMARY KEY,
        symbol TEXT,
        buy_price REAL,
        buy_txid TEXT,
        buy_time INTEGER
    );
    """,
    # Simple blacklist
    """
    CREATE TABLE IF NOT EXISTS blacklist (
        address TEXT PRIMARY KEY,
        reason TEXT,
        expires_at INTEGER
    );
    """,
    # Optional cache table (safe superset)
    """
    CREATE TABLE IF NOT EXISTS token_cache (
        address TEXT PRIMARY KEY,
        payload TEXT,
        updated_at INTEGER
    );
    """
]

def main():
    db_path = resolve_db_path()
    print(f"[bootstrap] Using DB: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        for stmt in SCHEMA:
            cur.execute(stmt)
        conn.commit()
        # show tables
        rows = cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;").fetchall()
        print("[bootstrap] Tables now:", [r[0] for r in rows])
    finally:
        conn.close()
    print("Done.")

if __name__ == "__main__":
    main()

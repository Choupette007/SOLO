# solana_trading_bot_bundle/trading_bot/database.py
from __future__ import annotations

import aiosqlite
import logging
import json
import time
import re
import os
import traceback
from datetime import datetime
from typing import Dict, Set, Optional, List, Tuple, Any, Sequence

from solana_trading_bot_bundle.common.constants import (
    token_cache_path,  # fallback if not set in config.yaml
)

from .utils import (
    load_config,
    custom_json_encoder,
    WHITELISTED_TOKENS,
)

logger = logging.getLogger("TradingBot")

# =========================
# Connection / Path helpers
# =========================

logger = logging.getLogger(__name__)

class _ConnCtx:
    def __init__(self, path: str):
        self._path = path
        self._db: Optional[aiosqlite.Connection] = None

    async def __aenter__(self) -> aiosqlite.Connection:
        self._db = await aiosqlite.connect(self._path)
        try:
            await self._db.execute("PRAGMA journal_mode=WAL;")
            await self._db.execute("PRAGMA busy_timeout=30000;")
            await self._db.execute("PRAGMA synchronous=NORMAL;")
            await self._db.execute("PRAGMA foreign_keys=ON;")
            self._db.row_factory = aiosqlite.Row
            # ★ Ensure schema exists on every connection (idempotent)
            await _ensure_core_schema(self._db)
        except Exception:
            # We don't want connection to fail due to a PRAGMA/DDL hiccup
            logger.debug(
                "Schema bootstrap on enter encountered a non-fatal issue:\n%s",
                traceback.format_exc()
            )
        return self._db

    async def __aexit__(self, exc_type, exc, tb):
        try:
            if self._db is not None:
                await self._db.close()
        finally:
            self._db = None


def _default_token_cache_path_value() -> str:
    """
    Safely obtain the default token cache path.

    `token_cache_path` may be either a string/Path or a function returning one.
    Also normalize JSON defaults to SQLite.
    """
    try:
        val = token_cache_path() if callable(token_cache_path) else token_cache_path
    except TypeError:
        val = token_cache_path

    # normalize to absolute filesystem path
    p = os.path.abspath(os.path.expanduser(os.path.expandvars(str(val))))
    # If a legacy JSON default slipped in, upgrade filename to tokens.sqlite3
    if p.lower().endswith(".json"):
        p = os.path.join(os.path.dirname(p), "tokens.sqlite3")
    return p


# one-time suppression for noisy Windows-path-on-POSIX warnings
_WARNED_WINPATH = False
def _warn_once_winpath(msg: str) -> None:
    global _WARNED_WINPATH
    if not _WARNED_WINPATH:
        logger.warning(msg)
        _WARNED_WINPATH = True


def _looks_windows_path(s: str) -> bool:
    # Drive + backslashes is a simple robust heuristic
    return ("\\" in s) and (":" in s)


def _resolve_db_path() -> str:
    """
    Resolve DB path from config with a safe absolute path, and ensure parent dir exists.

    Fixes:
      - on macOS/Linux, ignore Windows-style paths (e.g. 'C:\\Users\\...\\tokens.db')
      - if a directory or suffixless path is provided, use 'tokens.sqlite3' inside it
      - if a legacy '*.json' path is provided, upgrade to 'tokens.sqlite3'
    """
    # Platform/default
    default_dbp = _default_token_cache_path_value()

    # Read configured value (be careful to avoid GUI import cycles)
    try:
        # Prefer a local safe loader if you have one; otherwise very light YAML read:
        import yaml
        cfg_path = os.path.join(os.getcwd(), "config.yaml")
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}

    db_section = (cfg.get("database") or {}) if isinstance(cfg, dict) else {}
    cfg_dbp = db_section.get("token_cache_path") or db_section.get("path")

    # Choose candidate
    if cfg_dbp:
        candidate = os.path.expanduser(os.path.expandvars(str(cfg_dbp)))

        # If we're NOT on Windows and the path looks Windows-y, fall back to default
        if os.name != "nt" and _looks_windows_path(candidate):
            _warn_once_winpath(
                f"Configured token_cache_path looks Windows-style on this OS ({candidate}). "
                "Ignoring and using default AppData path instead."
            )
            dbp = default_dbp
        else:
            dbp = os.path.abspath(candidate)
    else:
        dbp = default_dbp

    # If a directory (or suffixless) was provided, put a sensible filename inside it
    base = os.path.basename(dbp)
    if (not os.path.splitext(base)[1]) or dbp.endswith(("/", "\\")):
        dbp = os.path.join(dbp, "tokens.sqlite3") if os.path.isdir(dbp) else os.path.join(os.path.dirname(dbp), "tokens.sqlite3")

    # Legacy JSON -> coerce to sqlite3 (same directory)
    if dbp.lower().endswith(".json"):
        logger.info("Upgrading legacy JSON DB path to SQLite: %s -> tokens.sqlite3", dbp)
        dbp = os.path.join(os.path.dirname(dbp), "tokens.sqlite3")

    os.makedirs(os.path.dirname(dbp), exist_ok=True)
    return dbp


def _connect(dbp: Optional[str] = None) -> _ConnCtx:
    """Return an async context manager for a DB connection without pre-awaiting."""
    path = dbp or _resolve_db_path()
    return _ConnCtx(path)

# Public alias used by GUI and bot code
def connect_db(dbp: Optional[str] = None) -> _ConnCtx:
    return _connect(dbp)


async def _exec(db: aiosqlite.Connection, sql: str, params: Tuple = ()) -> None:
    await db.execute(sql, params)


# ============================
# Compat view helper (minimal)
# ============================
async def _create_compat_views(db: aiosqlite.Connection) -> None:
    """
    Create/refresh compatibility views expected by older GUI code.
    Safe to call repeatedly.
    """
    await db.executescript("""
        DROP VIEW IF EXISTS eligible_tokens_view;
        CREATE VIEW eligible_tokens_view AS
        SELECT
            address        AS token_address,
            name,
            symbol,
            volume_24h,
            liquidity,
            market_cap,
            price,
            price_change_1h,
            price_change_6h,
            price_change_24h,
            score,
            categories,
            timestamp,
            data,
            created_at
        FROM eligible_tokens;
    """)
    await db.commit()


# =====================
# Core schema bootstrap
# =====================
async def _ensure_core_schema(db: aiosqlite.Connection) -> None:
    """
    Create/patch all tables and indexes the app relies on.
    Safe to call repeatedly. Keeps GUI/tests working even if init_db() wasn’t called.
    """
    # --- Base tables ---
    await _exec(db, """
        CREATE TABLE IF NOT EXISTS tokens (
            address TEXT PRIMARY KEY,
            name TEXT,
            symbol TEXT,
            volume_24h REAL,
            liquidity REAL,
            market_cap REAL,
            price REAL,
            price_change_1h REAL,
            price_change_6h REAL,
            price_change_24h REAL,
            categories TEXT,
            timestamp INTEGER,
            buy_price REAL,
            buy_txid TEXT,
            buy_time INTEGER,
            sell_price REAL,
            sell_txid TEXT,
            sell_time INTEGER,
            is_trading BOOLEAN
        );
    """)

    await _exec(db, """
        CREATE TABLE IF NOT EXISTS blacklist (
            address TEXT PRIMARY KEY,
            reason TEXT,
            timestamp INTEGER
        );
    """)

    await _exec(db, """
        CREATE TABLE IF NOT EXISTS eligible_tokens (
            address TEXT PRIMARY KEY,
            name TEXT,
            symbol TEXT,
            volume_24h REAL,
            liquidity REAL,
            market_cap REAL,
            price REAL,
            price_change_1h REAL,
            price_change_6h REAL,
            price_change_24h REAL,
            score REAL,
            categories TEXT,
            timestamp INTEGER,
            data TEXT,
            created_at INTEGER
        );
    """)

    # --- Backfill columns if table pre-existed ---
    cols = []
    async with db.execute("PRAGMA table_info(eligible_tokens)") as cur:
        async for row in cur:
            cols.append(row[1])
    if "data" not in cols:
        await _exec(db, "ALTER TABLE eligible_tokens ADD COLUMN data TEXT;")
    if "created_at" not in cols:
        await _exec(db, "ALTER TABLE eligible_tokens ADD COLUMN created_at INTEGER;")

    # --- Other core tables ---
    await _exec(db, """
        CREATE TABLE IF NOT EXISTS cached_token_data (
            address TEXT PRIMARY KEY,
            symbol TEXT,
            market_cap REAL,
            data TEXT
        );
    """)
    await _exec(db, """
        CREATE TABLE IF NOT EXISTS cached_creation_time (
            address TEXT PRIMARY KEY,
            creation_time TEXT
        );
    """)
    await _exec(db, """
        CREATE TABLE IF NOT EXISTS trade_history (
            trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_address TEXT,
            symbol TEXT,
            buy_price REAL,
            sell_price REAL,
            buy_amount REAL,
            sell_amount REAL,
            buy_txid TEXT,
            sell_txid TEXT,
            buy_time INTEGER,
            sell_time INTEGER,
            profit REAL,
            FOREIGN KEY (token_address) REFERENCES tokens(address)
        );
    """)

    # shortlist tokens (JSON-per-row: address, data)
    await _exec(db, """
        CREATE TABLE IF NOT EXISTS shortlist_tokens (
            address TEXT PRIMARY KEY,
            data    TEXT,
            created_at INTEGER DEFAULT (strftime('%s','now'))
        );
    """)
    sl_cols = []
    async with db.execute("PRAGMA table_info(shortlist_tokens)") as cur:
        async for row in cur:
            sl_cols.append(row[1])
    if "data" not in sl_cols:
        await _exec(db, "ALTER TABLE shortlist_tokens ADD COLUMN data TEXT;")
    if "created_at" not in sl_cols:
        await _exec(db, "ALTER TABLE shortlist_tokens ADD COLUMN created_at INTEGER;")

    # discovered_tokens (raw discovery JSON rows for GUI)
    await _exec(db, """
        CREATE TABLE IF NOT EXISTS discovered_tokens (
            address TEXT PRIMARY KEY,
            data TEXT,
            name TEXT,
            symbol TEXT,
            price REAL,
            liquidity REAL,
            market_cap REAL,
            v24hUSD REAL,
            volume_24h REAL,
            dexscreenerUrl TEXT,
            dsPairAddress TEXT,
            links TEXT,
            created_at INTEGER,
            creation_timestamp INTEGER
        );
    """)

    # --- Useful indexes ---
    await _exec(db, "CREATE INDEX IF NOT EXISTS idx_eligible_tokens_timestamp ON eligible_tokens(timestamp);")
    await _exec(db, "CREATE INDEX IF NOT EXISTS idx_eligible_tokens_score ON eligible_tokens(score);")
    await _exec(db, "CREATE INDEX IF NOT EXISTS idx_tokens_is_trading ON tokens(is_trading);")
    await _exec(db, "CREATE INDEX IF NOT EXISTS idx_trade_history_token ON trade_history(token_address);")
    await _exec(db, "CREATE INDEX IF NOT EXISTS idx_trade_history_buy_time ON trade_history(buy_time);")
    await _exec(db, "CREATE INDEX IF NOT EXISTS idx_trade_history_sell_time ON trade_history(sell_time);")
    await _exec(db, "CREATE INDEX IF NOT EXISTS idx_shortlist_created_at ON shortlist_tokens(created_at);")

    # --- Hygiene ---
    await _exec(db, "DELETE FROM tokens WHERE address IS NULL;")
    await _exec(db, "DELETE FROM eligible_tokens WHERE address IS NULL;")

    # --- Compatibility views for GUI/older queries ---
    await _create_compat_views(db)

    await db.commit()


# =====================
# Schema init/migration
# =====================
async def init_db(conn: Optional[aiosqlite.Connection] = None) -> None:
    """
    Initialize the trading bot database (idempotent).

    If `conn` is provided, uses that open connection.
    Otherwise, resolves the DB path and opens/closes its own connection.
    """
    if conn is not None:
        await _ensure_core_schema(conn)
        await _create_compat_views(conn)  # ensure views for existing external connection
        # Light verification on the provided connection
        try:
            async with conn.execute("SELECT name FROM sqlite_master WHERE type='table'") as cur:
                tables = {row[0] async for row in cur}
            expected = {
                'tokens','blacklist','eligible_tokens','cached_token_data',
                'cached_creation_time','trade_history','shortlist_tokens','discovered_tokens'
            }
            missing = expected - tables
            if missing:
                raise aiosqlite.OperationalError(f"Missing tables after init: {missing}")
            logger.info("Database initialized successfully (existing connection)")
        except Exception as e:
            logger.error("Verification on provided connection failed: %s\n%s", e, traceback.format_exc())
            raise
        return

    dbp = _resolve_db_path()
    logger.info(f"Initializing database at {dbp}")
    try:
        async with _connect(dbp) as db:
            await _ensure_core_schema(db)
            await _create_compat_views(db)  # ensure views for fresh connection

            # Quick verification
            async with db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cur:
                tables = {row[0] async for row in cur}
            expected = {
                'tokens','blacklist','eligible_tokens','cached_token_data',
                'cached_creation_time','trade_history','shortlist_tokens','discovered_tokens'
            }
            missing = expected - tables
            if missing:
                raise aiosqlite.OperationalError(f"Missing tables after init: {missing}")

            logger.info("Database initialized successfully")

    except aiosqlite.OperationalError as e:
        logger.error(f"Failed to initialize database at {dbp}: {e}\n{traceback.format_exc()}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error initializing database at {dbp}: {e}\n{traceback.format_exc()}")
        raise


# ==================
# Blacklist helpers
# ==================
async def load_blacklist() -> Set[str]:
    """Load the blacklist from the database."""
    try:
        async with _connect() as db:
            bl: Set[str] = set()
            async with db.execute("SELECT address, reason, timestamp FROM blacklist") as cur:
                rows = await cur.fetchall()
                for address, reason, ts in rows:
                    bl.add(address)
                    logger.debug(
                        "Blacklist entry: %s, Reason: %s, Timestamp: %s",
                        address, reason, datetime.fromtimestamp(ts)
                    )
            logger.info("Loaded %d blacklisted tokens", len(bl))
            return bl
    except Exception as e:
        logger.error(f"Failed to load blacklist: {e}\n{traceback.format_exc()}")
        return set()


async def add_to_blacklist(token_address: str, reason: str) -> None:
    """Add or update a token in the blacklist.""" 
    try:
        async with _connect() as db:
            await db.execute(
                "INSERT INTO blacklist(address, reason, timestamp) VALUES (?, ?, ?) "
                "ON CONFLICT(address) DO UPDATE SET reason=excluded.reason, timestamp=excluded.timestamp;",
                (token_address, reason, int(time.time())),
            )
            await db.commit()
        logger.info("Added %s to blacklist: %s", token_address, reason)
    except Exception as e:
        logger.error(f"Failed to add {token_address} to blacklist: {e}\n{traceback.format_exc()}")


async def clear_whitelisted_from_blacklist() -> None:
    """Remove whitelisted tokens from the blacklist table."""
    try:
        if not WHITELISTED_TOKENS:
            return
        async with _connect() as db:
            qmarks = ",".join(["?"] * len(WHITELISTED_TOKENS))
            await db.execute(
                f"DELETE FROM blacklist WHERE address IN ({qmarks});",
                tuple(WHITELISTED_TOKENS.keys())
            )
            await db.commit()
        logger.info("Removed whitelisted tokens from blacklist: %s", list(WHITELISTED_TOKENS.keys()))
    except Exception as e:
        logger.error(f"Failed to clear whitelisted tokens from blacklist: {e}\n{traceback.format_exc()}")


# ---------- OPTION B: keyword-friendly alias ----------
async def clear_expired_blacklist(max_age_hours: float = 24, *, hours: Optional[float] = None) -> None:
    """
    Remove blacklist entries older than `max_age_hours`.

    Compatibility:
      - Supports alias keyword `hours` used by callers like:
            await clear_expired_blacklist(hours=24.0)
      - If both are provided, `hours` takes precedence.
    """
    effective_hours = float(hours) if hours is not None else float(max_age_hours)
    cutoff = int(time.time()) - int(effective_hours * 3600)
    try:
        async with _connect() as db:
            before = db.total_changes
            await db.execute("DELETE FROM blacklist WHERE timestamp < ?;", (cutoff,))
            await db.commit()
            changed = db.total_changes - before
        logger.info(
            "Cleared %d expired blacklist entries older than %.1f hours",
            changed, effective_hours
        )
    except Exception as e:
        logger.error(f"Failed to clear expired blacklist: {e}\n{traceback.format_exc()}")
# ---------- end OPTION B ----------


async def review_blacklist() -> None:
    """Remove blacklist entries with transient reasons (e.g., rate limits)."""
    try:
        async with _connect() as db:
            async with db.execute("SELECT address, reason FROM blacklist") as cur:
                rows = await cur.fetchall()
            transient = ("rate limit", "http 429", "temporary")
            removed = 0
            for address, reason in rows:
                r = (reason or "").lower()
                if any(t in r for t in transient):
                    await db.execute("DELETE FROM blacklist WHERE address = ?;", (address,))
                    logger.info("Removed transient blacklist entry: %s (%s)", address, reason)
                    removed += 1
            if removed:
                await db.commit()
        logger.info("Reviewed blacklist, removed %d transient entries", removed)
    except Exception as e:
        logger.error(f"Failed to review blacklist: {e}\n{traceback.format_exc()}")


# =========================
# Creation-time cache helpers
# =========================
async def cache_creation_time(token_address: str, creation_time: Optional[datetime]) -> None:
    """Cache the creation time for a token in the database."""
    try:
        async with _connect() as db:
            # ensure table exists (in case init_db not called yet)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS cached_creation_time (
                    address TEXT PRIMARY KEY,
                    creation_time TEXT
                );
            """)
            creation_time_str = creation_time.isoformat() if creation_time else None
            await db.execute(
                "INSERT INTO cached_creation_time(address, creation_time) VALUES(?, ?) "
                "ON CONFLICT(address) DO UPDATE SET creation_time=excluded.creation_time;",
                (token_address, creation_time_str),
            )
            await db.commit()
            logger.debug("Cached creation time for %s: %s", token_address, creation_time_str)
    except Exception as e:
        logger.error(f"Failed to cache creation time for {token_address}: {e}\n{traceback.format_exc()}")


async def get_cached_creation_time(token_address: str) -> Optional[datetime]:
    """Retrieve cached creation time for a token from the database."""
    try:
        async with _connect() as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS cached_creation_time (
                    address TEXT PRIMARY KEY,
                    creation_time TEXT
                );
            """)
            async with db.execute(
                "SELECT creation_time FROM cached_creation_time WHERE address = ?;", (token_address,)
            ) as cur:
                row = await cur.fetchone()
                if row and row[0]:
                    logger.debug("Retrieved cached creation time for %s: %s", token_address, row[0])
                    return datetime.fromisoformat(row[0])
                return None
    except Exception as e:
        logger.error(f"Failed to retrieve cached creation time for {token_address}: {e}\n{traceback.format_exc()}")
        return None


# ======================
# Token data cache table
# ======================

async def cache_token_data(token: Dict) -> None:
    """Cache token data in the database."""
    token_address = token.get("address")
    if not token_address:
        logger.warning("Cannot cache token data: missing address for %s", token.get("symbol", "UNKNOWN"))
        return
    try:
        async with _connect() as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS cached_token_data (
                    address TEXT PRIMARY KEY,
                    symbol TEXT,
                    market_cap REAL,
                    data TEXT
                );
            """)
            token_data_json = json.dumps(token, default=custom_json_encoder)
            await db.execute(
                "INSERT INTO cached_token_data(address, symbol, market_cap, data) VALUES(?, ?, ?, ?) "
                "ON CONFLICT(address) DO UPDATE SET "
                "symbol=excluded.symbol, market_cap=excluded.market_cap, data=excluded.data;",
                (token_address, token.get("symbol", "UNKNOWN"), token.get("market_cap", 0), token_data_json),
            )
            await db.commit()
            logger.debug("Cached token data for %s (%s)", token.get("symbol", "UNKNOWN"), token_address)
    except Exception as e:
        logger.error(f"Failed to cache token data for {token.get('symbol','UNKNOWN')} ({token_address}): {e}\n{traceback.format_exc()}")


async def get_cached_token_data(token_address: str) -> Optional[Dict]:
    """Retrieve cached token data from the database."""
    try:
        async with _connect() as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS cached_token_data (
                    address TEXT PRIMARY KEY,
                    symbol TEXT,
                    market_cap REAL,
                    data TEXT
                );
            """)
            async with db.execute("SELECT data FROM cached_token_data WHERE address = ?;", (token_address,)) as cur:
                row = await cur.fetchone()
                if not row:
                    return None
                try:
                    data = json.loads(row[0])
                    if not isinstance(data, dict):
                        logger.error("Invalid cached data format for %s: %s", token_address, type(data))
                        return None
                    logger.debug("Retrieved cached token data for %s: %s...", token_address, json.dumps(data)[:100])
                    return data
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse cached data for {token_address}: {e}\n{traceback.format_exc()}")
                    return None
    except Exception as e:
        logger.error(f"Failed to retrieve cached token data for {token_address}: {e}\n{traceback.format_exc()}")
        return None


async def clear_birdeye_cache() -> None:
    """Clear corrupted birdeye_tokenlist cache."""
    try:
        async with _connect() as db:
            await db.execute("DELETE FROM cached_token_data WHERE address = ?;", ("birdeye_tokenlist",))
            await db.commit()
            logger.info("Cleared birdeye_tokenlist cache")
    except Exception as e:
        logger.error(f"Failed to clear birdeye_tokenlist cache: {e}\n{traceback.format_exc()}")


# =========================
# Tokens / Trades tables
# =========================

async def update_token_record(
    token: Dict,
    buy_price: float,
    buy_txid: str,
    buy_time: int,
    is_trading: bool = True,
) -> None:
    """
    Upsert a token row without nuking unrelated columns (avoid REPLACE).
    Only updates the provided field set on conflict.
    """
    try:
        categories_json = json.dumps(token.get("categories", []), default=custom_json_encoder)
        async with _connect() as db:
            await db.execute("""
                INSERT INTO tokens (
                    address, name, symbol, volume_24h, liquidity, market_cap, price,
                    price_change_1h, price_change_6h, price_change_24h, categories,
                    timestamp, buy_price, buy_txid, buy_time, is_trading
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(address) DO UPDATE SET
                    name=excluded.name,
                    symbol=excluded.symbol,
                    volume_24h=excluded.volume_24h,
                    liquidity=excluded.liquidity,
                    market_cap=excluded.market_cap,
                    price=excluded.price,
                    price_change_1h=excluded.price_change_1h,
                    price_change_6h=excluded.price_change_6h,
                    price_change_24h=excluded.price_change_24h,
                    categories=excluded.categories,
                    timestamp=excluded.timestamp,
                    buy_price=excluded.buy_price,
                    buy_txid=excluded.buy_txid,
                    buy_time=excluded.buy_time,
                    is_trading=excluded.is_trading;
            """, (
                token.get("address"),
                token.get("name", "UNKNOWN"),
                token.get("symbol", "UNKNOWN"),
                float(token.get("volume_24h", 0)),
                float(token.get("liquidity", 0)),
                float(token.get("market_cap", 0)),
                float(token.get("price", 0)),
                float(token.get("price_change_1h", 0)),
                float(token.get("price_change_6h", 0)),
                float(token.get("price_change_24h", 0)),
                categories_json,
                int(token.get("timestamp", time.time())),
                float(buy_price),
                buy_txid,
                int(buy_time),
                bool(is_trading),
            ))
            await db.commit()
        logger.debug("Updated token record for %s (%s)", token.get("symbol", "UNKNOWN"), token.get("address"))
    except Exception as e:
        logger.error(f"Failed to update token record for {token.get('address','UNKNOWN')}: {e}\n{traceback.format_exc()}")


async def record_trade(
    token: Dict,
    buy_price: float,
    buy_amount: float,
    buy_txid: str,
    buy_time: int,
    sell_price: float = None,
    sell_amount: float = None,
    sell_txid: str = None,
    sell_time: int = None,
) -> None:
    """Record a trade in the trade_history table."""
    profit = (sell_price - buy_price) * min(buy_amount, sell_amount or buy_amount) if (sell_price is not None and sell_amount is not None) else None
    try:
        async with _connect() as db:
            await db.execute("""
                INSERT INTO trade_history (
                    token_address, symbol, buy_price, sell_price, buy_amount, sell_amount,
                    buy_txid, sell_txid, buy_time, sell_time, profit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                token.get("address"),
                token.get("symbol", "UNKNOWN"),
                float(buy_price),
                float(sell_price) if sell_price is not None else None,
                float(buy_amount),
                float(sell_amount) if sell_amount is not None else None,
                buy_txid,
                sell_txid,
                int(buy_time),
                int(sell_time) if sell_time is not None else None,
                float(profit) if profit is not None else None,
            ))
            await db.commit()
        logger.debug("Recorded trade for %s (%s)", token.get("symbol", "UNKNOWN"), token.get("address"))
    except Exception as e:
        logger.error(f"Failed to record trade for {token.get('address','UNKNOWN')}: {e}\n{traceback.format_exc()}")


# ✅ open-positions reader
async def get_open_positions() -> List[Dict]:
    """
    Return rows from tokens that are currently marked as trading (open positions).
    """
    try:
        async with _connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM tokens WHERE is_trading = 1;") as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"get_open_positions failed: {e}\n{traceback.format_exc()}")
        return []


# ✅ sold-marker
async def mark_token_sold(
    token_address: str,
    sell_price: float,
    sell_txid: str,
    sell_time: int,
) -> None:
    """
    Update a token row as sold and no longer trading.
    """
    try:
        async with _connect() as db:
            await db.execute(
                """
                UPDATE tokens
                   SET sell_price = ?,
                       sell_txid  = ?,
                       sell_time  = ?,
                       is_trading = 0
                 WHERE address = ?;
                """,
                (float(sell_price), str(sell_txid), int(sell_time), token_address),
            )
            await db.commit()
        logger.debug("Marked %s sold @ %.8f (tx %s)", token_address, sell_price, sell_txid)
    except Exception as e:
        logger.error(f"mark_token_sold failed for {token_address}: {e}\n{traceback.format_exc()}")


# ============================
# eligible_tokens (shortlist)
# ============================

async def prune_old_eligible_tokens(max_age_hours: float = 168) -> None:
    """Remove eligible_tokens entries older than max_age_hours.""" 
    cutoff = int(time.time()) - int(max_age_hours * 3600)
    try:
        async with _connect() as db:
            before = db.total_changes
            await db.execute("DELETE FROM eligible_tokens WHERE timestamp < ?;", (cutoff,))
            await db.commit()
            deleted = db.total_changes - before
        logger.info("Pruned %d old eligible_tokens entries older than %.1f hours", deleted, max_age_hours)
    except Exception as e:
        logger.error(f"Failed to prune old eligible tokens: {e}\n{traceback.format_exc()}")


async def get_token_trade_status(token_address: str) -> Optional[Dict]:
    """Retrieve the trade status of a token from the tokens table.""" 
    try:
        async with _connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM tokens WHERE address = ?;", (token_address,)) as cur:
                row = await cur.fetchone()
                if row:
                    logger.debug("Retrieved trade status for %s", token_address)
                    return dict(row)
                return None
    except Exception as e:
        logger.error(f"Failed to retrieve trade status for {token_address}: {e}\n{traceback.format_exc()}")
        return None


async def upsert_eligible_token(token: Dict) -> None:
    """
    Upsert a single row into eligible_tokens.
    Expects keys: address, name, symbol, volume_24h, liquidity, market_cap, price,
                  price_change_1h, price_change_6h, price_change_24h, score, categories, timestamp
    """
    try:
        categories_json = json.dumps(token.get("categories", []), default=custom_json_encoder)
        async with _connect() as db:
            await db.execute("""
                INSERT INTO eligible_tokens (
                    address, name, symbol, volume_24h, liquidity, market_cap, price,
                    price_change_1h, price_change_6h, price_change_24h, score, categories, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(address) DO UPDATE SET
                    name=excluded.name,
                    symbol=excluded.symbol,
                    volume_24h=excluded.volume_24h,
                    liquidity=excluded.liquidity,
                    market_cap=excluded.market_cap,
                    price=excluded.price,
                    price_change_1h=excluded.price_change_1h,
                    price_change_6h=excluded.price_change_6h,
                    price_change_24h=excluded.price_change_24h,
                    score=excluded.score,
                    categories=excluded.categories,
                    timestamp=excluded.timestamp;
            """, (
                token.get("address"),
                token.get("name", "UNKNOWN"),
                token.get("symbol", "UNKNOWN"),
                float(token.get("volume_24h", 0)),
                float(token.get("liquidity", 0)),
                float(token.get("market_cap", 0)),
                float(token.get("price", 0)),
                float(token.get("price_change_1h", 0)),
                float(token.get("price_change_6h", 0)),
                float(token.get("price_change_24h", 0)),
                float(token.get("score", 0)),
                categories_json,
                int(token.get("timestamp", time.time())),
            ))

            # Write canonical JSON blob and backfill created_at if missing
            json_blob = json.dumps(token, default=custom_json_encoder)
            await db.execute(
                "UPDATE eligible_tokens "
                "SET data = ?, created_at = COALESCE(created_at, strftime('%s','now')) "
                "WHERE address = ?;",
                (json_blob, token.get("address")),
            )

            await db.commit()
        logger.debug("Upserted eligible_token %s (%s)", token.get("symbol", "UNKNOWN"), token.get("address"))
    except Exception as e:
        logger.error("Failed to upsert eligible_token %s: %s\n%s",
                     token.get("address", "UNKNOWN"), e, traceback.format_exc())


async def bulk_upsert_eligible_tokens(tokens: List[Dict]) -> int:
    """
    Bulk upsert into eligible_tokens. Returns number of rows written.
    """
    if not tokens:
        return 0
    try:
        async with _connect() as db:
            rows = []
            now = int(time.time())
            for t in tokens:
                rows.append((
                    t.get("address"),
                    t.get("name", "UNKNOWN"),
                    t.get("symbol", "UNKNOWN"),
                    float(t.get("volume_24h", 0)),
                    float(t.get("liquidity", 0)),
                    float(t.get("market_cap", 0)),
                    float(t.get("price", 0)),
                    float(t.get("price_change_1h", 0)),
                    float(t.get("price_change_6h", 0)),
                    float(t.get("price_change_24h", 0)),
                    float(t.get("score", 0)),
                    json.dumps(t.get("categories", []), default=custom_json_encoder),
                    int(t.get("timestamp", now)),
                ))
            sql = """
                INSERT INTO eligible_tokens (
                    address, name, symbol, volume_24h, liquidity, market_cap, price,
                    price_change_1h, price_change_6h, price_change_24h, score, categories, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(address) DO UPDATE SET
                    name=excluded.name,
                    symbol=excluded.symbol,
                    volume_24h=excluded.volume_24h,
                    liquidity=excluded.liquidity,
                    market_cap=excluded.market_cap,
                    price=excluded.price,
                    price_change_1h=excluded.price_change_1h,
                    price_change_6h=excluded.price_change_6h,
                    price_change_24h=excluded.price_change_24h,
                    score=excluded.score,
                    categories=excluded.categories,
                    timestamp=excluded.timestamp;
            """
            before = db.total_changes
            await db.executemany(sql, rows)

            # also store the raw token dict as JSON for GUI consumption
            json_rows = []
            for t in tokens:
                addr = t.get("address")
                if addr:
                    json_rows.append((json.dumps(t, default=custom_json_encoder), addr))
            if json_rows:
                await db.executemany(
                    "UPDATE eligible_tokens "
                    "SET data = ?, created_at = COALESCE(created_at, strftime('%s','now')) "
                    "WHERE address = ?;",
                    json_rows,
                )

            await db.commit()
            written = db.total_changes - before
        logger.debug("Bulk upserted %d eligible_tokens", written)
        return written
    except Exception as e:
        logger.error("Bulk upsert eligible_tokens failed: %s\n%s", e, traceback.format_exc())
        return 0


async def list_eligible_tokens(
    limit: int = 100,
    min_score: Optional[float] = None,
    newer_than: Optional[int] = None,
    order_by_score_desc: bool = True,
) -> List[Dict]:
    """
    Retrieve eligible tokens with optional filters.

    Also excludes common stables, staking wrappers, and LPs so the GUI "High Cap"
    (and other buckets) aren't polluted with non-tradable wrappers.
    """
    try:
        async with _connect() as db:
            db.row_factory = aiosqlite.Row

            where: List[str] = []
            params: List[Any] = []

            # caller-provided filters
            if isinstance(min_score, (int, float)):
                where.append("score >= ?")
                params.append(float(min_score))
            if isinstance(newer_than, int):
                where.append("timestamp >= ?")
                params.append(int(newer_than))

            # ------------------------------------------------------------------
            # Exclusions: stables, staking wrappers, and LPs
            # ------------------------------------------------------------------
            STABLES = (
                "usdc", "usdt", "usdce", "pyusd", "dai", "tusd",
                "usde", "susd", "lusd", "eurc"
            )
            WRAPPERS_LPS = (
                "jitosol", "bsol", "msol", "lsol", "stsol", "jlp",
                "lp"  # generic but safe to exclude at the symbol level
            )

            # Exclude by symbol (ensure NULL symbols are not excluded accidentally)
            placeholders = ",".join(["?"] * (len(STABLES) + len(WRAPPERS_LPS)))
            where.append(f"COALESCE(LOWER(symbol), '') NOT IN ({placeholders})")
            params.extend([*STABLES, *WRAPPERS_LPS])

            # Exclude by name substrings (staking wrappers only; LPs are handled by symbol/categories)
            where.append(
                "LOWER(name) NOT LIKE ? AND LOWER(name) NOT LIKE ?"
            )
            params.extend(["%staked%", "%staking%"])

            # Exclude by categories JSON text (stored as TEXT)
            where.append(
                "(categories IS NULL OR ("
                "LOWER(categories) NOT LIKE ? AND "
                "LOWER(categories) NOT LIKE ? AND "
                "LOWER(categories) NOT LIKE ?))"
            )
            params.extend(['%"stable"%', '%"lp"%', '%"staking"%'])

            where_sql = f"WHERE {' AND '.join(where)}" if where else ""
            order_sql = (
                "ORDER BY score DESC, timestamp DESC"
                if order_by_score_desc else
                "ORDER BY timestamp DESC"
            )

            sql = f"""
                SELECT address, name, symbol, volume_24h, liquidity, market_cap, price,
                       price_change_1h, price_change_6h, price_change_24h, score, categories, timestamp
                FROM eligible_tokens
                {where_sql}
                {order_sql}
                LIMIT ?;
            """
            params.append(int(max(1, limit)))

            async with db.execute(sql, params) as cur:
                rows = await cur.fetchall()
                res = []
                for r in rows:
                    d = dict(r)
                    # categories were stored as JSON
                    try:
                        d["categories"] = json.loads(d.get("categories") or "[]")
                    except Exception:
                        pass
                    res.append(d)
                return res
    except Exception as e:
        logger.error("list_eligible_tokens failed: %s\n%s", e, traceback.format_exc())
        return []


# ================
# Shortlist bridge
# ================

def _to_eligible_row(t: Dict) -> Optional[Dict]:
    """
    Normalize an in-memory token dict (from Dexscreener/Birdeye/etc.) to the eligible_tokens row schema.
    Accepts synonyms found across the codebase (e.g., v24hUSD, mc/fdv, priceChange1h/6h/24h).
    """
    now = int(time.time())

    # --- Volume normalization ---
    vol = t.get("volume_24h")
    if vol is None:
        vol = t.get("v24hUSD") or t.get("v24h")
        if vol is None:
            vol_field = t.get("volume")
            if isinstance(vol_field, dict):
                vol = vol_field.get("h24")
    if vol is None:
        vol = 0

    # --- Market cap normalization ---
    mc = t.get("market_cap")
    if mc is None:
        mc = t.get("mc", t.get("fdv", 0))

    price = t.get("price", 0)

    # --- Price changes ---
    pc1h = t.get("price_change_1h", t.get("priceChange1h", 0))
    pc6h = t.get("price_change_6h", t.get("priceChange6h", 0))
    pc24h = t.get("price_change_24h", t.get("priceChange24h", 0))

    # --- Timestamp normalization (fix for “Last update age”) ---
    ts = t.get("timestamp")
    if ts is None:
        ts = t.get("pairCreatedAt", t.get("createdAt", now))

    try:
        ts = int(ts)
        # If timestamp looks like milliseconds, convert to seconds
        if ts > 10_000_000_000:
            ts //= 1000
    except Exception:
        ts = now

    # --- Category normalization ---
    cats = t.get("categories", [])
    if isinstance(cats, str):
        cats = [cats]

    # --- Exclude staking wrappers / LP tokens (safer heuristics) ---
    sym = str(t.get("symbol") or "").lower()
    name = str(t.get("name") or "").lower()

    # match 'lp' as a separate word; allow legit words like 'alpha'/'help' to pass
    has_lp_word = bool(re.search(r"\blp\b", name))
    has_staking = ("staked" in name) or ("staking" in name)

    if has_lp_word or has_staking or sym in ("jitosol", "bsol", "jlp", "msol", "lsol", "psol", "stsol"):
        return None  # skip these tokens entirely


    return {
        "address": t.get("address"),
        "name": t.get("name", "UNKNOWN"),
        "symbol": t.get("symbol", "UNKNOWN"),
        "volume_24h": float(vol or 0),
        "liquidity": float(t.get("liquidity", 0)),
        "market_cap": float(mc or 0),
        "price": float(price or 0),
        "price_change_1h": float(pc1h or 0),
        "price_change_6h": float(pc6h or 0),
        "price_change_24h": float(pc24h or 0),
        "score": float(t.get("score", 0) or 0),
        "categories": cats,
        "timestamp": int(ts or now),
    }


async def persist_eligible_shortlist(
    top_tokens: Any,
    prune_hours: int = 168,
) -> int:
    """
    Normalize incoming 'top 5 per category' results and persist them into the existing
    eligible_tokens table, then prune old rows.

    Accepts:
      - dict[str, list[dict]] like {"new": [...], "low_cap": [...], ...}
      - list[dict] (treated as category 'unknown')
      - iterable of (category, token_dict) tuples

    Behavior:
      - Converts each token to our eligible_tokens row schema via _to_eligible_row()
      - Adds the category label (and 'shortlist') into 'categories'
      - Uses bulk_upsert_eligible_tokens(...) to write
      - Calls prune_old_eligible_tokens(prune_hours) if >0
    """
    # Ensure the eligible_tokens table exists even if init_db() wasn't called yet
    try:
        async with _connect() as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS eligible_tokens (
                    address TEXT PRIMARY KEY,
                    name TEXT,
                    symbol TEXT,
                    volume_24h REAL,
                    liquidity REAL,
                    market_cap REAL,
                    price REAL,
                    price_change_1h REAL,
                    price_change_6h REAL,
                    price_change_24h REAL,
                    score REAL,
                    categories TEXT,
                    timestamp INTEGER,
                    data TEXT,
                    created_at INTEGER
                );
            """)
            # Ensure JSON columns present for GUI
            cols = []
            async with db.execute("PRAGMA table_info(eligible_tokens)") as cur:
                async for row in cur:
                    cols.append(row[1])
            if "data" not in cols:
                await db.execute("ALTER TABLE eligible_tokens ADD COLUMN data TEXT;")
            if "created_at" not in cols:
                await db.execute("ALTER TABLE eligible_tokens ADD COLUMN created_at INTEGER;")
            await db.commit()
    except Exception:
        # If this fails, bulk_upsert_eligible_tokens will log a clear error anyway
        pass

    def _ensure_cat(tags: list, cat: str) -> list:
        cat = (cat or "unknown").strip().lower()
        out = [*tags]
        if cat and cat not in out:
            out.append(cat)
        if "shortlist" not in out:
            out.append("shortlist")
        return out

    rows: List[Dict[str, Any]] = []

    # Case 1: dict of categories -> list of tokens
    if isinstance(top_tokens, dict):
        for cat, arr in top_tokens.items():
            if not isinstance(arr, (list, tuple)):
                continue
            for t in arr:
                if not isinstance(t, dict):
                    continue
                row = _to_eligible_row(t)
                if not row:
                    continue
                row["categories"] = _ensure_cat(row.get("categories", []), str(cat))
                rows.append(row)

    # Case 2: list of tokens (no category info)
    elif isinstance(top_tokens, (list, tuple)):
        for item in top_tokens:
            if isinstance(item, dict):
                row = _to_eligible_row(item)
                if not row:
                    continue
                row["categories"] = _ensure_cat(row.get("categories", []), "unknown")
                rows.append(row)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                cat, tok = item[0], item[1]
                if isinstance(tok, dict):
                    row = _to_eligible_row(tok)
                    if not row:
                        continue
                    row["categories"] = _ensure_cat(row.get("categories", []), str(cat))
                    rows.append(row)

    # Case 3: any other iterable of (category, token_dict)
    else:
        try:
            for pair in top_tokens:  # type: ignore
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                cat, tok = pair[0], pair[1]
                if isinstance(tok, dict):
                    row = _to_eligible_row(tok)
                    if not row:
                        continue
                    row["categories"] = _ensure_cat(row.get("categories", []), str(cat))
                    rows.append(row)
        except Exception:
            # Not iterable or malformed – nothing to write
            pass

    written = await bulk_upsert_eligible_tokens(rows) if rows else 0

    # Optional pruning of stale rows
    if prune_hours and prune_hours > 0:
        await prune_old_eligible_tokens(max_age_hours=float(prune_hours))

    return int(written)


# ==============================
# GUI/Runner alignment helpers
# ==============================

async def ensure_eligible_tokens_schema(db: Optional[aiosqlite.Connection] = None) -> None:
    """
    Ensure eligible_tokens exists and has a 'data' column for the GUI to SELECT.
    Can be called with an existing connection, or with no args to manage its own.
    """
    if db is None:
        async with _connect() as _db:
            await ensure_eligible_tokens_schema(_db)
        return

    await db.execute("""
        CREATE TABLE IF NOT EXISTS eligible_tokens (
            address TEXT PRIMARY KEY,
            name TEXT,
            symbol TEXT,
            volume_24h REAL,
            liquidity REAL,
            market_cap REAL,
            price REAL,
            price_change_1h REAL,
            price_change_6h REAL,
            price_change_24h REAL,
            score REAL,
            categories TEXT,
            timestamp INTEGER,
            data TEXT,
            created_at INTEGER
        );
    """)
    cols = []
    async with db.execute("PRAGMA table_info(eligible_tokens)") as cur:
        async for row in cur:
            cols.append(row[1])
    if "data" not in cols:
        await db.execute("ALTER TABLE eligible_tokens ADD COLUMN data TEXT;")
    if "created_at" not in cols:
        await db.execute("ALTER TABLE eligible_tokens ADD COLUMN created_at INTEGER DEFAULT (strftime('%s','now'));")
    await db.commit()


async def ensure_shortlist_tokens_schema(db: Optional[aiosqlite.Connection] = None) -> None:
    """
    Ensure shortlist_tokens (JSON table) exists with (address, data, created_at).
    Can be called with an existing connection, or with no args to manage its own.
    """
    if db is None:
        async with _connect() as _db:
            await ensure_shortlist_tokens_schema(_db)
        return

    await db.execute("""
        CREATE TABLE IF NOT EXISTS shortlist_tokens (
            address TEXT PRIMARY KEY,
            data    TEXT,
            created_at INTEGER DEFAULT (strftime('%s','now'))
        );
    """)
    cols = []
    async with db.execute("PRAGMA table_info(shortlist_tokens)") as cur:
        async for row in cur:
            cols.append(row[1])
    if "data" not in cols:
        await db.execute("ALTER TABLE shortlist_tokens ADD COLUMN data TEXT;")
    if "created_at" not in cols:
        await db.execute("ALTER TABLE shortlist_tokens ADD COLUMN created_at INTEGER;")
    await db.commit()


# Provide an ensure_discovered_tokens_schema helper for parity with other ensure_* helpers
async def ensure_discovered_tokens_schema(db: Optional[aiosqlite.Connection] = None) -> None:
    """
    Ensure discovered_tokens exists (JSON row storage for raw discovery) and has needed columns.
    Can be called with an existing connection, or with no args to manage its own.
    """
    if db is None:
        async with _connect() as _db:
            await ensure_discovered_tokens_schema(_db)
        return

    await db.execute("""
        CREATE TABLE IF NOT EXISTS discovered_tokens (
            address TEXT PRIMARY KEY,
            data TEXT,
            name TEXT,
            symbol TEXT,
            price REAL,
            liquidity REAL,
            market_cap REAL,
            v24hUSD REAL,
            volume_24h REAL,
            dexscreenerUrl TEXT,
            dsPairAddress TEXT,
            links TEXT,
            created_at INTEGER,
            creation_timestamp INTEGER
        );
    """)
    # No backfill needed currently; commit and return
    await db.commit()


async def upsert_token_row(db: aiosqlite.Connection, table: str, token: Dict) -> None:
    """
    Upsert a single JSON token row into 'table' (address, data, created_at).
    Only allows known tables to avoid dynamic-SQL footguns.
    """
    # Allow discovered_tokens in addition to pre-existing shortlist/eligible
    if table not in {"shortlist_tokens", "eligible_tokens", "discovered_tokens"}:
        raise ValueError(f"Invalid table name: {table}")

    addr = token.get("address")
    if not addr:
        return

    payload = json.dumps(
        token,
        default=custom_json_encoder,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    sql = (
        f"INSERT INTO {table} (address, data, created_at) "
        "VALUES (?, ?, strftime('%s','now')) "
        "ON CONFLICT(address) DO UPDATE SET data=excluded.data, created_at=excluded.created_at;"
    )
    await db.execute(sql, (addr, payload))


async def bulk_upsert_tokens(db: aiosqlite.Connection, table: str, tokens: List[Dict]) -> None:
    """
    Bulk upsert JSON tokens into a table with (address, data, created_at).
    """
    for t in tokens or []:
        await upsert_token_row(db, table, t)
    await db.commit()


# ============================
# Atomic shortlist replacement
# ============================

async def clear_eligible_tokens() -> None:
    """Clear all rows from eligible_tokens (leaves table/schema intact)."""
    try:
        async with _connect() as db:
            await ensure_eligible_tokens_schema(db)
            await db.execute("DELETE FROM eligible_tokens;")
            await db.commit()
        logger.info("Cleared eligible_tokens")
    except Exception as e:
        logger.error(f"Failed to clear eligible_tokens: {e}\n{traceback.format_exc()}")


async def save_eligible_tokens(tokens: Sequence[Dict[str, Any]]) -> int:
    """
    Atomically replace the shortlist in eligible_tokens with `tokens`.
    GUI expects rows in this table; this function guarantees a fresh view each cycle.
    """
    try:
        async with _connect() as db:
            await ensure_eligible_tokens_schema(db)
            await db.execute("DELETE FROM eligible_tokens;")

            now = int(time.time())
            core_rows = []
            json_rows = []
            for t in tokens or []:
                addr = t.get("address")
                if not addr:
                    continue
                core_rows.append((
                    addr,
                    t.get("name", "UNKNOWN"),
                    t.get("symbol", "UNKNOWN"),
                    float(t.get("volume_24h", 0)),
                    float(t.get("liquidity", 0)),
                    float(t.get("market_cap", 0)),
                    float(t.get("price", 0)),
                    float(t.get("price_change_1h", 0)),
                    float(t.get("price_change_6h", 0)),
                    float(t.get("price_change_24h", 0)),
                    float(t.get("score", 0)),
                    json.dumps(t.get("categories", []), default=custom_json_encoder),
                    int(t.get("timestamp", now)),
                ))
                json_rows.append((json.dumps(t, default=custom_json_encoder), addr))

            if core_rows:
                await db.executemany("""
                    INSERT INTO eligible_tokens (
                        address, name, symbol, volume_24h, liquidity, market_cap, price,
                        price_change_1h, price_change_6h, price_change_24h, score, categories, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, core_rows)

            if json_rows:
                await db.executemany(
                    "UPDATE eligible_tokens "
                    "SET data = ?, created_at = COALESCE(created_at, strftime('%s','now')) "
                    "WHERE address = ?;",
                    json_rows,
                )

            await db.commit()
            written = len(core_rows)
        logger.info("Persisted %d shortlisted tokens into eligible_tokens", written)
        return written
    except Exception as e:
        logger.error(f"Failed to save eligible_tokens: {e}\n{traceback.format_exc()}")
        return 0
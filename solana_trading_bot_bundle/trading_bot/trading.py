from __future__ import annotations

import os
import time
import sqlite3
import asyncio
import logging
from logging import Logger
import json
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Set, Tuple, Optional, Callable, Awaitable

# --- Solana AsyncClient (runtime-safe, type-safe) ---------------------------
try:
    from solana.rpc.async_api import AsyncClient  # runtime import (may fail)
except Exception:
    AsyncClient = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    # type-only alias so Pylance has a real type at check time
    from solana.rpc.async_api import AsyncClient as AsyncClientType
else:
    class AsyncClientType:  # tiny stub so annotations don't crash at runtime
        ...


def _new_async_client(rpc_url: str):
    """
    Return a new AsyncClient. If the import was deferred/hidden, do a late import
    so runtime still succeeds.
    """
    if AsyncClient is None:  # type: ignore[name-defined]
        from solana.rpc.async_api import AsyncClient as _AC  # type: ignore[reportMissingImports]
        return _AC(rpc_url)
    return AsyncClient(rpc_url)  # type: ignore[call-arg]


# --- DB helpers -------------------------------------------------------------
from .database import persist_eligible_shortlist  # local package import

# Best-effort import for discovered tokens persistence; fall back to inline upsert if missing
try:
    from .database import persist_discovered_tokens as _persist_discovered_tokens  # type: ignore
except Exception:
    _persist_discovered_tokens = None

from . import database as _db  # module alias available for other DB helpers

# module logger (configured later by setup_logging)
logger: Logger = logging.getLogger("TradingBot")
logger.setLevel(logging.INFO)

# Best-effort candlestick patterns classifier import (package -> relative -> disabled)
_classify_patterns = None
try:
    # package path first (works when installed as bundle)
    from solana_trading_bot_bundle.trading_bot.candlestick_patterns import classify_patterns as _classify_patterns  # type: ignore
    logger.debug("Candlestick patterns classifier loaded (packaged path).")
except Exception:
    try:
        # relative local module
        from .candlestick_patterns import classify_patterns as _classify_patterns  # type: ignore
        logger.debug("Candlestick patterns classifier loaded (local path).")
    except Exception:
        logger.debug("Candlestick patterns classifier not available; pattern signals disabled.", exc_info=False)
        _classify_patterns = None

# --- Fetching (prefer packaged bundle; fallback to local; final shims) ------
USING_FETCHING = "missing"
try:
    # Absolute bundle import (preferred; works in both GUI and packaged runs)
    from solana_trading_bot_bundle.trading_bot.fetching import (
        fetch_dexscreener_search,
        fetch_raydium_tokens,
        fetch_birdeye_tokens,
        ensure_rugcheck_status_file,
        fetch_raydium_pool_for_mint,  # optional in some builds, but export if present
    )
    try:
        from solana_trading_bot_bundle.trading_bot.fetching import (
            batch_enrich_tokens_with_signals,  # type: ignore
        )
    except Exception:
        batch_enrich_tokens_with_signals = None  # type: ignore
    USING_FETCHING = "bundle"
except Exception:
    try:
        # Relative local package fallback
        from .fetching import (
            fetch_dexscreener_search,
            fetch_raydium_tokens,
            fetch_birdeye_tokens,
            ensure_rugcheck_status_file,
        )
        try:
            from .fetching import fetch_raydium_pool_for_mint  # type: ignore
        except Exception:
            async def fetch_raydium_pool_for_mint(
                session: "aiohttp.ClientSession",
                mint: str,
                *,
                timeout_s: float = 6.0,
            ) -> dict:
                return {}
        try:
            from .fetching import batch_enrich_tokens_with_signals  # type: ignore
        except Exception:
            batch_enrich_tokens_with_signals = None  # type: ignore
        USING_FETCHING = "local"
    except Exception:
        # Final shims if neither import path exists
        async def fetch_dexscreener_search(*args, **kwargs):  # type: ignore
            return []
        async def fetch_raydium_tokens(*args, **kwargs):  # type: ignore
            return []
        async def fetch_birdeye_tokens(*args, **kwargs):  # type: ignore
            return []
        def ensure_rugcheck_status_file(*args, **kwargs):  # type: ignore
            return None
        async def fetch_raydium_pool_for_mint(
            session: "aiohttp.ClientSession",
            mint: str,
            *,
            timeout_s: float = 6.0,
        ) -> dict:
            return {}
        batch_enrich_tokens_with_signals = None  # type: ignore
        USING_FETCHING = "missing"

logger.debug("USING FETCHING FROM: %s", USING_FETCHING)

# --- Metrics Engine (best-effort import; no-op if unavailable) --------------
METRICS = None
_make_store = None
try:
    # Try package (absolute) path first
    from solana_trading_bot_bundle.trading_bot.metrics_engine import make_store as _make_store  # type: ignore
    METRICS = _make_store(logger=logger, replay_on_init=True)
    logger.debug("Metrics engine loaded (packaged path).")
except Exception:
    try:
        # Try local relative module
        from .metrics_engine import make_store as _make_store  # type: ignore
        METRICS = _make_store(logger=logger, replay_on_init=True)
        logger.debug("Metrics engine loaded (local path).")
    except Exception:
        logger.debug("Metrics engine not available; metrics recording disabled.", exc_info=False)
        METRICS = None

# Export alias for external wiring/tests
make_metrics_store = _make_store if _make_store is not None else None

def _record_metric_fill(
    token_addr: str,
    symbol: str,
    side: str,
    quote: dict | None,
    buy_amount_sol: float | None = None,
    token_price_sol: float | None = None,
    txid: str | None = None,
    simulated: bool = False,
    source: str = "jupiter",
    **extra_metadata
) -> None:
    """
    Safe wrapper to record a standardized fill event to metrics store.
    
    Infers qty and price_usd from Jupiter quote when available, falls back to 
    get_token_price_in_sol. Computes fee_usd if possible. Calls METRICS.record_fill 
    (or METRICS.record if record_fill doesn't exist) with structured payload.
    
    Swallows exceptions and logs them to avoid breaking trading logic.
    
    Args:
        token_addr: Token mint address
        symbol: Token symbol
        side: "BUY" or "SELL"
        quote: Jupiter quote dict (optional)
        buy_amount_sol: Amount in SOL (for buys)
        token_price_sol: Token price in SOL (fallback)
        txid: Transaction ID
        simulated: True for dry-run/simulated trades
        source: Data source (jupiter/birdeye/dexscreener)
        **extra_metadata: Additional fields to include
    """
    if METRICS is None:
        logger.debug("Metrics store not available; skipping fill recording.")
        return
    
    try:
        # Infer quantity and price from quote or fallback
        qty = 0.0
        price_usd = 0.0
        fee_usd = 0.0
        amount_sol = buy_amount_sol or 0.0
        
        if quote and isinstance(quote, dict):
            # Try to extract from Jupiter quote
            in_amount = quote.get("inAmount", 0)
            out_amount = quote.get("outAmount", 0)
            
            # Get decimals for conversion (SOL is 9, most SPL tokens are 6 or 9)
            # For simplicity, assume WSOL decimals = 9
            LAMPORTS_PER_SOL_LOCAL = 1_000_000_000
            
            if side.upper() == "BUY":
                # BUY: inAmount is SOL lamports, outAmount is token amount
                if in_amount:
                    amount_sol = float(in_amount) / LAMPORTS_PER_SOL_LOCAL
                if out_amount:
                    # Assume token has decimals info in quote or fallback to 6
                    token_decimals = 6  # common default
                    if "outputMint" in quote:
                        # Could look up decimals but for now use reasonable default
                        pass
                    qty = float(out_amount) / (10 ** token_decimals)
            else:  # SELL
                # SELL: inAmount is token amount, outAmount is SOL lamports
                if in_amount:
                    token_decimals = 6
                    qty = float(in_amount) / (10 ** token_decimals)
                if out_amount:
                    amount_sol = float(out_amount) / LAMPORTS_PER_SOL_LOCAL
            
            # Try to extract fee
            if "platformFee" in quote:
                fee_lamports = quote.get("platformFee", {}).get("amount", 0)
                if fee_lamports:
                    fee_usd = float(fee_lamports) / LAMPORTS_PER_SOL_LOCAL
            elif "priceImpactPct" in quote:
                # Approximate fee from price impact if available
                impact = float(quote.get("priceImpactPct", 0))
                fee_usd = amount_sol * (abs(impact) / 100.0)
        
        # Fallback: use provided parameters
        if qty == 0.0 and amount_sol > 0 and token_price_sol and token_price_sol > 0:
            qty = amount_sol / token_price_sol
        
        # Convert to USD (approximate: assume SOL price or use 1:1 for now)
        # In production, you'd fetch SOL/USD price
        sol_price_usd = 1.0  # Placeholder - ideally fetch real SOL price
        price_usd = (amount_sol / qty * sol_price_usd) if qty > 0 else 0.0
        
        if price_usd == 0.0 and token_price_sol:
            price_usd = token_price_sol * sol_price_usd
        
        # Record the fill
        if hasattr(METRICS, 'record_fill'):
            METRICS.record_fill(
                token_addr=token_addr,
                side=side.upper(),
                qty=qty,
                price_usd=price_usd,
                fee_usd=fee_usd,
                dry_run=simulated
            )
        elif hasattr(METRICS, 'record'):
            # Fallback to generic record method
            METRICS.record({
                "type": "fill",
                "token": token_addr,
                "symbol": symbol,
                "side": side.upper(),
                "qty": qty,
                "price_usd": price_usd,
                "amount_sol": amount_sol,
                "fee_usd": fee_usd,
                "txid": txid,
                "simulated": simulated,
                "source": source,
                "quote": quote,
                **extra_metadata
            })
        
        logger.debug(
            "Recorded metric fill: %s %s qty=%.6f price_usd=%.6f simulated=%s",
            side, symbol, qty, price_usd, simulated
        )
    except Exception as e:
        logger.debug("Failed to record metric fill for %s: %s", symbol, e, exc_info=False)

# --- Shared helper: derive a unix 'creation' timestamp in *seconds* ----------
def _derive_creation_ts_s(d: Dict[str, Any]) -> int:
    """
    Robustly derive a unix 'creation' timestamp in *seconds* from common fields.
    Handles ms→s, ISO8601 strings, ints/floats, and simple nested dicts.
    Priority order favors already-canonical fields.
    """
    from datetime import datetime, timezone

    def _to_intish(v) -> int:
        if v is None:
            return 0
        if isinstance(v, (int, float)):
            try:
                return int(v)
            except Exception:
                return 0
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return 0
            if s.isdigit():
                try:
                    return int(s)
                except Exception:
                    return 0
            # ISO8601 string?
            try:
                s2 = s.replace("Z", "+00:00")
                dt = datetime.fromisoformat(s2)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return int(dt.timestamp())
            except Exception:
                return 0
        if isinstance(v, dict):
            # common nested carriers like {"ms":...}, {"ts":...}
            for k in ("ms", "sec", "seconds", "ts", "timestamp", "createdAt", "listedAt"):
                if k in v:
                    return _to_intish(v[k])
            return 0
        return 0

    # Try in this order; some sources use milliseconds
    for key in (
        "creation_timestamp",  
        "created_at",          
        "pairCreatedAt",       
        "listedAt",            
        "createdAt",           
        "timestamp",           
    ):
        val = d.get(key)
        ts = _to_intish(val)
        if ts > 2_000_000_000:  # clearly milliseconds → seconds
            ts //= 1000
        if ts > 0:
            return ts
    return 0

# --- Categories backfill (restore/compute) -------------------------------------
def _restore_or_compute_categories(tokens: list[dict]) -> list[dict]:
    """
    Ensure each token has a 'categories' list before selector runs.
    If missing, compute a coarse cap-bucket and attach:
      - 'low_cap'    : mc < 100_000
      - 'mid_cap'    : 100_000 ≤ mc < 500_000
      - 'large_cap'  : mc ≥ 500_000
    If an explicit 'bucket' exists ('low_cap'/'mid_cap'/'large_cap'/'newly_launched'), prefer that.
    """
    out: list[dict] = []
    for t in tokens or []:
        tt = dict(t)
        cats = tt.get("categories")
        # If categories already present and non-empty, keep them
        if isinstance(cats, list) and cats:
            out.append(tt)
            continue

        # Prefer an existing 'bucket' if present
        bucket = (tt.get("bucket") or "").strip().lower()
        if bucket not in ("low_cap", "mid_cap", "large_cap", "newly_launched"):
            # derive from market cap if needed
            try:
                mc = float(tt.get("market_cap", tt.get("mc", 0.0)) or 0.0)
            except Exception:
                mc = 0.0
            if mc < 100_000:
                bucket = "low_cap"
            elif mc < 500_000:
                bucket = "mid_cap"
            else:
                bucket = "large_cap"

        tt["categories"] = [bucket]
        out.append(tt)
    return out

# --- AsyncClient / Commitment (Pylance-safe) --------------------------------
try:
    # Preferred: solana-py Commitment object (what AsyncClient expects)
    from solana.rpc.commitment import Commitment as _RpcCommitment
    COMMIT_CONFIRMED = _RpcCommitment("confirmed")
    COMMIT_PROCESSED = _RpcCommitment("processed")
except Exception:
    COMMIT_CONFIRMED = None
    COMMIT_PROCESSED = None

try:
    # Pylance sometimes flags this even when it exists at runtime; ignore its warning.
    from solana.rpc.async_api import AsyncClient  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover
    AsyncClient = None  # type: ignore[misc, assignment]

# ---- Core constants / paths
from solana_trading_bot_bundle.common.constants import (
    APP_NAME, local_appdata_dir, appdata_dir, logs_dir, data_dir,
    config_path, env_path, db_path, token_cache_path, ensure_app_dirs, prefer_appdata_file
)

# ---- Third-party
import aiohttp
from dotenv import load_dotenv
from solders.commitment_config import CommitmentLevel
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from spl.token.instructions import get_associated_token_address

# ---- Feature gates (env + config aware)
from solana_trading_bot_bundle.common.feature_flags import (
    is_enabled_raydium,
    is_enabled_birdeye,
    resolved_run_flags,  # optional: for truthful log line
)

# Package utilities
from .utils import (
    load_config,
    setup_logging,
    WHITELISTED_TOKENS,
    price_cache,
    token_balance_cache,
    get_buy_amount,
    execute_jupiter_swap,   # stays here
    deduplicate_tokens,
    score_token_dispatch,
    log_scoring_telemetry,
)

from typing import Callable, Awaitable, Union, Any
from inspect import iscoroutinefunction

async def _resolve_buy_amount_sol(
    *,
    get_buy_amount_fn,
    config: dict,
    sol_price: float,
    wallet_balance: float,
    token: dict | None,
) -> float:
    def _clamp(v: Any) -> float:
        try:
            v = float(v)
            # guard NaN/inf/zero/negatives
            if v != v or v in (float("inf"), float("-inf")) or v <= 0:
                return 0.01
            return v
        except Exception:
            return 0.01

    # Call the provided sizing function (async or sync)
    if iscoroutinefunction(get_buy_amount_fn):
        try:
            v = await get_buy_amount_fn(
                token=token,
                wallet_balance=wallet_balance,
                sol_price=sol_price,
                config=config,
            )
        except TypeError:
            # legacy signature support
            v = await get_buy_amount_fn(config, sol_price)
    else:
        try:
            v = get_buy_amount_fn(
                token=token,
                wallet_balance=wallet_balance,
                sol_price=sol_price,
                config=config,
            )
        except TypeError:
            # legacy signature support
            v = get_buy_amount_fn(config, sol_price)

    # --- Normalize return to a scalar SOL amount ---
    if isinstance(v, (tuple, list)) and v:
        # (amount_sol, usd_amount) → amount_sol
        v = v[0]
    elif isinstance(v, dict):
        # tolerate dict-style returns
        v = v.get("amount_sol") or v.get("sol") or next(
            (x for x in v.values() if isinstance(x, (int, float))), 0.0
        )
    elif v is None:
        v = 0.0

    return _clamp(v)

# Market/chain data
from .market_data import (
    validate_token_mint,
    get_token_price_in_sol,
    get_sol_balance,        # comes from market_data
    get_sol_price,
    get_jupiter_quote,
    check_token_account,
    create_token_account,
)
# --- DB schema helper (tolerant import) --------------------------------------
# Try the schema function; if it's not present, fall back to init_db; else no-op.
_ensure_schema_impl: Callable[[], Awaitable[None]] | Callable[[], None] | None
try:
    # Preferred name if present in this codebase
    from .database import ensure_eligible_tokens_schema as _ensure_schema_impl  # type: ignore[attr-defined]
except Exception:
    try:
        # Fall back to init_db (which ensures/patches schema idempotently)
        from .database import init_db as _ensure_schema_impl  # type: ignore[attr-defined]
    except Exception:
        _ensure_schema_impl = None  # last resort: no-op marker
        
# --- Discovered-tokens persistence (tolerant import) -------------------------
try:
    from .database import persist_discovered_tokens  # real implementation
except Exception:
    # No-op fallback so the call site won’t crash if the symbol is missing
    async def persist_discovered_tokens(tokens, prune_hours: int = 24):
        return 0      

async def _ensure_schema_once(logger: logging.Logger) -> None:
    """Run whichever schema helper we found, if any, without blocking the event loop."""
    fn = _ensure_schema_impl
    if fn is None:
        logger.debug("DB schema helper not available; continuing without schema bootstrap.")
        return
    try:
        if asyncio.iscoroutinefunction(fn):
            await fn()                     # async impl
        else:
            await asyncio.to_thread(fn)    # sync impl without blocking loop
    except Exception:
        logger.debug("Schema bootstrap failed (continuing):", exc_info=True)

def _default_db_path() -> Path:
    # Prefer explicit env var, otherwise app local dir on Windows, or ~/.local/share on others
    env = os.getenv("SOLO_BOT_DB")
    if env:
        return Path(env)
    win_base = os.getenv("LOCALAPPDATA")
    if win_base:
        return Path(win_base) / "SOLOTradingBot" / "tokens.sqlite3"
    # Fallback for *nix
    return Path(os.path.expanduser("~")) / ".local" / "share" / "SOLOTradingBot" / "tokens.sqlite3"

def _db_path_from_config(config: dict | None) -> Path:
    try:
        db_cfg = (config or {}).get("database") or {}
        paths_cfg = (config or {}).get("paths") or {}
        raw = (
            db_cfg.get("token_cache_path")
            or paths_cfg.get("token_cache_path")
            or (config or {}).get("token_cache_path")
        )
        if raw:
            return Path(raw)
    except Exception:
        pass
    try:
        return Path(token_cache_path())  # normal path
    except Exception:
        return _default_db_path()        # defensive fallback

    # Fallback to bundled helper (imported above from constants)
    return Path(token_cache_path())  # type: ignore[name-defined]

def _load_persisted_shortlist_from_db(
    config: dict | None = None,
    max_age_s: int = 300,   # 5m TTL so we don't reuse stale shortlists
    min_count: int = 10     # require at least N rows or force live discovery
) -> list[dict]:
    """
    Read the most recent shortlist from the local tokens DB.
    Returns [] if the DB is missing, empty, too old, or too small.

    Why:
      - Prevents the UI getting "stuck" on a tiny/stale persisted shortlist
        that then filters to zero eligible tokens every cycle.
    """
    db_path = _db_path_from_config(config)
    if not db_path.exists():
        logger.info("Shortlist DB not found at %s; skipping DB-based shortlist.", db_path)
        return []

    conn = None
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Discover columns to handle different schema versions gracefully
        cur.execute("PRAGMA table_info(eligible_tokens)")
        cols = {row[1] for row in cur.fetchall()} if conn else set()
        if not cols:
            logger.info("Shortlist table 'eligible_tokens' not present; skipping DB-based shortlist.")
            return []

        # Build SELECT that tolerates schema variants
        order_col = "updated_ts" if "updated_ts" in cols else ("updated_at" if "updated_at" in cols else None)
        shortlist_flag_col = "is_shortlist" if "is_shortlist" in cols else None

        sql = "SELECT * FROM eligible_tokens"
        where = []
        if shortlist_flag_col:
            where.append(f"{shortlist_flag_col}=1")
        if where:
            sql += " WHERE " + " AND ".join(where)
        if order_col:
            sql += f" ORDER BY {order_col} DESC"
        sql += " LIMIT 200"

        cur.execute(sql)
        rows = cur.fetchall()
        if not rows:
            logger.info("Shortlist DB query returned 0 rows; skipping DB-based shortlist.")
            return []

        now = time.time()
        out: list[dict] = []
        seen: set[str] = set()
        youngest_ts: float | None = None

        def _get_ts(row: sqlite3.Row) -> float | None:
            if "updated_ts" in row.keys():
                try:
                    return float(row["updated_ts"])
                except Exception:
                    return None
            if "updated_at" in row.keys():
                try:
                    # tolerate string/iso/epoch
                    v = row["updated_at"]
                    if isinstance(v, (int, float)):
                        return float(v)
                    return float(str(v))
                except Exception:
                    return None
            return None

        for r in rows:
            addr = (
                r["address"] if "address" in r.keys()
                else (r["token_address"] if "token_address" in r.keys() else None)
            )
            if not addr or addr in seen:
                continue
            seen.add(addr)

            ts = _get_ts(r)
            if ts is not None:
                try:
                    age = now - float(ts)
                    if age > max_age_s:
                        # rows are newest-first; once stale, we can stop scanning
                        break
                    youngest_ts = float(ts) if youngest_ts is None else max(youngest_ts, float(ts))
                except Exception:
                    pass

            tok = {
                "address": addr,
                "symbol": (
                    r["symbol"] if "symbol" in r.keys()
                    else (r["baseSymbol"] if "baseSymbol" in r.keys() else "UNKNOWN")
                ),
                "name": (
                    r["name"] if "name" in r.keys()
                    else (r["baseName"] if "baseName" in r.keys() else "UNKNOWN")
                ),
                "categories": ["shortlist"],
            }

            # Copy over common numeric fields when present
            for k in (
                "price","volume24h","v24hUSD","liquidity","market_cap","score","bucket",
                "price_change_5min","price_change_1h","price_change_24h"
            ):
                if k in r.keys():
                    tok[k] = r[k]

            # Optional signal fields if your schema stored them
            for k in ("rsi_5m","rsi_15m","rsi_1h","sma_5m_5","sma_5m_10","sma_5m_20","atr_5m","atr_15m"):
                if k in r.keys():
                    tok[k] = r[k]

            out.append(tok)

        # ----- shortlist-level guards -----
        if not out:
            logger.info("Persisted shortlist empty after TTL filtering; forcing live discovery.")
            return []

        if len(out) < min_count:
            logger.info("Persisted shortlist size %d < min_count %d; forcing live discovery.", len(out), min_count)
            return []

        if youngest_ts is not None:
            age_s = int(now - youngest_ts)
            logger.info(
                "Loaded %d token(s) from persisted shortlist DB %s (youngest age=%ds)",
                len(out), db_path, age_s
            )
        else:
            logger.info("Loaded %d token(s) from persisted shortlist DB %s", len(out), db_path)

        return out

    except Exception as e:
        logger.warning("Failed loading persisted shortlist from DB: %s", e)
        return []
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass

# ---- Raydium shortlist enricher (safe, budgeted per-mint lookups) ----

async def _raydium_enrich_shortlist(session: aiohttp.ClientSession, shortlist: list[dict]) -> list[dict]:
    if os.getenv("ENABLE_RAYDIUM", "0") != "1":
        logger.info("Raydium disabled by ENABLE_RAYDIUM=0 - skipping enrichment.")
        return shortlist

    max_mints      = int(os.getenv("RAY_MINTS_PER_CYCLE", "40"))
    concurrency    = int(os.getenv("RAY_CONCURRENCY", "3"))
    per_timeout    = float(os.getenv("RAY_REQ_TIMEOUT_S", "6"))
    global_budget  = float(os.getenv("RAY_GLOBAL_BUDGET_S", "10"))
    min_liq_usd    = float(os.getenv("RAY_MIN_LIQ_USD", "50000"))
    break_on_empty = os.getenv("RAY_BREAK_ON_EMPTY", "1") == "1"
    empty_abort    = int(os.getenv("RAY_FIRST_EMPTY_ABORT", "2"))

    # first N unique mints
    mints, seen = [], set()
    for t in shortlist:
        a = t.get("address")
        if a and a not in seen:
            seen.add(a)
            mints.append(a)
        if len(mints) >= max_mints:
            break
    if not mints:
        return shortlist

    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    empty_batches = 0

    async def _fetch_one(mint: str) -> Dict:
        try:
            async with sem:
                info = await fetch_raydium_pool_for_mint(session, mint, timeout_s=per_timeout)
                if not info:
                    return {}
                if float(info.get("ray_liquidity_usd", 0) or 0) < min_liq_usd:
                    return {}
                return info
        except Exception:
            return {}

    out_map: Dict[str, Dict] = {}
    chunk = max(1, concurrency * 3)
    for i in range(0, len(mints), chunk):
        if time.time() - t0 > global_budget:
            logger.info("Raydium budget exhausted (%.1fs), stopping.", global_budget)
            break
        batch = mints[i:i+chunk]
        results = await asyncio.gather(*(_fetch_one(m) for m in batch), return_exceptions=True)
        added = 0
        for r in results:
            if isinstance(r, dict) and r.get("address"):
                out_map[r["address"]] = r
                added += 1
        if added == 0:
            empty_batches += 1
            if break_on_empty and empty_batches >= empty_abort:
                logger.info("Raydium enrichment aborting early due to consecutive empties.")
                break

    # merge fields back
    for t in shortlist:
        addr = t.get("address")
        info = out_map.get(addr)
        if info:
            t.setdefault("sources", []).append("raydium")
            t["liquidity"]   = max(float(t.get("liquidity", 0) or 0), float(info.get("ray_liquidity_usd", 0) or 0))
            t["ray_pool_id"] = info.get("ray_pool_id", "")
            t["ray_amm"]     = info.get("ray_amm", "")
            t["ray_version"] = info.get("ray_version", "")
    return shortlist

# --- LAZY import to break circular dependency (works for both package & script) ---
async def _enrich_tokens_with_price_change_lazy(session, tokens, logger, blacklist, failure_count):
    # tokens may be None if upstream short-circuited; normalize to list
    tokens = list(tokens or [])

    # small helpers
    def _f(x, d=0.0):
        try:
            return float(x)
        except Exception:
            return float(d)

    def _i(x, d=0):
        try:
            return int(x)
        except Exception:
            return int(d)

    def _neutral_fallback(src):
        out = []
        for t in (src or []):
            t2 = dict(t)

            # keep any existing core numeric fields; add neutral deltas
            def _fnum(x, dflt=0.0):
                try:
                    return float(x)
                except Exception:
                    return dflt

            def _inum(x, dflt=0):
                try:
                    return int(x)
                except Exception:
                    return dflt

            # price / mc / liq
            t2.setdefault("price", _fnum(t2.get("price", 0.0)))
            # mirror market cap to a canonical key as well as 'mc'
            mc_val = _fnum(t2.get("mc", t2.get("market_cap", 0.0)))
            t2.setdefault("mc", mc_val)
            t2.setdefault("market_cap", mc_val)
            t2.setdefault("liquidity", _fnum(t2.get("liquidity", 0.0)))

            # **IMPORTANT**: canonical 24h volume field for the GUI
            vol = (
                t2.get("volume_24h")
                or t2.get("dex_volume_24h")
                or t2.get("v24hUSD")
                or t2.get("volume24h")
                or 0.0
            )
            t2["volume_24h"] = _fnum(vol, 0.0)

            # misc
            t2.setdefault("holderCount", _inum(t2.get("holderCount", 0)))
            t2.setdefault("pct_change_5m", 0.0)
            t2.setdefault("pct_change_1h", 0.0)
            t2.setdefault("pct_change_24h", 0.0)
            t2.setdefault("dexscreenerUrl", t2.get("dexscreenerUrl", "") or "")
            t2.setdefault("dsPairAddress",  t2.get("dsPairAddress",  "") or "")

            # pretty market-cap string (use canonical market_cap/mc)
            try:
                from .utils import format_market_cap
            except Exception:
                from solana_trading_bot_bundle.trading_bot.utils import format_market_cap
            t2["mcFormatted"] = format_market_cap(_fnum(t2.get("market_cap", t2.get("mc", 0.0))))

            out.append(t2)
        return out

    # -------------------------
    # Lazy resolve the real enrichment function
    # -------------------------
    enrich_fn = None
    try:
        # package path first
        import solana_trading_bot_bundle.trading_bot.fetching as _fetch_mod  # type: ignore
        enrich_fn = getattr(_fetch_mod, "enrich_tokens_with_price_change", None)
    except Exception:
        pass
    if enrich_fn is None:
        try:
            # relative import if running as a package module
            from .fetching import enrich_tokens_with_price_change as _enrich_impl  # type: ignore
            enrich_fn = _enrich_impl
        except Exception:
            pass

    # If we still don't have an impl, keep it neutral
    if not callable(enrich_fn):
        try:
            logger.debug("Price-change enrichment unavailable (no impl); using neutral fallback.")
        except Exception:
            pass
        return _neutral_fallback(tokens)

    # -------------------------
    # Call real enrichment with safety nets
    # -------------------------
    try:
        result = await enrich_fn(session=session, tokens=tokens, logger=logger, blacklist=blacklist, failure_count=failure_count)
    except Exception as e:
        try:
            logger.warning("Price-change enrichment raised %s; using neutral fallback.", e)
        except Exception:
            pass
        return _neutral_fallback(tokens)

    # -------------------------
    # Small sanity checks (optional but helpful)
    # -------------------------

    # 1) If enrichment is empty or not a list → neutral fallback
    if not isinstance(result, list) or not result:
        try:
            logger.warning("Price-change enrichment returned empty; using neutral fallback.")
        except Exception:
            pass
        return _neutral_fallback(tokens)

    # 2) If all deltas are zero and price is ≤0 across the whole batch → neutral fallback
    def _has_any_signal(row: dict) -> bool:
        return (
            _f(row.get("price"), 0.0) > 0.0 or
            _f(row.get("priceChange5m",  row.get("price_change_5m",  0.0))) != 0.0 or
            _f(row.get("priceChange1h",  row.get("price_change_1h",  0.0))) != 0.0 or
            _f(row.get("priceChange24h", row.get("price_change_24h", 0.0))) != 0.0
        )

    if sum(1 for r in result if isinstance(r, dict) and _has_any_signal(r)) == 0:
        try:
            logger.warning("Price-change enrichment looks all-zero; using neutral fallback.")
        except Exception:
            pass
        return _neutral_fallback(tokens)

    # 3) Per-row canonicalization so downstream merge always finds the fields it expects
    fixed: list[dict] = []
    for r in result:
        if not isinstance(r, dict):
            continue
        rr = dict(r)

        # coalesce volume
        vol = rr.get("volume_24h") or rr.get("dex_volume_24h") or rr.get("v24hUSD") or rr.get("volume24h") or 0.0
        rr["volume_24h"] = _f(vol, 0.0)

        # mc <-> market_cap mirrors
        mc = _f(rr.get("mc", rr.get("market_cap", 0.0)), 0.0)
        rr["mc"] = mc if mc > 0 else _f(rr.get("mc", 0.0), 0.0)
        if _f(rr.get("market_cap", 0.0), 0.0) <= 0:
            rr["market_cap"] = rr["mc"]

        # deltas (zeros allowed)
        rr["price"] = _f(rr.get("price", rr.get("lastPrice", rr.get("value", rr.get("last_price", 0.0)))), 0.0)
        rr["priceChange5m"]  = _f(rr.get("priceChange5m",  rr.get("price_change_5m",  0.0)), 0.0)
        rr["priceChange1h"]  = _f(rr.get("priceChange1h",  rr.get("price_change_1h",  0.0)), 0.0)
        rr["priceChange24h"] = _f(rr.get("priceChange24h", rr.get("price_change_24h", 0.0)), 0.0)

        # ensure strings for these
        rr["dexscreenerUrl"] = rr.get("dexscreenerUrl") or ""
        rr["dsPairAddress"]  = rr.get("dsPairAddress")  or ""

        fixed.append(rr)

    return fixed or _neutral_fallback(tokens)

    # Resolve the real enrichment function (package or bundled)
    try:
        from ..price_enrichment_helper_file import enrich_tokens_with_price_change  # type: ignore
    except Exception:
        try:
            from solana_trading_bot_bundle.price_enrichment_helper_file import enrich_tokens_with_price_change  # type: ignore
        except Exception:
            logger.warning("Price-change enrichment module missing; skipping enrichment this cycle.")
            return _neutral_fallback(tokens)

    # Decide whether to disable Birdeye for this call
    import os
    global _BIRDEYE_401_SEEN
    disable_birdeye = False
    try:
        _birdeye_allowed_fn = globals().get("_birdeye_allowed")
        _global_cfg = globals().get("GLOBAL_CONFIG")
        if callable(_birdeye_allowed_fn) and not _birdeye_allowed_fn(_global_cfg):
            disable_birdeye = True
    except Exception:
        pass
    if os.getenv("BIRDEYE_ENABLE", "1") == "0":
        disable_birdeye = True
    if os.getenv("FORCE_DISABLE_BIRDEYE", "0") == "1":
        disable_birdeye = True
    if "_BIRDEYE_401_SEEN" in globals() and _BIRDEYE_401_SEEN:
        disable_birdeye = True
    if not os.getenv("BIRDEYE_API_KEY"):
        disable_birdeye = True
    if "..."
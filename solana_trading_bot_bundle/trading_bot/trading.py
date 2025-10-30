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
    
    NOTE: This is the canonical implementation. A duplicate definition that appeared
    later in the file (around line ~397) has been removed to avoid overshadowing.
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

# -------------------------------------------------------------------------
# Optional runtime MetricsStore wrapper and pattern/bollinger helpers
# Defensive: best-effort imports and lazy init so core logic keeps working
# -------------------------------------------------------------------------
_metrics_store = None  # lazy-initialized MetricsStore wrapper instance

def _metrics_on_fill(
    token_addr: str,
    symbol: str,
    side: str,
    qty: float,
    price_usd: float,
    fee_usd: float,
    txid: str | None,
    simulated: bool,
    **meta
) -> None:
    """
    Best-effort call into a MetricsStore-like object.
    Prefer new _metrics_store API (record_fill), fall back to METRICS alias.
    Swallows any errors.
    """
    try:
        global _metrics_store, METRICS
        ms = _metrics_store or METRICS
        if not ms:
            return
        # Prefer structured record_fill if available
        if hasattr(ms, "record_fill"):
            try:
                ms.record_fill(
                    token_addr=token_addr,
                    side=side.upper(),
                    qty=qty,
                    price_usd=price_usd,
                    fee_usd=fee_usd,
                    dry_run=simulated,
                    txid=txid,
                    symbol=symbol,
                    **meta,
                )
                return
            except Exception:
                pass
        # Fallback to generic 'record' method if present
        if hasattr(ms, "record"):
            try:
                ms.record({
                    "type": "fill",
                    "token": token_addr,
                    "symbol": symbol,
                    "side": side.upper(),
                    "qty": qty,
                    "price_usd": price_usd,
                    "fee_usd": fee_usd,
                    "txid": txid,
                    "simulated": simulated,
                    **meta,
                })
            except Exception:
                pass
    except Exception:
        # Never let metrics fail trading logic
        try:
            logger.debug("Metrics on_fill failed", exc_info=True)
        except Exception:
            pass


def _metrics_snapshot_equity_point(logger: logging.Logger | None = None) -> None:
    """
    Best-effort call to snapshot equity / point-in-time metrics.
    No-op if metrics store lacks such API.
    """
    try:
        global _metrics_store, METRICS
        ms = _metrics_store or METRICS
        if not ms:
            return
        if hasattr(ms, "snapshot_equity_point"):
            try:
                ms.snapshot_equity_point()
                return
            except Exception:
                pass
        # fallback: record a "snapshot" event if generic record exists
        if hasattr(ms, "record"):
            try:
                ms.record({"type": "equity_snapshot", "ts": int(time.time())})
            except Exception:
                pass
    except Exception:
        if logger:
            try:
                logger.debug("Metrics snapshot failed", exc_info=True)
            except Exception:
                pass

# Candlestick / BB helpers (best-effort; inert if numpy/pandas absent)
_patterns_module = None
_pd = None
_np = None
try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None
try:
    import pandas as _pd  # type: ignore
except Exception:
    _pd = None
try:
    # Support a numpy-based patterns module or a local candlestick_patterns module
    import candlestick_patterns as _patterns_module  # type: ignore
except Exception:
    try:
        # package-style import
        from solana_trading_bot_bundle.trading_bot import candlestick_patterns as _patterns_module  # type: ignore
    except Exception:
        _patterns_module = None

def attach_patterns_if_available(token: dict) -> None:
    """
    If token has OHLC arrays or a prebuilt pandas DataFrame, attempt to attach
    pattern booleans (pat_<name>_last) in-place. Defensive: returns silently if
    required libs or data are missing.
    """
    try:
        if not _patterns_module:
            return
        # Prefer a pandas DataFrame if present
        df = None
        if _pd and isinstance(token.get("_ohlc_df"), _pd.DataFrame):
            df = token.get("_ohlc_df")
        else:
            # try to build a DataFrame from _ohlc_close/_ohlc_open/_ohlc_high/_ohlc_low
            if _pd and isinstance(token.get("_ohlc_close"), (list, tuple)):
                try:
                    close = list(token.get("_ohlc_close") or [])
                    open_ = list(token.get("_ohlc_open") or [])
                    high = list(token.get("_ohlc_high") or [])
                    low = list(token.get("_ohlc_low") or [])
                    vol = list(token.get("_ohlc_volume") or token.get("_ohlc_vol") or [])
                    if len(close) and len(close) == len(open_) == len(high) == len(low):
                        df = _pd.DataFrame({
                            "open": open_,
                            "high": high,
                            "low": low,
                            "close": close,
                            "volume": vol if vol else [0] * len(close)
                        })
                except Exception:
                    df = None
        if df is None:
            return
        # call classifier if present
        try:
            pat_hits = _patterns_module.classify_patterns(df)  # defensive: may raise
        except Exception:
            pat_hits = {}
        # attach booleans for last row existence if classifier returns series/dicts
        for name, arr in (pat_hits or {}).items():
            try:
                # arr could be list-like (per-bar); set token field to last value boolean
                if isinstance(arr, (list, tuple)):
                    token[f"pat_{name}_last"] = bool(arr[-1])
                else:
                    token[f"pat_{name}_last"] = bool(arr)
            except Exception:
                token[f"pat_{name}_last"] = False
    except Exception:
        try:
            logger.debug("attach_patterns_if_available failed for %s", token.get("symbol", token.get("address")), exc_info=True)
        except Exception:
            pass

def _attach_bbands_if_available(token: dict, window: int = 20, stdev: float = 2.0) -> None:
    """
    If token contains _ohlc_close (list) or _ohlc_df (pandas), compute Bollinger Bands
    (basis, upper, lower) and attach boolean signals bb_long, bb_short for last row.
    Defensive: no-op if pandas/numpy not available.
    """
    try:
        if not _pd or not _np:
            return
        df = None
        if isinstance(token.get("_ohlc_df"), _pd.DataFrame):
            df = token.get("_ohlc_df")
        elif isinstance(token.get("_ohlc_close"), (list, tuple)):
            close = list(token.get("_ohlc_close") or [])
            if len(close) < window:
                return
            df = _pd.DataFrame({"close": close})
        else:
            return

        if "close" not in df.columns:
            return
        # compute rolling mean and std
        try:
            basis = df["close"].rolling(window=window, min_periods=window).mean()
            dev = df["close"].rolling(window=window, min_periods=window).std()
            upper = basis + dev * stdev
            lower = basis - dev * stdev
            # attach last values if present
            token["bb_basis"] = float(basis.iloc[-1]) if not _np.isnan(basis.iloc[-1]) else None
            token["bb_upper"] = float(upper.iloc[-1]) if not _np.isnan(upper.iloc[-1]) else None
            token["bb_lower"] = float(lower.iloc[-1]) if not _np.isnan(lower.iloc[-1]) else None
            # simple signals: price cross over / under
            close_last = float(df["close"].iloc[-1])
            token["bb_long"] = (token.get("bb_lower") is not None and close_last <= token["bb_lower"])
            token["bb_short"] = (token.get("bb_upper") is not None and close_last >= token["bb_upper"])
        except Exception:
            # safe to ignore any numeric oddities
            return
    except Exception:
        try:
            logger.debug("attach_bbands_if_available failed for %s", token.get("symbol", token.get("address")), exc_info=True)
        except Exception:
            pass

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
        cols = {row["name"] for row in cur.fetchall()}
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

    restore_env = None
    if disable_birdeye:
        restore_env = os.getenv("BIRDEYE_ENABLE")
        os.environ["BIRDEYE_ENABLE"] = "0"

    try:
        # Call the real thing
        result = await enrich_tokens_with_price_change(session, tokens, logger, blacklist, failure_count)

        # Coerce to list and guard against empties
        if not result:
            logger.warning("Enrichment returned no data; using neutral deltas for %d tokens.", len(tokens))
            return _neutral_fallback(tokens)
        if not isinstance(result, list):
            try:
                result = list(result)
            except Exception:
                logger.warning("Enrichment returned non-list; using neutral deltas.")
                return _neutral_fallback(tokens)
        if not result:
            return _neutral_fallback(tokens)

        # Merge enriched rows back onto originals by address + normalize
        src_by_addr = {t.get("address"): t for t in tokens if isinstance(t.get("address"), str) and t.get("address")}
        sanitized: list[dict] = []
        for item in result:
            item = dict(item) if isinstance(item, dict) else {}
            addr = item.get("address")
            base = src_by_addr.get(addr, {}) if addr else {}

            t = dict(base)  # retain original identity fields

            # --- pull fields from enrichment row (Birdeye or blended) ---
            price = item.get("price", item.get("lastPrice"))
            if price is None:
                price = item.get("value") or item.get("last_price")

            mc = item.get("mc", item.get("market_cap"))
            if mc is None:
                mc = item.get("fdv")

            liq        = item.get("liquidity")
            holder_cnt = item.get("holderCount")

            # percent changes (various spellings)
            pc1h  = item.get("priceChange1h",  item.get("price_change_1h"))
            pc6h  = item.get("priceChange6h",  item.get("price_change_6h"))
            pc24h = item.get("priceChange24h", item.get("price_change_24h"))
            pc5m  = item.get("priceChange5m",  item.get("price_change_5m"))

            # ---- canonical 24h volume (NEVER leave missing) ----
            vol_candidates = [
                item.get("volume_24h"),
                item.get("dex_volume_24h"),
                item.get("v24hUSD"),
                item.get("volume24h"),
                base.get("volume_24h"),
                base.get("v24hUSD"),
                base.get("volume24h"),
            ]
            best_vol = next((v for v in vol_candidates if isinstance(v, (int, float)) and float(v) > 0.0), 0.0)
            # keep if enrichment gave 0 but base had a value
            t["volume_24h"] = _f(best_vol, _f(base.get("volume_24h", 0.0)))

            # --- augment, don't overwrite with zeros/None ---
            existing_price = _f(t.get("price", 0.0))
            if isinstance(price, (int, float)) and float(price) > 0:
                t["price"] = float(price)
            else:
                t["price"] = existing_price

            existing_mc = _f(t.get("mc", t.get("market_cap", 0.0)))
            if isinstance(mc, (int, float)) and float(mc) > 0:
                t["mc"] = float(mc)
                t["market_cap"] = float(mc)  # mirror for downstream readers
            else:
                t["mc"] = existing_mc
                if _f(t.get("market_cap", 0.0)) <= 0:
                    t["market_cap"] = existing_mc

            # Liquidity / holders
            t["liquidity"]   = _f(liq, t.get("liquidity", 0.0))
            t["holderCount"] = _i(holder_cnt, t.get("holderCount", 0))

            # Percent changes (0 can be legit)
            t["pct_change_1h"]  = _f(pc1h,  0.0)
            t["pct_change_6h"]  = _f(pc6h,  0.0)
            t["pct_change_24h"] = _f(pc24h, 0.0)
            t["pct_change_5m"]  = _f(pc5m,  0.0)

            # --- timestamp mapping; normalize ms→s and clamp future ---
            ts_candidates = [
                item.get("updated_at"),
                item.get("last_trade_unix"),
                item.get("created_at"),
                item.get("pairCreatedAt"),
                item.get("timestamp"),
            ]
            ts_new = next((x for x in ts_candidates if isinstance(x, (int, float))), None)
            if ts_new is not None:
                ts_new = int(ts_new)
                if ts_new > 10_000_000_000:  # ms→s
                    ts_new //= 1000
                now = int(time.time())
                if ts_new > now + 600:       # clamp accidental future timestamps
                    ts_new = now
                t["timestamp"] = max(int(t.get("timestamp") or 0), ts_new)

            ds_url  = item.get("dexscreenerUrl") or t.get("dexscreenerUrl") or ""
            ds_pair = item.get("dsPairAddress")  or t.get("dsPairAddress")  or ""
            t["dexscreenerUrl"] = ds_url if isinstance(ds_url, str) else ""
            t["dsPairAddress"]  = ds_pair if isinstance(ds_pair, str) else ""

            try:
                from .utils import format_market_cap
            except Exception:
                from solana_trading_bot_bundle.trading_bot.utils import format_market_cap
            t["mcFormatted"] = format_market_cap(_f(t.get("market_cap", t.get("mc", 0.0)), 0.0))

            if "address" not in t and "address" in base:
                t["address"] = base["address"]

            sanitized.append(t)

        return sanitized


    except Exception as e:
        msg = str(e)
        if "401" in msg or "Unauthorized" in msg or "unauthorized" in msg:
            logger.warning("Birdeye 401 encountered during enrichment; disabling for the rest of this run.")
            _BIRDEYE_401_SEEN = True
        else:
            logger.debug("Enrichment raised; continuing without enriched data", exc_info=True)
        return _neutral_fallback(tokens)
    finally:
        if disable_birdeye:
            if restore_env is None:
                os.environ.pop("BIRDEYE_ENABLE", None)
            else:
                os.environ["BIRDEYE_ENABLE"] = restore_env
   

# ---- Merge & dedupe helper (address-canonical) ------------------------------
def _merge_by_address(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge heterogeneous token rows by address, keeping the 'best' row and
    backfilling missing fields from others. 'Best' is chosen by:
      1) higher liquidity, then
      2) higher 24h volume (v24hUSD), then
      3) newer created/listed timestamp.
    """
    def _f(x: Any, d: float = 0.0) -> float:
        try:
            if isinstance(x, dict):  # liquidity may arrive as {"usd": ...}
                return float(x.get("usd", d))
            return float(x)
        except Exception:
            return float(d)

    def _ts(x: Any) -> int:
        # accept seconds or ms
        try:
            t = int(x)
            return t // 1000 if t > 2_000_000_000 else t
        except Exception:
            return 0

    # NEW: derive a canonical creation timestamp (seconds) from common fields
    def _creation_ts(d: dict[str, Any]) -> int:
        for key in (
            "creation_timestamp",  # already-seconds preferred if present
            "pairCreatedAt",       # ms (Dexscreener/Raydium)
            "created_at",          # ms or s (Birdeye/ours)
            "listedAt",            # s
            "createdAt",           # s
            "timestamp",           # s (fallback)
        ):
            v = d.get(key)
            ts = _ts(v)
            if ts > 0:
                return ts
        return 0

    def _score(d: dict[str, Any]) -> tuple[float, float, int]:
        liq = _f(d.get("liquidity"), 0.0)
        vol = _f(d.get("v24hUSD", d.get("volume24h", 0.0)), 0.0)
        ts  = _ts(d.get("pairCreatedAt") or d.get("listedAt") or d.get("createdAt"))
        return (liq, vol, ts)

    merged: dict[str, dict[str, Any]] = {}
    for row in rows or []:
        addr = row.get("address") or (row.get("baseToken") or {}).get("address")
        if not addr:
            continue
        cand: dict[str, Any] = dict(row)

        # Normalize volume/liquidity aliases so downstream coalesce can pick them up
        if "v24hUSD" not in cand and "volume24h" in cand:
            cand["v24hUSD"] = _f(cand.get("volume24h"), 0.0)
        cand["liquidity"] = _f(cand.get("liquidity"), 0.0)

        # NEW: provide dex_ aliases used by coalesce step
        cand.setdefault("dex_volume_24h", cand.get("v24hUSD"))
        if cand.get("dex_liquidity") is None:
            cand["dex_liquidity"] = cand.get("liquidity")
        # if you have FDV/MC from Dexscreener in another field, mirror it here:
        if cand.get("dex_market_cap") is None and cand.get("fdv") is not None:
            cand["dex_market_cap"] = cand.get("fdv")

        # NEW: ensure every candidate carries a canonical creation_timestamp (seconds)
        cand["creation_timestamp"] = _creation_ts(cand)

        best = merged.get(addr)
        if best is None:
            merged[addr] = cand
            continue

        # choose the better of (best, cand) and also backfill missing values
        if _score(cand) > _score(best):
            # prefer cand but keep any extra fields from best that cand lacks
            for k, v in best.items():
                if k not in cand or cand.get(k) in (None, "", 0, 0.0):
                    cand[k] = v
            merged[addr] = cand
        else:
            # keep best, but fill any gaps from cand
            for k, v in cand.items():
                if k not in best or best.get(k) in (None, "", 0, 0.0):
                    best[k] = v
            merged[addr] = best

    # NEW: final safeguard – recompute/normalize creation_timestamp on outputs
    out = list(merged.values())
    for d in out:
        d["creation_timestamp"] = _creation_ts(d)
    return out


# -------------------------------------------------------------------------------
async def _build_live_shortlist(
    *,
    session,
    solana_client,  # kept for signature parity; not used here
    config: dict[str, Any],
    queries: List[str],
    blacklist: Set[str],            # kept for parity; not used by this function
    failure_count: dict[str, int],  # kept for parity; not used by this function
) -> List[dict[str, Any]]:
    """
    Live discovery → merge → dedupe → pre-cut → enrich → diagnostics → coalesce → shortlist → persist.
    Also persists the raw discovered set for the GUI "Discovered Tokens" tab.
    """
    logger.info("LIVE-SHORTLIST(build=v3) starting with queries=%s", queries)

    disc = (config.get("discovery") or {})
    post_cap = int(disc.get("dexscreener_post_cap", 300) or 300)
    pages    = int(disc.get("dexscreener_pages", 1) or 1)

    # 1) FETCH (Dexscreener paged search; keep your per-query loop)
    rows: List[dict[str, Any]] = []
    seen: Set[str] = set()
    for q in queries:
        try:
            chunk = await fetch_dexscreener_search(session, query=q, pages=pages)
        except Exception as e:
            logger.warning("Dexscreener fetch failed for query=%s: %s", q, e)
            chunk = []
        for pair in (chunk or []):
            addr = pair.get("address")
            if not addr or addr in seen:
                continue
            seen.add(addr)
            rows.append(pair)
        # Soft cap early if we’re already above the intended working set
        if len(rows) >= (post_cap * 2):
            break

    logger.info("Pre-dedupe counts: total=%d", len(rows))

    # 2) MERGE/DEDUPE to address-canonical list
    merged: List[dict[str, Any]] = _merge_by_address(rows)
    logger.info("After merge_by_address: %d", len(merged))

    # 3) PERSIST raw discovery for GUI “Discovered Tokens”
    try:
        prune_hours = int((config.get("discovery") or {}).get("discovered_prune_hours", 24))

        # ---- Normalize creation/created_at on the discovered set BEFORE persisting ----
        now_s = int(time.time())
        for _t in merged:
            ts = _derive_creation_ts_s(_t)  # seconds, robust helper
            _t["creation_timestamp"] = ts if ts > 0 else 0

            # created_at is what the DB prune uses; make sure it's present & sane
            try:
                created_at = int(_t.get("created_at") or 0)
            except Exception:
                created_at = 0
            if created_at <= 0:
                # prefer derived creation ts; otherwise fall back to "now"
                created_at = ts or now_s
            _t["created_at"] = created_at

            # keep a generic timestamp aligned as a fallback for any downstream readers
            try:
                stamp = int(_t.get("timestamp") or 0)
            except Exception:
                stamp = 0
            if stamp <= 0:
                _t["timestamp"] = created_at
        # ------------------------------------------------------------------------------

        if callable(_persist_discovered_tokens or None):
            # DB helper if available
            await _persist_discovered_tokens(merged, prune_hours=prune_hours)  # type: ignore[misc]
        else:
            # Inline fallback using existing database helpers
            async with _db.connect_db() as db:
                await _db.ensure_discovered_tokens_schema(db)
                await _db.bulk_upsert_tokens(db, "discovered_tokens", merged)
                if prune_hours and prune_hours > 0:
                    cutoff = now_s - int(prune_hours * 3600)
                    await db.execute("DELETE FROM discovered_tokens WHERE created_at < ?;", (cutoff,))
                    await db.commit()

        logger.info(
            "Persisted %d discovered tokens into discovered_tokens (prune=%dh)",
            len(merged), prune_hours
        )
    except Exception:
        logger.debug("Persisting discovered tokens failed", exc_info=True)

    # 4) OPTIONAL pre-cut (keep top-N by liquidity & v24hUSD before heavier steps)
    pre_cut = sorted(
        merged,
        key=lambda d: (
            float(d.get("liquidity", 0.0)),
            float(d.get("v24hUSD", d.get("volume24h", 0.0))),
        ),
        reverse=True,
    )[:post_cap]

    # 5) CAP workset so enrichment can cover 100% this cycle (align with Birdeye per-cycle cap)
    ENRICH_MAX = int(os.getenv("ENRICH_MAX", "60"))
    workset = pre_cut if len(pre_cut) <= ENRICH_MAX else pre_cut[:ENRICH_MAX]
    if len(pre_cut) > ENRICH_MAX:
        logger.info("Workset capped for enrichment: %d (of %d pre_cut)", ENRICH_MAX, len(pre_cut))

    # 6) ENRICH using your existing lazy enrichment helper
    try:
        if not workset:
            logger.info("Workset is empty after pre-cut; skipping enrichment.")
            enriched: List[dict[str, Any]] = []
        else:
            logger.info("Enriching %d tokens with price change data", len(workset))
            enriched = await _enrich_tokens_with_price_change_lazy(
                session=session,
                tokens=workset,
                logger=logger,
                blacklist=blacklist,
                failure_count=failure_count,
            )
        if not enriched:
            logger.warning("Enrichment returned no data; falling back to pre_cut (%d)", len(pre_cut))
            enriched = pre_cut
            for t in enriched:
                t.setdefault("_enriched", False)
    except Exception:
        logger.warning("Signal enrichment failed; using pre_cut without enrichment", exc_info=True)
        enriched = pre_cut
        for t in enriched:
            t.setdefault("_enriched", False)

    # 7) DIAGNOSTICS: enrichment coverage
    def _count_truthy(xs, key):
        c = 0
        for x in xs:
            v = x.get(key)
            if v is not None and (not isinstance(v, float) or v == v):  # not NaN
                c += 1
        return c

    n = len(enriched)
    have_mc    = _count_truthy(enriched, "market_cap")
    have_liq   = _count_truthy(enriched, "liquidity")
    have_vol24 = _count_truthy(enriched, "volume_24h")
    have_age   = _count_truthy(enriched, "token_age_minutes")
    logger.info("Enrichment coverage: n=%d mc=%d liq=%d vol24=%d age=%d", n, have_mc, have_liq, have_vol24, have_age)

    # 8) COALESCE metrics (fallback to Dexscreener fields when Birdeye missing)
    def _coalesce_metrics(t: dict[str, Any]) -> dict[str, Any]:
        if t.get("market_cap") is None:
            t["market_cap"] = t.get("fdv") or t.get("dex_market_cap")
        if t.get("liquidity") is None:
            t["liquidity"] = t.get("dex_liquidity") or t.get("reserve_usd") or t.get("liquidity")
        if t.get("volume_24h") is None:
            t["volume_24h"] = t.get("dex_volume_24h") or t.get("v24hUSD") or t.get("volume24h")
        return t

    enriched = [_coalesce_metrics(t) for t in enriched]

    # 9) Apply tech tie-breaker (already present in this file)
    try:
        _apply_tech_tiebreaker(enriched, config, logger=logger)
        logger.info("Applied tech-score tie-breaker to shortlist.")
    except Exception:
        logger.debug("tech tie-breaker failed; continuing.", exc_info=True)

    # 10) Fallback shortlist if coverage is weak
    eligible_tokens_preselect: List[dict[str, Any]] | None = None
    coverage_ratio = (have_mc / max(1, n)) if n else 0.0
    if coverage_ratio < 0.2 and n > 0:
        logger.warning("Weak enrichment (mc coverage=%.0f%%). Using fallback shortlist by 24h volume.", coverage_ratio * 100.0)
        MIN_FALLBACK = int(os.getenv("FALLBACK_SHORTLIST_MIN", "5"))
        fallback = sorted(enriched, key=lambda t: float(t.get("volume_24h") or 0.0), reverse=True)[:MIN_FALLBACK]
        for t in fallback:
            t.setdefault("_fallback_eligible", True)
        eligible_tokens_preselect = fallback

    tokens_for_selection = eligible_tokens_preselect if eligible_tokens_preselect is not None else enriched

    # 11) Bucket/shortlist with existing helper
    logger.info("Selecting mid-cap-centric shortlist from %d eligible tokens", len(tokens_for_selection))
    shortlist = await select_top_five_per_category(tokens_for_selection)

        # 11.5) FINAL CANONICALIZE for GUI/DB before persistence & return
    def _canon(t: dict[str, Any]) -> dict[str, Any]:
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

        t = dict(t)  # copy

        # price / mc / liq
        if "mc" in t and ("market_cap" not in t or _f(t.get("market_cap"), 0.0) <= 0):
            t["market_cap"] = _f(t["mc"], 0.0)
        if "market_cap" in t and ("mc" not in t or _f(t.get("mc"), 0.0) <= 0):
            t["mc"] = _f(t["market_cap"], 0.0)

        # coalesce 24h volume (NEVER leave missing for the GUI)
        vol = (
            t.get("volume_24h")
            or t.get("dex_volume_24h")
            or t.get("v24hUSD")
            or t.get("volume24h")
            or 0.0
        )
        t["volume_24h"] = _f(vol, 0.0)

        # deltas (ensure presence; zeros are fine)
        t["pct_change_5m"]  = _f(t.get("pct_change_5m"),  0.0)
        t["pct_change_1h"]  = _f(t.get("pct_change_1h"),  0.0)
        t["pct_change_24h"] = _f(t.get("pct_change_24h"), 0.0)

        # holders/liquidity present and numeric
        t["holderCount"] = _i(t.get("holderCount"), 0)
        t["liquidity"]   = _f(t.get("liquidity"), 0.0)

        # Dexscreener fields as strings
        t["dexscreenerUrl"] = t.get("dexscreenerUrl") or ""
        t["dsPairAddress"]  = t.get("dsPairAddress")  or ""

        # Pretty market cap for UI
        try:
            from .utils import format_market_cap
        except Exception:
            from solana_trading_bot_bundle.trading_bot.utils import format_market_cap
        t["mcFormatted"] = format_market_cap(_f(t.get("market_cap", t.get("mc", 0.0)), 0.0))

        # ---- canonical creation timestamp (seconds) for bucket logic ----
        ts = _derive_creation_ts_s(t)   # uses the shared helper you added earlier
        t["creation_timestamp"] = ts if ts > 0 else 0

        # mirror into created_at if DB/GUI consumers expect it
        if _i(t.get("created_at"), 0) <= 0:
            t["created_at"] = t["creation_timestamp"]

        # mark enrichment default if missing
        t.setdefault("_enriched", True)

        return t

    shortlist = [_canon(t) for t in (shortlist or [])]

    # 12) Persist shortlist for GUI “Shortlist/Eligible” view
    try:
        await persist_eligible_shortlist(shortlist, prune_hours=168)
        logger.info("Persisted %d tokens into eligible_tokens (shortlist view)", len(shortlist))
    except Exception:
        logger.debug("Persisting shortlist failed", exc_info=True)

    return shortlist

   
# --- tiny tie-breaker helper ---
def _tech_score(t: dict) -> float:
    """Lightweight score just for stable ordering; no hard trading logic."""
    def _f(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return float(default)

    # Price changes (assumed as %)
    pc5  = _f(t.get("price_change_5min"))
    pc1h = _f(t.get("price_change_1h"))
    pc24 = _f(t.get("price_change_24h"))

    # RSI normalized around 50 -> [-1, +1]
    rsi5  = (_f(t.get("rsi_5m"),  50.0) - 50.0) / 50.0
    rsi15 = (_f(t.get("rsi_15m"), 50.0) - 50.0) / 50.0
    rsi1h = (_f(t.get("rsi_1h"),  50.0) - 50.0) / 50.0

    # Use your existing weights to avoid behavior change
    score = (
        0.45 * pc5
        + 0.35 * pc1h
        + 0.10 * pc24
        + 0.05 * rsi5
        + 0.03 * rsi15
        + 0.02 * rsi1h
    )

    try:
        s = float(score)
        if s != s or s in (float("inf"), float("-inf")):
            return 0.0
        return s
    except Exception:
        return 0.0


# --- canonical tie-breaker (use once, call everywhere) ---
def _apply_tech_tiebreaker(
    eligible_tokens: list[dict],
    config: dict,
    *,
    logger: logging.Logger
) -> None:
    """
    In-place tie-breaker using your main scoring function when available,
    otherwise fall back to existing 'score' then _tech_score.
    No-ops if signals.ranking_tilt is disabled or list empty.
    """
    _sig_cfg = (config.get("signals") or {})
    if not (eligible_tokens and bool(_sig_cfg.get("ranking_tilt", True))):
        return

    try:
        if callable(score_token_dispatch):
            def _key(_t: dict) -> float:
                # Prefer your real scoring; fall back smoothly if it returns None/NaN
                s = score_token_dispatch(_t, config)
                if s is None:
                    s = _t.get("score", None)
                if s is None:
                    s = _tech_score(_t)
                return float(s)
        else:
            def _key(_t: dict) -> float:
                s = _t.get("score", None)
                if s is None:
                    s = _tech_score(_t)
                return float(s)

        eligible_tokens.sort(key=_key, reverse=True)
        logger.info("Applied tech-score tie-breaker to shortlist.")
    except Exception as _e:
        logger.debug("Tech tie-breaker skipped: %s", _e, exc_info=True)

# ----  _compute_buy_price_sol_per_token helper ----

def _compute_buy_price_sol_per_token(quote: dict[str, Any] | None) -> float | None:
    """
    Compute SOL per token from a Jupiter quote.
    Expects base-unit amounts: inAmount (lamports), outAmount (token base units).
    Returns None if not computable.
    """
    if not isinstance(quote, dict):
        return None
    try:
        in_amount  = float(quote.get("inAmount") or 0.0)   # lamports
        out_amount = float(quote.get("outAmount") or 0.0)  # token base units
        if in_amount > 0.0 and out_amount > 0.0:
            lamports_per_token = in_amount / out_amount
            return lamports_per_token / 1_000_000_000.0  # 1 SOL = 1e9 lamports
    except Exception:
        pass
    return None
   
# Centralized DB helpers
from .database import (
    init_db,
    load_blacklist as db_load_blacklist,
    add_to_blacklist as db_add_to_blacklist,
    clear_expired_blacklist,
    review_blacklist,
    cache_creation_time,
    get_cached_creation_time,
    cache_token_data,
    get_cached_token_data,
    clear_birdeye_cache,
    update_token_record,
    mark_token_sold,
    persist_eligible_shortlist,
    get_open_positions,
    get_token_trade_status,
)

# --- Build+trim+enrich live shortlist once -----------------------------------
async def _fetch_live_shortlist_once(
    *,
    session: aiohttp.ClientSession,
    solana_client,
    config: dict,
    queries: list[str],
    blacklist: set[str] | list[str],
    failure_count: dict,
    logger: logging.Logger
) -> list[dict]:
    """
    Build a live shortlist once, then apply:
      1) select_top_five_per_category
      2) _enrich_shortlist_with_signals (mark _enriched=False if enrichment empty)

    Returns [] if nothing viable could be produced.
    """
    try:
        tokens = await _build_live_shortlist(
            session=session,
            solana_client=solana_client,
            config=config,
            queries=queries,
            blacklist=blacklist,
            failure_count=failure_count,
        )
    except Exception:
        logger.debug("Live shortlist build failed", exc_info=True)
        return []

    if not tokens:
        return []

    # Keep shortlist small for trading path
    try:
        tokens = await select_top_five_per_category(tokens)
    except Exception:
        logger.debug("select_top_five_per_category failed", exc_info=True)

    # Enrich, but keep neutral path if it returns empty
    try:
        enriched = await _enrich_shortlist_with_signals(tokens)
        if enriched:
            tokens = enriched
        else:
            logger.warning(
                "Live enrichment returned empty; keeping un-enriched shortlist."
            )
            for t in tokens:
                t.setdefault("_enriched", False)
    except Exception:
        logger.debug("Signal enrichment on live shortlist failed", exc_info=True)

    # --- FINAL CANONICALIZE (recommended) ---
    try:
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

        def _canon_token(t: dict) -> dict:
            t = dict(t or {})

            # Ensure canonical market cap mirrors
            mc = _f(t.get("market_cap", t.get("mc", 0.0)))
            t["mc"] = mc
            t["market_cap"] = mc

            # Canonical 24h volume for GUI (coalesce common provider keys)
            if t.get("volume_24h") is None:
                t["volume_24h"] = _f(
                    t.get("dex_volume_24h") or t.get("v24hUSD") or t.get("volume24h"),
                    0.0
                )

            # Safe defaults the GUI expects
            t.setdefault("price", _f(t.get("price"), 0.0))
            t.setdefault("liquidity", _f(t.get("liquidity"), 0.0))
            t.setdefault("holderCount", _i(t.get("holderCount"), 0))
            t.setdefault("pct_change_5m", _f(t.get("pct_change_5m"), 0.0))
            t.setdefault("pct_change_1h", _f(t.get("pct_change_1h"), 0.0))
            t.setdefault("pct_change_24h", _f(t.get("pct_change_24h"), 0.0))

            # Pretty market cap
            try:
                from .utils import format_market_cap
            except Exception:
                from solana_trading_bot_bundle.trading_bot.utils import format_market_cap
            t["mcFormatted"] = format_market_cap(mc)

            return t

        tokens = [_canon_token(t) for t in (tokens or [])]
    except Exception:
        logger.debug("final canonicalize failed; continuing.", exc_info=True)

    return tokens or []

# --- Dexscreener per-token cache / rate limit (for creation time & any per-token fallbacks) ---

_DS_TOKEN_CACHE: dict[str, tuple[float, dict]] = {}  # addr -> (ts, data)
_DS_CACHE_TTL = int(os.getenv("DS_TOKEN_CACHE_TTL", "600"))  # seconds
_DS_SEM = asyncio.Semaphore(int(os.getenv("DS_TOKEN_CONCURRENCY", "6")))

# Birdeye per-cycle disable flag when a 401 is encountered
_BIRDEYE_401_SEEN = False

# --- PID / Heartbeat files (module-scope constants) ---
# prefer_appdata_file must already be imported from your constants module
PID_FILE        = prefer_appdata_file("bot.pid")
HEARTBEAT_FILE  = prefer_appdata_file("heartbeat")
STARTED_AT_FILE = prefer_appdata_file("started_at")  # single source of truth

_last_hb = 0  # for throttled heartbeat


def _write_pid_file() -> None:
    try:
        Path(PID_FILE).write_text(str(os.getpid()), encoding="utf-8")
    except Exception:
        logger.debug("PID write failed", exc_info=True)

    # Write started_at once (durable)
    try:
        p = Path(STARTED_AT_FILE)
        if not p.exists():
            p.write_text(str(int(time.time())), encoding="utf-8")
    except Exception:
        logger.debug("started_at write failed", exc_info=True)


def _heartbeat(throttle_s: int = 5) -> None:
    """
    Write two files the GUI can read:
      - STARTED_AT_FILE: written once (first call only)
      - HEARTBEAT_FILE: updated periodically (throttled)
    """
    global _last_hb
    now = int(time.time())

    # Ensure started_at exists (write-once)
    try:
        started_at_path = Path(STARTED_AT_FILE)
        if not started_at_path.exists():
            started_at_path.write_text(str(now), encoding="utf-8")
    except Exception:
        logger.debug("Started-at write failed", exc_info=True)

    # Throttled heartbeat
    if now - _last_hb < throttle_s:
        return
    _last_hb = now
    try:
        Path(HEARTBEAT_FILE).write_text(str(now), encoding="utf-8")
    except Exception:
        logger.debug("Heartbeat write failed", exc_info=True)

# --- Single-instance guard & heartbeat loop ------------------------------------
def _read_int_file(p: str) -> int | None:
    try:
        return int(Path(p).read_text(encoding="utf-8").strip())
    except Exception:
        return None

def _process_alive(pid: int) -> bool:
    try:
        import psutil  # type: ignore
    except Exception:
        # Be conservative if psutil isn't available: treat as alive;
        # staleness will still be checked via heartbeat timestamp.
        return True
    try:
        p = psutil.Process(pid)
        return p.is_running() and (p.status() != "zombie")
    except Exception:
        return False

def acquire_single_instance_or_explain(stale_after_s: int = 180) -> bool:
    """
    If another instance is likely running (fresh heartbeat), refuse to start.
    If it looks stale (no recent heartbeat), take over.
    """
    pid_path = Path(PID_FILE)

    # If no previous PID, we become the owner.
    if not pid_path.exists():
        _write_pid_file()
        _heartbeat(throttle_s=0)
        logger.info("PID guard: acquired lock (first instance).")
        return True

    # There is a previous PID file—check staleness
    prev_pid = _read_int_file(PID_FILE)
    hb_ts    = _read_int_file(HEARTBEAT_FILE) or 0
    now      = int(time.time())
    fresh    = (now - hb_ts) <= max(30, stale_after_s)

    if prev_pid and _process_alive(prev_pid) and fresh:
        logger.warning(
            "Another bot instance appears to be running (pid=%s, heartbeat %ds ago). "
            "Refusing to start a duplicate.",
            prev_pid, max(0, now - hb_ts)
        )
        return False

    # Stale or dead—take over
    ago = (now - hb_ts) if hb_ts else None
    ago_txt = f"{int(ago)}s ago" if ago is not None else "unknown"
    logger.info(
        "PID guard: previous instance looks stale (pid=%s, last heartbeat=%s). Taking ownership.",
        prev_pid, ago_txt
    )

    _write_pid_file()
    _heartbeat(throttle_s=0)
    return True

async def heartbeat_task(stop_event: asyncio.Event, interval: int = 5) -> None:
    """
    Background task that updates heartbeat regularly.
    """
    try:
        while not stop_event.is_set():
            _heartbeat(throttle_s=interval)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                # timeout just means "emit another heartbeat"
                continue
    except asyncio.CancelledError:
        # Optional last stamp
        _heartbeat(throttle_s=0)
        raise
    except Exception:
        logger.debug("heartbeat_task error", exc_info=True)

def release_single_instance() -> None:
    """Remove PID/heartbeat files on clean exit (only if we own the lock)."""
    try:
        existing_pid = _read_int_file(PID_FILE)
        if not existing_pid or existing_pid == os.getpid():
            for p in (PID_FILE, HEARTBEAT_FILE, STARTED_AT_FILE):
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    logger.debug("Cleanup failed for %s", p, exc_info=True)
        else:
            logger.info("Not removing PID file; owned by PID %s.", existing_pid)
    except Exception:
        logger.debug("PID cleanup failed", exc_info=True)


# --- Uniform trade-event logger (single definition) ---
def _log_trade_event(event: str, **kw):
    try:
        msg = " | ".join([event] + [f"{k}={kw[k]}" for k in sorted(kw.keys())])
        logger.info(msg)
    except Exception:
        logger.info("%s %s", event, kw)


def _peek_tokens(tokens: list[dict], n: int = 5) -> str:
    out = []
    for t in tokens[:n]:
        sym = t.get("symbol", "?")
        addr = t.get("address", "?")
        out.append(f"{sym}({addr[:6]}…)")
    return ", ".join(out)

# =========================
# DRY RUN / TEST MODE HELPERS (master switch)
# =========================
def _dry_run_on(cfg: Dict[str, Any] | None = None) -> bool:
    """
    Returns True if DRY RUN mode is active via either the environment or config.

    Sources:
      - ENV: DRY_RUN in {"1","true","yes","on","y"} (case-insensitive)
      - CONFIG: config["trading"]["dry_run"] == True (or config["bot"]["dry_run"] for back-compat)

    Use this at execution choke points (buy/sell executors, ATA creation, send layer).
    """

    env_flag = str(os.getenv("DRY_RUN", "0")).lower() in ("1", "true", "yes", "on", "y")
    if cfg is None:
        return env_flag
    # Prefer 'trading' section; fall back to 'bot' for older configs.
    t = (cfg.get("trading") or {})
    b = (cfg.get("bot") or {})
    return env_flag or bool(t.get("dry_run", b.get("dry_run", False)))

def _dry_run_txid(kind: str = "OP") -> str:
    """Generate a synthetic txid for paper trades (e.g., DRYRUN-BUY / DRYRUN-SELL)."""
    return f"DRYRUN-{str(kind or 'OP').upper()}"

def _skip_onchain_send(cfg: Dict[str, Any] | None = None) -> bool:
    """
    True if we should NOT broadcast:
      - DRY RUN is on
      - or ENV DISABLE_SEND_TX in {"1","true","yes","on","y"}
      - or config.trading.send_transactions is False
    """

    if _dry_run_on(cfg):
        return True
    if str(os.getenv("DISABLE_SEND_TX", "0")).lower() in ("1", "true", "yes", "on", "y"):
        return True
    if cfg:
        t = (cfg.get("trading") or {})
        if t.get("send_transactions") is False:
            return True
    return False

LAMPORTS_PER_SOL = 1_000_000_000  # keep your existing constant if already defined

async def has_min_balance(
    wallet,                 # solders.Keypair
    config: dict,           # full config dict
    session=None,           # unused, kept for call-site compatibility
    min_sol: float | None = None,
) -> bool:
    """
    Returns True if wallet's SOL balance >= threshold.
    - Threshold: min_sol if provided, else config['bot']['required_sol'] (fallback 0.0).
    - Creates a short-lived AsyncClient from config['solana']['rpc_endpoint'].
    """
    try:
        threshold = float(
            (min_sol if min_sol is not None else (config.get("bot", {}) or {}).get("required_sol", 0.0))
        )
    except Exception:
        threshold = 0.0

    rpc_url = (config.get("solana") or {}).get("rpc_endpoint") or "https://api.mainnet-beta.solana.com"
    client = _new_async_client(rpc_url)
    try:
        # Use the pre-defined solana-py Commitment constant
        if COMMIT_PROCESSED is not None:
            resp = await client.get_balance(wallet.pubkey(), commitment=COMMIT_PROCESSED)
        else:
            resp = await client.get_balance(wallet.pubkey())

        lamports = getattr(resp, "value", None)
        if isinstance(lamports, int):
            sol = lamports / LAMPORTS_PER_SOL
            logger.debug("Wallet balance: %.6f SOL (need >= %.6f)", sol, threshold)
            return sol >= threshold

        # Rare defensive fallback if value shape differs
        if isinstance(lamports, dict):
            maybe_lamports = lamports.get("value")
            if isinstance(maybe_lamports, int):
                sol = maybe_lamports / LAMPORTS_PER_SOL
                logger.debug("Wallet balance: %.6f SOL (need >= %.6f)", sol, threshold)
                return sol >= threshold

    except Exception as e:
        logger.warning("has_min_balance: balance check failed: %s", e, exc_info=True)
        return False
    finally:
        try:
            await client.close()
        except Exception:
            pass

    return False

# =========================
# Birdeye master gate (env + optional config)
# =========================
def _birdeye_allowed(config: Dict[str, Any] | None = None) -> bool:
    """
    Central switch for Birdeye usage.
    Priority: ENV overrides > config['sources']['birdeye_enabled'] (defaults True).
    - FORCE_DISABLE_BIRDEYE=1 -> OFF
    - BIRDEYE_ENABLE=0        -> OFF
    - else -> ON unless config disables
    """

    if str(os.getenv("FORCE_DISABLE_BIRDEYE", "0")).strip().lower() in ("1", "true", "yes", "on", "y"):
        return False
    env_enable = str(os.getenv("BIRDEYE_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on", "y")
    if not env_enable:
        return False
    try:
        if config and isinstance(config, dict):
            return bool((config.get("sources") or {}).get("birdeye_enabled", True))
    except Exception:
        pass
    return True

# =========================
# PATCH: Aggressive exits (partial TPs + trailing)
# =========================

def _load_exit_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    b = config.get("bot", {})
    return {
        "stop_loss_x": float(b.get("stop_loss", 0.85)),
        "tp1_x": float(b.get("tp1_x", 1.30)),
        "tp2_x": float(b.get("tp2_x", 2.00)),
        "tp1_pct_to_sell": float(b.get("tp1_pct_to_sell", 0.50)),
        "tp2_pct_of_remaining": float(b.get("tp2_pct_of_remaining", 0.60)),
        "breakeven_plus": float(b.get("breakeven_plus", 0.02)),
        "trail_pct_after_tp1": float(b.get("trail_pct_after_tp1", 0.18)),
        "trail_pct_moonbag": float(b.get("trail_pct_moonbag", 0.30)),
        "no_tp_cutoff_hours": float(b.get("no_tp_cutoff_hours", 6.0)),
        "no_tp_min_profit_x": float(b.get("no_tp_min_profit_x", 1.10)),
        "max_hold_hours": float(b.get("max_hold_hours", 48.0)),
        "sell_slippage_bps": int(b.get("sell_slippage_bps", 150)),
    }

async def _get_trade_state(token_address: str) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "tp1_done": False,
        "tp2_done": False,
        "highest_price": None,
        "breakeven_floor": None,
        "no_tp_deadline": None,
        "hard_exit_deadline": None,
        "trail_pct": None,
    }
    try:
        status = await get_token_trade_status(token_address)
        if status and isinstance(status, dict):
            ts = status.get("trade_state") or {}
            if not ts and "highest_price" in status:
                ts = {k: status.get(k) for k in state.keys() if k in status}
            if isinstance(ts, dict):
                state.update({k: ts.get(k, state[k]) for k in state})
                return state
    except Exception:
        pass

    try:
        cached = await get_cached_token_data(token_address)
        if cached and isinstance(cached, dict):
            ts = cached.get("trade_state") or {}
            if isinstance(ts, dict):
                state.update({k: ts.get(k, state[k]) for k in state})
    except Exception:
        pass
    return state

async def _save_trade_state(token_address: str, trade_state: Dict[str, Any]) -> None:
    try:
        await update_token_record(
            token={"address": token_address, "trade_state": dict(trade_state)},
            buy_price=None, buy_txid=None, buy_time=None, is_trading=True
        )
    except Exception:
        logger.debug("Failed to persist trade_state for %s", token_address, exc_info=True)

def _update_highest(state: Dict[str, Any], current_price: float) -> None:
    hp = state.get("highest_price")
    if hp is None or current_price > float(hp or 0):
        state["highest_price"] = float(current_price)

def _sell_amount_lamports(token_balance_tokens: float, pct: float, default_decimals: int = 9) -> int:
    """
    Compute base-unit amount to sell from a whole-token balance.
    token_balance_tokens: balance in WHOLE tokens (not lamports/base units)
    pct: 0..1 fraction of the balance to sell
    default_decimals: SPL token decimals (9 for SOL-wrapped style tokens unless overridden)
    """
    pct = max(0.0, min(1.0, float(pct)))
    return max(1, int(token_balance_tokens * pct * (10 ** default_decimals)))

# ---- Jupiter helpers -------------------------------------------------
import asyncio
from aiohttp import ClientConnectorError, ClientOSError, ServerDisconnectedError

# Trip this to True for the remainder of the *current cycle* when Jupiter DNS/connector fails.
# Remember to reset `_JUP_OFFLINE = False` at the start of each trading cycle.
_JUP_OFFLINE = False

async def _safe_get_jupiter_quote(
    *,
    input_mint: str,
    output_mint: str,
    amount: int,
    user_pubkey: str,
    session: "aiohttp.ClientSession",
    slippage_bps: int,
) -> tuple[dict | None, str | None]:
    """
    Wrap market_data.get_jupiter_quote. If DNS/network fails once in a cycle,
    mark _JUP_OFFLINE and stop calling Jupiter again until next cycle.
    Returns (quote_dict_or_None, error_or_None).
    """
    global _JUP_OFFLINE
    if _JUP_OFFLINE:
        return None, "JUPITER_OFFLINE"

    try:
        # CALL THE REAL FUNCTION (do not recurse)
        quote, err = await get_jupiter_quote(
            input_mint=input_mint,
            output_mint=output_mint,
            amount=amount,
            user_pubkey=user_pubkey,
            session=session,
            slippage_bps=slippage_bps,
        )
        return quote, err
    except (ClientConnectorError, ClientOSError, ServerDisconnectedError, asyncio.TimeoutError) as e:
        _JUP_OFFLINE = True
        logger.warning("Jupiter looks offline this cycle (%s); skipping further probes.", e)
        return None, "JUPITER_OFFLINE"
    except Exception as e:
        # Other errors shouldn't trip the offline breaker
        return None, str(e)


async def _is_jupiter_tradable(
    input_mint: str,
    output_mint: str,
    lamports: int,
    user_pubkey: str,
    session: "aiohttp.ClientSession",
    slippage_bps: int,
) -> tuple[bool, str | None]:
    """
    Quick probe: ask Jupiter for a tiny quote via the safe wrapper.
    Returns (is_tradable, error_text_if_any).
    """
    try:
        quote, error = await _safe_get_jupiter_quote(
            input_mint=input_mint,
            output_mint=output_mint,
            amount=lamports,
            user_pubkey=user_pubkey,
            session=session,
            slippage_bps=slippage_bps,
        )
        if not quote or error:
            return False, (error or "No quote")
        return True, None
    except Exception as e:
        return False, str(e)

# ---- Correct program IDs (previous short caused ValueError: wrong size) ----
TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
ASSOCIATED_TOKEN_PROGRAM_ID = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
# --------------------------------------------------------------------------------------

def log_error_with_stacktrace(message: str, error: Exception) -> None:
    logger.error(f"{message}: {str(error)}\n{traceback.format_exc()}")

def _clamp_bps(bps: int, lo: int = 1, hi: int = 5000) -> int:
    """Clamp slippage basis points to a sane range to avoid accidental extremes."""
    try:
        return max(lo, min(hi, int(bps)))
    except Exception:
        return 50

# =============================================================================
# Mid-cap focus & stable-like skip (NEW)
# =============================================================================

# Symbols we never want to trade (stablecoins, wrappers, staked-SOL wrappers, etc.)
CORE_STABLE_SYMS = {
    "usdc", "usdt", "wusdc", "wusdt", "usdc.e", "usdt.e", "usdcet", "usdtet",
    "wsol", "jupsol"
}
# Canonical mints to skip outright
CORE_STABLE_ADDRS = {
    # WSOL
    "So11111111111111111111111111111111111111112",
    # USDC
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    # USDT
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
    # Jupiter Staked SOL (as reported)
    "jupSoLaHXQiZZTSfEWMTRRgpnyFm8f6sZdosWBjx93v",
}

def _is_core_stable_like(token: Dict[str, Any]) -> bool:
    sym = (token.get("symbol") or token.get("baseToken", {}).get("symbol") or "").strip().lower()
    addr = (token.get("address") or token.get("token_address") or "").strip()
    name = (token.get("name") or "").strip().lower()
    # Treat anything that looks like *staked sol* as a wrapper we do not trade
    if "staked sol" in name or "stsol" in sym:
        return True
    return (sym in CORE_STABLE_SYMS) or (addr in CORE_STABLE_ADDRS)

# Tunable targets (env) for mid-cap overweight:
MID_CAP_TARGET  = int(os.getenv("MID_CAP_TARGET", "12"))  # how many mids to aim for
LOW_CAP_SPILLOVER  = int(os.getenv("LOW_CAP_SPILLOVER", "3"))
HIGH_CAP_SPILLOVER = int(os.getenv("HIGH_CAP_SPILLOVER", "2"))

# -------------------------- External data fetchers --------------------------

async def fetch_dexscreener_search(
    session: aiohttp.ClientSession,
    query: str = "solana",
    pages: int = 10,
) -> List[Dict]:
    logger.info(f"Fetching Solana pairs from Dexscreener with query: {query}")
    import os, random

    # --- Dexscreener rate-limit controls ---
    _DS_MAX_CONC = int(os.getenv("DEXSCREENER_MAX_CONC", "2"))  # tiny; DS is strict
    _DS_TIMEOUT  = aiohttp.ClientTimeout(total=12)
    _DS_HEADERS  = {"accept": "application/json", "User-Agent": "SOLOTradingBot/1.0"}
    _DS_SEM      = asyncio.Semaphore(_DS_MAX_CONC)

    class _DexRateLimit(Exception):
        def __init__(self, retry_after: float | None = None):
            super().__init__("Dexscreener rate limit")
            self.retry_after = retry_after

    async def fetch_page_once(page: int) -> List[Dict]:
        # NOTE: uses outer-scope `session` and `query`
        url = f"https://api.dexscreener.com/latest/dex/search/?q={query}&page={page}"
        try:
            async with _DS_SEM:
                async with session.get(url, headers=_DS_HEADERS, timeout=_DS_TIMEOUT) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        pairs = data.get("pairs", []) or []
                        return [{
                            "address": pair.get("baseToken", {}).get("address"),
                            "symbol":  pair.get("baseToken", {}).get("symbol"),
                            "name":    pair.get("baseToken", {}).get("name"),
                            "v24hUSD": float((pair.get("volume", {}) or {}).get("h24", 0) or 0),
                            "liquidity": float((pair.get("liquidity", {}) or {}).get("usd", 0) or 0),
                            "mc": float(pair.get("fdv", 0) or 0),
                            "pairCreatedAt": pair.get("pairCreatedAt", 0),
                            "links": (pair.get("info", {}) or {}).get("socials", []),
                        } for pair in pairs
                            if pair.get("chainId") == "solana"
                            and (pair.get("baseToken", {}) or {}).get("address")]
                    if resp.status == 429:
                        ra = resp.headers.get("Retry-After")
                        try:
                            retry_after = float(ra) if ra is not None else None
                        except Exception:
                            retry_after = None
                        raise _DexRateLimit(retry_after)
                    text = (await resp.text())[:200]
                    logger.warning(f"Dexscreener non-200 {resp.status} on page {page}: {text}")
                    return []
        except _DexRateLimit:
            raise
        except Exception as e:
            log_error_with_stacktrace(f"Dexscreener fetch failed on page {page}", e)
            return []

    async def fetch_page(page: int) -> List[Dict]:
        backoff, max_backoff = 1.0, 30.0
        attempts = 0
        while attempts < 5:
            attempts += 1
            try:
                return await fetch_page_once(page)
            except _DexRateLimit as rl:
                sleep_s = rl.retry_after if rl.retry_after is not None else backoff
                logger.warning(f"Dexscreener rate limit hit on page {page}, sleeping {sleep_s:.1f}s")
                await asyncio.sleep(sleep_s + random.uniform(0, 0.5))
                backoff = min(backoff * 2.0, max_backoff)
                continue
            except Exception as e:
                logger.warning(f"Dexscreener fetch exception on page {page} (attempt {attempts}): {e}")
                await asyncio.sleep(backoff + random.uniform(0, 0.5))
                backoff = min(backoff * 2.0, max_backoff)
        logger.error(f"Dexscreener fetch failed on page {page} after {attempts} attempts")
        return []

    # Limit total pages to avoid bursts
    page_limit = int(os.getenv("DEX_PAGE_LIMIT", "6"))
    pages = max(1, min(pages, page_limit))
    page_indices = list(range(1, pages + 1))

    # Fetch with per-host concurrency controlled by _DS_SEM
    results = await asyncio.gather(*(fetch_page(p) for p in page_indices), return_exceptions=True)

    # Flatten + dedupe by token address
    out: List[Dict] = []
    seen: set[str] = set()
    for r in results:
        if isinstance(r, list):
            for t in r:
                addr = t.get("address")
                if addr and addr not in seen:
                    seen.add(addr)
                    out.append(t)
        else:
            logger.warning(f"Dexscreener page task failed: {r}")

    return out    

async def fetch_raydium_tokens(
    session: aiohttp.ClientSession,
    solana_client: Any,
    max_pairs: int = 100
) -> List[Dict]:
    def _truthy_env(name: str, default: str = "0") -> bool:
        v = os.getenv(name, default).strip().lower()
        return v in ("1", "true", "yes", "on", "y")

    # master gates
    if _truthy_env("FORCE_DISABLE_RAYDIUM", "0"):
        logger.info("Raydium disabled by FORCE_DISABLE_RAYDIUM - skipping fetch.")
        return []
    if not _truthy_env("RAYDIUM_ENABLE", "0"):
        logger.info("Raydium disabled by RAYDIUM_ENABLE=0 - skipping fetch.")
        return []
    if max_pairs <= 0:
        logger.info("Raydium disabled by config (max_pairs=0) - skipping fetch.")
        return []

    tokens: List[Dict] = []
    url = "https://api.raydium.io/v2/main/pairs"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "TradingBot/1.0"
    }

    try:
        max_mb = float(os.getenv("RAYDIUM_MAX_DOWNLOAD_MB", "40"))
    except Exception:
        max_mb = 40.0
    max_bytes = int(max_mb * 1024 * 1024)

    try:
        logger.info("Fetching tokens from Raydium")
        timeout = aiohttp.ClientTimeout(total=20)
        async with session.get(url, headers=headers, timeout=timeout) as resp:
            if resp.status == 429:
                logger.warning("Raydium rate limit (429); skipping this cycle.")
                return []
            if resp.status != 200:
                body = (await resp.text())[:512]
                logger.error(f"Raydium API failed: HTTP {resp.status}, Response: {body}")
                return []

            cl = resp.headers.get("Content-Length")
            if cl and cl.isdigit() and int(cl) > max_bytes:
                logger.warning("Raydium response exceeded safe size (Content-Length=%s > %s bytes). Skipping.", cl, max_bytes)
                return []

            buf = bytearray()
            async for chunk in resp.content.iter_chunked(64 * 1024):
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    logger.warning("Raydium streamed body exceeded safe size (%s > %s bytes). Skipping.", len(buf), max_bytes)
                    return []

            try:
                data = json.loads(bytes(buf))
            except MemoryError:
                logger.error("Raydium parse MemoryError on %s bytes.", len(buf))
                return []
            except Exception as e:
                logger.error("Raydium JSON parse error: %s", e)
                return []

            pairs = data if isinstance(data, list) else data.get("data", [])
            if not isinstance(pairs, list):
                logger.error(f"Raydium API returned unexpected structure: {type(data)}")
                return []

            logger.info("Fetched %d pairs from Raydium", len(pairs))
            for pair in pairs[:max_pairs]:
                if not isinstance(pair, dict):
                    continue

                token_address = pair.get("baseMint")
                name_val = pair.get("name", "") or "UNKNOWN"
                symbol = name_val.split("/")[0] if "/" in name_val else name_val
                if not token_address or not symbol:
                    continue

                try:
                    Pubkey.from_string(token_address)
                except Exception:
                    logger.warning("Invalid Raydium token address: %s", token_address)
                    continue

                # tolerant volume/liquidity
                v24 = pair.get("volume24h")
                if isinstance(v24, dict):
                    v24 = v24.get("usd", 0)
                v24 = float(v24 or 0)

                liq = pair.get("liquidity")
                if isinstance(liq, dict):
                    liq = liq.get("usd", 0)
                liq = float(liq or 0)

                market_cap = float(pair.get("fdv", 0) or pair.get("marketCap", 0) or 0)

                try:
                    cached_token = await get_cached_token_data(token_address)
                    if cached_token and cached_token.get("v24hUSD", 0) >= v24:
                        continue
                except Exception:
                    pass

                try:
                    pair_created_at = await get_token_creation_time(token_address, solana_client, config=None, session=session)
                except Exception:
                    pair_created_at = None

                token = {
                    "address": token_address,
                    "symbol": symbol,
                    "name": name_val,
                    "v24hUSD": v24,
                    "liquidity": liq,
                    "mc": market_cap,
                    "pairCreatedAt": int(pair_created_at.timestamp() * 1000) if pair_created_at else 0,
                    "links": (pair.get("extensions") or {}).get("socials", []),
                    # NEW
                    "creation_timestamp": _derive_creation_ts_s({
                        "pairCreatedAt": int(pair_created_at.timestamp() * 1000) if pair_created_at else 0
                    }),
                }
                try:
                    await cache_token_data(token)
                except Exception:
                    pass
                tokens.append(token)

            logger.info("Processed %d valid Raydium tokens", len(tokens))
            return tokens

    except Exception as e:
        log_error_with_stacktrace("Raydium API fetch failed", e)
        return []

async def fetch_birdeye_tokens(
    session: aiohttp.ClientSession,
    solana_client: Any,
    max_tokens: int = 50,
    config: Dict[str, Any] | None = None,
) -> List[Dict]:
    """
    Birdeye fetch with hard OFF support and clean 401 handling.
    NO-OP when:
      - _birdeye_allowed(config) is False, or
      - missing BIRDEYE_API_KEY, or
      - max_tokens <= 0
    """
    global _BIRDEYE_401_SEEN  # <-- added: trip the per-cycle breaker on 401

    if not _birdeye_allowed(config):
        logger.info("Birdeye disabled by config/env - skipping tokenlist fetch.")
        return []

    api_key = (os.getenv("BIRDEYE_API_KEY") or "").strip()
    if not api_key:
        logger.info("Birdeye disabled - missing BIRDEYE_API_KEY.")
        return []
    if max_tokens <= 0:
        logger.info("Birdeye disabled by config (max_tokens=0) - skipping fetch.")
        return []

    logger.info("Fetching tokens from Birdeye")
    try:
        url = (
            "https://public-api.birdeye.so/defi/tokenlist"
            f"?sort_by=v24hUSD&sort_type=desc&offset=0&limit={max_tokens}"
        )
        headers = {
            "X-API-KEY": api_key,
            "Accept": "application/json",
            "User-Agent": "TradingBot/1.0",
        }
        timeout = aiohttp.ClientTimeout(total=15)
        async with session.get(url, headers=headers, timeout=timeout) as resp:
            if resp.status == 429:
                logger.warning("Birdeye tokenlist rate limit (429); skipping this cycle.")
                return []
            if resp.status == 401:
                _BIRDEYE_401_SEEN = True  # <-- added: flip breaker so the rest of the cycle avoids Birdeye
                body = await resp.text()
                logger.error("Birdeye unauthorized (401). Check key/plan. Resp[:256]=%s", body[:256])
                return []
            if resp.status != 200:
                body = await resp.text()
                logger.error("Birdeye API error: HTTP %s, Response: %s", resp.status, body[:512])
                return []

            try:
                data = await resp.json()
            except Exception as e:
                logger.error("Birdeye JSON parse error: %s", e)
                return []

            if not isinstance(data, dict) or not data.get("success"):
                logger.error("Birdeye API error: %s", data.get("message", "Unknown error") if isinstance(data, dict) else "Bad payload")
                return []

            tokens = (data.get("data") or {}).get("tokens", [])
            if not isinstance(tokens, list):
                logger.error("Birdeye API returned unexpected data structure: %s", type(tokens))
                return []

            valid_tokens: List[Dict] = []
            for token in tokens:
                if not isinstance(token, dict):
                    continue
                token_address = token.get("address")
                if not token_address:
                    continue
                if token_address in WHITELISTED_TOKENS:
                    continue
                try:
                    Pubkey.from_string(token_address)
                except Exception:
                    logger.warning("Invalid Birdeye token address: %s", token_address)
                    continue

                cached_token = await get_cached_token_data(token_address)
                if cached_token and cached_token.get("v24hUSD", 0) >= float(token.get("v24hUSD", 0) or 0):
                    continue

                # IMPORTANT: pass config through so the same gate applies to creation time
                pair_created_at = await get_token_creation_time(token_address, solana_client, config=config, session=session)

                valid_token = {
                    "address": token_address,
                    "symbol": token.get("symbol", "UNKNOWN"),
                    "name": token.get("name", "UNKNOWN"),
                    "v24hUSD": float(token.get("v24hUSD", 0) or 0),
                    "liquidity": float(token.get("liquidity", 0) or 0),
                    "mc": float(token.get("mc", 0) or token.get("fdv", 0) or 0),
                    "pairCreatedAt": pair_created_at.timestamp() * 1000 if pair_created_at else 0,
                    "links": (token.get("extensions") or {}).get("socials", []),
                    "created_at": token.get("created_at"),
                    # NEW: canonical seconds field
                    "creation_timestamp": _derive_creation_ts_s({
                        "pairCreatedAt": pair_created_at.timestamp() * 1000 if pair_created_at else 0,
                        "created_at": token.get("created_at"),
                    }),
                }
                await cache_token_data(valid_token)
                valid_tokens.append(valid_token)

            logger.info("Processed %d valid Birdeye tokens", len(valid_tokens))
            return valid_tokens

    except Exception as e:
        log_error_with_stacktrace("Birdeye token fetch failed", e)
        return []

# -------------------------- Creation time helpers --------------------------

def _truthy_env(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on", "y")

async def fetch_birdeye_creation_time(
    token_address: str,
    session: aiohttp.ClientSession,
) -> datetime | None:
    """
    Safe stub: returns None so callers can gracefully fall back.
    Replace with real Birdeye call when ready.
    """
    try:
        logger.debug("Birdeye creation time fetch not implemented; %s", token_address)
    except Exception:
        # if logger isn't available this early, fail silently
        pass
    return None

    """
    Fetch creation time from Birdeye token_security endpoint.

    Obeys OFF switches:
      - FORCE_DISABLE_BIRDEYE=1  -> skip
      - BIRDEYE_ENABLE=0         -> skip
      - missing BIRDEYE_API_KEY  -> skip

    Returns:
      datetime (UTC) if found, else None.
    """
    try:
        if token_address in WHITELISTED_TOKENS:
            return None

        # Respect cache
        cached_time = await get_cached_creation_time(token_address)
        if cached_time:
            return cached_time

        # Feature gating
        if _truthy_env("FORCE_DISABLE_BIRDEYE", "0"):
            logger.info("Birdeye creation-time disabled by FORCE_DISABLE_BIRDEYE - skipping.")
            return None
        if not _truthy_env("BIRDEYE_ENABLE", "0"):
            logger.info("Birdeye creation-time disabled by BIRDEYE_ENABLE=0 - skipping.")
            return None

        api_key = (os.getenv("BIRDEYE_API_KEY") or "").strip()
        if not api_key:
            logger.info("Birdeye creation-time disabled = missing BIRDEYE_API_KEY.")
            return None

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "X-API-KEY": api_key,
            "User-Agent": "TradingBot/1.0",
        }
        url = f"https://public-api.birdeye.so/public/token_security?address={token_address}"
        timeout = aiohttp.ClientTimeout(total=10)

        async with session.get(url, headers=headers, timeout=timeout) as resp:
            if resp.status == 429:
                logger.warning("Birdeye creation time rate limit (429) for %s - skipping this cycle.", token_address)
                return None
            if resp.status == 401:
                body = await resp.text()
                logger.error("Birdeye creation time unauthorized (401). Check key/plan. Resp[:256]=%s", body[:256])
                return None
            if resp.status != 200:
                body = await resp.text()
                logger.warning("Birdeye creation time failed for %s: HTTP %s, Resp[:256]=%s",
                               token_address, resp.status, body[:256])
                return None

            try:
                data = await resp.json()
            except Exception as e:
                logger.warning("Birdeye creation time JSON parse error for %s: %s", token_address, e)
                return None

            creation_ts = (data.get("data") or {}).get("created_timestamp")
            if creation_ts:
                # Birdeye returns ms; guard if seconds
                if creation_ts > 10_000_000_000:  # crude ms vs s check
                    seconds = creation_ts / 1000.0
                else:
                    seconds = float(creation_ts)
                parsed_time = datetime.fromtimestamp(seconds, tz=timezone.utc)
                await cache_creation_time(token_address, parsed_time)
                return parsed_time

            return None

    except Exception as e:
        log_error_with_stacktrace(f"Birdeye creation time error for {token_address}", e)
        return None


async def fetch_dexscreener_creation_time(
    token_address: str,
    session: aiohttp.ClientSession,
) -> datetime | None:
    """
    Fetch creation time from Dexscreener token endpoint, with in-process cache,
    concurrency cap, and jittered backoff on 429s. Uses the earliest pairCreatedAt (ms).
    """
    try:
        import time, random  # local import to avoid module-level churn

        if token_address in WHITELISTED_TOKENS:
            return None

        # 1) DB cache first
        cached_time = await get_cached_creation_time(token_address)
        if cached_time:
            return cached_time

        # 2) In-process response cache + rate limit primitives (lazy init if 1A not added)
        global _DS_TOKEN_CACHE, _DS_CACHE_TTL, _DS_SEM
        if "_DS_TOKEN_CACHE" not in globals():
            _DS_TOKEN_CACHE = {}  # addr -> (ts, data)
        if "_DS_CACHE_TTL" not in globals():
            _DS_CACHE_TTL = int(os.getenv("DS_TOKEN_CACHE_TTL", "600"))  # seconds
        if "_DS_SEM" not in globals():
            _DS_SEM = asyncio.Semaphore(int(os.getenv("DS_TOKEN_CONCURRENCY", "6")))

        now = time.time()
        cached = _DS_TOKEN_CACHE.get(token_address)
        if cached and (now - cached[0]) < _DS_CACHE_TTL:
            data = cached[1]
        else:
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
                "User-Agent": "TradingBot/1.0",
            }
            url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            timeout = aiohttp.ClientTimeout(total=10)

            backoff = 2.0
            while True:
                async with _DS_SEM:
                    resp = await session.get(url, headers=headers, timeout=timeout)

                if resp.status == 429:
                    # polite exponential backoff with a touch of jitter
                    await asyncio.sleep(backoff + random.uniform(0, 0.5))
                    backoff = min(backoff * 1.8, 30.0)
                    continue

                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(
                        "Dexscreener creation time failed for %s: HTTP %s, Resp[:256]=%s",
                        token_address, resp.status, body[:256]
                    )
                    return None

                try:
                    data = await resp.json()  # type: ignore[assignment]
                except Exception as e:
                    logger.warning(
                        "Dexscreener creation time JSON parse error for %s: %s",
                        token_address, e
                    )
                    return None

                _DS_TOKEN_CACHE[token_address] = (now, data)
                break

        pairs = data.get("pairs", []) if isinstance(data, dict) else []
        # Choose the earliest pairCreatedAt if multiple pairs exist
        best_ts_ms: int | None = None
        for p in pairs or []:
            ts = p.get("pairCreatedAt")
            if ts is None:
                continue
            try:
                ts = int(ts)
            except Exception:
                continue
            if best_ts_ms is None or ts < best_ts_ms:
                best_ts_ms = ts

        if best_ts_ms:
            seconds = best_ts_ms / 1000.0
            creation_time = datetime.fromtimestamp(seconds, tz=timezone.utc)
            await cache_creation_time(token_address, creation_time)
            return creation_time

        return None

    except Exception as e:
        log_error_with_stacktrace(
            f"Dexscreener creation time error for {token_address}", e
        )
        return None

async def get_token_creation_time(
    token_address: str,
    solana_client: Any,
    config: dict[str, Any] | None = None,
    session: aiohttp.ClientSession | None = None,
) -> datetime | None:
    """
    Chooses provider order based on Birdeye availability:
      - If _birdeye_allowed(config) and API key present -> try Birdeye first then Dexscreener.
      - Otherwise -> Dexscreener only.
    """
    logger.info("Fetching token creation time for %s", token_address)
    # ... keep your existing implementation ...


    # Decide whether Birdeye should be tried (unified gate + has key)
    birdeye_on = _birdeye_allowed(config) and bool((os.getenv("BIRDEYE_API_KEY") or "").strip())

    _owns = False
    if session is None:
        session = aiohttp.ClientSession()
        _owns = True
    try:
        fetchers = [fetch_dexscreener_creation_time]
        if birdeye_on:
            fetchers = [fetch_birdeye_creation_time, fetch_dexscreener_creation_time]

        for fetch_func in fetchers:
            try:
                creation_time = await fetch_func(token_address, session)
                if creation_time:
                    return creation_time
            except Exception as e:
                logger.warning(
                    "Failed to fetch creation time for %s using %s: %s",
                    token_address,
                    getattr(fetch_func, "__name__", "wrapped_fetcher"),
                    str(e),
                )
        return None
    finally:
        if _owns:
            await session.close()

# -------------------------- Eligibility / scoring --------------------------

async def verify_token_with_rugcheck(token_address: str, token: Dict, session: aiohttp.ClientSession, config: Dict) -> Tuple[float, List[Dict], str]:
    logger.warning(f"RugCheck API not configured for {token.get('symbol', 'UNKNOWN')} ({token_address}). Using max_rugcheck_score from config.")
    max_rugcheck_score = config['discovery'].get('low_cap', {}).get('max_rugcheck_score', 5000)
    if 'newly_launched' in token.get('categories', []):
        max_rugcheck_score = config['discovery'].get('newly_launched', {}).get('max_rugcheck_score', 2000)
    return 0.0, [], "Placeholder: RugCheck not implemented"

async def is_token_eligible(token: Dict, session: aiohttp.ClientSession, config: Dict) -> Tuple[bool, List[str]]:
    logger.info(f"Checking eligibility for {token.get('symbol', 'UNKNOWN')} ({token.get('address', 'UNKNOWN')})")
    token_address = token.get('address')
    if not token_address:
        logger.warning(f"Token missing address: {token}")
        return False, []

    # >>> Hard skip for stables / wrappers (jupSOL, USDC, USDT, WSOL, etc.)
    if _is_core_stable_like(token):
        return False, []

    categories: List[str] = []

    market_cap = float(token.get('mc', 0) or token.get('fdv', 0))
    liquidity = float(token.get('liquidity', 0))
    volume_24h = float(token.get('v24hUSD', 0))
    pair_created_at = token.get('pairCreatedAt', 0)

    if market_cap > 0:
        if market_cap < config['discovery']['low_cap']['max_market_cap']:
            categories.append('low_cap')
        elif market_cap < config['discovery']['mid_cap']['max_market_cap']:
            categories.append('mid_cap')
        else:
            categories.append('large_cap')

    if pair_created_at:
        age_minutes = (time.time() - pair_created_at / 1000) / 60
        if age_minutes <= config['discovery']['newly_launched']['max_token_age_minutes']:
            categories.append('newly_launched')

    if market_cap < 1000:
        return False, categories
    if market_cap > config['discovery']['mid_cap']['max_market_cap'] and 'large_cap' not in categories:
        return False, categories

    if 'low_cap' in categories and liquidity < config['discovery']['low_cap']['liquidity_threshold']:
        return False, categories
    if 'mid_cap' in categories and liquidity < config['discovery']['mid_cap']['liquidity_threshold']:
        return False, categories
    if 'large_cap' in categories and liquidity < config['discovery']['large_cap']['liquidity_threshold']:
        return False, categories
    if 'newly_launched' in categories and liquidity < config['discovery']['newly_launched']['liquidity_threshold']:
        return False, categories

    if 'low_cap' in categories and volume_24h < config['discovery']['low_cap']['volume_threshold']:
        return False, categories
    if 'mid_cap' in categories and volume_24h < config['discovery']['mid_cap']['volume_threshold']:
        return False, categories
    if 'large_cap' in categories and volume_24h < config['discovery']['large_cap']['volume_threshold']:
        return False, categories
    if 'newly_launched' in categories and volume_24h < config['discovery']['newly_launched']['volume_threshold']:
        return False, categories

    price_change_24h = float(token.get('priceChange24h', 0) or token.get('price_change_24h', 0))
    if abs(price_change_24h) > config['discovery']['max_price_change']:
        return False, categories

    # Not implemented holder count yet
    categories.append('pending_holder_count')
    return True, categories

async def select_top_five_per_category(eligible_tokens: List[Dict]) -> List[Dict]:
    # Nicety: log clearly when nothing to buy this cycle
    if not eligible_tokens:
        logger.info("No buy candidates this cycle (eligible_tokens=0)")
        return []

    # Mid-cap-centric shortlist (overweight mid, limited spillover from low/high)
    logger.info(f"Selecting mid-cap-centric shortlist from {len(eligible_tokens)} eligible tokens")

    # Split into groups
    groups = {"low_cap": [], "mid_cap": [], "large_cap": [], "newly_launched": []}
    for t in eligible_tokens:
        cats = (t.get('categories') or [])
        if 'mid_cap' in cats:
            groups['mid_cap'].append(t)
        elif 'low_cap' in cats:
            groups['low_cap'].append(t)
        elif 'large_cap' in cats:
            groups['large_cap'].append(t)
        elif 'newly_launched' in cats:
            groups['newly_launched'].append(t)

    # ---- Fallback if no categories made it through ----
    total_cats = sum(len(v) for v in groups.values())
    if total_cats == 0:
        # Prefer tokens explicitly marked by upstream fallback
        marked = [t for t in eligible_tokens if t.get("_fallback_eligible")]
        pool = marked if marked else eligible_tokens

        def _f(x):
            try:
                return float(x)
            except Exception:
                return 0.0

        pool_sorted = sorted(
            pool,
            key=lambda d: (
                _f(d.get("volume_24h") or d.get("v24hUSD") or d.get("volume24h")),
                _f(d.get("liquidity") or d.get("dex_liquidity")),
            ),
            reverse=True,
        )

        # Size via env (default 5) so you don’t need to change the signature
        try:
            take = int(os.getenv("FALLBACK_SHORTLIST_MIN", "5"))
        except Exception:
            take = 5

        shortlist = pool_sorted[:max(1, take)]
        # Tag for clarity (optional)
        for t in shortlist:
            t.setdefault("categories", [])
            if "fallback" not in t["categories"]:
                t["categories"].append("fallback")

        logger.warning(
            "Selector fallback engaged: no categories present; returning top %d by vol/liquidity (pool=%d).",
            len(shortlist), len(pool_sorted)
        )
        return deduplicate_tokens(shortlist)

    # ---- Normal path: categories present ----
    # Sort each by your existing score (already set with utils.score_token)
    for k in groups:
        groups[k].sort(key=lambda x: x.get('score', 0.0), reverse=True)

    # Take mid caps first
    picks: List[Dict[str, Any]] = []
    picks.extend(groups['mid_cap'][:MID_CAP_TARGET])

    # If we did not hit the mid target, fill with low caps first, then large caps
    need_more = max(0, MID_CAP_TARGET - len([*picks]))
    if need_more > 0:
        picks.extend(groups['low_cap'][:max(LOW_CAP_SPILLOVER, need_more)])
        need_more = max(0, MID_CAP_TARGET + LOW_CAP_SPILLOVER - len([*picks]))
        if need_more > 0:
            picks.extend(groups['large_cap'][:max(HIGH_CAP_SPILLOVER, need_more)])

    # Optionally: we keep newly_launched out of shortlist for now.
    # dedupe just in case
    picks = deduplicate_tokens(picks)

    logger.info(
        "Shortlist (mid-first): mid=%d low=%d high=%d new=%d total=%d",
        len(groups['mid_cap']), len(groups['low_cap']), len(groups['large_cap']),
        len(groups['newly_launched']), len(picks)
    )
    return picks


# ---- Signals enrichment wiring (RSI/MACD/Bollinger/TD9 + patterns) ----

def _resolve_ohlcv_fetcher():
    """
    Try to discover an async OHLCV fetcher in the current runtime.
    Must return: async fn(token_address, *, interval: str, limit: int) -> dict(open, high, low, close, volume)
    """
    try:
        import solana_trading_bot_bundle.trading_bot.market_data as market_data  # type: ignore
        if hasattr(market_data, "fetch_ohlcv"):
            async def _f(addr: str, *, interval: str = "1m", limit: int = 200):
                o = await market_data.fetch_ohlcv(token_address=addr, timeframe=interval, limit=limit)
                return {"open": o["open"], "high": o["high"], "low": o["low"], "close": o["close"], "volume": o["volume"]}
            return _f
        if hasattr(market_data, "get_ohlcv"):
            async def _g(addr: str, *, interval: str = "1m", limit: int = 200):
                o = await market_data.get_ohlcv(token_address=addr, timeframe=interval, limit=limit)
                return {"open": o["open"], "high": o["high"], "low": o["low"], "close": o["close"], "volume": o["volume"]}
            return _g
    except Exception:
        pass
    cand = globals().get("fetch_ohlcv_for_signals") or globals().get("get_ohlcv") or globals().get("fetch_ohlcv")
    if callable(cand):
        return cand
    return None

async def _enrich_shortlist_with_signals(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # If the import failed (e.g., when running as a script), skip enrichment safely
    if batch_enrich_tokens_with_signals is None:
        try:
            logger.debug("Signals enrichment unavailable: fetching module not importable; proceeding without.")
        except Exception:
            pass
        return tokens

    try:
        fetcher = _resolve_ohlcv_fetcher()
        if not fetcher:
            try:
                logger.debug("Signals enrichment skipped: no OHLCV fetcher found.")
            except Exception:
                pass
            return tokens

        # Run the batch enrichment implementation (may be package or local)
        enriched = await batch_enrich_tokens_with_signals(tokens, fetch_ohlcv_func=fetcher, concurrency=8)

        # Safety: if enrichment didn't return a list, fall back to original tokens
        if not isinstance(enriched, list):
            return tokens

        # --- Pattern classification enrichment (best-effort) -----------------
        if _classify_patterns is not None:
            try:
                cfg = load_config()
            except Exception:
                cfg = {}
            signals_enabled = bool((cfg.get("signals") or {}).get("enable", True))
            patterns_enabled = bool((cfg.get("patterns") or {}).get("enable", True))

            if signals_enabled and patterns_enabled:
                patterns_cfg = (cfg.get("patterns") or {})
                pat_list = patterns_cfg.get("list") or [
                    "bullish_engulfing", "bearish_engulfing", "hammer", "shooting_star",
                    "morning_star", "evening_star", "doji", "hanging_man"
                ]

                async def _ensure_ohlcv5(tok: dict):
                    """
                    Return a 5m OHLCV dict with list values for keys:
                    { "open": [...], "high": [...], "low": [...], "close": [...], "volume": [...] }
                    Prefer already-attached OHLCV from the batch enricher; fallback to fetcher(addr, interval="5m").
                    """
                    # Common locations batch enricher may use
                    candidates = []
                    try:
                        candidates.append(tok.get("ohlcv_5m"))
                    except Exception:
                        pass
                    try:
                        candidates.append(tok.get("_ohlcv_5m"))
                    except Exception:
                        pass
                    try:
                        ohlcv_container = tok.get("ohlcv")
                        if isinstance(ohlcv_container, dict):
                            candidates.append(ohlcv_container.get("5m"))
                    except Exception:
                        pass

                    for c in candidates:
                        if isinstance(c, dict) and all(k in c for k in ("open", "high", "low", "close")):
                            return {
                                "open": list(c.get("open") or []),
                                "high": list(c.get("high") or []),
                                "low":  list(c.get("low") or []),
                                "close": list(c.get("close") or []),
                                "volume": list(c.get("volume") or []),
                            }

                    # Fallback: use resolved OHLCV fetcher (if available)
                    addr = tok.get("address") or tok.get("token_address") or tok.get("mint") or (tok.get("baseToken") or {}).get("address")
                    if addr and callable(fetcher):
                        try:
                            o = await fetcher(addr, interval="5m", limit=200)
                            if isinstance(o, dict) and all(k in o for k in ("open", "high", "low", "close")):
                                return {
                                    "open": list(o.get("open") or []),
                                    "high": list(o.get("high") or []),
                                    "low":  list(o.get("low") or []),
                                    "close": list(o.get("close") or []),
                                    "volume": list(o.get("volume") or []),
                                }
                        except Exception:
                            return None
                    return None

                # Replace the per-token loop body with the augmented logic below.
                for tok in (enriched or []):
                    try:
                        ohlcv5 = await _ensure_ohlcv5(tok)
                        if not ohlcv5:
                            continue

                        # --- make OHLC lists available on the token for helper functions ---
                        # These are lightweight list copies (safe even if source is already list-like)
                        try:
                            tok["_ohlc_open"] = list(ohlcv5.get("open") or [])
                            tok["_ohlc_high"] = list(ohlcv5.get("high") or [])
                            tok["_ohlc_low"] = list(ohlcv5.get("low") or [])
                            tok["_ohlc_close"] = list(ohlcv5.get("close") or [])
                            tok["_ohlc_volume"] = list(ohlcv5.get("volume") or [])
                        except Exception:
                            # If any of the above fails, continue -- helpers are defensive
                            pass

                        # --- Existing classifier-based enrichment (preserve current behavior) ---
                        try:
                            pat_hits = {}
                            if _classify_patterns is not None:
                                try:
                                    # Your existing call used ohlcv5 and pat_list
                                    pat_hits = _classify_patterns(ohlcv5, names=pat_list, params=None)
                                except Exception:
                                    pat_hits = {}
                        except Exception:
                            pat_hits = {}

                        def _last_bool(arr):
                            try:
                                return bool(arr[-1])
                            except Exception:
                                return False

                        for name, arr in (pat_hits or {}).items():
                            try:
                                tok[f"pat_{name}_5m"] = _last_bool(arr)
                            except Exception:
                                tok[f"pat_{name}_5m"] = False

                        # Composite momentum-friendly flags (conservative defaults)
                        try:
                            tok["pat_bullish_5m"] = any(tok.get(k, False) for k in (
                                "pat_bullish_engulfing_5m", "pat_morning_star_5m", "pat_hammer_5m"
                            ))
                            tok["pat_bearish_5m"] = any(tok.get(k, False) for k in (
                                "pat_bearish_engulfing_5m", "pat_evening_star_5m", "pat_shooting_star_5m", "pat_hanging_man_5m"
                            ))
                        except Exception:
                            tok["pat_bullish_5m"] = False
                            tok["pat_bearish_5m"] = False

                        # --- Additional, best-effort helpers (safe when libs absent) ---
                        # These helpers will attach pat_<name>_last fields and bb_* fields when possible.
                        try:
                            # attach_patterns_if_available understands _ohlc_close/_ohlc_open etc or _ohlc_df
                            attach_patterns_if_available(tok)
                        except Exception:
                            # never fail the enrichment pipeline
                            logger.debug("attach_patterns_if_available failed for %s", tok.get("symbol", tok.get("address")), exc_info=True)

                        try:
                            # computes bb_basis, bb_upper, bb_lower and bb_long/short booleans
                            _attach_bbands_if_available(tok)
                        except Exception:
                            logger.debug("_attach_bbands_if_available failed for %s", tok.get("symbol", tok.get("address")), exc_info=True)

                        # Optional: if you do NOT want to keep the raw OHLC arrays on the token (to save memory),
                        # uncomment the cleanup below. I leave it commented so external tools/tests that expect
                        # ohlc lists still see them.
                        # try:
                        #     del tok["_ohlc_open"]
                        #     del tok["_ohlc_high"]
                        #     del tok["_ohlc_low"]
                        #     del tok["_ohlc_close"]
                        #     del tok["_ohlc_volume"]
                        # except Exception:
                        #     pass

                    except Exception:
                        # Fail per-token quietly so enrichment pipeline remains robust
                        continue

        return enriched

    except Exception:
        try:
            logger.debug("Signals enrichment failed (falling back): %s", traceback.format_exc())
        except Exception:
            pass
        return tokens

# -------------------------- Wallet / trading actions --------------------------

async def get_token_balance(
    wallet,
    token_mint: str,
    solana_client,
    token_data: dict | None = None,
) -> float:
    """
    Local helper aligned with market_data.get_token_balance signature:
    (wallet, token_mint, solana_client, token_data=None)
    """
    try:
        cache_key = f"balance:{wallet.pubkey()}:{token_mint}"
        cached = token_balance_cache.get(cache_key)
        if cached is not None:
            return float(cached)

        # Associated Token Account (ATA) PDA
        ata = get_associated_token_address(
            owner=wallet.pubkey(),
            mint=Pubkey.from_string(token_mint),
        )

        # Choose coroutine first, then await once
        _balance_coro = (
            solana_client.get_token_account_balance(ata, commitment=COMMIT_CONFIRMED)
            if COMMIT_CONFIRMED is not None
            else solana_client.get_token_account_balance(ata)
        )
        resp = await _balance_coro

        balance = float(resp.value.ui_amount) if getattr(resp, "value", None) else 0.0
        token_balance_cache[cache_key] = balance
        return balance

    except Exception as e:
        sym = (token_data or {}).get("symbol", "UNKNOWN")
        log_error_with_stacktrace(f"Error fetching token balance for {sym} ({token_mint})", e)
        return 0.0

# Cache for token mint decimals
token_decimals_cache: Dict[str, int] = {}

async def get_token_decimals(
    token_mint: str,
    solana_client: Any,
    logger: logging.Logger,
) -> int:
    """
    Fetch SPL token mint decimals (cached).
    Returns 9 if unknown or on RPC limitations.
    """
    if token_mint in token_decimals_cache:
        return token_decimals_cache[token_mint]

    # Fast path: get_token_supply exposes decimals
    try:
        supply = await solana_client.get_token_supply(Pubkey.from_string(token_mint))
        dec = getattr(getattr(supply, "value", None), "decimals", None)
        if isinstance(dec, int) and 0 <= dec <= 18:
            token_decimals_cache[token_mint] = dec
            return dec
    except Exception:
        pass

    # Fallback: try largest accounts -> read one account's balance (includes decimals)
    try:
        la = await solana_client.get_token_largest_accounts(Pubkey.from_string(token_mint))
        value = getattr(la, "value", None)
        if isinstance(value, list) and value:
            v0 = value[0]
            first_address = v0.get("address") if isinstance(v0, dict) else getattr(v0, "address", None)
            if first_address:
                bal = await solana_client.get_token_account_balance(Pubkey.from_string(first_address))
                dec = getattr(getattr(bal, "value", None), "decimals", None)
                if isinstance(dec, int) and 0 <= dec <= 18:
                    token_decimals_cache[token_mint] = dec
                    return dec
    except Exception:
        pass

    logger.debug(f"Using default decimals=9 for {token_mint}")
    token_decimals_cache[token_mint] = 9
    return 9

async def real_on_chain_buy(
    token: dict[str, Any],
    buy_amount: float,
    wallet: Keypair,
    solana_client: Any,
    session: aiohttp.ClientSession,
    blacklist: set[str],
    failure_count: dict[str, int],
    config: dict[str, Any],
    ) -> str | None:
    token_address = token.get("address")
    symbol = token.get("symbol", "UNKNOWN")
    if not token_address:
        logger.warning(f"Token missing address: {token}")
        return None
    try:
        # Safety: never buy stables/stake-wrappers if they somehow slipped through
        if _is_core_stable_like(token):
            logger.info(f"Skipping stable/wrapper {symbol} ({token_address})")
            return None

        if not await validate_token_mint(token_address, solana_client):
            logger.warning(f"Invalid token mint {token_address} for {symbol}")
            if token_address not in WHITELISTED_TOKENS:
                await db_add_to_blacklist(token_address, "Invalid token mint")
                blacklist.add(token_address)
            return None
       
        # FIX: signature (no logger arg)
        token_price = await get_token_price_in_sol(token_address, session, price_cache, token)
        if token_price <= 0:
            logger.warning(f"Invalid token price for {symbol} ({token_address}): {token_price}")
            failure_count[token_address] = failure_count.get(token_address, 0) + 1
            if failure_count[token_address] >= 5 and token_address not in WHITELISTED_TOKENS:
                await db_add_to_blacklist(token_address, "Persistent invalid token price")
                blacklist.add(token_address)
            return None

        sol_amount_lamports = int(buy_amount * 1_000_000_000)
        expected_tokens_est = (buy_amount / token_price) if token_price > 0 else 0.0
        logger.debug(f"Buying ~{expected_tokens_est:.6f} units of {symbol} ({token_address}) for {buy_amount:.6f} SOL")
        # Quick preflight: ensure route is tradable with a dust quote to avoid full-failures
        tradable, t_error = await _is_jupiter_tradable(
            input_mint="So11111111111111111111111111111111111111112",
            output_mint=token_address,
            lamports=max(1000000, int(sol_amount_lamports * 0.01)),
            user_pubkey=str(wallet.pubkey()),
            session=session,
            slippage_bps=_clamp_bps(int(config['bot'].get('buy_slippage_bps', 50)))
        )
        if not tradable:
            logger.error(f"Preflight Jupiter probe indicates {symbol} not tradable: {t_error}")
            failure_count[token_address] = failure_count.get(token_address, 0) + 1
            if failure_count[token_address] >= 5 and token_address not in WHITELISTED_TOKENS:
                await db_add_to_blacklist(token_address, f"Not tradable: {t_error}")
                blacklist.add(token_address)
            return None

        # FIX: signature (no logger), plus dry-run ATA short-circuit (belt & suspenders)
        assoc_addr_exists = await check_token_account(
            str(wallet.pubkey()), token_address, solana_client, blacklist, failure_count, token
        )
                  
        # ---- Associated Token Account (ATA) ensure/create ----
        # Expect: assoc_addr_exists (bool) computed earlier, plus: wallet, token_address, solana_client, token, symbol
        # Uses helpers/flags: _skip_onchain_send(config), _dry_run_on(config), WHITELISTED_TOKENS, db_add_to_blacklist

        if not assoc_addr_exists:
            dry_run = _skip_onchain_send(config)

            if dry_run:
                # In dry-run we expect no ATA until a real buy happens; don't blacklist
                logger.info(
                    "Watch-only: no ATA yet for %s (%s) — expected in dry-run.",
                    symbol, token_address,
                )
                return None
            else:
                # Live mode: try to create the ATA once before blacklisting
                try:
                    created_ata = await create_token_account(
                        wallet=wallet,
                        mint_address=token_address,    # explicit name avoids arg-order confusion
                        solana_client=solana_client,
                        token=token,
                    )
                except TypeError as e:
                    logger.error(
                        "Failed to create token account for %s (%s): %s",
                        symbol, token_address, e,
                    )
                    created_ata = None
                except Exception as e:
                    logger.exception(
                        "Failed to create token account for %s (%s): %s",
                        symbol, token_address, e,
                    )
                    created_ata = None

                if not created_ata:
                    logger.error("ATA creation unavailable for %s (%s); blacklisting.", symbol, token_address)
                    if token_address not in WHITELISTED_TOKENS:
                        await db_add_to_blacklist(token_address, "Failed to create token account")
                        blacklist.add(token_address)
                    return None

                # Live mode: attempt to create the ATA
                try:
                    # Use explicit keywords to avoid signature/ordering mix-ups.
                    assoc_addr = await create_token_account(
                        wallet=wallet,
                        mint_address=token_address,    # explicit name avoids arg-order confusion
                        solana_client=solana_client,
                        token=token,
                    )
                except TypeError as e:
                    # Common pitfall: duplicate 'opts' to AsyncClient.send_transaction(...), etc.
                    logger.error(
                        "Failed to create token account for %s (%s): signature/type error: %s",
                        symbol, token_address, e,
                    )
                    assoc_addr = None
                except Exception as e:
                    logger.exception(
                        "Failed to create token account for %s (%s): %s",
                        symbol, token_address, e,
                    )
                    assoc_addr = None

                if not assoc_addr:
                    # Live mode and creation failed → log + blacklist to avoid repeated retries
                    logger.error("ATA creation unavailable for %s (%s); blacklisting.", symbol, token_address)
                    if token_address not in WHITELISTED_TOKENS:
                        await db_add_to_blacklist(token_address, "Failed to create token account")
                        blacklist.add(token_address)
                    return None   

        # FIX: signature (no logger kwarg)
        quote, error = await _safe_get_jupiter_quote(
            input_mint="So11111111111111111111111111111111111111112",  # SOL
            output_mint=token_address,
            amount=sol_amount_lamports,
            user_pubkey=str(wallet.pubkey()),
            session=session,
            slippage_bps=_clamp_bps(int(config['bot'].get('buy_slippage_bps', 50)))
        )
        if not quote or error:
            logger.error(f"Failed to get Jupiter quote for {symbol} ({token_address}): {error}")
            failure_count[token_address] = failure_count.get(token_address, 0) + 1
            if failure_count[token_address] >= 5 and token_address not in WHITELISTED_TOKENS:
                await db_add_to_blacklist(token_address, f"Jupiter quote error: {error}")
                blacklist.add(token_address)
            return None

        # DRY-RUN / SEND GATE
        if _skip_onchain_send(config):
            txid = _dry_run_txid("buy")
            _log_trade_event(
                "DRYRUN_BUY",
                token=symbol,
                address=token_address,
                lamports=sol_amount_lamports,
                slippage_bps=_clamp_bps(int(config['bot'].get('buy_slippage_bps', 50))),
                txid=txid
            )
            # Record metrics for dry-run buy in real_on_chain_buy
            _record_metric_fill(
                token_addr=token_address,
                symbol=symbol,
                side="BUY",
                quote=quote,
                buy_amount_sol=buy_amount,
                token_price_sol=token_price,
                txid=txid,
                simulated=True,
                source="jupiter",
            )
        else:
            txid = await execute_jupiter_swap(quote, str(wallet.pubkey()), wallet, solana_client)
            if not txid:
                logger.error(f"Failed to execute swap for {symbol} ({token_address})")
                failure_count[token_address] = failure_count.get(token_address, 0) + 1
                if failure_count[token_address] >= 5 and token_address not in WHITELISTED_TOKENS:
                    await db_add_to_blacklist(token_address, "Swap execution failed")
                    blacklist.add(token_address)
                return None
            # Record metrics for live buy in real_on_chain_buy
            _record_metric_fill(
                token_addr=token_address,
                symbol=symbol,
                side="BUY",
                quote=quote,
                buy_amount_sol=buy_amount,
                token_price_sol=token_price,
                txid=txid,
                simulated=False,
                source="jupiter",
            )

        # Persist buy state (+ initialize trade_state)
        exit_cfg = _load_exit_cfg(config)
        now_ts = int(time.time())
        trade_state = {
            "tp1_done": False,
            "tp2_done": False,
            "highest_price": float(token_price),  # start at buy price
            "breakeven_floor": float(token_price) * (1.0 + exit_cfg["breakeven_plus"]),  # activated after TP1
            "no_tp_deadline": now_ts + int(exit_cfg["no_tp_cutoff_hours"] * 3600),
            "hard_exit_deadline": now_ts + int(exit_cfg["max_hold_hours"] * 3600),
            "trail_pct": None,  # set to trail_pct_after_tp1 once TP1 hits
        }

        await update_token_record(
            token={
                "address": token_address,
                "name": token.get("name", "UNKNOWN"),
                "symbol": symbol,
                "volume_24h": float(token.get('v24hUSD', 0)),
                "liquidity": float(token.get('liquidity', 0)),
                "market_cap": float(token.get('mc', 0) or token.get('fdv', 0)),
                "price": float(token.get('price', 0) or 0),
                "price_change_1h": float(token.get('price_change_1h', 0) or token.get('priceChange1h', 0) or 0),
                "price_change_6h": float(token.get('price_change_6h', 0) or 0),
                "price_change_24h": float(token.get('price_change_24h', 0) or token.get('priceChange24h', 0) or 0),
                "categories": token.get('categories', []),
                "timestamp": now_ts,
                "trade_state": trade_state,
            },
            buy_price=float(token_price),
            buy_txid=str(txid),
            buy_time=now_ts,
            is_trading=True,
        )

        logger.info(f"Successfully bought {symbol} ({token_address}) for {buy_amount:.6f} SOL, txid: {txid}")
        return txid

    except Exception as e:
        log_error_with_stacktrace(f"Error executing buy for {symbol} ({token_address})", e)
        failure_count[token_address] = failure_count.get(token_address, 0) + 1
        if failure_count[token_address] >= 5 and token_address not in WHITELISTED_TOKENS:
            await db_add_to_blacklist(token_address, "Repeated buy failures")
            blacklist.add(token_address)
        return None

async def real_on_chain_sell(
    token: dict[str, Any],
    wallet: Keypair,
    solana_client: Any,
    session: aiohttp.ClientSession,
    blacklist: set[str],
    failure_count: dict[str, int],
    config: dict[str, Any],
) -> str | None:
    token_address = token.get("address")
    symbol = token.get("symbol", "UNKNOWN")
    if not token_address:
        logger.warning(f"Token missing address: {token}")
        return None
    try:
        # Basic safety checks
        if not await validate_token_mint(token_address, solana_client):
            logger.warning(f"Invalid token mint {token_address} for {symbol}")
            if token_address not in WHITELISTED_TOKENS:
                await db_add_to_blacklist(token_address, "Invalid token mint")
                blacklist.add(token_address)
            return None
        
        assoc_addr_exists = await check_token_account(
            str(wallet.pubkey()), token_address, solana_client, blacklist, failure_count, token
        )

        # Detect dry-run / simulate / send-tx disabled
        dry_run = _skip_onchain_send(config)

        if not assoc_addr_exists:
            if dry_run:
                # In dry-run we expect no ATA until a real buy happens; don't blacklist
                logger.info("Watch-only: no ATA yet for %s (%s) — expected in dry-run.", symbol, token_address)
                return None
            else:
                logger.error("No token account exists for %s (%s)", symbol, token_address)
                if token_address not in WHITELISTED_TOKENS:
                    await db_add_to_blacklist(token_address, "No token account")
                    blacklist.add(token_address)
                return None

        token_balance = await get_token_balance(wallet, token_address, solana_client, token)
        if token_balance <= 0:
            logger.debug(f"No tokens available to sell for {symbol} ({token_address})")
            return None

        status = await get_token_trade_status(token_address)
        if not status or status.get("buy_price") is None or status.get("buy_time") is None:
            if dry_run:
                # In dry-run there may never be a recorded buy; don't blacklist
                logger.info("No valid buy record for %s (%s) — expected in dry-run.", symbol, token_address)
                return None
            else:
                logger.warning(f"No valid buy record for {symbol} ({token_address})")
                if token_address not in WHITELISTED_TOKENS:
                    await db_add_to_blacklist(token_address, "No buy record")
                    blacklist.add(token_address)
                return None

        exit_cfg = _load_exit_cfg(config)

        buy_price = float(status.get("buy_price") or 0)
        buy_time = int(status.get("buy_time") or 0)
        now_ts = int(time.time())

        # FIX: signature (no logger)
        current_price = await get_token_price_in_sol(token_address, session, price_cache, token)
        if current_price <= 0 or buy_price <= 0:
            logger.warning(
                f"Invalid price(s) for {symbol} ({token_address}): current={current_price}, buy={buy_price}"
            )
            failure_count[token_address] = failure_count.get(token_address, 0) + 1
            if not dry_run and failure_count[token_address] >= 5 and token_address not in WHITELISTED_TOKENS:
                await db_add_to_blacklist(token_address, "Persistent invalid current/buy price")
                blacklist.add(token_address)
            return None

        # Mint decimals (use everywhere we compute amounts)
        decimals = await get_token_decimals(token_address, solana_client, logger)

        # Per-trade state
        ts = await _get_trade_state(token_address)
        _update_highest(ts, current_price)

        profit_ratio = current_price / buy_price
        hours_held = (now_ts - buy_time) / 3600.0

        # ----- TIME-BASED EXITS -----

        # Stagnant exit
        if (not ts.get("tp1_done", False)
            and hours_held >= exit_cfg["no_tp_cutoff_hours"]
            and profit_ratio < exit_cfg["no_tp_min_profit_x"]):
            logger.info(
                f"[{symbol}] No TP1 within {exit_cfg['no_tp_cutoff_hours']}h and "
                f"profit {profit_ratio:.2f}x < {exit_cfg['no_tp_min_profit_x']}x â†’ exit full."
            )
            amount = _sell_amount_lamports(token_balance, 1.0, default_decimals=decimals)
            # FIX: signature (no logger kwarg)
            quote, error = await _safe_get_jupiter_quote(
                input_mint=token_address,
                output_mint="So11111111111111111111111111111111111111112",
                amount=amount,
                user_pubkey=str(wallet.pubkey()),
                session=session,
                slippage_bps=exit_cfg["sell_slippage_bps"]
            )
            if not quote or error:
                logger.error(f"Failed stagnant-exit quote for {symbol} ({token_address}): {error}")
                return None

            if _skip_onchain_send(config):
                txid = _dry_run_txid("sell")
                _log_trade_event(
                    "DRYRUN_SELL",
                    token=symbol,
                    address=token_address,
                    amount_lamports=amount,
                    reason="STALE",
                    slippage_bps=exit_cfg["sell_slippage_bps"],
                    txid=txid
                )
                # Record metrics for dry-run sell (stagnant exit)
                _record_metric_fill(
                    token_addr=token_address,
                    symbol=symbol,
                    side="SELL",
                    quote=quote,
                    buy_amount_sol=None,
                    token_price_sol=current_price,
                    txid=txid,
                    simulated=True,
                    source="jupiter",
                    reason="STALE",
                )
            else:
                txid = await execute_jupiter_swap(quote, str(wallet.pubkey()), wallet, solana_client)
                if not txid:
                    logger.error(f"Failed stagnant-exit sell for {symbol} ({token_address})")
                    return None
                # Record metrics for live sell (stagnant exit)
                _record_metric_fill(
                    token_addr=token_address,
                    symbol=symbol,
                    side="SELL",
                    quote=quote,
                    buy_amount_sol=None,
                    token_price_sol=current_price,
                    txid=txid,
                    simulated=False,
                    source="jupiter",
                    reason="STALE",
                )

            await mark_token_sold(
                token_address=token_address,
                sell_price=float(current_price),
                sell_txid=str(txid),
                sell_time=now_ts,
            )
            logger.info(f"[{symbol}] Stagnant exit complete. txid={txid}")
            return txid

        # Max-hold hard exit
        if hours_held >= exit_cfg["max_hold_hours"]:
            logger.info(f"[{symbol}] Max-hold {exit_cfg['max_hold_hours']}h reached â†’ exit full.")
            amount = _sell_amount_lamports(token_balance, 1.0, default_decimals=decimals)
            quote, error = await _safe_get_jupiter_quote(
                input_mint=token_address,
                output_mint="So11111111111111111111111111111111111111112",
                amount=amount,
                user_pubkey=str(wallet.pubkey()),
                session=session,
                slippage_bps=exit_cfg["sell_slippage_bps"]
            )
            if not quote or error:
                logger.error(f"Failed max-hold quote for {symbol} ({token_address}): {error}")
                return None

            if _skip_onchain_send(config):
                txid = _dry_run_txid("sell")
                _log_trade_event(
                    "DRYRUN_SELL",
                    token=symbol,
                    address=token_address,
                    amount_lamports=amount,
                    reason="MAX_HOLD",
                    slippage_bps=exit_cfg["sell_slippage_bps"],
                    txid=txid
                )
                # Record metrics for dry-run sell (max-hold exit)
                _record_metric_fill(
                    token_addr=token_address,
                    symbol=symbol,
                    side="SELL",
                    quote=quote,
                    buy_amount_sol=None,
                    token_price_sol=current_price,
                    txid=txid,
                    simulated=True,
                    source="jupiter",
                    reason="MAX_HOLD",
                )
            else:
                txid = await execute_jupiter_swap(quote, str(wallet.pubkey()), wallet, solana_client)
                if not txid:
                    logger.error(f"Failed max-hold sell for {symbol} ({token_address})")
                    return None
                # Record metrics for live sell (max-hold exit)
                _record_metric_fill(
                    token_addr=token_address,
                    symbol=symbol,
                    side="SELL",
                    quote=quote,
                    buy_amount_sol=None,
                    token_price_sol=current_price,
                    txid=txid,
                    simulated=False,
                    source="jupiter",
                    reason="MAX_HOLD",
                )

            await mark_token_sold(
                token_address=token_address,
                sell_price=float(current_price),
                sell_txid=str(txid),
                sell_time=now_ts
            )
            logger.info(f"[{symbol}] Max-hold exit complete. txid={txid}")
            return txid

        # ----- PRICE-BASED EXITS -----

        # Pre-TP1 stop-loss
        if not ts.get("tp1_done", False) and profit_ratio <= exit_cfg["stop_loss_x"]:
            logger.info(f"[{symbol}] Stop-loss hit at {profit_ratio:.2f}x (â‰¤ {exit_cfg['stop_loss_x']}x) â†’ sell all.")
            amount = _sell_amount_lamports(token_balance, 1.0, default_decimals=decimals)
            quote, error = await _safe_get_jupiter_quote(
                input_mint=token_address,
                output_mint="So11111111111111111111111111111111111111112",
                amount=amount,
                user_pubkey=str(wallet.pubkey()),
                session=session,
                slippage_bps=exit_cfg["sell_slippage_bps"]
            )
            if not quote or error:
                logger.error(f"Failed stop-loss quote for {symbol} ({token_address}): {error}")
                return None

            if _skip_onchain_send(config):
                txid = _dry_run_txid("sell")
                _log_trade_event(
                    "DRYRUN_SELL",
                    token=symbol,
                    address=token_address,
                    amount_lamports=amount,
                    reason="STOP_LOSS",
                    slippage_bps=exit_cfg["sell_slippage_bps"],
                    txid=txid
                )
                # Record metrics for dry-run sell (stop-loss)
                _record_metric_fill(
                    token_addr=token_address,
                    symbol=symbol,
                    side="SELL",
                    quote=quote,
                    buy_amount_sol=None,
                    token_price_sol=current_price,
                    txid=txid,
                    simulated=True,
                    source="jupiter",
                    reason="STOP_LOSS",
                )
            else:
                txid = await execute_jupiter_swap(quote, str(wallet.pubkey()), wallet, solana_client)
                if not txid:
                    logger.error(f"Failed stop-loss sell for {symbol} ({token_address})")
                    return None
                # Record metrics for live sell (stop-loss)
                _record_metric_fill(
                    token_addr=token_address,
                    symbol=symbol,
                    side="SELL",
                    quote=quote,
                    buy_amount_sol=None,
                    token_price_sol=current_price,
                    txid=txid,
                    simulated=False,
                    source="jupiter",
                    reason="STOP_LOSS",
                )

            await mark_token_sold(token_address=token_address, sell_price=float(current_price), sell_txid=str(txid), sell_time=now_ts)
            logger.info(f"[{symbol}] Stop-loss executed. txid={txid}")
            return txid

        # TP1: partial take profit + start trailing
        if not ts.get("tp1_done", False) and profit_ratio >= exit_cfg["tp1_x"]:
            pct = exit_cfg["tp1_pct_to_sell"]
            amount = _sell_amount_lamports(token_balance, pct, default_decimals=decimals)
            logger.info(f"[{symbol}] TP1 {profit_ratio:.2f}x â†’ sell {pct*100:.1f}% and start {exit_cfg['trail_pct_after_tp1']*100:.0f}% trailing.")
            quote, error = await _safe_get_jupiter_quote(
                input_mint=token_address,
                output_mint="So11111111111111111111111111111111111111112",
                amount=amount,
                user_pubkey=str(wallet.pubkey()),
                session=session,
                slippage_bps=exit_cfg["sell_slippage_bps"]
            )
            if not quote or error:
                logger.error(f"Failed TP1 quote for {symbol} ({token_address}): {error}")
                return None

            if _skip_onchain_send(config):
                txid = _dry_run_txid("sell")
                _log_trade_event(
                    "DRYRUN_SELL",
                    token=symbol,
                    address=token_address,
                    amount_lamports=amount,
                    reason="TP1",
                    slippage_bps=exit_cfg["sell_slippage_bps"],
                    txid=txid
                )
                # Record metrics for dry-run sell (TP1)
                _record_metric_fill(
                    token_addr=token_address,
                    symbol=symbol,
                    side="SELL",
                    quote=quote,
                    buy_amount_sol=None,
                    token_price_sol=current_price,
                    txid=txid,
                    simulated=True,
                    source="jupiter",
                    reason="TP1",
                )
            else:
                txid = await execute_jupiter_swap(quote, str(wallet.pubkey()), wallet, solana_client)
                if not txid:
                    logger.error(f"Failed TP1 sell for {symbol} ({token_address})")
                    return None
                # Record metrics for live sell (TP1)
                _record_metric_fill(
                    token_addr=token_address,
                    symbol=symbol,
                    side="SELL",
                    quote=quote,
                    buy_amount_sol=None,
                    token_price_sol=current_price,
                    txid=txid,
                    simulated=False,
                    source="jupiter",
                    reason="TP1",
                )

            # Activate trailing & breakeven protection
            ts["tp1_done"] = True
            ts["trail_pct"] = exit_cfg["trail_pct_after_tp1"]
            ts["breakeven_floor"] = max(float(ts.get("breakeven_floor") or 0), buy_price * (1.0 + exit_cfg["breakeven_plus"]))
            _update_highest(ts, current_price)
            await _save_trade_state(token_address, ts)
            logger.info(f"[{symbol}] TP1 partial sell done. txid={txid}")
            return txid  # one action per cycle

        # TP2: second partial + widen trailing
        if ts.get("tp1_done", False) and (not ts.get("tp2_done", False)) and profit_ratio >= exit_cfg["tp2_x"]:
            pct = exit_cfg["tp2_pct_of_remaining"]
            amount = _sell_amount_lamports(token_balance, pct, default_decimals=decimals)
            logger.info(f"[{symbol}] TP2 {profit_ratio:.2f}x â†’ sell {pct*100:.1f}% of remaining; widen trail to {exit_cfg['trail_pct_moonbag']*100:.0f}%.")
            quote, error = await _safe_get_jupiter_quote(
                input_mint=token_address,
                output_mint="So11111111111111111111111111111111111111112",
                amount=amount,
                user_pubkey=str(wallet.pubkey()),
                session=session,
                slippage_bps=exit_cfg["sell_slippage_bps"]
            )
            if not quote or error:
                logger.error(f"Failed TP2 quote for {symbol} ({token_address}): {error}")
                return None

            if _skip_onchain_send(config):
                txid = _dry_run_txid("sell")
                _log_trade_event(
                    "DRYRUN_SELL",
                    token=symbol,
                    address=token_address,
                    amount_lamports=amount,
                    reason="TP2",
                    slippage_bps=exit_cfg["sell_slippage_bps"],
                    txid=txid
                )
                # Record metrics for dry-run sell (TP2)
                _record_metric_fill(
                    token_addr=token_address,
                    symbol=symbol,
                    side="SELL",
                    quote=quote,
                    buy_amount_sol=None,
                    token_price_sol=current_price,
                    txid=txid,
                    simulated=True,
                    source="jupiter",
                    reason="TP2",
                )
            else:
                txid = await execute_jupiter_swap(quote, str(wallet.pubkey()), wallet, solana_client)
                if not txid:
                    logger.error(f"Failed TP2 sell for {symbol} ({token_address})")
                    return None
                # Record metrics for live sell (TP2)
                _record_metric_fill(
                    token_addr=token_address,
                    symbol=symbol,
                    side="SELL",
                    quote=quote,
                    buy_amount_sol=None,
                    token_price_sol=current_price,
                    txid=txid,
                    simulated=False,
                    source="jupiter",
                    reason="TP2",
                )

            ts["tp2_done"] = True
            ts["trail_pct"] = exit_cfg["trail_pct_moonbag"]
            _update_highest(ts, current_price)
            await _save_trade_state(token_address, ts)
            logger.info(f"[{symbol}] TP2 partial sell done. txid={txid}")
            return txid  # one action per cycle

        # Trailing stops (post-TP1 or post-TP2)
        if ts.get("tp1_done", False):
            hp = float(ts.get("highest_price") or current_price)
            trail_pct = float(ts.get("trail_pct") or exit_cfg["trail_pct_after_tp1"])
            trail_floor = hp * (1.0 - trail_pct)
            # Ensure floor respects breakeven after TP1
            breakeven_floor = float(ts.get("breakeven_floor") or (buy_price * (1.0 + exit_cfg["breakeven_plus"])))
            active_floor = max(trail_floor, breakeven_floor)

            if current_price <= active_floor:
                logger.info(f"[{symbol}] Trailing stop @ {current_price:.8f} SOL (floor {active_floor:.8f}) â†’ exit full.")
                amount = _sell_amount_lamports(token_balance, 1.0, default_decimals=decimals)
                quote, error = await _safe_get_jupiter_quote(
                    input_mint=token_address,
                    output_mint="So11111111111111111111111111111111111111112",
                    amount=amount,
                    user_pubkey=str(wallet.pubkey()),
                    session=session,
                    slippage_bps=exit_cfg["sell_slippage_bps"]
                )
                if not quote or error:
                    logger.error(f"Failed trailing-stop quote for {symbol} ({token_address}): {error}")
                    return None

                if _skip_onchain_send(config):
                    txid = _dry_run_txid("sell")
                    _log_trade_event(
                        "DRYRUN_SELL",
                        token=symbol,
                        address=token_address,
                        amount_lamports=amount,
                        reason="TRAIL",
                        slippage_bps=exit_cfg["sell_slippage_bps"],
                        txid=txid
                    )
                    # Record metrics for dry-run sell (trailing stop)
                    _record_metric_fill(
                        token_addr=token_address,
                        symbol=symbol,
                        side="SELL",
                        quote=quote,
                        buy_amount_sol=None,
                        token_price_sol=current_price,
                        txid=txid,
                        simulated=True,
                        source="jupiter",
                        reason="TRAIL",
                    )
                else:
                    txid = await execute_jupiter_swap(quote, str(wallet.pubkey()), wallet, solana_client)
                    if not txid:
                        logger.error(f"Failed trailing-stop sell for {symbol} ({token_address})")
                        return None
                    # Record metrics for live sell (trailing stop)
                    _record_metric_fill(
                        token_addr=token_address,
                        symbol=symbol,
                        side="SELL",
                        quote=quote,
                        buy_amount_sol=None,
                        token_price_sol=current_price,
                        txid=txid,
                        simulated=False,
                        source="jupiter",
                        reason="TRAIL",
                    )

                await mark_token_sold(token_address=token_address, sell_price=float(current_price), sell_txid=str(txid), sell_time=now_ts)
                logger.info(f"[{symbol}] Trailing stop exit complete. txid={txid}")
                return txid

            # Keep tracking the high and persist occasionally
            if current_price > hp * 1.001:  # tiny hysteresis to reduce DB churn
                ts["highest_price"] = float(current_price)
                await _save_trade_state(token_address, ts)

        # Nothing to do this cycle
        return None

    except Exception as e:
        log_error_with_stacktrace(f"Error executing sell for {symbol} ({token_address})", e)
        failure_count[token_address] = failure_count.get(token_address, 0) + 1

        # Only blacklist repeated failures when actually sending transactions
        if (not _skip_onchain_send(config)) and failure_count[token_address] >= 5 and token_address not in WHITELISTED_TOKENS:
            await db_add_to_blacklist(token_address, "Repeated sell failures")
            blacklist.add(token_address)

        return None

# -------------------------- Diagnostics: open-position snapshot --------------------------

async def _log_positions_snapshot(
    open_positions: List[Dict[str, Any]],
    session: aiohttp.ClientSession,
    solana_client: Any,            # <- use Any so Pylance doesn't require a real type
    config: Dict[str, Any],
) -> None:
    if not open_positions:
        logger.info("[WATCH] No open positions.")
        return

    exit_cfg = _load_exit_cfg(config)

    for pos in open_positions:
        addr = pos.get("address")
        sym = pos.get("symbol", "UNKNOWN")
        if not addr:
            continue
        try:
            status = await get_token_trade_status(addr)  # contains buy_price, buy_time, possibly trade_state
            buy_price = float((status or {}).get("buy_price") or 0.0)
            buy_time = int((status or {}).get("buy_time") or 0)

            # Current price
            current_price = await get_token_price_in_sol(
                addr, session, price_cache, {"address": addr, "symbol": sym}
            )
            profit_x = (current_price / buy_price) if (current_price > 0 and buy_price > 0) else 0.0
            hours_held = (time.time() - buy_time) / 3600.0 if buy_time else 0.0

            # Trade-state fields (if any)
            ts = ((status or {}).get("trade_state") or {})
            tp1_done = bool(ts.get("tp1_done", False))
            tp2_done = bool(ts.get("tp2_done", False))
            highest_price = float(ts.get("highest_price") or 0.0)
            trail_pct = ts.get("trail_pct")  # None, 0.18, or 0.30
            breakeven_floor = ts.get("breakeven_floor")

            trail_floor = None
            if highest_price > 0 and trail_pct not in (None, 0, 0.0):
                trail_floor = highest_price * (1.0 - float(trail_pct))

            # Active floor after TP1 = max(trailing floor, breakeven+)
            active_floor = None
            if tp1_done:
                be = float(breakeven_floor or (buy_price * (1.0 + exit_cfg["breakeven_plus"])))
                tr = float(trail_floor or 0.0)
                active_floor = max(be, tr)

            logger.info(
                "[WATCH] %s %s | price=%.8f SOL | buy=%.8f | pnl=%.2fx | held=%.2fh | "
                "TP1=%s TP2=%s | high=%s | trail=%s | floor=%s",
                sym,
                f"{addr[:4]}…{addr[-4:]}",
                current_price,
                buy_price,
                profit_x,
                hours_held,
                tp1_done,
                tp2_done,
                f"{highest_price:.8f}" if highest_price else "-",
                f"{float(trail_pct)*100:.1f}%" if trail_pct not in (None, 0, 0.0) else "-",
                f"{active_floor:.8f}" if active_floor else "-",
            )
        except Exception as e:
            logger.warning(f"[WATCH] Snapshot failed for {sym} ({addr}): {e}", exc_info=True)

# -------------------------- Main loop --------------------------
async def main() -> None:
    logger.info("Starting trading bot")
    solana_client: Optional[AsyncClientType] = None
    try:  # ==== OUTER TRY ====
        # --- .env loading + diagnostics (do this first) ---
        from dotenv import load_dotenv, find_dotenv
        from pathlib import Path

        candidate_envs = [
            # preferred app-data location used by the GUI
            Path.home() / "AppData" / "Local" / "SOLOTradingBot" / ".env",
            # repo / working dir fallback
            Path.cwd() / ".env",
        ]
        loaded_from = None
        for p in candidate_envs:
            try:
                if p.exists():
                    load_dotenv(dotenv_path=str(p), override=True)
                    loaded_from = str(p)
                    break
            except Exception:
                pass
        if not loaded_from:
            discovered = find_dotenv(usecwd=True)
            if discovered:
                load_dotenv(discovered, override=True)
                loaded_from = discovered
                

        # quick env echo so we know what actually applied
        logger.info("ENV: loaded from %s", loaded_from or "<none>")
        logger.info(
            "ENV FLAGS: DRY_RUN=%s DISABLE_SEND_TX=%s JUPITER_QUOTE_ONLY=%s JUPITER_EXECUTE=%s",
            os.getenv("DRY_RUN", "0"), os.getenv("DISABLE_SEND_TX", "0"),
            os.getenv("JUPITER_QUOTE_ONLY", "0"), os.getenv("JUPITER_EXECUTE", "0"),
        )
        logger.info(
            "Birdeye: ENABLE=%s FORCE_DISABLE=%s KEY_PRESENT=%s RPS=%s cycle_cap=%s run_cap=%s",
            os.getenv("BIRDEYE_ENABLE", "0") in ("1", "true", "True"),
            os.getenv("FORCE_DISABLE_BIRDEYE", "0"),
            "yes" if os.getenv("BIRDEYE_API_KEY") else "no",
            os.getenv("BIRDEYE_RPS", "?"),
            os.getenv("BIRDEYE_MAX_CALLS_PER_CYCLE", "?"),
            os.getenv("BIRDEYE_MAX_CALLS_PER_RUN", "?"),
        )
        logger.info(
            "Dexscreener: DEX_PAGES=%s DEX_PER_PAGE=%s DEX_MAX=%s DEX_FORCE_IPV4=%s",
            os.getenv("DEX_PAGES", "?"), os.getenv("DEX_PER_PAGE", "?"),
            os.getenv("DEX_MAX", "?"), os.getenv("DEX_FORCE_IPV4", "auto"),
        )

        # --- NOW load config and init logging BEFORE using `config` anywhere ---
        config = load_config()
        setup_logging(config)

        # --- Testing override: loosen selection thresholds so shortlist doesn't collapse ---
        TEST_LOOSEN = bool(int(os.getenv("TEST_LOOSEN_SELECTION", "0")))
        if TEST_LOOSEN:
            sel = (config.get("selection") or {})
            sel.setdefault("min_liquidity_usd", 0)
            sel.setdefault("min_volume_24h_usd", 0)
            sel.setdefault("min_age_minutes", 0)
            sel.setdefault("allow_missing_mc", True)
            sel.setdefault("allow_missing_liq", True)
            config["selection"] = sel
            logger.info("Selection thresholds loosened for testing (TEST_LOOSEN_SELECTION=1).")

       
        # --- SINGLE-INSTANCE GUARD (claim before any side effects) ---
        if not acquire_single_instance_or_explain(int(os.getenv("PID_STALE_AFTER_S", "180"))):
            return

        # Start a background heartbeat that stays fresh even during long cycles
        stop_hb = asyncio.Event()
        hb_task = asyncio.create_task(heartbeat_task(stop_hb, int(os.getenv("HEARTBEAT_INTERVAL_S", "5"))))

        # Persist our PID & emit an immediate heartbeat
        _write_pid_file()
        _heartbeat(throttle_s=0)
        # --- END GUARD ---

        # RPC client (create ONCE)
        solana_client = _new_async_client(config["solana"]["rpc_endpoint"])

        # Wallet
        private_key = os.getenv(config["wallet"]["private_key_env"])
        if not private_key:
            logger.error("%s not set", config["wallet"]["private_key_env"])
            return
        wallet = Keypair.from_base58_string(private_key)
        logger.info("Wallet loaded: %s", wallet.pubkey())

        # RPC quick check (resilient to response shape changes)
        try:
            bh = await solana_client.get_latest_blockhash()
            slot = getattr(getattr(bh, "context", None), "slot", None)
            blockhash = getattr(getattr(bh, "value", None), "blockhash", None)
            short_bh = (blockhash[:12] + "…") if isinstance(blockhash, str) else str(blockhash)

            ver = await solana_client.get_version()

            # Normalize version payload to something we can query safely
            v = getattr(ver, "value", ver)
            core = None
            if isinstance(v, dict):
                core = (
                    v.get("solana-core")
                    or v.get("solana_core")
                    or v.get("solanaCore")
                    or v.get("solana")
                    or v.get("version")
                )
            else:
                # Sometimes it's a typed object; probe common attribute names
                core = (
                    getattr(v, "solana_core", None)
                    or getattr(v, "solanaCore", None)
                    or getattr(v, "solana", None)
                    or getattr(v, "version", None)
                )

            logger.info(
                "RPC OK via %s - slot=%s, blockhash=%s, core=%s",
                config["solana"]["rpc_endpoint"], slot, short_bh, core or "unknown"
            )
        except Exception as e:
            logger.warning("RPC health check failed: %s", e)


        # DB init
        await init_db()
        await _ensure_schema_once(logger)

        blacklist = await db_load_blacklist()
        failure_count: Dict[str, int] = {}
        cycle_index = 0

        async with aiohttp.ClientSession() as session:
            while True:
                # Stop flag check (operator-controlled)
                try:
                    with open("bot_stop_flag.txt", "r") as f:
                        if f.read().strip() == "1":
                            logger.info("Stop flag detected, exiting trading bot")
                            break
                except FileNotFoundError:
                    pass

                # ===== PER-CYCLE START =====
                try:  # <-- per-cycle try MUST align with except below
                    # Reset Jupiter offline breaker for this cycle
                    global _JUP_OFFLINE
                    _JUP_OFFLINE = False

                    # Hygiene
                    await clear_expired_blacklist(max_age_hours=24)
                    await review_blacklist()

                    # Balance (telemetry)
                    startup_balance = await get_sol_balance(wallet, solana_client)
                    required_min = float((config.get("bot") or {}).get("required_sol", 0.0))
                    logger.info("Startup SOL balance: %.6f (min required=%.6f)", startup_balance, required_min)

                    # Source queries for trading pass
                    _discovery_cfg = (config.get("discovery") or {})
                    queries = _discovery_cfg.get("dexscreener_queries") or ["solana"]
                    queries = list(dict.fromkeys(queries))
                    logger.info("Trading: using dexscreener queries %s", queries)

                    # Build shortlist (DB first, then live)
                    eligible_tokens: List[Dict[str, Any]] = []
                    using_db_shortlist = False

                    prefer_db_shortlist = bool((config.get("trading") or {}).get("prefer_persisted_shortlist", True))
                    db_max_age_s = int((config.get("trading") or {}).get("db_shortlist_max_age_s", 300))
                    db_min_count = int((config.get("trading") or {}).get("db_shortlist_min_count", 10))

                    try:
                        eligible_tokens_from_db = _load_persisted_shortlist_from_db(
                            config=config, max_age_s=db_max_age_s, min_count=db_min_count
                        )
                    except Exception as _e:
                        logger.warning("DB shortlist load failed: %s; falling back to live.", _e)
                        eligible_tokens_from_db = []

                    if prefer_db_shortlist and eligible_tokens_from_db:
                        eligible_tokens = eligible_tokens_from_db
                        using_db_shortlist = True
                        logger.info("Using %d token(s) from persisted shortlist.", len(eligible_tokens))
                    else:
                        logger.info("Persisted shortlist stale/small/empty; running live discovery.")
                        eligible_tokens = await _build_live_shortlist(
                            session=session,
                            solana_client=solana_client,
                            config=config,
                            queries=queries,
                            blacklist=blacklist,
                            failure_count=failure_count,
                        )

                    # Common shortlist processing
                    fallback_hours = int((config.get("trading") or {}).get("db_fallback_hours", 1))
                    _fallback_age_s = max(0, fallback_hours) * 3600

                    if using_db_shortlist:
                        try:
                            enriched = await _enrich_shortlist_with_signals(eligible_tokens)
                            eligible_tokens = enriched or eligible_tokens
                            if not enriched:
                                for t in eligible_tokens:
                                    t.setdefault("_enriched", False)
                        except Exception:
                            logger.warning("Signal enrichment failed on DB shortlist; keeping original.", exc_info=True)
                            for t in eligible_tokens:
                                t.setdefault("_enriched", False)
                        try:
                            await persist_eligible_shortlist(eligible_tokens, prune_hours=168)
                        except Exception:
                            logger.debug("Persisting shortlist (DB path refresh) failed", exc_info=True)
                    else:
                        # Ensure categories exist before per-bucket selection (live path)
                        eligible_tokens = _restore_or_compute_categories(eligible_tokens)
                        eligible_tokens = await select_top_five_per_category(eligible_tokens)
                        try:
                            enriched = await _enrich_shortlist_with_signals(eligible_tokens)
                            if enriched:
                                eligible_tokens = enriched
                            else:
                                logger.warning("Live shortlist collapsed; attempting DB fallback (<=%ds).", _fallback_age_s)
                                db_fallback = _load_persisted_shortlist_from_db(
                                    config=config, max_age_s=_fallback_age_s, min_count=5
                                )
                                if db_fallback:
                                    # Backfill categories on DB fallback so downstream logic/UI has buckets
                                    eligible_tokens = _restore_or_compute_categories(db_fallback)
                                    using_db_shortlist = True
                                    logger.info("Pulled %d tokens from DB fallback.", len(eligible_tokens))
                                else:
                                    # Backfill again before the fallback selector
                                    eligible_tokens = _restore_or_compute_categories(eligible_tokens)
                                    eligible_tokens = await select_top_five_per_category(eligible_tokens)
                                    for t in eligible_tokens:
                                        t.setdefault("_enriched", False)
                        except Exception:
                            logger.warning("Signal enrichment failed on live shortlist; attempting DB fallback.", exc_info=True)
                            db_fallback = _load_persisted_shortlist_from_db(
                                config=config, max_age_s=_fallback_age_s, min_count=5
                            )
                            if db_fallback:
                                # Backfill categories on DB fallback too
                                eligible_tokens = _restore_or_compute_categories(db_fallback)
                                using_db_shortlist = True
                                logger.info("Pulled %d tokens from DB fallback.", len(eligible_tokens))
                            else:
                                for t in eligible_tokens:
                                    t.setdefault("_enriched", False)

                    _apply_tech_tiebreaker(eligible_tokens, config, logger=logger)
                    logger.info("Found %d eligible tokens", len(eligible_tokens))
                    log_scoring_telemetry(eligible_tokens, where="shortlist")

                    if using_db_shortlist and not eligible_tokens:
                        logger.info("DB shortlist yielded zero; forcing live discovery once.")
                        eligible_tokens = await _fetch_live_shortlist_once(
                            session=session, solana_client=solana_client, config=config,
                            queries=queries, blacklist=blacklist, failure_count=failure_count, logger=logger
                        )

                    # Mode flags
                    tsec = (config.get("trading") or {})
                    bsec = (config.get("bot") or {})

                    dry = _dry_run_on(config)
                    send_disabled = _skip_onchain_send(config)
                    simulate = bool(tsec.get("simulate", bsec.get("simulate", False)))

                    env_dry = os.getenv("DRY_RUN", "0") == "1"
                    env_disable_send = os.getenv("DISABLE_SEND_TX", "0") == "1"
                    env_jup_quote_only = os.getenv("JUPITER_QUOTE_ONLY", "0") == "1"
                    env_jup_execute = os.getenv("JUPITER_EXECUTE", "0") == "1"

                    dry = dry or env_dry
                    simulate = simulate or env_dry or env_jup_quote_only
                    send_disabled = send_disabled or env_disable_send or env_jup_quote_only
                    if env_jup_execute and not env_jup_quote_only:
                        simulate = False

                    _dry_run       = bool(locals().get("dry_run", locals().get("dry", False)))
                    _send_disabled = bool(locals().get("send_disabled", False))
                    _simulate      = bool(locals().get("simulate", False))

                    try:
                        _has_balance = await has_min_balance(wallet, config, session=session)
                    except Exception as _e:
                        logger.warning("Balance check failed; treating as no-balance. (%s)", _e)
                        _has_balance = False

                    _enabled = bool((config.get("trading") or {}).get("enabled", True))
                    _send_tx = not _send_disabled
                    can_trade = True if (_dry_run or _send_disabled or _simulate) else bool(_has_balance)

                    logger.info(
                        "TRADING-CYCLE: shortlist=%d enabled=%s dry_run=%s simulate=%s send_tx=%s can_trade=%s",
                        len(eligible_tokens), _enabled, _dry_run, _simulate, _send_tx, can_trade
                    )

                    # ---------------- Buys ----------------
                    candidates = eligible_tokens[:10]
                    logger.info("TRADING-CYCLE: entering buy loop with %d candidate(s)", len(candidates))
                    try:
                        if not candidates:
                            logger.info("No buy candidates this cycle")
                        else:
                            WSOL = "So11111111111111111111111111111111111111112"
                            USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
                            user_pubkey = str(wallet.pubkey())
                            LAMPORTS_PER_SOL = 1_000_000_000

                            # Resolve SOL price
                            sol_price: Optional[float] = None
                            try:
                                sol_price = await get_sol_price(session)
                                if sol_price and sol_price > 0:
                                    try:
                                        price_cache["SOLUSD"] = float(sol_price)
                                    except Exception:
                                        pass
                            except Exception:
                                sol_price = None

                            if not sol_price:
                                try:
                                    cached = float(price_cache.get("SOLUSD") or 0.0)
                                    sol_price = cached if cached > 0 else None
                                except Exception:
                                    sol_price = None

                            if not sol_price:
                                try:
                                    quote, qerr = await _safe_get_jupiter_quote(
                                        input_mint=WSOL,
                                        output_mint=USDC,
                                        amount=LAMPORTS_PER_SOL,  # 1 SOL
                                        user_pubkey=user_pubkey,
                                        session=session,
                                        slippage_bps=50,
                                    )
                                    if quote and not qerr:
                                        usdc_out_raw = float(quote.get("outAmount") or 0.0)  # USDC 6dp
                                        derived = usdc_out_raw / 1_000_000 if usdc_out_raw > 0 else 0.0
                                        if derived > 0:
                                            sol_price = derived
                                            try:
                                                price_cache["SOLUSD"] = float(sol_price)
                                            except Exception:
                                                pass
                                except Exception:
                                    pass
                            if not sol_price:
                                try:
                                    env_fallback = os.getenv("SOL_FALLBACK_PRICE")
                                    sol_price = float(env_fallback) if env_fallback else None
                                except Exception:
                                    sol_price = None

                            if not sol_price:
                                logger.warning("Unable to determine SOL price; skipping buys this cycle.")
                            else:
                                # --- Sizing ---
                                wallet_balance = await get_sol_balance(wallet, solana_client)

                                buy_amount_sol = await _resolve_buy_amount_sol(
                                    get_buy_amount_fn=get_buy_amount,
                                    config=config,
                                    sol_price=sol_price,
                                    wallet_balance=wallet_balance,
                                    token=None,  # pass a token dict if you tilt sizing per-asset
                                )

                                # Optional: do not size above available balance (minus a dust buffer)
                                try:
                                    max_afford = max(0.0, float(wallet_balance) - 0.0003)
                                    if buy_amount_sol > max_afford:
                                        logger.info(
                                            "Clamping buy amount from %.6f SOL to wallet max %.6f SOL",
                                            buy_amount_sol,
                                            max_afford,
                                        )
                                        buy_amount_sol = max_afford
                                except Exception:
                                    pass

                                # --- Compute slippage once for the cycle ---
                                slippage_bps = _clamp_bps(int((config.get("trading") or {}).get("slippage_bps", 150)))

                                # rank/tilt
                                enriched = await _enrich_shortlist_with_signals(candidates)
                                if enriched:
                                    candidates = enriched
                                else:
                                    for t in candidates:
                                        t.setdefault("_enriched", False)
                                    logger.warning(
                                        "Candidate enrichment returned empty; keeping un-enriched candidates."
                                    )
                                _apply_tech_tiebreaker(candidates, config, logger=logger)

                                for tok in candidates:
                                    token_addr = tok.get("address")
                                    symbol = tok.get("symbol") or "UNKNOWN"
                                    source = tok.get("source", "dexscreener")
                                    if not token_addr:
                                        continue

                                    # --- Per-token sizing override (nice-to-have) ---
                                    per_token_buy_amount_sol = buy_amount_sol  # default to cycle-level amount
                                    if (config.get("bot") or {}).get("per_asset_sizing", True):
                                        try:
                                            # unwraps (sol, usd) → sol inside the resolver
                                            per_token_buy_amount_sol = await _resolve_buy_amount_sol(
                                                get_buy_amount_fn=get_buy_amount,
                                                config=config,
                                                sol_price=sol_price,
                                                wallet_balance=wallet_balance,
                                                token=tok,  # pass the actual token for richer logs + sizing
                                            )
                                        except Exception as e:
                                            logger.warning(
                                                "Per-asset sizing failed for %s: %s",
                                                tok.get("symbol") or tok.get("address") or "unknown",
                                                e,
                                            )
                                            per_token_buy_amount_sol = buy_amount_sol

                                    # Skip tokens with zero/negative sizing
                                    if per_token_buy_amount_sol <= 0:
                                        logger.warning(
                                            "Buy sizing <= 0 for %s; skipping.",
                                            tok.get("symbol") or tok.get("address") or "unknown",
                                        )
                                        continue

                                    # --- Convert to lamports for this token ---
                                    amount_lamports_full = max(1, int(per_token_buy_amount_sol * LAMPORTS_PER_SOL))

                                    # Small probe amount for preflight / route sanity (~0.0001 SOL or 10k lamports minimum)
                                    amount_lamports_probe = min(
                                        amount_lamports_full,
                                        max(10_000, int(0.0001 * LAMPORTS_PER_SOL)),
                                    )

                                    # Optional: avoid dust routes
                                    if amount_lamports_full < 10_000:
                                        logger.debug(
                                            "Skipping dust-size buy for %s (full=%d lamports)",
                                            tok.get("symbol") or tok.get("address") or "unknown",
                                            amount_lamports_full,
                                        )
                                        continue

                                    # Preflight
                                    is_tradable, errtxt = await _is_jupiter_tradable(
                                        input_mint=WSOL,
                                        output_mint=token_addr,
                                        lamports=amount_lamports_probe,
                                        user_pubkey=user_pubkey,
                                        session=session,
                                        slippage_bps=slippage_bps,
                                    )
                                    if not is_tradable:
                                        msg = (
                                            "TOKEN_NOT_TRADABLE"
                                            if errtxt and "TOKEN_NOT_TRADABLE" in errtxt
                                            else (errtxt or "unknown")
                                        )
                                        logger.info("Skip %s (%s): %s", symbol, token_addr, msg)
                                        continue

                                    # Full quote
                                    quote, qerr = await _safe_get_jupiter_quote(
                                        input_mint=WSOL,
                                        output_mint=token_addr,
                                        amount=amount_lamports_full,
                                        user_pubkey=user_pubkey,
                                        session=session,
                                        slippage_bps=slippage_bps,
                                    )
                                    if not quote or qerr:
                                        logger.error(
                                            "Failed full-quote for %s (%s): %s",
                                            symbol,
                                            token_addr,
                                            qerr or "no route",
                                        )
                                        continue

                                    out_amount = None
                                    try:
                                        out_amount = float(quote.get("outAmount") or 0)
                                    except Exception:
                                        pass
                                    logger.info(
                                        "BUY-CANDIDATE %s (%s): preflight OK, got full quote (lamports_in=%d, out≈%s).",
                                        symbol,
                                        token_addr,
                                        amount_lamports_full,
                                        f"{int(out_amount):,}" if out_amount is not None else "?",
                                    )

                                    buy_price_sol = _compute_buy_price_sol_per_token(quote)
                                    if buy_price_sol is None:
                                        try:
                                            buy_price_sol = await get_token_price_in_sol(
                                                token_addr,
                                                session,
                                                price_cache,
                                                {"address": token_addr, "symbol": symbol},
                                            )
                                            buy_price_sol = (
                                                float(buy_price_sol)
                                                if buy_price_sol and buy_price_sol > 0
                                                else None
                                            )
                                        except Exception:
                                            buy_price_sol = None

                                    # Simulate or execute
                                    if _dry_run or _send_disabled or _simulate:
                                        txid = _dry_run_txid("BUY")
                                        try:
                                            await update_token_record(
                                                token={
                                                    "address": token_addr,
                                                    "symbol": symbol,
                                                    "source": source,
                                                },
                                                buy_price=buy_price_sol,
                                                buy_txid=txid,
                                                buy_time=int(time.time()),
                                                is_trading=True,
                                            )
                                        except Exception:
                                            logger.debug(
                                                "Paper trade persistence failed for %s",
                                                token_addr,
                                                exc_info=True,
                                            )
                                        logger.info(
                                            "DRYRUN-BUY %s (%s): txid=%s amount_in_SOL=%.6f price≈%s SOL/token",
                                            symbol,
                                            token_addr,
                                            txid,
                                            per_token_buy_amount_sol,
                                            f"{buy_price_sol:.12f}"
                                            if isinstance(buy_price_sol, (float, int))
                                            else "?",
                                        )
                                        # Record metrics for dry-run buy (best-effort via wrapper)
                                        try:
                                            _metrics_on_fill(
                                                token_addr=token_addr,
                                                symbol=symbol,
                                                side="BUY",
                                                qty=0.0,  # qty unknown here; metrics store will infer from quote if possible
                                                price_usd=0.0,
                                                fee_usd=0.0,
                                                txid=txid,
                                                simulated=True,
                                                quote=quote,
                                                buy_amount_sol=per_token_buy_amount_sol,
                                                token_price_sol=buy_price_sol,
                                                source=source,
                                            )
                                        except Exception:
                                            logger.debug("metrics wrapper failed (dryrun buy)", exc_info=True)
                                        continue

                                    try:
                                        exec_res = await execute_jupiter_swap(
                                            input_mint=WSOL,
                                            output_mint=token_addr,
                                            amount=amount_lamports_full,
                                            user_pubkey=user_pubkey,
                                            session=session,
                                            slippage_bps=slippage_bps,
                                            config=config,
                                        )
                                        txid = None
                                        if isinstance(exec_res, dict):
                                            txid = (
                                                exec_res.get("txid")
                                                or exec_res.get("signature")
                                                or exec_res.get("id")
                                            )
                                        logger.info(
                                            "LIVE-BUY %s (%s) executed: txid=%s price≈%s SOL/token",
                                            symbol,
                                            token_addr,
                                            txid or "?",
                                            f"{buy_price_sol:.12f}"
                                            if isinstance(buy_price_sol, (float, int))
                                            else "?",
                                        )
                                        # Record metrics for live buy (best-effort via wrapper)
                                        try:
                                            _metrics_on_fill(
                                                token_addr=token_addr,
                                                symbol=symbol,
                                                side="BUY",
                                                qty=0.0,
                                                price_usd=0.0,
                                                fee_usd=0.0,
                                                txid=txid,
                                                simulated=False,
                                                quote=quote,
                                                buy_amount_sol=per_token_buy_amount_sol,
                                                token_price_sol=buy_price_sol,
                                                source=source,
                                            )
                                        except Exception:
                                            logger.debug("metrics wrapper failed (live buy)", exc_info=True)

                                        # Snapshot equity point for metrics (best-effort)
                                        try:
                                            _metrics_snapshot_equity_point(logger=logger)
                                        except Exception:
                                            logger.debug("metrics snapshot failed after BUY", exc_info=True)

                                    except Exception as e:
                                        logger.error(
                                            "Jupiter execute failed for %s (%s): %s",
                                            symbol,
                                            token_addr,
                                            e,
                                        )
                                        continue
                    except Exception as e:
                        logger.error("BUY-LOOP error: %s", e, exc_info=True)

                    # ---------------- Sells ----------------
                    open_positions = await get_open_positions()
                    cooldown_sec = int(float((config.get("bot") or {}).get("cooldown_seconds", 3)))

                    if cycle_index % 5 == 0:
                        try:
                            await _log_positions_snapshot(open_positions, session, solana_client, config)
                        except Exception:
                            logger.debug("positions snapshot failed", exc_info=True)

                    for pos in open_positions:
                        try:
                            addr = (pos.get("address") if isinstance(pos, dict) else None) or ""
                            if not addr:
                                logger.debug("Skipping position with no address: %s", pos)
                                await asyncio.sleep(0.25)
                                continue
                            sym = (pos.get("symbol") if isinstance(pos, dict) else None) or "UNKNOWN"
                            token_data = {"address": addr, "symbol": sym}

                            if _dry_run or _simulate or _send_disabled:
                                txid = _dry_run_txid("SELL")
                                try:
                                    await update_token_record(
                                        token={"address": addr, "symbol": sym},
                                        sell_txid=txid, sell_time=int(time.time()), is_trading=False
                                    )
                                except Exception:
                                    logger.debug("Paper sell persistence failed for %s", addr, exc_info=True)
                                logger.info("DRYRUN-SELL %s (%s): txid=%s", sym, addr, txid)

                                # Best-effort metrics record for dry-run sell
                                try:
                                    _metrics_on_fill(
                                        token_addr=addr,
                                        symbol=sym,
                                        side="SELL",
                                        qty=0.0,
                                        price_usd=0.0,
                                        fee_usd=0.0,
                                        txid=txid,
                                        simulated=True,
                                        source="jupiter",
                                    )
                                except Exception:
                                    logger.debug("metrics wrapper failed (dryrun sell)", exc_info=True)

                                await asyncio.sleep(cooldown_sec)
                                continue

                            txid = await real_on_chain_sell(
                                token_data, wallet, solana_client, session, blacklist, failure_count, config
                            )
                            if txid:
                                logger.info("LIVE-SELL %s (%s) txid=%s", sym, addr, txid)
                            else:
                                logger.warning("Sell returned no txid for %s (%s)", sym, addr)

                            # Best-effort metrics snapshot after sells (live & paper)
                            try:
                                _metrics_snapshot_equity_point(logger=logger)
                            except Exception:
                                logger.debug("metrics snapshot failed after SELL", exc_info=True)

                            await asyncio.sleep(cooldown_sec)

                        except Exception as e:
                            log_error_with_stacktrace(
                                f"Error processing sell for {pos.get('symbol','UNKNOWN')} ({pos.get('address')})", e
                            )
                            await asyncio.sleep(0.5)

                    # ---- end-of-cycle ----
                    interval = int(float((config.get("bot") or {}).get("cycle_interval", 30)))
                    logger.info("Completed trading cycle, waiting %d seconds", interval)
                    _heartbeat()
                    await asyncio.sleep(interval)
                    cycle_index += 1

                except Exception as e:  # <-- pairs with per-cycle try
                    log_error_with_stacktrace("Error in trading cycle", e)
                    await asyncio.sleep(60)
                    continue
                # ===== PER-CYCLE END =====

    except Exception as e:  # <-- pairs with OUTER try
        log_error_with_stacktrace("Fatal error in main loop", e)

    finally:
        # stop heartbeat task and cleanup PID/heartbeat files
        try:
            if 'stop_hb' in locals():
                stop_hb.set()
            if 'hb_task' in locals():
                hb_task.cancel()
                try:
                    await asyncio.gather(hb_task, return_exceptions=True)
                except Exception:
                    pass
        except Exception:
            logger.debug("Heartbeat task shutdown failed", exc_info=True)

        # final heartbeat is nice-to-have
        try:
            _heartbeat(throttle_s=0)
        except Exception:
            logger.debug("Final heartbeat failed", exc_info=True)

        # Close RPC client if opened
        try:
            if solana_client is not None:
                await solana_client.close()
        except Exception:
            logger.debug("solana_client.close() failed", exc_info=True)

        # Release our single-instance lock (removes PID, heartbeat, started_at if we own them)
        try:
            release_single_instance()
        except Exception:
            logger.debug("release_single_instance failed", exc_info=True)

        logger.info("Trading bot stopped")

if __name__ == "__main__":
    asyncio.run(main())
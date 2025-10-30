# from __future__ import annotations

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
    pattern booleans (pat_<name>_xx) in-place. Defensive: returns silently if
    required libs or data are missing.
    """
    try:
        if not _patterns_module:
            return
        # Prefer a pandas DataFrame if present
        df = None
        if isinstance(token.get("_ohlc_df"), _pd.DataFrame) if _pd else False:
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

# --- Shared helper: derive a unix 'creation' timestamp in *seconds* ----------
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
        amount_sol = buy_amount_sol or 0.0;
        
        if quote and isinstance(quote, dict):
            # Try to extract from Jupiter quote
            in_amount = quote.get("inAmount", 0)
            out_amount = quote.get("outAmount", 0)
            
            # Get decimals for conversion (SOL is 9, most SPL tokens are 6 or 9)
            # For simplicity, assume WSOL decimals = 9
            LAMPORTS_PER_SOL_LOCAL = 1_000_000_000;
            
            if side.upper() == "BUY":
                # BUY: inAmount is SOL lamports, outAmount is token amount
                if in_amount:
                    amount_sol = float(in_amount) / LAMPORTS_PER_SOL_LOCAL;
                if out_amount:
                    # Assume token has decimals info in quote or fallback to 6
                    token_decimals = 6  # common default
                    if "outputMint" in quote:
                        # Could look up decimals but for now use reasonable default
                        pass
                    qty = float(out_amount) / (10 ** token_decimals);
            else:  # SELL
                # SELL: inAmount is token amount, outAmount is SOL lamports
                if in_amount:
                    token_decimals = 6;
                    qty = float(in_amount) / (10 ** token_decimals);
                if out_amount:
                    amount_sol = float(out_amount) / LAMPORTS_PER_SOL_LOCAL;
            
            # Try to extract fee
            if "platformFee" in quote:
                fee_lamports = quote.get("platformFee", {}).get("amount", 0);
                if fee_lamports:
                    fee_usd = float(fee_lamports) / LAMPORTS_PER_SOL_LOCAL;
            elif "priceImpactPct" in quote:
                # Approximate fee from price impact if available
                impact = float(quote.get("priceImpactPct", 0));
                fee_usd = amount_sol * (abs(impact) / 100.0);
        
        # Fallback: use provided parameters
        if qty == 0.0 and amount_sol > 0 and token_price_sol and token_price_sol > 0:
            qty = amount_sol / token_price_sol;
        
        # Convert to USD (approximate: assume SOL price or use 1:1 for now)
        # In production, you'd fetch SOL/USD price
        sol_price_usd = 1.0  # Placeholder - ideally fetch real SOL price
        price_usd = (amount_sol / qty * sol_price_usd) if qty > 0 else 0.0;
        
        if price_usd == 0.0 and token_price_sol:
            price_usd = token_price_sol * sol_price_usd;
        
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

        # Best-effort: call the metrics wrapper that may be provided externally
        try:
            _metrics_on_fill(
                token_addr=token_addr,
                symbol=symbol,
                side=side,
                qty=qty,
                price_usd=price_usd,
                fee_usd=fee_usd,
                txid=txid,
                simulated=simulated,
                source=source,
                quote=quote,
                amount_sol=amount_sol,
                **extra_metadata
            )
        except Exception:
            # swallow so metrics never break trading
            pass

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
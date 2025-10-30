# solana_trading_bot_bundle/trading_bot/fetching.py — imports

import os
import json
import time
import asyncio
import logging
import traceback
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import socket  # IPv4/IPv6 connector control

import aiohttp
import aiosqlite
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Solana / solders
from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey

# Project constants / paths
from solana_trading_bot_bundle.common.constants import appdata_dir

# Optional feature flags (env + config aware hard switches)
# These let us log/skip Raydium/Birdeye fetches cleanly even if code paths are reached.
try:
    from solana_trading_bot_bundle.common.feature_flags import (
        is_enabled_raydium,
        is_enabled_birdeye,
        FORCE_DISABLE_RAYDIUM,
        FORCE_DISABLE_BIRDEYE,
    )
except Exception:
    # Soft fallbacks if the feature_flags module isn't present.
    # IMPORTANT: make them arg-flexible so both is_enabled_raydium() and is_enabled_raydium(cfg) work.
    def _env_on(name: str, default: bool) -> bool:
        v = os.getenv(name, "1" if default else "0").strip().lower()
        return v in ("1", "true", "yes", "on", "y")
    def is_enabled_raydium(*_args, **_kwargs) -> bool:
        # default OFF is safer while we stabilize; flip to True if you prefer legacy behavior
        return _env_on("RAYDIUM_ENABLE", False)
    def is_enabled_birdeye(*_args, **_kwargs) -> bool:
        return _env_on("BIRDEYE_ENABLE", False)
    FORCE_DISABLE_RAYDIUM = _env_on("FORCE_DISABLE_RAYDIUM", False)
    FORCE_DISABLE_BIRDEYE = _env_on("FORCE_DISABLE_BIRDEYE", False)


# Local utilities / config loader
try:
    from .utils import WHITELISTED_TOKENS, load_config
except Exception:
    from .utils import WHITELISTED_TOKENS
    def load_config() -> Dict[str, Any]:
        return {}

# DB cache helpers
from .database import (
    get_cached_token_data,
    cache_token_data,
    get_cached_creation_time,
    cache_creation_time,
)

# ---------------------------------------------------------------------
# DB writer shim: tolerate both cache_token_data(token) and (addr, token)
# ---------------------------------------------------------------------
import inspect as _ins

async def _maybe_await(x):
    import inspect as _inspect
    return (await x) if _inspect.isawaitable(x) else x

try:
    _sig = _ins.signature(cache_token_data)
    # count required positional params (no defaults)
    _CACHE_TOKEN_DATA_PARAM_COUNT = sum(
        1 for p in _sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
    )
except Exception:
    # if we can't introspect, assume the newer 1-arg form
    _CACHE_TOKEN_DATA_PARAM_COUNT = 1

async def _write_token_row(addr: str, token: dict) -> None:
    # ensure the token dict carries its address for the 1-arg form
    token.setdefault("address", addr)
    if _CACHE_TOKEN_DATA_PARAM_COUNT <= 1:
        # newer DB writer: cache_token_data(token_dict)
        await _maybe_await(cache_token_data(token))
    else:
        # legacy DB writer: cache_token_data(address, token_dict)
        await _maybe_await(cache_token_data(addr, token))



# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger("SOLOTradingBot")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------
# Appdata paths
# ---------------------------------------------------------------------
def _appdir_path(filename: str) -> Path:
    try:
        base = appdata_dir() if callable(appdata_dir) else Path(appdata_dir)
    except Exception:
        base = Path(os.getenv("LOCALAPPDATA") or (Path.home() / ".local" / "share")) / "SOLOTradingBot"
    base.mkdir(parents=True, exist_ok=True)
    return base / filename

STATUS_FILE = _appdir_path("rugcheck_status.json")
FAILURES_FILE = _appdir_path("rugcheck_failures.json")

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Env defaults (config can override)
# ---------------------------------------------------------------------
def _env_bool(var: str, default: bool) -> bool:
    return os.getenv(var, str(default)).strip().lower() in ("1", "true", "yes", "on", "y")

RUGCHECK_ENABLE = _env_bool("RUGCHECK_ENABLE", True)
RUGCHECK_HARD_FAIL = _env_bool("RUGCHECK_HARD_FAIL", False)
BLOCK_RUGCHECK_DANGEROUS = _env_bool("BLOCK_RUGCHECK_DANGEROUS", True)

RUGCHECK_DISCOVERY_CHECK  = _env_bool("RUGCHECK_DISCOVERY_CHECK", False)
RUGCHECK_DISCOVERY_FILTER = _env_bool("RUGCHECK_DISCOVERY_FILTER", False)

# Raydium defaults — keep OFF by default while testing
RAYDIUM_ENABLE = _env_bool("RAYDIUM_ENABLE", False)

# Safe byte cap derivation (no NameError; robust to junk values)
# Priority: explicit bytes (RAYDIUM_MAX_BYTES) → MB (RAYDIUM_MAX_DOWNLOAD_MB) → default 40MB
_MIN_MB_FLOOR = 8
try:
    _ray_bytes_env = os.getenv("RAYDIUM_MAX_BYTES")
    if _ray_bytes_env is not None and _ray_bytes_env.strip():
        RAYDIUM_MAX_BYTES = max(1, int(_ray_bytes_env))
    else:
        _mb_env = os.getenv("RAYDIUM_MAX_DOWNLOAD_MB")
        _mb = int(_mb_env) if (_mb_env is not None and _mb_env.strip()) else 40
        _mb = max(_MIN_MB_FLOOR, _mb)
        RAYDIUM_MAX_BYTES = _mb * 1024 * 1024
except Exception:
    RAYDIUM_MAX_BYTES = 40 * 1024 * 1024  # final fallback


RAYDIUM_MAX_PAIRS = int(os.getenv("RAYDIUM_MAX_PAIRS", "100"))
# Start small to avoid oversize thrash; still overridable via env/config
RAYDIUM_PAGE_SIZE_ENV = int(os.getenv("RAYDIUM_PAGE_SIZE", "50"))
RAYDIUM_MAX_POOLS_ENV = int(os.getenv("RAYDIUM_MAX_POOLS", "1500"))

# Birdeye toggles — default OFF while testing
BIRDEYE_ENABLE_ENV = _env_bool("BIRDEYE_ENABLE", False)
BIRDEYE_MAX_TOKENS_ENV = int(os.getenv("BIRDEYE_MAX_TOKENS", "500"))

DEFAULT_SOLANA_TIMEOUT = int(os.getenv("SOLANA_RPC_TIMEOUT", "15"))
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")

DEXSCREENER_PAGES_ENV = int(os.getenv("DEX_PAGES") or os.getenv("DEXSCREENER_PAGES", "10"))
DEXSCREENER_PER_PAGE_ENV = int(os.getenv("DEX_PER_PAGE", "100"))
DEXSCREENER_MAX_ENV = int(os.getenv("DEX_MAX", os.getenv("DEXSCREENER_MAX", "0")))
DEXSCREENER_QUERY_ENV = os.getenv("DEXSCREENER_QUERY", "solana")
DEXSCREENER_QUERIES_ENV = [q.strip() for q in os.getenv("DEXSCREENER_QUERIES", "").split(",") if q.strip()]


# >>> WINDOWS DEX PATCH: UA + IPv4 controls ------------------------------------
DEX_USER_AGENT_ENV = os.getenv("DEX_USER_AGENT", "").strip()
DEFAULT_BROWSER_UA = (
    DEX_USER_AGENT_ENV
    or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)
DEX_FORCE_IPV4_ENV = os.getenv("DEX_FORCE_IPV4", "auto").strip().lower()

def _dex_force_ipv4() -> bool:
    # 'auto' forces IPv4 on Windows; override with 1/0 if desired.
    if DEX_FORCE_IPV4_ENV in ("1", "true", "yes", "on", "force"):
        return True
    if DEX_FORCE_IPV4_ENV in ("0", "false", "no", "off"):
        return False
    return os.name == "nt"

async def _make_dex_session() -> Tuple[aiohttp.ClientSession, Dict[str, str]]:
    """
    Dedicated Dexscreener session:
      • IPv4-only connector (on Windows by default)
      • Browser-like User-Agent
    """
    family = socket.AF_INET if _dex_force_ipv4() else 0
    connector = aiohttp.TCPConnector(
        family=family, limit=30, ttl_dns_cache=300, enable_cleanup_closed=True
    )
    timeout = aiohttp.ClientTimeout(total=30, sock_connect=10, sock_read=20)
    sess = aiohttp.ClientSession(connector=connector, timeout=timeout)
    headers = {"Accept": "application/json", "User-Agent": DEFAULT_BROWSER_UA}
    return sess, headers
# <<< WINDOWS DEX PATCH ---------------------------------------------------------

# --- Birdeye cooldown (simple exponential backoff) ---
_BIRDEYE_COOLDOWN_UNTIL = 0
_BIRDEYE_LAST_BACKOFF = 0


# ---- Quality thresholds (optional; leave 0 to disable) -----------------------
MIN_LIQUIDITY_USD   = float(os.getenv("MIN_LIQUIDITY_USD",   "0"))   # e.g. 10000
MIN_VOLUME24_USD    = float(os.getenv("MIN_VOLUME24_USD",    "0"))   # e.g. 5000
MIN_MARKETCAP_USD   = float(os.getenv("MIN_MARKETCAP_USD",   "0"))   # e.g. 500000

def _num(v, d=0.0) -> float:
    try:
        return float(v if v not in (None, "") else d)
    except Exception:
        return float(d)

def _mc(obj: Dict[str, Any]) -> float:
    # Try multiple common fields used by the three sources
    return _num(
        obj.get("market_cap")
        or obj.get("mc")
        or obj.get("fdv")
        or obj.get("fdvUsd")
        or obj.get("marketCap")
        or 0.0,
        0.0,
    )

def _better(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """
    Return True if 'a' should replace 'b' when de-duplicating.
    Priority: liquidity > volume_24h > market cap.
    """
    a_key = (_num(a.get("liquidity")), _num(a.get("volume_24h")), _mc(a))
    b_key = (_num(b.get("liquidity")), _num(b.get("volume_24h")), _mc(b))
    return a_key > b_key

def _passes_minimums(tok: Dict[str, Any]) -> bool:
    if MIN_LIQUIDITY_USD   and _num(tok.get("liquidity"))  < MIN_LIQUIDITY_USD:   return False
    if MIN_VOLUME24_USD    and _num(tok.get("volume_24h")) < MIN_VOLUME24_USD:    return False
    if MIN_MARKETCAP_USD   and _mc(tok)                    < MIN_MARKETCAP_USD:   return False
    return True



# ---------------------------------------------------------------------
# Cooperative shutdown
# ---------------------------------------------------------------------
from threading import Event as _ThreadEvent
_SHUTDOWN = _ThreadEvent()

def signal_shutdown() -> None:
    _SHUTDOWN.set()

def clear_shutdown_signal() -> None:
    _SHUTDOWN.clear()

def _should_stop() -> bool:
    if _SHUTDOWN.is_set():
        return True
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    return loop.is_closed()

# ---------------------------------------------------------------------
# Helpers to read config with sane fallbacks
# ---------------------------------------------------------------------
def _cfg() -> Dict[str, Any]:
    try:
        return load_config() or {}
    except Exception:
        return {}

def _disc(key: str, default: Any) -> Any:
    c = _cfg()
    return (c.get("discovery") or {}).get(key, default)

def _dex_queries() -> List[str]:
    q = _disc("dexscreener_queries", None)
    if isinstance(q, list) and q:
        return [str(x).strip() for x in q if str(x).strip()]
    if DEXSCREENER_QUERIES_ENV:
        return DEXSCREENER_QUERIES_ENV
    return [DEXSCREENER_QUERY_ENV]

def _dex_pages_perpage() -> Tuple[int, int]:
    pages = int(_disc("dexscreener_pages", DEXSCREENER_PAGES_ENV))
    per_page = int(_disc("dexscreener_per_page", DEXSCREENER_PER_PAGE_ENV))
    return max(1, pages), max(1, per_page)

def _dex_post_cap() -> int:
    cap = 0
    try:
        cap = int(_disc("dexscreener_post_cap", 0))
    except Exception:
        cap = 0
    if cap <= 0:
        try:
            cap = int(DEXSCREENER_MAX_ENV or 0)
        except Exception:
            cap = 0
    return max(0, cap)

def _ray_limits(max_pairs_hint: Optional[int] = None) -> Tuple[int, int]:
    """Returns tuple of (parse_cap, page_size) for Raydium API."""
    cfg_max_pairs = int(_disc("raydium_max_pairs", RAYDIUM_MAX_PAIRS))
    parse_cap = int(max_pairs_hint or cfg_max_pairs)
    parse_cap = min(parse_cap, int(_disc("raydium_max_pools", RAYDIUM_MAX_POOLS_ENV)))
    page_size = int(_disc("raydium_page_size", RAYDIUM_PAGE_SIZE_ENV))
    page_size = max(50, min(page_size, 250))  # Clamp
    return max(1, parse_cap), page_size

def _bird_max() -> int:
    try:
        v = int(_disc("birdeye_max_tokens", BIRDEYE_MAX_TOKENS_ENV))
        return max(0, v)
    except Exception:
        return max(0, BIRDEYE_MAX_TOKENS_ENV)

def _rc_discovery_switches() -> Tuple[bool, bool]:
    check = bool(_disc("rugcheck_in_discovery", RUGCHECK_DISCOVERY_CHECK))
    filt = bool(_disc("require_rugcheck_pass", RUGCHECK_DISCOVERY_FILTER))
    return check, filt

# ---------------------------------------------------------------------
# Fallback prices (DISABLED by default)
# ---------------------------------------------------------------------
FALLBACK_PRICES: Dict[str, Dict[str, float]] = {}
if os.getenv("FALLBACK_ENABLE", "0").lower() not in ("1", "true", "yes"):
    FALLBACK_PRICES.clear()

logger.warning(f"USING FETCHING FROM: {__file__}")
logger.warning("FALLBACK enabled? %s (count=%d)", bool(FALLBACK_PRICES), len(FALLBACK_PRICES))

# ---------------------------------------------------------------------
# In-memory failures & status
# ---------------------------------------------------------------------
RUGCHECK_FAILURES: List[Dict[str, str]] = []

def _write_status(enabled: bool, available: bool, message: str) -> None:
    try:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with STATUS_FILE.open("w", encoding="utf-8") as f:
            json.dump(
                {"enabled": enabled, "available": available, "message": message, "timestamp": int(time.time())},
                f, indent=2,
            )
    except Exception as e:
        logger.warning("Failed to write Rugcheck status: %s", e)

def _save_failures() -> None:
    try:
        FAILURES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with FAILURES_FILE.open("w", encoding="utf-8") as f:
            json.dump(RUGCHECK_FAILURES, f, indent=2)
    except Exception as e:
        logger.error("Failed to save Rugcheck failures: %s", e)

# NEW: seed the GUI banner immediately on startup (no network calls)
def ensure_rugcheck_status_file() -> None:
    """
    Seed the GUI's Rugcheck banner with an immediate status snapshot.
    We don't hit Rugcheck here; we only check whether Rugcheck is enabled
    and whether an Authorization header can be formed from the env.
    """
    try:
        # Determine enabled flag robustly even if the module-level constant isn't set yet
        if "RUGCHECK_ENABLE" in globals():
            enabled = bool(globals()["RUGCHECK_ENABLE"])
        else:
            enabled = str(os.getenv("RUGCHECK_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")

        try:
            # Prefer the light-weight header getter; safe even if it later refreshes the token.
            hdrs = get_rugcheck_headers() if "get_rugcheck_headers" in globals() else {}
        except Exception:
            hdrs = {}

        available = bool(hdrs.get("Authorization"))
        msg = "JWT configured" if available else "JWT missing (set RUGCHECK_JWT or RUGCHECK_JWT_TOKEN)"
        _write_status(enabled, available, msg)
    except Exception as e:
        # Never break discovery because of banner bookkeeping
        _write_status(False, False, f"status unavailable: {e}")

_MAX_FAILURES = 500

def _append_failure(addr: str, reason: str) -> None:
    RUGCHECK_FAILURES.append({"address": addr, "reason": reason})
    if len(RUGCHECK_FAILURES) > _MAX_FAILURES:
        del RUGCHECK_FAILURES[: len(RUGCHECK_FAILURES) - _MAX_FAILURES]
    _save_failures()

def log_error_with_stacktrace(message: str, error: Exception) -> None:
    logger.error("%s: %s\n%s", message, error, traceback.format_exc())
    
# -----------------------------------------------------------------------------
# Sanity tweak #1: normalize config with safe defaults (discovery/bot)
# -----------------------------------------------------------------------------
def _normalized_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    raw = raw or {}
    disc = dict(raw.get("discovery") or {})
    # default buckets
    def _b(defaults):
        b = dict(defaults)
        b.update(dict(disc.get(defaults["name"], {})))
        b.pop("name", None)
        return b

    # sensible defaults that won't stall discovery
    low_cap_defaults = {
        "name": "low_cap",
        "max_market_cap": 750_000,
        "liquidity_threshold": 10_000,
        "volume_threshold": 10_000,
        "max_rugcheck_score": 5000,
    }
    mid_cap_defaults = {
        "name": "mid_cap",
        "max_market_cap": 5_000_000,
        "liquidity_threshold": 50_000,
        "volume_threshold": 40_000,
        "max_rugcheck_score": 7000,
    }
    large_cap_defaults = {
        "name": "large_cap",
        "liquidity_threshold": 100_000,
        "volume_threshold": 100_000,
        "max_rugcheck_score": 9000,
    }
    new_defaults = {
        "name": "newly_launched",
        "max_token_age_minutes": 180,
        "liquidity_threshold": 5_000,
        "volume_threshold": 5_000,
        "max_rugcheck_score": 2000,
    }

    disc_out = {
        "low_cap": _b(low_cap_defaults),
        "mid_cap": _b(mid_cap_defaults),
        "large_cap": _b(large_cap_defaults),
        "newly_launched": _b(new_defaults),
        "max_price_change": float(disc.get("max_price_change", 85)),  # absolute %
    }

    bot = dict(raw.get("bot") or {})
    bot.setdefault("cycle_interval", 30)
    bot.setdefault("cooldown_seconds", 3)
    bot.setdefault("dry_run", False)

    out = dict(raw)
    out["discovery"] = disc_out
    out["bot"] = bot
    return out

# -----------------------------------------------------------------------------
# Sanity tweak #2: clamp discovery/fetch limits to safe ranges
# -----------------------------------------------------------------------------
def _clamp_discovery_limits(cfg: Dict[str, Any]) -> None:
    # Dexscreener
    disc = cfg.setdefault("discovery", {})
    dex = disc.setdefault("dexscreener", {})
    dex.setdefault("pages", 2)
    dex.setdefault("per_page", 100)
    dex.setdefault("max", 0)
    dex["pages"] = max(1, min(int(dex.get("pages", 2)), 3))
    dex["per_page"] = max(50, min(int(dex.get("per_page", 100)), 150))
    dex["max"] = max(0, min(int(dex.get("max", 0)), 1000))

    # Raydium
    ray = disc.setdefault("raydium", {})
    ray.setdefault("max_pairs", 100)
    ray.setdefault("max_download_mb", 40)  # small to avoid stalls on Windows
    ray["max_pairs"] = max(25, min(int(ray.get("max_pairs", 100)), 250))
    ray["max_download_mb"] = max(16, min(int(ray.get("max_download_mb", 40)), 85))

    # Birdeye
    be = disc.setdefault("birdeye", {})
    be.setdefault("max_tokens", 250)
    be["max_tokens"] = max(50, min(int(be.get("max_tokens", 250)), 750))
    

# ---------------------------------------------------------------------
# Rugcheck header helpers
# ---------------------------------------------------------------------
try:
    try:
        from .rugcheck_auth import ensure_valid_rugcheck_headers as _rc_hdrs
    except Exception:
        _rc_hdrs = None
    from .rugcheck_auth import get_rugcheck_headers
except Exception:
    try:
        try:
            from solana_trading_bot_bundle.trading_bot.rugcheck_auth import ensure_valid_rugcheck_headers as _rc_hdrs
        except Exception:
            _rc_hdrs = None
        from solana_trading_bot_bundle.trading_bot.rugcheck_auth import get_rugcheck_headers
    except Exception:
        _rc_hdrs = None
        def get_rugcheck_headers() -> Dict[str, str]:
            tok = (os.getenv("RUGCHECK_JWT_TOKEN") or os.getenv("RUGCHECK_JWT") or os.getenv("RUGCHECK_API_TOKEN") or "")
            return {"Authorization": f"Bearer {tok}"} if tok else {}
# ---------------------------------------------------------------------
# Streaming JSON helper (defensive, size-capped)
# ---------------------------------------------------------------------
async def _safe_json_stream(
    resp: aiohttp.ClientResponse,
    max_bytes: Optional[int] = None,
    chunk_size: int = 262_144,  # 256 KiB chunks keep memory flatter
) -> Any:
    """
    Stream-response -> JSON with a hard byte cap.

    - Aborts early if Content-Length exceeds cap.
    - Streams in chunks; raises MemoryError if cap is exceeded mid-stream.
    - Returns {} on empty/204 bodies.
    - Tries a second decode pass with explicit UTF-8 if the first fails.
    """
    # --- resolve byte cap safely (no NameError if constant is missing)
    try:
        default_cap = int(RAYDIUM_MAX_BYTES)  # type: ignore[name-defined]
    except Exception:
        default_cap = 40 * 1024 * 1024  # 40 MiB final fallback

    limit = int(max_bytes) if max_bytes is not None else default_cap
    if limit < 1:
        limit = default_cap

    # --- hard fail on clearly-oversize content-length
    cl = resp.content_length
    if cl is not None and cl > limit:
        # Make sure to release the connection pool slot ASAP
        try:
            await resp.release()
        finally:
            pass
        raise MemoryError(f"Response too large: {cl} bytes > {limit}")

    # --- stream the body under the cap
    buf = bytearray()
    try:
        async for chunk in resp.content.iter_chunked(chunk_size):
            if not chunk:
                continue
            buf.extend(chunk)
            if len(buf) > limit:
                # Stop reading more data; free socket promptly
                try:
                    await resp.release()
                finally:
                    pass
                raise MemoryError(f"Response exceeded {limit} bytes")
    except asyncio.CancelledError:
        # Propagate cancellations cleanly (task shutdown, etc.)
        raise
    except Exception:
        # Ensure the connection is not left hanging on unexpected errors
        try:
            await resp.release()
        finally:
            pass
        raise

    # --- empty body handling
    if not buf or resp.status == 204:
        return {}

    # --- decode JSON (two attempts)
    try:
        return json.loads(buf)
    except json.JSONDecodeError:
        # Retry with explicit UTF-8 string decode (tolerant replacement)
        try:
            text = buf.decode("utf-8", errors="replace")
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON ({e}) from {len(buf)} bytes") from e

# ---------------------------------------------------------------------
# Rugcheck validation (boolean for trading-time checks)
# ---------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=30))
async def validate_rugcheck(token_address: str, session: aiohttp.ClientSession) -> bool:
    if not RUGCHECK_ENABLE:
        _write_status(False, False, "Rugcheck disabled by config; tokens allowed")
        return True
    if token_address in WHITELISTED_TOKENS:
        _write_status(True, True, "Rugcheck active (whitelist bypass)")
        return True
    try:
        headers = await _rc_hdrs(session, force_refresh=False) if _rc_hdrs else get_rugcheck_headers()
    except Exception:
        headers = {}
    if not headers.get("Authorization"):
        msg = "Rugcheck unavailable: missing/invalid JWT"
        logger.warning(f"{msg} for {token_address}")
        _append_failure(token_address, msg)
        _write_status(True, False, msg)
        return not RUGCHECK_HARD_FAIL
    try:
        async def _fetch_report(hdrs: Dict[str, str]):
            async with session.get(
                f"https://api.rugcheck.xyz/v1/tokens/{token_address}/report",
                headers=hdrs,
                timeout=aiohttp.ClientTimeout(total=12, sock_connect=6, sock_read=8),
            ) as r:
                if r.status != 200:
                    return r.status, None
                return r.status, await r.json(content_type=None)
        status, data = await _fetch_report(headers)
        if status in (401, 403) and _rc_hdrs:
            try:
                headers = await _rc_hdrs(session, force_refresh=True) or headers
            except Exception:
                pass
            if headers.get("Authorization"):
                status, data = await _fetch_report(headers)
        if data is None:
            msg = f"Rugcheck HTTP {status}"
            logger.warning(f"{msg} for {token_address}")
            _append_failure(token_address, msg)
            _write_status(True, False, msg)
            return not RUGCHECK_HARD_FAIL
        risk_level = data.get("risk_level") or (data.get("risk", {}) or {}).get("label", "unknown")
        if BLOCK_RUGCHECK_DANGEROUS and str(risk_level).lower() in ("high", "medium"):
            msg = f"High/Medium risk ({risk_level})"
            logger.warning(f"Rugcheck blocked {token_address}: {msg}")
            _append_failure(token_address, msg)
            _write_status(True, True, "Rugcheck active")
            return False
        _write_status(True, True, "Rugcheck active")
        return True
    except (aiohttp.ClientError, ValueError) as e:
        msg = f"Rugcheck error: {e}"
        logger.warning(f"{msg} for {token_address}")
        _append_failure(token_address, str(e))
        _write_status(True, False, msg)
        return not RUGCHECK_HARD_FAIL

# NEW: lightweight annotator used during discovery (config-aware)
async def _annotate_rugcheck_fields(token_address: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
    safety = "unknown"
    dangerous = False
    check, filt = _rc_discovery_switches()
    if RUGCHECK_ENABLE and check:
        try:
            ok = await validate_rugcheck(token_address, session)
            safety = "ok" if ok else "dangerous"
            dangerous = not ok
            if (not ok) and filt and token_address not in WHITELISTED_TOKENS:
                return {"drop": True}
        except Exception:
            safety = "unknown"
            dangerous = False
    return {"safety": safety, "dangerous": dangerous}

# ---------------------------------------------------------------------
# Dexscreener (search)
#  - Classic paged version (stable, predictable, fewer duplicates)
#  - Auto-fallback to sharded search with alt-queries if classic is sparse
#  Both paths DEDUPE BY BASE TOKEN ADDRESS before returning.
#  IMPORTANT: Solana mints are case-sensitive — we DO NOT lowercase them.
# ---------------------------------------------------------------------
from string import ascii_lowercase, digits

#   DEX_USE_SHARDS=false  -> we still try classic first, and only shard if classic is sparse
#   DEX_USE_SHARDS=true   -> (kept for back-compat; wrapper no longer depends on it strictly)
DEX_USE_SHARDS_ENV = os.getenv("DEX_USE_SHARDS", "false").lower() in ("1", "true", "yes", "on")


def _prefer(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Return the better row between a and b using (liquidity, vol24h, fdv/mcap)."""
    def f(x, k, d=0.0):
        try:
            return float((x or {}).get(k, d) or d)
        except Exception:
            return float(d)

    liq_a, liq_b = f(a, "liquidity"), f(b, "liquidity")
    if liq_a != liq_b:
        return a if liq_a > liq_b else b

    vol_a, vol_b = f(a, "volume_24h"), f(b, "volume_24h")
    if vol_a != vol_b:
        return a if vol_a > vol_b else b

    # market_cap already normalized in row builder; still guard for alternates
    def mc(x):
        try:
            return float(
                (x.get("market_cap")
                 or x.get("mc")
                 or x.get("fdv")
                 or x.get("fdvUsd")
                 or x.get("marketCap")
                 or 0.0)
            )
        except Exception:
            return 0.0

    return a if mc(a) >= mc(b) else b


def _dedupe_by_base_address(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collapse duplicates by base token address, keeping best row.
    Preserve original Solana address casing (case-sensitive base58).
    Also keep both 'address' and 'token_address' populated for downstream code.
    """
    best: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        addr = (r.get("address") or r.get("token_address") or "").strip()  # NO .lower()
        if not addr:
            continue
        prev = best.get(addr)
        best[addr] = r if prev is None else _prefer(prev, r)

    out: List[Dict[str, Any]] = []
    for r in best.values():
        a = (r.get("address") or r.get("token_address") or "").strip()
        r["address"] = a
        r["token_address"] = a
        out.append(r)
    return out


def _dex_query_shards(pages_to_fetch: int) -> List[str]:
    """
    Generate shard suffixes (" a", " b", ...) to broaden Dexscreener search.
    Front-load vowels + 't' then the rest + digits to bias early, common results.
    """
    base = list("aeioubt") + [c for c in ascii_lowercase if c not in "aeioubt"] + list(digits)
    n = max(1, int(pages_to_fetch))
    return [f" {s}" for s in base[:n]]


# ---------- Windows-friendly browser-like headers ----------
def _default_dex_headers(overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    ua = os.getenv("DEX_USER_AGENT", "").strip() or (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
    )
    hdrs = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://dexscreener.com",
        "Referer": "https://dexscreener.com/",
        "Connection": "keep-alive",
        "User-Agent": ua,
    }
    if overrides:
        hdrs.update(overrides)
    return hdrs


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=6, max=120),
    retry=retry_if_exception_type((aiohttp.ClientError, ValueError, asyncio.TimeoutError)),
)
async def fetch_dexscreener_search_classic(
    session: aiohttp.ClientSession,
    query: str = DEXSCREENER_QUERY_ENV,
    pages: Optional[int] = None,
    per_page: Optional[int] = None,
    headers: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Classic Dexscreener search using sequential ?page= pagination.
    Dedupes by base address. Uses browser-like headers and tries both URL variants.
    """
    cfg_pages, cfg_per_page = _dex_pages_perpage()
    total_pages = max(1, int(pages) if pages is not None else cfg_pages)
    per = max(1, min(int(per_page) if per_page is not None else cfg_per_page, 200))

    # try without and with trailing slash — observed to behave differently behind some CDNs
    bases = [
        "https://api.dexscreener.com/latest/dex/search",
        "https://api.dexscreener.com/latest/dex/search/",
    ]
    hdrs = _default_dex_headers(headers)
    timeout = aiohttp.ClientTimeout(total=30, sock_connect=10, sock_read=20)

    rows: List[Dict[str, Any]] = []
    now_ms = int(time.time() * 1000)
    sol_launch_ms = 1581465600000  # 2020-02-12T00:00:00Z

    for page in range(1, total_pages + 1):
        if _should_stop():
            break
        got_page = False

        for base in bases:
            url = f"{base}?q={query}&page={page}"
            try:
                async with session.get(url, headers=hdrs, timeout=timeout) as resp:
                    if resp.status == 429:
                        ra = resp.headers.get("Retry-After") or resp.headers.get("X-RateLimit-Reset")
                        sleep_s = float(ra) - time.time() if ra and str(ra).isdigit() else 15.0
                        await asyncio.sleep(max(5.0, sleep_s))
                        continue
                    if resp.status != 200:
                        body = (await resp.text())[:300]
                        logger.warning("Dexscreener page=%d HTTP %s: %s", page, resp.status, body)
                        continue

                    data = await resp.json(content_type=None)
                    pairs = (data.get("pairs") or [])[:per]
                    for pair in pairs:
                        if pair.get("chainId") != "solana":
                            continue
                        token_address = (pair.get("baseToken") or {}).get("address")
                        if not token_address:
                            continue
                        try:
                            Pubkey.from_string(token_address)
                        except Exception:
                            continue
                        token_address = token_address.strip()

                        created = pair.get("pairCreatedAt") or 0
                        if not (sol_launch_ms <= (created or 0) <= now_ms):
                            created = 0

                        rows.append({
                            "address": token_address,
                            "token_address": token_address,
                            "symbol": (pair.get("baseToken") or {}).get("symbol", "UNKNOWN"),
                            "name":   (pair.get("baseToken") or {}).get("name", "UNKNOWN"),
                            "volume_24h": float((pair.get("volume") or {}).get("h24", 0) or 0),
                            "liquidity":  float((pair.get("liquidity") or {}).get("usd", 0) or 0),
                            "market_cap": float(pair.get("fdv", 0) or pair.get("marketCap", 0) or 0),
                            "creation_timestamp": (created // 1000) if created else 0,
                            "timestamp": int(time.time()),
                            "categories": ["no_creation_time"] if created == 0 else [],
                            "price": float(pair.get("priceUsd") or 0),
                            "priceChange": pair.get("priceChange") or {},
                            "pair_address": pair.get("pairAddress"),
                            "source": "dexscreener",
                        })
                    got_page = True
                    break  # success on one base variant
            except (aiohttp.ClientError, ValueError, asyncio.TimeoutError) as e:
                logger.warning("Dexscreener page=%d fetch failed (%s): %s", page, base, e)
                continue

        # small politeness delay (helps with bot heuristics on Windows)
        await asyncio.sleep(0.25 if got_page else 0.6)

    deduped = _dedupe_by_base_address(rows)
    cap = _dex_post_cap()
    if cap and len(deduped) > cap:
        deduped = deduped[:cap]
    logger.info("Dexscreener classic: %d unique", len(deduped))
    return deduped


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=10, max=180),
    retry=retry_if_exception_type((aiohttp.ClientError, ValueError)),
)
async def fetch_dexscreener_search(
    session: aiohttp.ClientSession,
    query: str = DEXSCREENER_QUERY_ENV,
    pages: Optional[int] = None,
    headers: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Wrapper: try classic first; if too few uniques, fallback to sharded search
    over multiple alt queries and merge (dedup by base address).
    """
    hdrs = _default_dex_headers(headers)
    cfg_pages, cfg_per_page = _dex_pages_perpage()
    pages_to_fetch = int(pages) if pages is not None else cfg_pages
    per_page_cap = cfg_per_page

    # --- 1) Classic pass ---
    deduped = await fetch_dexscreener_search_classic(
        session, query=query, pages=pages_to_fetch, per_page=per_page_cap, headers=hdrs
    )

    # If classic yielded enough, stop here
    fallback_min = int(os.getenv("DEX_FALLBACK_MIN_UNIQUES", "10"))
    if len(deduped) >= fallback_min:
        return deduped

    # --- 2) Fallback: sharded search over alt queries ---
    # Keep this gentle to avoid rate limits, but broad enough to escape filters.
    alt_env = [q.strip() for q in os.getenv("DEXSCREENER_ALT_QUERIES", "").split(",") if q.strip()]
    alt_defaults = [query, f"{query} ", "sol", "sola", "a", "e", "i", "o", "u", "t"]
    alt = (alt_env or alt_defaults)[:8]  # cap attempts

    logger.info("Dexscreener fallback: trying shards/alt queries: %r", alt)

    shards = _dex_query_shards(max(1, pages_to_fetch))
    timeout = aiohttp.ClientTimeout(total=30, sock_connect=10, sock_read=20)

    async def fetch_shard(q: str, sh: str) -> List[Dict[str, Any]]:
        url = f"https://api.dexscreener.com/latest/dex/search?q={q}{sh}"
        try:
            async with session.get(url, headers=hdrs, timeout=timeout) as resp:
                if resp.status == 429:
                    # gentle throttling; skip this shard
                    await asyncio.sleep(1.2)
                    return []
                if resp.status != 200:
                    return []
                data = await resp.json(content_type=None)
                pairs = (data.get("pairs") or [])[:per_page_cap]
                now_ms = int(time.time() * 1000)
                sol_launch_ms = 1581465600000
                out: List[Dict[str, Any]] = []
                for pair in pairs:
                    if pair.get("chainId") != "solana":
                        continue
                    token_address = (pair.get("baseToken") or {}).get("address")
                    if not token_address:
                        continue
                    try:
                        Pubkey.from_string(token_address)
                    except Exception:
                        continue
                    token_address = token_address.strip()
                    created = pair.get("pairCreatedAt") or 0
                    if not (sol_launch_ms <= created <= now_ms):
                        created = 0
                    out.append({
                        "address": token_address,
                        "token_address": token_address,
                        "symbol": (pair.get("baseToken") or {}).get("symbol", "UNKNOWN"),
                        "name":   (pair.get("baseToken") or {}).get("name", "UNKNOWN"),
                        "volume_24h": float((pair.get("volume") or {}).get("h24", 0) or 0),
                        "liquidity":  float((pair.get("liquidity") or {}).get("usd", 0) or 0),
                        "market_cap": float(pair.get("fdv", 0) or pair.get("marketCap", 0) or 0),
                        "creation_timestamp": (created // 1000) if created else 0,
                        "timestamp": int(time.time()),
                        "categories": ["no_creation_time"] if created == 0 else [],
                        "price": float(pair.get("priceUsd") or 0),
                        "priceChange": pair.get("priceChange") or {},
                        "pair_address": pair.get("pairAddress"),
                        "source": "dexscreener",
                    })
                return out
        except (aiohttp.ClientError, ValueError, asyncio.TimeoutError):
            return []

    # Moderate parallelism across alt queries × shards
    tasks = [fetch_shard(q, sh) for q in alt for sh in shards]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    extra: List[Dict[str, Any]] = []
    for r in results:
        if isinstance(r, list):
            extra.extend(r)

    merged = _dedupe_by_base_address(deduped + extra)
    cap = _dex_post_cap()
    if cap and len(merged) > cap:
        merged = merged[:cap]

    return merged

# ---------------------------------------------------------------------
# Raydium (pairs) — streaming early-stop
#   • GET /v2/main/pairs (no paging)
#   • 30s timeout
#   • Byte budget via RAYDIUM_READ_BYTE_BUDGET (default 16 MiB)
#   • Unique-token cap via RAYDIUM_PARSE_CAP (default 300)
#   • Per-address de-duplication selecting best (liq, vol24h, mc)
#   • Obeys FORCE_DISABLE_RAYDIUM / is_enabled_raydium() / RAYDIUM_ENABLE
# ---------------------------------------------------------------------
async def fetch_raydium_tokens(
    session: aiohttp.ClientSession,
    solana_client: AsyncClient,  # kept for signature parity
    max_pairs: Optional[int] = None,
    *,
    max_download_bytes: Optional[int] = None,
) -> List[Dict[str, Any]]:
    # --- Feature flags: bail out early if disabled ---
    try:
        if 'FORCE_DISABLE_RAYDIUM' in globals() and FORCE_DISABLE_RAYDIUM:
            logger.info("Raydium disabled by FORCE_DISABLE_RAYDIUM — skipping fetch.")
            return []
        if 'is_enabled_raydium' in globals():
            try:
                enabled = is_enabled_raydium()  # some builds take 0 args
            except TypeError:
                # fallback: some builds take (config); else read env
                enabled = os.getenv("RAYDIUM_ENABLE", "false").strip().lower() not in ("0", "false", "no", "off")
            if not enabled:
                logger.info("Raydium disabled by config/env — skipping fetch.")
                return []
        else:
            if os.getenv("RAYDIUM_ENABLE", "false").strip().lower() in ("0", "false", "no", "off"):
                logger.info("Raydium disabled via env/config — skipping fetch.")
                return []
    except Exception:
        logger.info("Raydium disabled (flag error fallback) — skipping fetch.")
        return []

    # ---- helpers ----
    def _num(v, d: float = 0.0) -> float:
        try:
            return float(v if v not in (None, "") else d)
        except Exception:
            return float(d)

    def _better_row(new_t: Dict[str, Any], old_t: Dict[str, Any]) -> bool:
        """Prefer higher (liquidity, volume_24h, market_cap)."""
        a = (_num(new_t.get("liquidity")), _num(new_t.get("volume_24h")), _num(new_t.get("market_cap")))
        b = (_num(old_t.get("liquidity")), _num(old_t.get("volume_24h")), _num(old_t.get("market_cap")))
        return a > b

    # Light quality gates (envs; default to 0)
    min_liq = float(os.getenv("MIN_LIQUIDITY_USD", "0"))
    min_vol = float(os.getenv("MIN_VOLUME24_USD", "0"))
    min_mc  = float(os.getenv("MIN_MARKETCAP_USD", "0"))

    def _passes_minimums(liq: float, vol: float, mc: float) -> bool:
        return (
            (not min_liq or liq >= min_liq) and
            (not min_vol or vol >= min_vol) and
            (not min_mc  or mc  >= min_mc)
        )

    # Unique-token cap
    try:
        parse_cap, _ = _ray_limits(max_pairs_hint=max_pairs)  # if present in your codebase
        parse_cap = int(parse_cap or 0)
    except Exception:
        parse_cap = int(max_pairs or 0)
    if parse_cap <= 0:
        parse_cap = int(os.getenv("RAYDIUM_PARSE_CAP", "300"))

    # Byte budget (streaming); max_download_bytes takes precedence
    if max_download_bytes is not None:
        byte_budget = int(max_download_bytes)
    else:
        # prefer env; fallback 16 MiB
        byte_budget = int(os.getenv("RAYDIUM_READ_BYTE_BUDGET", str(16 * 1024 * 1024)))

    url = "https://api.raydium.io/v2/main/pairs"
    headers = {"Accept": "application/json", "User-Agent": "SOLOTradingBot/1.0"}
    timeout = aiohttp.ClientTimeout(total=30, sock_connect=10, sock_read=20)

    # ---- streaming array-of-objects parser (no extra deps) ----
    async def _iter_top_level_objects_from_stream(resp: aiohttp.ClientResponse, *, budget: int):
        """
        Yields dict objects from a JSON like: [ {...}, {...}, ... ]
        Stops if 'budget' bytes consumed.
        """
        buf = bytearray()
        consumed = 0
        depth = 0
        in_string = False
        escape = False
        saw_array_start = False
        obj_start = None  # index where current object starts in buf

        async for chunk in resp.content.iter_chunked(65536):
            buf.extend(chunk)
            consumed += len(chunk)
            if consumed > budget:
                break

            i = 0
            while i < len(buf):
                c = chr(buf[i])

                if not saw_array_start:
                    if c == '[':
                        saw_array_start = True
                    i += 1
                    continue

                if obj_start is None:
                    if c == '{':
                        obj_start = i
                        depth = 1
                        in_string = False
                        escape = False
                    i += 1
                    continue

                if in_string:
                    if escape:
                        escape = False
                    elif c == '\\':
                        escape = True
                    elif c == '"':
                        in_string = False
                else:
                    if c == '"':
                        in_string = True
                    elif c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            obj_bytes = bytes(buf[obj_start:i+1])
                            try:
                                yield json.loads(obj_bytes)
                            except Exception:
                                pass
                            # drop processed bytes
                            del buf[:i+1]
                            i = -1
                            obj_start = None
                i += 1
        # ignore trailing partials

    # ---- request + stream ----
    best_by_addr: Dict[str, Dict[str, Any]] = {}
    now_ts = int(time.time())

    try:
        async with session.get(url, headers=headers, timeout=timeout) as resp:
            if resp.status == 429:
                reset = resp.headers.get("X-RateLimit-Reset")
                try:
                    to_sleep = max(5.0, float(reset) - time.time()) if reset else 15.0
                except Exception:
                    to_sleep = 15.0
                logger.warning("Raydium 429; sleeping %.1fs", to_sleep)
                await asyncio.sleep(to_sleep)
                return []

            if resp.status != 200:
                body = (await resp.text())[:500]
                logger.error("Raydium HTTP %s: %s", resp.status, body)
                return []

            # Stream objects and stop early when we have enough uniques
            async for p in _iter_top_level_objects_from_stream(resp, budget=byte_budget):
                # Extract base token address robustly
                addr = (
                    ((p.get("base") or {}).get("address"))
                    or p.get("baseMint")
                    or p.get("mint")
                    or ""
                )
                addr = (addr or "").strip()
                if not addr:
                    continue
                try:
                    Pubkey.from_string(addr)
                except Exception:
                    continue

                # Metrics
                def _vol24(obj: Dict[str, Any]) -> float:
                    return _num(obj.get("volume24h") or (obj.get("volume") or {}).get("h24") or 0)

                liq = _num(p.get("liquidity") or (p.get("liquidity") or {}).get("usd") or 0)
                vol = _vol24(p)
                mc  = _num(p.get("fdv") or p.get("marketCap") or 0)

                if not _passes_minimums(liq, vol, mc):
                    continue

                # created time (ms) -> seconds
                created_ms = None
                for k in ("createdAt", "createdTimestamp", "createdTime", "timestamp", "created_at"):
                    v = p.get(k)
                    if v is not None:
                        try:
                            v = float(v)
                            if v and v < 10_000_000_000:
                                v *= 1000.0
                            created_ms = v
                            break
                        except Exception:
                            pass
                creation_timestamp = int(created_ms / 1000) if created_ms else 0

                symbol = (
                    (p.get("base") or {}).get("symbol")
                    or (p.get("name") or "").split("/")[0]
                    or p.get("symbol")
                    or "UNKNOWN"
                )
                name = (
                    (p.get("base") or {}).get("name")
                    or p.get("name")
                    or symbol
                )

                tok = {
                    "address": addr,
                    "symbol": symbol,
                    "name": name,
                    "volume_24h": vol,
                    "liquidity": liq,
                    "market_cap": mc,
                    "creation_timestamp": creation_timestamp,
                    "timestamp": now_ts,
                    "price": _num(p.get("price") or 0),
                    "source": "raydium",
                }

                prev = best_by_addr.get(addr)
                if prev is None or _better_row(tok, prev):
                    best_by_addr[addr] = tok

                if len(best_by_addr) >= parse_cap:
                    break  # EARLY STOP

    except asyncio.TimeoutError:
        logger.warning("Raydium request timed out.")
        return []
    except MemoryError as e:
        logger.warning("Raydium response exceeded safe size: %s", e)
        return []
    except (aiohttp.ClientError, ValueError) as e:
        logger.error("Raydium error: %s", e, exc_info=True)
        return []

    tokens = list(best_by_addr.values())
    logger.info(
        "Raydium streamed: %d unique tokens (cap=%d, budget=%d bytes)",
        len(tokens), parse_cap, byte_budget
    )
    return tokens



## ---------------------------------------------------------------------
# Birdeye (tokens) — public-first (minimal headers), per-host backoff,
#                    IPv6→IPv4 fallback, tiny-probe-first, 30s cache,
#                    success floor with auto-fill
# ---------------------------------------------------------------------
def _bd_num(x, default=0.0) -> float:
    try:
        return float(x if x not in (None, "") else default)
    except Exception:
        return float(default)

def _rl_from_headers(h) -> float:
    """
    Read Birdeye rate-limit headers and return seconds to wait.
    Prefers x-ratelimit-reset (epoch seconds), falls back to retry-after.
    """
    try:
        reset = float(h.get("x-ratelimit-reset", "0"))
        now = time.time()
        if reset > now:  # header is epoch seconds
            return max(0.8, reset - now)  # small cushion
    except Exception:
        pass
    try:
        ra = float(h.get("retry-after", "0"))
        return max(0.8, ra)
    except Exception:
        return 0.0

# per-host circuit breaker & backoff state
try:
    _BIRDEYE_HOST_BLOCK_UNTIL
except NameError:
    _BIRDEYE_HOST_BLOCK_UNTIL = {}  # {host: unix_ts}

# ensure global cooldown vars exist
try:
    _BIRDEYE_COOLDOWN_UNTIL
except NameError:
    _BIRDEYE_COOLDOWN_UNTIL = 0.0
try:
    _BIRDEYE_LAST_BACKOFF
except NameError:
    _BIRDEYE_LAST_BACKOFF = 0.0

# probe + paging controls (bumped defaults)
BIRDEYE_TINY_PROBE_LIMIT = 5            # small seed to avoid zero
BIRDEYE_PAGE_LIMIT        = 12          # gentle but bigger pages
BIRDEYE_MAX_PAGES_PRIMARY = 4           # main pass
BIRDEYE_MAX_PAGES_REFILL  = 3           # extra pass if below floor

async def _ipv_session(force_ipv4: bool = False, force_ipv6: bool = False) -> aiohttp.ClientSession:
    fam = 0
    if force_ipv4:
        fam = socket.AF_INET
    elif force_ipv6:
        fam = socket.AF_INET6
    conn = aiohttp.TCPConnector(family=fam or 0, ttl_dns_cache=300)
    return aiohttp.ClientSession(connector=conn)

@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
)
async def fetch_birdeye_tokens(
    session: aiohttp.ClientSession,
    solana_client: AsyncClient,
    max_tokens: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if _should_stop():
        return []

    cfg_max = _bird_max()
    if not (BIRDEYE_ENABLE_ENV and cfg_max != 0):
        logger.info("Birdeye disabled via config/env; skipping.")
        return []

    global _BIRDEYE_COOLDOWN_UNTIL, _BIRDEYE_LAST_BACKOFF, _BIRDEYE_HOST_BLOCK_UNTIL
    now = time.time()
    if now < _BIRDEYE_COOLDOWN_UNTIL:
        logger.warning("Birdeye API on cooldown, skipping this cycle.")
        return []

    api_key = os.getenv("BIRDEYE_API_KEY")
    if not api_key:
        logger.warning("BIRDEYE_API_KEY not set; returning 0 Birdeye tokens.")
        return []

    target = max(1, int(max_tokens if max_tokens is not None else cfg_max))
    page_limit = min(BIRDEYE_PAGE_LIMIT, target)
    max_pages  = min(BIRDEYE_MAX_PAGES_PRIMARY, (target + page_limit - 1) // page_limit)
    # require at least this many raw items before we call it a “pass”
    success_floor = min(max(12, page_limit), target)

    host_public  = "https://public-api.birdeye.so/defi/tokenlist"
    host_private = "https://api.birdeye.so/defi/tokenlist"   # flaky (521); used only if public gives nothing
    host_candidates = [host_public, host_private]

    # try preferred sort, but fall back to none
    sort_keys = ["v24hUSD", None]

    # **Minimal** headers (match curl success)
    headers_min = {
        "Accept": "application/json",
        "X-API-KEY": api_key,
        "User-Agent": "curl/8.4.0",
    }
    timeout = aiohttp.ClientTimeout(total=20, sock_connect=6, sock_read=10)

    # ------- 30s in-process cache -------
    _cache = getattr(fetch_birdeye_tokens, "_CACHE", {"until": 0.0, "items": []})
    if now < float(_cache.get("until", 0)) and _cache.get("items"):
        cached = _cache["items"]
        logger.debug("Birdeye: served %d items from 30s cache", len(cached))
        return cached[:target]

    # small jitter to avoid synchronized bursts
    await asyncio.sleep(0.08 + (time.time() % 0.12))

    items: List[Dict[str, Any]] = []
    last_status: Optional[int] = None
    last_host: Optional[str] = None
    last_headers: Dict[str, str] = {}
    offset = 0

    async def _do_get(base: str, params: dict, family: str = "normal"):
        """
        Single GET with optional IPv4/IPv6 connector.
        family ∈ {"normal","ipv6","ipv4"}
        """
        headers = headers_min
        if family == "normal":
            return await session.get(base, headers=headers, params=params, timeout=timeout)
        tmp = await _ipv_session(force_ipv6=(family == "ipv6"), force_ipv4=(family == "ipv4"))
        try:
            resp = await tmp.get(base, headers=headers, params=params, timeout=timeout)
            async def _closer():
                await tmp.close()
            setattr(resp, "close_parent", _closer)
            return resp
        except Exception:
            await tmp.close()
            raise

    async def _tiny_probe(limit: int = BIRDEYE_TINY_PROBE_LIMIT) -> List[Dict[str, Any]]:
        """
        Minimal call that mirrors your successful curl:
          - public host
          - small limit (default 5)
          - NO sort_by
          - IPv6 first, then IPv4
        """
        base = host_public
        params = {"chain": "solana", "limit": limit, "offset": 0}
        for fam in ("ipv6", "ipv4"):
            resp_ctx = None
            try:
                await asyncio.sleep(0.5)
                resp_ctx = await _do_get(base, params, family=fam)
                async with resp_ctx as resp:
                    txt = await resp.text()
                    if resp.status != 200:
                        logger.warning("Tiny-probe HTTP %s via %s; body[:120]=%r",
                                       resp.status, fam, txt[:120])
                        continue
                    try:
                        data = json.loads(txt)
                    except Exception:
                        data = await resp.json(content_type=None)
                    payload = data.get("data") or {}
                    li = payload.get("items") or payload.get("tokens") or []
                    return li if isinstance(li, list) else []
            except Exception as e:
                logger.warning("Tiny-probe exception via %s: %s", fam, e)
                continue
            finally:
                closer = getattr(resp_ctx, "close_parent", None)
                if callable(closer):
                    try:
                        await closer()
                    except Exception:
                        pass
        return []

    async def _force_fill_public(start_offset: int, need: int) -> int:
        """
        If we’re below the success floor, grab a couple of unsorted pages
        from the public host with gentle pacing.
        Returns number of new items added.
        """
        added = 0
        base = host_public
        off = start_offset
        for _ in range(BIRDEYE_MAX_PAGES_REFILL):
            params = {"offset": off, "limit": page_limit, "chain": "solana"}
            got = False
            for fam in ("normal", "ipv6", "ipv4"):
                resp_ctx = None
                try:
                    resp_ctx = await _do_get(base, params, family=fam)
                    async with resp_ctx as resp:
                        txt = await resp.text()
                        if resp.status == 200:
                            try:
                                data = json.loads(txt)
                            except Exception:
                                data = await resp.json(content_type=None)
                            payload = data.get("data") or {}
                            li = payload.get("items") or payload.get("tokens") or []
                            if isinstance(li, list) and li:
                                items.extend(li)
                                added += len(li)
                                got = True
                                break
                        elif resp.status == 429:
                            wait = _rl_from_headers(resp.headers) or 1.2
                            await asyncio.sleep(wait)
                            continue
                except Exception as e:
                    logger.debug("force_fill_public error via %s: %s", fam, e)
                finally:
                    closer = getattr(resp_ctx, "close_parent", None)
                    if callable(closer):
                        try:
                            await closer()
                        except Exception:
                            pass
            if not got:
                break
            off += page_limit
            if len(items) >= need:
                break
            await asyncio.sleep(1.2)
        return added

    # ---- Try tiny-probe up front to avoid “zero” result ----
    try:
        seed = await _tiny_probe()
        if seed:
            items.extend(seed)
            logger.info("Birdeye tiny-probe prefetch: captured %d item(s).", len(seed))
            await asyncio.sleep(1.2)  # respect ~1 req/sec ceiling after probe
    except Exception:
        pass

    # Main paged fetch
    for _page in range(max_pages):
        if _should_stop():
            break
        if len(items) >= target:
            break

        got_page = False
        for sort_by in sort_keys:
            if _should_stop():
                break

            params = {"offset": offset, "limit": page_limit, "chain": "solana"}
            if sort_by:
                params["sort_by"] = sort_by
                params["sort_type"] = "desc"

            page_captured = False

            # filter out circuit-broken hosts; prefer public; avoid private once we already have items
            usable_hosts = []
            now = time.time()
            for base in host_candidates:
                if base is host_private and items:
                    continue
                if now >= _BIRDEYE_HOST_BLOCK_UNTIL.get(base, 0.0):
                    usable_hosts.append(base)

            for base in usable_hosts:
                if _should_stop():
                    break

                # try normal → IPv6 → IPv4
                for fam in ("normal", "ipv6", "ipv4"):
                    resp_ctx = None
                    try:
                        resp_ctx = await _do_get(base, params, family=fam)
                        async with resp_ctx as resp:
                            body_text = await resp.text()
                            last_status = resp.status
                            last_host   = base
                            last_headers = {k.lower(): v for k, v in resp.headers.items()
                                            if k.lower() in ("x-ratelimit-limit","x-ratelimit-remaining","x-ratelimit-reset","retry-after")}

                            if resp.status == 401:
                                logger.error("Birdeye unauthorized (401) on %s — check key/plan.", base)
                                return []

                            if resp.status == 429 or (
                                resp.status == 200 and '"success":false' in body_text and "Too many requests" in body_text
                            ):
                                logger.warning("Birdeye 429 on %s; attempting tiny-probe (limit=%d, no sort)…",
                                               base, BIRDEYE_TINY_PROBE_LIMIT)
                                tiny = await _tiny_probe()
                                if isinstance(tiny, list) and tiny:
                                    items.extend(tiny)
                                    page_captured = True
                                    got_page = True
                                    logger.info("Birdeye tiny-probe succeeded post-429: captured %d item(s).", len(tiny))
                                    break  # exit family loop
                                # tiny-probe failed → set host/global backoff and move on
                                wait = _rl_from_headers(resp.headers) or 6.0
                                _BIRDEYE_HOST_BLOCK_UNTIL[base] = time.time() + max(45.0, wait * 5)
                                _BIRDEYE_LAST_BACKOFF = min(25, max(8, (_BIRDEYE_LAST_BACKOFF * 2) if _BIRDEYE_LAST_BACKOFF else 8))
                                _BIRDEYE_COOLDOWN_UNTIL = time.time() + max(_BIRDEYE_LAST_BACKOFF, wait)
                                logger.warning("Birdeye 429 persists; host paused ~%.0fs; global cool %.0fs. headers=%s",
                                               _BIRDEYE_HOST_BLOCK_UNTIL[base] - time.time(), _BIRDEYE_COOLDOWN_UNTIL - time.time(), last_headers)
                                await asyncio.sleep(min(1.5, wait))
                                break

                            if resp.status >= 500:
                                if fam != "ipv4":
                                    logger.warning("Birdeye %d on %s via %s; trying another family…", resp.status, base, fam)
                                    continue
                                _BIRDEYE_HOST_BLOCK_UNTIL[base] = time.time() + 600.0
                                logger.warning("Birdeye %d on %s; blocking host 10m.", resp.status, base)
                                await asyncio.sleep(0.6)
                                break

                            if resp.status == 400:
                                logger.warning("Birdeye 400 params=%s on %s; trying next host.", params, base)
                                break

                            if resp.status != 200:
                                logger.warning("Birdeye HTTP %d on %s; Body: %s", resp.status, base, body_text[:300])
                                break

                            # Parse JSON (tolerate wrong content-type)
                            try:
                                data = json.loads(body_text)
                            except Exception:
                                data = await resp.json(content_type=None)

                            if isinstance(data, dict) and data.get("success") is False:
                                break

                            payload = data.get("data") if isinstance(data, dict) else {}
                            page_items = (
                                (payload.get("items") or payload.get("tokens") or
                                 payload.get("list")  or payload.get("records"))
                                if isinstance(payload, dict) else []
                            )
                            if not isinstance(page_items, list):
                                logger.warning("Birdeye: unexpected JSON shape on %s params=%s", base, params)
                                break

                            items.extend(page_items)
                            page_captured = True
                            got_page = True
                            logger.debug("Birdeye page %d (%s via %s %s): %d items",
                                         _page + 1, sort_by or "default", base, fam, len(page_items))
                            break  # success

                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        logger.warning("Birdeye request failed for %s via %s: %s", base, fam, e)
                        continue
                    finally:
                        closer = getattr(resp_ctx, "close_parent", None)
                        if callable(closer):
                            try:
                                await closer()
                            except Exception:
                                pass

                if page_captured:
                    break  # host loop

            if page_captured:
                break  # sort loop

        if not got_page:
            break

        offset += page_limit
        await asyncio.sleep(1.1)  # gentle pace

    # If still below success floor, force-fill a couple of unsorted pages
    if len(items) < success_floor:
        need = success_floor
        added = await _force_fill_public(offset, need)
        logger.info("Birdeye auto-fill added %d item(s); raw total=%d (floor=%d).",
                    added, len(items), success_floor)

    # last-ditch: one more tiny-probe if still empty
    if not items:
        tiny = await _tiny_probe()
        if tiny:
            items.extend(tiny)

    if not items:
        if last_status is not None:
            logger.error("Birdeye tokenlist failed (last HTTP %s on %s; headers=%s); skipping Birdeye this cycle",
                         last_status, (last_host or "?"), last_headers)
        return []

    # --- Build tokens, dedupe by address, Rugcheck/creation time ---
    seen: set[str] = set()
    tokens: List[Dict[str, Any]] = []

    for it in items:
        if _should_stop():
            break
        try:
            token_address = (it.get("address") or "").strip()
            if not token_address:
                continue
            try:
                Pubkey.from_string(token_address)
            except Exception:
                continue

            rc = await _annotate_rugcheck_fields(token_address, session)
            if rc.get("drop"):
                continue
            if token_address in seen:
                continue
            seen.add(token_address)

            symbol = it.get("symbol") or "UNKNOWN"
            name   = it.get("name") or symbol
            vol24  = it.get("v24hUSD") or it.get("volume24hUSD") or it.get("v24h") or 0
            liq    = it.get("liquidity") or it.get("liquidityUSD") or 0
            mc     = it.get("mc") or it.get("marketCap") or it.get("market_cap") or 0
            price  = it.get("price") or it.get("priceUsd") or None

            try:
                created_dt = await fetch_birdeye_creation_time(token_address, session)
            except Exception:
                created_dt = None
            creation_ts = int(created_dt.timestamp()) if created_dt else 0

            tokens.append({
                "address": token_address,
                "symbol": symbol,
                "name": name,
                "volume_24h": _bd_num(vol24),
                "liquidity": _bd_num(liq),
                "market_cap": _bd_num(mc),
                "creation_timestamp": creation_ts,
                "timestamp": int(time.time()),
                "categories": ["no_creation_time"] if creation_ts == 0 else [],
                "links": (it.get("extensions") or {}).get("socials", []),
                "score": None,
                "price": _bd_num(price if price is not None else (FALLBACK_PRICES.get(token_address, {}).get("price", 0.0))),
                "price_change_1h": _bd_num(FALLBACK_PRICES.get(token_address, {}).get("price_change_1h", 0.0)),
                "price_change_6h": _bd_num(FALLBACK_PRICES.get(token_address, {}).get("price_change_6h", 0.0)),
                "price_change_24h": _bd_num(FALLBACK_PRICES.get(token_address, {}).get("price_change_24h", 0.0)),
                "source": "birdeye",
                "safety": rc.get("safety", "unknown"),
                "dangerous": bool(rc.get("dangerous", False)),
            })
        except Exception:
            continue

    if tokens:
        _BIRDEYE_LAST_BACKOFF = 0
        _BIRDEYE_COOLDOWN_UNTIL = 0
        # 30s cache
        fetch_birdeye_tokens._CACHE = {"until": time.time() + 30.0, "items": list(tokens)}

    # Success logging with floor
    if len(tokens) >= success_floor:
        logger.info("Birdeye PASS: fetched %d ≥ %d tokens (public-first, tiny-probe, paged+refill).",
                    len(tokens), success_floor)
    else:
        logger.warning("Birdeye SOFT-OK: fetched %d < %d tokens (rate/host limits likely).",
                       len(tokens), success_floor)

    return tokens[:target]




# ---------------------------------------------------------------------
# Birdeye creation time — minimal headers, respects cooldown & flags
# ---------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
)
async def fetch_birdeye_creation_time(
    token_address: str,
    session: aiohttp.ClientSession
) -> Optional[datetime]:
    global _BIRDEYE_COOLDOWN_UNTIL, _BIRDEYE_LAST_BACKOFF

    # --- Feature flags: skip entirely when disabled ---
    try:
        # Prefer central feature flags if available
        if 'FORCE_DISABLE_BIRDEYE' in globals() and FORCE_DISABLE_BIRDEYE:
            return None
        if 'is_enabled_birdeye' in globals():
            try:
                # Some versions accept (config); others are 0-arg reading env
                enabled = is_enabled_birdeye()  # type: ignore[call-arg]
            except TypeError:
                enabled = os.getenv("BIRDEYE_ENABLE", "false").strip().lower() not in ("0", "false", "no", "off")
            if not enabled:
                return None
        else:
            # Fallback to env if the helper isn't imported
            if os.getenv("BIRDEYE_ENABLE", "false").strip().lower() in ("0", "false", "no", "off"):
                return None
    except Exception:
        # On any unexpected flag error, be safe and skip
        return None

    if _should_stop():
        return None

    now = time.time()
    if now < _BIRDEYE_COOLDOWN_UNTIL:
        return None

    api_key = os.getenv("BIRDEYE_API_KEY")
    # Also bail if the key looks missing or obviously malformed
    if not api_key or token_address in WHITELISTED_TOKENS:
        return None

    # Basic sanity on token address
    try:
        Pubkey.from_string(token_address)
    except Exception:
        return None

    # Use cache if present
    cached_time = await get_cached_creation_time(token_address)
    if cached_time:
        return cached_time

    headers = {
        "Accept": "application/json",
        "X-API-KEY": api_key,
        "User-Agent": "curl/8.4.0",
    }
    timeout = aiohttp.ClientTimeout(total=12, sock_connect=6, sock_read=8)

    # Try both endpoints (Birdeye has moved things around over time)
    endpoints = [
        f"https://public-api.birdeye.so/defi/token_security?address={token_address}",
        f"https://public-api.birdeye.so/public/token_security?address={token_address}",
    ]

    data_obj: Optional[Dict[str, Any]] = None

    for url in endpoints:
        if _should_stop():
            return None
        try:
            async with session.get(url, headers=headers, timeout=timeout) as resp:
                text = await resp.text()

                # 429 — rate limited: exponential cool-down
                if resp.status == 429:
                    wait = _rl_from_headers(resp.headers) or 0.8
                    _BIRDEYE_LAST_BACKOFF = min(25, max(8, (_BIRDEYE_LAST_BACKOFF * 2) if _BIRDEYE_LAST_BACKOFF else 8))
                    _BIRDEYE_COOLDOWN_UNTIL = time.time() + max(_BIRDEYE_LAST_BACKOFF, wait)
                    await asyncio.sleep(min(1.5, wait))
                    raise aiohttp.ClientError("birdeye-429")

                # 401 — unauthorized: set a longer cool-down but don't spam logs
                if resp.status == 401:
                    _BIRDEYE_LAST_BACKOFF = max(_BIRDEYE_LAST_BACKOFF or 0, 20)
                    _BIRDEYE_COOLDOWN_UNTIL = time.time() + _BIRDEYE_LAST_BACKOFF
                    return None

                if resp.status != 200:
                    # try next endpoint
                    continue

                # Parse JSON (some responses have no/odd content-type header)
                try:
                    data_obj = json.loads(text)
                except Exception:
                    data_obj = await resp.json(content_type=None)

                # Soft errors in body
                if isinstance(data_obj, dict) and data_obj.get("success") is False:
                    msg = str(data_obj.get("message", "")).lower()
                    if "too many requests" in msg:
                        wait = _rl_from_headers(resp.headers) or 0.8
                        _BIRDEYE_LAST_BACKOFF = min(25, max(8, (_BIRDEYE_LAST_BACKOFF * 2) if _BIRDEYE_LAST_BACKOFF else 8))
                        _BIRDEYE_COOLDOWN_UNTIL = time.time() + max(_BIRDEYE_LAST_BACKOFF, wait)
                        await asyncio.sleep(min(1.5, wait))
                        raise aiohttp.ClientError("birdeye-soft-429")
                    data_obj = None
                    continue

                if isinstance(data_obj, dict):
                    break

        except (aiohttp.ClientError, asyncio.TimeoutError):
            data_obj = None
            continue

    if not isinstance(data_obj, dict):
        return None

    d = data_obj.get("data") or data_obj

    # Extract creation timestamp (support multiple possible keys)
    created_ms: Optional[float] = None
    for key in [
        "created_timestamp", "createdTimestamp", "createdTime",
        "created_at", "createdAt", "mintTime", "launchDate"
    ]:
        v = d.get(key) or (isinstance(d.get("score"), dict) and d["score"].get(key))
        if v is not None:
            try:
                created_ms = float(v)
                break
            except Exception:
                continue

    if created_ms is None:
        return None
    # If it looks like seconds, convert to ms
    if created_ms < 10_000_000_000:
        created_ms *= 1000.0

    # Sanity: 2020-02-12T00:00:00Z .. now
    now_ms = time.time() * 1000.0
    if not (1581465600000.0 <= created_ms <= now_ms):
        return None

    dt = datetime.fromtimestamp(created_ms / 1000.0, tz=timezone.utc)

    try:
        await cache_creation_time(token_address, dt)
    except Exception:
        pass

    # success — reset backoff
    _BIRDEYE_LAST_BACKOFF = 0
    _BIRDEYE_COOLDOWN_UNTIL = 0
    return dt


# ---------------------------------------------------------------------
# Dexscreener creation time
# ---------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
)
async def fetch_dexscreener_creation_time(
    token_address: str,
    session: aiohttp.ClientSession,
    headers: Optional[Dict[str, str]] = None,   # <- allow caller to pass browser-ish headers
) -> Optional[datetime]:
    if _should_stop():
        return None
    if token_address in WHITELISTED_TOKENS:
        return None
    try:
        Pubkey.from_string(token_address)
    except Exception:
        return None

    cached = await get_cached_creation_time(token_address)
    if cached:
        return cached

    # --- headers (use strong browser defaults if not provided) ---
    try:
        DEFAULT_BROWSER_UA  # type: ignore[name-defined]
    except NameError:
        DEFAULT_BROWSER_UA = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
        )
    try:
        BROWSERISH_DEX_HEADERS  # type: ignore[name-defined]
    except NameError:
        BROWSERISH_DEX_HEADERS = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Origin": "https://dexscreener.com",
            "Referer": "https://dexscreener.com/",
            "User-Agent": DEFAULT_BROWSER_UA,
        }

    hdrs = dict(BROWSERISH_DEX_HEADERS)
    hdrs["Accept"] = "application/json"
    if headers:
        # allow explicit overrides from caller
        hdrs.update(headers)

    timeout = aiohttp.ClientTimeout(total=12, sock_connect=6, sock_read=8)
    url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"

    # Helper: attempt the request using a provided session; on IPv6/DNS issues in Windows,
    # transparently retry once with a temporary IPv4-only Dex session (if available).
    async def _do_request(sess: aiohttp.ClientSession) -> Optional[dict]:
        async with sess.get(url, headers=hdrs, timeout=timeout) as resp:
            if resp.status == 429:
                retry_after = resp.headers.get("Retry-After") or resp.headers.get("X-RateLimit-Reset")
                sleep_s = 8.0
                if retry_after:
                    try:
                        sleep_s = max(5.0, float(retry_after) - time.time())
                    except Exception:
                        # if it's not a float/epoch, just use a gentle backoff
                        sleep_s = 8.0
                await asyncio.sleep(max(5.0, sleep_s))
                raise aiohttp.ClientError("dexscreener-429")
            if resp.status != 200:
                body = (await resp.text())[:500]
                logger.warning("Dexscreener creation time HTTP %s for %s; body: %s", resp.status, url, body)
                return None
            try:
                return await resp.json(content_type=None)
            except Exception as je:
                logger.warning("Dexscreener creation time JSON decode failed: %s", je)
                return None

    data: Optional[dict] = None
    try:
        if _should_stop():
            return None
        data = await _do_request(session)
    except asyncio.TimeoutError:
        logger.error("Dexscreener creation time request timed out for %s", token_address)
        data = None
    except (aiohttp.ClientError, ValueError) as e:
        # If this looks like a DNS/IPv6 family issue on Windows, retry once via IPv4-only session
        retry_via_ipv4 = False
        emsg = str(e).lower()
        if ("getaddrinfo failed" in emsg) or ("temporary failure in name resolution" in emsg):
            retry_via_ipv4 = True
        if retry_via_ipv4:
            try:
                # Use the dedicated IPv4 Dex session if available
                dex_sess = None
                dex_headers = None
                try:
                    dex_sess, dex_headers = await _make_dex_session()  # type: ignore[name-defined]
                except NameError:
                    dex_sess = None
                if dex_sess is not None:
                    try:
                        data = await _do_request(dex_sess)
                    finally:
                        try:
                            await dex_sess.close()
                        except Exception:
                            pass
                else:
                    logger.error("Dexscreener creation time error for %s (IPv4 retry unavailable): %s", token_address, e, exc_info=True)
                    data = None
            except Exception:
                data = None
        else:
            logger.error("Dexscreener creation time error for %s: %s", token_address, e, exc_info=True)
            data = None

    pairs = data.get("pairs", []) if isinstance(data, dict) else []
    if not isinstance(pairs, list) or not pairs:
        return None

    earliest_ms: Optional[float] = None
    now_ms = time.time() * 1000.0
    solana_launch_ms = 1581465600000.0  # 2020-02-12T00:00:00Z

    for p in pairs:
        if _should_stop():
            return None
        try:
            # only consider Solana
            if p.get("chainId") != "solana":
                continue
            created = p.get("pairCreatedAt") or p.get("createdAt") or 0
            created = float(created or 0)
            if created and created < 10_000_000_000:
                created *= 1000.0
            if not (solana_launch_ms <= created <= now_ms):
                continue
            if earliest_ms is None or created < earliest_ms:
                earliest_ms = created
        except Exception:
            continue

    if earliest_ms is None:
        return None

    dt = datetime.fromtimestamp(earliest_ms / 1000.0, tz=timezone.utc)
    try:
        await cache_creation_time(token_address, dt)
    except Exception:
        pass
    return dt


# ---------------------------------------------------------------------
# Unified creation time helper (used by refresh hook)
# ---------------------------------------------------------------------
async def get_token_creation_time(
    token_address: str,
    solana_client: AsyncClient,
    session: Optional[aiohttp.ClientSession] = None
) -> Optional[datetime]:
    close_session = False
    sess = session
    if sess is None:
        sess = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15, sock_connect=6, sock_read=8))
        close_session = True
    try:
        # Prefer Birdeye (cheap when key/plan allows). If it fails/blocked, try Dex.
        dt = await fetch_birdeye_creation_time(token_address, sess)
        if dt:
            return dt

        # Use a dedicated Dex session (IPv4 + browser-like headers) for Windows quirks
        dex_sess = None
        dex_headers = None
        try:
            dex_sess, dex_headers = await _make_dex_session()  # type: ignore[name-defined]
        except NameError:
            # Fallback: reuse existing session and internal defaults
            dex_sess = sess
            dex_headers = None

        try:
            return await fetch_dexscreener_creation_time(token_address, dex_sess, headers=dex_headers)
        finally:
            # Only close if we created a separate Dex session
            if dex_sess is not None and dex_sess is not sess:
                try:
                    await dex_sess.close()
                except Exception:
                    pass

    finally:
        if close_session:
            try:
                await sess.close()
            except Exception:
                pass


# ---------------------------------------------------------------------
# Latest tokens aggregator
# ---------------------------------------------------------------------
async def get_latest_tokens(max_items: int = 500) -> List[Dict[str, Any]]:
    if _should_stop():
        logger.info("Shutdown signaled; skipping latest token fetch.")
        return []

    connector = aiohttp.TCPConnector(limit=50, enable_cleanup_closed=True)
    session_timeout = aiohttp.ClientTimeout(total=30, sock_connect=10, sock_read=20)

    # >>> WINDOWS DEX PATCH: dedicated Dex session (IPv4 + browser UA)
    try:
        dex_session, dex_headers = await _make_dex_session()  # type: ignore[name-defined]
    except NameError:
        # Fallback if helper not present
        dex_session, dex_headers = aiohttp.ClientSession(timeout=session_timeout), {
            "Accept": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
            ),
            "Origin": "https://dexscreener.com",
            "Referer": "https://dexscreener.com/",
        }

    async with aiohttp.ClientSession(connector=connector, timeout=session_timeout) as session:
        solana_client = AsyncClient(SOLANA_RPC_URL, timeout=DEFAULT_SOLANA_TIMEOUT)
        try:
            # ---------- Raydium ----------
            ray: List[Dict[str, Any]] = []
            if not _should_stop():
                try:
                    parse_cap, _ = _ray_limits(max_items)
                    ray = await fetch_raydium_tokens(session, solana_client, max_pairs=parse_cap)
                except MemoryError as e:
                    logger.warning("Raydium oversize MemoryError suppressed: %s", e)
                    ray = []
                except Exception as e:
                    logger.error("Raydium list error: %s", e, exc_info=True)
                    ray = []

            # ---------- Dexscreener ----------
            dex: List[Dict[str, Any]] = []
            if not _should_stop():
                try:
                    queries = _dex_queries()
                    pages, _ = _dex_pages_perpage()
                    rs = await asyncio.gather(
                        *[fetch_dexscreener_search(dex_session, query=q, pages=pages, headers=dex_headers) for q in queries],
                        return_exceptions=True
                    )
                    for r in rs:
                        if isinstance(r, list):
                            dex.extend(r)
                except Exception as e:
                    logger.error("Dexscreener list error: %s", e, exc_info=True)
                    dex = []

            # ---------- Birdeye ----------
            bird: List[Dict[str, Any]] = []
            if not _should_stop():
                try:
                    bird_cap = _bird_max()
                    if bird_cap > 0:
                        target = max(1, min(bird_cap, int(max_items or bird_cap)))
                        bird = await fetch_birdeye_tokens(session, solana_client, max_tokens=target)
                    else:
                        logger.info("Birdeye disabled by config; skipping.")
                        bird = []
                except Exception as e:
                    logger.error("Birdeye list error: %s", e, exc_info=True)
                    bird = []

            # We’re done with Dex session
            try:
                await dex_session.close()
            except Exception:
                pass

            # ---------- Combine ----------
            combined = (ray or []) + (dex or []) + (bird or [])
            if _should_stop():
                return []

            # ---------- Best per ADDRESS ----------
            def _fnum(o, key, default=0.0):
                try:
                    return float((o or {}).get(key, default) or default)
                except Exception:
                    return float(default)

            def _mc(o):
                try:
                    return float(
                        (o.get("market_cap")
                         or o.get("mc")
                         or o.get("fdv")
                         or o.get("fdvUsd")
                         or o.get("marketCap")
                         or 0.0)
                    )
                except Exception:
                    return 0.0

            best_by_addr: Dict[str, Dict[str, Any]] = {}
            for t in combined:
                addr = (t.get("address") or t.get("token_address"))
                if not addr:
                    continue
                prev = best_by_addr.get(addr)
                if prev is None:
                    best_by_addr[addr] = t
                else:
                    liq_t, liq_p = _fnum(t, "liquidity"), _fnum(prev, "liquidity")
                    vol_t, vol_p = _fnum(t, "volume_24h"), _fnum(prev, "volume_24h")
                    mc_t,  mc_p  = _mc(t), _mc(prev)
                    if (liq_t, vol_t, mc_t) > (liq_p, vol_p, mc_p):
                        best_by_addr[addr] = t

            # ---------- Item 4: collapse dupes by human *title* ----------
            def _norm_title(s: str) -> str:
                return "".join(ch for ch in str(s or "").lower() if ch.isalnum())

            def _title_key(t: Dict[str, Any]) -> Optional[str]:
                name = _norm_title(t.get("name", ""))
                sym  = _norm_title(t.get("symbol", ""))
                if len(name) >= 4:
                    return f"name:{name}"
                if len(sym) >= 4:
                    return f"sym:{sym}"
                return None

            def _prefer(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
                # keep the stronger row
                ka = (_fnum(a, "liquidity"), _fnum(a, "volume_24h"), _mc(a))
                kb = (_fnum(b, "liquidity"), _fnum(b, "volume_24h"), _mc(b))
                return a if ka >= kb else b

            by_title: Dict[str, Dict[str, Any]] = {}
            passthrough: List[Dict[str, Any]] = []

            for t in best_by_addr.values():
                key = _title_key(t)
                if not key:
                    passthrough.append(t)
                    continue
                prev = by_title.get(key)
                by_title[key] = t if prev is None else _prefer(t, prev)

            rows = list(by_title.values()) + passthrough

            # ---------- Normalize numeric fields & timestamps ----------
            def _num(x, d=0.0):
                try:
                    return float(x if x is not None else d)
                except Exception:
                    return float(d)

            out: List[Dict[str, Any]] = []
            for tok in rows:
                if "mc" not in tok or tok.get("mc") in (None, "", 0, "0"):
                    tok["mc"] = tok.get("market_cap") or tok.get("fdv") or tok.get("fdvUsd") or tok.get("marketCap") or 0.0

                tok["liquidity"]  = _num(tok.get("liquidity", 0.0))
                tok["volume_24h"] = _num(tok.get("volume_24h", 0.0))
                tok["price"]      = _num(tok.get("price", 0.0))
                tok["mc"]         = _num(tok.get("mc", 0.0))

                pc = tok.get("priceChange")
                if isinstance(pc, dict):
                    tok["price_change_1h"]  = _num(pc.get("h1", tok.get("price_change_1h", 0.0)))
                    tok["price_change_6h"]  = _num(pc.get("h6", tok.get("price_change_6h", 0.0)))
                    tok["price_change_24h"] = _num(pc.get("h24", tok.get("price_change_24h", 0.0)))
                else:
                    tok["price_change_1h"]  = _num(tok.get("price_change_1h", 0.0))
                    tok["price_change_6h"]  = _num(tok.get("price_change_6h", 0.0))
                    tok["price_change_24h"] = _num(tok.get("price_change_24h", 0.0))

                created_raw = tok.get("creation_timestamp") or tok.get("createdAt") or tok.get("pairCreatedAt") or 0
                try:
                    created_val = int(created_raw)
                    if created_val > 10**12:
                        created_val //= 1000
                    tok["creation_timestamp"] = created_val
                except Exception:
                    pass

                out.append(tok)

            out.sort(
                key=lambda x: (x["liquidity"] * 2.0) + (x["volume_24h"] * 1.5) + (x["mc"]),
                reverse=True,
            )

            if max_items and len(out) > int(max_items):
                out = out[: int(max_items)]

            logger.info(
                "Combined pool: ray=%d, dex=%d, bird=%d → unique_by_addr=%d (returned=%d)",
                len(ray or []), len(dex or []), len(bird or []), len(best_by_addr), len(out)
            )
            return out

        finally:
            try:
                await solana_client.close()
            except Exception:
                pass

# ---------------------------------------------------------------------
# Cache refresh hook (GUI calls this via run_async_task)
# ---------------------------------------------------------------------
async def refresh_token_cache(max_items: Optional[int] = None) -> int:
    global _CACHE_TOKEN_DATA_PARAM_COUNT

    pages, per_page = _dex_pages_perpage()
    dex_cap = _dex_post_cap()
    bird_cap = _bird_max()
    ray_cap, _ = _ray_limits()
    default_total = (dex_cap or (pages * per_page)) + max(0, bird_cap) + ray_cap

    tokens = await get_latest_tokens(max_items=max_items or default_total)

    # detect cache_token_data signature once
    if _CACHE_TOKEN_DATA_PARAM_COUNT is None:
        try:
            _CACHE_TOKEN_DATA_PARAM_COUNT = len(inspect.signature(cache_token_data).parameters)
        except Exception:
            _CACHE_TOKEN_DATA_PARAM_COUNT = 1  # safe default

    written = 0
    loud_errors = 0  # promote first few errors to WARNING so we can see them
    for t in tokens:
        try:
            # --- normalize keys so downstream writers are happy ---
            addr = t.get("address") or t.get("token_address")
            if not addr:
                raise ValueError("token missing address / token_address")

            # ensure both keys are present
            t["address"] = addr
            t.setdefault("token_address", addr)

            # optional: ensure a couple of common fields exist
            t.setdefault("symbol", t.get("baseToken", {}).get("symbol") or t.get("symbol"))
            t.setdefault("liquidity", t.get("liquidity"))
            t.setdefault("volume_24h", t.get("volume_24h") or t.get("volume24h") or t.get("v24h"))

            # --- call DB writer with the right arity ---
            if _CACHE_TOKEN_DATA_PARAM_COUNT == 1:
                # database.cache_token_data(row: dict)
                await cache_token_data(t)
            else:
                # database.cache_token_data(address: str, row: dict)
                await _write_token_row(addr, t)

            written += 1

        except Exception as e:
            loud_errors += 1
            if loud_errors <= 5:
                logger.warning(
                    "cache_token_data failed for %s: %s",
                    t.get("address") or t.get("token_address"), e, exc_info=True
                )
            else:
                logger.debug(
                    "cache_token_data failed silently for %s",
                    t.get("address") or t.get("token_address"), exc_info=True
                )
            continue

    logger.info("Refreshed token cache with %d entries", written)
    return written

# --- Back-compat shim -------------------------------------------------
async def collect_candidates(limit: int = 500):
    """
    Backwards-compatible alias for older code paths that still import
    `collect_candidates`. Calls the new aggregator and normalizes keys.
    """
    rows = await get_latest_tokens(max_items=limit)
    for r in rows:
        addr = r.get("address") or r.get("token_address")
        if addr:
            r["address"] = addr
            r.setdefault("token_address", addr)
    return rows

# If your module defines __all__, expose the alias too:
try:
    __all__  # may not exist
except NameError:
    pass
else:
    if "collect_candidates" not in __all__:
        __all__.append("collect_candidates")


# ---------------------------------------------------------------------
# Optional: quick sync wrapper (handy for CLI/local testing)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio as _a
    async def _main():
        try:
            toks = await get_latest_tokens(max_items=int(os.getenv("TEST_MAX_ITEMS", "500")))
            print(f"Got {len(toks)} tokens (sample 3):")
            for t in toks[:3]:
                print(json.dumps({k: t.get(k) for k in ("address","symbol","liquidity","volume_24h","mc")}, indent=2))
        except Exception as e:
            print("Error:", e)
    _a.run(_main())


# =====================================================================
# === Signals & Patterns (RSI, MACD, Bollinger, TD9, Candlestick)   ===
# === Lightweight, NumPy-only, post-shortlist enrichment utilities  ===
# =====================================================================
from typing import Iterable

# ---- optional config shim (tries to read a dict-like config if present) ----
def _cfg():
    try:
        # prefer a global 'load_config' if fetching.py defines it
        lc = globals().get("load_config")
        if callable(lc):
            return lc()
    except Exception:
        pass
    try:
        # or maybe a global 'CONFIG' dict exists
        cfg = globals().get("CONFIG")
        if isinstance(cfg, dict):
            return cfg
    except Exception:
        pass
    return {}

try:
    import numpy as _np  # use private alias to avoid collisions
except Exception:  # pragma: no cover
    _np = None

def _np_arr(a):
    if _np is None:
        raise ImportError("NumPy is required for indicators; please add numpy to requirements.")
    return _np.asarray(a, dtype=float)

# ----------------------------- Indicators -----------------------------
def _ema(arr, period: int):
    x = _np_arr(arr)
    if x.size < period:
        out = _np.empty_like(x); out[:] = _np.nan
        return out
    k = 2.0 / (period + 1)
    out = _np.empty_like(x); out[:] = _np.nan
    sma = _np.nanmean(x[:period])
    out[period-1] = sma
    for i in range(period, x.size):
        out[i] = x[i] * k + out[i-1] * (1 - k)
    return out

def ind_rsi(close, period: int = 14):
    c = _np_arr(close)
    if c.size < period + 1:
        rsis = _np.empty_like(c); rsis[:] = _np.nan
        return rsis
    diff = _np.diff(c, prepend=c[0])
    gains = _np.clip(diff, 0, None)
    losses = -_np.clip(diff, None, 0)
    rsis = _np.empty_like(c); rsis[:] = _np.nan
    avg_gain = _np.convolve(gains, _np.ones(period), 'valid')[:1].mean()
    avg_loss = _np.convolve(losses, _np.ones(period), 'valid')[:1].mean()
    if avg_loss == 0:
        rsis[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsis[period] = 100.0 - (100.0 / (1.0 + rs))
    ag, al = avg_gain, avg_loss
    for i in range(period + 1, c.size):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
        if al == 0:
            rsis[i] = 100.0
        else:
            rs = ag / al
            rsis[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsis

def ind_macd(close, fast: int = 12, slow: int = 26, signal: int = 9):
    c = _np_arr(close)
    ema_fast = _ema(c, fast)
    ema_slow = _ema(c, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "hist": hist}

def ind_bollinger(close, period: int = 20, stddev: float = 2.0):
    c = _np_arr(close)
    n = c.size
    mid = _np.full(n, _np.nan)
    upper = _np.full(n, _np.nan)
    lower = _np.full(n, _np.nan)
    width = _np.full(n, _np.nan)
    percent_b = _np.full(n, _np.nan)
    if n < period:
        return {"mid": mid, "upper": upper, "lower": lower, "width": width, "percent_b": percent_b}
    # O(n*period) rolling — fine for 100–500 bars
    for i in range(period-1, n):
        w = c[i-period+1:i+1]
        m = _np.mean(w); s = _np.std(w, ddof=0)
        mid[i] = m
        upper[i] = m + stddev * s
        lower[i] = m - stddev * s
        width[i] = (upper[i] - lower[i]) / m if m != 0 and not _np.isnan(m) else _np.nan
        if not _np.isnan(upper[i]) and not _np.isnan(lower[i]) and upper[i] != lower[i]:
            percent_b[i] = (c[i] - lower[i]) / (upper[i] - lower[i])
    return {"mid": mid, "upper": upper, "lower": lower, "width": width, "percent_b": percent_b}

def ind_td9(close, lookback: int = 30):
    c = _np_arr(close); n = c.size
    up = _np.zeros(n, dtype=int)     # bearish setup count
    down = _np.zeros(n, dtype=int)   # bullish setup count
    setup = _np.full(n, "", dtype=object)
    td9_up = _np.zeros(n, dtype=bool)
    td9_down = _np.zeros(n, dtype=bool)
    for i in range(n):
        if i < 4:
            continue
        if c[i] > c[i-4]:
            up[i] = (up[i-1] + 1) if up[i-1] > 0 else 1
            down[i] = 0
        elif c[i] < c[i-4]:
            down[i] = (down[i-1] + 1) if down[i-1] > 0 else 1
            up[i] = 0
        else:
            up[i] = 0; down[i] = 0
        if up[i] == 9:
            setup[i] = "td9_up"; td9_up[i] = True
        elif down[i] == 9:
            setup[i] = "td9_down"; td9_down[i] = True
    if lookback and n > lookback:
        start = n - lookback
        td9_up[:start] = False
        td9_down[:start] = False
    return {"up": up, "down": down, "setup": setup, "td9_up": td9_up, "td9_down": td9_down}

# -------------------------- Candle Patterns --------------------------
def pat_bullish_engulfing(open_, close):
    o, c = _np_arr(open_), _np_arr(close)
    prev_bear = c[:-1] < o[:-1]
    curr_bull = c[1:]  > o[1:]
    engulf    = (c[1:] >= o[:-1]) & (o[1:] <= c[:-1])
    out = _np.zeros_like(c, dtype=bool); out[1:] = prev_bear & curr_bull & engulf
    return out

def pat_bearish_engulfing(open_, close):
    o, c = _np_arr(open_), _np_arr(close)
    prev_bull = c[:-1] > o[:-1]
    curr_bear = c[1:]  < o[1:]
    engulf    = (o[1:] >= c[:-1]) & (c[1:] <= o[:-1])
    out = _np.zeros_like(c, dtype=bool); out[1:] = prev_bull & curr_bear & engulf
    return out

def pat_hammer(open_, high, low, close, tol: float = 0.3):
    o, h, l, c = map(_np_arr, (open_, high, low, close))
    body = _np.abs(c - o)
    upper = h - _np.maximum(c, o)
    lower = _np.minimum(c, o) - l
    return (lower >= 2*body) & (upper <= body) & ((h - c) <= tol * (h - l))

def pat_shooting_star(open_, high, low, close, tol: float = 0.3):
    o, h, l, c = map(_np_arr, (open_, high, low, close))
    body = _np.abs(c - o)
    upper = h - _np.maximum(c, o)
    lower = _np.minimum(c, o) - l
    return (upper >= 2*body) & (lower <= body) & ((c - l) <= tol * (h - l))

def pat_doji(open_, close, eps: float = 1e-6, rel: float = 0.1):
    o, c = _np_arr(open_), _np_arr(close)
    rng = _np.maximum(_np.abs(c - o), eps)
    return (_np.abs(c - o) / rng) < rel

def classify_patterns_arrays(ohlcv: Dict[str, Iterable[float]], names: Iterable[str]):
    o, h, l, c = map(_np_arr, (ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]))
    out: Dict[str, "_np.ndarray"] = {}
    for name in names:
        if name == "bullish_engulfing": out[name] = pat_bullish_engulfing(o, c)
        elif name == "bearish_engulfing": out[name] = pat_bearish_engulfing(o, c)
        elif name == "hammer": out[name] = pat_hammer(o, h, l, c)
        elif name == "shooting_star": out[name] = pat_shooting_star(o, h, l, c)
        elif name == "doji": out[name] = pat_doji(o, c)
    return out

# ------------------------ Enrichment Orchestrator ---------------------
def _signals_cfg():
    c = _cfg() if callable(_cfg) else {}
    sig = (c.get("signals") if isinstance(c, dict) else None) or {}
    pats = (c.get("patterns") if isinstance(c, dict) else None) or {}
    return {
        "enable": bool(sig.get("enable", True)),
        "rsi": {"enable": bool(sig.get("rsi", {}).get("enable", True)), "period": int(sig.get("rsi", {}).get("period", 14))},
        "macd": {
            "enable": bool(sig.get("macd", {}).get("enable", True)),
            "fast": int(sig.get("macd", {}).get("fast", 12)),
            "slow": int(sig.get("macd", {}).get("slow", 26)),
            "signal": int(sig.get("macd", {}).get("signal", 9)),
        },
        "bollinger": {
            "enable": bool(sig.get("bollinger", {}).get("enable", True)),
            "period": int(sig.get("bollinger", {}).get("period", 20)),
            "stddev": float(sig.get("bollinger", {}).get("stddev", 2.0)),
        },
        "td9": {"enable": bool(sig.get("td9", {}).get("enable", True)), "lookback": int(sig.get("td9", {}).get("lookback", 30))},
        "patterns": {
            "enable": bool(pats.get("enable", True)),
            "list": list(pats.get("list", ["bullish_engulfing","bearish_engulfing","hammer","shooting_star","doji"])),
        },
        "ohlcv": {
            "interval": (sig.get("ohlcv", {}) or {}).get("interval", "1m"),
            "limit": int((sig.get("ohlcv", {}) or {}).get("limit", 200)),
        }
    }

async def enrich_token_with_signals(token: Dict[str, Any], ohlcv: Dict[str, Iterable[float]], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Mutates and returns `token` with last-bar signals.
    Expected OHLCV: dict with keys: 'open','high','low','close','volume' arrays.
    """
    cfg = cfg or _signals_cfg()
    if not cfg.get("enable", True):
        return token
    close = _np_arr(ohlcv["close"])
    if close.size < 30:  # minimal bars
        return token

    # RSI
    if cfg["rsi"]["enable"]:
        _rsi = ind_rsi(close, cfg["rsi"]["period"])
        token["rsi"] = float(_rsi[-1]) if _rsi.size else None

    # MACD
    if cfg["macd"]["enable"]:
        m = ind_macd(close, cfg["macd"]["fast"], cfg["macd"]["slow"], cfg["macd"]["signal"])
        token["macd"] = float(m["macd"][-1]) if m["macd"].size else None
        token["macd_signal"] = float(m["signal"][-1]) if m["signal"].size else None
        token["macd_hist"] = float(m["hist"][-1]) if m["hist"].size else None

    # Bollinger
    if cfg["bollinger"]["enable"]:
        bb = ind_bollinger(close, cfg["bollinger"]["period"], cfg["bollinger"]["stddev"])
        token["bb_percent_b"] = float(bb["percent_b"][-1]) if bb["percent_b"].size else None
        token["bb_width"]     = float(bb["width"][-1]) if bb["width"].size else None

    # TD9
    if cfg["td9"]["enable"]:
        td = ind_td9(close, cfg["td9"]["lookback"])
        token["td9_up"]   = bool(td["td9_up"][-1]) if td["td9_up"].size else False
        token["td9_down"] = bool(td["td9_down"][-1]) if td["td9_down"].size else False

    # Patterns
    if cfg["patterns"]["enable"]:
        pats = classify_patterns_arrays(ohlcv, cfg["patterns"]["list"])
        token["patterns"] = [k for k, v in pats.items() if v.size and bool(v[-1])]

    return token

async def batch_enrich_tokens_with_signals(
    tokens: List[Dict[str, Any]],
    fetch_ohlcv_func,  # async callable: (address, interval, limit) -> Dict[str, List[float]]
    interval: Optional[str] = None,
    limit: Optional[int] = None,
    cfg: Optional[Dict[str, Any]] = None,
    concurrency: int = 8,
) -> List[Dict[str, Any]]:
    """Generic post-shortlist enricher. You provide how to fetch OHLCV for a token.
    We call it for top-N tokens and attach signals/patterns to each.
    """
    cfg_base = _signals_cfg()
    interval = interval or cfg_base["ohlcv"]["interval"]
    limit = int(limit or cfg_base["ohlcv"]["limit"])
    cfg = cfg or cfg_base

    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def _enrich_one(tok: Dict[str, Any]):
        addr = tok.get("address") or tok.get("token_address")
        if not addr:
            return tok
        async with sem:
            try:
                ohlcv = await fetch_ohlcv_func(addr, interval=interval, limit=limit)
                if isinstance(ohlcv, dict) and all(k in ohlcv for k in ("open","high","low","close","volume")):
                    await enrich_token_with_signals(tok, ohlcv, cfg=cfg)
            except Exception as e:
                try:
                    logger.debug("batch_enrich: OHLCV fetch/enrich failed for %s: %s", addr, e)
                except Exception:
                    pass
        return tok

    return await asyncio.gather(*[_enrich_one(t) for t in tokens])

# Integration note:
# Call `batch_enrich_tokens_with_signals` right AFTER you compute your shortlist
# and BEFORE ranking/scoring for final picks. Provide your own async OHLCV fetcher.

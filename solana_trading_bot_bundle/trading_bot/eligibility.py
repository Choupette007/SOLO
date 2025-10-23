# solana_trading_bot_bundle/trading_bot/eligibility.py
from __future__ import annotations

import os
import logging
import asyncio
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import Counter
import aiohttp

# ---- Noise filter: stables & WSOL shouldn't be shortlisted in categories ----
KNOWN_STABLES = {
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
}
ALWAYS_HIDE_IN_CATEGORIES = KNOWN_STABLES | {
    "So11111111111111111111111111111111111111112",  # WSOL
}

from .utils import (
    add_to_blacklist,
    load_config,
    get_rugcheck_token,  # fallback if rugcheck_auth headers aren’t available
)

logger = logging.getLogger("TradingBot")

# Standardize eligibility rejections at INFO so they're easy to grep in logs
REJECT_TAG = "ELIG-REJECT"
def _log_reject(addr: str, reason: str) -> None:
    logger.info("%s %s: %s", REJECT_TAG, addr, reason)


# --- Rugcheck headers import (supports both of your helper locations) ---
try:
    # utils variant
    from utils.rugcheck_auth import ensure_valid_rugcheck_headers as _ensure_headers  # type: ignore
    from utils.rugcheck_auth import get_rugcheck_headers as _get_headers  # type: ignore
except Exception:
    try:
        # bundle variant
        from solana_trading_bot_bundle.trading_bot.rugcheck_auth import (  # type: ignore
            ensure_valid_rugcheck_headers as _ensure_headers,
            get_rugcheck_headers as _get_headers,
        )
    except Exception:
        _ensure_headers = None  # type: ignore
        _get_headers = None     # type: ignore


# -----------------------------------------------------------------------------#
# Optional TTL caches (avoid refetching within a short window)
# -----------------------------------------------------------------------------#
try:
    from cachetools import TTLCache
except Exception:  # very small fallback if cachetools isn't installed
    class TTLCache(dict):  # type: ignore
        def __init__(self, maxsize: int, ttl: int) -> None:
            super().__init__()
            self._ttl = ttl
        def __setitem__(self, key, value) -> None:  # store (value, expires_at)
            super().__setitem__(key, (value, time.time() + self._ttl))
        def __getitem__(self, key):
            v, exp = super().__getitem__(key)
            if time.time() > exp:
                super().pop(key, None)
                raise KeyError(key)
            return v
        def get(self, key, default=None):
            try:
                return self.__getitem__(key)
            except KeyError:
                return default

# Cache horizons (seconds)
_RUGCHECK_TTL = 30 * 60
_HOLDERS_TTL = 10 * 60
_CONC_TTL    = 10 * 60

_rugcheck_cache: TTLCache = TTLCache(maxsize=5000, ttl=_RUGCHECK_TTL)
_holders_cache: TTLCache  = TTLCache(maxsize=10000, ttl=_HOLDERS_TTL)  # key: token -> int
_conc_cache: TTLCache     = TTLCache(maxsize=5000, ttl=_CONC_TTL)      # key: (token, top_n) -> float

# -----------------------------------------------------------------------------#
# Env fallback thresholds (used only if config values are missing)
# -----------------------------------------------------------------------------#
MIN_LIQ_USD_ENV   = float(os.getenv("MIN_LIQ_USD", "0"))     # e.g., 1000
MIN_VOL24_USD_ENV = float(os.getenv("MIN_VOL24_USD", "0"))   # e.g., 2500

# -----------------------------------------------------------------------------#
# Small helpers
# -----------------------------------------------------------------------------#
def get_env_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return str(value).strip().lower() in ("true", "1", "yes", "on")

def _num(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _lower(s: Any) -> str:
    try:
        return str(s or "").strip().lower()
    except Exception:
        return ""

def _label_names_from_any(labels_any: Any) -> List[str]:
    """
    Normalize labels coming from RugCheck (array of dicts/strings/mixed) into a list of lowercase names.
    Accepts: [{"label":"scam"}, "dangerous", {"name": "Honeypot"}]
    -> ["scam","dangerous","honeypot"]
    """
    out: List[str] = []
    if labels_any is None:
        return out
    if isinstance(labels_any, dict):
        for k, v in labels_any.items():
            if v:
                out.append(_lower(k))
        return out
    if isinstance(labels_any, list):
        for item in labels_any:
            if isinstance(item, dict):
                name = item.get("label") or item.get("name") or item.get("value") or item.get("type")
                out.append(_lower(name))
            else:
                out.append(_lower(item))
        return out
    out.append(_lower(labels_any))
    return out

def _best_first(*vals) -> Any:
    """Return the first non-None/non-empty value from vals (0 is valid)."""
    for v in vals:
        if v not in (None, "", [], {}):
            return v
    return None

# ---------------- New helper: require recent positive momentum ----------------
def _recent_momentum_allowed(
    token: Dict[str, Any],
    *,
    cfg: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    Enforce minima for short-term momentum with fail-closed behavior.
    - Reads price_change_1h and price_change_6h (with fallbacks from dexscreener/birdeye).
    - Config keys (discovery):
        min_price_change_1h  (default: 0.0)  # require >= this percent over 1h
        min_price_change_6h  (default: 0.0)  # require >= this percent over 6h
    - If a min threshold is explicitly present (> 0) and the corresponding field is missing, fail-closed (return False).
    Returns (allowed, reason)
    """
    try:
        disc = cfg.get("discovery") or {}
        min_1h = _num(disc.get("min_price_change_1h", 0.0), 0.0)
        min_6h = _num(disc.get("min_price_change_6h", 0.0), 0.0)
    except Exception:
        min_1h, min_6h = 0.0, 0.0

    # find candidate fields (many APIs use different names)
    # Use _best_first to detect presence (returns None if all are None/empty)
    p1_raw = _best_first(
        token.get("price_change_1h"),
        token.get("priceChange1h"),
        (token.get("dexscreener") or {}).get("priceChange1h"),
        (token.get("birdeye") or {}).get("price_change_1h"),
    )

    p6_raw = _best_first(
        token.get("price_change_6h"),
        token.get("priceChange6h"),
        (token.get("dexscreener") or {}).get("priceChange6h"),
        (token.get("birdeye") or {}).get("price_change_6h"),
    )

    # Fail-closed: if threshold is explicitly set and field is missing, reject
    if min_1h > 0.0 and p1_raw is None:
        return False, f"price_change_1h missing but min_price_change_1h={min_1h}% required"
    if min_6h > 0.0 and p6_raw is None:
        return False, f"price_change_6h missing but min_price_change_6h={min_6h}% required"

    # Convert to numeric (now safe since we checked presence if threshold exists)
    p1 = _num(p1_raw, 0.0)
    p6 = _num(p6_raw, 0.0)

    # Enforce minima if threshold is set
    if min_1h > 0.0 and p1 < float(min_1h):
        return False, f"price_change_1h={p1}% < min_required={min_1h}%"
    if min_6h > 0.0 and p6 < float(min_6h):
        return False, f"price_change_6h={p6}% < min_required={min_6h}%"

    return True, f"recent_momentum_ok 1h={p1}% 6h={p6}%"
# -----------------------------------------------------------------------------

# ---------------- New: Price-change guard helper -------------------------------
def _price_change_allowed(
    token: Dict[str, Any],
    price_change_pct: float,
    *,
    cfg: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    Decide whether a token's 24h price change is acceptable (signed-aware).
    Returns (allowed: bool, reason: str).

    Behavior:
    - Use discovery config values when present (sensible defaults otherwise).
    - For negative price_change_pct (drops): if abs(pct) > max_price_drop, reject; otherwise allow small drops.
    - For positive pct: 
        - Always reject beyond a hard cap (max_price_change_hard).
        - Allow pct <= max_price_change unconditionally.
        - For pct between max_price_change and max_price_change_hard, require market depth:
            liquidity >= min_liq_for_big_pct OR volume_24h >= min_vol_for_big_pct.
        - If price is extremely small (< min_price_for_pct), treat pct as unreliable unless market depth present.
    """
    try:
        disc = cfg.get("discovery") or {}
        max_pct = _num(disc.get("max_price_change", 350), 350.0)
        max_pct_hard = _num(disc.get("max_price_change_hard", max(max_pct * 4, 2000)), max(max_pct * 4, 2000.0))
        min_price_for_pct = _num(disc.get("min_price_for_pct", 0.0001), 0.0001)
        min_liq_for_big_pct = _num(disc.get("min_liq_for_big_pct", 5000.0), 5000.0)
        min_vol_for_big_pct = _num(disc.get("min_vol_for_big_pct", 10000.0), 10000.0)
        max_price_drop = _num(disc.get("max_price_drop", max_pct), max_pct)  # default to max_price_change if not set
    except Exception:
        max_pct, max_pct_hard, min_price_for_pct, min_liq_for_big_pct, min_vol_for_big_pct = 350.0, 2000.0, 0.0001, 5000.0, 10000.0
        max_price_drop = max_pct

    pct = float(price_change_pct or 0.0)

    # Handle negative price changes (drops)
    if pct < 0:
        drop_magnitude = abs(pct)
        if drop_magnitude > float(max_price_drop):
            return False, f"price_change_24h={pct}% drop too large (abs>{max_price_drop}%)"
        # Allow small drops
        return True, f"price_change_24h={pct}% drop acceptable (abs<={max_price_drop}%)"

    # Positive price change handling (using signed value)
    # Hard safety cap
    if pct > float(max_pct_hard):
        return False, f"price_change_24h={pct}% too extreme (>{max_pct_hard}%)"

    # Within primary threshold -> allow (these are runners but not absurd)
    if pct <= float(max_pct):
        return True, f"price_change_24h={pct}% <= max={max_pct}%"

    # Extract market depth
    liq = _num(_best_first(token.get("liquidity"), token.get("liquidity_usd"), 0.0), 0.0)
    vol24 = _num(_best_first(token.get("volume_24h"), token.get("v24hUSD"), token.get("volume"), 0.0), 0.0)
    price_val = None
    try:
        price_val = None if token.get("price") is None else float(token.get("price"))
    except Exception:
        price_val = None

    # Tiny-price heuristic: percent unreliable if price is tiny and no depth
    if price_val is not None and price_val > 0 and price_val < float(min_price_for_pct):
        if liq >= float(min_liq_for_big_pct) or vol24 >= float(min_vol_for_big_pct):
            return True, f"tiny_price={price_val} but market depth present (liq={liq},vol24={vol24})"
        return False, f"tiny_price={price_val} -> pct unreliable (pct={pct}%)"

    # Percent above max_pct but below hard cap: require depth
    if liq >= float(min_liq_for_big_pct) or vol24 >= float(min_vol_for_big_pct):
        return True, f"pct={pct}% > {max_pct}% but market depth ok (liq={liq},vol24={vol24})"

    return False, f"price_change_24h={pct}% > {max_pct}% and market depth too low (liq={liq},vol24={vol24})"

# -----------------------------------------------------------------------------#
# Centralized HTTP helper (retries for 429/5xx, consistent headers)
# -----------------------------------------------------------------------------#
async def _json_request(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[aiohttp.ClientTimeout] = None,
    max_retries: int = 2,
    backoff_initial: float = 1.5,
) -> Optional[Any]:  # <-- broadened from Optional[Dict[str, Any]] because some APIs return lists
    hdrs = {"accept": "application/json", "User-Agent": "SOLOTradingBot/1.0"}
    if headers:
        hdrs.update(headers)

    attempt = 0
    backoff = backoff_initial
    while True:
        try:
            async with session.request(
                method.upper(),
                url,
                headers=hdrs,
                params=params,
                json=json_body,
                timeout=timeout,
            ) as resp:
                if resp.status == 200:
                    try:
                        return await resp.json(content_type=None)
                    except Exception as e:
                        logger.warning("Failed to decode JSON from %s: %s", url, e)
                        return None
                if resp.status in (429, 500, 502, 503, 504) and attempt < max_retries:
                    attempt += 1
                    body = await resp.text()
                    logger.warning("HTTP %s from %s, retrying in %.1fs (attempt %d/%d). Body: %s",
                                   resp.status, url, backoff, attempt, max_retries, body[:200])
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                body = await resp.text()
                logger.debug("Non-200 from %s: %s %s", url, resp.status, body[:300])
                return None
        except Exception as e:
            if attempt < max_retries:
                attempt += 1
                logger.warning("Request error to %s: %s; retrying in %.1fs (%d/%d)",
                               url, e, backoff, attempt, max_retries)
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            logger.warning("Request error to %s (final): %s", url, e)
            return None

# -----------------------------------------------------------------------------#
# RugCheck verification
# -----------------------------------------------------------------------------#
async def verify_token_with_rugcheck(token_address, token, session, config):
    """
    Resilient RugCheck verifier:
    - Tries /tokens/scan/solana/{mint} then /tokens/{mint}
    - Supports JWT (Authorization: Bearer) and API key (X-API-KEY)
    - Caches results and tolerates API downtime if configured
    Returns (risk_score: float, labels: list[str], message: str)
    """
    rcfg = (config.get("rugcheck") or {}) if isinstance(config, dict) else {}
    if not bool(rcfg.get("enabled", False)):
        return 0.0, [], "RugCheck disabled"

    # Cache hit
    cache = _rugcheck_cache  # type: ignore
    if token_address in cache:
        return cache[token_address]

    # API base + endpoints (scan first, then fallback)
    api_base = str(rcfg.get("api_base", "https://api.rugcheck.xyz/v1")).rstrip("/")
    endpoints = [
        f"{api_base}/tokens/scan/solana/{token_address}",
        f"{api_base}/tokens/{token_address}",
    ]

    # ---- Build headers robustly (prefer module helpers, then utils/env) ----
    _rc_hdrs_async = None
    _rc_hdrs_simple = None
    try:
        from .rugcheck_auth import ensure_valid_rugcheck_headers as _rc_hdrs_async  # type: ignore
    except Exception:
        try:
            from solana_trading_bot_bundle.trading_bot.rugcheck_auth import ensure_valid_rugcheck_headers as _rc_hdrs_async  # type: ignore
        except Exception:
            _rc_hdrs_async = None
    try:
        from .rugcheck_auth import get_rugcheck_headers as _rc_hdrs_simple  # type: ignore
    except Exception:
        try:
            from solana_trading_bot_bundle.trading_bot.rugcheck_auth import get_rugcheck_headers as _rc_hdrs_simple  # type: ignore
        except Exception:
            _rc_hdrs_simple = None

    headers: Dict[str, str] = {}
    # Try async refresher (may be sync in some builds)
    if _rc_hdrs_async is not None:
        try:
            maybe = _rc_hdrs_async(session, force_refresh=False)  # type: ignore
            if asyncio.iscoroutine(maybe):
                maybe = await maybe
            if isinstance(maybe, dict):
                headers.update(maybe)
        except Exception:
            pass
    # Try simple getter if Authorization still missing
    h_lower = {k.lower(): v for k, v in headers.items()}
    if "authorization" not in h_lower and _rc_hdrs_simple is not None:
        try:
            maybe = _rc_hdrs_simple()  # type: ignore
            if isinstance(maybe, dict):
                headers.update(maybe)
        except Exception:
            pass
    # Fallback to env/config JWT
    h_lower = {k.lower(): v for k, v in headers.items()}
    if "authorization" not in h_lower:
        try:
            tok = get_rugcheck_token()
            if tok:
                headers["Authorization"] = f"Bearer {tok}"
        except Exception:
            pass
    # Add API key if not already present
    h_lower = {k.lower(): v for k, v in headers.items()}
    if "x-api-key" not in h_lower:
        api_key = rcfg.get("api_key") or os.getenv("RUGCHECK_API_KEY") or ""
        if api_key:
            headers["X-API-KEY"] = str(api_key)

    timeout_total = float(rcfg.get("timeout_sec", 15.0))
    timeout = aiohttp.ClientTimeout(total=timeout_total, sock_connect=min(7, timeout_total), sock_read=min(10, timeout_total))
    data, last_err = None, None

    async def _fetch(url: str, force_refresh: bool = False):
        nonlocal data, last_err, headers
        local_headers = dict(headers)
        if force_refresh and _rc_hdrs_async is not None:
            try:
                refreshed = _rc_hdrs_async(session, force_refresh=True)  # type: ignore
                if asyncio.iscoroutine(refreshed):
                    refreshed = await refreshed
                if isinstance(refreshed, dict) and refreshed.get("Authorization"):
                    local_headers.update(refreshed)
            except Exception:
                pass
        try:
            async with session.get(url, headers=local_headers, timeout=timeout) as resp:
                if resp.status == 200:
                    try:
                        data = await resp.json(content_type=None)
                    except Exception:
                        txt = await resp.text()
                        data = {"raw": txt}
                    return resp.status
                last_err = f"HTTP {resp.status} at {url}"
                return resp.status
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            return None

    # Try scan endpoint first (with optional refresh on 401/403), then /tokens
    status = await _fetch(endpoints[0], force_refresh=False)
    if status in (401, 403):
        status = await _fetch(endpoints[0], force_refresh=True)
    if data is None:
        status = await _fetch(endpoints[1], force_refresh=False)
        if status in (401, 403):
            await _fetch(endpoints[1], force_refresh=True)

    # No usable data from both endpoints
    if data is None:
        msg = f"RugCheck API error for {token_address}: {last_err or 'unknown'}"
        logger.warning(msg)
        if bool(rcfg.get("allow_if_api_down", True)):
            tup = (0.0, [], "API down - allowed")
            cache[token_address] = tup  # type: ignore
            return tup
        tup = (9999.0, [], msg)
        cache[token_address] = tup  # type: ignore
        return tup

    # ---- Parse response ----
    # Some responses wrap in {"data": {...}}, others are flat dicts
    payload = data.get("data", data) if isinstance(data, dict) else {}
    # Normalize labels robustly
    labels_raw = payload.get("labels") or payload.get("riskLabels") or payload.get("flags") or []
    labels_list = _label_names_from_any(labels_raw)

    # Extract risk score from several possible shapes
    cand_scores = [
        payload.get("risk_score"),
        payload.get("riskScore"),
        payload.get("score"),
        (payload.get("risk") or {}).get("score") if isinstance(payload, dict) else None,
        (payload.get("result") or {}).get("risk_score") if isinstance(payload, dict) else None,
        (payload.get("analysis") or {}).get("risk_score") if isinstance(payload, dict) else None,
    ]
    risk_score = 0.0
    for v in cand_scores:
        try:
            if v is not None:
                risk_score = float(v)
                break
        except Exception:
            continue

    # Apply local danger-label blocklist
    danger = {str(x).lower() for x in (rcfg.get("danger_labels_blocklist") or ["dangerous", "scam", "honeypot"])}
    present = set(labels_list)
    if present & danger:
        tup = (9999.0, labels_list, f"Blocked by labels: {sorted(present & danger)}")
        cache[token_address] = tup  # type: ignore
        return tup

    tup = (risk_score, labels_list, "OK")
    cache[token_address] = tup  # type: ignore
    return tup


# -----------------------------------------------------------------------------#
# Holders helpers (Birdeye/Solscan/Helius/public)
# -----------------------------------------------------------------------------#
async def _fetch_holders_birdeye(
    token_address: str,
    session: aiohttp.ClientSession,
    cfg: Dict[str, Any],
    timeout: aiohttp.ClientTimeout,
) -> Optional[int]:
    p = (cfg.get("providers", {}) or {}).get("birdeye", {}) or {}
    api_key = p.get("api_key") or ""
    if not api_key:
        return None
    url = p.get("holders_url") or "https://api.birdeye.so/defi/token/holders"
    headers = {"x-api-key": api_key}
    payload = {"address": token_address, "page": 1, "page_size": 1}

    data = await _json_request(
        session, "POST", url, headers=headers, json_body=payload, timeout=timeout
    )
    if not data:
        return None
    d = data.get("data", data)
    total = d.get("total")
    return int(total) if total is not None else None

async def _fetch_holders_solscan_pro(
    token_address: str,
    session: aiohttp.ClientSession,
    cfg: Dict[str, Any],
    timeout: aiohttp.ClientTimeout,
) -> Optional[int]:
    p = (cfg.get("providers", {}) or {}).get("solscan_pro", {}) or {}
    api_key = p.get("api_key") or ""
    if not api_key:
        return None

    url = "https://pro-api.solscan.io/v2.0/token/holders"
    headers = {"token": api_key}
    params = {"address": token_address, "page": 1, "page_size": 10}

    data = await _json_request(
        session, "GET", url, headers=headers, params=params, timeout=timeout
    )
    if not data:
        return None
    d = data.get("data") or {}
    total = d.get("total")
    return int(total) if total is not None else None

async def _fetch_holders_helius(
    token_address: str,
    session: aiohttp.ClientSession,
    cfg: Dict[str, Any],
    timeout: aiohttp.ClientTimeout,
) -> Optional[int]:
    p = (cfg.get("providers", {}) or {}).get("helius", {}) or {}
    api_key = p.get("api_key") or ""
    if not api_key:
        return None
    url = p.get("url") or f"https://mainnet.helius-rpc.com/?api-key={api_key}"
    method = p.get("method_summary") or "getTokenHoldersSummary"

    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": {"mint": token_address}}
    data = await _json_request(session, "POST", url, json_body=payload, timeout=timeout)
    if not data:
        return None
    result = (data or {}).get("result") | {} if isinstance((data or {}).get("result"), dict) else (data or {}).get("result", {})
    total = result.get("holderCount") or result.get("total")
    return int(total) if total is not None else None

async def _fetch_holders_public_solscan(
    token_address: str,
    session: aiohttp.ClientSession,
    timeout: aiohttp.ClientTimeout,
) -> Optional[int]:
    base = "https://public-api.solscan.io/token/holders"
    params = {"tokenAddress": token_address, "limit": 1, "offset": 0}
    data = await _json_request(session, "GET", base, params=params, timeout=timeout)
    if not data:
        return None
    if isinstance(data, dict):
        if "total" in data:
            return int(data["total"])
        d = data.get("data")
        if isinstance(d, dict) and "total" in d:
            return int(d["total"])
    return None

async def fetch_holder_count(
    token_address: str,
    session: aiohttp.ClientSession,
    config: Dict[str, Any],
) -> Tuple[Optional[int], str]:
    cached = _holders_cache.get(token_address)
    if isinstance(cached, int):
        return cached, "cache"

    hcfg = config.get("holders", {}) or {}
    timeout_sec = int(hcfg.get("timeout_seconds", 8))
    timeout = aiohttp.ClientTimeout(total=timeout_sec, sock_connect=5, sock_read=timeout_sec)

    count = await _fetch_holders_birdeye(token_address, session, hcfg, timeout)
    if isinstance(count, int) and count >= 0:
        _holders_cache[token_address] = count
        return count, "birdeye"

    count = await _fetch_holders_solscan_pro(token_address, session, hcfg, timeout)
    if isinstance(count, int) and count >= 0:
        _holders_cache[token_address] = count
        return count, "solscan_pro"

    count = await _fetch_holders_helius(token_address, session, hcfg, timeout)
    if isinstance(count, int) and count >= 0:
        _holders_cache[token_address] = count
        return count, "helius"

    if (hcfg.get("providers", {}).get("public_solscan", {}) or {}).get("enabled", True):
        count = await _fetch_holders_public_solscan(token_address, session, timeout)
        if isinstance(count, int) and count >= 0:
            _holders_cache[token_address] = count
            return count, "public_solscan"

    return None, "unavailable"

def _min_holders_for_categories(categories: List[str], config: Dict[str, Any]) -> Optional[int]:
    hcfg = config.get("holders", {}) or {}
    mins = hcfg.get("min_holders", {}) or {}
    vals: List[int] = []
    for cat in categories:
        v = mins.get(cat)
        if isinstance(v, (int, float)):
            vals.append(int(v))
    return max(vals) if vals else None

# -----------------------------------------------------------------------------#
# Concentration — top N share
# -----------------------------------------------------------------------------#
async def _birdeye_supply_and_top(
    token_address: str,
    session: aiohttp.ClientSession,
    cfg: Dict[str, Any],
    top_n: int,
    timeout: aiohttp.ClientTimeout,
) -> Optional[Tuple[float, float]]:
    p = (cfg.get("providers", {}) or {}).get("birdeye", {}) or {}
    api_key = p.get("api_key") or ""
    if not api_key:
        return None
    headers = {"x-api-key": api_key}
    holders_url  = p.get("top_holders_url") or p.get("holders_url") or "https://api.birdeye.so/defi/token/holders"
    overview_url = p.get("overview_url") or "https://api.birdeye.so/defi/token/overview"

    payload = {"address": token_address, "page": 1, "page_size": int(max(1, top_n))}
    hdata = await _json_request(session, "POST", holders_url, headers=headers, json_body=payload, timeout=timeout)
    if not hdata:
        return None
    d = hdata.get("data", hdata)
    items = d.get("items") or d.get("holders") or []
    top_sum = 0.0
    for it in items:
        qty = _best_first(it.get("balance"), it.get("amount"), it.get("uiAmount"))
        if qty is None:
            continue
        top_sum += _num(qty, 0.0)

    payload_overview = {"address": token_address}
    odata = await _json_request(session, "POST", overview_url, headers=headers, json_body=payload_overview, timeout=timeout)
    if not odata:
        return None
    od = odata.get("data", odata)
    supply = _best_first(od.get("supply"), od.get("circulating_supply"), od.get("total_supply"))
    if supply is None:
        return None
    total_supply = _num(supply, 0.0)
    if total_supply <= 0:
        return None
    return top_sum, total_supply

async def _solscan_supply_and_top(
    token_address: str,
    session: aiohttp.ClientSession,
    cfg: Dict[str, Any],
    top_n: int,
    timeout: aiohttp.ClientTimeout,
) -> Optional[Tuple[float, float]]:
    p = (cfg.get("providers", {}) or {}).get("solscan_pro", {}) or {}
    api_key = p.get("api_key") or ""
    if not api_key:
        return None
    headers = {"token": api_key}

    holders_url = "https://pro-api.solscan.io/v2.0/token/holders"
    params = {"address": token_address, "page": 1, "page_size": int(max(1, top_n))}
    hdata = await _json_request(session, "GET", holders_url, headers=headers, params=params, timeout=timeout)
    if not hdata:
        return None
    hd = hdata.get("data") or {}
    items = hd.get("items") or []
    decimals = hd.get("decimals")
    top_sum_raw = 0.0
    for it in items:
        amt = it.get("amount")
        if amt is None:
            continue
        top_sum_raw += _num(amt, 0.0)

    info_url = "https://pro-api.solscan.io/v2.0/token/info"
    params_info = {"address": token_address}
    idata = await _json_request(session, "GET", info_url, headers=headers, params=params_info, timeout=timeout)
    if not idata:
        return None
    idd = idata.get("data") or {}
    supply_raw = idd.get("supply")
    dec = idd.get("decimals") if idd.get("decimals") is not None else decimals
    if supply_raw is None or dec is None:
        return None
    scale = 10 ** int(dec)
    total_supply_units = _num(supply_raw, 0.0) / scale
    top_sum_units = top_sum_raw / scale
    if total_supply_units <= 0:
        return None
    return top_sum_units, total_supply_units

async def _helius_supply_and_top(
    token_address: str,
    session: aiohttp.ClientSession,
    cfg: Dict[str, Any],
    top_n: int,
    timeout: aiohttp.ClientTimeout,
) -> Optional[Tuple[float, float]]:
    p = (cfg.get("providers", {}) or {}).get("helius", {}) or {}
    api_key = p.get("api_key") or ""
    if not api_key:
        return None
    url = p.get("url") or f"https://mainnet.helius-rpc.com/?api-key={api_key}"
    method_top = p.get("method_top") or "getTokenTopHolders"
    method_summary = p.get("method_summary") or "getTokenHoldersSummary"

    payload_top = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method_top,
        "params": {"mint": token_address, "limit": int(max(1, top_n))},
    }
    tdata = await _json_request(session, "POST", url, json_body=payload_top, timeout=timeout)
    if not tdata:
        return None
    titems = (tdata or {}).get("result") or []
    top_sum_units = 0.0
    for it in titems:
        qty = _best_first(it.get("uiAmount"), it.get("amountUi"))
        if qty is None:
            continue
        top_sum_units += _num(qty, 0.0)

    payload_sum = {"jsonrpc": "2.0", "id": 2, "method": method_summary, "params": {"mint": token_address}}
    sdata = await _json_request(session, "POST", url, json_body=payload_sum, timeout=timeout)
    if not sdata:
        return None
    result = (sdata or {}).get("result") or {}
    supply_units = _best_first(result.get("supplyUi"), result.get("supply"), result.get("circulatingUi"))
    if supply_units is None:
        return None
    total_supply_units = _num(supply_units, 0.0)
    if total_supply_units <= 0:
        return None
    return top_sum_units, total_supply_units

async def _public_solscan_supply_and_top(
    token_address: str,
    session: aiohttp.ClientSession,
    top_n: int,
    timeout: aiohttp.ClientTimeout,
) -> Optional[Tuple[float, float]]:
    holders_url = "https://public-api.solscan.io/token/holders"
    params = {"tokenAddress": token_address, "limit": int(max(1, top_n)), "offset": 0}
    hdata = await _json_request(session, "GET", holders_url, params=params, timeout=timeout)
    if not hdata:
        return None
    data = hdata
    if isinstance(hdata, dict) and isinstance(hdata.get("data"), dict):
        items = hdata["data"].get("items")
    else:
        items = (data.get("holders") or data.get("items") or []) if isinstance(data, dict) else []
    top_sum_units = 0.0
    for it in items or []:
        qty = _best_first(it.get("uiAmount"), it.get("amountUi"), it.get("amount"))
        if qty is None:
            continue
        top_sum_units += _num(qty, 0.0)

    info_url = "https://public-api.solscan.io/token/meta"
    params2 = {"tokenAddress": token_address}
    idata = await _json_request(session, "GET", info_url, params=params2, timeout=timeout)
    if not idata:
        return None
    supply_units = _best_first(
        (idata.get("data") or {}).get("supplyUi") if isinstance(idata.get("data"), dict) else None,
        idata.get("supplyUi"),
        idata.get("supply"),
    )
    if supply_units is None:
        return None
    total_supply_units = _num(supply_units, 0.0)
    if total_supply_units <= 0:
        return None
    return top_sum_units, total_supply_units

async def fetch_top_holder_concentration(
    token_address: str,
    session: aiohttp.ClientSession,
    config: Dict[str, Any],
    top_n: int,
) -> Tuple[Optional[float], str]:
    cache_key = (token_address, int(max(1, top_n)))
    cached = _conc_cache.get(cache_key)
    if isinstance(cached, (int, float)):
        return float(cached), "cache"

    hcfg = config.get("holders", {}) or {}
    timeout_sec = int(hcfg.get("timeout_seconds", 8))
    timeout = aiohttp.ClientTimeout(total=timeout_sec, sock_connect=5, sock_read=timeout_sec)

    pair = await _birdeye_supply_and_top(token_address, session, hcfg, top_n, timeout)
    if pair:
        top_sum, supply = pair
        pct = (top_sum / supply) * 100.0
        _conc_cache[cache_key] = pct
        return pct, "birdeye"

    pair = await _solscan_supply_and_top(token_address, session, hcfg, top_n, timeout)
    if pair:
        top_sum, supply = pair
        pct = (top_sum / supply) * 100.0
        _conc_cache[cache_key] = pct
        return pct, "solscan_pro"

    pair = await _helius_supply_and_top(token_address, session, hcfg, top_n, timeout)
    if pair:
        top_sum, supply = pair
        pct = (top_sum / supply) * 100.0
        _conc_cache[cache_key] = pct
        return pct, "helius"

    if (hcfg.get("providers", {}).get("public_solscan", {}) or {}).get("enabled", True):
        pair = await _public_solscan_supply_and_top(token_address, session, top_n, timeout)
        if pair:
            top_sum, supply = pair
            pct = (top_sum / supply) * 100.0
            _conc_cache[cache_key] = pct
            return pct, "public_solscan"

    return None, "unavailable"

def _max_conc_for_categories(categories: List[str], config: Dict[str, Any]) -> Optional[float]:
    ccfg = config.get("concentration", {}) or {}
    per = ccfg.get("max_percent_per_category", {}) or {}
    caps: List[float] = []
    for cat in categories:
        v = per.get(cat)
        if isinstance(v, (int, float)):
            caps.append(float(v))
    if caps:
        return float(min(caps))
    global_cap = ccfg.get("max_percent")
    if isinstance(global_cap, (int, float)):
        return float(global_cap)
    return None

# -----------------------------------------------------------------------------#
# Eligibility (with holders + concentration)  — plus single-primary category
# -----------------------------------------------------------------------------#

def _primary_category_list(categories: List[str]) -> List[str]:
    """
    Reduce any list of category tags to a single primary tag.
    Precedence: newly_launched > large_cap > mid_cap > low_cap > unknown_cap
    """
    s = {c.strip().lower() for c in categories or []}
    if "newly_launched" in s:
        return ["newly_launched"]
    if "large_cap" in s:
        return ["large_cap"]
    if "mid_cap" in s:
        return ["mid_cap"]
    if "low_cap" in s:
        return ["low_cap"]
    return ["unknown_cap"] if s else ["unknown_cap"]


async def is_token_eligible(
    token: Dict[str, Any],
    session: aiohttp.ClientSession,
    config: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Check if a token is eligible for trading based on discovery criteria, RugCheck (optional at discovery),
    and optional holders/concentration policies.
    Returns: (eligible: bool, [single_primary_category])
    """
    token_address = token.get("address")
    if not token_address:
        logger.warning("Token missing address: %s", token)
        return False, []

    # Immediately ignore WSOL/stables in category selection flows
    if token_address in ALWAYS_HIDE_IN_CATEGORIES:
        logger.debug("Ignoring stable/WSOL token in eligibility flow: %s", token_address)
        return False, []

    categories: List[str] = []
    reasons: List[str] = []

    # --- Dexscreener / Birdeye fallbacks ---
    ds = token.get("dexscreener") or {}
    be = token.get("birdeye") or {}

    # Normalize core fields (prefer normalized, then Birdeye, then Dexscreener)
    market_cap = _num(_best_first(
        token.get("mc"),
        token.get("fdv"),
        token.get("market_cap"),
        be.get("market_cap") if isinstance(be, dict) else None,
        ds.get("fdv"),
        ds.get("mcap"),
        ds.get("marketCap"),
    ), 0)

    liquidity = _num(_best_first(
        token.get("liquidity"),
        token.get("liquidity_usd"),
        ds.get("liquidityUsd"),
        (ds.get("liquidity") or {}).get("usd") if isinstance(ds.get("liquidity"), dict) else None,
    ), 0)

    volume_24h = _num(_best_first(
        token.get("volume_24h"),
        token.get("v24hUSD"),
        token.get("volume"),
        ds.get("v24hUsd") or ds.get("v24hUSD"),
        ds.get("volume24h") or ds.get("h24"),
        ds.get("volume"),
    ), 0)

    # creation: accept seconds or ms; also accept Dexscreener pairCreatedAt
    creation_ts_sec = _best_first(
        token.get("creation_timestamp"),
        (token.get("pairCreatedAt") or 0) / 1000.0 if token.get("pairCreatedAt") else None,
        (ds.get("pairCreatedAt") or 0) / 1000.0 if ds.get("pairCreatedAt") else None,
    )
    creation_ts_sec = _num(creation_ts_sec or 0)
    pair_created_at_ms = int(creation_ts_sec * 1000) if creation_ts_sec else 0

    # Assign categories by market cap
    try:
        if market_cap > 0:
            if market_cap < _num(config["discovery"]["low_cap"]["max_market_cap"], 1e6):
                categories.append("low_cap")
            elif market_cap < _num(config["discovery"]["mid_cap"]["max_market_cap"], 1e8):
                categories.append("mid_cap")
            else:
                categories.append("large_cap")
    except Exception:
        pass

    # "newly_launched" by age
    if pair_created_at_ms:
        age_minutes = (time.time() - pair_created_at_ms / 1000.0) / 60.0
        try:
            max_new_age = _num(config["discovery"]["newly_launched"]["max_token_age_minutes"], 180.0)
        except Exception:
            max_new_age = 180.0
        if age_minutes <= max_new_age:
            categories.append("newly_launched")

    # If nothing yet, keep token visible downstream
    if not categories:
        categories = ["unknown_cap"]

    # Reduce to a single primary category *before* thresholds/logics below
    categories = _primary_category_list(categories)

    # ---------------------- Guardrails: market cap ----------------------
    _min_mc = 1000  # default; can be overridden via config["discovery"]["min_market_cap"]
    try:
        _min_mc = _num(config.get("discovery", {}).get("min_market_cap", 1000), 1000)
    except Exception:
        pass

    if market_cap > 0 and market_cap < _min_mc:
        reason = f"market_cap={market_cap} too low (<{_min_mc})"
        reasons.append(reason)
        _log_reject(token_address, reason)
        token = dict(token); token["eligibility_reasons"] = reasons
        return False, categories

    # Optional upper bound sanity
    try:
        mid_cap_max = _num(config["discovery"]["mid_cap"]["max_market_cap"], 1e8)
        if market_cap > 0 and market_cap > mid_cap_max and categories == ["mid_cap"]:
            reason = f"market_cap={market_cap} too high (>mid_cap_max={mid_cap_max})"
            reasons.append(reason)
            _log_reject(token_address, reason)
            token = dict(token); token["eligibility_reasons"] = reasons
            return False, categories
    except Exception:
        pass

    # ---------------------- Guardrails: liquidity ----------------------
    def _liquidity_threshold(cat: str) -> float:
        try:
            v = config["discovery"][cat].get("liquidity_threshold")
            return _num(v, MIN_LIQ_USD_ENV)
        except Exception:
            return MIN_LIQ_USD_ENV

    cat = categories[0]
    if cat == "low_cap" and liquidity < _liquidity_threshold("low_cap"):
        reason = f"liquidity={liquidity} too low for low_cap"
        reasons.append(reason)
        _log_reject(token_address, reason)
        token = dict(token); token["eligibility_reasons"] = reasons
        return False, categories
    if cat == "mid_cap" and liquidity < _liquidity_threshold("mid_cap"):
        reason = f"liquidity={liquidity} too low for mid_cap"
        reasons.append(reason)
        _log_reject(token_address, reason)
        token = dict(token); token["eligibility_reasons"] = reasons
        return False, categories
    if cat == "large_cap" and liquidity < _liquidity_threshold("large_cap"):
        reason = f"liquidity={liquidity} too low for large_cap"
        reasons.append(reason)
        _log_reject(token_address, reason)
        token = dict(token); token["eligibility_reasons"] = reasons
        return False, categories
    if cat == "newly_launched":
        try:
            nl_th = _num(config["discovery"]["newly_launched"].get("liquidity_threshold"), MIN_LIQ_USD_ENV)
        except Exception:
            nl_th = MIN_LIQ_USD_ENV
        if liquidity < nl_th:
            reason = f"liquidity={liquidity} too low for newly_launched"
            reasons.append(reason)
            _log_reject(token_address, reason)
            token = dict(token); token["eligibility_reasons"] = reasons
            return False, categories

    # ---------------------- Guardrails: volume ----------------------
    def _volume_threshold(cat: str) -> float:
        try:
            v = config["discovery"][cat]["volume_threshold"]
            return _num(v, MIN_VOL24_USD_ENV)
        except Exception:
            return MIN_VOL24_USD_ENV

    if cat == "low_cap" and volume_24h < _volume_threshold("low_cap"):
        reason = f"volume_24h={volume_24h} too low for low_cap"
        reasons.append(reason)
        _log_reject(token_address, reason)
        token = dict(token); token["eligibility_reasons"] = reasons
        return False, categories
    if cat == "mid_cap" and volume_24h < _volume_threshold("mid_cap"):
        reason = f"volume_24h={volume_24h} too low for mid_cap"
        reasons.append(reason)
        _log_reject(token_address, reason)
        token = dict(token); token["eligibility_reasons"] = reasons
        return False, categories
    if cat == "large_cap" and volume_24h < _volume_threshold("large_cap"):
        reason = f"volume_24h={volume_24h} too low for large_cap"
        reasons.append(reason)
        _log_reject(token_address, reason)
        token = dict(token); token["eligibility_reasons"] = reasons
        return False, categories
    if cat == "newly_launched" and volume_24h < _volume_threshold("newly_launched"):
        reason = f"volume_24h={volume_24h} too low for newly_launched"
        reasons.append(reason)
        _log_reject(token_address, reason)
        token = dict(token); token["eligibility_reasons"] = reasons
        return False, categories

    
    # ---------------------- Guardrail: recent momentum (1h/6h) ----------------------
    recent_ok, recent_reason = _recent_momentum_allowed(token, cfg=config)
    if not recent_ok:
        reason = recent_reason
        reasons.append(reason)
        _log_reject(token_address, reason)
        token = dict(token); token["eligibility_reasons"] = reasons
        return False, categories
    else:
        token.setdefault("eligibility_reasons", [])
        token["eligibility_reasons"].append(f"recent_momentum_ok: {recent_reason}")
    # ---------------------------------------------------------------------------
# ---------------------- Guardrail: price change ----------------------
    # Replaced unconditional rejection with conditional logic that favors real runners when depth exists.
    price_change_24h = _num(_best_first(
        token.get("price_change_24h"),
        token.get("priceChange24h"),
        ds.get("priceChange24h"),
        ds.get("priceChange"),
    ), 0)
    allowed, pc_reason = _price_change_allowed(token, price_change_24h, cfg=config)
    if not allowed:
        reason = pc_reason
        reasons.append(reason)
        _log_reject(token_address, reason)
        token = dict(token); token["eligibility_reasons"] = reasons
        return False, categories
    else:
        # annotate reason for visibility (helps audits)
        token.setdefault("eligibility_reasons", [])
        token["eligibility_reasons"].append(f"price_change_ok: {pc_reason}")

    # ---------------------- Rugcheck at discovery (optional) ----------------------
    disc_cfg = (config.get("discovery") or {})
    use_rc = get_env_bool("RUGCHECK_DISCOVERY_CHECK", False) or bool(disc_cfg.get("rugcheck_in_discovery", False))
    filter_rc = get_env_bool("RUGCHECK_DISCOVERY_FILTER", False)
    if disc_cfg.get("require_rugcheck_pass", None) is not None:
        filter_rc = bool(disc_cfg.get("require_rugcheck_pass"))
        if filter_rc:
            use_rc = True

    if not use_rc:
        token.setdefault("safety", token.get("safety", "unknown"))
        token.setdefault("dangerous", bool(token.get("dangerous", False)))
        logger.debug("Rugcheck skipped at discovery for %s (safety=%s, dangerous=%s)",
                     token_address, token.get("safety"), token.get("dangerous"))
    else:
        risk_score, risk_labels_raw, _ = await verify_token_with_rugcheck(
            token_address, token, session, config
        )
        risk_label_names = _label_names_from_any(risk_labels_raw)

        token["rugcheck_labels"] = risk_label_names
        token["rugcheck_score"] = float(risk_score)
        token["safety"] = "ok"
        token["dangerous"] = False

        if filter_rc:
            block_dangerous_env = get_env_bool("BLOCK_RUGCHECK_DANGEROUS", True)
            block_dangerous_cfg = disc_cfg.get("require_rugcheck_pass", None)
            block_dangerous = block_dangerous_cfg if block_dangerous_cfg is not None else block_dangerous_env

            danger_blocklist: Set[str] = {"dangerous", "scam", "honeypot"}
            try:
                cfg_labels = (config.get("rugcheck", {}) or {}).get("danger_labels_blocklist")
                if isinstance(cfg_labels, list) and cfg_labels:
                    danger_blocklist = {_lower(x) for x in (set(cfg_labels) | danger_blocklist)}
            except Exception as e:
                logger.warning("Failed to load rugcheck.danger_labels_blocklist: %s", e)

            if block_dangerous:
                hit = set(risk_label_names) & danger_blocklist
                if hit:
                    msg = f"Dangerous label(s): {sorted(hit)}"
                    _log_reject(token_address, msg)
                    await add_to_blacklist(token_address, msg)
                    token["safety"] = "dangerous"
                    token["dangerous"] = True
                    token = dict(token); token["eligibility_reasons"] = [msg]
                    return False, categories

            def _max_rug(cat: str, default_val: float = 80.0) -> float:
                try:
                    return _num(config["discovery"][cat].get("max_rugcheck_score", default_val), default_val)
                except Exception:
                    return float(default_val)

            if cat == "newly_launched":
                max_rug_score = _max_rug("newly_launched")
            elif cat == "mid_cap":
                max_rug_score = _max_rug("mid_cap")
            elif cat == "large_cap":
                max_rug_score = _max_rug("large_cap")
            else:
                max_rug_score = _max_rug("low_cap")

            if risk_score > max_rug_score:
                msg = f"RugCheck high risk (score={risk_score} >= limit={max_rug_score})"
                _log_reject(token_address, msg)
                await add_to_blacklist(token_address, msg)
                token["safety"] = "dangerous"
                token["dangerous"] = True
                token = dict(token); token["eligibility_reasons"] = [msg]
                return False, categories

        if ("dangerous" in risk_label_names) or ("scam" in risk_label_names) or ("honeypot" in risk_label_names):
            token["safety"] = "dangerous"
            token["dangerous"] = True
        elif risk_score >= 1e6:
            token["safety"] = "unknown"
            token["dangerous"] = False
        else:
            token["safety"] = "ok"
            token["dangerous"] = False

    # -------------------- Holders / Concentration (unchanged) --------------------
    hcfg = config.get("holders", {}) or {}
    if hcfg.get("enabled", True):
        min_required = _min_holders_for_categories(categories, config)
        if isinstance(min_required, int):
            allow_if_down = bool(hcfg.get("allow_if_api_down", True))
            holders_count, source_h = await fetch_holder_count(token_address, session, config)
            if holders_count is None:
                note = f"holders unavailable (src={source_h})"
                if allow_if_down:
                    logger.warning(
                        "Holder count unavailable for %s; proceeding due to allow_if_api_down",
                        token_address,
                    )
                    categories.append("pending_holder_count")
                else:
                    _log_reject(token_address, note)
                    await add_to_blacklist(token_address, note)
                    token = dict(token); token["eligibility_reasons"] = [note]
                    return False, categories
            else:
                token["holders"] = holders_count
                token["holders_source"] = source_h
                if holders_count < min_required:
                    msg = f"holders={holders_count} < min_required={min_required} (src={source_h})"
                    _log_reject(token_address, msg)
                    await add_to_blacklist(token_address, msg)
                    token = dict(token); token["eligibility_reasons"] = [msg]
                    return False, categories

    ccfg = config.get("concentration", {}) or {}
    if ccfg.get("enabled", False):
        top_n = int(max(1, ccfg.get("top_n", 10)))
        allow_if_down_c = bool(ccfg.get("allow_if_api_down", True))
        max_pct_allowed = _max_conc_for_categories(categories, config)

        pct, source_c = await fetch_top_holder_concentration(token_address, session, config, top_n)
        if pct is None:
            note = f"concentration unavailable (src={source_c})"
            if not allow_if_down_c:
                _log_reject(token_address, note)
                await add_to_blacklist(token_address, note)
                token = dict(token); token["eligibility_reasons"] = [note]
                return False, categories
            else:
                logger.warning(
                    "Concentration unavailable for %s; proceeding due to allow_if_api_down",
                    token_address,
                )
        else:
            token["topN_concentration_pct"] = pct
            token["topN_concentration_source"] = source_c
            if isinstance(max_pct_allowed, (int, float)) and pct > float(max_pct_allowed):
                msg = f"top{top_n}_concentration={pct:.2f}% > limit={float(max_pct_allowed):.2f}% (src={source_c})"
                _log_reject(token_address, msg)
                await add_to_blacklist(token_address, msg)
                token = dict(token); token["eligibility_reasons"] = [msg]
                return False, categories

    logger.debug("ELIGIBLE %s | category=%s", token_address, categories)
    return True, categories


# -----------------------------------------------------------------------------#
# Scoring + Selection
# -----------------------------------------------------------------------------#
def score_token(token: Dict[str, Any], config: Dict[str, Any]) -> float:
    """
    Score a token based on various metrics (field names normalized).
    Returns [0,1].

    Note: price-change contribution is amplified when market depth indicates a real runner.
    """
    ds = token.get("dexscreener") or {}
    be = token.get("birdeye") or {}

    market_cap = _num(_best_first(
        token.get("mc"),
        token.get("fdv"),
        token.get("market_cap"),
        be.get("market_cap") if isinstance(be, dict) else None,
        ds.get("fdv"),
        ds.get("mcap"),
        ds.get("marketCap"),
    ), 0)

    liquidity = _num(_best_first(
        token.get("liquidity"),
        token.get("liquidity_usd"),
        ds.get("liquidityUsd"),
        (ds.get("liquidity") or {}).get("usd") if isinstance(ds.get("liquidity"), dict) else None,
    ), 0)

    volume_24h = _num(_best_first(
        token.get("volume_24h"),
        token.get("v24hUSD"),
        token.get("volume"),
        ds.get("v24hUsd") or ds.get("v24hUSD"),
        ds.get("volume24h") or ds.get("h24"),
        ds.get("volume"),
    ), 0)

    # Treat negative 24h % as zero for scoring (we don't want negative momentum to be rewarded).
    price_change_24h = _num(_best_first(
        token.get("price_change_24h"),
        token.get("priceChange24h"),
        ds.get("priceChange24h"),
        ds.get("priceChange"),
    ), 0.0)
    price_change_24h = max(0.0, price_change_24h)

    weights = config.get(
        "weights",
        {
            "market_cap": 0.3,
            "liquidity": 0.2,
            "volume": 0.3,
            "price_change": 0.2,
            "new_token_bonus": 0.3,
        },
    )

    score = 0.0

    max_market_cap = _num(
        config.get("discovery", {}).get("mid_cap", {}).get("max_market_cap", 1e9),
        1e9,
    )
    if market_cap > 0 and max_market_cap > 0:
        score += (max(0.0, max_market_cap - market_cap) / max_market_cap) * weights.get("market_cap", 0.3)

    max_liquidity = max(
        _num(config.get("discovery", {}).get("low_cap", {}).get("liquidity_threshold", 1.0), 1.0),
        _num(config.get("discovery", {}).get("mid_cap", {}).get("liquidity_threshold", 1.0), 1.0),
        _num(config.get("discovery", {}).get("large_cap", {}).get("liquidity_threshold", 1.0), 1.0),
    )
    if liquidity > 0 and max_liquidity > 0:
        score += min(liquidity / max_liquidity, 1.0) * weights.get("liquidity", 0.2)

    max_volume = max(
        _num(config.get("discovery", {}).get("low_cap", {}).get("volume_threshold", 1.0), 1.0),
        _num(config.get("discovery", {}).get("mid_cap", {}).get("volume_threshold", 1.0), 1.0),
        _num(config.get("discovery", {}).get("large_cap", {}).get("volume_threshold", 1.0), 1.0),
    )
    if volume_24h > 0 and max_volume > 0:
        score += min(volume_24h / max_volume, 1.0) * weights.get("volume", 0.3)

    # Price change scoring: encourage real runners.
    max_price_change_allowed = _num(config.get("discovery", {}).get("max_price_change", 350), 350)
    # Depth amplification: if token has decent depth amplify price-change contribution
    min_liq_for_big_pct = _num(config.get("discovery", {}).get("min_liq_for_big_pct", 5000.0), 5000.0)
    min_vol_for_big_pct = _num(config.get("discovery", {}).get("min_vol_for_big_pct", 10000.0), 10000.0)
    depth_multiplier = 1.0
    if liquidity >= min_liq_for_big_pct or volume_24h >= min_vol_for_big_pct:
        depth_multiplier = 1.5  # boost price-change influence for tokens with real market depth

    if price_change_24h > 0 and max_price_change_allowed > 0:
        pct_score = min(price_change_24h / max_price_change_allowed, 1.0) * weights.get("price_change", 0.2)
        score += pct_score * depth_multiplier

    if "newly_launched" in token.get("categories", []):
        score += weights.get("new_token_bonus", 0.0)

    return max(0.0, min(score, 1.0))

# -----------------------------------------------------------------------------#
# Shortlisting (canonical buckets; no alias lists that cause UI duplication)
# -----------------------------------------------------------------------------#

# (1) Canonical bucket synonyms
_BUCKET_SYNONYMS: Dict[str, Set[str]] = {
    "high": {"high", "high_cap", "large", "large_cap"},
    "mid": {"mid", "mid_cap", "medium", "medium_cap"},
    "low": {"low", "low_cap", "small", "small_cap"},
    "new": {"new", "newly_launched", "newly_listed", "new_tokens"},
}

def _canon_bucket_for(categories: List[str]) -> Optional[str]:
    if not categories:
        return None
    cats = {str(c).lower().strip() for c in categories}
    for canon, syns in _BUCKET_SYNONYMS.items():
        if cats & syns:
            return canon
    return None

# (2) Defaults that keep obvious stables/wrappers out of discovery
_DEFAULT_EXCLUDE_SYMBOLS: Set[str] = {
    "SOL", "WSOL", "W SOL", "wSOL",
    "USDC", "USDT",
    "Wrapped SOL", "Wrapped Solana",
}
_DEFAULT_EXCLUDE_NAMES: Set[str] = {"wrapped solana", "wrapped sol"}  # substr match (lowercase)
_DEFAULT_EXCLUDE_ADDR: Set[str] = {
    # canonical Solana wrappers / stables
    "So11111111111111111111111111111111111111112",  # wSOL
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
}

def _mc(tok: Dict[str, Any]) -> float:
    return _num(
        tok.get("mc")
        or tok.get("market_cap")
        or tok.get("fdv")
        or tok.get("fdvUsd")
        or tok.get("marketCap")
        or 0.0
    )

def _liq(tok: Dict[str, Any]) -> float:
    v = tok.get("liquidity")
    if isinstance(v, dict):
        return _num(v.get("usd"), 0.0)
    return _num(v, 0.0)

def _vol24(tok: Dict[str, Any]) -> float:
    v = tok.get("volume_24h")
    if v is not None:
        return _num(v, 0.0)
    v = tok.get("volume")
    if isinstance(v, dict):
        return _num(v.get("h24"), 0.0)
    return _num(v, 0.0)

def _created_s(tok: Dict[str, Any]) -> int:
    raw = tok.get("creation_timestamp") or tok.get("pairCreatedAt") or tok.get("createdAt") or 0
    ts = _num(raw, 0.0)
    if ts > 10**12:  # ms -> s
        ts = ts / 1000.0
    return int(ts or 0)

# (4) Config readers
def _cfg() -> Dict[str, Any]:
    try:
        return load_config() or {}
    except Exception:
        return {}

def _disc(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return (cfg.get("discovery") or {})

def _per_bucket_limit(cfg: Dict[str, Any], default: int = 5) -> int:
    d = _disc(cfg)
    v = d.get("shortlist_per_bucket", default)
    try:
        return max(1, int(v))
    except Exception:
        return default

def _rules(cfg: Dict[str, Any]) -> Dict[str, Any]:
    d = _disc(cfg)
    return {
        "new": {
            "max_age_min": _num((d.get("newly_launched") or {}).get("max_token_age_minutes"), 180),
            "liq_min":      _num((d.get("newly_launched") or {}).get("liquidity_threshold"), 100),
            "vol_min":      _num((d.get("newly_launched") or {}).get("volume_threshold"), 50),
        },
        "low": {
            "mc_max": _num((d.get("low_cap") or {}).get("max_market_cap"), 100_000),
            "liq_min": _num((d.get("low_cap") or {}).get("liquidity_threshold"), 50),
            "vol_min": _num((d.get("low_cap") or {}).get("volume_threshold"), 50),
        },
        "mid": {
            "mc_max": _num((d.get("mid_cap") or {}).get("max_market_cap"), 500_000),
            "liq_min": _num((d.get("mid_cap") or {}).get("liquidity_threshold"), 300),
            "vol_min": _num((d.get("mid_cap") or {}).get("volume_threshold"), 100),
        },
        "high": {
            "liq_min": _num((d.get("large_cap") or {}).get("liquidity_threshold"), 1000),
            "vol_min": _num((d.get("large_cap") or {}).get("volume_threshold"), 500),
        },
    }

# (5) Global de-dup
def _better_row(new_t: Dict[str, Any], old_t: Dict[str, Any]) -> bool:
    a = (_liq(new_t), _vol24(new_t), _mc(new_t))
    b = (_liq(old_t), _vol24(old_t), _mc(old_t))
    return a > b

def _dedupe_best_by_address(tokens: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    merges = 0
    for t in tokens:
        addr = (t.get("address") or t.get("token_address") or "").strip()
        if not addr:
            continue
        prev = best.get(addr)
        if prev is None:
            best[addr] = t
        else:
            merges += 1
            best[addr] = t if _better_row(t, prev) else prev
    if merges:
        logger.info("Eligibility: merged %d duplicate token rows by address.", merges)
    return best

# (6) Build exclude lists (config + defaults + constants if available)
def _exclude_filters(cfg: Dict[str, Any]) -> Dict[str, Set[str]]:
    d = _disc(cfg)
    sym_cfg = {s.strip() for s in (d.get("exclude_symbols") or []) if s and isinstance(s, str)}
    addr_cfg = {s.strip() for s in (d.get("exclude_addresses") or []) if s and isinstance(s, str)}
    # merge with defaults
    sym = set(_DEFAULT_EXCLUDE_SYMBOLS) | sym_cfg
    addr = set(_DEFAULT_EXCLUDE_ADDR) | addr_cfg

    # IMPORTANT: do NOT union a “whitelist” here; we only want to hide those from High-Cap Top-5,
    # which is filtered later in solana_trading_bot.run_discovery_cycle(...)
    # (i.e., whitelist ≠ exclude)

    try:
        # Keep canonical stables/wrappers out of discovery buckets
        addr |= set(ALWAYS_HIDE_IN_CATEGORIES)  # type: ignore
    except Exception:
        pass

    return {"symbols": sym, "addresses": addr}


# (7) One canonical category per token
def _primary_category(tok: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[str]:
    cat = _canon_bucket_for(tok.get("categories") or [])
    if cat in ("new", "low", "mid", "high"):
        return cat

    rules = _rules(cfg)
    now_s = int(time.time())
    created = _created_s(tok)
    age_min = None if not created else max(0.0, (now_s - created) / 60.0)

    liq = _liq(tok)
    vol = _vol24(tok)
    mc  = _mc(tok)

    r_new = rules["new"]
    if age_min is not None and age_min <= r_new["max_age_min"]:
        if liq >= r_new["liq_min"] and vol >= r_new["vol_min"]:
            return "new"

    r_low, r_mid, r_high = rules["low"], rules["mid"], rules["high"]
    if mc <= r_low["mc_max"]:
        if liq >= r_low["liq_min"] and vol >= r_low["vol_min"]:
            return "low"
        return None
    if mc <= r_mid["mc_max"]:
        if liq >= r_mid["liq_min"] and vol >= r_mid["vol_min"]:
            return "mid"
        return None

    if liq >= r_high["liq_min"] and vol >= r_high["vol_min"]:
        return "high"

    return None

def select_top_five_per_category(
    tokens: List[Dict[str, Any]],
    per_bucket: int = 5,
    blacklist: Optional[Set[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    cfg = _cfg()
    limit = max(1, int(per_bucket or _per_bucket_limit(cfg, default=5)))
    excludes = _exclude_filters(cfg)
    exclude_syms = {s.lower() for s in excludes["symbols"]}
    exclude_names_lc = _DEFAULT_EXCLUDE_NAMES  # lower-case substrings
    exclude_addrs = excludes["addresses"]
    bl = set(blacklist or set())

    # honor env/consts if present
    try:
        bypass_stables = get_env_bool("BYPASS_STABLE_FILTER", False)  # type: ignore
    except Exception:
        bypass_stables = False

    prepped: List[Dict[str, Any]] = []
    dropped_by_exclude = 0

    for t in tokens or []:
        addr = (t.get("address") or t.get("token_address") or "").strip()
        if not addr or addr in bl:
            continue

        # Symbol/name based excludes
        sym = (t.get("symbol") or t.get("baseSymbol") or "").strip()
        name = (t.get("name") or t.get("baseName") or "").strip()
        sym_lc = sym.lower()
        name_lc = name.lower()

        excluded = False
        if not bypass_stables:
            if addr in exclude_addrs:
                excluded = True
            elif sym and sym_lc in exclude_syms:
                excluded = True
            elif any(substr in name_lc for substr in exclude_names_lc):
                excluded = True

        if excluded:
            dropped_by_exclude += 1
            continue

        # normalize numerics
        t = dict(t)
        t.setdefault("liquidity", _liq(t))
        t.setdefault("volume_24h", _vol24(t))
        if "mc" not in t:
            t["mc"] = _mc(t)

        # score (use your util if available)
        if "score" not in t or t["score"] is None:
            try:
                t["score"] = score_token(t, cfg)  # type: ignore
            except Exception:
                w = (cfg.get("weights") or {})
                t["score"] = (
                    _mc(t) * _num(w.get("market_cap"), 0.3)
                    + _liq(t) * _num(w.get("liquidity"), 0.2)
                    + _vol24(t) * _num(w.get("volume"), 0.3)
                    + _num(t.get("price_change_24h") or 0.0) * _num(w.get("price_change"), 0.2)
                )

        if not t.get("categories"):
            t["categories"] = ["unknown_cap"]

        prepped.append(t)

    if dropped_by_exclude:
        logger.info("Eligibility: excluded %d rows by symbol/name/address filters.", dropped_by_exclude)

    # de-dup globally
    best_by_addr = _dedupe_best_by_address(prepped)

    # bucket strictly by primary category (no cross-bucket padding)
    buckets: Dict[str, List[Dict[str, Any]]] = {"high": [], "mid": [], "low": [], "new": []}

    for tok in best_by_addr.values():
        canon = _primary_category(tok, cfg)
        if not canon:
            continue
        if len(buckets[canon]) < limit:
            buckets[canon].append(tok)

    logger.info(
        "Shortlist per-bucket (canonical): high=%d mid=%d low=%d new=%d",
        len(buckets["high"]), len(buckets["mid"]), len(buckets["low"]), len(buckets["new"]),
    )
    return buckets


# Back-compat async wrapper for older imports/call sites
async def shortlist_by_category(
    tokens: List[Dict[str, Any]],
    blacklist: Optional[Set[str]] = None,
    top_n: int = 5,
    new_minutes: int = 180,  # retained for signature compatibility; not used here
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Legacy async wrapper. Accepts either `top_n` or `per_bucket` (kwargs),
    and an optional `blacklist`. Returns the same as `select_top_five_per_category`.
    """
    per_bucket = int(kwargs.get("per_bucket", top_n))
    return select_top_five_per_category(tokens, per_bucket=per_bucket, blacklist=blacklist)

# ==== BEGIN FILTER-REASON SUMMARY PATCH ======================================
# This summarizes *why* candidates are rejected, so discovery "0 coins" is debuggable.
# Call log_filter_summary(...) from your orchestrator right after you build `eligible`.

@dataclass
class FilterStats:
    total_in: int = 0
    kept: int = 0
    reasons: Counter = field(default_factory=Counter)

    @property
    def rejected(self) -> int:
        return max(self.total_in - self.kept, 0)

    def add(self, reason: str) -> None:
        if reason:
            self.reasons[reason] += 1

def _infer_primary_bucket(tok: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """
    Use this module's canonical category logic to pick one of: new|low|mid|high.
    Falls back to thresholds if the token didn't carry categories.
    """
    # Reuse canonical logic defined above in this file
    canon = _primary_category(tok, cfg)  # returns "new"/"low"/"mid"/"high" or None
    return canon or "low"  # harmless fallback

def summarize_filter_outcomes(
    all_candidates: List[Dict[str, Any]],
    kept_candidates: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> FilterStats:
    stats = FilterStats()
    stats.total_in = len(all_candidates)
    kept_set = { (t.get("address") or t.get("token_address") or "").strip() for t in (kept_candidates or []) }
    stats.kept = len(kept_set)

    rules = _rules(cfg)  # uses your discovery thresholds per bucket

    for t in all_candidates or []:
        addr = (t.get("address") or t.get("token_address") or "").strip()
        if not addr:
            stats.add("missing_address")
            continue
        if addr in kept_set:
            continue  # not rejected

        # Determine which bucket’s thresholds to compare against
        bucket = _infer_primary_bucket(t, cfg)
        r = rules.get(bucket, {})
        liq_min = float(r.get("liq_min", 0))
        vol_min = float(r.get("vol_min", 0))

        # Numerics (reuse file helpers)
        liq = _liq(t)
        vol = _vol24(t)
        mc  = _mc(t)
        created_s = _created_s(t)  # 0 if unknown

        # --- market cap sanity (your discovery.min_market_cap, if set) ---
        min_mc = float((cfg.get("discovery") or {}).get("min_market_cap", 0) or 0)
        if mc > 0 and min_mc and mc < min_mc:
            stats.add("marketcap<min")

        # --- liquidity & volume thresholds per bucket ---
        if liq_min and liq < liq_min:
            stats.add(f"liquidity<{bucket}_min")
        if vol_min and vol < vol_min:
            stats.add(f"volume24h<{bucket}_min")

        # --- "new" age window (only if token lands in "new") ---
        if bucket == "new":
            max_age_min = float(((_disc(cfg).get("newly_launched") or {}).get("max_token_age_minutes") or 0) or 0)
            if created_s:
                age_min = max(0.0, (time.time() - created_s) / 60.0)
                if max_age_min and age_min > max_age_min:
                    stats.add("too_old_for_new")

        # --- price change guardrail (matches your discovery.max_price_change) ---
        pc = t.get("price_change_24h") or (t.get("dexscreener") or {}).get("priceChange24h") or 0
        try:
            pc = float(pc)
        except Exception:
            pc = 0.0
        max_pc = float((cfg.get("discovery") or {}).get("max_price_change", 350) or 350)
        if abs(pc) > max_pc:
            # We still record these; with the new logic some may be allowed if depth present.
            stats.add("price_change_24h>max")

        # --- Rugcheck filtering, if enabled at discovery ---
        disc = (cfg.get("discovery") or {})
        require_rc_pass = bool(disc.get("require_rugcheck_pass", False))
        if require_rc_pass:
            labels = {str(x).lower() for x in (t.get("rugcheck_labels") or [])}
            dangerous = bool(t.get("dangerous"))
            if dangerous or labels.intersection({"dangerous", "scam", "honeypot"}):
                stats.add("rugcheck_danger")
            # If RC not present at all
            if not labels and not t.get("rugcheck_score"):
                stats.add("rugcheck_missing")

        # --- cosmetic gaps that often cause later drops ---
        sym = (t.get("symbol") or t.get("baseSymbol") or "").strip()
        if not sym:
            stats.add("missing_symbol")
        src = t.get("source")
        if not src:
            stats.add("missing_source")

    return stats

def log_filter_summary(
    all_candidates: List[Dict[str, Any]],
    kept_candidates: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    logger: logging.Logger,
    *,
    label: str = "pre-shortlist",
) -> None:
    stats = summarize_filter_outcomes(all_candidates, kept_candidates, cfg)
    top_reasons = ", ".join(f"{k}:{v}" for k, v in stats.reasons.most_common(12))
    logger.info(
        "FILTER-SUMMARY [%s] in=%d kept=%d rejected=%d | %s",
        label, stats.total_in, stats.kept, stats.rejected, top_reasons or "no-reasons"
    )
# ==== END FILTER-REASON SUMMARY PATCH ========================================
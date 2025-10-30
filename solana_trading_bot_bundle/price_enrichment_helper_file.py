# solana_trading_bot_bundle/price_enrichment_helper_file.py
from __future__ import annotations

import os
import time
import asyncio
import aiohttp
import logging
from collections import deque
from typing import Dict, List, Set, TypedDict, Optional, Any

from solders.pubkey import Pubkey

from solana_trading_bot_bundle.trading_bot.utils import (
    load_config,
    format_market_cap,
    add_to_blacklist,
    price_cache,            # simple in-process cache (dict-like) shared across modules
    WHITELISTED_TOKENS,
)

logger = logging.getLogger("PriceEnrichment")

# ------------ Tunables (env overridable) ------------
# Hard caps to prevent long stalls per token
_ENRICH_TOKEN_TIMEOUT_S = float(os.getenv("PRICE_ENRICH_TOKEN_TIMEOUT_S", "6"))
# Tiny jitter used instead of long Retry-After sleeps on 429s
_SMALL_BACKOFF_S = float(os.getenv("PRICE_ENRICH_SMALL_BACKOFF_S", "0.2"))
# HTTP request timeouts (per request)
_HTTP_TOTAL_TIMEOUT_S = float(os.getenv("PRICE_ENRICH_HTTP_TIMEOUT_S", "8"))

# Cache tunables
_PRICE_CACHE_TTL_POS_S = float(os.getenv("PRICE_CACHE_TTL_S", "60"))     # positive (hit) TTL
_PRICE_CACHE_TTL_NEG_S = float(os.getenv("PRICE_CACHE_NEG_TTL_S", "10")) # negative (miss) TTL
_PRICE_CACHE_MAX_SIZE  = int(os.getenv("PRICE_CACHE_MAX_SIZE", "5000"))

# --- Dexscreener limits (RPM + budgets) ---
# Requests-per-minute across BOTH token+search endpoints
_DS_RPM = max(1, int(os.getenv("DEXSCREENER_RPM", os.getenv("DEX_RPM", "200"))))
_DS_MAX_CALLS_PER_CYCLE = max(0, int(os.getenv("DEXSCREENER_MAX_CALLS_PER_CYCLE", "400")))
_DS_MAX_CALLS_PER_RUN   = max(0, int(os.getenv("DEXSCREENER_MAX_CALLS_PER_RUN",   "5000")))

# --- Birdeye limits (RPS + budgets) ---
_BIRDEYE_RPS = max(1, int(os.getenv("BIRDEYE_RPS", "5")))
_BIRDEYE_MAX_CALLS_PER_CYCLE = max(0, int(os.getenv("BIRDEYE_MAX_CALLS_PER_CYCLE", "60")))
_BIRDEYE_MAX_CALLS_PER_RUN   = max(0, int(os.getenv("BIRDEYE_MAX_CALLS_PER_RUN",   "2000")))

# Shared headers
_UA = {"accept": "application/json", "User-Agent": "SOLOTradingBot/1.0"}

# Helper
def _bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    v = str(val).strip().lower()
    return v in {"1", "true", "yes", "on"}

# ------------------------ Rate-limit gates ------------------------
# Dexscreener: RPM sliding window over 60s
_ds_lock = asyncio.Lock()
_ds_recent = deque()              # timestamps (float) of last DS calls in 60s window
_ds_cycle_count = 0               # reset per enrichment cycle
_ds_run_count = 0                 # process lifetime

async def _ds_gate(reset_cycle: bool = False) -> bool:
    """Enforce Dexscreener RPM + per-cycle/run budgets. Return True if a call may proceed."""
    global _ds_recent, _ds_cycle_count, _ds_run_count
    if reset_cycle:
        _ds_cycle_count = 0
        return True

    # Budget checks first
    if _DS_MAX_CALLS_PER_RUN and _ds_run_count >= _DS_MAX_CALLS_PER_RUN:
        return False
    if _DS_MAX_CALLS_PER_CYCLE and _ds_cycle_count >= _DS_MAX_CALLS_PER_CYCLE:
        return False

    # RPM window
    async with _ds_lock:
        now = time.time()
        cutoff = now - 60.0
        # prune
        while _ds_recent and _ds_recent[0] < cutoff:
            _ds_recent.popleft()
        if len(_ds_recent) >= _DS_RPM:
            # sleep until the oldest call exits the 60s window
            sleep_for = 60.0 - (now - _ds_recent[0]) + 0.01
            await asyncio.sleep(max(0.01, sleep_for))
            # prune again
            now = time.time()
            cutoff = now - 60.0
            while _ds_recent and _ds_recent[0] < cutoff:
                _ds_recent.popleft()
        _ds_recent.append(time.time())

    _ds_cycle_count += 1
    _ds_run_count += 1
    return True

# Birdeye: simple RPS sliding window over 1s
_birdeye_lock = asyncio.Lock()
_birdeye_recent: list[float] = []     # timestamps (last 1s window for RPS)
_birdeye_cycle_count = 0              # reset per enrichment cycle
_birdeye_run_count = 0                # process lifetime

async def _birdeye_gate(reset_cycle: bool = False) -> bool:
    """
    Enforce Birdeye per-run cap, per-cycle cap, and RPS limit.
    Return True if we may perform a Birdeye call now; False if budget is exhausted.
    """
    global _birdeye_recent, _birdeye_cycle_count, _birdeye_run_count
    if reset_cycle:
        _birdeye_cycle_count = 0
        return True

    if _BIRDEYE_MAX_CALLS_PER_RUN and _birdeye_run_count >= _BIRDEYE_MAX_CALLS_PER_RUN:
        return False
    if _BIRDEYE_MAX_CALLS_PER_CYCLE and _birdeye_cycle_count >= _BIRDEYE_MAX_CALLS_PER_CYCLE:
        return False

    # RPS sliding window (1 second)
    async with _birdeye_lock:
        now = time.time()
        cutoff = now - 1.0
        _birdeye_recent = [t for t in _birdeye_recent if t >= cutoff]
        if len(_birdeye_recent) >= _BIRDEYE_RPS:
            sleep_for = 1.0 - (now - _birdeye_recent[0]) + 0.01
            await asyncio.sleep(max(0.01, sleep_for))
            # prune again after sleep
            now = time.time()
            cutoff = now - 1.0
            _birdeye_recent = [t for t in _birdeye_recent if t >= cutoff]
        _birdeye_recent.append(time.time())

    _birdeye_cycle_count += 1
    _birdeye_run_count += 1
    return True
# ------------------------------------------------------------------


class TokenData(TypedDict, total=False):
    address: str
    symbol: str
    price: float
    mc: float
    liquidity: float
    holderCount: int
    priceChange1h: float
    priceChange6h: float
    priceChange24h: float
    mcFormatted: str
    dexscreenerUrl: str
    dsPairAddress: str


def _url_from_ds_pair(chosen: Dict[str, Any]) -> str:
    """Build a Dexscreener URL from a pair object."""
    if not isinstance(chosen, dict):
        return ""
    url = chosen.get("url")
    if isinstance(url, str) and url.startswith("http"):
        return url
    pair_addr = chosen.get("pairAddress")
    if isinstance(pair_addr, str) and pair_addr:
        return f"https://dexscreener.com/solana/{pair_addr}"
    return ""


async def _best_solana_pair_from_list(
    pairs: List[Dict[str, Any]],
    token_address: str,
) -> Optional[Dict[str, Any]]:
    """
    Pick the highest-liquidity Solana pair where the mint appears as base OR quote.
    Return the chosen pair object.
    """
    best: Optional[Dict[str, Any]] = None
    best_liq = -1.0
    for p in pairs or []:
        try:
            if p.get("chainId") not in ("solana", "SOLANA", 101):
                continue
            base = (p.get("baseToken") or {}).get("address")
            quote = (p.get("quoteToken") or {}).get("address")
            if token_address not in {base, quote}:
                continue
            liq = p.get("liquidity")
            if isinstance(liq, dict):
                liq = float(liq.get("usd", 0) or 0)
            else:
                liq = float(liq or 0)
            if liq > best_liq:
                best_liq = liq
                best = p
        except Exception:
            continue
    return best


# ------------------------ Simple in-process cache ------------------------
def _now_monotonic() -> float:
    try:
        return asyncio.get_running_loop().time()
    except RuntimeError:
        # outside event loop (e.g., unit tests)
        return asyncio.get_event_loop().time()

def _cache_get(addr: str) -> Optional[Dict[str, Any] | None]:
    """Return cached normalized dict, or None if expired/missing.
    Note: returns None for both 'no entry' and 'negative cache' lookups;
    caller distinguishes by checking existence first.
    """
    try:
        entry = price_cache.get(addr)  # expected: { "ts": float, "data": dict|None, "neg": bool }
        if not entry:
            return None
        ts = float(entry.get("ts", 0))
        neg = bool(entry.get("neg", False))
        ttl = _PRICE_CACHE_TTL_NEG_S if neg else _PRICE_CACHE_TTL_POS_S
        if (_now_monotonic() - ts) <= ttl:
            # Return whatever was cached: dict (positive) or None (negative)
            return entry.get("data")
    except Exception:
        # any unexpected structure => treat as miss
        pass
    # stale or invalid → drop
    try:
        price_cache.pop(addr, None)
    except Exception:
        pass
    return None

def _cache_put(addr: str, data: Optional[Dict[str, Any]]) -> None:
    """Store normalized dict (or None for negative cache)."""
    try:
        if len(price_cache) >= _PRICE_CACHE_MAX_SIZE:
            # drop an arbitrary item (simple cap; LRU not required here)
            price_cache.pop(next(iter(price_cache)))
    except Exception:
        # if eviction fails, keep going
        pass
    try:
        price_cache[addr] = {
            "ts": _now_monotonic(),
            "data": data,
            "neg": data is None,
        }
    except Exception:
        # ignore cache write failures silently
        pass
# ------------------------------------------------------------------------


async def fetch_price_by_token_endpoint(
    session: aiohttp.ClientSession,
    token_address: str,
) -> Optional[Dict[str, Any]]:
    """
    GET https://api.dexscreener.com/latest/dex/tokens/{mint}
    Choose best SOL pair and normalize fields.
    """
    # Respect Dexscreener RPM/budgets
    can_call = await _ds_gate()
    if not can_call:
        logger.info("Dexscreener budget reached; skipping tokens endpoint for %s", token_address)
        return None

    try:
        async with session.get(
            f"https://api.dexscreener.com/latest/dex/tokens/{token_address}",
            headers=_UA,
            timeout=aiohttp.ClientTimeout(total=_HTTP_TOTAL_TIMEOUT_S),
        ) as resp:
            if resp.status != 200:
                logger.warning(
                    "Dexscreener tokens endpoint failed for %s: HTTP %s",
                    token_address, resp.status,
                )
                return None

            data = await resp.json(content_type=None)
            pairs = data.get("pairs") or []
            chosen = await _best_solana_pair_from_list(pairs, token_address)
            if not chosen:
                logger.warning("No SOL pair for %s via tokens endpoint", token_address)
                return None

            price = float(chosen.get("priceUsd", 0) or 0)
            pc = chosen.get("priceChange") or {}
            vol = chosen.get("volume") or {}
            ds_url = _url_from_ds_pair(chosen)

            return {
                "price": price,
                "priceChange1h": float(pc.get("h1", 0) or 0),
                "priceChange6h": float(pc.get("h6", 0) or 0),
                "priceChange24h": float(pc.get("h24", 0) or 0),
                "volume_24h": float((vol.get("h24", 0) if isinstance(vol, dict) else 0) or 0),
                "dexscreenerUrl": ds_url,
                "dsPairAddress": chosen.get("pairAddress") or "",
            }
    except Exception as e:
        logger.warning("Token endpoint fetch failed for %s: %s", token_address, e)
        return None


async def enrich_tokens_with_price_change(
    session: aiohttp.ClientSession,
    tokens: List[TokenData],
    logger: logging.Logger,
    blacklist: Set[str],
    failure_count: Dict[str, int],
) -> List[TokenData]:
    """
    Enrich tokens with price and deltas.
    Order: Dexscreener token endpoint → Birdeye (if key) → Dexscreener search → neutral fallback.
    Always returns a list.
    """
    logger.info("Enriching %d tokens with price change data", len(tokens))

    # Reset per-cycle budgets for both providers
    await _ds_gate(reset_cycle=True)
    await _birdeye_gate(reset_cycle=True)

    config = load_config()
    batch_size  = int((config.get("discovery", {}) or {}).get("batch_size", 2))
    batch_sleep = float((config.get("discovery", {}) or {}).get("batch_sleep", 1.0))

    # --- Birdeye enablement: read env here (AFTER load_dotenv has run in main) ---
    api_key_raw = os.getenv("BIRDEYE_API_KEY", "")
    api_key = (api_key_raw or "").strip()

    # Normalize bool-ish envs so "1"/"true"/"yes"/"on" all work
    flag_on    = _bool_env("BIRDEYE_ENABLE", default=bool(api_key))  # default ON if key exists
    forced_off = _bool_env("FORCE_DISABLE_BIRDEYE", default=False)

    # Effective decision (single source of truth for the whole process)
    birdeye_enabled = bool(api_key) and flag_on and not forced_off
    os.environ["BIRDEYE_ENABLE_EFFECTIVE"] = "1" if birdeye_enabled else "0"

    # One clear log line with the reason if disabled
    if not api_key:
        logger.warning("Birdeye disabled: no BIRDEYE_API_KEY in env.")
    elif forced_off:
        logger.warning("Birdeye disabled: FORCE_DISABLE_BIRDEYE=1.")
    elif not flag_on:
        logger.warning("Birdeye disabled: BIRDEYE_ENABLE not truthy (use 1/true/yes/on).")
    else:
        logger.info(
            "Birdeye enabled (key_len=%d, rps=%s, cycle_cap=%s, run_cap=%s)",
            len(api_key),
            os.getenv("BIRDEYE_RPS", "?"),
            os.getenv("BIRDEYE_MAX_CALLS_PER_CYCLE", "?"),
            os.getenv("BIRDEYE_MAX_CALLS_PER_RUN", "?"),
        )

    birdeye_broken = False  # trip to True on first 401

    results: List[TokenData] = []

    async def _normalize_and_fill(
        token: TokenData,
        price: float | None,
        pc1h: float | None,
        pc6h: float | None,
        pc24h: float | None,
        ds_url: str = "",
        ds_pair: str = "",
    ) -> TokenData:
        out: TokenData = dict(token)
        out["price"] = float(price or 0.0)
        out["priceChange1h"] = float(pc1h or 0.0)
        out["priceChange6h"] = float(pc6h or 0.0)
        out["priceChange24h"] = float(pc24h or 0.0)
        out["mc"] = float(token.get("mc", 0) or 0)
        out["liquidity"] = float(token.get("liquidity", 0) or 0)
        out["holderCount"] = int(token.get("holderCount", 0) or 0)
        out["mcFormatted"] = format_market_cap(out["mc"])
        out["dexscreenerUrl"] = ds_url
        out["dsPairAddress"] = ds_pair
        return out

    async def _enrich_one(token: TokenData) -> TokenData | None:
        nonlocal birdeye_broken

        addr = token.get("address")
        sym  = token.get("symbol", "UNKNOWN")

        if not addr:
            logger.debug("Skipping %s: missing address", sym)
            return None
        if addr in blacklist:
            logger.debug("Skipping blacklisted token: %s (%s)", sym, addr)
            return None

        # Whitelisted → neutral deltas + configured fallback price
        if addr in WHITELISTED_TOKENS:
            fallback_price = float((config.get("bot", {}) or {}).get("fallback_token_price", 0.0001) or 0.0001)
            return await _normalize_and_fill(token, fallback_price, 0.0, 0.0, 0.0)

        # Pubkey sanity check
        try:
            Pubkey.from_string(addr)
        except Exception:
            logger.warning("Invalid public key for %s (%s)", sym, addr)
            if addr not in WHITELISTED_TOKENS:
                await add_to_blacklist(addr, "Invalid public key")
                blacklist.add(addr)
            return None

        # ---------------- Cache check (normalized) ----------------
        cached = _cache_get(addr)
        if cached is not None:
            # If cached is dict → positive cache. If None → negative cache.
            if isinstance(cached, dict):
                out = dict(token)
                out.update({
                    "price": float(cached.get("price", 0) or 0),
                    "priceChange1h": float(cached.get("priceChange1h", 0) or 0),
                    "priceChange6h": float(cached.get("priceChange6h", 0) or 0),
                    "priceChange24h": float(cached.get("priceChange24h", 0) or 0),
                    "mc": float(token.get("mc", 0) or 0),
                    "liquidity": float(token.get("liquidity", 0) or 0),
                    "holderCount": int(token.get("holderCount", 0) or 0),
                    "mcFormatted": format_market_cap(float(token.get("mc", 0) or 0)),
                    "dexscreenerUrl": cached.get("dexscreenerUrl", ""),
                    "dsPairAddress": cached.get("dsPairAddress", ""),
                })
                return out
            else:
                # Negative cached → fast neutral
                return await _normalize_and_fill(token, None, None, None, None, "", "")
        # ----------------------------------------------------------------

        # 1) Dexscreener token endpoint (FAST CAP)
        try:
            ds = await asyncio.wait_for(
                fetch_price_by_token_endpoint(session, addr),
                timeout=_ENRICH_TOKEN_TIMEOUT_S,
            )
            if ds:
                out = await _normalize_and_fill(
                    token,
                    ds.get("price"),
                    ds.get("priceChange1h"),
                    ds.get("priceChange6h"),
                    ds.get("priceChange24h"),
                    ds.get("dexscreenerUrl", ""),
                    ds.get("dsPairAddress", ""),
                )
                _cache_put(addr, {
                    "price": out["price"],
                    "priceChange1h": out["priceChange1h"],
                    "priceChange6h": out["priceChange6h"],
                    "priceChange24h": out["priceChange24h"],
                    "dexscreenerUrl": out["dexscreenerUrl"],
                    "dsPairAddress": out["dsPairAddress"],
                })
                return out
        except asyncio.TimeoutError:
            # Don’t sleep; fast-fail to next source
            logger.debug("Token endpoint timeout for %s (%.2fs)", addr, _ENRICH_TOKEN_TIMEOUT_S)
        except Exception as e:
            logger.debug("Dexscreener token endpoint failed for %s: %s", addr, e, exc_info=True)

        # 2) Birdeye (only if enabled and not broken) — obey RPS/budgets; NO long sleeps on 429
        if birdeye_enabled and not birdeye_broken:
            can_call = await _birdeye_gate()
            if not can_call:
                logger.info("Birdeye budget reached (cycle/run); skipping %s", addr)
            else:
                try:
                    url = f"https://public-api.birdeye.so/defi/price?address={addr}"
                    headers = {
                        "accept": "application/json",
                        "X-API-KEY": api_key,  # type: ignore[arg-type]
                        "User-Agent": "SOLOTradingBot/1.0",
                    }
                    async with session.get(
                        url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=_HTTP_TOTAL_TIMEOUT_S),
                    ) as resp:
                        if resp.status == 401:
                            birdeye_broken = True
                            logger.warning("Birdeye unauthorized (401). Disabling for the session.")
                        elif resp.status == 429:
                            # Fast tiny backoff and move on — do NOT stall the whole cycle
                            logger.warning("Birdeye rate limit for %s; skipping quickly.", addr)
                            await asyncio.sleep(_SMALL_BACKOFF_S)
                        elif resp.status == 200:
                            data = await resp.json(content_type=None)
                            pd = data.get("data", {}) or {}
                            if "value" in pd:
                                out = await _normalize_and_fill(
                                    token,
                                    float(pd.get("value") or 0),
                                    float(pd.get("priceChange1h", 0) or 0),
                                    float(pd.get("priceChange6h", 0) or 0),
                                    float(pd.get("priceChange24h", 0) or 0),
                                )
                                _cache_put(addr, {
                                    "price": out["price"],
                                    "priceChange1h": out["priceChange1h"],
                                    "priceChange6h": out["priceChange6h"],
                                    "priceChange24h": out["priceChange24h"],
                                    "dexscreenerUrl": out["dexscreenerUrl"],
                                    "dsPairAddress": out["dsPairAddress"],
                                })
                                return out
                except Exception as e:
                    logger.debug("Birdeye fetch error for %s: %s", addr, e, exc_info=True)

        # 3) Dexscreener search fallback — obey RPM/budgets; NO long sleeps on 429
        try:
            # Small jitter to avoid thundering herd
            await asyncio.sleep(0.25)

            can_call = await _ds_gate()
            if not can_call:
                logger.info("Dexscreener budget reached; skipping search for %s", addr)
            else:
                async with session.get(
                    f"https://api.dexscreener.com/latest/dex/search?q={addr}",
                    headers=_UA,
                    timeout=aiohttp.ClientTimeout(total=_HTTP_TOTAL_TIMEOUT_S),
                ) as resp:
                    if resp.status == 429:
                        logger.warning("Dexscreener search 429 for %s; skipping quickly.", addr)
                        # Tiny backoff, then neutral fallback if no success
                        await asyncio.sleep(_SMALL_BACKOFF_S)
                    elif resp.status == 200:
                        data = await resp.json(content_type=None)
                        pairs = data.get("pairs", []) or []
                        chosen = await _best_solana_pair_from_list(pairs, addr)
                        if chosen:
                            pc = chosen.get("priceChange") or {}
                            out = await _normalize_and_fill(
                                token,
                                float(chosen.get("priceUsd", 0) or 0),
                                float(pc.get("h1", 0) or 0),
                                float(pc.get("h6", 0) or 0),
                                float(pc.get("h24", 0) or 0),
                                _url_from_ds_pair(chosen),
                                chosen.get("pairAddress") or "",
                            )
                            _cache_put(addr, {
                                "price": out["price"],
                                "priceChange1h": out["priceChange1h"],
                                "priceChange6h": out["priceChange6h"],
                                "priceChange24h": out["priceChange24h"],
                                "dexscreenerUrl": out["dexscreenerUrl"],
                                "dsPairAddress": out["dsPairAddress"],
                            })
                            return out
        except Exception as e:
            logger.debug("Dexscreener search fallback failed for %s: %s", addr, e, exc_info=True)

        # 4) Neutral fallback — never stall here
        logger.warning("No enrichment sources succeeded for %s (%s); using neutral deltas.", sym, addr)
        _cache_put(addr, None)  # negative cache briefly to avoid re-hammering this mint
        return await _normalize_and_fill(token, None, None, None, None, "", "")

    # Batch politely to avoid rate limits
    for i in range(0, len(tokens), max(1, batch_size)):
        batch = tokens[i : i + batch_size]
        enriched_batch = await asyncio.gather(*(_enrich_one(t) for t in batch), return_exceptions=True)
        for item in enriched_batch:
            if isinstance(item, dict):
                results.append(item)
            elif isinstance(item, Exception):
                logger.debug("Enrich batch item failed: %s", item, exc_info=True)
        if i + batch_size < len(tokens):
            await asyncio.sleep(batch_sleep)

    return results




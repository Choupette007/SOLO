# solana_trading_bot_bundle/trading_bot/utils/__init__.py
# Compatibility-focused initializer for trading_bot.utils package.
# Provides conservative fallbacks for legacy helpers so imports like
# `from .utils import format_market_cap` succeed even if the full
# single-file utils.py isn't present.
from __future__ import annotations

import importlib.util
import os
import sys
import logging
import base64
import math
import statistics
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Iterable, Any, Optional, Tuple, List, Dict

# ---- Re-export env helpers from the utils package --------------------
from .env_loader import (  # type: ignore
    APP_DIR_NAME as APP_NAME,
    ensure_appdata_env_bootstrap,
    load_env_first_found,
    prefer_appdata_path,
    prefer_appdata_file,
)

# ---- Local cache placeholders (may be aliased to legacy objects) -----
price_cache: dict[str, float] = {}
token_balance_cache: dict[str, float] = {}
recent_price_cache: dict[str, dict] = {}

def clear_caches() -> None:
    price_cache.clear()
    token_balance_cache.clear()
    recent_price_cache.clear()

# ---- Try to load sibling legacy module trading_bot/utils.py ----------
_legacy_mod: ModuleType | None = None

def _load_legacy_utils() -> ModuleType | None:
    try:
        trading_bot_dir = Path(__file__).resolve().parents[1]  # .../trading_bot
        legacy_path = trading_bot_dir / "utils.py"
        if not legacy_path.exists():
            return None

        spec = importlib.util.spec_from_file_location(
            "solana_trading_bot_bundle.trading_bot._legacy_utils",
            str(legacy_path),
        )
        if not spec or not spec.loader:
            return None

        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)  # type: ignore[assignment]
        return mod
    except Exception:
        return None

def _export_from_legacy(mod: ModuleType, names: Iterable[str] | None = None) -> None:
    if mod is None:
        return
    public = [n for n in dir(mod) if not n.startswith("_")] if names is None else list(names)
    for n in public:
        if n in globals():
            continue
        try:
            globals()[n] = getattr(mod, n)
        except Exception:
            continue

_legacy_mod = _load_legacy_utils()

# If legacy has cache dicts, alias ours to *the same objects* for consistency.
if _legacy_mod is not None:
    for _name in ("price_cache", "token_balance_cache", "recent_price_cache", "token_account_existence_cache"):
        if hasattr(_legacy_mod, _name):
            globals()[_name] = getattr(_legacy_mod, _name)

# Export a curated set of names from legacy utils if present.
_legacy_names_to_import = [
    "_best_first",
    "safe_create_task",
    "remove_none_keys",
    "deduplicate_tokens",
    "custom_json_encoder",
    "get_rugcheck_token",
    "get_rugcheck_headers",
    "get_rugcheck_token_async",
    "add_to_blacklist",
    "token_account_existence_cache",
    "load_config",
    "setup_logging",
    # buy-sizing helpers
    "compute_buy_size_usd",
    "get_buy_amount",
    # jupiter helper
    "execute_jupiter_swap",
    # scoring / labels
    "normalize_labels",
    "is_flagged",
    "score_token",
    "score_token_conservative",
    "score_token_dispatch",
    "log_scoring_telemetry",
    # formatting helpers
    "format_market_cap",
    "format_usd",
    "format_ts_human",
    "fmt_usd",
    "fmt_ts",
]
if _legacy_mod is not None:
    _export_from_legacy(_legacy_mod, _legacy_names_to_import)

# ---- Compatibility shims (provide minimal fallbacks if legacy didn't) --------

# _best_first: legacy private helper; provide fallback if not present.
if "_best_first" not in globals():
    def _best_first(*vals: Any) -> Optional[Any]:
        for v in vals:
            if v not in (None, "", [], {}):
                return v
        return None
    globals()["_best_first"] = _best_first

# token_account_existence_cache
if "token_account_existence_cache" not in globals():
    token_account_existence_cache: dict[str, bool] = {}

# BLACKLIST / WHITELIST sets
if "BLACKLIST" not in globals():
    BLACKLIST: set[str] = set()

if "WHITELISTED_TOKENS" not in globals():
    WHITELISTED_TOKENS: set[str] = set()

# DB-backed helpers (best-effort; fall back to in-memory behaviour)
def _db_get_blacklist_status(addr: str) -> Optional[bool]:
    try:
        from ..database import get_blacklist_status  # type: ignore
    except Exception:
        return None
    try:
        return bool(get_blacklist_status(addr))
    except Exception:
        return None

def _db_add_to_blacklist(addr: str) -> bool:
    try:
        from ..database import add_to_blacklist as _db_add  # type: ignore
    except Exception:
        return False
    try:
        _db_add(addr)
        return True
    except Exception:
        return False

# is_blacklisted
if "is_blacklisted" not in globals():
    def is_blacklisted(address: str) -> bool:  # type: ignore[redef]
        addr = (address or "").strip()
        if not addr:
            return False
        db_status = _db_get_blacklist_status(addr)
        if db_status is not None:
            return db_status
        return addr in BLACKLIST
    globals()["is_blacklisted"] = is_blacklisted

# add_to_blacklist
if "add_to_blacklist" not in globals():
    def add_to_blacklist(address: str) -> None:  # type: ignore[redef]
        addr = (address or "").strip()
        if not addr:
            return
        if not _db_add_to_blacklist(addr):
            BLACKLIST.add(addr)
    globals()["add_to_blacklist"] = add_to_blacklist

# custom_json_encoder fallback (keeps minimal behaviour)
if "custom_json_encoder" not in globals():
    def custom_json_encoder(obj: Any) -> Any:  # type: ignore[redef]
        if isinstance(obj, Path):
            return str(obj)
        try:
            import datetime as _dt
            if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
                return obj.isoformat()
        except Exception:
            pass
        try:
            from decimal import Decimal as _Decimal  # type: ignore
            if isinstance(obj, _Decimal):
                return float(obj)
        except Exception:
            pass
        if isinstance(obj, (bytes, bytearray, memoryview)):
            return bytes(obj).hex()
        if isinstance(obj, set):
            try:
                return sorted(obj)  # type: ignore[arg-type]
            except Exception:
                return list(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    globals()["custom_json_encoder"] = custom_json_encoder

# get_rugcheck_token fallback: check multiple env vars
if "get_rugcheck_token" not in globals():
    def get_rugcheck_token() -> Optional[str]:  # type: ignore[redef]
        for key in (
            "RUGCHECK_JWT_TOKEN", "RUGCHECK_JWT", "RUGCHECK_API_TOKEN",
            "RUGCHECK_API_KEY", "RUGCHECK_TOKEN", "RUGCHECK"
        ):
            val = os.environ.get(key)
            if val:
                return val
        return None
    globals()["get_rugcheck_token"] = get_rugcheck_token

# Provide simple no-op placeholders for helpers that may be referenced
if "safe_create_task" not in globals():
    safe_create_task = None  # type: ignore
if "remove_none_keys" not in globals():
    remove_none_keys = None  # type: ignore
if "load_config" not in globals():
    load_config = None  # type: ignore
if "get_rugcheck_headers" not in globals():
    get_rugcheck_headers = None  # type: ignore

# setup_logging fallback (minimal, safe)
if "setup_logging" not in globals():
    def setup_logging(config: dict | None = None) -> logging.Logger:  # type: ignore[redef]
        level_name = None
        try:
            if isinstance(config, dict):
                level_name = (config.get("logging") or {}).get("log_level")
        except Exception:
            level_name = None
        level_name = (level_name or os.environ.get("LOG_LEVEL") or "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)

        root = logging.getLogger()
        root.setLevel(level)
        for h in list(root.handlers):
            try:
                h.setLevel(level)
            except Exception:
                pass

        logger = logging.getLogger("TradingBot")
        logger.setLevel(level)
        return logger
    globals()["setup_logging"] = setup_logging

# ---- Buy-sizing fallbacks --------------------------------------------
if "compute_buy_size_usd" not in globals():
    def compute_buy_size_usd(wallet_sol_balance: float, sol_usd_price: float, config: dict | None) -> float:  # type: ignore[redef]
        try:
            cfg = dict(config or {})
            botcfg = dict(cfg.get("bot") or {})
            tradingcfg = dict(cfg.get("trading") or {})
        except Exception:
            botcfg = {}
            tradingcfg = {}

        try:
            prefer_risk = bool(botcfg.get("prefer_risk_sizing", True))
        except Exception:
            prefer_risk = True
        try:
            max_risk_pct = float(botcfg.get("max_risk_pct_of_portfolio", 0.05))
        except Exception:
            max_risk_pct = 0.05
        try:
            buy_percentage = float(botcfg.get("buy_percentage", 0.10))
        except Exception:
            buy_percentage = 0.10
        try:
            min_usd = float(botcfg.get("min_usd_per_buy", tradingcfg.get("min_order_usd", 50.0)))
        except Exception:
            min_usd = 50.0
        try:
            max_usd = float(botcfg.get("max_order_usd", tradingcfg.get("max_order_usd", 500.0)))
        except Exception:
            max_usd = 500.0

        portfolio_usd = 0.0
        try:
            if wallet_sol_balance is not None and sol_usd_price and sol_usd_price > 0:
                portfolio_usd = float(wallet_sol_balance) * float(sol_usd_price)
        except Exception:
            portfolio_usd = 0.0

        risk_buy = portfolio_usd * max_risk_pct if portfolio_usd > 0 else 0.0
        legacy_buy = portfolio_usd * buy_percentage if portfolio_usd > 0 else 0.0

        if prefer_risk and risk_buy > 0:
            chosen = max(min_usd, min(risk_buy, max_usd))
        elif legacy_buy > 0:
            chosen = max(min_usd, min(legacy_buy, max_usd))
        else:
            chosen = min_usd

        global_min = float(tradingcfg.get("min_order_usd", min_usd)) if isinstance(tradingcfg, dict) else min_usd
        if chosen < global_min:
            chosen = global_min

        return float(chosen)
    globals()["compute_buy_size_usd"] = compute_buy_size_usd

# get_buy_amount fallback (async)
if "get_buy_amount" not in globals():
    async def get_buy_amount(token: Optional[dict], wallet_balance: float, sol_price: float, config: dict) -> Tuple[float, float]:  # type: ignore[redef]
        try:
            usd_target = compute_buy_size_usd(wallet_balance, sol_price, config)
        except Exception:
            usd_target = 0.0
        amount_sol = 0.0
        try:
            if sol_price and sol_price > 0:
                amount_sol = float(usd_target) / float(sol_price)
        except Exception:
            amount_sol = 0.0
        usd_amount = amount_sol * sol_price if sol_price and sol_price > 0 else 0.0
        return float(amount_sol), float(usd_amount)
    globals()["get_buy_amount"] = get_buy_amount

# ---- Jupiter swap fallback --------------------------------------------
if "execute_jupiter_swap" not in globals():
    async def execute_jupiter_swap(
        quote: dict,
        user_pubkey: str,
        wallet: object,
        solana_client: object,
    ) -> Optional[str]:  # type: ignore[redef]
        try:
            swap_b64 = quote.get("swapTransaction") if isinstance(quote, dict) else None
            if not swap_b64:
                logging.getLogger("TradingBot").debug("execute_jupiter_swap: no swapTransaction in quote")
                return None
            raw_tx = base64.b64decode(swap_b64)
            try:
                resp = await solana_client.send_raw_transaction(raw_tx)
                sig = getattr(resp, "value", None) or (resp.get("result") if isinstance(resp, dict) else None)
                if sig:
                    return str(sig)
            except Exception as e:
                logging.getLogger("TradingBot").debug("execute_jupiter_swap raw submit failed: %s", e)
            logging.getLogger("TradingBot").warning("execute_jupiter_swap fallback did not succeed")
            return None
        except Exception as e:
            logging.getLogger("TradingBot").exception("execute_jupiter_swap error: %s", e)
            return None
    globals()["execute_jupiter_swap"] = execute_jupiter_swap

# ---- Deduplicate tokens fallback --------------------------------------
if "deduplicate_tokens" not in globals():
    def deduplicate_tokens(tokens: List[dict]) -> List[dict]:  # type: ignore[redef]
        try:
            seen: set[str] = set()
            out: List[dict] = []
            for t in tokens or []:
                addr = t.get("address") if isinstance(t, dict) else None
                if addr and addr not in seen:
                    seen.add(addr)
                    out.append(t)
            return out
        except Exception:
            return tokens or []
    globals()["deduplicate_tokens"] = deduplicate_tokens

# ---- Scoring / labels fallbacks --------------------------------------
if "normalize_labels" not in globals():
    def normalize_labels(labels_any: Any) -> List[str]:
        out: List[str] = []
        if isinstance(labels_any, dict):
            labels_any = [labels_any]
        if not isinstance(labels_any, (list, tuple)):
            return out
        for item in labels_any:
            if isinstance(item, str):
                out.append((item or "").strip().lower())
            elif isinstance(item, dict):
                name = item.get("label") or item.get("name") or item.get("id")
                if name:
                    out.append(str(name).strip().lower())
        return list(dict.fromkeys([x for x in out if x]))
    globals()["normalize_labels"] = normalize_labels

if "is_flagged" not in globals():
    def is_flagged(labels_any: Any, bad_keys: Optional[set[str]] = None) -> bool:
        bad = bad_keys or {"honeypot", "scam", "blocked", "malicious", "dangerous"}
        labs = set(normalize_labels(labels_any))
        return len(bad.intersection(labs)) > 0
    globals()["is_flagged"] = is_flagged

if "score_token" not in globals():
    def score_token(token: Dict[str, Any]) -> float:
        def _num(x: Any, default: float = 0.0) -> float:
            try:
                return float(x)
            except Exception:
                return float(default)

        def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
            return max(lo, min(hi, v))

        if is_flagged(token.get("rugcheck_labels")):
            return 0.0
        status = str(token.get("rugcheck_status") or "").strip().lower()
        if status in {"fail", "blocked"}:
            return 0.0

        mc  = _num(token.get("mc") or token.get("market_cap") or token.get("fdv"), 0.0)
        liq = _num(token.get("liquidity"), 0.0)
        vol = _num(token.get("v24hUSD") or token.get("volume_24h"), 0.0)

        pc = token.get("priceChange") or {}
        pc1h  = _num(token.get("price_change_1h",  pc.get("h1",  0.0)), 0.0)
        pc6h  = _num(token.get("price_change_6h",  pc.get("h6",  0.0)), 0.0)
        pc24h = _num(token.get("price_change_24h", pc.get("h24", 0.0)), 0.0)

        created_raw = token.get("pairCreatedAt") or token.get("creation_timestamp") or 0
        try:
            created_ms = int(created_raw)
        except Exception:
            created_ms = 0
        if 0 < created_ms < 10_000_000_000:
            created_ms *= 1000
        age_h = (time.time() - (created_ms / 1000.0)) / 3600.0 if created_ms else None

        MID_MIN, MID_MAX = 100_000.0, 500_000.0
        if mc <= 0:
            mc_s = 0.0
        elif mc < MID_MIN:
            mc_s = _clip(mc / MID_MIN) * 0.8
        elif mc <= MID_MAX:
            mc_s = 1.0
        else:
            mc_s = _clip((MID_MAX / max(mc, 1.0)) ** 0.4)

        LIQ_SOFT = 25_000.0
        VOL_SOFT = 150_000.0
        liq_s = _clip(math.tanh(liq / max(LIQ_SOFT, 1.0)))
        vol_s = _clip(math.tanh(vol / max(VOL_SOFT, 1.0)))

        def _mom(x: float, cap: float) -> float:
            if x <= 0:
                return 0.0
            x = min(x, cap)
            return _clip(math.tanh(x / cap))
        m1  = _mom(pc1h, 40.0)
        m6  = _mom(pc6h, 120.0)
        m24 = _mom(pc24h, 250.0)
        momentum = (0.50 * m1) + (0.30 * m6) + (0.20 * m24)

        if age_h is None:
            age_s = 0.7
        else:
            if age_h < (15.0 / 60.0):
                age_s = 0.2
            elif age_h < 2.0:
                age_s = 0.7
            elif age_h < 72.0:
                age_s = 1.0
            elif age_h < 720.0:
                age_s = 0.8
            else:
                age_s = 0.6

        birdeye_present = bool(token.get("created_at")) or (str(token.get("source") or "").lower() == "birdeye")
        presence_mult = 1.10 if birdeye_present else 0.85

        cats = (token.get("categories") or []) or []
        cat_mult = 1.0
        if "mid_cap" in cats:
            cat_mult *= 1.10
        if "large_cap" in cats:
            cat_mult *= 0.92
        if "low_cap" in cats:
            cat_mult *= 0.97

        base = (0.35 * mc_s) + (0.25 * liq_s) + (0.20 * vol_s) + (0.20 * momentum)
        final = base * age_s * presence_mult * cat_mult
        score = max(0.0, min(100.0, final * 100.0))
        return float(score)
    globals()["score_token"] = score_token

if "score_token_conservative" not in globals():
    def score_token_conservative(token: Dict[str, Any]) -> float:
        if is_flagged(token.get("rugcheck_labels")):
            return 0.0
        status = str(token.get("rugcheck_status") or "").strip().lower()
        if status in {"fail", "blocked"}:
            return 0.0

        def _num(x: Any, d: float = 0.0) -> float:
            try:
                return float(x)
            except Exception:
                return d

        def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
            return max(lo, min(hi, v))

        mc  = _num(token.get("mc") or token.get("market_cap") or token.get("fdv"))
        liq = _num(token.get("liquidity"))
        vol = _num(token.get("v24hUSD") or token.get("volume_24h"))

        pc = token.get("priceChange") or {}
        pc1h  = _num(token.get("price_change_1h",  pc.get("h1",  0.0)))
        pc6h  = _num(token.get("price_change_6h",  pc.get("h6",  0.0)))
        pc24h = _num(token.get("price_change_24h", pc.get("h24", 0.0)))

        LO, HI = 50_000.0, 2_500_000.0
        if mc <= 0:
            mc_s = 0.0
        elif mc < LO:
            mc_s = _clip(mc / LO) * 0.7
        elif mc <= HI:
            mc_s = 1.0
        else:
            mc_s = _clip((HI / max(mc, 1.0)) ** 0.25)

        liq_s = _clip(math.tanh(liq / 20_000.0))
        vol_s = _clip(math.tanh(vol / 100_000.0))

        def _mom(x: float, cap: float) -> float:
            if x <= 0:
                return 0.0
            x = min(x, cap)
            return _clip(math.tanh(x / cap))
        m1  = _mom(pc1h, 25.0)
        m6  = _mom(pc6h, 80.0)
        m24 = _mom(pc24h, 200.0)
        momentum = (0.35 * m1) + (0.35 * m6) + (0.30 * m24)

        created_raw = token.get("pairCreatedAt") or token.get("creation_timestamp") or 0
        try:
            created_ms = int(created_raw)
        except Exception:
            created_ms = 0
        if 0 < created_ms < 10_000_000_000:
            created_ms *= 1000
        age_h = (time.time() - (created_ms / 1000.0)) / 3600.0 if created_ms else None
        if age_h is None:
            age_s = 0.7
        elif age_h < 2.0:
            age_s = 0.6
        elif age_h < 168.0:
            age_s = 1.0
        elif age_h < 720.0:
            age_s = 0.8
        else:
            age_s = 0.6

        presence_mult = 1.05 if (bool(token.get("created_at")) or (str(token.get("source") or "").lower() == "birdeye")) else 0.9

        base = (0.30 * mc_s) + (0.35 * liq_s) + (0.25 * vol_s) + (0.10 * momentum)
        final = base * age_s * presence_mult
        return float(max(0.0, min(100.0, final * 100.0)))
    globals()["score_token_conservative"] = score_token_conservative

if "score_token_dispatch" not in globals():
    def score_token_dispatch(token: Dict[str, Any], cfg: Dict[str, Any]) -> float:
        mode = str(((cfg or {}).get("scoring", {}) or {}).get("mode") or "midcap").lower()
        if mode == "conservative":
            score = score_token_conservative(token)
        else:
            score = score_token(token)

        bb_cfg: Dict[str, Any] = (((cfg or {}).get("signals") or {}).get("bbands_flip") or {})
        if bool(bb_cfg.get("enable", True)):
            bb_long = bool(token.get("bb_long"))
            bb_short = bool(token.get("bb_short"))
            if bb_short and bool(bb_cfg.get("reject_shorts", True)):
                token["_reject_reason"] = "bb_short_flip"
                return -1e9  # hard-reject
            if bb_long:
                basis = token.get("bb_basis")
                upper = token.get("bb_upper")
                lower = token.get("bb_lower")
                try:
                    width_min = float(bb_cfg.get("min_width", 0.0))
                except Exception:
                    width_min = 0.0
                width_pct = 0.0
                try:
                    if basis is not None:
                        b = float(basis)
                        if b != 0 and upper is not None and lower is not None:
                            width_pct = (float(upper) - float(lower)) / b
                except Exception:
                    width_pct = 0.0

                long_bonus_val = bb_cfg.get("long_bonus_points", 15)
                try:
                    long_bonus = int(long_bonus_val)
                except Exception:
                    long_bonus = 15

                narrow_bonus_val = bb_cfg.get("long_bonus_points_narrow", 5)
                try:
                    narrow_bonus = int(narrow_bonus_val)
                except Exception:
                    narrow_bonus = 5

                bump = long_bonus if width_pct >= width_min else narrow_bonus
                score += bump

        cats = (token.get("categories") or []) or []
        minimum_floor = 5.0
        if ("whitelisted" in cats) and score < minimum_floor:
            score = minimum_floor
        return float(score)
    globals()["score_token_dispatch"] = score_token_dispatch

if "log_scoring_telemetry" not in globals():
    def log_scoring_telemetry(tokens: List[Dict[str, Any]], where: str = "eligible") -> None:
        if not tokens:
            logging.getLogger("TradingBot").info(f"[SCORE] {where}: empty set")
            return
        total = 0.0
        have_be = 0
        n = 0
        for t in tokens:
            try:
                total += float(t.get("score", 0.0))
                have_be += 1 if (bool(t.get("created_at")) or (str(t.get("source") or "").lower() == "birdeye")) else 0
                n += 1
            except Exception:
                continue
        avg = total / max(1, n)
        pct = (have_be / max(1, n)) * 100.0
        logging.getLogger("TradingBot").info(f"[SCORE] {where}: avg={avg:.1f} birdeye_coverage={have_be}/{n} ({pct:.1f}%)")
    globals()["log_scoring_telemetry"] = log_scoring_telemetry

# ---- Formatting helpers (fallbacks) --------------------------------------
if "format_market_cap" not in globals():
    def format_market_cap(value: float) -> str:
        try:
            num = float(value)
        except Exception:
            return "N/A"
        abs_num = abs(num)
        try:
            if abs_num >= 1_000_000_000:
                return f"{num / 1_000_000_000:.2f}B"
            elif abs_num >= 1_000_000:
                return f"{num / 1_000_000:.2f}M"
            elif abs_num >= 1_000:
                return f"{num / 1_000:.2f}K"
            else:
                return f"{num:.2f}"
        except Exception:
            return "N/A"
    globals()["format_market_cap"] = format_market_cap

if "format_usd" not in globals():
    def format_usd(value: float | int | None, *, compact: bool = False, decimals: int = 2) -> str:
        try:
            num = float(value)
        except Exception:
            return "$0.00"
        if not math.isfinite(num):
            return "$0.00"
        sign = "-" if num < 0 else ""
        n = abs(num)
        try:
            if compact:
                if n >= 1_000_000_000:
                    return f"{sign}${n/1_000_000_000:.{decimals}f}B"
                if n >= 1_000_000:
                    return f"{sign}${n/1_000_000:.{decimals}f}M"
                if n >= 1_000:
                    return f"{sign}${n/1_000:.{decimals}f}K"
                return f"{sign}${n:.{decimals}f}"
            return f"{sign}${n:,.{decimals}f}"
        except Exception:
            return "$0.00"
    globals()["format_usd"] = format_usd

if "format_ts_human" not in globals():
    def format_ts_human(ts: int | float | str | None, *, with_time: bool = True) -> str:
        if ts is None:
            return "-"
        try:
            t = float(ts)
        except Exception:
            return "-"
        try:
            if 0 < t < 10_000_000_000:
                dt = datetime.fromtimestamp(t)
            else:
                dt = datetime.fromtimestamp(t / 1000.0)
            return dt.strftime("%Y-%m-%d %H:%M:%S") if with_time else dt.strftime("%Y-%m-%d")
        except Exception:
            return "-"
    globals()["format_ts_human"] = format_ts_human

if "fmt_usd" not in globals():
    def fmt_usd(value: Any) -> str:
        try:
            return format_usd(value, compact=True, decimals=2)
        except Exception:
            return "$0.00"
    globals()["fmt_usd"] = fmt_usd

if "fmt_ts" not in globals():
    def fmt_ts(ts: Any) -> str:
        try:
            return format_ts_human(ts, with_time=True)
        except Exception:
            return "-"
    globals()["fmt_ts"] = fmt_ts

# ---- Final public surface --------------------------------------------
_legacy_public = set(n for n in dir(_legacy_mod) if _legacy_mod and not n.startswith("_"))
_ours_public = {
    "APP_NAME",
    "ensure_appdata_env_bootstrap",
    "load_env_first_found",
    "prefer_appdata_path",
    "prefer_appdata_file",
    "price_cache",
    "token_balance_cache",
    "recent_price_cache",
    "clear_caches",
    "token_account_existence_cache",
    "BLACKLIST",
    "WHITELISTED_TOKENS",
    "is_blacklisted",
    "add_to_blacklist",
    "custom_json_encoder",
    "get_rugcheck_token",
    "_best_first",
    "setup_logging",
    "compute_buy_size_usd",
    "get_buy_amount",
    "execute_jupiter_swap",
    "deduplicate_tokens",
    "normalize_labels",
    "is_flagged",
    "score_token",
    "score_token_conservative",
    "score_token_dispatch",
    "log_scoring_telemetry",
    "format_market_cap",
    "format_usd",
    "format_ts_human",
    "fmt_usd",
    "fmt_ts",
}

__all__ = sorted(_legacy_public | _ours_public)
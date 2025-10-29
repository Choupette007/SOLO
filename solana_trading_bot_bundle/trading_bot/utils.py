# solana_trading_bot_bundle/trading_bot/utils.py
from __future__ import annotations

import base64
import json
import logging
from logging import Logger
from logging.handlers import RotatingFileHandler
import os
import re
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import random

import yaml
from cachetools import TTLCache
from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey
from solders.signature import Signature

# Prefer modern solders types (solana-py >= 0.36)
# Backwards-compatible fallbacks: try solders first, then fall back to solana-py types when available.
try:
    from solders.transaction import Transaction, VersionedTransaction  # type: ignore
except Exception:
    try:
        # Older solana-py provides a Transaction class (not VersionedTransaction)
        from solana.transaction import Transaction  # type: ignore
    except Exception as e:
        logging.getLogger("TradingBot").error(
            "Failed to import Transaction type from 'solders' or 'solana'. Ensure compatible versions are installed. "
            "Recommended: solana>=0.36,<0.37 and solders>=0.26,<0.27. Underlying error: %s",
            e,
        )
        raise
    # VersionedTransaction is not available in older solana-py; set to None to guard usage sites
    VersionedTransaction = None  # type: ignore

# Optional signing support with fallback to solana.keypair.Keypair (older solana-py)
try:
    from solders.keypair import Keypair  # type: ignore
except Exception:
    try:
        from solana.keypair import Keypair  # type: ignore
    except Exception:
        Keypair = None  # type: ignore

from solana_trading_bot_bundle.common.constants import (
    APP_NAME,
    local_appdata_dir,
    appdata_dir,
    logs_dir,
    data_dir,
    config_path,
    env_path,
    db_path,
    token_cache_path,
    ensure_app_dirs,
    prefer_appdata_file,
)

# Back-compat logger (keep 'TradingBot'), plus alias a clean name for dashboards, etc.
logger = logging.getLogger("TradingBot")
logging.getLogger("SoloTradingBot").propagate = True  # alias; handlers attached by setup_logging

# -----------------------------------------------------------------------------
# Whitelist & caches
# -----------------------------------------------------------------------------
WHITELISTED_TOKENS: Dict[str, str] = {
    "So11111111111111111111111111111111111111112": "SOL",
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": "USDC",
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": "USDT",
}

price_cache = TTLCache(maxsize=1000, ttl=300)
token_balance_cache = TTLCache(maxsize=1000, ttl=60)
token_account_existence_cache = TTLCache(maxsize=1000, ttl=300)

# -----------------------------------------------------------------------------
# Path helpers (robust across Windows/Mac)
# -----------------------------------------------------------------------------
def _to_path(p: Any) -> Path:
    """Coerce constants (which may be str/Path/callable) into a Path safely."""
    try:
        v = p() if callable(p) else p
    except Exception:
        v = p
    try:
        v2 = prefer_appdata_file(v)  # pass-through if already absolute
    except Exception:
        v2 = v
    return Path(v2)

def _appdata() -> Path:
    try:
        return _to_path(appdata_dir)
    except Exception:
        # last-ditch: LOCALAPPDATA on Windows, ~/.local/share on POSIX
        return Path(os.getenv("LOCALAPPDATA") or (Path.home() / ".local" / "share")) / APP_NAME

def _logs_dir() -> Path:
    p = _to_path(logs_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _config_path() -> Path:
    return _to_path(config_path)

def _env_file_path() -> Path:
    return _to_path(env_path)

# Ensure base folders exist as early as possible
try:
    ensure_app_dirs()
except Exception:
    pass
try:
    _logs_dir()
except Exception:
    pass

# -----------------------------------------------------------------------------
# Config & logging
# -----------------------------------------------------------------------------
_missing_cfg_last_log_ts: float = 0.0

def _resolve_config_path(path: Optional[str] = None) -> Path:
    """
    Pick a sane config path in this order:
      1) explicit 'path' if given
      2) constants' config_path (already normalized)
      3) <appdata>/config.yaml
      4) <cwd>/config.yaml  (fallback)
    Auto-creates parent dirs.
    """
    candidates: List[Path] = []
    if path:
        candidates.append(Path(path))
    candidates.append(_config_path())
    candidates.append(_appdata() / "config.yaml")
    candidates.append(Path.cwd() / "config.yaml")

    for c in candidates:
        try:
            c.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            if c.exists() or os.access(str(c.parent), os.W_OK):
                return c
        except Exception:
            continue
    p = _appdata() / "config.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _create_default_config(cfg_path: Path) -> None:
    """
    Write a minimal default config **once** so first-run testers don't see scary errors.
    Safe to overwrite empty files; won't clobber real configs.
    """
    try:
        if not cfg_path.exists() or (cfg_path.stat().st_size == 0):
            default_cfg = {
                "logging": {
                    "file": str(_logs_dir() / "bot.log"),
                    "log_level": "INFO",
                    "log_rotation_size_mb": 10,
                    "log_max_files": 5,
                },
                "discovery": {
                    # conservative defaults; GUI can change these
                    "dexscreener_pages": 5,
                    "dexscreener_per_page": 75,
                    "dexscreener_post_cap": 0,
                    "raydium_max_pairs": 400,
                    "raydium_page_size": 100,
                    "raydium_max_pools": 800,
                    "birdeye_max_tokens": 200,
                    "rugcheck_in_discovery": False,
                    "require_rugcheck_pass": False,
                },
                "database": {
                    "token_cache_path": str(_to_path(token_cache_path)),
                },
            }
            cfg_path.write_text("# Auto-generated default config\n" + yaml.safe_dump(default_cfg), encoding="utf-8")
    except Exception as e:
        logging.getLogger("TradingBot").debug("Could not create default config at %s: %s", cfg_path, e)

def load_config(path: str | None = None) -> Dict:
    global _missing_cfg_last_log_ts
    cfg_path = _resolve_config_path(path)
    _create_default_config(cfg_path)
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        logger.debug("Loaded configuration from %s", cfg_path)
        return config
    except Exception as e:
        now = time.time()
        if now - _missing_cfg_last_log_ts > 30:
            logger.error("Failed to load config from %s: %s", cfg_path, e)
            _missing_cfg_last_log_ts = now
        return {}

# sentinel to avoid reconfig churn on reruns
_LOG_SENTINEL_ATTR = "_solo_logging_file"

def setup_logging(config: dict[str, Any] | None) -> logging.Logger:
    log_cfg = (config or {}).get("logging", {}) if isinstance(config, dict) else {}
    from solana_trading_bot_bundle.common.constants import logs_dir as _logs_dir_fn

    # Resolve path & options
    raw_file = log_cfg.get("file") or (_logs_dir_fn() / "bot.log")
    log_file = Path(raw_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    level_name = str(log_cfg.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    max_size_mb = int(log_cfg.get("log_rotation_size_mb", 10))
    max_files = int(log_cfg.get("log_max_files", 5))

    root = logging.getLogger()
    root.setLevel(level)

    # Fast-exit if we’ve already configured to this file
    if getattr(root, _LOG_SENTINEL_ATTR, None) == str(log_file):
        for h in root.handlers:
            h.setLevel(level)
        return logging.getLogger("TradingBot")

    # ----- File handler (dedupe by exact path on ROOT) -----
    file_handler: Optional[RotatingFileHandler] = None
    for h in list(root.handlers):
        if isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", None) == str(log_file):
            file_handler = h
            break
    if file_handler is None:
        file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=max_files,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        file_handler.setLevel(level)
        root.addHandler(file_handler)
    else:
        file_handler.setLevel(level)

    # ----- Console handler (single) -----
    has_console = any(isinstance(h, logging.StreamHandler) and not hasattr(h, "baseFilename") for h in root.handlers)
    if not has_console:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        sh.setLevel(level)
        root.addHandler(sh)

    # ----- Project loggers: no extra handlers, just propagate -----
    for name in ("TradingBot", "TradingBotGUI", "SoloTradingBot"):
        lx = logging.getLogger(name)
        lx.handlers.clear()
        lx.propagate = True
        lx.setLevel(level)

    setattr(root, _LOG_SENTINEL_ATTR, str(log_file))
    logging.getLogger("TradingBot").info("Logging configured: level=%s, file=%s", level_name, str(log_file))
    return logging.getLogger("TradingBot")

# -----------------------------------------------------------------------------
# .env helpers (local, safe)
# -----------------------------------------------------------------------------
ENV_PATH = _env_file_path()

def _parse_dotenv_file(path: Path) -> dict:
    data: Dict[str, str] = {}
    try:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                data[k.strip()] = v.strip().strip('"')
    except Exception as e:
        logger.debug("Failed to parse .env: %s", e)
    return data

def _write_env_key(key: str, value: str) -> None:
    try:
        ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
        current = _parse_dotenv_file(ENV_PATH)
        current[key] = value
        lines = ["# SOLOTradingBot .env"]
        for k, v in current.items():
            v_clean = str(v).replace("\n", "").replace("\r", "").replace('"', '\\"')
            lines.append(f'{k}="{v_clean}"')
        ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if os.name != "nt":
            try:
                os.chmod(ENV_PATH, 0o600)
            except Exception:
                pass
    except Exception as e:
        logger.warning("Failed to write %s to .env: %s", key, e)

# -----------------------------------------------------------------------------
# Rugcheck token with expiry + auto-login
# -----------------------------------------------------------------------------
def _base64url_decode_noverify(seg: str) -> bytes:
    seg_b = seg.encode() if isinstance(seg, str) else seg
    pad = b"=" * ((4 - (len(seg_b) % 4)) % 4)
    import base64 as _b64
    return _b64.urlsafe_b64decode(seg_b + pad)

def _jwt_exp_unix(jwt_token: str) -> Optional[int]:
    try:
        parts = jwt_token.split(".")
        if len(parts) != 3:
            return None
        payload = json.loads(_base64url_decode_noverify(parts[1]).decode("utf-8"))
        exp = payload.get("exp")
        return int(exp) if exp is not None else None
    except Exception:
        return None

def _is_token_still_valid(jwt_token: str, leeway_seconds: int = 300) -> bool:
    exp = _jwt_exp_unix(jwt_token)
    if not exp:
        return False
    now = int(time.time())
    return (exp - now) > leeway_seconds

def _load_wallet_keypair_from_env() -> Any:
    """
    Load a wallet Keypair from environment variables. We return `Any` here to
    keep Pylance happy even when `Keypair` is a runtime variable that may be
    None if `solders` isn't installed.
    """
    if Keypair is None:
        raise RuntimeError("Keypair type not available; cannot sign Rugcheck login.")
    pk = (os.getenv("SOLANA_PRIVATE_KEY") or os.getenv("WALLET_PRIVATE_KEY") or "").strip()
    if not pk:
        raise ValueError("Set SOLANA_PRIVATE_KEY or WALLET_PRIVATE_KEY in your .env to auto-login Rugcheck.")
    # Try base58 64-byte secret key
    try:
        return Keypair.from_base58_string(pk)  # type: ignore[attr-defined]
    except Exception:
        pass
    # Try JSON array of 64 bytes
    try:
        arr = json.loads(pk)
        if isinstance(arr, list) and len(arr) == 64 and all(isinstance(x, int) for x in arr):
            return Keypair.from_bytes(bytes(arr))
    except Exception:
        pass
    raise ValueError("Private key format not recognized (expect base58 64-byte secret or JSON array of 64 ints).")

def get_rugcheck_token(refresh: bool = False) -> Optional[str]:
    """
    Return a Rugcheck JWT:
      • Use env if present and not expired,
      • else sign a login message with wallet and request a fresh token,
      • write fresh token back to .env and os.environ.

    NOTE: This function uses synchronous requests and will block the event loop if called
    from an async context. Consider using get_rugcheck_token_async when calling from
    async code to avoid blocking the event loop.
    """
    tok = (os.getenv("RUGCHECK_JWT_TOKEN") or "").strip()
    if tok and not refresh and _is_token_still_valid(tok):
        return tok

    # Try to auto-login using wallet
    try:
        import base58  # type: ignore
        import requests  # type: ignore

        wallet = _load_wallet_keypair_from_env()
        message_data = {
            "message": "Sign-in to Rugcheck.xyz",
            "timestamp": int(time.time() * 1000),
            "publicKey": str(wallet.pubkey()),
        }
        message_json = json.dumps(message_data, separators=(",", ":")).encode("utf-8")
        signature = wallet.sign_message(message_json)
        sig_b58 = str(signature)
        signature_data = list(base58.b58decode(sig_b58))
        payload = {
            "signature": {"data": signature_data, "type": "ed25519"},
            "wallet": str(wallet.pubkey()),
            "message": message_data,
        }
        resp = requests.post(
            "https://api.rugcheck.xyz/auth/login/solana",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            new_tok = data.get("token") or data.get("jwt") or data.get("access_token")
            if new_tok:
                os.environ["RUGCHECK_JWT_TOKEN"] = new_tok
                _write_env_key("RUGCHECK_JWT_TOKEN", new_tok)
                return new_tok
            logger.warning("Rugcheck login succeeded but token field not found in response.")
        else:
            logger.warning("Rugcheck login failed: %s %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.warning("Rugcheck auto-login error: %s", e)

    # Fallback: return existing token even if we couldn't refresh
    return tok or None

async def get_rugcheck_token_async(refresh: bool = False) -> Optional[str]:
    """Async wrapper that runs the blocking get_rugcheck_token in a thread to avoid
    blocking the event loop. Keeps existing signing/requests logic.
    """
    return await asyncio.to_thread(get_rugcheck_token, refresh)

def get_rugcheck_headers(force_refresh: bool = False) -> Dict[str, str]:
    tok = get_rugcheck_token(refresh=force_refresh)
    return {"Authorization": f"Bearer {tok}"} if tok else {}

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
async def add_to_blacklist(token_address: str, reason: str) -> None:
    cfg = load_config()
    db_file = _to_path((cfg.get("database", {}) or {}).get("token_cache_path", token_cache_path))
    try:
        import aiosqlite
        async with aiosqlite.connect(db_file) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS blacklist (
                    address   TEXT PRIMARY KEY,
                    reason    TEXT,
                    timestamp INTEGER
                )
                """
            )
            await db.execute(
                "INSERT OR REPLACE INTO blacklist (address, reason, timestamp) VALUES (?, ?, ?)",
                (token_address, reason, int(datetime.now().timestamp())),
            )
            await db.commit()
        logger.info("Added %s to blacklist: %s", token_address, reason)
    except Exception as e:
        logger.error("Failed to add %s to blacklist: %s", token_address, e)

def deduplicate_tokens(tokens: List[Dict]) -> List[Dict]:
    seen: set[str] = set()
    out: List[Dict] = []
    for t in tokens or []:
        addr = t.get("address")
        if addr and addr not in seen:
            seen.add(addr)
            out.append(t)
    return out

def remove_none_keys(d: Dict) -> Dict:
    return {k: v for k, v in (d or {}).items() if v is not None and v != ""}

def custom_json_encoder(obj: object) -> str:
    if isinstance(obj, (Pubkey, Signature)):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# -----------------------------------------------------------------------------
# Backoff / Circuit-breaker helpers (shared utilities)
# -----------------------------------------------------------------------------
async def backoff_sleep(attempt: int, base: float = 1.5, max_sleep: float = 120.0, jitter: bool = True) -> float:
    """
    Jittered exponential backoff helper for async code.
    Returns the sleep seconds after awaiting.
    """
    sleep = min(max_sleep, base * (2 ** max(0, attempt - 1)))
    if jitter:
        factor = random.uniform(0.8, 1.25)
        sleep = max(0.0, sleep * factor)
    sleep = max(0.1, float(sleep))
    await asyncio.sleep(sleep)
    return sleep

class CircuitBreaker429:
    """
    Simple in-memory circuit breaker for consecutive HTTP 429 events.
    Use record(429=True/False) and inspect is_open().
    """
    def __init__(self, threshold: int = 5, cooldown_seconds: int = 60):
        self.threshold = int(threshold)
        self.cooldown_seconds = int(cooldown_seconds)
        self._count = 0
        self._last_trip_time: Optional[float] = None

    def record(self, is_429: bool) -> None:
        if is_429:
            self._count += 1
            if self._count >= self.threshold:
                self._last_trip_time = time.time()
        else:
            self._count = 0
            self._last_trip_time = None

    def is_open(self) -> bool:
        if self._last_trip_time is None:
            return False
        if (time.time() - self._last_trip_time) > self.cooldown_seconds:
            self._count = 0
            self._last_trip_time = None
            return False
        return True

    def remaining_cooldown(self) -> float:
        if self._last_trip_time is None:
            return 0.0
        rc = max(0.0, self.cooldown_seconds - (time.time() - self._last_trip_time))
        return rc

# -----------------------------------------------------------------------------
# Buy-sizing / risk-aware sizing
# -----------------------------------------------------------------------------
def compute_buy_size_usd(wallet_sol_balance: float, sol_usd_price: float, config: Dict[str, Any]) -> float:
    """
    Compute USD amount for a buy following risk-based sizing with safe fallbacks.

    Rules:
      - Prefer bot.max_risk_pct_of_portfolio if prefer_risk_sizing is True and portfolio value available.
      - Fall back to legacy buy_percentage if necessary.
      - Enforce min_usd_per_buy and max_order_usd clamps.
      - Respect global trading.min_order_usd when present.
    """
    botcfg = (config or {}).get("bot", {}) or {}
    tradingcfg = (config or {}).get("trading", {}) or {}

    prefer_risk = bool(botcfg.get("prefer_risk_sizing", True))
    max_risk_pct = float(botcfg.get("max_risk_pct_of_portfolio", 0.05))
    buy_percentage = float(botcfg.get("buy_percentage", 0.10))
    min_usd = float(botcfg.get("min_usd_per_buy", tradingcfg.get("min_order_usd", 50.0)))
    max_usd = float(botcfg.get("max_order_usd", tradingcfg.get("max_order_usd", 500.0)))

    # portfolio USD estimate
    portfolio_usd = 0.0
    try:
        if wallet_sol_balance is not None and sol_usd_price and sol_usd_price > 0:
            portfolio_usd = float(wallet_sol_balance) * float(sol_usd_price)
    except Exception:
        portfolio_usd = 0.0

    # compute candidate sizes
    risk_buy = portfolio_usd * float(max_risk_pct) if portfolio_usd > 0 else 0.0
    legacy_buy = portfolio_usd * float(buy_percentage) if portfolio_usd > 0 else 0.0

    if prefer_risk and risk_buy > 0:
        chosen = max(min_usd, min(risk_buy, max_usd))
        reason = "risk_pct"
    elif legacy_buy > 0:
        chosen = max(min_usd, min(legacy_buy, max_usd))
        reason = "legacy_pct"
    else:
        chosen = min_usd
        reason = "min_floor"

    # respect global trading minimum if configured
    global_min = float(tradingcfg.get("min_order_usd", min_usd))
    if chosen < global_min:
        chosen = global_min

    logger.debug(
        "compute_buy_size_usd: wallet_sol=%.6f sol_price=%.4f portfolio_usd=%.2f prefer_risk=%s max_risk_pct=%.4f "
        "risk_buy=%.2f legacy_buy=%.2f chosen_usd=%.2f reason=%s",
        wallet_sol_balance or 0.0, sol_usd_price or 0.0, portfolio_usd, prefer_risk, max_risk_pct,
        risk_buy, legacy_buy, chosen, reason,
    )

    return float(chosen)

async def get_buy_amount(
    token: Optional[Dict[str, Any]],
    wallet_balance: float,
    sol_price: float,
    config: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Determine buy amount (SOL) and USD amount for a given token.

    Returns:
      (amount_sol, usd_amount)
    """
    token = token or {}

    botcfg = (config or {}).get("bot", {}) or {}
    tradingcfg = (config or {}).get("trading", {}) or {}

    dry_run = bool(botcfg.get("dry_run", tradingcfg.get("dry_run", True)))
    per_trade_sol = float(botcfg.get("per_trade_sol", 0.0))  # if >0, cap in SOL

    usd_target = compute_buy_size_usd(wallet_balance, sol_price, config)
    usd_min_config = float(botcfg.get("min_usd_per_buy", tradingcfg.get("min_order_usd", 0.0)))
    usd_target = max(usd_target, usd_min_config)

    amount_sol = 0.0
    try:
        if sol_price and sol_price > 0:
            amount_sol = float(usd_target) / float(sol_price)
    except Exception:
        amount_sol = 0.0

    if per_trade_sol and per_trade_sol > 0:
        amount_sol = min(amount_sol, float(per_trade_sol))

    if not dry_run and wallet_balance is not None:
        amount_sol = min(amount_sol, float(wallet_balance))

    if dry_run and amount_sol <= 0 and usd_target > 0 and sol_price and sol_price > 0:
        amount_sol = float(usd_target) / float(sol_price)

    usd_amount = amount_sol * sol_price if sol_price and sol_price > 0 else 0.0

    try:
        tok_name = token.get("symbol") or token.get("address") or "unknown"
        logger.info(
            "Buy sizing: token=%s wallet_sol=%.6f sol_price=%.4f portfolio_usd=%.2f "
            "usd_target=%.2f amount_sol=%.6f usd_amount=%.2f dry_run=%s",
            tok_name,
            wallet_balance or 0.0,
            sol_price or 0.0,
            (wallet_balance or 0.0) * (sol_price or 0.0),
            usd_target,
            amount_sol,
            usd_amount,
            dry_run,
        )
    except Exception:
        pass

    return float(amount_sol), float(usd_amount)

# -----------------------------------------------------------------------------
# Labels + Scoring
# -----------------------------------------------------------------------------
def normalize_labels(labels_any: Any) -> List[str]:
    """
    Accepts Rugcheck-like label arrays (strings/dicts/mixed) and returns lowercase names.
    e.g. [{"label":"scam"}, "dangerous", {"name": "Honeypot"}] -> ["scam","dangerous","honeypot"]
    Preserves first-seen order.
    """
    out: list[str] = []
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
    # preserve order and deduplicate
    return list(dict.fromkeys([x for x in out if x]))

def is_flagged(labels_any: Any, bad_keys: Optional[set[str]] = None) -> bool:
    bad = bad_keys or {"honeypot", "scam", "blocked", "malicious", "dangerous"}
    labs = set(normalize_labels(labels_any))
    return len(bad.intersection(labs)) > 0

def score_token(token: Dict[str, Any]) -> float:
    """
    Aggressive, mid-cap biased scoring (Birdeye-aware) with 0..100 output.
    """
    # --------- Hard safety gates ----------
    if is_flagged(token.get("rugcheck_labels")):
        return 0.0
    status = str(token.get("rugcheck_status") or "").strip().lower()
    if status in {"fail", "blocked"}:
        return 0.0

    # --------- Safe extractors ----------
    def _num(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, v))

    # Core features (use your existing token schema keys)
    mc  = _num(token.get("mc") or token.get("market_cap") or token.get("fdv"), 0.0)
    liq = _num(token.get("liquidity"), 0.0)
    vol = _num(token.get("v24hUSD") or token.get("volume_24h"), 0.0)

    # Price-change (percentage) fields with fallbacks
    pc = token.get("priceChange") or {}
    pc1h  = _num(token.get("price_change_1h",  pc.get("h1",  0.0)), 0.0)
    pc6h  = _num(token.get("price_change_6h",  pc.get("h6",  0.0)), 0.0)
    pc24h = _num(token.get("price_change_24h", pc.get("h24", 0.0)), 0.0)

    # Age shaping
    created_raw = token.get("pairCreatedAt") or token.get("creation_timestamp") or 0
    try:
        created_ms = int(created_raw)
    except Exception:
        created_ms = 0
    if 0 < created_ms < 10_000_000_000:
        created_ms *= 1000
    age_h = (time.time() - (created_ms / 1000.0)) / 3600.0 if created_ms else None

    # 1) Market cap window (mid-cap sweet spot)
    MID_MIN, MID_MAX = 100_000.0, 500_000.0
    if mc <= 0:
        mc_s = 0.0
    elif mc < MID_MIN:
        mc_s = _clip(mc / MID_MIN) * 0.8
    elif mc <= MID_MAX:
        mc_s = 1.0
    else:
        mc_s = _clip((MID_MAX / max(mc, 1.0)) ** 0.4)

    # 2) Liquidity & Volume scaling (aggressive)
    LIQ_SOFT = 25_000.0   # USD
    VOL_SOFT = 150_000.0  # USD
    liq_s = _clip(math.tanh(liq / max(LIQ_SOFT, 1.0)))
    vol_s = _clip(math.tanh(vol / max(VOL_SOFT, 1.0)))

    # 3) Momentum (cap blow-offs; favor 1h > 6h > 24h)
    def _mom(x: float, cap: float) -> float:
        if x <= 0:
            return 0.0
        x = min(x, cap)
        return _clip(math.tanh(x / cap))
    m1  = _mom(pc1h, 40.0)
    m6  = _mom(pc6h, 120.0)
    m24 = _mom(pc24h, 250.0)
    momentum = (0.50 * m1) + (0.30 * m6) + (0.20 * m24)

    # 4) Age shaping (avoid <15m and very old)
    if age_h is None:
        age_s = 0.7
    else:
        if age_h < (15.0 / 60.0):   # < 15 minutes
            age_s = 0.2
        elif age_h < 2.0:          # 15m..2h
            age_s = 0.7
        elif age_h < 72.0:         # 2h..3d
            age_s = 1.0
        elif age_h < 720.0:        # 3d..30d
            age_s = 0.8
        else:
            age_s = 0.6

    # 5) Birdeye presence bonus / absence penalty
    birdeye_present = bool(token.get("created_at")) or (str(token.get("source") or "").lower() == "birdeye")
    presence_mult = 1.10 if birdeye_present else 0.85

    # 6) Category nudges
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

def _birdeye_present(token: Dict[str, Any]) -> bool:
    return bool(token.get("created_at")) or (str(token.get("source") or "").lower() == "birdeye")

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

    presence_mult = 1.05 if _birdeye_present(token) else 0.9

    base = (0.30 * mc_s) + (0.35 * liq_s) + (0.25 * vol_s) + (0.10 * momentum)
    final = base * age_s * presence_mult
    return float(max(0.0, min(100.0, final * 100.0)))

def score_token_dispatch(token: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    """
    Dispatch scoring mode, plus optional BBands-flip gating/bonus controlled via:
      config:
        signals:
          bbands_flip:
            enable: true
            reject_shorts: true
            long_bonus_points: 15
            long_bonus_points_narrow: 5
            min_width: 0.015
    Expect these fields on token if enrichment ran:
      bb_long (bool), bb_short (bool), bb_upper, bb_basis, bb_lower
    """
    mode = str(((cfg or {}).get("scoring", {}) or {}).get("mode") or "midcap").lower()
    if mode == "conservative":
        score = score_token_conservative(token)
    else:
        score = score_token(token)

    # ---- Optional BBands flip hook (non-breaking; defaults safe) ----
    bb_cfg = (((cfg or {}).get("signals") or {}).get("bbands_flip") or {})
    if bool(bb_cfg.get("enable", True)):
        bb_long  = bool(token.get("bb_long"))
        bb_short = bool(token.get("bb_short"))
        if bb_short and bool(bb_cfg.get("reject_shorts", True)):
            token["_reject_reason"] = "bb_short_flip"
            return -1e9  # hard-reject
        if bb_long:
            basis = token.get("bb_basis")
            upper = token.get("bb_upper")
            lower = token.get("bb_lower")
            width_min = float(bb_cfg.get("min_width", 0.0))
            width_pct = 0.0
            try:
                if basis not in (None, 0.0) and upper is not None and lower is not None:
                    width_pct = (float(upper) - float(lower)) / float(basis)
            except Exception:
                width_pct = 0.0
            bump = int(bb_cfg.get("long_bonus_points", 15)) if width_pct >= width_min else int(bb_cfg.get("long_bonus_points_narrow", 5))
            score += bump

    # Keep a small floor for whitelisted items
    cats = (token.get("categories") or []) or []
    minimum_floor = 5.0
    if ("whitelisted" in cats) and score < minimum_floor:
        score = minimum_floor
    return float(score)

def log_scoring_telemetry(tokens: List[Dict[str, Any]], where: str = "eligible") -> None:
    if not tokens:
        logger.info(f"[SCORE] {where}: empty set")
        return
    total = 0.0
    have_be = 0
    n = 0
    for t in tokens:
        try:
            total += float(t.get("score", 0.0))
            have_be += 1 if _birdeye_present(t) else 0
            n += 1
        except Exception:
            continue
    avg = total / max(1, n)
    pct = (have_be / max(1, n)) * 100.0
    logger.info(f"[SCORE] {where}: avg={avg:.1f} birdeye_coverage={have_be}/{n} ({pct:.1f}%)")

# ---------- formatting helpers ----------------------------------------------------------
def format_market_cap(value: float) -> str:
    try:
        num = float(value)
    except Exception:
        return "N/A"

    abs_num = abs(num)
    if abs_num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif abs_num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif abs_num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return f"{num:.2f}"
    
def format_usd(value: float | int | None, *, compact: bool = False, decimals: int = 2) -> str:
    try:
        num = float(value)
    except Exception:
        return "$0.00"

    if not math.isfinite(num):
        return "$0.00"

    sign = "-" if num < 0 else ""
    n = abs(num)

    if compact:
        if n >= 1_000_000_000:
            return f"{sign}${n/1_000_000_000:.{decimals}f}B"
        if n >= 1_000_000:
            return f"{sign}${n/1_000_000:.{decimals}f}M"
        if n >= 1_000:
            return f"{sign}${n/1_000:.{decimals}f}K"
        return f"{sign}${n:.{decimals}f}"

    return f"{sign}${n:,.{decimals}f}"

def format_ts_human(ts: int | float | str | None, *, with_time: bool = True) -> str:
    if ts is None:
        return "-"
    try:
        t = float(ts)
    except Exception:
        return "-"
    if 0 < t < 10_000_000_000:
        dt = datetime.fromtimestamp(t)
    else:
        dt = datetime.fromtimestamp(t / 1000.0)
    return dt.strftime("%Y-%m-%d %H:%M:%S") if with_time else dt.strftime("%Y-%m-%d")
    
def fmt_usd(value):
    return format_usd(value, compact=True, decimals=2)

def fmt_ts(ts):
    return format_ts_human(ts, with_time=True)

# -----------------------------------------------------------------------------
# Jupiter swap execution (robust)
# -----------------------------------------------------------------------------
async def execute_jupiter_swap(
    quote: Dict,
    user_pubkey: str,
    wallet: "Keypair",
    solana_client: AsyncClient,
) -> Optional[str]:
    try:
        swap_b64 = quote.get("swapTransaction")
        if not swap_b64:
            logger.error("No 'swapTransaction' field present in quote.")
            return None

        raw_tx = base64.b64decode(swap_b64)

        # 1) Try raw submit first
        try:
            resp = await solana_client.send_raw_transaction(raw_tx)
            sig = getattr(resp, "value", None) or (resp.get("result") if isinstance(resp, dict) else None)
            if sig:
                return str(sig)
        except Exception as e_raw:
            logger.debug("send_raw_transaction failed; will try signing route: %s", e_raw)

        # 2) Try legacy Transaction with signer
        try:
            tx = Transaction.deserialize(raw_tx)
            if Keypair is None or wallet is None:
                logger.error("Wallet/Keypair not available to sign the transaction.")
                return None
            resp2 = await solana_client.send_transaction(tx, wallet)
            sig2 = getattr(resp2, "value", None) or (resp2.get("result") if isinstance(resp2, dict) else None)
            if sig2:
                return str(sig2)
        except Exception as e_legacy:
            logger.debug("Legacy Transaction path failed; will try VersionedTransaction: %s", e_legacy)

        # 3) If versioned and VersionedTransaction is available, retry raw (network transient)
        if VersionedTransaction is not None:
            try:
                _ = VersionedTransaction.deserialize(raw_tx)
                resp3 = await solana_client.send_raw_transaction(raw_tx)
                sig3 = getattr(resp3, "value", None) or (resp3.get("result") if isinstance(resp3, dict) else None)
                if sig3:
                    return str(sig3)
            except Exception as e_vtx:
                logger.debug("VersionedTransaction path failed: %s", e_vtx)

        logger.error("Failed to submit Jupiter swap transaction after multiple attempts.")
        return None

    except Exception as e:
        logger.error("Failed to execute Jupiter swap: %s", e)
        return None

# -----------------------------------------------------------------------------
# Secret management (first-run prompt + persistence)
# -----------------------------------------------------------------------------
def _is_json_64int_secret(s: str) -> bool:
    try:
        arr = json.loads(s)
        return isinstance(arr, list) and len(arr) == 64 and all(isinstance(x, int) and 0 <= x <= 255 for x in arr)
    except Exception:
        return False

def _looks_base58_secret(s: str) -> bool:
    return isinstance(s, str) and len(s.strip()) >= 64 and s.strip().isalnum()

def validate_solana_secret(s: str) -> bool:
    s = (s or "").strip()
    return _is_json_64int_secret(s) or _looks_base58_secret(s)

def validate_birdeye_key(s: str) -> bool:
    return isinstance(s, str) and len(s.strip()) >= 12

def get_secret_from_env(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v and isinstance(v, str) and v.strip():
        return v.strip()
    data = _parse_dotenv_file(ENV_PATH)
    v2 = data.get(name)
    return v2.strip() if isinstance(v2, str) and v2.strip() else None

def set_secret(name: str, value: str) -> None:
    os.environ[name] = value
    _write_env_key(name, value)

def ensure_required_secrets(interactive: bool = True) -> Dict[str, bool]:
    import sys

    sol = get_secret_from_env("SOLANA_PRIVATE_KEY")
    bir = get_secret_from_env("BIRDEYE_API_KEY")

    have_tty = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()

    flags = {"SOLANA_PRIVATE_KEY": False, "BIRDEYE_API_KEY": False}
    if sol and validate_solana_secret(sol):
        flags["SOLANA_PRIVATE_KEY"] = True
    if bir and validate_birdeye_key(bir):
        flags["BIRDEYE_API_KEY"] = True

    if interactive and have_tty:
        if not flags["SOLANA_PRIVATE_KEY"]:
            print("\n=== First-run setup: SOLANA_PRIVATE_KEY ===")
            print("Paste your Solana secret EITHER as JSON array of 64 ints OR as a base58 64-byte secret.")
            sol_in = input("SOLANA_PRIVATE_KEY: ").strip()
            if validate_solana_secret(sol_in):
                set_secret("SOLANA_PRIVATE_KEY", sol_in)
                flags["SOLANA_PRIVATE_KEY"] = True
            else:
                print("(!) That doesn't look valid. You can also put it in your .env later.")

        if not flags["BIRDEYE_API_KEY"]:
            print("\n=== First-run setup: BIRDEYE_API_KEY ===")
            bir_in = input("BIRDEYE_API_KEY: ").strip()
            if validate_birdeye_key(bir_in):
                set_secret("BIRDEYE_API_KEY", bir_in)
                flags["BIRDEYE_API_KEY"] = True
            else:
                print("(!) That doesn't look valid. You can also put it in your .env later.")

    return flags

# -----------------------------------------------------------------------------
# Config helpers (Dexscreener post-cap)
# -----------------------------------------------------------------------------
def get_dex_post_cap(config: Dict) -> int:
    try:
        disc = dict((config or {}).get("discovery") or {})
    except Exception:
        disc = {}
    try:
        cap = int(disc.get("dexscreener_post_cap", 0))
    except Exception:
        cap = 0
    if cap <= 0:
        try:
            cap = int(os.getenv("DEX_MAX") or os.getenv("DEXSCREENER_MAX", "0"))
        except Exception:
            cap = 0
    return max(0, cap)

# -----------------------------------------------------------------------------
# Public exports
# -----------------------------------------------------------------------------
__all__ = [
    "WHITELISTED_TOKENS",
    "price_cache",
    "token_balance_cache",
    "token_account_existence_cache",
    "load_config",
    "setup_logging",
    "ENV_PATH",
    "_parse_dotenv_file",
    "_write_env_key",
    "get_rugcheck_token",
    "get_rugcheck_headers",
    "get_rugcheck_token_async",
    "add_to_blacklist",
    "deduplicate_tokens",
    "remove_none_keys",
    "custom_json_encoder",
    "get_buy_amount",
    "compute_buy_size_usd",
    "backoff_sleep",
    "CircuitBreaker429",
    "normalize_labels",
    "is_flagged",
    "score_token",
    "format_market_cap",
    "format_usd",
    "format_ts_human",
    "execute_jupiter_swap",
    # secret management
    "get_secret_from_env",
    "set_secret",
    "validate_solana_secret",
    "validate_birdeye_key",
    "ensure_required_secrets",
    # scoring selector + telemetry
    "score_token_dispatch",
    "score_token_conservative",
    "log_scoring_telemetry",
    # config helpers
    "get_dex_post_cap",
    # convenience shorthands
    "fmt_usd",
    "fmt_ts",
    # volume & pattern helpers (re-exports)
    "rolling_median",
    "rolling_mad",
    "is_volume_spike",
    "is_evening_star",
    "is_morning_star",
    "is_three_black_crows",
    "is_three_white_soldiers",
    "is_hammer",
    "is_inverted_hammer",
    "is_bullish_engulfing",
    "is_bearish_engulfing",
    "is_doji",
]
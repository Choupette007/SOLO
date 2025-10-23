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

import yaml
from cachetools import TTLCache
from logging.handlers import RotatingFileHandler
from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey
from solders.signature import Signature

# Prefer modern solders types (solana-py >= 0.36)
try:
    from solders.transaction import Transaction, VersionedTransaction  # type: ignore
except Exception as e:
    logging.getLogger("TradingBot").error(
        "Failed to import solders.transaction. Ensure compatible versions of 'solana' and 'solders' are installed. "
        "Recommended: solana>=0.36,<0.37 and solders>=0.26,<0.27. Underlying error: %s",
        e,
    )
    raise

# Optional signing support
try:
    from solders.keypair import Keypair  # type: ignore
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

logger = logging.getLogger("TradingBot")

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
        # Normalize via constants' resolver when available
        v2 = prefer_appdata_file(v)  # may pass through unchanged if already absolute
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
        # If it exists or its parent is writable, use it
        try:
            if c.exists() or os.access(str(c.parent), os.W_OK):
                return c
        except Exception:
            continue
    # Fallback to appdata
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
                    # keep conservative defaults; GUI can change these
                    "dexscreener_pages": 10,
                    "dexscreener_per_page": 100,
                    "dexscreener_post_cap": 0,
                    "raydium_max_pairs": 1000,
                    "raydium_page_size": 100,
                    "raydium_max_pools": 1500,
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
        # Non-fatal; we'll load as {} below
        logging.getLogger("TradingBot").debug("Could not create default config at %s: %s", cfg_path, e)

def load_config(path: str | None = None) -> Dict:
    global _missing_cfg_last_log_ts
    cfg_path = _resolve_config_path(path)
    # Create a small default if missing/empty (out-of-the-box UX)
    _create_default_config(cfg_path)
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        logger.debug("Loaded configuration from %s", cfg_path)
        return config
    except Exception as e:
        # Throttle spammy log lines when file is absent/locked on first run
        now = time.time()
        if now - _missing_cfg_last_log_ts > 30:
            logger.error("Failed to load config from %s: %s", cfg_path, e)
            _missing_cfg_last_log_ts = now
        return {}

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

# sentinel to avoid reconfig churn on reruns
_LOG_SENTINEL_ATTR = "_solo_logging_file"

def setup_logging(config: dict[str, Any] | None) -> logging.Logger:
    log_cfg = (config or {}).get("logging", {}) if isinstance(config, dict) else {}
    from solana_trading_bot_bundle.common.constants import logs_dir as _logs_dir  

    # Resolve path & options
    raw_file = log_cfg.get("file") or (_logs_dir() / "bot.log")
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
        # still ensure level stays in sync
        for h in root.handlers:
            h.setLevel(level)
        return logging.getLogger("TradingBot")

    # ----- File handler (dedupe by exact path on ROOT) -----
    file_handler: Optional[RotatingFileHandler] = None
    for h in list(root.handlers):
        # keep only the first rotating handler pointing to the same file
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
    bot_logger = logging.getLogger("TradingBot")
    bot_logger.handlers.clear()
    bot_logger.propagate = True
    bot_logger.setLevel(level)

    gui_logger = logging.getLogger("TradingBotGUI")
    gui_logger.handlers.clear()
    gui_logger.propagate = True
    gui_logger.setLevel(level)

    # mark configured to this file (prevents duplicate setup on reruns)
    setattr(root, _LOG_SENTINEL_ATTR, str(log_file))

    bot_logger.info("Logging configured: level=%s, file=%s", level_name, str(log_file))
    return bot_logger

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
        raise RuntimeError("solders.Keypair not available; cannot sign Rugcheck login.")
    pk = (os.getenv("SOLANA_PRIVATE_KEY") or os.getenv("WALLET_PRIVATE_KEY") or "").strip()
    if not pk:
        raise ValueError("Set SOLANA_PRIVATE_KEY or WALLET_PRIVATE_KEY in your .env to auto-login Rugcheck.")
    # Try base58 64-byte secret key
    try:
        # solders supports base58 secret directly
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

# utils.py  (≈ lines 421–440 in your current file)

async def get_buy_amount(token: Dict, wallet_balance: float, sol_price: float, config: Dict) -> Tuple[float, float]:
    per_trade_sol = float(config.get("bot", {}).get("per_trade_sol", 0.01))
    buy_percentage = float(config.get("bot", {}).get("buy_percentage", 0.10))
    dry_run = bool(config.get("bot", {}).get("dry_run", True))  # honor DRY_RUN from config
    min_usd_per_buy = float(config.get("bot", {}).get("min_usd_per_buy", 0.0))  # e.g. 50.0

    # baseline sizing from config
    amount = min(per_trade_sol, wallet_balance * buy_percentage)
    usd_amount = amount * sol_price if sol_price and sol_price > 0 else 0.0

    # enforce per-buy USD floor if configured and we have a price
    if min_usd_per_buy > 0 and sol_price and sol_price > 0:
        floor_sol = min_usd_per_buy / sol_price

        if dry_run:
            # in dry-run, allow simulating at least the floor regardless of wallet balance
            amount = max(amount, floor_sol)
        else:
            # in live mode, do not exceed available balance
            amount = min(wallet_balance, max(amount, floor_sol))

        usd_amount = amount * sol_price

    # final guard: if still zero in dry-run, simulate the floor outright
    if dry_run and amount <= 0 and min_usd_per_buy > 0 and sol_price and sol_price > 0:
        amount = min_usd_per_buy / sol_price
        usd_amount = amount * sol_price

    return amount, usd_amount


# -----------------------------------------------------------------------------
# Labels + Scoring (new)
# -----------------------------------------------------------------------------
def normalize_labels(labels_any: Any) -> List[str]:
    """
    Accepts Rugcheck-like label arrays (strings/dicts/mixed) and returns lowercase names.
    e.g. [{"label":"scam"}, "dangerous", {"name": "Honeypot"}] -> ["scam","dangerous","honeypot"]
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
    # de-dup and drop empties
    return list({x for x in out if x})

def is_flagged(labels_any: Any, bad_keys: Optional[set[str]] = None) -> bool:
    bad = bad_keys or {"honeypot", "scam", "blocked", "malicious", "dangerous"}
    labs = set(normalize_labels(labels_any))
    return len(bad.intersection(labs)) > 0

def score_token(token: Dict[str, Any]) -> float:
    """
    Aggressive, mid-cap biased scoring (Birdeye-aware) with 0..100 output.

    Highlights:
      - Prefers mc ~100k..500k (mid-cap window)
      - Momentum tilt (1h > 6h > 24h), caps blow-offs
      - Liquidity & volume scaling (tanh saturation)
      - +10% boost if Birdeye data present, −15% penalty if missing
      - Light age shaping (de-emphasize <15m and very old)
      - Small penalty for large-caps; tiny penalty for low-caps
    """

    # --------- Hard safety gates (keep these up front) ----------
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

    # Age shaping: Dexscreener 'pairCreatedAt' is ms; allow other fields too.
    created_raw = token.get("pairCreatedAt") or token.get("creation_timestamp") or 0
    try:
        created_ms = int(created_raw)
    except Exception:
        created_ms = 0
    # Heuristic: if seconds, convert to ms
    if 0 < created_ms < 10_000_000_000:
        created_ms *= 1000
    age_h = (time.time() - (created_ms / 1000.0)) / 3600.0 if created_ms else None

    # --------- 1) Market cap window (mid-cap sweet spot) ----------
    MID_MIN, MID_MAX = 100_000.0, 500_000.0
    if mc <= 0:
        mc_s = 0.0
    elif mc < MID_MIN:
        mc_s = _clip(mc / MID_MIN) * 0.8  # ramp toward 100k
    elif mc <= MID_MAX:
        mc_s = 1.0  # ideal band
    else:
        # decay as it grows beyond mid-cap (softly)
        mc_s = _clip((MID_MAX / max(mc, 1.0)) ** 0.4)

    # --------- 2) Liquidity & Volume scaling (aggressive) ----------
    # Soft targets; shape with tanh for diminishing returns
    LIQ_SOFT = 25_000.0   # USD
    VOL_SOFT = 150_000.0  # USD
    liq_s = _clip(math.tanh(liq / max(LIQ_SOFT, 1.0)))
    vol_s = _clip(math.tanh(vol / max(VOL_SOFT, 1.0)))

    # --------- 3) Momentum (cap blow-offs; favor 1h > 6h > 24h) ----------
    def _mom(x: float, cap: float) -> float:
        if x <= 0:
            return 0.0
        x = min(x, cap)
        return _clip(math.tanh(x / cap))
    m1  = _mom(pc1h, 40.0)
    m6  = _mom(pc6h, 120.0)
    m24 = _mom(pc24h, 250.0)
    momentum = (0.50 * m1) + (0.30 * m6) + (0.20 * m24)

    # --------- 4) Age shaping (avoid <15m and very old) ----------
    if age_h is None:
        age_s = 0.7  # unknown age; neutral-ish
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

    # --------- 5) Birdeye presence bonus / absence penalty ----------
    # Heuristics: Birdeye fetch usually sets 'source' or a Birdeye timestamp like 'created_at'
    birdeye_present = bool(token.get("created_at")) or (str(token.get("source") or "").lower() == "birdeye")
    presence_mult = 1.10 if birdeye_present else 0.85

    # --------- 6) Category nudges ----------
    cats = (token.get("categories") or []) or []
    cat_mult = 1.0
    if "mid_cap" in cats:
        cat_mult *= 1.10
    if "large_cap" in cats:
        cat_mult *= 0.92
    if "low_cap" in cats:
        cat_mult *= 0.97

    # Base blend (aggressive weights). Each component is ~[0..1].
    base = (0.35 * mc_s) + (0.25 * liq_s) + (0.20 * vol_s) + (0.20 * momentum)

    # Final composition
    final = base * age_s * presence_mult * cat_mult

    # Return in 0..100 range for GUI/sorting consistency (clamped)
    score = max(0.0, min(100.0, final * 100.0))
    return float(score)

# -----------------------------------------------------------------------------
# Scoring dispatcher, conservative mode, and lightweight telemetry
# -----------------------------------------------------------------------------

def _birdeye_present(token: Dict[str, Any]) -> bool:
    """Heuristic: treat Birdeye presence as a simple data-quality signal."""
    return bool(token.get("created_at")) or (str(token.get("source") or "").lower() == "birdeye")


def score_token_conservative(token: Dict[str, Any]) -> float:
    """
    A calmer baseline scorer (0..100) for A/B or fallback:
      - Less momentum weight, more volume/liquidity
      - Wider mcap comfort zone (50k..2.5M)
      - Still honors Rugcheck 'blocked' / bad labels -> score 0
    """
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

    # Mcap comfort band: 50k..2.5M (gentler decay outside)
    LO, HI = 50_000.0, 2_500_000.0
    if mc <= 0:
        mc_s = 0.0
    elif mc < LO:
        mc_s = _clip(mc / LO) * 0.7
    elif mc <= HI:
        mc_s = 1.0
    else:
        mc_s = _clip((HI / max(mc, 1.0)) ** 0.25)

    # Liquidity/Volume saturate
    liq_s = _clip(math.tanh(liq / 20_000.0))
    vol_s = _clip(math.tanh(vol / 100_000.0))

    # Momentum (lower weight, capped)
    def _mom(x: float, cap: float) -> float:
        if x <= 0:
            return 0.0
        x = min(x, cap)
        return _clip(math.tanh(x / cap))
    m1  = _mom(pc1h, 25.0)
    m6  = _mom(pc6h, 80.0)
    m24 = _mom(pc24h, 200.0)
    momentum = (0.35 * m1) + (0.35 * m6) + (0.30 * m24)

    # Age: prefer 2h..7d
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
    elif age_h < 168.0:   # 2h..7d
        age_s = 1.0
    elif age_h < 720.0:   # 7d..30d
        age_s = 0.8
    else:
        age_s = 0.6

    presence_mult = 1.05 if _birdeye_present(token) else 0.9

    base = (0.30 * mc_s) + (0.35 * liq_s) + (0.25 * vol_s) + (0.10 * momentum)
    final = base * age_s * presence_mult
    return float(max(0.0, min(100.0, final * 100.0)))


def score_token_dispatch(token: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    """
    Switch between midcap (aggressive) and conservative scorers via config:
        scoring.mode: "midcap" | "conservative"
    Also enforces a small **minimum floor for whitelisted tokens** so manual curation
    can override a low computed score (guardrail).
    """
    mode = str(((cfg or {}).get("scoring", {}) or {}).get("mode") or "midcap").lower()
    if mode == "conservative":
        s = score_token_conservative(token)
    else:
        s = score_token(token)

    # Guardrail: never bury a whitelisted token (trading.py tags them as 'whitelisted')
    cats = (token.get("categories") or []) or []
    minimum_floor = 5.0
    if ("whitelisted" in cats) and s < minimum_floor:
        s = minimum_floor
    return float(s)


def log_scoring_telemetry(tokens: List[Dict[str, Any]], where: str = "eligible") -> None:
    """
    Lightweight per-cycle telemetry: average score and Birdeye coverage rate.
    Call from trading loop after you've assigned token['score'].
    """
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
    """
    Format a market cap number into a human-readable string with suffixes.
    Examples:
      1234        -> "1.23K"
      1234567     -> "1.23M"
      1234567890  -> "1.23B"
      123         -> "123"
    """
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
    """
    Format a number as a USD string.
      - compact=True  -> uses K/M/B suffixes, e.g. "$1.23M"
      - compact=False -> comma-separated dollars, e.g. "$1,234,567.89"
    Safe for None/NaN/str inputs.
    """
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

    # Non-compact: full dollars with thousands separators
    return f"{sign}${n:,.{decimals}f}"


def format_ts_human(ts: int | float | str | None, *, with_time: bool = True) -> str:
    """
    Render epoch timestamp (seconds OR milliseconds) as a readable local time.
    Examples:
      1759595901     -> "2025-12-04 15:18:21"
      1759595901000  -> "2025-12-04 15:18:21"
    Falls back to "-" on bad values.
    """
    if ts is None:
        return "-"

    try:
        t = float(ts)
    except Exception:
        return "-"
    
    # --- Adapters for the GUI table ---
def fmt_usd(value):
    # Compact USD like $12.3K / $4.5M
    return format_usd(value, compact=True, decimals=2)

def fmt_ts(ts):
    # Human local timestamp YYYY-MM-DD HH:MM:SS
    return format_ts_human(ts, with_time=True)


    # Heuristic: treat < 1e10 as seconds, otherwise milliseconds.
    if 0 < t < 10_000_000_000:
        dt = datetime.fromtimestamp(t)
    else:
        dt = datetime.fromtimestamp(t / 1000.0)

    return dt.strftime("%Y-%m-%d %H:%M:%S") if with_time else dt.strftime("%Y-%m-%d")
    

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

        # 3) If versioned, retry raw (network transient)
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
    # Loose pre-check only; real validation happens in _load_wallet_keypair_from_env()
    return isinstance(s, str) and len(s.strip()) >= 64 and s.strip().isalnum()

def validate_solana_secret(s: str) -> bool:
    s = (s or "").strip()
    return _is_json_64int_secret(s) or _looks_base58_secret(s)

def validate_birdeye_key(s: str) -> bool:
    # Birdeye keys are opaque strings; tweak length/pattern if you know their format.
    return isinstance(s, str) and len(s.strip()) >= 12

def get_secret_from_env(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v and isinstance(v, str) and v.strip():
        return v.strip()
    # also check .env managed by our utils
    data = _parse_dotenv_file(ENV_PATH)
    v2 = data.get(name)
    return v2.strip() if isinstance(v2, str) and v2.strip() else None

def set_secret(name: str, value: str) -> None:
    os.environ[name] = value
    _write_env_key(name, value)

def ensure_required_secrets(interactive: bool = True) -> Dict[str, bool]:
    """
    Ensure SOLANA_PRIVATE_KEY and BIRDEYE_API_KEY exist.

    Returns flags dict indicating validity:
      {"SOLANA_PRIVATE_KEY": bool, "BIRDEYE_API_KEY": bool}

    Behavior:
      - If present: validate and return.
      - If missing and interactive & TTY: prompt in console, then persist to .env and process env.
      - If missing and not interactive or no TTY: return False flags (let GUI collect them).
    """
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
    """
    Returns the overall Dexscreener post-cap from config (discovery.dexscreener_post_cap).
    If 0/missing, falls back to env vars DEX_MAX or DEXSCREENER_MAX.
    0 means "no cap".
    """
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
    "add_to_blacklist",
    "deduplicate_tokens",
    "remove_none_keys",
    "custom_json_encoder",
    "get_buy_amount",
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


]

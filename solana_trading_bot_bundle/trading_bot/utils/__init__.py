# solana_trading_bot_bundle/trading_bot/utils/__init__.py
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable, Any

# ---- Re-export env helpers from the utils package --------------------
from .env_loader import (  # type: ignore
    APP_DIR_NAME as APP_NAME,
    ensure_appdata_env_bootstrap,
    load_env_first_found,
    prefer_appdata_path,
    prefer_appdata_file,
)

# ---- Base caches (will alias to legacy if present) -------------------
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
        sys.modules[spec.name] = mod  # aid relative imports/debugging
        spec.loader.exec_module(mod)  # type: ignore[assignment]
        return mod
    except Exception:
        return None

def _export_from_legacy(mod: ModuleType, names: Iterable[str] | None = None) -> None:
    # Export all public symbols from legacy unless a specific set provided.
    public = [n for n in dir(mod) if not n.startswith("_")] if names is None else list(names)
    for n in public:
        # Never clobber symbols we already defined here.
        if n in globals():
            continue
        globals()[n] = getattr(mod, n)

_legacy_mod = _load_legacy_utils()

# If legacy has cache dicts, alias ours to *the same objects* for consistency.
if _legacy_mod is not None:
    for _name in ("price_cache", "token_balance_cache", "recent_price_cache"):
        if hasattr(_legacy_mod, _name):
            globals()[_name] = getattr(_legacy_mod, _name)

    # Export all other public helpers (e.g., add_to_blacklist, get_rugcheck_token, etc.)
    _export_from_legacy(_legacy_mod)

# ---- Compatibility shims (only if legacy didn't provide them) --------

# Sets
if "BLACKLIST" not in globals():
    BLACKLIST: set[str] = set()

if "WHITELISTED_TOKENS" not in globals():
    WHITELISTED_TOKENS: set[str] = set()

# Cache for token account existence checks
if "token_account_existence_cache" not in globals():
    token_account_existence_cache: dict[str, bool] = {}

def _norm_addr(a: str | None) -> str:
    # Solana addresses are case-sensitive; do not lowercase.
    return (a or "").strip()

# Optional DB hooks for blacklist persistence
def _db_get_blacklist_status(addr: str) -> bool | None:
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
        """
        True if the address is blacklisted. Prefers DB-backed check if available,
        otherwise falls back to in-memory BLACKLIST.
        """
        addr = _norm_addr(address)
        if not addr:
            return False
        db_status = _db_get_blacklist_status(addr)
        if db_status is not None:
            return db_status
        return addr in BLACKLIST

# add_to_blacklist
if "add_to_blacklist" not in globals():
    def add_to_blacklist(address: str) -> None:  # type: ignore[redef]
        """
        Add address to blacklist (DB if available, else in-memory set).
        """
        addr = _norm_addr(address)
        if not addr:
            return
        # Try DB first; fall back to in-memory set
        if not _db_add_to_blacklist(addr):
            BLACKLIST.add(addr)

# custom_json_encoder
if "custom_json_encoder" not in globals():
    def custom_json_encoder(obj: Any) -> Any:  # type: ignore[redef]
        """
        Minimal JSON encoder for common non-JSON types used in the bot.
        Return a JSON-serializable representation, or raise TypeError.
        """
        # pathlib.Path
        if isinstance(obj, Path):
            return str(obj)
        # datetime/date/time
        try:
            import datetime as _dt
            if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
                return obj.isoformat()
        except Exception:
            pass
        # Decimal
        try:
            from decimal import Decimal as _Decimal  # type: ignore
            if isinstance(obj, _Decimal):
                return float(obj)
        except Exception:
            pass
        # bytes-like -> hex
        if isinstance(obj, (bytes, bytearray, memoryview)):
            return bytes(obj).hex()
        # sets -> sorted list
        if isinstance(obj, set):
            try:
                return sorted(obj)  # type: ignore[arg-type]
            except Exception:
                return list(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# get_rugcheck_token (tolerant to multiple env var names)
if "get_rugcheck_token" not in globals():
    def get_rugcheck_token() -> str | None:  # type: ignore[redef]
        """
        Return Rugcheck API token from environment if configured.
        Accepts several common env var names.
        """
        for key in (
            "RUGCHECK_JWT_TOKEN", "RUGCHECK_JWT", "RUGCHECK_API_TOKEN",
            "RUGCHECK_API_KEY", "RUGCHECK_TOKEN", "RUGCHECK"
        ):
            val = os.environ.get(key)
            if val:
                return val
        return None

# ---- Final public surface --------------------------------------------
_legacy_public = set(n for n in dir(_legacy_mod) if _legacy_mod and not n.startswith("_"))
_ours_public = {
    # env helpers
    "APP_NAME",
    "ensure_appdata_env_bootstrap",
    "load_env_first_found",
    "prefer_appdata_path",
    "prefer_appdata_file",
    # caches & helpers
    "price_cache",
    "token_balance_cache",
    "recent_price_cache",
    "clear_caches",
    "token_account_existence_cache",
    # lists/sets
    "BLACKLIST",
    "WHITELISTED_TOKENS",
    # shims/helpers
    "is_blacklisted",
    "add_to_blacklist",
    "custom_json_encoder",
    "get_rugcheck_token",
}

__all__ = sorted(_legacy_public | _ours_public)

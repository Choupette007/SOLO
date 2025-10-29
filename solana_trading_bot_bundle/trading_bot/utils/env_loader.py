# solana_trading_bot_bundle/utils/env_loader.py
from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Name of the per-user application directory
APP_DIR_NAME = "SOLOTradingBot"

# Relative resource names used by other modules (kept as strings on purpose)
# These are meant to be passed into prefer_appdata_path(...)
token_cache_path: str = "tokens.sqlite3"     # yields .../SOLOTradingBot/tokens.sqlite3
log_file_path: str   = "logs/bot.log"        # yields .../SOLOTradingBot/logs/bot.log


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _exe_dir() -> Path:
    """
    Folder containing the running binary (PyInstaller) or repo root in dev.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys.argv[0]).resolve().parent
    # dev: go up to the project root (bundle/utils/env_loader.py -> bundle -> project)
    return Path(__file__).resolve().parents[2]


def _appdata_base_dir() -> Path:
    """
    Cross-platform per-user appdata base for this app:
      - Windows: %LOCALAPPDATA%/SOLOTradingBot
      - macOS:   ~/Library/Application Support/SOLOTradingBot
      - Linux:   ${XDG_DATA_HOME:-~/.local/share}/SOLOTradingBot
    """
    if os.name == "nt":
        base = (os.environ.get("LOCALAPPDATA")
                or os.environ.get("APPDATA")
                or str(Path.home() / "AppData" / "Local"))
        return Path(base) / APP_DIR_NAME

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME

    # Linux / other POSIX
    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data:
        return Path(xdg_data) / APP_DIR_NAME
    return Path.home() / ".local" / "share" / APP_DIR_NAME


def _appdata_env_path() -> Path:
    return _appdata_base_dir() / ".env"


def _candidate_env_paths() -> list[Path]:
    exe_dir = _exe_dir()
    return [
        Path.cwd() / ".env",   # project CWD (dev)
        exe_dir / ".env",      # alongside exe/binary
        _appdata_env_path(),   # user appdata
    ]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def load_env_first_found(override: bool = False) -> Optional[Path]:
    """
    Load the first existing .env from candidates; return the path used or None.
    """
    for p in _candidate_env_paths():
        if p.exists():
            load_dotenv(dotenv_path=p, override=override)
            return p
    return None


def ensure_appdata_env_bootstrap(
    template_names: tuple[str, ...] = (".env", "default.env")
) -> Path:
    """
    Ensure appdata has a .env. If missing, copy from (1) CWD/.env,
    (2) exe_dir/.env, (3) exe_dir/default.env. If none found, write a skeleton.
    Returns the appdata path (created or existing).
    """
    dst = _appdata_env_path()
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return dst

    exe_dir = _exe_dir()
    # order: CWD .env, exe_dir .env, exe_dir default.env
    sources = [
        Path.cwd() / template_names[0],
        exe_dir / template_names[0],
        exe_dir / template_names[-1],
    ]
    for src in sources:
        if src.exists():
            shutil.copy2(src, dst)
            return dst

    # nothing found; create a minimal skeleton
    dst.write_text(
        "SOLANA_PRIVATE_KEY=\n"
        "BIRDEYE_API_KEY=\n"
        "RUGCHECK_JWT_TOKEN=\n"
        "RUGCHECK_API_KEY=\n"
        "RUGCHECK_ENABLE=true\n"
        "RUGCHECK_DISCOVERY_CHECK=true\n"
        "RUGCHECK_DISCOVERY_FILTER=false\n",
        encoding="utf-8",
    )
    return dst


def prefer_appdata_path(
    relative_name: str,
    *,
    ensure_parent: bool = True,
    default_contents: Optional[str] = None,
) -> Path:
    """
    Return a Path under the per-user appdata directory for this app.
    If ensure_parent is True, create parent directories.
    If default_contents is provided and file is missing, write it.
    """
    # Normalize separators and strip leading slashes to avoid absolute hijack
    rel = str(relative_name).lstrip("/\\")
    p = _appdata_base_dir() / rel
    if ensure_parent:
        p.parent.mkdir(parents=True, exist_ok=True)
    if default_contents is not None and not p.exists():
        p.write_text(default_contents, encoding="utf-8")
    return p


# --- Back-compat alias (some modules still import/call this older name) ---
def prefer_appdata_file(
    relative_name: str,
    *,
    ensure_parent: bool = True,
    default_contents: Optional[str] = None,
) -> str:
    """
    DEPRECATED: use prefer_appdata_path().
    Accepts a relative filename; returns a string path.
    """
    return str(
        prefer_appdata_path(
            relative_name,
            ensure_parent=ensure_parent,
            default_contents=default_contents,
        )
    )


def get_active_private_key() -> Optional[str]:
    """
    Fetch SOLANA_PRIVATE_KEY from the environment (after load_env_first_found).
    Trims surrounding quotes and whitespace.
    """
    v = os.environ.get("SOLANA_PRIVATE_KEY")
    if not v:
        return None
    v = v.strip().strip('"').strip("'")
    return v or None


def ensure_one_time_credentials() -> None:
    """
    Placeholder for any one-time seeding of credentials.
    Currently a no-op; kept for API compatibility with existing imports.
    """
    return None


# Export list (nice to have, not required)
__all__ = [
    "APP_DIR_NAME",
    "token_cache_path",
    "log_file_path",
    "load_env_first_found",
    "ensure_appdata_env_bootstrap",
    "prefer_appdata_path",
    "prefer_appdata_file",  # back-compat
    "get_active_private_key",
    "ensure_one_time_credentials",
]

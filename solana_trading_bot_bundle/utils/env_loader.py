# solana_trading_bot_bundle/utils/env_loader.py
from __future__ import annotations
import os, sys, shutil
from pathlib import Path
from dotenv import load_dotenv

APP_DIR_NAME = "SOLOTradingBot"

def _exe_dir() -> Path:
    # Works for dev & PyInstaller
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys.argv[0]).resolve().parent
    return Path(__file__).resolve().parents[2]  # project root in dev

def _appdata_env_path() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_DIR_NAME / ".env"
    # macOS / Linux
    mac = Path.home() / "Library" / "Application Support" / APP_DIR_NAME / ".env"
    xdg = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / APP_DIR_NAME / ".env"
    # Prefer mac path on macOS; else XDG
    return mac if sys.platform == "darwin" else xdg

def _candidate_env_paths() -> list[Path]:
    exe_dir = _exe_dir()
    return [
        Path.cwd() / ".env",                 # project CWD (dev)
        exe_dir / ".env",                    # alongside exe/binary
        _appdata_env_path(),                 # appdata
    ]

def load_env_first_found(override: bool = False) -> Path | None:
    """Load the first existing .env from candidates; return the path used."""
    for p in _candidate_env_paths():
        if p.exists():
            load_dotenv(dotenv_path=p, override=override)
            return p
    return None

def ensure_appdata_env_bootstrap(template_names: tuple[str, ...] = (".env", "default.env")) -> Path:
    """
    Ensure appdata has a .env. If missing, copy from (1) CWD/.env,
    (2) exe_dir/.env, (3) exe_dir/default.env (packaged template).
    Returns the appdata path (created or existing).
    """
    dst = _appdata_env_path()
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return dst

    exe_dir = _exe_dir()
    # order: CWD .env, exe_dir .env, exe_dir default.env
    sources = [Path.cwd() / template_names[0], exe_dir / template_names[0], exe_dir / template_names[-1]]
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

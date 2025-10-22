# ----------solana_bot_gui_corrected_working.py-----------------
# -------------------- top-of-file preamble --------------------
from __future__ import annotations

import os, sys, json
from pathlib import Path
import streamlit as st

# ✅ MUST be the first Streamlit call (before any st.markdown / st.write)
st.set_page_config(
    page_title="SOLO Meme Coin Trading Bot",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Package bootstrap so 'solana_trading_bot_bundle' is importable everywhere ---
# Place this BEFORE any 'from solana_trading_bot_bundle ...' imports.
try:
    _GUI_ROOT = Path(__file__).resolve().parent
    _PKG_ROOT = _GUI_ROOT                     # repo root (folder that contains solana_trading_bot_bundle)
    _PKG_DIR  = _PKG_ROOT / "solana_trading_bot_bundle"

    # Make repo root importable in THIS process…
    if str(_PKG_ROOT) not in sys.path:
        sys.path.insert(0, str(_PKG_ROOT))

    # …and in ALL child processes (bot subprocess, etc.)
    _existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(_PKG_ROOT)] + ([p for p in _existing.split(os.pathsep) if p] if _existing else [])
    )

    # Ensure subpackages are real packages for the bot’s -m import
    for sub in ("", "utils", "trading_bot", "common"):
        ip = _PKG_DIR / sub / "__init__.py"
        ip.parent.mkdir(parents=True, exist_ok=True)
        if not ip.exists():
            ip.write_text("# package marker\n", encoding="utf-8")
except Exception as _e:
    # logger isn't initialized yet; use print to avoid NameError
    print(f"[package bootstrap] skipped: {_e}", flush=True)

# --- Import the cosmetic alias helper (safe fallback if constants wasn't updated yet) ---
from solana_trading_bot_bundle.common import constants as _const
display_cap = getattr(_const, "display_cap", lambda s: {"High Cap": "Large Cap"}.get(s, s))

# ---------- Sticky Tab (remember + restore across reruns) ----------
DEFAULT_TAB = "⚙️ Bot Control"

if "active_tab" not in st.session_state:
    st.session_state.active_tab = st.query_params.get("tab") or DEFAULT_TAB

def goto_tab(tab_label: str) -> None:
    st.session_state.active_tab = tab_label
    st.query_params["tab"] = tab_label
    wanted_js = json.dumps(tab_label)
    st.markdown(
        f"""
        <script>
          (function(){{
            const wanted = {wanted_js};
            function tryClick(){{
              const tabs = window.parent.document.querySelectorAll('button[role="tab"]');
              for (const t of tabs) {{ if (t.innerText.trim() === wanted) {{ t.click(); return; }} }}
              setTimeout(tryClick, 80);
            }}
            setTimeout(tryClick, 0);
          }})();
        </script>
        """,
        unsafe_allow_html=True,
    )

_wanted = st.session_state.get("active_tab", DEFAULT_TAB)
st.markdown(
    f"""
    <script>
      (function(){{
        const wanted = {json.dumps(_wanted)};
        function tryClick(){{
          const tabs = window.parent.document.querySelectorAll('button[role="tab"]');
          for (const t of tabs) {{ if (t.innerText.trim() === wanted) {{ t.click(); return; }} }}
          setTimeout(tryClick, 80);
        }}
        setTimeout(tryClick, 0);
      }})();
    </script>
    """,
    unsafe_allow_html=True,
)

# ---- Market-cap buckets & helpers (GUI-side) -------------------------------
# >>> Stable/pegged/wrapped/LP detection (GUI-side)
MC_THRESHOLDS = {
    "high_min": 50_000_000,   # >= 50M → High (tune in the GUI)
    "mid_min":    150_000,    # 150K .. < 50M → Mid
}

GUI_HIDE_STABLES = False

# Symbols we always treat as "stable-like" or pegged/wrapped majors.
# Use lowercase; we normalize the incoming symbol to lowercase.
_STABLE_LIKE_SYMBOLS = {
    # fiat-stables
    "usdc", "usdt", "usdt.e", "usdc.e", "pai", "usdd", "dai",
    # solana staking wrappers / common pegs
    "jitosol", "msol", "stsol", "bsol", "lsol",
    # majors (wrapped/pegged) commonly cluttering "High"
    "sol", "wsol", "weth", "wbtc", "eth", "btc",
    # common generic LP symbol
    "lp",
}

# Name substrings that strongly indicate stables, wrapped majors, or LPs.
# All checks are done in lowercase.
_STABLE_NAME_HINTS = (
    # fiat-stable hints
    "usd", "usdc", "usdt", "usdd", "dai", "pai", "tether", "circle",
    # wrapped majors / bridges
    "wrapped btc", "wrapped bitcoin",
    "wrapped eth", "wrapped ether",
    "(wormhole)", "wormhole",
    # staking wrappers / perps / LPs
    "staked sol", "binance staked sol", "jito staked sol",
    "perps", "perps lp", "liquidity provider", "lp token",
)

def _is_stable_like_row(row: dict) -> bool:
    sym  = str((row.get("symbol") or "")).strip().lower()
    name = str((row.get("name")   or "")).strip().lower()

    # direct symbol blocks
    if sym in _STABLE_LIKE_SYMBOLS:
        return True

    # catch obvious LP names without false positives like "help"/"alpha"
    if " lp" in name or name.endswith(" lp"):
        return True

    # generic name hints
    return any(h in name for h in _STABLE_NAME_HINTS)

def _coerce_float(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if v == v else float(default)  # handle NaN
    except Exception:
        return float(default)

def market_cap_bucket(
    mc_like,
    *,
    mid_floor: float | None = None,
    high_floor: float | None = None,
) -> str:
    """
    Returns 'High' | 'Mid' | 'Low' based on thresholds.
    Uses MC_THRESHOLDS by default, but callers can override.
    """
    if high_floor is None:
        high_floor = float(MC_THRESHOLDS["high_min"])
    if mid_floor is None:
        mid_floor = float(MC_THRESHOLDS["mid_min"])

    mc = _coerce_float(mc_like, 0.0)
    if mc <= 0:
        return "Low"  # treat unknown/zero as Low for grouping

    if mc >= high_floor:
        return "High"
    if mc >= mid_floor:
        return "Mid"
    return "Low"

def market_cap_badge(
    mc_like,
    *,
    mid_floor: float | None = None,
    high_floor: float | None = None,
) -> str:
    """
    Badge used in tables. Accepts mc/fdv/strings; graceful on None/NaN.
    Default thresholds come from MC_THRESHOLDS, but can be overridden.
    """
    mc = _coerce_float(mc_like, 0.0)
    if mc <= 0:
        return "⚪"  # unknown/zero

    bucket = market_cap_bucket(mc, mid_floor=mid_floor, high_floor=high_floor)
    return {"High": "🟢 High", "Mid": "🟡 Mid", "Low": "🔴 Low"}[bucket]

# --- Category normalization helper (canonical names used by trading/eligibility) ---
def _normalize_categories(tok: dict) -> None:
    """
    Ensure categories contain canonical names used by trading/eligibility:
      - low_cap, mid_cap, large_cap, newly_launched
    Accept common aliases and map them to canonical names in-place.
    """
    if tok is None:
        return
    cats = tok.get("categories") or []
    if isinstance(cats, str):
        cats = [cats]

    out = set()
    for c in cats:
        if not c:
            continue
        cs = str(c).strip().lower()
        if cs in {"high", "high_cap", "large", "large_cap"}:
            out.add("large_cap")
        elif cs in {"mid", "mid_cap"}:
            out.add("mid_cap")
        elif cs in {"low", "low_cap"}:
            out.add("low_cap")
        elif cs in {"new", "newly_launched", "newlylaunched"}:
            out.add("newly_launched")
        elif cs in {"shortlist", "fallback", "unknown_cap"}:
            out.add(cs)
        else:
            out.add(cs)

    # honor/mirror any explicit bucket hint
    try:
        bk = str(tok.get("_bucket") or "").strip().lower()
        if bk:
            if bk in {"high", "high_cap", "large", "large_cap"}:
                out.add("large_cap")
            elif bk in {"mid", "mid_cap"}:
                out.add("mid_cap")
            elif bk in {"low", "low_cap"}:
                out.add("low_cap")
            elif bk in {"new", "newly_launched"}:
                out.add("newly_launched")
    except Exception:
        pass

    tok["categories"] = list(out)

# ---------- UI state defaults ----------
if "disc_auto_refresh" not in st.session_state:
    st.session_state.disc_auto_refresh = False
if "disc_refresh_interval_ms" not in st.session_state:
    st.session_state.disc_refresh_interval_ms = 60_000
if "last_token_refresh" not in st.session_state:
    st.session_state.last_token_refresh = 0.0
def _pct_or_none(v):
    """Return a float percent if numeric, else None.
    Accepts either a percent (e.g. 3.12 or 0.77 meaning 0.77%) or a fraction
    (e.g. 0.0312 meaning 3.12%). Only normalize fractions when the original
    textual form clearly indicates a fraction (e.g. starts with "0." or ".")."""
    try:
        s = str(v).strip()
        if not s:
            return None
        has_percent_sign = s.endswith("%")
        if has_percent_sign:
            s = s[:-1].strip()
        x = float(s)
        if x != x:  # NaN
            return None
    except Exception:
        return None

    s_l = str(v).strip()
    if s_l.startswith(".") or s_l.startswith("0."):
        if abs(x) > 0.0:
            return x * 100.0
    return x

def bind_discovery_autorefresh(label: str = "Enable Auto-Refresh (every 60 seconds)") -> None:
    st.checkbox(label, key="disc_auto_refresh")
    if st.session_state.disc_auto_refresh:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=st.session_state.disc_refresh_interval_ms, key="disc_auto_tick")
        except Exception:
            if not st.session_state.get("_disc_tick_warned"):
                st.info("Auto-refresh active. For precise intervals, install: pip install streamlit-autorefresh==1.0.1")
                st.session_state["_disc_tick_warned"] = True

# ---------- DataFrame styling + numeric right alignment + uniform buttons ----------
# IMPORTANT: delete any older CSS blocks; keep ONLY this one.
st.markdown("""
<style>
/* 1) Stable table layout */
div[data-testid="stDataFrame"] table { table-layout: fixed; width: 100%; }

/* Truncation + allow inner wrappers to shrink */
div[data-testid="stDataFrame"] thead th,
div[data-testid="stDataFrame"] tbody td {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  box-sizing: border-box;
}
div[data-testid="stDataFrame"] thead th > div,
div[data-testid="stDataFrame"] tbody td > div { min-width: 0 !important; }

/* Header look */
div[data-testid="stDataFrame"] thead th { text-align: center; font-weight: 800; }
div[data-testid="stDataFrame"] thead th > div { display: flex; justify-content: center; }

/* Default body alignment (NO !important) so per-column rules can win */
div[data-testid="stDataFrame"] tbody td { text-align: left; }

/* 2) Right-align numeric columns.
   st.dataframe wraps each cell in a flex container; we right-align by pushing
   content to the end of that flex container.
   Column order (no index): 1 Name | 2 Token Address | 3 Dex | 4 Safety | 5 Price | 6 Liquidity | 7 Market Cap | 8 Volume (24h) | 9 1H | 10 6H | 11 24H
*/
div[data-testid="stDataFrame"] tbody tr td:nth-child(5)  > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(6)  > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(7)  > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(8)  > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(9)  > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(10) > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(11) > div {
  justify-content: flex-end !important;
  text-align: right !important;
}

/* With an (accidental) index column, numeric columns shift by +1 (6..12) */
div[data-testid="stDataFrame"] tbody tr td:nth-child(6)  > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(7)  > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(8)  > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(9)  > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(10) > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(11) > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(12) > div {
  justify-content: flex-end !important;
  text-align: right !important;
}

/* Optional: tighten padding on numeric columns a bit */
div[data-testid="stDataFrame"] tbody tr td:nth-child(5),
div[data-testid="stDataFrame"] tbody tr td:nth-child(6),
div[data-testid="stDataFrame"] tbody tr td:nth-child(7),
div[data-testid="stDataFrame"] tbody tr td:nth-child(8),
div[data-testid="stDataFrame"] tbody tr td:nth-child(9),
div[data-testid="stDataFrame"] tbody tr td:nth-child(10),
div[data-testid="stDataFrame"] tbody tr td:nth-child(11) { padding-right: 8px; }

/* 3) Buttons: Start/Stop same size, no wrapping */
div.stButton > button {
  min-width: 120px;
  height: 36px;
  border-radius: 12px;
  white-space: nowrap;
}
div.stButton > button:disabled { opacity: 0.7; }
</style>
""", unsafe_allow_html=True)

# Prevent inner DataFrame scrollbar
def _df_height(n_rows: int, row_px: int = 34, header_px: int = 38, pad_px: int = 16, max_px: int = 520) -> int:
    return min(header_px + n_rows * row_px + pad_px, max_px)
# -------------------- end top-of-file preamble --------------------

# --- Package bootstrap so 'solana_trading_bot_bundle' is importable everywhere ---
# Place this BEFORE any 'from solana_trading_bot_bundle ...' imports.
try:
    _GUI_ROOT = Path(__file__).resolve().parent
    _PKG_ROOT = _GUI_ROOT                     # repo root (folder that contains solana_trading_bot_bundle)
    _PKG_DIR  = _PKG_ROOT / "solana_trading_bot_bundle"

    # Make repo root importable in THIS process…
    if str(_PKG_ROOT) not in sys.path:
        sys.path.insert(0, str(_PKG_ROOT))

    # …and in ALL child processes (bot subprocess, etc.)
    _existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(_PKG_ROOT)] + ([p for p in _existing.split(os.pathsep) if p] if _existing else [])
    )

    # Ensure subpackages are real packages for the bot’s -m import
    for sub in ("", "utils", "trading_bot"):
        ip = _PKG_DIR / sub / "__init__.py"
        ip.parent.mkdir(parents=True, exist_ok=True)
        if not ip.exists():
            ip.write_text("# package marker\n", encoding="utf-8")
except Exception as _e:
    # logger isn't initialized yet; use print to avoid NameError
    print(f"[package bootstrap] skipped: { _e }", flush=True)

# Import shared paths + optional status seeder (works whether you run from package or flat folder)
try:
    from solana_trading_bot_bundle.trading_bot.fetching import (
        STATUS_FILE,
        FAILURES_FILE,
        ensure_rugcheck_status_file,
    )
except Exception:
    try:
        # Fallback import when running a flat checkout
        from trading_bot.fetching import (
            STATUS_FILE,
            FAILURES_FILE,
            ensure_rugcheck_status_file,
        )
    except Exception:
        # Last-ditch defaults so the GUI still starts (no-ops if real module isn’t importable)
        from pathlib import Path as _Path
        import json as _json, time as _time

        if os.name == "nt":
            _base = _Path(os.getenv("LOCALAPPDATA", _Path.home() / "AppData/Local"))
        else:
            _base = _Path(os.getenv("XDG_DATA_HOME", _Path.home() / ".local/share"))
        _app = _base / "SOLOTradingBot"
        _app.mkdir(parents=True, exist_ok=True)

        STATUS_FILE   = _app / "rugcheck_status.json"
        FAILURES_FILE = _app / "rugcheck_failures.json"

        def ensure_rugcheck_status_file() -> None:
            """Seed a neutral status file if missing (GUI banner depends on it)."""
            try:
                if not STATUS_FILE.exists():
                    STATUS_FILE.write_text(
                        _json.dumps(
                            {"enabled": False, "available": False,
                             "message": "Rugcheck status unknown",
                             "timestamp": int(_time.time())},
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
            except Exception:
                pass


# --- One-time live refresh bootstrap for the Discovered Tokens tab (deduplicated) ---
def _bootstrap_discovery_once() -> None:
    """
    Ensure the Discovered Tokens tab shows a fresh scan on first visit.
    Call this at the very top of the 'Discovered Tokens' tab block.
    """
    ss = st.session_state
    if not ss.get("discovery_bootstrapped"):
        ss["discovery_bootstrapped"] = True
        # Clear any cached DB fetch so the next call performs a live pull
        try:
            # This will exist by the time the tab calls us.
            fetch_tokens_from_db.clear()
        except Exception:
            pass
        # Force the auto-refresh window to be 'expired' so a pull happens now
        ss["last_token_refresh"] = 0.0

# Coerce status/failures files to Path for safe .exists()/.open() downstream
from pathlib import Path as _P
try:
    STATUS_FILE   = _P(STATUS_FILE)
    FAILURES_FILE = _P(FAILURES_FILE)
except Exception:
    pass

# Seed the Rugcheck banner once at app boot (safe no-op if already written)
try:
    ensure_rugcheck_status_file()
except Exception:
    pass

# --- Enforce non-migrating mode for .env handling (GUI should never migrate env on users) ---
os.environ.setdefault("SOLANABOT_NO_ENV_MIGRATE", "1")

# ---- Stdlib / third-party ----
import subprocess
import logging
import yaml
import time
import threading
import atexit
import asyncio
from logging.handlers import RotatingFileHandler

# ---- Third-party
import psutil
import aiosqlite
import aiohttp
from pandas import DataFrame
from dotenv import load_dotenv

# ---- Currency/time formatters for Status tab ----
# Try short aliases first (preferred), then long names, then package import.
try:
    from utils import fmt_usd, fmt_ts  # preferred short aliases
except Exception:
    try:
        # fallback to long names (older utils.py)
        from utils import format_usd as fmt_usd
        from utils import format_ts_human as fmt_ts
    except Exception:
        try:
            # final fallback if running as a package
            from solana_trading_bot_bundle.trading_bot.utils import (
                format_usd as fmt_usd,
                format_ts_human as fmt_ts,
            )
        except Exception:
            # last-resort stubs so the GUI still renders
            def fmt_usd(v, *, compact=False, decimals=2):
                try:
                    n = float(v)
                except Exception:
                    return "$0.00"
                return f"${n:,.{decimals}f}"

            def fmt_ts(v, *, with_time=True):
                from datetime import datetime
                try:
                    x = float(v)
                    if 0 < x < 10_000_000_000:
                        dt = datetime.fromtimestamp(x)
                    else:
                        dt = datetime.fromtimestamp(x / 1000.0)
                    return dt.strftime("%Y-%m-%d %H:%M:%S") if with_time else dt.strftime("%Y-%m-%d")
                except Exception:
                    return "-"

# --- Rugcheck import shim (unchanged behavior) ---
import types as _types
def _install_rugcheck_shim():
    # Try utils.rugcheck_auth
    _fn = None
    try:
        import rugcheck_auth as _rcmod  # utils on sys.path above
        _fn = getattr(_rcmod, "get_rugcheck_token", None)
    except Exception:
        try:
            from utils import rugcheck_auth as _rcmod  # if package-style
            _fn = getattr(_rcmod, "get_rugcheck_token", None)
        except Exception:
            _fn = None
    mname = "solana_trading_bot_bundle.rugcheck_auth"
    if mname not in sys.modules:
        _shim = _types.ModuleType(mname)
        if callable(_fn):
            _shim.get_rugcheck_token = _fn
        else:
            async def get_rugcheck_token():
                # Fallback: read JWT/API key from env; empty string is OK (public endpoints)
                return os.getenv("RUGCHECK_JWT", "") or os.getenv("RUGCHECK_API_TOKEN", "")
            _shim.get_rugcheck_token = get_rugcheck_token  # async fallback
        sys.modules[mname] = _shim

_install_rugcheck_shim()

# ---- Local package
from solana_trading_bot_bundle.common.constants import (
    APP_NAME, appdata_dir, config_path, env_path, token_cache_path, prefer_appdata_file,
    display_cap,  
)


from solana_trading_bot_bundle.trading_bot.trading import main as trading_main

# ---- In-process trading globals (place them here)
import threading, asyncio

# ---- Resolve paths (force GUI to use the SAME AppData as the launcher) ----
import sys as _sys, os as _os, shutil as _shutil
from pathlib import Path as _Path

_APPNAME = "SOLOTradingBot"

if _os.name == "nt":
    _base = _os.getenv("LOCALAPPDATA") or _os.getenv("APPDATA") or str(_Path.home() / "AppData" / "Local")
    APP_DIR: _Path = _Path(_base) / _APPNAME
elif _sys.platform == "darwin":
    APP_DIR: _Path = _Path.home() / "Library" / "Application Support" / _APPNAME
else:
    APP_DIR: _Path = _Path.home() / ".local" / "share" / _APPNAME

APP_DIR.mkdir(parents=True, exist_ok=True)

# Primary (canonical) paths
CONFIG_PATH: _Path    = APP_DIR / "config.yaml"
ENV_PATH: _Path       = APP_DIR / ".env"
DB_PATH: _Path        = APP_DIR / "tokens.sqlite3"   
LOG_PATH: _Path       = APP_DIR / "logs" / "bot.log"
STOP_FLAG_PATH: _Path = APP_DIR / "bot_stop_flag.txt"
# Persisted process metadata
PID_FILE: _Path     = APP_DIR / "bot.pid"
SUBPROC_LOG: _Path  = APP_DIR / "bot_subprocess.log"

try:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)                    
        os.environ["SOLBOT_ENV"] = str(ENV_PATH) 
except Exception as _e:
    print(f"[env] failed to load {ENV_PATH}: {_e}", flush=True)

# --- Ensure config.yaml exists in the canonical APP_DIR; no XDG/mac fallback ---
try:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        # Seed from best available local candidates (project / bundled)
        _candidates = [
            _Path(__file__).with_name("config.yaml"),
            _Path.cwd() / "config.yaml",
            _Path(getattr(_sys, "_MEIPASS", _Path(__file__).parent)) / "config.yaml",
        ]
        for _src in _candidates:
            try:
                if _src.exists():
                    _shutil.copy2(_src, CONFIG_PATH)
                    break
            except Exception:
                pass
except Exception:
    # non-fatal — downstream loader will surface a clear error if still missing
    pass


# (dirs for logs/stop flag created later)

# =============================
# Bot spawn/stop helpers (idempotent)  — with dynamic launcher resolution
# =============================

if "start_bot" not in globals() or "_bot_running" not in globals():

    # --- Persisted process metadata (fallback if not defined earlier; no bare expressions) ---
    if "PID_FILE" not in globals():
        PID_FILE: _Path = APP_DIR / "bot.pid"
    if "SUBPROC_LOG" not in globals():
        SUBPROC_LOG: _Path = APP_DIR / "bot_subprocess.log"
            

    # =============================
    # Bot spawn/stop helpers (PID-aware)
    # =============================

    def _build_pythonpath_for_spawn() -> str:
        """Compose a PYTHONPATH that lets the child import the bundle as a package."""
        parts: list[str] = []
        parts.extend([str(_PKG_ROOT)])  # repo/package root
        existing = os.environ.get("PYTHONPATH", "")
        if existing:
            parts.extend([p for p in existing.split(os.pathsep) if p])
        seen, out = set(), []
        for p in parts:
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return os.pathsep.join(out)

    # ---- small pidfile utilities ----
    def _read_pidfile() -> int | None:
        try:
            pid = int(PID_FILE.read_text(encoding="utf-8").strip())
            return pid if pid > 0 else None
        except Exception:
            return None

    def _write_pidfile(pid: int) -> None:
        try:
            PID_FILE.parent.mkdir(parents=True, exist_ok=True)
            PID_FILE.write_text(f"{int(pid)}\n", encoding="utf-8")
        except Exception:
            pass

    def _clean_pidfile() -> None:
        try:
            PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass

    def _is_alive(pid: int) -> bool:
        try:
            p = psutil.Process(int(pid))
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except Exception:
            return False

    def _bot_running() -> bool:
        """Return True if a bot process is alive; restores PID from file across reruns."""
        pid = st.session_state.get("bot_pid")
        if pid and _is_alive(pid):
            return True
        st.session_state.pop("bot_pid", None)  # drop bad session PID

        pid = _read_pidfile()
        if pid and _is_alive(pid):
            st.session_state["bot_pid"] = int(pid)
            return True

        _clean_pidfile()  # stale pidfile
        return False

    # ---- Resolve a runnable bot command (module-first, then script) ----
    def _resolve_bot_cmd() -> tuple[list[str], str]:
        """
        Return (cmd_list, human_label). Prefers `python -m solana_trading_bot`
        (your working top-level module), then a package-level __main__ at
        `solana_trading_bot_bundle.trading_bot`, and finally a direct script path.
        """
        import importlib.util as iu

        # 1) Top-level module: solana_trading_bot.py in repo root
        if iu.find_spec("solana_trading_bot"):
            return ([sys.executable, "-m", "solana_trading_bot"], "module: solana_trading_bot")

        # 2) Package __main__: solana_trading_bot_bundle/trading_bot/__main__.py (optional)
        if iu.find_spec("solana_trading_bot_bundle.trading_bot"):
            # Only works if that package has a __main__.py; harmless if not chosen.
            return ([sys.executable, "-m", "solana_trading_bot_bundle.trading_bot"],
                    "module: solana_trading_bot_bundle.trading_bot")

        # 3) Direct script path fallbacks
        for candidate in (
            _PKG_ROOT / "solana_trading_bot.py",
            Path(__file__).with_name("solana_trading_bot.py"),
        ):
            try:
                if candidate.exists():
                    return ([sys.executable, str(candidate)], f"script: {candidate}")
            except Exception:
                pass

        # Nothing found; keep the old (broken) target so we can surface a clear error in the log
        return ([sys.executable, "-m", "solana_trading_bot_bundle.trading_bot.solana_trading_bot"],
                "module: solana_trading_bot_bundle.trading_bot.solana_trading_bot (legacy/missing)")

    def start_bot() -> None:
        """Spawn the trading bot as a subprocess and persist its PID (session + pidfile)."""
        if _bot_running():
            return  # already running

        # Respect stop flag: if user requested stop, don't start
        try:
            if STOP_FLAG_PATH.exists():
                return
        except Exception:
            pass

        # clear any previous error since we're attempting a fresh start
        st.session_state.pop("_bot_last_error", None)

        env = os.environ.copy()
        try:
            env["PYTHONPATH"] = _build_pythonpath_for_spawn()
        except Exception:
            pass

        # Decide how to launch the bot (no hard-coded legacy -m target)
        cmd, label = _resolve_bot_cmd()

        # Build environment for the child so it inherits our loaded .env and PYTHONPATH
        env = dict(os.environ)
        # Ensure helpers know where the .env lives (belt-and-suspenders)
        try:
            if "SOLBOT_ENV" not in env and 'ENV_PATH' in globals() and ENV_PATH:
                env["SOLBOT_ENV"] = str(ENV_PATH)
        except Exception:
            pass
        # Always pass PYTHONPATH we constructed so the child can import the bundle
        env["PYTHONPATH"] = _build_pythonpath_for_spawn()

        # Decide how to launch the bot (no hard-coded legacy -m target)
        cmd, label = _resolve_bot_cmd()
        
        SUBPROC_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(SUBPROC_LOG, "a", encoding="utf-8") as f:
            f.write("\n" + ("=" * 78) + "\n")
            import time as _t, shlex as _shlex
            f.write(_t.strftime("[%Y-%m-%d %H:%M:%S] ") + "Starting bot…\n")
            f.write(f"launcher={label}\n")
            f.write(f"cwd={os.getcwd()}\n")
            try:
                f.write(f"cmd={_shlex.join(cmd)}\n")
            except Exception:
                f.write(f"cmd={cmd}\n")
            f.flush()
            try:
                proc = subprocess.Popen(
                    cmd, env=env, stdout=f, stderr=subprocess.STDOUT, text=True,
                )
                # >>> Make the first click “stick”
                st.session_state["bot_pid"] = int(proc.pid)
                _write_pidfile(proc.pid)
                # Clear any autostart locks/flags to avoid fights
                for k in ("_autostart_done","_autostart_pending","_autostart_error","_autostart_last_try_ts"):
                    st.session_state.pop(k, None)
                try:
                    _release_autostart_lock()
                except Exception:
                    pass
                f.write(f"spawned pid={proc.pid}\n"); f.flush()
            except Exception as e:
                f.write(f"Spawn failed: {e}\n")
                st.session_state["_bot_last_error"] = f"Spawn failed: {e}"
                return

                pid = int(proc.pid)
                st.session_state["bot_pid"] = pid
                _write_pidfile(pid)

                # Detect instant crash and surface the tail to the UI
                import time as _t
                _t.sleep(0.9)
                if not _is_alive(pid):
                    try:
                        tail = "".join(
                            SUBPROC_LOG.read_text(encoding="utf-8", errors="ignore").splitlines(True)[-120:]
                        )
                        st.session_state["_bot_last_error"] = tail
                    except Exception:
                        st.session_state["_bot_last_error"] = "(failed to read bot_subprocess.log)"
                    _clean_pidfile()
                    st.session_state.pop("bot_pid", None)

def stop_bot() -> None:
    """Signal the bot to stop, wait briefly for exit, then force-kill if needed. Cleans all state."""
    ss = st.session_state

    # 1) Ask the bot to exit gracefully (your loop should watch this file)
    try:
        STOP_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        STOP_FLAG_PATH.write_text("stop\n", encoding="utf-8")
    except Exception:
        pass

    # 2) Find the PID (prefer session, fall back to pidfile)
    pid = ss.get("bot_pid") or _read_pidfile()

    # 3) Try to terminate the process cleanly; kill if it won't exit
    if pid:
        try:
            import psutil, time
            proc = psutil.Process(int(pid))
            if proc.is_running():
                try:
                    proc.terminate()             # gentle
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)         # give it up to ~5s to honor STOP flag
                except psutil.TimeoutExpired:
                    try:
                        proc.kill()              # force if still alive
                    except Exception:
                        pass
        except psutil.NoSuchProcess:
            pass
        except Exception:
            pass

    # 4) Clean up files (so UI can't misread stale state)
    try:
        PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        HEARTBEAT_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        STARTED_AT_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        # optional: remove STOP flag now that we've acted on it
        STOP_FLAG_PATH.unlink(missing_ok=True)
    except Exception:
        pass

    # 5) Clear session flags so Start is immediately available and autostart can re-run if enabled
    for k in ("bot_pid", "_autostart_done", "_autostart_pending", "_autostart_error", "_autostart_last_try_ts"):
        ss.pop(k, None)

    # 6) Release multi-tab autostart lock (must be defined earlier in the file)
    try:
        _release_autostart_lock()
    except Exception:
        pass

# --- Auto-start guard (call once per rerun; respects STOP_FLAG_PATH; double-spawn safe) ---
AUTO_START_BOT = os.getenv("AUTO_START_BOT", "0").strip().lower() in ("1", "true", "yes", "y")
_AUTOSTART_BACKOFF_SEC = 2.0

def _maybe_autostart_bot() -> None:
    """
    Safe autostart: respects STOP_FLAG_PATH, runs once per session,
    and confirms the bot is actually running before marking done.
    Includes a tiny backoff and a multi-tab lock.
    """
    ss = st.session_state
    now = time.time()

    # If autostart disabled, clear pending/error and bail
    if not AUTO_START_BOT:
        ss.pop("_autostart_pending", None)
        ss.pop("_autostart_error", None)
        return

    # Backoff throttle to avoid thrash across rapid reruns
    last_try = ss.get("_autostart_last_try_ts")
    if last_try and (now - last_try) < _AUTOSTART_BACKOFF_SEC:
        return

    # One-shot per session
    if ss.get("_autostart_done", False):
        return

    # Respect STOP flag
    try:
        if STOP_FLAG_PATH.exists():
            ss["_autostart_pending"] = False
            return
    except Exception:
        pass

    # Ensure helpers exist
    sb = globals().get("start_bot")
    br = globals().get("_bot_running")
    if not callable(sb) or not callable(br):
        ss["_autostart_pending"] = True
        return

    # If already running, mark done and release any lock
    if br():
        ss["_autostart_done"] = True
        ss["_autostart_pending"] = False
        ss.pop("_autostart_error", None)
        _release_autostart_lock()
        return

    # Acquire multi-tab lock; if another tab is handling autostart, skip
    if not _has_autostart_lock():
        return

    # Try starting
    ss["_autostart_last_try_ts"] = now
    ss["_autostart_pending"] = True
    try:
        sb()
        # Let PID/heartbeat land
        time.sleep(0.6)
        if br():
            ss["_autostart_done"] = True
            ss["_autostart_pending"] = False
            ss.pop("_autostart_error", None)
            _release_autostart_lock()
        else:
            # leave done=False; next rerun will retry after backoff
            ss["_autostart_done"] = False
            # keep the lock so another tab doesn't race; it will be released on success/stop
    except Exception as e:
        ss["_autostart_error"] = str(e)
        ss["_autostart_pending"] = False   # allow manual Start
        _release_autostart_lock()

# ... define start_bot / _bot_running first 
_maybe_autostart_bot()

# =============================
# Config Helpers
# =============================

@st.cache_data(show_spinner=False)
def load_config(path: Path | None = None) -> dict:
    """Load YAML config with caching. If path is None, uses current CONFIG_PATH."""
    cfg_path = Path(path) if path is not None else Path(CONFIG_PATH)
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg
    except Exception as e:
        logging.getLogger("SolanaMemeBotGUI").error("Failed to load config: %s", e)
        try:
            st.error(f"❌ Failed to load configuration: {e}")
        except Exception:
            pass
        return {}  # allow app to continue

def save_config(config: dict, path: Path = CONFIG_PATH) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        return True
    except Exception as e:
        logging.getLogger("SolanaMemeBotGUI").error("Failed to save config: %s", e)
        try:
            st.error(f"❌ Failed to save configuration: {e}")
        except Exception:
            pass
        return False


def _resolve_db_and_logs_from_config(cfg: dict) -> None:
    r"""
    Resolve DB/log paths from config, expanding ~ and %VARS%, with safe fallbacks.
    Also probes common locations so GUI and bot stay in sync even if filenames differ.

    Extra hardening:
      - On POSIX, if a Windows-style path is provided, ignore it and fall back to AppData defaults.
      - Never interpolate a function object into a path.
    """
    global DB_PATH, LOG_PATH
    lg = logging.getLogger("SolanaMemeBotGUI")

    def _looks_windows_path(s: str) -> bool:
        # Simple heuristic: contains backslashes *and* a drive-colon
        return ("\\" in s) and (":" in s)

    # --------- Determine the database default via the database module ----------
    # This avoids ever touching token_cache_path directly.
    default_db: Path
    try:
        try:
            from solana_trading_bot_bundle.trading_bot import database as _dbmod  # type: ignore
        except Exception:
            import trading_bot.database as _dbmod  # type: ignore

        if hasattr(_dbmod, "_resolve_db_path"):
            default_db = Path(_dbmod._resolve_db_path())  # uses same logic as connect_db()
        else:
            default_db = APP_DIR / "tokens.sqlite3"
    except Exception:
        default_db = APP_DIR / "tokens.sqlite3"

    default_log = APP_DIR / "logs" / "bot.log"

    # ---------- Database path ----------
    try:
        db_section = cfg.get("database", {}) if isinstance(cfg, dict) else {}
        cand = db_section.get("token_cache_path") or db_section.get("path")

        if cand:
            # Expand env vars and ~
            cand = os.path.expandvars(os.path.expanduser(str(cand)))

            # If we're on POSIX but got a Windows-looking path, ignore it
            if os.name != "nt" and _looks_windows_path(cand):
                lg.warning(
                    "Configured token_cache_path looks Windows-style on this OS (%s). "
                    "Ignoring and using default AppData path instead.", cand
                )
                db_path = Path(default_db)
            else:
                db_path = Path(cand)
                # If a directory (or suffixless path) was provided, pick a sensible file name
                if db_path.suffix == "" and (str(db_path).endswith(("\\", "/")) or not db_path.name.count(".")):
                    db_path = db_path / "tokens.db"
                db_path = db_path.resolve()
        else:
            # No explicit path in config — probe common locations
            candidates = [
                Path(default_db),
                APP_DIR / "tokens.sqlite3",
                APP_DIR / "tokens.db",
            ]
            la = os.getenv("LOCALAPPDATA", "")
            if la:
                candidates += [
                    Path(la) / "SOLOTradingBot" / "tokens.db",
                    Path(la) / "SOLOTradingBot" / "tokens.sqlite3",
                ]

            existing = [p for p in candidates if p and p.exists()]
            db_path = existing[0] if existing else candidates[0]

        db_path.parent.mkdir(parents=True, exist_ok=True)
        DB_PATH = db_path
        lg.info("Using token DB at %s", DB_PATH)
    except Exception as e:
        lg.warning("Could not resolve DB path from config, falling back: %s", e)
        DB_PATH = Path(default_db)
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
  
    # ---------- Log file path ----------
    try:
        log_section = cfg.get("logging", {}) if isinstance(cfg, dict) else {}
        lf = log_section.get("log_file_path")

        if isinstance(lf, str) and lf.strip():
            lf = os.path.expandvars(os.path.expanduser(lf))
            if os.name != "nt" and _looks_windows_path(lf):
                lg.warning(
                    "Configured log_file_path looks Windows-style on this OS (%s). "
                    "Ignoring and using default AppData log path instead.", lf
                )
                log_path = default_log
            else:
                log_path = Path(lf)
                # If a directory (or suffixless path) was provided, use bot.log inside it
                if log_path.suffix == "" and (str(log_path).endswith(("\\", "/")) or not log_path.name.count(".")):
                    log_path = log_path / "bot.log"
                log_path = log_path.resolve()
        else:
            candidates = [default_log]
            la = os.getenv("LOCALAPPDATA", "")
            if la:
                candidates.insert(0, Path(la) / "SOLOTradingBot" / "logs" / "bot.log")
            existing_dir = [p for p in candidates if p.parent.exists()]
            log_path = existing_dir[0] if existing_dir else candidates[0]

        log_path.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH = log_path
        lg.info("Using log file at %s", LOG_PATH)
    except Exception as e:
        lg.warning("Could not resolve LOG path from config; using default: %s", e)
        LOG_PATH = default_log
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    

# ---- Load environment once (early)  (REPLACE this whole block) ----
import os
from pathlib import Path

try:
    from dotenv import load_dotenv, find_dotenv
except Exception:
    load_dotenv = None
    find_dotenv = None

if load_dotenv:
    try:
        # 1) Canonical AppData .env (preferred)
        if 'ENV_PATH' in globals() and ENV_PATH:
            load_dotenv(dotenv_path=str(ENV_PATH), override=False)

        # 2) Also read a local project .env if present (won’t override step #1)
        proj_env = Path(__file__).with_name(".env")
        if proj_env.exists() and (not 'ENV_PATH' in globals() or proj_env != ENV_PATH):
            load_dotenv(dotenv_path=str(proj_env), override=False)

        # 3) As a last resort, let python-dotenv auto-find in CWD
        auto_env = find_dotenv(usecwd=True) if find_dotenv else ""
        if auto_env:
            auto_env_path = Path(auto_env)
            if auto_env_path != proj_env and (not 'ENV_PATH' in globals() or auto_env_path != ENV_PATH):
                load_dotenv(dotenv_path=auto_env, override=False)
    except Exception:
        pass

# ---- Bootstrap config-driven paths before logger init ----
try:
    cfg_boot = load_config()  # cached; respects current CONFIG_PATH
except Exception:
    cfg_boot = {}

if isinstance(cfg_boot, dict):
    _resolve_db_and_logs_from_config(cfg_boot)

# Ensure directories now that LOG_PATH may have changed
try:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
try:
    STOP_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# --- Bot metadata files (GUI-side), ensure these exist as Path objects ---
HEARTBEAT_FILE: Path   = APP_DIR / "heartbeat"
STARTED_AT_FILE: Path  = APP_DIR / "started_at.txt"
AUTOSTART_LOCK_FILE: Path = APP_DIR / "autostart.lock"

# --- Multi-tab autostart lock (prevents two browser tabs from starting the bot) ---
def _has_autostart_lock() -> bool:
    try:
        # Try to create exclusively; if it exists, another tab/session holds it
        fd = os.open(str(AUTOSTART_LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        st.session_state["_autostart_lock_owned"] = True
        return True
    except FileExistsError:
        return False
    except Exception:
        return False

def _release_autostart_lock() -> None:
    if st.session_state.get("_autostart_lock_owned"):
        try:
            os.remove(str(AUTOSTART_LOCK_FILE))
        except Exception:
            pass
        st.session_state.pop("_autostart_lock_owned", None)

# ---- Logging (configure a dedicated app logger once)
logger = logging.getLogger("SolanaMemeBotGUI")
logger.setLevel(logging.INFO)

if not logger.handlers:
    _fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(str(LOG_PATH), maxBytes=2_000_000, backupCount=3,
                                 encoding="utf-8", delay=True)
        fh.setFormatter(_fmt)
        logger.addHandler(fh)
    except Exception as e:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        logger.warning("Failed to attach RotatingFileHandler at %s: %s", LOG_PATH, e)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(_fmt)
    logger.addHandler(sh)

# Prevent duplicate lines via the root logger
logger.propagate = False
logger.info("Logging initialized at %s", LOG_PATH)

def _rebind_logger_file_handler() -> None:
    """Point the existing logger's RotatingFileHandler at the (possibly updated) LOG_PATH."""
    for h in list(logger.handlers):
        if isinstance(h, RotatingFileHandler):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(str(LOG_PATH), maxBytes=2_000_000, backupCount=3,
                                 encoding="utf-8", delay=True)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.propagate = False
        logger.info("Rebound file handler to %s", LOG_PATH)
    except Exception as e:
        logger.warning("Failed to rebind file handler to %s: %s", LOG_PATH, e)


# ---- App/env flags
BOT_ENTRY_MODULE = "solana_trading_bot_bundle.trading_bot.solana_trading_bot"
DEV_MODE = str(os.getenv("GUI_DEV_MODE", "0")).lower() in ("1", "true", "yes")
BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")

# Load settings from the already-cached config (avoid re-reading file)
_cfg = cfg_boot if isinstance(cfg_boot, dict) else {}

SAFETY_CFG = _cfg.get("safety", {})
NEW_TOKEN_BADGE_WINDOW_MIN = int(SAFETY_CFG.get("new_badge_minutes", 180))
NEW_TOKEN_SAFETY_WINDOW_MIN = int(SAFETY_CFG.get("new_safety_minutes", 1440))
SAFE_MIN_LIQUIDITY_USD = float(SAFETY_CFG.get("safe_min_liquidity_usd", 5000.0))
STRONG_SCORE = float(SAFETY_CFG.get("strong_score", 85.0))

# --- UI/Discovery knobs (shared with bot) ---
DISCOVERY_CFG = _cfg.get("discovery", {}) if isinstance(_cfg, dict) else {}
TOP_N_PER_CATEGORY = int(
    os.getenv("TOP_N_PER_CATEGORY", DISCOVERY_CFG.get("shortlist_per_bucket", 5))
)

# Fallback Prices (unchanged)
FALLBACK_PRICES = {
    "So11111111111111111111111111111111111111112": {
        "price": 175.81,
        "price_change_1h": -0.77,
        "price_change_6h": -0.70,
        "price_change_24h": -3.12,
        "volume_24h": 25000.00,
    }
}

# === DB SCHEMA BOOTSTRAP (run once per Streamlit session) ====================
try:
    # Keep this import EXACTLY on its own line
    try:
        from solana_trading_bot_bundle.trading_bot.database import (
            connect_db, init_db
        )
    except Exception:
        from trading_bot.database import (
            connect_db, init_db  # flat layout fallback
        )

    async def _maybe_await(x):
        import inspect
        return (await x) if inspect.isawaitable(x) else x

    async def _call_init_db_maybe_with_conn(conn):
        """
        init_db may be:
          - async def init_db(conn)
          - def init_db(conn)
          - async def init_db()
          - def init_db()
        Call it safely regardless of signature.
        """
        import inspect
        try:
            sig = inspect.signature(init_db)
        except Exception:
            sig = None

        if sig:
            params = [
                p for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
            ]
            if len(params) >= 1:
                return await _maybe_await(init_db(conn))  # expects a connection
            else:
                return await _maybe_await(init_db())      # expects no args
        # If introspection failed, try with conn first, then fallback to no-arg
        try:
            return await _maybe_await(init_db(conn))
        except TypeError:
            return await _maybe_await(init_db())

    async def _bootstrap_db_once() -> None:
        """
        Be tolerant: connect_db() may return (a) an awaitable connection or (b) an async context manager.
        """
        import inspect
        obj = connect_db()
        if hasattr(obj, "__aenter__") and hasattr(obj, "__aexit__"):
            async with obj as db:
                await _call_init_db_maybe_with_conn(db)
            return

        conn = await obj if inspect.isawaitable(obj) else obj
        try:
            await _call_init_db_maybe_with_conn(conn)
        finally:
            close = getattr(conn, "close", None)
            if callable(close):
                try:
                    import inspect as _ins
                    if _ins.iscoroutinefunction(close):
                        await close()
                    else:
                        close()
                except Exception:
                    pass

    @st.cache_resource(show_spinner=False)
    def _ensure_db_schema_ready() -> bool:
        asyncio.run(_bootstrap_db_once())
        return True

    _ = _ensure_db_schema_ready()
except Exception as _db_e:
    logger.warning("DB bootstrap failed: %s", _db_e)

# =============================
# Asyncio Event Loop Management
# =============================
# --- NO Streamlit decorator here ---
_LOOP = None
_LOOP_THREAD = None
_LOOP_LOCK = threading.Lock()

def _shutdown_loop():
    global _LOOP, _LOOP_THREAD
    try:
        if _LOOP and _LOOP.is_running():
            _LOOP.call_soon_threadsafe(_LOOP.stop)
        if _LOOP_THREAD and _LOOP_THREAD.is_alive():
            _LOOP_THREAD.join(timeout=1.5)
    except Exception:
        pass
    try:
        if _LOOP and not _LOOP.is_closed():
            _LOOP.close()
    except Exception:
        pass
    _LOOP = None
    _LOOP_THREAD = None

atexit.register(_shutdown_loop)

def get_event_loop():
    """Create (once) a dedicated asyncio event loop in a background thread (thread-safe, no Streamlit cache)."""
    global _LOOP, _LOOP_THREAD
    with _LOOP_LOCK:
        if _LOOP and _LOOP.is_running() and not _LOOP.is_closed():
            return _LOOP
        loop = asyncio.new_event_loop()
        def _run():
            try:
                asyncio.set_event_loop(loop)
                loop.run_forever()
            except Exception:
                pass
        t = threading.Thread(target=_run, daemon=True, name="gui-asyncio-loop")
        t.start()
        _LOOP, _LOOP_THREAD = loop, t
        return loop


def run_async_task(task, timeout: float | None = None):
    """
    Runs an async task safely from Streamlit (which does not run an event loop).

    Accepts:
      - a coroutine OBJECT,
      - a coroutine FUNCTION (async def),
      - a plain callable (lambda/def) which may return either a value or a coroutine.

    Always awaits the underlying coroutine on the dedicated background loop
    returned by get_event_loop(), preventing 'coroutine was never awaited' issues.
    """
    import inspect
    import concurrent.futures

    # 1) Normalize to a coroutine object (coro)
    if inspect.iscoroutine(task):
        coro = task
    elif inspect.iscoroutinefunction(task):
        coro = task()  # call to get the coroutine object
    elif callable(task):
        # Call the callable. It might return a value OR a coroutine.
        result = task()
        if inspect.iscoroutine(result):
            coro = result
        else:
            async def _wrap(val=result):
                return val
            coro = _wrap()
    else:  # bad input
        raise TypeError(f"run_async_task expected coroutine/corofunc/callable, got {type(task)!r}")

    # 2) Always schedule on our dedicated loop
    loop = get_event_loop()
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return fut.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        try:
            fut.cancel()
        except Exception:
            pass
        raise
    except Exception:
        raise

# -- minimal dependencies for live-first path --
import asyncio as _lf_asyncio
from typing import List as _lf_List, Dict as _lf_Dict, Any as _lf_Any, Optional as _lf_Optional

try:
    import aiohttp as _lf_aiohttp
except Exception:
    _lf_aiohttp = None

# Use the already-installed rugcheck shim.
try:
    from solana_trading_bot_bundle.rugcheck_auth import get_rugcheck_token as _lf_get_rugcheck_token
except Exception:
    async def _lf_get_rugcheck_token():
        import os as _os
        return _os.getenv("RUGCHECK_JWT", "") or _os.getenv("RUGCHECK_API_TOKEN", "")

# Stable/WSOL hide set (fallback if not defined)
try:
    _LF_HIDE = set(ALWAYS_HIDE_IN_CATEGORIES)
except Exception:
    _LF_HIDE = {
        "So11111111111111111111111111111111111111112",  # WSOL
        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
        "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
        "jupSoLaHXQiZZTSfEWMTRRgpnyFm8f6sZdosWBjx93v",  # JITOSOL
    }

# --- Also hide by symbol/name to catch non-canonical clones (WSOL/USDC/etc.) ---
# You can extend/override via ENV: ALWAYS_HIDE_SYMBOLS="SOL,WSOL,USDC,USDT,Wrapped Solana"
def _csv_env(name: str, default: str) -> set[str]:
    raw = os.getenv(name, default)
    return {s.strip().upper() for s in raw.split(",") if s.strip()}

ALWAYS_HIDE_SYMBOLS = _csv_env(
    "ALWAYS_HIDE_SYMBOLS",
    "SOL,WSOL,USDC,USDT,WRAPPED SOLANA,WRAPPED USD COIN,USD COIN, JITOSOL"
)

def _hidden_token(t: dict) -> bool:
    addr = (t.get("address") or t.get("token_address") or "").strip()
    if addr and addr in _LF_HIDE:
        return True
    sym_or_name = (t.get("symbol") or t.get("name") or "").strip().upper()
    return bool(sym_or_name and sym_or_name in ALWAYS_HIDE_SYMBOLS)


# Config fallbacks
try:
    _LF_NEW_BADGE_MIN = int(NEW_TOKEN_BADGE_WINDOW_MIN)
except Exception:
    _LF_NEW_BADGE_MIN = 180

try:
    _LF_BIRDEYE_API_KEY = BIRDEYE_API_KEY
except Exception:
    import os as _os
    _LF_BIRDEYE_API_KEY = _os.getenv("BIRDEYE_API_KEY", "")

async def _lf_best_solana_pair(pairs: _lf_List[_lf_Dict[str, _lf_Any]], token: str) -> _lf_Optional[str]:
    best, best_liq = None, -1.0
    for p in pairs or []:
        try:
            if p.get("chainId") != "solana":
                continue
            base = (p.get("baseToken") or {}).get("address")
            quote = (p.get("quoteToken") or {}).get("address")
            if token not in {base, quote}:
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
    return best and best.get("pairAddress")

async def _lf_fetch_price_for_token(addr: str, session) -> _lf_Dict[str, _lf_Any]:
    # 1) Dexscreener tokens endpoint
    try:
        async with session.get(f"https://api.dexscreener.com/latest/dex/tokens/{addr}", timeout=8) as r:
            if r.status == 200:
                j = await r.json()
                pairs = j.get("pairs") or []
                best = await _lf_best_solana_pair(pairs, addr)
                if best:
                    chosen = next((p for p in pairs if p.get("pairAddress") == best), None)
                    if chosen:
                        pc = chosen.get("priceChange") or {}
                        vol = chosen.get("volume") or {}
                        return {
                            "price": float(chosen.get("priceUsd", 0) or 0),
                            "price_change_1h": float(pc.get("h1", 0) or 0),
                            "price_change_6h": float(pc.get("h6", 0) or 0),
                            "price_change_24h": float(pc.get("h24", 0) or 0),
                            "volume_24h": float(vol.get("h24", 0) or 0),
                        }
    except Exception:
        pass
    # 2) Birdeye fallback
    try:
        if _LF_BIRDEYE_API_KEY:
            async with session.get(
                f"https://public-api.birdeye.so/defi/price?address={addr}",
                headers={"X-API-KEY": _LF_BIRDEYE_API_KEY},
                timeout=8
            ) as r:
                if r.status == 200:
                    j = await r.json()
                    d = j.get("data") or {}
                    return {
                        "price": float(d.get("value", 0) or 0),
                        "price_change_1h": 0.0,
                        "price_change_6h": 0.0,
                        "price_change_24h": 0.0,
                        "volume_24h": float(d.get("v24hUSD", 0) or 0),
                    }
    except Exception:
        pass
    return {}

async def _lf_discover_pairs(limit: int = 200) -> _lf_List[_lf_Dict[str, _lf_Any]]:
    if not _lf_aiohttp:
        return []
    out: _lf_List[_lf_Dict[str, _lf_Any]] = []
    try:
        async with _lf_aiohttp.ClientSession() as sess:
            async with sess.get("https://api.dexscreener.com/latest/dex/pairs/solana", timeout=10) as r:
                if r.status != 200:
                    return out
                j = await r.json()
                for p in (j.get("pairs") or [])[:max(0, int(limit))]:
                    try:
                        base = p.get("baseToken") or {}
                        addr = base.get("address") or ""
                        if not addr or addr in _LF_HIDE:
                            continue
                        out.append({
                            "address": addr,
                            "name": base.get("name") or base.get("symbol") or addr[:6]+"…"+addr[-4:],
                            "symbol": base.get("symbol") or "",
                            "pairAddress": p.get("pairAddress"),
                            "liquidity": float((p.get("liquidity") or {}).get("usd", 0) or 0),
                            "market_cap": float(p.get("fdv", 0) or 0),
                            "volume_24h": float((p.get("volume") or {}).get("h24", 0) or 0),
                            "dex": p.get("dexId") or "",
                            "createdAt": int((p.get("pairCreatedAt") or 0) / 1000),
                        })
                    except Exception:
                        continue
    except Exception:
        pass
    # dedupe by address
    seen = {}
    for t in out:
        a = t.get("address")
        if a and a not in seen:
            seen[a] = t
    return list(seen.values())

async def _lf_enrich_prices(tokens: _lf_List[_lf_Dict[str, _lf_Any]]) -> _lf_List[_lf_Dict[str, _lf_Any]]:
    if not _lf_aiohttp or not tokens:
        return tokens
    async with _lf_aiohttp.ClientSession() as sess:
        tasks = [_lf_fetch_price_for_token(t.get("address", ""), session=sess) for t in tokens]
        res = await _lf_asyncio.gather(*tasks, return_exceptions=True)
    out = []
    for t, r in zip(tokens, res):
        if isinstance(r, dict):
            t.update(r)
        out.append(t)
    return out

async def _lf_enrich_rugcheck(tokens: _lf_List[_lf_Dict[str, _lf_Any]]) -> _lf_List[_lf_Dict[str, _lf_Any]]:
    if not _lf_aiohttp or not tokens:
        return tokens
    jwt = ""
    try:
        jwt = await _lf_get_rugcheck_token()
    except Exception:
        jwt = ""
    headers = {"Authorization": f"Bearer {jwt}"} if jwt else {}
    async with _lf_aiohttp.ClientSession() as sess:
        tasks = []
        for t in tokens:
            addr = t.get("address", "")
            if not addr:
                tasks.append(_lf_asyncio.sleep(0, result=None))
                continue
            url = f"https://api.rugcheck.xyz/v1/tokens/{addr}"
            async def _one(u=url):
                try:
                    async with sess.get(u, headers=headers, timeout=8) as r:
                        if r.status != 200:
                            return None
                        j = await r.json()
                        return {
                            "rugcheck_label": (j.get("risk", {}) or {}).get("label") or j.get("label") or "",
                            "rugcheck_score": float((j.get("risk", {}) or {}).get("score", 0) or j.get("score", 0) or 0),
                        }
                except Exception:
                    return None
            tasks.append(_one())
        res = await _lf_asyncio.gather(*tasks, return_exceptions=True)
    for t, rc in zip(tokens, res):
        if isinstance(rc, dict) and rc:
            t.update(rc)
        else:
            t.setdefault("rugcheck_label", "")
            t.setdefault("rugcheck_score", 0.0)
    return tokens

def _lf_mcap_bucket(x: float) -> str:
    try:
        x = float(x or 0)
    except Exception:
        x = 0.0
    if x >= 500_000_000: return "High"
    if x >= 50_000_000: return "Mid"
    return "Low"

def _flatten_shortlist_to_list(shortlist, tokens_fallback: list[dict], limit_total: int) -> list[dict]:
    """Normalize shortlist (dict or list) to a flat list while tagging bucket when available."""
    flat: list[dict] = []
    if isinstance(shortlist, dict):
        for k in ("high", "mid", "low", "new", "large", "small"):
            for t in shortlist.get(k, []) or []:
                t["_bucket"] = k
                flat.append(t)
    elif isinstance(shortlist, list):
        flat = shortlist
    else:
        flat = tokens_fallback or []
    return flat[:limit_total]

# --- LIVE-FIRST IMPL ---------------------------------------------------------

# NEW: normalize field names coming from different sources (Dexscreener/Birdeye/etc.)
def _normalize_price_change_keys(t: dict) -> None:
    """Make sure GUI sees price_change_1h / _6h / _24h even if sources use other keys."""
    def _first_existing(keys: list[str]) -> float | None:
        for k in keys:
            v = t.get(k)
            if v is None:
                continue
            try:
                return float(v)
            except Exception:
                pass
        return None

    # only set if missing (don't clobber a proper value)
    if t.get("price_change_1h") is None:
        v = _first_existing(["pc1h", "h1", "1h", "change1h", "priceChange1h"])
        if v is not None:
            t["price_change_1h"] = v

    if t.get("price_change_6h") is None:
        v = _first_existing(["pc6h", "h6", "6h", "change6h", "priceChange6h"])
        if v is not None:
            t["price_change_6h"] = v

    if t.get("price_change_24h") is None:
        v = _first_existing(["pc24h", "h24", "24h", "change24h", "priceChange24h"])
        if v is not None:
            t["price_change_24h"] = v


async def _lf_pull_live_tokens_impl(limit_total: int = 90) -> _lf_List[_lf_Dict[str, _lf_Any]]:
    base = await _lf_discover_pairs(limit=250)
    if not base:
        return []
    base = await _lf_enrich_prices(base)
    base = await _lf_enrich_rugcheck(base)

    BAD_SYMBOLS = {"USDC", "USDT", "JUP", "SOL", "MSOL", "BSOL", "JITOSOL", "STSOL"}
    BAD_NAME_SNIPPETS = ("LP", "Perps", "Staked", "Jupiter", "USDC/USDT", "USDT/USDC")
    MAX_CAP_FOR_LIVE = 50_000_000  # adjust if you want

    filtered: list[dict] = []
    for t in base:
        sym   = (t.get("symbol") or "").upper()
        name  = t.get("name") or ""
        mc    = float(t.get("market_cap") or t.get("mc") or 0)
        price = float(t.get("price") or 0)
        liq   = float(t.get("liquidity") or 0)
        vol   = float(t.get("volume_24h") or 0)

        # normalize core numeric keys
        t["market_cap"] = mc
        t["liquidity"]  = float(t.get("liquidity")  or t.get("liq")    or liq)
        t["volume_24h"] = float(t.get("volume_24h") or t.get("vol24")  or vol)

        if sym in BAD_SYMBOLS:
            continue
        if any(snip.lower() in name.lower() for snip in BAD_NAME_SNIPPETS):
            continue
        if mc <= 0 or mc > MAX_CAP_FOR_LIVE:
            continue
        if not (price > 0 or liq > 1_000 or vol > 10_000):
            continue

        # ----- attach cap bucket + categories (mirror async path) -----
        HIGH_MIN = float(MC_THRESHOLDS.get("high_min", 50_000_000))
        MID_MIN  = float(MC_THRESHOLDS.get("mid_min",      500_000))

        if mc >= HIGH_MIN:
            cap = "large_cap"   # canonical name
        elif mc >= MID_MIN:
            cap = "mid_cap"
        else:
            cap = "low_cap"

        t["_bucket"] = cap

        existing_cats = t.get("categories") or []
        if isinstance(existing_cats, str):
            existing_cats = [existing_cats]
        existing_cats = [str(c).strip().lower() for c in existing_cats if c]
        if cap not in existing_cats:
            existing_cats.append(cap)
        t["categories"] = existing_cats

        # Canonicalize category aliases if helper is present
        try:
            _normalize_categories(t)
        except Exception:
            pass
        # ----- /cap tagging -----

        # normalize change keys so GUI won’t render 0.00% when they exist under pct_*
        t["price_change_1h"]  = t.get("price_change_1h")  or t.get("pct_change_1h")
        t["price_change_6h"]  = t.get("price_change_6h")  or t.get("pct_change_6h")
        t["price_change_24h"] = t.get("price_change_24h") or t.get("pct_change_24h")

        filtered.append(t)

    out = sorted(
        filtered,
        key=lambda x: float(x.get("liquidity") or 0) * 2
                    + float(x.get("volume_24h") or 0) * 1.5
                    + float(x.get("market_cap") or 0),
        reverse=True,
    )
    return out[:limit_total]

# Thin sync wrapper so the GUI can call the async impl by name expected elsewhere.
def _pull_live_tokens(limit_total: int = 90) -> list[dict]:
    """
    Synchronous entrypoint used by the GUI. Calls the async live-first implementation.
    """
    try:
        # If you already use run_async_task elsewhere (preferred)
        return run_async_task(_lf_pull_live_tokens_impl, limit_total)
    except NameError:
        # If run_async_task isn't in scope here, fall back to a local event loop.
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(_lf_pull_live_tokens_impl(limit_total))


# =========================== END LIVE-FIRST INJECTION ===========================

def _pull_live_tokens_legacy(
    limit_total: int = 90,
    *,
    max_age_hours: int | None = 24,
) -> list[dict]:
    """
    Legacy/DB-backed path. Optionally restricts DB fallback to rows newer than `max_age_hours`.
    If `max_age_hours` is 0 or None, returns the most recent rows without an age filter.

    NOTE: This block assumes `run_async_task` accepts either:
      - a coroutine function (callable) which it will call/schedule, or
      - a coroutine object.
    """
    try:
        # Optional refresh if available (pass CALLABLES, not already-created coroutines)
        try:
            from solana_trading_bot_bundle.trading_bot.fetching import refresh_token_cache as _rtc
            _ = run_async_task(_rtc, timeout=120)  # callable, no parentheses
        except Exception:
            try:
                from solana_trading_bot_bundle.trading_bot.market_data import refresh_token_cache as _rtc
                _ = run_async_task(_rtc, timeout=120)  # callable, no parentheses
            except Exception:
                pass

        # Latest tokens via network (best effort)
        tokens = None
        try:
            from solana_trading_bot_bundle.trading_bot.fetching import get_latest_tokens as _gl
            tokens = run_async_task(_gl, timeout=120)  # callable, no parentheses
        except Exception:
            try:
                from solana_trading_bot_bundle.trading_bot.market_data import get_latest_tokens as _gl
                tokens = run_async_task(_gl, timeout=120)  # callable, no parentheses
            except Exception:
                tokens = None

        # Safe DB fallback (list interface, not single-token getter)
        if not tokens:
            try:
                from solana_trading_bot_bundle.trading_bot.database import list_eligible_tokens
                newer_than = None
                if isinstance(max_age_hours, (int, float)) and max_age_hours > 0:
                    newer_than = int(time.time()) - int(float(max_age_hours) * 3600)
                tokens = run_async_task(
                lambda: list_eligible_tokens(
                    limit=limit_total,
                    newer_than=newer_than,
                    order_by_score_desc=True,
                )
            ) or []
            except Exception:
                tokens = []

        # Normalize for legacy path too
        for _t in (tokens or []):
            _normalize_price_change_keys(_t)

        # --- Bucket & flatten ---
        try:
            from solana_trading_bot_bundle.trading_bot.eligibility import select_top_five_per_category as _sel5
            shortlist = run_async_task(lambda: _sel5(tokens or [], blacklist=set()), timeout=120)
        except Exception:
            shortlist = None

        return _flatten_shortlist_to_list(shortlist, tokens_fallback=tokens or [], limit_total=limit_total)

    except Exception as e:
        try:
            st.error(f"Live scan (legacy) failed: {e}")
        except Exception:
            pass
        return []

def _pull_tokens_with_fallback(
    limit_total: int = 90,
    *,
    legacy_hours: int | None = None,
) -> list[dict]:
    """
    Try live-first; optionally fall back to DB-backed legacy if empty.
    - If `legacy_hours` is provided, it controls the DB window.
    - Otherwise, if `st.session_state['fallback_hours']` exists, that value is used.
    - A value of 0 disables the fallback.
    """
    tokens = _pull_live_tokens(limit_total)
    if tokens:
        return tokens

    # Decide the window (caller arg wins; else UI knob; else 0 = disabled)
    hours = legacy_hours
    if hours is None:
        try:
            hours = int(st.session_state.get("fallback_hours", 0) or 0)
        except Exception:
            hours = 0

    if isinstance(hours, int) and hours > 0:
        return _pull_live_tokens_legacy(limit_total, max_age_hours=hours)

    return []

# =============================
# SQLite Helpers (hybrid)
# =============================
class _ConnectDB:
    def __init__(self, path: Path):
        self._path = path
        self._conn: aiosqlite.Connection | None = None

    def __await__(self):
        # Allow: conn = await connect_db()
        return aiosqlite.connect(str(self._path)).__await__()

    async def __aenter__(self) -> aiosqlite.Connection:
        # Ensure directory exists (avoids failures when DB dir is missing)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning("Could not create DB directory %s: %s", self._path.parent, e)

        # Small diagnostic if we're on POSIX but got a Windows-looking path
        try:
            if os.name != "nt" and ("\\" in str(self._path)) and (":" in str(self._path)):
                logger.warning(
                    "DB path looks Windows-style on POSIX: %s (will be created as a literal file name).",
                    self._path,
                )
        except Exception:
            pass

        self._conn = await aiosqlite.connect(str(self._path))
        try:
            await self._conn.execute("PRAGMA journal_mode=WAL;")
            await self._conn.execute("PRAGMA busy_timeout=30000;")
            await self._conn.execute("PRAGMA synchronous=NORMAL;")
            await self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.row_factory = aiosqlite.Row
        except Exception as e:
            logger.warning("DB PRAGMA init failed: %s", e, exc_info=True)
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        try:
            if self._conn is not None:
                await self._conn.close()
        finally:
            self._conn = None


def connect_db() -> _ConnectDB:
    return _ConnectDB(DB_PATH)


# --- Minimal core tables safety net -------------------------------------------
# Guarantees the tables/columns the GUI queries depend on always exist,
# even if init_db() didn't run for some reason.
async def _ensure_core_tables(db: aiosqlite.Connection) -> None:
    # trade_history must support the P&L query in Tab 5
    await db.execute("""
        CREATE TABLE IF NOT EXISTS trade_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_address TEXT,
            symbol TEXT,
            buy_amount REAL DEFAULT 0,
            sell_amount REAL DEFAULT 0,
            buy_price REAL DEFAULT 0,
            sell_price REAL DEFAULT 0,
            profit REAL DEFAULT 0,
            buy_time INTEGER,
            sell_time INTEGER
        )
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_trade_history_sell_time
        ON trade_history (sell_time)
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_trade_history_token_addr
        ON trade_history (token_address)
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_trade_history_symbol
        ON trade_history (symbol)
    """)

    # tokens table is read by the GUI status snapshot
    await db.execute("""
        CREATE TABLE IF NOT EXISTS tokens (
            address TEXT PRIMARY KEY,
            symbol TEXT,
            name   TEXT,
            is_trading INTEGER DEFAULT 0,
            sell_time INTEGER
        )
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_tokens_is_trading
        ON tokens (is_trading)
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_tokens_sell_time
        ON tokens (sell_time)
    """)

    await db.commit()


# =============================
# eligible_tokens schema
# =============================
async def ensure_eligible_tokens_schema():
    """
    Create/upgrade the `eligible_tokens` table. This function is async and intended to be
    executed on a background event loop. It only logs errors to avoid Streamlit context
    warnings in background threads.

    Notes / fixes based on logs:
      - Add a `data` TEXT column because the GUI reads: SELECT data FROM eligible_tokens
      - Add a `created_at` INTEGER column for lightweight insertion timestamps.
      - Keep all numeric columns the GUI and bot expect.
      - Create indices on creation_timestamp, created_at, and timestamp to speed GUI filters.
    """
    try:
        async with connect_db() as db:
            # Ensure the two core tables exist first (prevents "no such table" in GUI queries)
            await _ensure_core_tables(db)

            # Base table (idempotent)
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS eligible_tokens (
                    address TEXT PRIMARY KEY,
                    name TEXT,
                    symbol TEXT,
                    volume_24h REAL,
                    liquidity REAL,
                    market_cap REAL,
                    price REAL,
                    price_change_1h REAL,
                    price_change_6h REAL,
                    price_change_24h REAL,
                    score REAL,
                    categories TEXT,
                    timestamp INTEGER,
                    creation_timestamp INTEGER
                )
                """
            )

            # Snapshot existing columns
            cols = set()
            async with db.execute("PRAGMA table_info(eligible_tokens)") as cur:
                async for row in cur:
                    cols.add(row[1])

            async def add_col(name: str, decl: str):
                """Add a column if it does not already exist."""
                if name not in cols:
                    await db.execute(f"ALTER TABLE eligible_tokens ADD COLUMN {name} {decl}")
                    cols.add(name)

            # Ensure all expected numeric/text columns exist (backfill for older DBs)
            for name, decl in (
                ("name", "TEXT"),
                ("symbol", "TEXT"),
                ("volume_24h", "REAL"),
                ("liquidity", "REAL"),
                ("market_cap", "REAL"),
                ("price", "REAL"),
                ("price_change_1h", "REAL"),
                ("price_change_6h", "REAL"),
                ("price_change_24h", "REAL"),
                ("score", "REAL"),
                ("categories", "TEXT"),
                ("timestamp", "INTEGER"),
                ("creation_timestamp", "INTEGER"),
            ):
                await add_col(name, decl)

            # >>> NEW: columns required by GUI reads <<<
            await add_col("data", "TEXT")  # canonical JSON blob the GUI selects directly

            if "created_at" not in cols:
                await db.execute(
                    "ALTER TABLE eligible_tokens ADD COLUMN created_at INTEGER DEFAULT (strftime('%s','now'))"
                )
                cols.add("created_at")

            # Helpful indices for GUI filtering / sorting
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_eligible_tokens_creation_ts ON eligible_tokens(creation_timestamp)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_eligible_tokens_created_at ON eligible_tokens(created_at)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_eligible_tokens_timestamp ON eligible_tokens(timestamp)"
            )

            await db.commit()

    except Exception as e:
        logger.error("Failed to ensure eligible_tokens schema: %s", e, exc_info=True)
        # Do NOT call st.* here.


# =============================
# shortlist_tokens schema
# =============================
async def ensure_shortlist_tokens_schema():
    """
    Create/upgrade the `shortlist_tokens` table (address, data, created_at).
    """
    try:
        async with connect_db() as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS shortlist_tokens (
                    address TEXT PRIMARY KEY,
                    data    TEXT,
                    created_at INTEGER DEFAULT (strftime('%s','now'))
                )
                """
            )
            cols = set()
            async with db.execute("PRAGMA table_info(shortlist_tokens)") as cur:
                async for row in cur:
                    cols.add(row[1])
            if "data" not in cols:
                await db.execute("ALTER TABLE shortlist_tokens ADD COLUMN data TEXT")
                cols.add("data")
            if "created_at" not in cols:
                await db.execute("ALTER TABLE shortlist_tokens ADD COLUMN created_at INTEGER")
                cols.add("created_at")
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_shortlist_tokens_created_at ON shortlist_tokens(created_at)"
            )
            await db.commit()
    except Exception as e:
        logger.error("Failed to ensure shortlist_tokens schema: %s", e, exc_info=True)


# =====================================================
# price_cache schema (corrected & hardened)
# =====================================================
async def ensure_price_cache_schema():
    """
    Create/upgrade the `price_cache` table. This function is async and intended to be
    executed on a background event loop. It only logs errors.

    Fixes & notes:
      - Guarantees presence of price, % change, volume, and timestamp columns.
      - Adds/refreshes indices (timestamp; address) for faster lookups.
      - Updates the local `cols` snapshot when adding columns to avoid duplicate ALTERs.
    """
    try:
        async with connect_db() as db:
            # Base table (idempotent)
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS price_cache (
                    address TEXT PRIMARY KEY,
                    price REAL DEFAULT 0,
                    price_change_1h REAL DEFAULT 0,
                    price_change_6h REAL DEFAULT 0,
                    price_change_24h REAL DEFAULT 0,
                    volume_24h REAL DEFAULT 0,
                    timestamp INTEGER DEFAULT (strftime('%s','now'))
                )
                """
            )

            # Snapshot existing columns
            cols = set()
            async with db.execute("PRAGMA table_info(price_cache)") as cur:
                async for row in cur:
                    cols.add(row[1])

            async def add_if_missing(col: str, sqltype: str, default_clause: str = "DEFAULT 0"):
                if col not in cols:
                    await db.execute(f"ALTER TABLE price_cache ADD COLUMN {col} {sqltype} {default_clause}")
                    cols.add(col)

            # Ensure all price-related columns exist
            await add_if_missing("price", "REAL")
            await add_if_missing("price_change_1h", "REAL")
            await add_if_missing("price_change_6h", "REAL")
            await add_if_missing("price_change_24h", "REAL")
            await add_if_missing("volume_24h", "REAL")

            # Ensure timestamp column with default now()
            if "timestamp" not in cols:
                await db.execute(
                    "ALTER TABLE price_cache ADD COLUMN timestamp INTEGER DEFAULT (strftime('%s','now'))"
                )
                cols.add("timestamp")

            # Helpful indices
            await db.execute("CREATE INDEX IF NOT EXISTS idx_price_cache_timestamp ON price_cache(timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_price_cache_address ON price_cache(address)")

            await db.commit()
    except Exception as e:
        logger.error("Failed to initialize/migrate price_cache: %s", e, exc_info=True)
        # Do NOT call st.* here.


# >>> Normalize scan results so UI code can safely .get(...) on category buckets
def _normalize_scan_result(obj):
    if isinstance(obj, list):
        return {"unknown": obj}
    if isinstance(obj, dict):
        return obj
    return {"unknown": []}


# Ensure GUI tables exist up-front so early SELECTs work (matches bot writer)
async def _ensure_gui_tables():
    await ensure_eligible_tokens_schema()
    await ensure_price_cache_schema()
    await ensure_shortlist_tokens_schema()

# Call once at import time (via the background loop) to avoid "no such column: data"
try:
    run_async_task(_ensure_gui_tables, timeout=20)
except Exception:
    # If run_async_task isn't ready yet, the first UI action will invoke them again (idempotent).
    pass

# =============================
# Price & API Helpers
# =============================
# mirror Part-3's optional import behavior
try:
    import aiohttp as _aiohttp_local
except Exception:
    _aiohttp_local = None

async def get_cached_price(address: str, cache_seconds=300) -> dict | None:
    try:
        async with connect_db() as db:
            async with db.execute(
                "SELECT price, price_change_1h, price_change_6h, price_change_24h, volume_24h, timestamp "
                "FROM price_cache WHERE address = ? AND timestamp >= ?",
                (address, int(time.time()) - cache_seconds),
            ) as cur:
                row = await cur.fetchone()
                if row:
                    return {
                        "price": row["price"],
                        "price_change_1h": row["price_change_1h"],
                        "price_change_6h": row["price_change_6h"],
                        "price_change_24h": row["price_change_24h"],
                        "volume_24h": row["volume_24h"],
                    }
        return None
    except Exception as e:
        logger.warning("Failed to retrieve cached price for %s: %s", address, e)
        return None


async def cache_price(address: str, price_data: dict):
    try:
        async with connect_db() as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO price_cache
                (address, price, price_change_1h, price_change_6h, price_change_24h, volume_24h, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    address,
                    float(price_data.get("price", 0) or 0),
                    float(price_data.get("price_change_1h", 0) or 0),
                    float(price_data.get("price_change_6h", 0) or 0),
                    float(price_data.get("price_change_24h", 0) or 0),
                    float(price_data.get("volume_24h", 0) or 0),
                    int(time.time()),
                ),
            )
            await db.commit()
    except Exception as e:
        logger.warning("Failed to cache price for %s: %s", address, e)


# Prefer de-duplicated best-pair logic from Part 3 when available
async def _best_solana_pair_from_list(pairs: list, token_address: str):
    try:
        # Use Part-3 helper if present to avoid duplication
        return await _lf_best_solana_pair(pairs, token_address)  # type: ignore[name-defined]
    except Exception:
        # Fallback local implementation (kept for resilience)
        best = None
        best_liq = -1.0
        for p in pairs or []:
            try:
                if p.get("chainId") != "solana":
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
        return best and best.get("pairAddress")


async def fetch_price_by_token_endpoint(token_address: str, session):
    if not _aiohttp_local:
        return None
    try:
        async with session.get(
            f"https://api.dexscreener.com/latest/dex/tokens/{token_address}",
            timeout=8,
        ) as resp:
            if resp.status != 200:
                logger.warning("Dexscreener tokens endpoint failed for %s: HTTP %s", token_address, resp.status)
                return None
            data = await resp.json()
            pairs = data.get("pairs") or []
            best_pair_addr = await _best_solana_pair_from_list(pairs, token_address)
            if not best_pair_addr:
                logger.warning("No SOL pair for %s via tokens endpoint", token_address)
                return None
            chosen = next((p for p in pairs if p.get("pairAddress") == best_pair_addr), None)
            if not chosen:
                return None
            pc = chosen.get("priceChange") or {}
            vol = chosen.get("volume") or {}
            return {
                "price": float(chosen.get("priceUsd", 0) or 0),
                "price_change_1h": float(pc.get("h1", 0) or 0),
                "price_change_6h": float(pc.get("h6", 0) or 0),
                "price_change_24h": float(pc.get("h24", 0) or 0),
                "volume_24h": float(vol.get("h24", 0) or 0),
            }
    except Exception as e:
        logger.warning("Token endpoint fetch failed for %s: %s", token_address, e)
        return None


async def fetch_pair_address(token_address: str, session) -> str | None:
    if not _aiohttp_local:
        return None
    try:
        async with session.get(
            f"https://api.dexscreener.com/latest/dex/search?q={token_address}",
            timeout=8,
        ) as response:
            if response.status != 200:
                logger.warning("Dexscreener pair search failed for %s: HTTP %s", token_address, response.status)
                return None
            data = await response.json()
            pairs = data.get("pairs", [])
            best = await _best_solana_pair_from_list(pairs, token_address)
            return best
    except Exception as e:
        logger.warning("Failed to fetch pair address for %s: %s", token_address, e)
        return None


async def fetch_price_changes_dexscreener(pair_address: str, session) -> dict | None:
    if not _aiohttp_local:
        return None
    try:
        async with session.get(
            f"https://api.dexscreener.com/latest/dex/pairs/solana/{pair_address}",
            timeout=8,
        ) as response:
            if response.status != 200:
                logger.warning("Dexscreener failed for pair %s: HTTP %s", pair_address, response.status)
                return None
            data = await response.json()
            pair = data.get("pair", {})
            return {
                "price": float(pair.get("priceUsd", 0) or 0),
                "price_change_1h": float((pair.get("priceChange") or {}).get("h1", 0) or 0),
                "price_change_6h": float((pair.get("priceChange") or {}).get("h6", 0) or 0),
                "price_change_24h": float((pair.get("priceChange") or {}).get("h24", 0) or 0),
                "volume_24h": float((pair.get("volume") or {}).get("h24", 0) or 0),
            }
    except Exception as e:
        logger.warning("Failed to fetch Dexscreener data for pair %s: %s", pair_address, e)
        return None


async def fetch_price_data(
    token_address: str,
    pair_address: str | None = None,
    retries: int = 2,
    backoff: int = 5,
    session: any | None = None,
) -> dict:
    cached_price = await get_cached_price(token_address)
    if cached_price:
        return cached_price

    # Without aiohttp, fall back quickly to cache/seed
    if _aiohttp_local is None and session is None:
        cached_fallback = await get_cached_price(token_address, cache_seconds=86400)
        if cached_fallback:
            return cached_fallback
        fd = FALLBACK_PRICES.get(
            token_address,
            {"price": 0.0, "price_change_1h": 0.0, "price_change_6h": 0.0, "price_change_24h": 0.0, "volume_24h": 0.0},
        )
        await cache_price(token_address, fd)
        return fd

    own = False
    if session is None:
        session = _aiohttp_local.ClientSession()  # type: ignore[union-attr]
        own = True
    try:
        tok_data = await fetch_price_by_token_endpoint(token_address, session=session)
        if tok_data:
            await cache_price(token_address, tok_data)
            return tok_data

        if pair_address is None:
            pair_address = await fetch_pair_address(token_address, session=session)
        if pair_address:
            pair_data = await fetch_price_changes_dexscreener(pair_address, session=session)
            if pair_data:
                await cache_price(token_address, pair_data)
                return pair_data

        if BIRDEYE_API_KEY:
            endpoint = f"https://public-api.birdeye.so/defi/price?address={token_address}"
            for attempt in range(retries):
                try:
                    async with session.get(endpoint, headers={"X-API-KEY": BIRDEYE_API_KEY}, timeout=8) as response:
                        if response.status == 429:
                            if attempt < retries - 1:
                                logger.warning("Birdeye 429 for %s; retrying in %ss", token_address, backoff)
                                await asyncio.sleep(backoff)
                                backoff *= 2
                                continue
                            logger.warning("Rate limit exceeded for %s on Birdeye after %s attempts", token_address, retries)
                            break
                        if response.status != 200:
                            logger.warning("Birdeye failed for %s: HTTP %s", token_address, response.status)
                            break
                        data = await response.json()
                        bd = {
                            "price": float((data.get("data") or {}).get("value") or 0.0),
                            "price_change_1h": 0.0,
                            "price_change_6h": 0.0,
                            "price_change_24h": 0.0,
                            "volume_24h": 0.0,
                        }
                        await cache_price(token_address, bd)
                        return bd
                except Exception as e:
                    logger.warning("Failed to fetch Birdeye data for %s: %s", token_address, e)
                    break
        else:
            logger.info("BIRDEYE_API_KEY not set; skipping Birdeye fallback for %s", token_address)
    finally:
        if own:
            try:
                await session.close()
            except Exception:
                pass

    cached_fallback = await get_cached_price(token_address, cache_seconds=86400)
    if cached_fallback:
        return cached_fallback

    fd = FALLBACK_PRICES.get(
        token_address,
        {"price": 0.0, "price_change_1h": 0.0, "price_change_6h": 0.0, "price_change_24h": 0.0, "volume_24h": 0.0},
    )
    await cache_price(token_address, fd)
    return fd

# ==========================
# Data Load/Save & Maintenance
# ==========================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_tokens_from_db(_cache_buster: int = 0) -> list[dict]:
    """
    Read the GUI-facing shortlist from tokens.sqlite3 and normalize it so:
      - created_at filter is applied ONLY when fallback_hours > 0
      - buckets (high/mid/low/newly_launched) are assigned reliably even for JSON rows
      - stable/whitelisted tokens are hidden from category tables
      - percent-change fields are mapped from whatever the writer used, instead of
        defaulting to zeros (all-neutral 0.0 are shown as '—')
      - duplicates are removed by token address, keeping the most recent/best row
    """
    try:
        # --- settings pulled from session state (safe defaults) ---
        try:
            fallback_hours = int(st.session_state.get("fallback_hours", 24))
        except Exception:
            fallback_hours = 24
        cutoff_ts = None
        if isinstance(fallback_hours, int) and fallback_hours > 0:
            cutoff_ts = int(time.time()) - fallback_hours * 3600

        # --- helper: number coercion ---
        def _f(x, default=0.0):
            if x is None:
                return float(default)
            if isinstance(x, (int, float)):
                return float(x)
            try:
                xs = str(x).replace(",", "").strip()
                if xs in ("", "-", "—", "None", "null"):
                    return float(default)
                return float(xs)
            except Exception:
                return float(default)

        # --- helper: detect obvious stablecoins (extra safety) ---
        _STABLE_SYMBOLS = {"USDC", "USDT", "USD", "USDE", "DAI", "USDS", "PYUSD"}
        def _looks_stable(tok: dict) -> bool:
            sym = str(tok.get("symbol") or "").upper()
            nm  = str(tok.get("name") or "").upper()
            if sym in _STABLE_SYMBOLS:
                return True
            for kw in ("USDC", "USDT", "STABLE", "DOLLAR", "USD "):
                if kw in sym or kw in nm:
                    return True
            return False

        # --- helper: bucket by market cap, add "newly_launched" by creation_timestamp ---
        NEW_TOKEN_BADGE_WINDOW_MIN = 60 * 24  # <= 24h
        def _assign_bucket(tok: dict) -> str:
            try:
                now = int(time.time())
                cts = tok.get("creation_timestamp")
                if cts is not None:
                    try:
                        cts = int(cts)
                        if cts >= now - NEW_TOKEN_BADGE_WINDOW_MIN * 60:
                            return "newly_launched"
                    except Exception:
                        pass
            except Exception:
                pass

            mc = _f(tok.get("market_cap") or tok.get("mc"))
            if mc >= 500_000:
                # IMPORTANT: GUI expects "high_cap" (not "large_cap")
                return "high_cap"
            if 100_000 <= mc < 500_000:
                return "mid_cap"
            return "low_cap"

        # --- helper: normalize one token row from JSON/legacy into canonical keys ---
        def _normalize(tok: dict) -> dict:
            out = dict(tok) if isinstance(tok, dict) else {}

            # canonical numeric fields
            out["market_cap"]   = _f(tok.get("market_cap", tok.get("mc")))
            out["liquidity"]    = _f(tok.get("liquidity"))
            out["volume_24h"]   = _f(
                tok.get("volume_24h", tok.get("vol24h", tok.get("vol24", tok.get("volume24h"))))
            )
            out["price"]        = _f(tok.get("price"))

            # percent-change fields (map synonyms)
            pc1h  = tok.get("price_change_1h",  tok.get("pc1h",  tok.get("change_1h",  tok.get("pct_change_1h"))))
            pc6h  = tok.get("price_change_6h",  tok.get("pc6h",  tok.get("change_6h",  tok.get("pct_change_6h"))))
            pc24h = tok.get("price_change_24h", tok.get("pc24h", tok.get("change_24h", tok.get("pct_change_24h"))))

            # Treat "neutral" zero triplets as missing so UI shows "—"
            def _maybe_none(v): 
                return None if v in (None, "", "null") else _f(v, 0.0)

            p1, p6, p24 = _maybe_none(pc1h), _maybe_none(pc6h), _maybe_none(pc24h)
            if (p1 or 0.0) == 0.0 and (p6 or 0.0) == 0.0 and (p24 or 0.0) == 0.0:
                p1 = p6 = p24 = None

            out["price_change_1h"]  = p1
            out["price_change_6h"]  = p6
            out["price_change_24h"] = p24

            # address/symbol/name
            out["address"] = tok.get("address") or tok.get("mint") or tok.get("token_address")
            out["symbol"]  = tok.get("symbol") or ""
            out["name"]    = tok.get("name")   or out["symbol"] or ""

            # creation timestamp (best-effort)
            if out.get("creation_timestamp") is None:
                maybe_cts = tok.get("creation_timestamp") or tok.get("created_at")
                try:
                    out["creation_timestamp"] = int(maybe_cts) if maybe_cts is not None else None
                except Exception:
                    out["creation_timestamp"] = None

            # bucket (respect existing _bucket, else assign)
            out["_bucket"] = tok.get("_bucket") or _assign_bucket(out)

            # score (keep if present for tie-breakers)
            if "score" in tok:
                try:
                    out["score"] = float(tok["score"])
                except Exception:
                    pass

            return out

        # --- read from DB ---
        tokens: list[dict] = []
        now = int(time.time())

        async def inner() -> list[dict]:
            await ensure_eligible_tokens_schema()
            await ensure_price_cache_schema()

            out: list[dict] = []
            async with connect_db() as db:
                # detect schema
                cols = set()
                async with db.execute("PRAGMA table_info(eligible_tokens)") as cur_cols:
                    async for row in cur_cols:
                        cols.add(row[1])

                # new JSON layout
                # new tolerant JSON+legacy reader
                if "data" in cols:
                    # Read rows regardless of whether data is null, so we can handle mixed schema states.
                    params: tuple = ()
                    where = "1=1"
                    if cutoff_ts is not None and "created_at" in cols:
                        where += " AND created_at >= ?"
                        params = (cutoff_ts,)

                    q = f"SELECT *, created_at FROM eligible_tokens WHERE {where} ORDER BY created_at DESC LIMIT 1000"
                    async with db.execute(q, params) as cur:
                        async for row in cur:
                            try:
                                # Prefer JSON blob when present
                                raw_data = row.get("data") if isinstance(row, dict) else row["data"]
                                if raw_data:
                                    tok = json.loads(raw_data)
                                    if isinstance(tok, dict):
                                        t = _normalize(tok)
                                        t["_created_at_row"] = int(row.get("created_at") or 0)
                                        out.append(t)
                                        continue

                                # If data is NULL or not parseable, reconstruct from scalar columns in this row
                                # Build a dict from column names present in this row
                                # Support both sqlite3.Row (mapping) and row tuples by PRAGMA table_info earlier
                                # Use safe .get() accesses for dict-style rows
                                rdict = dict(row) if isinstance(row, dict) else {}
                                # pick common legacy columns, tolerant to missing keys
                                legacy = {
                                    "address": rdict.get("address") or rdict.get("mint") or rdict.get("token_address"),
                                    "symbol": rdict.get("symbol") or "",
                                    "name": rdict.get("name") or "",
                                    "market_cap": rdict.get("market_cap") or rdict.get("mc"),
                                    "liquidity": rdict.get("liquidity"),
                                    "volume_24h": rdict.get("volume_24h"),
                                    "price": rdict.get("price"),
                                    "price_change_1h": rdict.get("price_change_1h"),
                                    "price_change_6h": rdict.get("price_change_6h"),
                                    "price_change_24h": rdict.get("price_change_24h"),
                                    "score": rdict.get("score"),
                                    "categories": rdict.get("categories"),
                                    "creation_timestamp": rdict.get("creation_timestamp") or rdict.get("created_at") or rdict.get("timestamp"),
                                }
                                t = _normalize(legacy)
                                t["_created_at_row"] = int(rdict.get("created_at") or rdict.get("timestamp") or 0)
                                out.append(t)
                            except Exception:
                                continue

        raw_tokens = run_async_task(inner) or []

        # --- address de-dupe: keep newest, then higher score, then better liq/vol/mc ---
        def _key_for_best(t):
            return (
                int(t.get("_created_at_row") or 0),
                float(t.get("score") or 0.0),
                float(t.get("liquidity") or 0.0),
                float(t.get("volume_24h") or 0.0),
                float(t.get("market_cap") or 0.0),
            )

        best_by_addr: dict[str, dict] = {}
        for t in raw_tokens:
            addr = str(t.get("address") or "")
            if not addr:
                continue
            prev = best_by_addr.get(addr)
            if prev is None or _key_for_best(t) > _key_for_best(prev):
                best_by_addr[addr] = t

        tokens = list(best_by_addr.values())

        # --- final GUI-side filters: hide stables/whitelisted in category lists ---
        hidden_addrs = set()
        try:
            hidden_addrs |= set(ALWAYS_HIDE_IN_CATEGORIES)   # e.g., stables/WSOL
        except Exception:
            pass
        try:
            hidden_addrs |= set(WHITELISTED_TOKENS)
        except Exception:
            pass

        filtered: list[dict] = []
        for t in tokens:
            addr = str(t.get("address") or "")
            if addr in hidden_addrs or _looks_stable(t):
                continue
            filtered.append(t)

        logger.info("Fetched %d tokens from database (post-filter, de-duped)", len(filtered))
        return filtered

    except Exception as e:
        logger.error("Failed to fetch tokens from database: %s", e)
        try:
            st.error(f"❌ Failed to fetch tokens from database: {e}")
        except Exception:
            pass
        return []


async def prune_old_tokens(days: int = 7) -> bool:
    try:
        cutoff = int(time.time()) - days * 24 * 3600
        async with connect_db() as db:
            await db.execute("DELETE FROM eligible_tokens WHERE timestamp < ?", (cutoff,))
            await db.commit()
        logger.info("Pruned tokens older than %d days", days)
        return True
    except Exception as e:
        logger.error("Failed to prune old tokens: %s", e)
        # Avoid calling st.* from background threads
        return False

# =============================
# Safety, Formatting & Grouping
# =============================

def emoji_safety(score):
    if score is None:
        return "❓"
    try:
        s = float(score)
        if s >= 70:
            return "🛡️"
        elif s >= 40:
            return "⚠️"
        else:
            return "💀"
    except Exception:
        return "❓"


def _normalize_labels(labels_raw: list | str | None) -> set:
    if not labels_raw:
        return set()
    if isinstance(labels_raw, str):
        try:
            labels_raw = json.loads(labels_raw)
        except Exception:
            labels_raw = [labels_raw]
    if not isinstance(labels_raw, (list, set, tuple)):
        return set()
    return {str(l).lower().strip() for l in labels_raw if l}

DANGER_LABELS = {
    "rugpull", "scam", "honeypot", "blacklisted", "malicious", "high_risk",
    "locked_liquidity_low", "no_liquidity", "suspicious_contract"
}

WARNING_LABELS = {
    "new_token", "low_liquidity", "unverified_contract", "recently_deployed",
    "moderate_risk", "unknown_team"
}

def _normalize_categories(tok: dict) -> None:
    """
    Ensure categories contain canonical names used by trading/eligibility:
      - low_cap, mid_cap, large_cap, newly_launched
    Accept common aliases and map them to canonical names in-place.
    Place this helper with the other small helpers so static analyzers see it
    before any call sites.
    """
    if tok is None:
        return
    cats = tok.get("categories") or []
    if isinstance(cats, str):
        cats = [cats]
    out = set()
    for c in cats:
        if not c:
            continue
        cs = str(c).strip().lower()
        # map common aliases -> canonical
        if cs in {"high", "high_cap", "large", "large_cap"}:
            out.add("large_cap")
        elif cs in {"mid", "mid_cap"}:
            out.add("mid_cap")
        elif cs in {"low", "low_cap"}:
            out.add("low_cap")
        elif cs in {"new", "newly_launched", "newlylaunched"}:
            out.add("newly_launched")
        elif cs in {"shortlist", "fallback", "unknown_cap"}:
            out.add(cs)  # preserve other tags
        else:
            out.add(cs)
    # if there was a _bucket field, also honor/mirror that
    try:
        bk = str(tok.get("_bucket") or "").strip().lower()
        if bk:
            if bk in {"high", "high_cap", "large", "large_cap"}:
                out.add("large_cap")
            elif bk in {"mid", "mid_cap"}:
                out.add("mid_cap")
            elif bk in {"low", "low_cap"}:
                out.add("low_cap")
            elif bk in {"new", "newly_launched"}:
                out.add("newly_launched")
    except Exception:
        pass
    # Preserve any other tags like 'shortlist' etc. and write back
    tok["categories"] = list(out)


# ---- Whitelists / UI hide lists (typed + import-friendly) ----
from typing import Set

# Local defaults (safe to ship)
DEFAULT_WHITELISTED_TOKENS: Set[str] = {
    "So11111111111111111111111111111111111111112",  # WSOL
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
    "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",   # JUP
    "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",   # mSOL
    "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj",  # stSOL
    "J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn",  # JitoSOL
    "bSo13r4TkiE4KumL71LsHTPpL2euBYLFx6h9HP3piy1",   # bSOL
    "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3",  # PYTH
}
DEFAULT_ALWAYS_HIDE_IN_CATEGORIES: Set[str] = {
    "So11111111111111111111111111111111111111112",  # WSOL
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
}

# Start with defaults
WHITELISTED_TOKENS: Set[str] = set(DEFAULT_WHITELISTED_TOKENS)
ALWAYS_HIDE_IN_CATEGORIES: Set[str] = set(DEFAULT_ALWAYS_HIDE_IN_CATEGORIES)

# Try to merge in package-level definitions if available
try:  # bundle path
    from solana_trading_bot_bundle.trading_bot.eligibility import (  # type: ignore[reportMissingImports]
        WHITELISTED_TOKENS as _PKG_WHITELIST,
        ALWAYS_HIDE_IN_CATEGORIES as _PKG_HIDE,
    )
    if _PKG_WHITELIST:
        WHITELISTED_TOKENS |= set(_PKG_WHITELIST)
    if _PKG_HIDE:
        ALWAYS_HIDE_IN_CATEGORIES |= set(_PKG_HIDE)
except Exception:
    pass


def determine_safety(token: dict) -> tuple[str, str]:
    addr = (token.get("address") or "").strip()
    liq_raw = token.get("liquidity") or 0
    try:
        liq = float(liq_raw)
    except Exception:
        liq = 0.0
    cts = int(token.get("creation_timestamp") or 0)
    is_new = (cts > 0 and (time.time() - cts) <= NEW_TOKEN_SAFETY_WINDOW_MIN * 60)

    if addr in WHITELISTED_TOKENS:
        base, reason = "safe", "whitelist"
    else:
        labels_raw = token.get("labels") or token.get("rugcheck_labels") or []
        labels = _normalize_labels(labels_raw)
        if not labels:
            base, reason = "warning", "no_labels"
        else:
            lset = set(labels)
            if lset & DANGER_LABELS:
                base, reason = "dangerous", "labels"
            elif lset & WARNING_LABELS:
                base, reason = "warning", "labels"
            else:
                base, reason = "safe", "labels_ok"

    if base == "safe" and (liq < SAFE_MIN_LIQUIDITY_USD or is_new):
        base = "warning"; reason = "new_or_low_liquidity"
    return base, reason


def new_coin_badge(token: dict) -> str:
    try:
        cts = int(token.get("creation_timestamp") or 0)  # epoch seconds
    except Exception:
        cts = 0
    addr = (token.get("address") or "").strip()
    if addr in WHITELISTED_TOKENS:
        return ""
    if cts and (time.time() - cts) <= NEW_TOKEN_BADGE_WINDOW_MIN * 60:
        return "🔵 New"
    return ""

# --- Numeric helpers and deduplication ---
import re as _re
_NUM_RE = _re.compile(r"[^0-9.\-]")

def _num(x, default=0.0) -> float:
    if x is None:
        return float(default)
    try:
        if isinstance(x, str):
            x = _NUM_RE.sub("", x)
        return float(x)
    except Exception:
        return float(default)

def enrich_mc_in_place(t: dict) -> None:
    mc = t.get("mc")
    if mc in (None, "", 0, "0"):
        mc = t.get("marketCap") or t.get("fdv") or t.get("fdvUsd") or t.get("market_cap")
    t["mc"] = _num(mc, 0.0)

def normalize_token_fields_in_place(t: dict) -> None:
    """
    Make live-scan tokens consistent with DB schema:
    - creation_timestamp: derive from createdAt if missing
    - liquidity: accept dicts (e.g., {'usd': ...})
    - labels: if a single 'rugcheck_label' exists, map to ['...']
    """
    # creation_timestamp from createdAt (already sec in our live path; guard if ms elsewhere)
    if not t.get("creation_timestamp"):
        created = t.get("createdAt") or t.get("created_at") or 0
        try:
            created = int(created or 0)
            if created > 0:
                t["creation_timestamp"] = created if created < 10**12 else created // 1000
        except Exception:
            pass

    # liquidity normalization
    liq = t.get("liquidity")
    if isinstance(liq, dict):
        t["liquidity"] = _num(liq.get("usd", 0), 0.0)
    else:
        t["liquidity"] = _num(liq, 0.0)

    # labels normalization (single -> list)
    if "labels" not in t:
        rl = t.get("rugcheck_label") or t.get("rugcheck_labels")
        if rl:
            t["labels"] = [rl] if isinstance(rl, str) else rl

def deduplicate_tokens(tokens: list[dict]) -> list[dict]:
    by_addr = {}
    for t in tokens or []:
        addr = (t.get("address") or "").strip()
        if not addr:
            continue
        prev = by_addr.get(addr)
        if not prev:
            by_addr[addr] = t
            continue
        def _f(k):
            try:
                return float(t.get(k) or 0), float(prev.get(k) or 0)
            except:
                return 0.0, 0.0
        for k in ("timestamp", "score", "liquidity", "mc"):
            cur, old = _f(k)
            if cur > old:
                prev[k] = t.get(k)
        for k in ("symbol","name"):
            if (not prev.get(k)) and t.get(k):
                prev[k] = t.get(k)
        if t.get("labels"):
            prev["labels"] = t["labels"]
    return list(by_addr.values())

# --- Rugcheck verification (live/cache/whitelist/failed) ---
from typing import Tuple, Dict, Any, List, Optional

# === Canonical .env loader (single source) ===============================
import os as _os
from pathlib import Path as _Path
try:
    from dotenv import load_dotenv as _load_dotenv  # type: ignore
except Exception:
    _load_dotenv = None

_APPDATA_ENV = _Path(_os.getenv("LOCALAPPDATA", _Path.home() / "AppData" / "Local")) / "SOLOTradingBot" / ".env"
if _load_dotenv:
    try:
        _load_dotenv(_APPDATA_ENV, override=False)
    except Exception:
        pass
# ========================================================================

_RC_CACHE: Dict[str, Dict[str, Any]] = {}

def _now() -> float: 
    return time.time()

def _rc_age_sec(meta: Optional[Dict[str, Any]]) -> float:
    if not meta or "fetched_at" not in meta:
        return float("inf")
    return max(0.0, _now() - float(meta["fetched_at"]))

# --- Hide helpers --------------------------------------------------------
import os, time

# Fallback to whatever your file defined; otherwise empty
try:
    _ADDR_HIDE = set(ALWAYS_HIDE_IN_CATEGORIES)
except Exception:
    _ADDR_HIDE = set()

def _csv_env(name: str, default: str) -> set[str]:
    raw = os.getenv(name, default)
    return {s.strip().upper() for s in str(raw).split(",") if s.strip()}

# Symbols/names to hide even if the address is different (WSOL/USDC clones, etc.)
ALWAYS_HIDE_SYMBOLS = _csv_env(
    "ALWAYS_HIDE_SYMBOLS",
    "SOL,WSOL,USDC,USDT,WRAPPED SOLANA,WRAPPED USD COIN,USD COIN,USDC.E"
)


# ------------------------------------------------------------------------

async def _fetch_rugcheck_labels(session, jwt: str, addr: str) -> Tuple[List[str], int]:
    base = "https://api.rugcheck.xyz/v1"
    headers = {"Authorization": f"Bearer {jwt}"} if jwt else {}
    timeout = aiohttp.ClientTimeout(total=12, sock_connect=5, sock_read=8)
    urls = [
        f"{base}/tokens/scan/solana/{addr}",
        f"{base}/tokens/{addr}",
        f"{base}/tokens/{addr}/labels",
    ]
    last_status = 0
    for url in urls:
        try:
            async with session.get(url, headers=headers, timeout=timeout) as resp:
                last_status = resp.status
                if resp.status != 200:
                    continue
                data = await resp.json()
                payload = data.get("data", data) if isinstance(data, dict) else {}
                labels = payload.get("labels") or payload.get("riskLabels") or payload.get("flags") or []
                return labels, 200
        except Exception:
            continue
    return [], last_status

async def verify_rugcheck_for_tokens(tokens: List[Dict[str, Any]], minutes_ttl: int = 15) -> Dict[str, int]:
    from solana_trading_bot_bundle.rugcheck_auth import get_rugcheck_token
    import inspect
    ttl_sec = max(60, minutes_ttl * 60)
    counters = dict(total=0, live=0, cache=0, skipped_whitelist=0, failed=0)
    whitelist = WHITELISTED_TOKENS

    try:
        jwt = (await get_rugcheck_token() if inspect.iscoroutinefunction(get_rugcheck_token) else get_rugcheck_token())
    except Exception:
        jwt = ""

    async with aiohttp.ClientSession() as session:
        for t in tokens or []:
            counters["total"] += 1
            addr = (t.get("address") or "").strip()
            if not addr:
                t["rugcheck_meta"] = {"status":"failed","source":"n/a","http_status":0,"fetched_at":_now(),"age_sec":0,"labels_sample":[]}
                counters["failed"] += 1
                continue
            if addr in whitelist:
                t["rugcheck_meta"] = {"status":"skipped_whitelist","source":"whitelist","http_status":200,"fetched_at":_now(),"age_sec":0,"labels_sample":[]}
                counters["skipped_whitelist"] += 1
                continue
            cached = _RC_CACHE.get(addr)
            if cached and _rc_age_sec(cached) <= ttl_sec:
                t["rugcheck_meta"] = {
                    "status":"ok_cache","source":"cache","http_status":cached.get("http_status",200),
                    "fetched_at":cached.get("fetched_at",_now()),"age_sec":_rc_age_sec(cached),
                    "labels_sample": list(cached.get("labels", []))[:5],
                }
                t.setdefault("labels", cached.get("labels", []))
                counters["cache"] += 1
                continue
            labels, http_status = ([], 0)
            if jwt:
                labels, http_status = await _fetch_rugcheck_labels(session, jwt, addr)
            status = "ok_live" if (http_status == 200 and isinstance(labels, list)) else "failed"
            meta = {"status":status,"source":"live" if status=="ok_live" else "live_error",
                    "http_status":http_status,"fetched_at":_now(),"age_sec":0,"labels_sample": list(labels)[:5]}
            t["rugcheck_meta"] = meta
            if status == "ok_live":
                t.setdefault("labels", labels)
                _RC_CACHE[addr] = {"labels":labels,"fetched_at":meta["fetched_at"],"http_status":http_status,"source":"live"}
                counters["live"] += 1
            else:
                counters["failed"] += 1
    return counters

def format_percent(value):
    try:
        value = float(value)
        return f"🟢 {value:.2f}%" if value > 0 else f"🔴 {value:.2f}%" if value < 0 else f"{value:.2f}%"
    except Exception:
        return "N/A"

DEX_BASE = "https://dexscreener.com/solana"

def to_display_row(t: dict) -> dict:
    addr = (t.get("address") or "").strip()
    name = str(t.get('symbol') or t.get('name') or 'N/A')
    safety_label, _ = determine_safety(t)

    price = _num(t.get('price'), 0.0)
    liq   = _num(t.get('liquidity'), 0.0)
    # Robust MC fallback so categories work even if upstream uses other keys
    mc    = _num(t.get('mc') or t.get('market_cap') or t.get('fdv'), 0.0)
    vol   = _num(t.get('volume_24h') or (t.get('volume') or {}).get('h24'), 0.0)

    dex_url = f"{DEX_BASE}?q={addr}" if addr else ""
    row = {
        "Name": name,
        "Token Address": addr,
        "Dex": dex_url,
        "Safety": {"safe": "🛡️ Safe", "warning": "⚠️ Warning", "dangerous": "💀 Dangerous"}.get(safety_label, "❓ Unknown"),
        "Price": f"${price:,.6f}",
        "Liquidity": f"${liq:,.0f}",
        "Market Cap": f"${mc:,.0f}",
        "MC Tier": market_cap_badge(mc),
        "New": new_coin_badge(t),
        "Volume (24h)": f"${vol:,.0f}",
        "1H": format_percent(t.get("price_change_1h", 0)),
        "6H": format_percent(t.get("price_change_6h", 0)),
        "24H": format_percent(t.get("price_change_24h", 0)),
    }
    return row


def partition_by_category(tokens: list[dict]):
    """
    Return dict with keys: new, large, mid, low.
    - New tokens are exclusive.
    - Hide core assets by address AND symbol/name (to filter WSOL/USDC clones).
    - Deduplicate by address across all groups.
    """
    import time
    now = int(time.time())
    new_cutoff = now - NEW_TOKEN_BADGE_WINDOW_MIN * 60 if 'NEW_TOKEN_BADGE_WINDOW_MIN' in globals() else now - 180*60

    groups = {"new": [], "large": [], "mid": [], "low": []}
    seen: set[str] = set()

    for t in tokens or []:
        addr = (t.get("address") or t.get("token_address") or "").strip()
        if not addr:
            continue
        if addr in seen:
            continue
        if _hidden_token(t):
            continue

        try:
            cts = int(t.get("creation_timestamp") or t.get("created_at") or 0)
        except Exception:
            cts = 0

        mc_val = _num(t.get("mc") or t.get("market_cap") or t.get("fdv") or 0, 0.0)

        # New is exclusive (but don't consider whitelisted core assets as "new")
        if cts >= new_cutoff and addr not in WHITELISTED_TOKENS:
            groups["new"].append(t)
            seen.add(addr)
            continue

        if mc_val >= 500_000:
            groups["large"].append(t)
        elif mc_val >= 100_000:
            groups["mid"].append(t)
        else:
            groups["low"].append(t)
        seen.add(addr)

    # sort each group by score desc (fallback to market cap)
    def _score(tok):
        try:
            return float(tok.get("score") or tok.get("market_cap") or tok.get("mc") or tok.get("fdv") or 0)
        except Exception:
            return 0.0

    for k in groups:
        groups[k].sort(key=_score, reverse=True)
    return groups

# =============================
# Bot Control
# =============================
def _log_rc(proc: subprocess.Popen):
    rc = proc.poll()
    logger.info("Bot subprocess terminated with return code: %s", rc)
    if rc not in (0, None):
        logger.error("Bot subprocess exited abnormally (code: %s)", rc)

def _stop_proc_tree(proc: subprocess.Popen, *, write_stop_flag: bool, grace: float = 2.0, term_wait: float = 5.0):
    """Gracefully stop a subprocess; escalate to terminate/kill with child cleanup."""
    if not proc or proc.poll() is not None:
        return

    # Optional cooperative stop
    if write_stop_flag:
        try:
            STOP_FLAG_PATH.write_text("1", encoding="utf-8")
            logger.info("Stop flag written to %s", STOP_FLAG_PATH)
        except Exception:
            pass

    # Give the bot a moment to exit on its own
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass

    # Then terminate process tree if still running
    try:
        ps_proc = psutil.Process(proc.pid)
    except psutil.Error:
        ps_proc = None

    if ps_proc and ps_proc.is_running():
        for child in ps_proc.children(recursive=True):
            try:
                child.terminate()
            except psutil.Error:
                pass
        try:
            ps_proc.terminate()
        except psutil.Error:
            pass

        try:
            proc.wait(timeout=term_wait)
            logger.info("Stopped bot (PID: %s)", proc.pid)
        except subprocess.TimeoutExpired:
            try:
                ps_proc.kill()
            except psutil.Error:
                pass
            logger.warning("Forced termination of bot (PID: %s)", proc.pid)

# =============================
# Bot spawn/stop helpers
# =============================

def _build_pythonpath_for_spawn() -> str:
    """Compose a PYTHONPATH that lets the child import the bundle as a package."""
    parts: list[str] = []
    # Prefer repo dir and bundle dir first
    parts.extend([str(_PKG_ROOT), str(_PKG_DIR)])
    # Keep any existing PYTHONPATH entries (dedup)
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        parts.extend([p for p in existing.split(os.pathsep) if p])
    # Deduplicate while preserving order
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return os.pathsep.join(out)

# =============================
# UI Helpers
# =============================
from collections import deque

@st.cache_data(ttl=10, show_spinner=False)
def read_logs(_k: int, log_file: Path = LOG_PATH, max_lines: int = 300) -> str:
    """Tail logs efficiently; cache invalidates when _k (mtime) changes."""
    try:
        if log_file.exists():
            dq = deque(maxlen=int(max_lines))
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    dq.append(line)
            return "".join(dq) if dq else "(empty log file)"
        return "⚠️ No logs found."
    except Exception as e:
        # (keep this as-is)
        logger.error("Failed to read logs: %s", e)
        return f"❌ Failed to read logs: {e}"

def _mtime(p: Path) -> int:
    try:
        return int(p.stat().st_mtime)
    except Exception:
        return 0

# ---- DRY JSON file reader (cached via mtime key) ----
def _read_json_file(path: Path) -> dict | list | None:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def _read_rugcheck_status(_k: int) -> dict:
    """Read Rugcheck status JSON; cache invalidates when file mtime changes."""
    return _read_json_file(STATUS_FILE) or {}

def _fmt_age(ts) -> str:
    """Return a compact 'X ago' string for a unix timestamp, or ''."""
    try:
        if ts is None:
            return ""
        age = max(0, int(time.time() - int(ts)))
        if age < 60:
            return f"{age}s"
        m, s = divmod(age, 60)
        if m < 60:
            return f"{m}m {s}s"
        h, m = divmod(m, 60)
        if h < 24:
            return f"{h}h {m}m"
        d, h = divmod(h, 24)
        return f"{d}d {h}h"
    except Exception:
        return ""


def show_rugcheck_banner():
    """
    Displays a compact status banner for Rugcheck.
    Handles missing keys, stale timestamps, and multi-line messages gracefully.
    """
    try:
        status = _read_rugcheck_status(_mtime(STATUS_FILE)) or {}
    except Exception:
        status = {}

    enabled   = bool(status.get("enabled"))
    available = bool(status.get("available"))
    msg_raw   = status.get("message") or ""

    # Normalize message: collapse whitespace/newlines so the banner doesn't look broken
    msg = " ".join(str(msg_raw).split()).strip()
    age_text = _fmt_age(status.get("timestamp"))
    suffix = f" (updated {age_text} ago)" if age_text else ""

    if not enabled:
        # Disabled: info (blue)
        st.info(f"🛈 Rugcheck disabled — tokens allowed{suffix}" + (f". {msg}" if msg else ""))
        return

    if not available:
        # Enabled but currently unavailable: warning (yellow)
        st.warning(f"⚠️ Rugcheck unavailable{suffix}" + (f". {msg}" if msg else ""))
        return

    # Enabled and available: success (green)
    st.success(f"✅ Rugcheck active{suffix}" + (f". {msg}" if msg else ""))

@st.cache_data(show_spinner=False)
def _read_rugcheck_failures(_k: int) -> list[dict]:
    """Read Rugcheck failures JSON; cache invalidates when file mtime changes."""
    data = _read_json_file(FAILURES_FILE)
    return data if isinstance(data, list) else []

def display_rugcheck_failures_table():
    failures = _read_rugcheck_failures(_mtime(FAILURES_FILE))
    if not failures:
        st.info("No Rugcheck validation failures recorded.")
        return
    st.warning("Some tokens failed Rugcheck validation and were excluded (unless whitelisted):")
    st.dataframe(
        [{"Token Address": f.get("address", ""), "Reason": f.get("reason", "")} for f in failures],
        width="stretch",
        hide_index=True,
        height=_df_height(len(failures)),  # <-- use the helper here
    )

def _df_height(n_rows: int, row_px: int = 34, header_px: int = 38, pad_px: int = 16, max_px: int = 520) -> int:
    """Compute a DataFrame height that avoids the inner vertical scrollbar."""
    return min(header_px + n_rows * row_px + pad_px, max_px)


# =============================
# Streamlit UI
# =============================
st.title("📈 SOLO Meme Coin Trading Bot")
st.markdown("<style>.stAlert { margin-top: .25rem; }</style>", unsafe_allow_html=True)
show_rugcheck_banner()

# --- Background init error relay (avoid using st.* in background threads) ---
INIT_ERROR_MSG: str | None = None

# Defensive fallbacks for globals that other modules usually define
DEV_MODE = bool(globals().get("DEV_MODE", False))
NEW_TOKEN_BADGE_WINDOW_MIN = int(globals().get("NEW_TOKEN_BADGE_WINDOW_MIN", 180))

# Initialize session state (define BEFORE any widgets that use these keys)
st.session_state.setdefault("bot_process", None)
st.session_state.setdefault("bot_running", False)
st.session_state.setdefault("last_token_refresh", 0.0)
st.session_state.setdefault("auto_refresh_enabled", False)  # widget-owned key; set default only
st.session_state.setdefault("_auto_refresh_prev", False)     # helper to detect toggle changes
st.session_state.setdefault("token_container", st.empty())   # placeholder for safe .empty()

cfg = load_config()
if cfg is None:
    st.error("❌ Failed to load configuration. Please check your setup.")
    st.stop()

# Set BASE_DIR now that Streamlit is initialized (for bot run)
BASE_DIR = (Path(sys.executable).parent if getattr(sys, "frozen", False)
            else Path(__file__).resolve().parent)

_resolve_db_and_logs_from_config(cfg)
_rebind_logger_file_handler()

# Initialize async tasks in a background thread
def initialize_async_tasks():
    # IMPORTANT: Do not call any st.* APIs inside this thread to avoid
    # "missing ScriptRunContext" warnings. Use logging and module globals instead.
    global INIT_ERROR_MSG
    try:
        # Pass CALLABLES to run_async_task so the coroutine is created inside it
        run_async_task(ensure_eligible_tokens_schema)
        run_async_task(ensure_price_cache_schema)
    except Exception as e:
        logger.error("Initialization failed", exc_info=True)
        # Stash the message in a global that the main thread can read
        INIT_ERROR_MSG = str(e)

threading.Thread(target=initialize_async_tasks, daemon=True).start()

# Surface any background init error (will appear on next rerun/interaction)
if INIT_ERROR_MSG:
    st.error(f"❌ Initialization failed: {INIT_ERROR_MSG}")

# ======================= Tabs + Sticky-Tab Shim =======================
BASE_TAB_LABELS = [
    "🔧 Configuration",
    "⚙️ Bot Control",
    "🪙 Discovered Tokens",
    "💹 P&L",
    "📊 Status",
    "🛡 Rugcheck",
]
DEV_EXTRA_LABELS = ["🧪 Diagnostics", "📜 Trade Logs"]  # inserted before Rugcheck

if DEV_MODE:
    TAB_LABELS = BASE_TAB_LABELS[:-1] + DEV_EXTRA_LABELS + BASE_TAB_LABELS[-1:]
    (
        tab_config,
        tab_control,
        tab_tokens,
        tab_pl,
        tab_status,
        tab_diag,
        tab_tradelogs,
        tab_rc,
    ) = st.tabs(TAB_LABELS)
else:
    TAB_LABELS = BASE_TAB_LABELS[:]
    tab_config, tab_control, tab_tokens, tab_pl, tab_status, tab_rc = st.tabs(TAB_LABELS)

# Use existing sticky-tab util from Part 1 (no duplicate JS here)
desired_label = st.session_state.pop("_next_tab", None) or st.session_state.get("active_tab", "⚙️ Bot Control")
if desired_label not in TAB_LABELS:
    desired_label = "⚙️ Bot Control"
# Persist & nudge if needed
if desired_label != st.session_state.get("active_tab"):
    try:
        goto_tab(desired_label)  # from Part 1
    except Exception:
        st.session_state["active_tab"] = desired_label  # fallback: at least persist choice

# ======================= Rugcheck Tab =======================
with tab_rc:
    st.header("🛡 Rugcheck Validation Failures")
    display_rugcheck_failures_table()
    # Optionally also show the inline banner
    # show_rugcheck_banner()

# ======================= Configuration Tab =======================
with tab_config:
    st.header("🔧 Bot Configuration")
    st.subheader("Bot Settings")

    # ensure sections exist
    bot_cfg = cfg.setdefault("bot", {})
    disc_cfg = cfg.setdefault("discovery", {})
    for sec in ("low_cap", "mid_cap", "large_cap", "newly_launched"):
        disc_cfg.setdefault(sec, {})

    # ---- cross-platform path defaults ----
    def _default_token_db_path() -> str:
        if os.name == "nt":
            base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
            return str(base / "SOLOTradingBot" / "tokens.db")
        return str(Path.home() / ".cache" / "SOLOTradingBot" / "tokens.db")

    def _default_log_dir() -> str:
        if os.name == "nt":
            base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
            return str(base / "SOLOTradingBot" / "logs")
        return str(Path.home() / ".cache" / "SOLOTradingBot" / "logs")

    def _looks_windows_path(p: str) -> bool:
        return isinstance(p, str) and (":\\" in p or p.startswith("\\\\"))

    def _normalize_path_for_os(raw: str, is_dir: bool = False) -> tuple[str, bool]:
        """
        If running on POSIX and a Windows-style path is provided, replace with a sane default.
        Returns (normalized_path, was_rewritten)
        """
        if not raw:
            return (_default_log_dir() if is_dir else _default_token_db_path(), False)
        if os.name != "nt" and _looks_windows_path(raw):
            return (_default_log_dir() if is_dir else _default_token_db_path(), True)
        return (raw, False)

    # ---- bot knobs ----
    bot_cfg["minimum_wallet_balance_usd"] = st.number_input(
        "Minimum Wallet Balance (USD)",
        key="bot_min_wallet_usd",
        min_value=0.0,
        value=float(bot_cfg.get("minimum_wallet_balance_usd", 0.20)),
        step=0.10,
        format="%.2f",
        help="If your wallet balance is below this amount, the bot will pause new buys.",
    )
    bot_cfg["cycle_interval"] = st.number_input(
        "Cycle Interval (seconds)",
        key="bot_cycle_interval",
        min_value=30,
        value=int(bot_cfg.get("cycle_interval", 60)),
        step=10,
        help="How often the bot runs a discovery/trading cycle.",
    )
    bot_cfg["profit_target"] = st.number_input(
        "Profit Target Ratio",
        key="bot_profit_target",
        min_value=1.0,
        value=float(bot_cfg.get("profit_target", 1.5)),
        step=0.1,
        format="%.2f",
        help="Sell when price ≥ avg buy × this ratio. Example: 1.5 = +50%.",
    )
    bot_cfg["stop_loss"] = st.number_input(
        "Stop Loss Ratio",
        key="bot_stop_loss",
        min_value=0.1,
        max_value=1.0,
        value=float(bot_cfg.get("stop_loss", 0.7)),
        step=0.1,
        format="%.2f",
        help="Sell when price ≤ avg buy × this ratio. Example: 0.7 = −30%.",
    )

    # ---- discovery settings ----
    st.subheader("Discovery Settings")

    # ---------- Workset & Lane Caps (pre-enrichment cut) ----------
    with st.expander("Pre-Enrichment Work-Set & Lane Caps", expanded=False):
        disc_cfg.setdefault("workset_cap", 120)
        disc_cfg.setdefault("quality_cap", 120)
        disc_cfg.setdefault("new_coin_cap", 40)
        disc_cfg.setdefault("spike_cap", 0)

        disc_cfg["workset_cap"] = st.number_input(
            "WORKSET_CAP (max tokens enriched per cycle)",
            key="disc_workset_cap",
            min_value=20, max_value=500, step=10,
            value=int(disc_cfg.get("workset_cap", 120)),
            help="Hard cap on the number of tokens enriched per cycle. This directly controls rate/latency."
        )
        cols_caps = st.columns(3)
        with cols_caps[0]:
            disc_cfg["quality_cap"] = st.number_input(
                "QUALITY_CAP",
                key="disc_quality_cap",
                min_value=0, max_value=500, step=10,
                value=int(disc_cfg.get("quality_cap", 120)),
                help="Max from the quality lane (meets liq/vol floors)."
            )
        with cols_caps[1]:
            disc_cfg["new_coin_cap"] = st.number_input(
                "NEW_COIN_CAP",
                key="disc_new_cap",
                min_value=0, max_value=500, step=5,
                value=int(disc_cfg.get("new_coin_cap", 40)),
                help="Max from the newly-launched lane."
            )
        with cols_caps[2]:
            disc_cfg["spike_cap"] = st.number_input(
                "SPIKE_CAP",
                key="disc_spike_cap",
                min_value=0, max_value=500, step=5,
                value=int(disc_cfg.get("spike_cap", 0)),
                help="Max from the spike/experimental lane."
            )
        st.caption(
            "The work-set used for enrichment is built from lane caps (quality/new/spike) and then "
            "clamped by WORKSET_CAP."
        )

    # ---------- Newly Launched ----------
    with st.expander("Newly Launched Tokens"):
        new_cfg = disc_cfg["newly_launched"]
        new_cfg["max_token_age_minutes"] = st.number_input(
            "Max Token Age (minutes)",
            key="new_max_token_age_minutes",
            min_value=1,
            value=int(new_cfg.get("max_token_age_minutes", 180)),
            step=1,
        )
        new_cfg["liquidity_threshold"] = st.number_input(
            "Min Liquidity (USD)",
            key="new_liquidity_threshold",
            min_value=10,
            value=int(new_cfg.get("liquidity_threshold", 100)),
            step=10,
        )
        new_cfg["volume_threshold"] = st.number_input(
            "Min 24H Volume (USD)",
            key="new_volume_threshold",
            min_value=10,
            value=int(new_cfg.get("volume_threshold", 50)),
            step=10,
        )
        new_cfg["max_rugcheck_score"] = st.number_input(
            "Max Rugcheck Score",
            key="new_max_rugcheck_score",
            min_value=100,
            value=int(new_cfg.get("max_rugcheck_score", 2000)),
            step=100,
        )

    # ---------- Low Cap ----------
    with st.expander("Low Cap Tokens"):
        low_cfg = disc_cfg["low_cap"]
        low_cfg["max_market_cap"] = st.number_input(
            "Max Market Cap (USD)",
            key="low_max_market_cap",
            min_value=1000,
            value=int(low_cfg.get("max_market_cap", 100000)),
            step=1000,
        )
        low_cfg["liquidity_threshold"] = st.number_input(
            "Min Liquidity (USD)",
            key="low_liquidity_threshold",
            min_value=10,
            value=int(low_cfg.get("liquidity_threshold", 100)),
            step=10,
        )
        low_cfg["volume_threshold"] = st.number_input(
            "Min 24H Volume (USD)",
            key="low_volume_threshold",
            min_value=10,
            value=int(low_cfg.get("volume_threshold", 50)),
            step=10,
        )

    # ---------- Mid Cap ----------
    with st.expander("Mid Cap Tokens"):
        mid_cfg = disc_cfg["mid_cap"]
        mid_cfg["max_market_cap"] = st.number_input(
            "Max Market Cap (USD)",
            key="mid_max_market_cap",
            min_value=1000,
            value=int(mid_cfg.get("max_market_cap", 500000)),
            step=1000,
        )
        mid_cfg["liquidity_threshold"] = st.number_input(
            "Min Liquidity (USD)",
            key="mid_liquidity_threshold",
            min_value=10,
            value=int(mid_cfg.get("liquidity_threshold", 500)),
            step=10,
        )
        mid_cfg["volume_threshold"] = st.number_input(
            "Min 24H Volume (USD)",
            key="mid_volume_threshold",
            min_value=10,
            value=int(mid_cfg.get("volume_threshold", 250)),
            step=10,
        )

    # ---------- Large Cap ----------
    with st.expander("Large Cap Tokens"):
        lg_cfg = disc_cfg["large_cap"]
        lg_cfg["liquidity_threshold"] = st.number_input(
            "Min Liquidity (USD)",
            key="large_liquidity_threshold",
            min_value=10,
            value=int(lg_cfg.get("liquidity_threshold", 1000)),
            step=10,
        )
        lg_cfg["volume_threshold"] = st.number_input(
            "Min 24H Volume (USD)",
            key="large_volume_threshold",
            min_value=10,
            value=int(lg_cfg.get("volume_threshold", 500)),
            step=10,
        )

    # ---- global limits ----
    st.subheader("Global Discovery Limits")
    disc_cfg["max_price_change"] = st.number_input(
        "Max 24H Price Change (%)",
        key="global_max_price_change",
        min_value=0.0,
        value=float(disc_cfg.get("max_price_change", 100.0)),
        step=1.0,
        format="%.2f",
    )
    disc_cfg["min_holder_count"] = st.number_input(
        "Min Holder Count",
        key="global_min_holder_count",
        min_value=0,
        value=int(disc_cfg.get("min_holder_count", 50)),
        step=1,
    )

    # ---- shortlist (GUI + bot) ----
    _env_topn = os.getenv("TOP_N_PER_CATEGORY")
    try:
        _env_topn_val = int(_env_topn) if _env_topn not in (None, "") else None
    except Exception:
        _env_topn_val = None

    _config_topn_val = int(disc_cfg.get("shortlist_per_bucket", 5))
    _initial_topn = _env_topn_val if _env_topn_val is not None else _config_topn_val

    st.subheader("Shortlist")
    disc_cfg["shortlist_per_bucket"] = st.number_input(
        "Top N per category (GUI & Bot)",
        key="global_shortlist_per_bucket",
        min_value=1,
        max_value=50,
        value=_initial_topn,
        step=1,
        help=(
            "The bot and GUI will use the same shortlist size. "
            "If the environment variable TOP_N_PER_CATEGORY is set, it overrides the config at runtime."
        ),
    )
    if _env_topn_val is not None:
        st.caption(
            f"🔧 Environment override active: TOP_N_PER_CATEGORY={_env_topn_val} "
            f"(will take precedence over config)."
        )

    # ---- paths & storage ----
    st.subheader("Paths & Storage")
    paths_cfg = cfg.setdefault("paths", {})
    current_db = paths_cfg.get("token_cache_path") or cfg.get("token_cache_path") or _default_token_db_path()
    current_logs = paths_cfg.get("log_dir") or cfg.get("log_dir") or _default_log_dir()

    db_input_raw = st.text_input(
        "Token DB path",
        key="paths_token_db",
        value=str(current_db),
        help="Where to store the token cache database (SQLite).",
    )
    logs_input_raw = st.text_input(
        "Log directory",
        key="paths_log_dir",
        value=str(current_logs),
        help="Folder to write log files (e.g., bot.log).",
    )

    db_input_norm, db_rewritten = _normalize_path_for_os(db_input_raw, is_dir=False)
    logs_input_norm, logs_rewritten = _normalize_path_for_os(logs_input_raw, is_dir=True)

    paths_cfg["token_cache_path"] = db_input_norm
    paths_cfg["log_dir"] = logs_input_norm

    if db_rewritten:
        st.caption(f"⚠️ Provided DB path looked Windows-style on this OS; using '{db_input_norm}'.")
    if logs_rewritten:
        st.caption(f"⚠️ Provided log directory looked Windows-style on this OS; using '{logs_input_norm}'.")

    # ---- save config ----
    if st.button("💾 Save Config"):
        if save_config(cfg):
            st.success("✅ Configuration saved.")
            load_config.clear()                         # reset any cached config loaders
            _resolve_db_and_logs_from_config(cfg)       # rewire DB/log locations if needed
            _rebind_logger_file_handler()               # rotate bound file handler to new location


# ======================= Diagnostics (DEV only) =======================
if DEV_MODE:
    with tab_diag:
        st.header("🧪 Diagnostics")

        # Quick metrics
        try:
            import psutil  # optional
        except Exception:
            psutil = None

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Python", sys.version.split()[0])
            st.metric("Streamlit", st.__version__)
        with col2:
            st.metric("PID (GUI)", os.getpid())
            st.metric("Bot Running", "Yes" if st.session_state.get("bot_running") else "No")
        with col3:
            if psutil:
                proc = psutil.Process(os.getpid())
                st.metric("Memory (GUI)", f"{proc.memory_info().rss/1e6:,.0f} MB")
            else:
                st.write("Install `psutil` for memory stats")

        st.subheader("Paths / Config")
        _paths = cfg.get("paths", {})
        st.json(
            {
                "token_db": _paths.get("token_cache_path"),
                "log_dir": _paths.get("log_dir"),
                "config_source": st.session_state.get("config_source", "unknown"),
                "CONFIG_PATH": str(CONFIG_PATH),
                "DB_PATH": str(DB_PATH),
                "LOG_PATH": str(LOG_PATH),
            }
        )

        # ---- DB inspection helpers (async) ----
        async def _diag_prepare_schema():
            """Best-effort: ensure tables the GUI expects exist."""
            try:
                await ensure_eligible_tokens_schema()
            except Exception:
                pass
            try:
                await ensure_price_cache_schema()
            except Exception:
                pass
            try:
                await ensure_shortlist_tokens_schema()
            except Exception:
                pass
            try:
                async with connect_db() as _db:
                    try:
                        await _ensure_core_tables(_db)
                        await _db.commit()
                    except Exception:
                        pass
            except Exception:
                pass

        async def _diag_inspect_db():
            info = {"tables": [], "counts": {}}
            try:
                await _diag_prepare_schema()
                async with connect_db() as db:
                    async with db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cur:
                        rows = await cur.fetchall()
                        info["tables"] = [r[0] for r in rows] if rows else []
                    for t in ["eligible_tokens", "shortlist_tokens", "price_cache", "tokens", "trade_history"]:
                        try:
                            cur = await db.execute(f"SELECT COUNT(*) FROM {t}")
                            row = await cur.fetchone()
                            info["counts"][t] = int(row[0]) if row else 0
                        except Exception:
                            info["counts"][t] = "n/a"
            except Exception as e:
                info = {"error": str(e)}
            return info

        def _diag_tail_logs(lines: int = 300) -> str:
            try:
                p = Path(str(LOG_PATH))
                if p.exists():
                    dq = deque(maxlen=lines)
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            dq.append(line)
                    return "".join(dq) if dq else "(empty log file)"
                return "No log file yet. Trigger some actions first."
            except Exception as e:
                return f"Failed to read logs: {e}"

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🔍 Inspect DB", key="diag_btn_inspect_db"):
                res = run_async_task(_diag_inspect_db, timeout=40) or {}
                st.json(res or {"error": "inspection failed"})
        with c2:
            if st.button("📜 Tail Logs", key="diag_btn_tail_logs"):
                st.text_area(
                    "Logs (last 300 lines)",
                    value=_diag_tail_logs(),
                    height=320,
                    key="diag_txt_logs"
                )
        with c3:
            if st.button("🧩 Runtime Info", key="diag_btn_env"):
                st.json({
                    "python": sys.version,
                    "executable": sys.executable,
                    "cwd": os.getcwd(),
                    "frozen": bool(getattr(sys, "frozen", False)),
                    "app_dir": os.getcwd(),
                    "py_path_entry_0": sys.path[0] if sys.path else "",
                    "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                })
                # Optional heads-up if DB is empty
                try:
                    _tok = fetch_tokens_from_db()
                    if isinstance(_tok, list) and len(_tok) == 0:
                        st.warning("⚠️ No tokens found in database. Run the bot to populate token data.")
                except Exception:
                    pass

# ======================= Trade Logs (DEV only) =======================
if DEV_MODE:
    with tab_tradelogs:
        st.header("📜 Trade Logs")
        if st.button("🔄 Refresh Logs"):
            read_logs.clear()
        logs = read_logs(_mtime(LOG_PATH))
        st.text_area("Recent Logs", value=logs, height=500)

# ======================= Bot Control =================================
with tab_control:
    st.header("⚙️ Bot Control")

    # --- Compact styles ---
    st.markdown("""
    <style>
      .status-card {
        display:inline-block;
        padding:6px 10px;
        border:1px solid #e8e8e8;
        border-radius:10px;
        background:var(--background-color);
        width:fit-content;
        max-width:460px;
      }
      .status-badge {font-weight:700;font-size:.95rem;margin-bottom:2px;}
      .ok {color:#0c8d39;}
      .stop {color:#c43d3d;}
      .muted {opacity:.75;font-size:.85rem;}
      .thin-rule {margin:.5rem 0 .5rem 0;opacity:.12;}
    </style>
    """, unsafe_allow_html=True)

    # ---- Real process state via PID helpers ----
    running = _bot_running()                           # refresh session_state["bot_pid"] from pidfile if present
    pid = st.session_state.get("bot_pid")              # pick up any pid restored by _bot_running()

    # Seed start time on first recovery so we can show uptime
    if running and "bot_started_at" not in st.session_state:
        try:
            if pid:
                st.session_state.bot_started_at = psutil.Process(int(pid)).create_time()
            else:
                st.session_state.bot_started_at = time.time()
        except Exception:
            st.session_state.bot_started_at = time.time()

    # Pretty uptime for the card
    def _fmt_uptime(sec: int) -> str:
        sec = max(0, int(sec))
        m, s = divmod(sec, 60); h, m = divmod(m, 60)
        return f"{h}h {m:02d}m {s:02d}s"

    uptime = None
    if running and st.session_state.get("bot_started_at"):
        try:
            uptime = _fmt_uptime(time.time() - float(st.session_state["bot_started_at"]))
        except Exception:
            uptime = None

    # ---- Crash output (tail) if the last start failed ----
    err = st.session_state.get("_bot_last_error")
    if err and not running:
        with st.expander("Last bot startup error (tail)", expanded=True):
            st.code(err, language="text")
            if "SUBPROC_LOG" in globals():
                st.caption(f"Log file: {SUBPROC_LOG}")

    # ---------- helpers ----------
    import sqlite3
    from pathlib import Path
    from datetime import datetime

    def _fmt_tdelta(seconds: float) -> str:
        try:
            seconds = max(0, int(seconds))
            h, rem = divmod(seconds, 3600)
            m, s = divmod(rem, 60)
            return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        except Exception:
            return "—"

    def _get_db_path() -> str:
        for candidate in (
            lambda: str(DB_PATH) if "DB_PATH" in globals() else None,
            lambda: st.session_state.get("DB_PATH"),
            lambda: st.session_state.get("db_path"),
            lambda: os.path.join(os.path.expanduser("~"),
                                 "Library/Application Support/SOLOTradingBot",
                                 "tokens.sqlite3"),
        ):
            try:
                pth = candidate()
                if pth:
                    return pth
            except Exception:
                pass
        return "tokens.sqlite3"

    def _get_log_path() -> str:
        for candidate in (
            lambda: str(LOG_PATH) if "LOG_PATH" in globals() else None,
            lambda: st.session_state.get("LOG_PATH"),
            lambda: st.session_state.get("log_file"),
            lambda: os.path.join(os.path.expanduser("~"),
                                 "Library/Application Support/SOLOTradingBot",
                                 "logs", "bot.log"),
        ):
            try:
                pth = candidate()
                if pth:
                    return pth
            except Exception:
                pass
        return "bot.log"

    def _safe_int(x):
        try:
            return int(x)
        except Exception:
            return 0

    def _db_count(table: str) -> int:
        try:
            conn = sqlite3.connect(_get_db_path(), timeout=1)
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(1) FROM sqlite_master WHERE type='table' AND name=?", (table,))
                if _safe_int(cur.fetchone()[0]) == 0:
                    return 0
                cur.execute(f"SELECT COUNT(1) FROM {table}")
                row = cur.fetchone()
                return _safe_int(row[0]) if row else 0
            finally:
                conn.close()
        except Exception:
            return 0

    def _eligible_latest_ts() -> int:
        try:
            conn = sqlite3.connect(_get_db_path(), timeout=1)
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(1) FROM sqlite_master WHERE type='table' AND name='eligible_tokens'")
                if _safe_int(cur.fetchone()[0]) == 0:
                    return 0
                cur.execute("PRAGMA table_info(eligible_tokens)")
                cols = [r[1] for r in cur.fetchall()]
                if "timestamp" in cols:
                    cur.execute("SELECT MAX(COALESCE(timestamp,0)) FROM eligible_tokens")
                    v = cur.fetchone()
                    return _safe_int(v[0] or 0)
                return 0
            finally:
                conn.close()
        except Exception:
            return 0

    def _tail_last_timestamp_from_log() -> float | None:
        try:
            path = _get_log_path()
            if not Path(path).exists():
                return None
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 12_000))
                data = f.read().decode("utf-8", "ignore")
            for line in data.splitlines()[::-1]:
                if not line:
                    continue
                try:
                    ts = datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f")
                    return ts.timestamp()
                except Exception:
                    continue
            return None
        except Exception:
            return None

    # --------- NEW: one-time DB self-heal so Discovered Tokens works ----------
    def _backfill_json_data():
        """
        If eligible_tokens has a JSON 'data' column but it's empty, populate it
        from legacy scalar columns so the GUI list renders again.
        Runs once per session.
        """
        try:
            conn = sqlite3.connect(_get_db_path(), timeout=2)
            try:
                cur = conn.cursor()
                # table present?
                cur.execute("SELECT COUNT(1) FROM sqlite_master WHERE type='table' AND name='eligible_tokens'")
                if _safe_int(cur.fetchone()[0]) == 0:
                    return
                # which columns do we have?
                cur.execute("PRAGMA table_info(eligible_tokens)")
                cols = {r[1] for r in cur.fetchall()}
                if "data" not in cols:
                    return  # JSON path not in schema; nothing to do

                cur.execute("SELECT COUNT(1) FROM eligible_tokens WHERE data IS NOT NULL")
                if _safe_int(cur.fetchone()[0]) > 0:
                    return  # already populated

                # Build a json_object(...) from whatever legacy columns exist
                mapping = [
                    ("address", "address"),
                    ("name", "name"),
                    ("symbol", "symbol"),
                    ("volume_24h", "volume_24h"),
                    ("liquidity", "liquidity"),
                    ("mc", "market_cap"),
                    ("price", "price"),
                    ("price_change_1h", "price_change_1h"),
                    ("price_change_6h", "price_change_6h"),
                    ("price_change_24h", "price_change_24h"),
                    ("score", "score"),
                    ("categories", "categories"),
                    ("timestamp", "timestamp"),
                    ("creation_timestamp", "creation_timestamp"),
                ]
                kv = []
                for key, col in mapping:
                    if col in cols:
                        kv.append(f"'{key}', {col}")
                if not kv:
                    return

                sql = f"UPDATE eligible_tokens " \
                      f"SET data = json_object({', '.join(kv)}) " \
                      f"WHERE data IS NULL"
                cur.execute(sql)
                conn.commit()
                st.caption("✅ Backfilled JSON in eligible_tokens.data for display.")
            finally:
                conn.close()
        except Exception:
            # Non-fatal — keep GUI alive even if SQLite lacks JSON1, etc.
            pass

    if not st.session_state.get("_did_json_backfill", False):
        _backfill_json_data()
        st.session_state["_did_json_backfill"] = True
    # -------------------------------------------------------------------------

    # ---------- live metrics ----------
    uptime_s = (time.time() - st.session_state.get("bot_started_at", time.time())) if running else 0
    last_hb_ts = _tail_last_timestamp_from_log()
    last_hb_ago = (time.time() - last_hb_ts) if (last_hb_ts) else None

    eligible_count = _db_count("eligible_tokens")
    shortlisted = eligible_count
    db_records = _db_count("tokens") + _db_count("token_cache")

    latest_ts = _eligible_latest_ts()
    last_update_age = (time.time() - latest_ts) if latest_ts else None

    prev = st.session_state.get("_bot_control_prev", {})
    deltas = {
        "eligible": eligible_count - int(prev.get("eligible", 0)),
        "short": shortlisted - int(prev.get("short", 0)),
        "db": db_records - int(prev.get("db", 0)),
    }
    st.session_state["_bot_control_prev"] = {
        "eligible": eligible_count,
        "short": shortlisted,
        "db": db_records,
    }

    # ---------- Status + Actions (compact, single source of truth) ----------
    col_status, col_start, col_stop, col_spacer = st.columns([2.2, 0.9, 0.9, 7.0], gap="small")


    with col_status:
        st.markdown(
            f"""
            <div class="status-card">
              <div class="status-badge {'ok' if running else 'stop'}">
                Status: {'Running' if running else 'Stopped'}
              </div>
              <div class="muted">
                PID: {pid or '—'} &nbsp;&nbsp; Uptime: { _fmt_tdelta(uptime_s) if running else '—'}<br/>
                Last heartbeat: { (str(int(last_hb_ago)) + 's ago') if last_hb_ago is not None else '—' }
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_start:
        if st.button("▶ Start Bot", disabled=running, key="btn_start_bot_compact"):
            st.session_state["_next_tab"] = "⚙️ Bot Control"
            if not running:
                # clear any lingering stop flag that would block starts
                try:
                    STOP_FLAG_PATH.unlink(missing_ok=True)
                except Exception:
                    pass
                start_bot()
                st.session_state.bot_started_at = time.time()
                # Immediate refresh so the first click reflects new state
                try:
                    st.rerun()
                except Exception:
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass

    with col_stop:
        if st.button("🛑 Stop Bot", disabled=not running, key="btn_stop_bot_compact"):
            st.session_state["_next_tab"] = "⚙️ Bot Control"
            if running:
                stop_bot()
                # Immediate refresh so the first click reflects new state
                try:
                    st.rerun()
                except Exception:
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass

    st.markdown('<hr class="thin-rule" />', unsafe_allow_html=True)

    # ---------- Metrics (compact) ----------
    m1, m2, m3, m4 = st.columns(4, gap="large")
    with m1:
        st.metric("Eligible tokens", eligible_count, f"{deltas['eligible']:+d}")
    with m2:
        st.metric("Shortlisted", shortlisted, f"{deltas['short']:+d}")
    with m3:
        st.metric("Last update age", (f"{int(last_update_age)} s" if last_update_age is not None else "—"))
    with m4:
        st.metric("DB records", db_records, f"{deltas['db']:+d}")


# -------- Tab 4: Tokens (Discovered Tokens with one-time bootstrap + auto-refresh) --------
with tab_tokens:
    st.header("🪙 Discovered Tokens")

    # 1) One-time live refresh on first landing of this tab
    _bootstrap_discovery_once()  # defined near the top of the file

    # 2) Auto-refresh controls (persisted in session state)
    AUTO_SECS = int(os.getenv("GUI_AUTO_REFRESH_SECS", "60"))

    # ensure dedicated, non-colliding keys exist (avoid reuse of generic keys)
    st.session_state.setdefault("disc_auto_refresh", False)
    st.session_state.setdefault("disc_refresh_interval_ms", AUTO_SECS * 1000)
    st.session_state.setdefault("last_token_refresh", 0.0)

    # DB fallback window (hrs) — default to 24 so persisted shortlist shows if live scan is empty
    st.session_state.setdefault("fallback_hours", 24)

    # optional autorefresh helper (clean, non-blocking) if installed
    try:
        from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh
    except Exception:
        st_autorefresh = None  # graceful fallback

    # widget (stateful via key; initialized from st.session_state["fallback_hours"])
    enabled = st.checkbox(
        f"Enable Auto-Refresh (every {AUTO_SECS} seconds)",
        key="disc_auto_refresh",
        help="Keeps this tab fresh while you watch it.",
    )

    # If enabled and component available, register the timed rerun loop
    if enabled and st_autorefresh:
        st_autorefresh(
            interval=st.session_state.get("disc_refresh_interval_ms", AUTO_SECS * 1000),
            key="disc_autorefresh_loop",
        )
    elif enabled and not st_autorefresh:
        st.info(
            "Auto-refresh requires the optional `streamlit-autorefresh` package. "
            "Install it with: `pip install streamlit-autorefresh`. "
            "Manual refresh still works below."
        )

    # central refresh helper: clear caches, bump timers, and force a rerun
    def _do_refresh() -> None:
        try:
            fetch_tokens_from_db.clear()  # if it's an st.cache_data function
        except Exception:
            pass
        now = time.time()
        st.session_state.last_token_refresh = now
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()  # fallback for older Streamlit
            except Exception:
                pass

    if st.button("🔄 Refresh Tokens", key="manual_refresh_tokens"):
        _do_refresh()

    # --- DB fallback window (used when live scan returns nothing) ---
    _ = st.number_input(
        "Include DB fallback if live scan is empty (hours)",
        min_value=0,
        max_value=168,
        step=1,
        key="fallback_hours",  # uses and updates st.session_state["fallback_hours"]
        help="0 disables fallback. >0 shows tokens saved in the last N hours when live scan returns nothing.",
    )

    # ---------- helpers (hard sanitize + explicit formatting) ----------
    import re
    _NUM_ONLY = re.compile(r"[^0-9eE\.\-]+")

    def _to_float(v) -> float:
        if v is None:
            return 0.0
        if isinstance(v, (int, float)):
            return float(v)
        s = _NUM_ONLY.sub("", str(v))
        try:
            return float(s) if s else 0.0
        except Exception:
            return 0.0

    # One icon ONLY in Safety
    def _safety_str(t: dict) -> str:
        lvl, _ = determine_safety(t)
        return {
            "safe": "🛡️ Safe",
            "warning": "⚠️ Warning",
            "dangerous": "⛔ Dangerous",
        }.get(lvl, "❔ Unknown")

    # ---- Dexscreener URL helper (pair-first) ----
    def _dex_url_for(token: dict) -> str:
        """
        Prefer a pair address if present (Dexscreener pair pages), else token address.
        Fallback to the generic Solana page if neither is available.
        """
        pair = (
            token.get("pair_address")
            or token.get("pair")
            or token.get("pairAddress")
            or token.get("pairAddressBase58")
            or ""
        )
        pair = str(pair).strip()

        addr = (
            token.get("address")
            or token.get("token_address")
            or token.get("mint")
            or ""
        )
        addr = str(addr).strip()

        base = "https://dexscreener.com/solana"
        if pair:
            return f"{base}/{pair}"
        if addr:
            return f"{base}/{addr}"
        return base

    def _row_display(t: dict) -> dict:
        """
        Keep numeric types for money & percent columns so we can style them.
        """
        addr = (t.get("address") or t.get("token_address") or "").strip()
        name = t.get("name") or t.get("symbol") or (addr[:6] + "…" + addr[-4:] if addr else "N/A")
        price = _to_float(t.get("price"))
        liq   = _to_float(t.get("liquidity"))
        mc    = _to_float(t.get("mc") or t.get("market_cap"))
        vol24 = _to_float(t.get("volume_24h"))
        p1  = _to_float(t.get("price_change_1h", 0))
        p6  = _to_float(t.get("price_change_6h", 0))
        p24 = _to_float(t.get("price_change_24h", 0))
        return {
            "Name": name,
            "Token Address": addr,
            "Dex": _dex_url_for(t),
            "Safety": _safety_str(t),
            "Price": price,
            "Liquidity": liq,
            "Market Cap": mc,
            "Volume (24h)": vol24,
            "1H": p1, "6H": p6, "24H": p24,
        }

    # ---------- load + bucket (live-first, optional DB fallback) ----------
    tokens = _pull_tokens_with_fallback(limit_total=25)

    fhours = int(st.session_state.get("fallback_hours", 0) or 0)
    hint = f" (DB fallback up to last {fhours}h if live was empty)" if fhours > 0 else ""
    st.caption(f"Showing {len(tokens)} token(s){hint}.")

    # >>> USE THE "HIDE STABLES" TOGGLE HERE <<<
    if GUI_HIDE_STABLES:
        before = len(tokens)
        tokens = [t for t in tokens if not _is_stable_like_row(t)]
        removed = before - len(tokens)
        if removed > 0:
            st.caption(f"Filtered out {removed} stable-like token(s).")

        # [PATCH-E] mirror pct-change aliases into the names our table expects
    for _t in tokens:
        _t.setdefault("price_change_1h",  _t.get("pct_change_1h"))
        _t.setdefault("price_change_6h",  _t.get("pct_change_6h"))
        _t.setdefault("price_change_24h", _t.get("pct_change_24h"))

    # Partition into buckets
    import math

    def _num_or_nan(x) -> float:
        try:
            f = float(x)
            return f if (f == f) else math.nan  # keep NaN as NaN
        except Exception:
            return math.nan

    # Deterministic ranking: MC (primary) -> Liquidity -> 24h Volume (desc)
    def _score_tuple(t: dict) -> tuple[float, float, float]:
        mc  = _num_or_nan(t.get("mc") or t.get("market_cap"))
        liq = _num_or_nan(t.get("liquidity"))
        vol = _num_or_nan(t.get("volume_24h"))
        mc  = 0.0 if (mc != mc)   else mc
        liq = 0.0 if (liq != liq) else liq
        vol = 0.0 if (vol != vol) else vol
        return (mc, liq, vol)

    def _top_n(items: list[dict], n: int = 5) -> list[dict]:
        return sorted(items, key=_score_tuple, reverse=True)[:n]

    # Partition into buckets (prefer helper, fallback if needed)
    try:
        groups = partition_by_category(tokens)
        high = groups.get("large", []) or groups.get("high", [])
        mid  = groups.get("mid",   [])
        low  = groups.get("low",   [])
        new  = groups.get("new",   [])
    except Exception:
        HIGH_CAP_USD = 500_000
        MID_CAP_USD  = 100_000
        high, mid, low, new = [], [], [], []
        now = int(time.time())
        cutoff = now - NEW_TOKEN_BADGE_WINDOW_MIN * 60
        for t in tokens:
            try:
                cts = int(t.get("creation_timestamp") or 0)
            except Exception:
                cts = 0
            if cts and cts >= cutoff:
                new.append(t)
                continue
            mc_val = _num_or_nan(t.get("mc") or t.get("market_cap"))
            mc_val = 0.0 if (mc_val != mc_val) else mc_val  # NaN->0
            if mc_val >= HIGH_CAP_USD:
                high.append(t)
            elif mc_val >= MID_CAP_USD:
                mid.append(t)
            else:
                low.append(t)

    # Decide Top-N per bucket: env override -> config -> default (5), then clamp to 5
    try:
        env_topn = os.getenv("TOP_N_PER_CATEGORY")
        _env_topn_val = int(env_topn) if env_topn not in (None, "") else None
    except Exception:
        _env_topn_val = None

    try:
        cfg_topn = int(cfg.get("discovery", {}).get("shortlist_per_bucket", 5) or 5)
    except Exception:
        cfg_topn = 5

    # Choose: env overrides config if present
    _topn_cfg = _env_topn_val if _env_topn_val is not None else cfg_topn

    # Clamp to sensible upper/lower bounds. You asked for top 5 per bucket, so enforce 5 as max.
    _topn_cfg = max(1, min(int(_topn_cfg), 5))

    # Apply Top-N to each bucket
    high = _top_n(high, _topn_cfg)
    mid  = _top_n(mid,  _topn_cfg)
    low  = _top_n(low,  _topn_cfg)
    new  = _top_n(new,  _topn_cfg)

    def _render_bucket(label: str, items: list[dict]):
        if not items:
            return
        import pandas as pd

        df = pd.DataFrame([_row_display(t) for t in items], columns=[
            "Name","Token Address","Dex","Safety","Price","Liquidity",
            "Market Cap","Volume (24h)","1H","6H","24H"
        ])

        # Keep these as numeric; do not convert to strings before styling
        money_cols = ["Price","Liquidity","Market Cap","Volume (24h)"]
        pct_cols   = ["1H","6H","24H"]

        def _fmt_money(x):
            try:
                # show small prices with decimals, large as 0 decimals
                return f"${x:,.6f}" if isinstance(x, (int, float)) and x < 1 else f"${x:,.0f}"
            except Exception:
                return x

        def _fmt_pct(x):
            try:
                return f"{float(x):+.2f}%"
            except Exception:
                return x

        def _color_pct(v):
            try:
                v = float(v)
                if v > 0:  return "color: green"
                if v < 0:  return "color: red"
            except Exception:
                pass
            return ""

        # Build Styler and explicitly right-align numeric columns so it works regardless of CSS.
        styler = (
            df.style
              .format({**{c: _fmt_money for c in money_cols}, **{c: _fmt_pct for c in pct_cols}})
              .map(_color_pct, subset=pct_cols)
              .hide(axis="index")
              .set_properties(
                  subset=money_cols + pct_cols,
                  **{"text-align": "right", "padding-right": "6px"}
              )
              .set_table_styles([
                  {"selector": "td", "props": [("font-variant-numeric", "tabular-nums"),
                                               ("font-feature-settings", '"tnum" 1')]}
              ], overwrite=False)
        )

        st.subheader(label)

        st.dataframe(
            styler,
            width="stretch",
            height=_df_height(len(df)),
            hide_index=True,
            column_order=[
                "Name","Token Address","Dex","Safety","Price","Liquidity",
                "Market Cap","Volume (24h)","1H","6H","24H"
            ],
            column_config={
                "Name":          st.column_config.TextColumn("Name",          width=220),
                "Token Address": st.column_config.TextColumn("Token Address", width=340),
                "Dex":           st.column_config.LinkColumn("Dex", display_text="Link", help="Open on Dexscreener", width=20),
                "Safety":        st.column_config.TextColumn("Safety",        width=30),
                "Price":         st.column_config.TextColumn("Price",         width=30),
                "Liquidity":     st.column_config.TextColumn("Liquidity",     width=30),
                "Market Cap":    st.column_config.TextColumn("Market Cap",    width=30),
                "Volume (24h)":  st.column_config.TextColumn("Volume (24h)",  width=30),
                "1H":            st.column_config.TextColumn("1H",            width=20),
                "6H":            st.column_config.TextColumn("6H",            width=20),
                "24H":           st.column_config.TextColumn("24H",           width=20),
            },
        )

    # >>> Draw discovery buckets once per rerun
    if not st.session_state.get("_disc_tables_drawn_once"):
        st.session_state["_disc_tables_drawn_once"] = True
        labels_and_data = [
            (f"🟢 {display_cap('High Cap')}", high),
            ("🟡 Mid Cap",  mid),
            ("🔴 Low Cap",  low),
            ("🔵 New",      new),
        ]
        for label, df in labels_and_data:
            _render_bucket(label, df)


# -------- Tab 6: Status --------
with tab_status:
    st.subheader("📊 Token Status")

    async def _status_snapshot():
        counts = {"holding": 0, "sold": 0, "watchlist": 0}
        rows = []
        try:
            async with connect_db() as db:
                # Holding / Open
                cur1 = await db.execute(
                    "SELECT COUNT(*) AS c FROM tokens WHERE sell_time IS NULL OR is_trading = 1"
                )
                row1 = await cur1.fetchone()

                # Sold / Closed
                cur2 = await db.execute(
                    "SELECT COUNT(*) AS c FROM tokens WHERE sell_time IS NOT NULL"
                )
                row2 = await cur2.fetchone()

                counts["holding"] = int((row1["c"] if isinstance(row1, (dict,)) else row1[0]) if row1 else 0)
                counts["sold"]    = int((row2["c"] if isinstance(row2, (dict,)) else row2[0]) if row2 else 0)

                # On Watchlist (this cycle): prefer eligible_tokens (snapshot), fallback to shortlist_tokens (history)
                try:
                    curw = await db.execute("SELECT COUNT(*) AS c FROM eligible_tokens")
                    roww = await curw.fetchone()
                    counts["watchlist"] = int((roww["c"] if isinstance(roww, (dict,)) else roww[0]) if roww else 0)
                except Exception:
                    try:
                        curw2 = await db.execute("SELECT COUNT(*) AS c FROM shortlist_tokens")
                        roww2 = await curw2.fetchone()
                        counts["watchlist"] = int((roww2["c"] if isinstance(roww2, (dict,)) else roww2[0]) if roww2 else 0)
                    except Exception:
                        counts["watchlist"] = 0

                # Build a schema-aware SELECT for trade_history
                cols = set()
                async with db.execute("PRAGMA table_info(trade_history)") as curti:
                    async for r in curti:
                        cols.add(r[1])
                base_cols = [
                    "symbol", "token_address", "buy_amount", "sell_amount",
                    "buy_price", "sell_price", "profit", "buy_time", "sell_time"
                ]
                # Include 'action' only if present
                if "action" in cols:
                    base_cols.insert(2, "action")
                select_list = ", ".join(base_cols)
                cur = await db.execute(
                    f"SELECT {select_list} FROM trade_history "
                    f"ORDER BY COALESCE(sell_time, buy_time) DESC LIMIT 50"
                )
                rows = [dict(r) for r in await cur.fetchall()]
        except Exception as e:
            logger.error("Status snapshot failed: %s", e, exc_info=True)

        return counts, rows

    # Optional: preview of the active watchlist (current-cycle snapshot first)
    async def _watchlist_preview(limit: int = 25):
        import json
        preview = []
        try:
            async with connect_db() as db:
                # Prefer eligible_tokens (current snapshot)
                try:
                    cur = await db.execute(
                        "SELECT address, data, created_at FROM eligible_tokens "
                        "ORDER BY created_at DESC LIMIT ?", (limit,)
                    )
                    rows = await cur.fetchall()
                    for r in rows:
                        addr = r[0] if not isinstance(r, dict) else r.get("address")
                        data_json = r[1] if not isinstance(r, dict) else r.get("data")
                        db_created_at = r[2] if not isinstance(r, dict) else r.get("created_at")
                        try:
                            t = json.loads(data_json) if isinstance(data_json, str) else (data_json or {})
                        except Exception:
                            t = {}

                        # created_at fallback chain
                        created_at = (
                            db_created_at
                            or t.get("pairCreatedAt")
                            or t.get("createdAt")
                            or t.get("created_at")
                            or t.get("listedAt")
                            or t.get("timestamp")
                        )

                        preview.append({
                            "symbol": (t.get("symbol")
                                    or (isinstance(t.get("baseToken"), dict) and t["baseToken"].get("symbol"))
                                    or "—"),
                            "address": addr,
                            "bucket": t.get("_bucket") or "—",
                            "market_cap": t.get("market_cap") or t.get("mc"),
                            "liquidity": t.get("liquidity") or t.get("liquidity_usd"),
                            "volume_24h": t.get("volume_24h") or t.get("v24hUSD") or t.get("volume"),
                            "created_at": created_at,
                        })
                    return preview
                except Exception:
                    # Fallback to shortlist_tokens (history/upsert view)
                    try:
                        cur2 = await db.execute(
                            "SELECT address, data, created_at FROM shortlist_tokens "
                            "ORDER BY created_at DESC LIMIT ?", (limit,)
                        )
                        rows2 = await cur2.fetchall()
                        for r in rows2:
                            addr = r[0] if not isinstance(r, dict) else r.get("address")
                            data_json = r[1] if not isinstance(r, dict) else r.get("data")
                            db_created_at = r[2] if not isinstance(r, dict) else r.get("created_at")
                            try:
                                t = json.loads(data_json) if isinstance(data_json, str) else (data_json or {})
                            except Exception:
                                t = {}

                            # created_at fallback chain
                            created_at = (
                                db_created_at
                                or t.get("pairCreatedAt")
                                or t.get("createdAt")
                                or t.get("created_at")
                                or t.get("listedAt")
                                or t.get("timestamp")
                            )

                            preview.append({
                                "symbol": (t.get("symbol")
                                        or (isinstance(t.get("baseToken"), dict) and t["baseToken"].get("symbol"))
                                        or "—"),
                                "address": addr,
                                "bucket": t.get("_bucket") or "—",
                                "market_cap": t.get("market_cap") or t.get("mc"),
                                "liquidity": t.get("liquidity") or t.get("liquidity_usd"),
                                "volume_24h": t.get("volume_24h") or t.get("v24hUSD") or t.get("volume"),
                                "created_at": created_at,
                            })
                        return preview
                    except Exception:
                        return []
        except Exception:
            return []

    snap = run_async_task(_status_snapshot, timeout=90)
    counts, recent = (
        snap if isinstance(snap, tuple) and len(snap) == 2
        else ({'holding': 0, 'sold': 0, 'watchlist': 0}, [])
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Holding / Open", counts.get("holding", 0))
    c2.metric("Sold / Closed", counts.get("sold", 0))
    c3.metric("On Watchlist (this cycle)", counts.get("watchlist", 0))

    st.divider()
    st.markdown("**Recent Trades**")
    if recent:
        st.dataframe(recent, width="stretch", hide_index=True)
    else:
        st.info("No trade history yet.")

    # Active Watchlist preview (current-cycle snapshot) — Status tab only
    st.divider()
    st.markdown("**Active Watchlist (current snapshot)**")

    wl = run_async_task(_watchlist_preview, timeout=60)

    if wl:
        formatted_rows = []
        for r in wl:
            formatted_rows.append({
                "symbol": r.get("symbol", "—"),
                "address": r.get("address", "—"),
                "bucket": r.get("bucket", "—"),
                "Market Cap (USD)": fmt_usd(r.get("market_cap")),
                "Liquidity (USD)": fmt_usd(r.get("liquidity")),
                "Volume 24H (USD)": fmt_usd(r.get("volume_24h")),
                "Created At (local)": fmt_ts(
                    r.get("created_at")  # now includes fallback from token JSON
                ),
            })

        st.dataframe(formatted_rows, width="stretch", hide_index=True)
    else:
        st.info("No shortlist found yet. Run discovery from Bot Control to populate this.")





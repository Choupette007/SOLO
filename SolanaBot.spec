# SolanaBot.spec — ONEDIR build for the Streamlit GUI (no collect_tk_files)
# Build: pyinstaller .\SolanaBot.spec

import os
from pathlib import Path
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT
from PyInstaller.building.datastruct import Tree
from PyInstaller.utils.hooks import collect_all, collect_submodules

HERE = Path(os.getcwd())
MAIN_SCRIPT = str(HERE / "run_bot_with_setup.py")
APP_NAME = "SolanaBot"
ICON_PATH = HERE / "bot_icon.ico"

datas, binaries, hiddenimports = [], [], []

def add_pkg(pkg: str):
    try:
        d, b, h = collect_all(pkg)
        datas.extend(d); binaries.extend(b); hiddenimports.extend(h)
    except Exception:
        pass

# UI & plotting
for pkg in ["streamlit", "altair", "plotly", "pandas"]:
    add_pkg(pkg)

# Solana stack & helpers
for pkg in ["solana", "solders", "aiohttp", "aiosqlite", "requests", "dotenv", "psutil", "tenacity", "watchdog"]:
    add_pkg(pkg)

# Submodules (filter out slow/optional bits)
def safe_collect(package: str, exclude_prefixes=()):
    try:
        mods = collect_submodules(package)
        if exclude_prefixes:
            mods = [m for m in mods if not any(m.startswith(p) for p in exclude_prefixes)]
        hiddenimports.extend(mods)
    except Exception:
        pass

# Streamlit’s optional shim triggers a warning if langchain isn’t installed -> exclude it
safe_collect("streamlit", exclude_prefixes=("streamlit.external.langchain",))
safe_collect("altair")
safe_collect("plotly",   exclude_prefixes=("plotly.plotly",))  # deprecated module
# DO NOT collect pandas submodules (drags in tests/pytest) — collect_all("pandas") already did enough
# safe_collect("pandas")  # intentionally omitted
safe_collect("solana")
safe_collect("solders")
hiddenimports += ["tkinter"]

# Project data beside the app
if (HERE / "config.yaml").exists():
    datas.append((str(HERE / "config.yaml"), "."))

if (HERE / ".env.example").exists():
    datas.append((str(HERE / ".env.example"), "."))

if ICON_PATH.exists():
    datas.append((str(ICON_PATH), "."))

# Tk/Tcl fallback: PyInstaller’s tkinter hook usually handles this; we add a safe manual include
try:
    import tkinter
    tk_base = Path(tkinter.__file__).parent
    tcl_dir = tk_base / "tcl"
    tk_dir1 = tk_base / "tk"
    tk_dir2 = tk_base / "tkinter" / "tk"
    if tcl_dir.exists():
        datas.append(Tree(str(tcl_dir), prefix="tcl"))
    if tk_dir1.exists():
        datas.append(Tree(str(tk_dir1), prefix="tk"))
    elif tk_dir2.exists():
        datas.append(Tree(str(tk_dir2), prefix="tk"))
except Exception:
    pass

a = Analysis(
    [MAIN_SCRIPT],
    pathex=[str(HERE)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=(str(ICON_PATH) if ICON_PATH.exists() else None),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="SolanaBot",  # dist\SolanaBot\SolanaTradingBot.exe
)

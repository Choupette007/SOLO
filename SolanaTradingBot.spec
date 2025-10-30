# SolanaBot.spec — onefile build with Tcl/Tk under _internal for Tkinter prompts
# Place this file next to run_bot_with_setup.py and run: pyinstaller SolanaBot.spec

import sys
from pathlib import Path
import tkinter  # ensures availability & gives us a reliable path to Tcl/Tk
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT
from PyInstaller.building.datastruct import Tree

# ------------ Project layout ------------
HERE = Path(__file__).resolve().parent
MAIN_SCRIPT = str(HERE / "run_bot_with_setup.py")
APP_NAME = "SolanaBot"
ICON_PATH = HERE / "bot_icon.ico"  # optional

# ------------ Tcl/Tk bundling ------------
# Your launcher checks for:
#   base/_internal/tcl8.6  and  base/_internal/tk8.6
# We mirror that layout here.
tk_base = Path(tkinter.__file__).parent
tcl_dir = tk_base / "tcl"           # usually .../tkinter/tcl/
tk_dir  = tk_base / "tkinter" / "tk"  # sometimes empty; we still add it safely

tcl_tree = []
tk_tree = []

if tcl_dir.exists():
    # Copy whole tcl folder to _internal/tcl8.6 (covers tcl8.6, encodings, msgs, etc.)
    tcl_tree = [Tree(str(tcl_dir), prefix="_internal/tcl8.6")]
# Some Python dists ship "tk" directly under tkinter; include if present
if (tk_base / "tk").exists():
    tk_tree = [Tree(str(tk_base / "tk"), prefix="_internal/tk8.6")]
elif tk_dir.exists():
    tk_tree = [Tree(str(tk_dir), prefix="_internal/tk8.6")]

extra_datas = []
extra_datas += tcl_tree
extra_datas += tk_tree

# ------------ Hidden imports ------------
# PyInstaller is usually good at discovering these, but we help it along.
hiddenimports = []
hiddenimports += collect_submodules("streamlit")
hiddenimports += collect_submodules("tenacity")
hiddenimports += collect_submodules("dotenv")
# If you use these at runtime (you do in Rugcheck auto-login):
hiddenimports += ["requests", "base58", "solders", "solders.keypair"]

# ------------ Analysis ------------
a = Analysis(
    [MAIN_SCRIPT],
    pathex=[str(HERE)],
    binaries=[],
    datas=extra_datas,
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
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # set True if you want a console window
    icon=(str(ICON_PATH) if ICON_PATH.exists() else None),
)

# Onefile output goes into ./dist/SolanaBot.exe



exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SolanaBot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['bot_icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SolanaBot',
)

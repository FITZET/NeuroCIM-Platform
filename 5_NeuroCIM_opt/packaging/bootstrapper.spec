from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

PLATFORM = Path.cwd().resolve()
hiddenimports = collect_submodules("keyring") + ["keyring.backends.Windows"]

a = Analysis(
    [str(PLATFORM / "bootstrapper.py")],
    pathex=[str(PLATFORM)],
    binaries=[], datas=[], hiddenimports=hiddenimports,
    hookspath=[], hooksconfig={}, runtime_hooks=[],
    excludes=["torch", "torchvision", "numpy", "pandas", "matplotlib", "PySide6"],
    noarchive=False, optimize=1,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="NeuroCIM-Installer", console=False, debug=False,
    bootloader_ignore_signals=False, strip=False, upx=False,
)

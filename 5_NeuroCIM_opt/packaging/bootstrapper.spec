from pathlib import Path

PLATFORM = Path.cwd().resolve()
hiddenimports = ["keyring", "keyring.backends.Windows"]

a = Analysis(
    [str(PLATFORM / "bootstrapper.py")],
    pathex=[str(PLATFORM)],
    binaries=[], datas=[], hiddenimports=hiddenimports,
    hookspath=[], hooksconfig={}, runtime_hooks=[],
    excludes=[
        "torch", "torchvision", "numpy", "pandas", "matplotlib", "PySide6",
        "pytest", "keyring.testing", "keyring.backends.OS_X",
        "keyring.backends.SecretService", "keyring.backends.kwallet",
        "keyring.backends.libsecret",
    ],
    noarchive=False, optimize=1,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="NeuroCIM-Installer", console=False, debug=False,
    bootloader_ignore_signals=False, strip=False, upx=False,
)

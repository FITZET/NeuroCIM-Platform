from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_submodules


PLATFORM = Path.cwd().resolve()
PROJECT = PLATFORM.parent
SEARCH_ROOTS = [
    PLATFORM,
    PROJECT / "1_data_fitting",
    PROJECT / "2_model_basic",
    PROJECT / "3_models" / "LeNet_5",
    PROJECT / "3_models" / "Resnet20",
    PROJECT / "3_models" / "BNN",
    PROJECT / "3_models" / "TinyTransformer",
    PROJECT / "3_models" / "VisionZoo",
    PROJECT / "3_models" / "GPT2",
]
for root in reversed(SEARCH_ROOTS):
    sys.path.insert(0, str(root))

hiddenimports = []
for package in (
    "data_fitting", "fault_layers", "lenet5", "resnet20", "bnn",
    "tiny_transformer", "vision_zoo", "gpt2_cim", "boto3", "botocore", "s3transfer", "jmespath",
    "keyring", "safetensors",
):
    hiddenimports.extend(collect_submodules(package))
hiddenimports.extend([
    "torchvision.datasets",
    "torchvision.transforms",
    "matplotlib.backends.backend_agg",
    "openpyxl",
    "keyring.backends.Windows",
])

a = Analysis(
    [str(PLATFORM / "app.py")],
    pathex=[str(path) for path in SEARCH_ROOTS],
    binaries=[],
    datas=[],
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest", "IPython", "notebook", "jupyter",
        "dask", "distributed", "pyarrow", "numba", "llvmlite",
        "sphinx", "docutils", "babel", "lxml", "win32com",
        "cryptography", "bcrypt", "fsspec", "scipy",
        "torch.utils.tensorboard", "torchaudio",
        "PyQt5", "PyQt6", "PySide2", "tables", "sqlalchemy", "s3fs",
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NeuroCIM",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="NeuroCIM",
)

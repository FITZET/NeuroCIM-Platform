# NeuroCIM DRAM CIM Simulation Platform

Windows 11 desktop platform for device-error fitting, signed differential neural-network inference, batch evaluation, and error-aware training. The numerical workflow is built around the official NeuroSim codebase and supports BNN, LeNet-5, ResNet-20, TinyTransformer, and GPT-2 resources.

The active desktop application is in `5_NeuroCIM_opt`. Model source adapters are in `3_models`; datasets and trained checkpoints are distributed separately as authenticated release assets and are not stored in Git.

## Development

```powershell
cd 5_NeuroCIM_opt
python -m pip install -r requirements.txt
python app.py
```

## Windows release

CPU and NVIDIA GPU releases are built in separate Python environments:

```powershell
.\5_NeuroCIM_opt\scripts\build_installer.ps1 -Flavor CPU -Version 1.0.0
.\5_NeuroCIM_opt\scripts\build_installer.ps1 -Flavor GPU -Version 1.0.0
```

The installer uses disk slices below GitHub's per-asset limit. `update-manifest.json` records every filename and SHA-256 digest used by the authenticated updater.

# Platform architecture

## Data and execution flow

```text
CSV/LUT/MAC bundle
  -> validated canonical data + provenance
  -> cell LUT fitter or grouped MAC fitter
  -> versioned error model
  -> model registry and quantization configuration
  -> exact device inference / error-aware training
  -> NeuroSim trace export and PPA
  -> application result object
  -> CLI or desktop GUI
  -> JSON/CSV/XLSX/report artifacts
```

## Current modules

- `1_data_fitting`: canonical cell-grid loader and fitting algorithms.
- `neurocim_opt.error_model`: deterministic multiplication-LUT contract.
- `neurocim_opt.mac_error_model`: array-MAC import, grouped fits and persistence.
- `neurocim_opt.layers`: exact differential Conv2d/Linear and STE.
- `neurocim_opt.neurosim_adapter`: official trace export, invocation and PPA parser.
- `3_models/LeNet_5`: first model/dataset integration and training implementation.
- `cli.py`: current application entry point; future GUI must call shared services
  rather than shelling out to this CLI.

## Required boundaries

The measured LUT or MAC model determines numerical inference error. NeuroSim models
hardware mapping and PPA and does not directly consume the measured-current LUT.
Four non-negative `X+/X-` by `W+/W-` paths represent signed computation, while their
signed combination is performed outside the analog core.

Model-specific architecture, preprocessing and checkpoints remain under `3_models`.
Reusable error models, orchestration, UI, logging, exporting and packaging remain in
`5_NeuroCIM_opt`. Original files under `0_data` are read-only.

## Planned application/UI split

The next layer will expose typed application services for import, fitting, inference,
training and PPA. Both CLI and PySide6 GUI will use those services. Long operations
run in workers with progress, cancellation and exception propagation; no numerical
algorithm belongs in a widget callback.

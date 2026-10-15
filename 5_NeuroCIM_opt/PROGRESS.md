# Progress

## 2026-09-30 — completed implementation

- Audited the existing fitting, basic-model and LeNet-5 projects; the workspace root
  has no Git metadata, so changes were made directly in the requested source tree.
- Added versioned error-model import/export with source and LUT SHA-256 validation,
  explicit `[W, X]` order, lambda interpolation and interaction-surface validation.
- Added exact signed differential Linear and Conv2d with uint8 magnitude codes,
  per-product LUT access, int64 accumulation and floating bias after rescaling.
- Added exact LUT forward with a floating straight-through gradient, deterministic
  55k/5k/10k MNIST splits, training resume and best-validation checkpoint selection.
- Imported all four device tables and fitted `0_data/01_NMOS.csv` through the existing
  `DataFittingService`.
- Full 10,000-image baseline: float 98.69%, ideal differential 98.70%, and untrained
  01_NMOS differential 98.52% at lambda 1.
- One full device-aware epoch reached 99.02% on both device and ideal differential
  evaluation. Equal-budget ideal QAT reached 98.95% on the device and 99.07% on the
  ideal path; the device-aware advantage for this one seed is 0.07 percentage points.
- Integrated official `neurosim/NeuroSim`, branch `2DInferenceV1.4`, commit
  `8a88abf85844c0e1ba17cc771ea535fff6040456`, in one source directory.
- Added four-path trace export, isolated MinGW build, C++ invocation, PPA parser and
  upstream provenance. Fixed trace orientation against upstream
  `CopyInput[row][vector*bit]` and added exact dimension tests.
- Official NeuroSim completed all five LeNet-5 layers: 4.7236 mm^2 area, 22.1344 us
  pipelined latency, 230.426 nJ dynamic energy, 19.2858 TOPS/W and 45,178.6 FPS.

The local measured LUT determines numerical error and training behavior. NeuroSim
accounts for the four physical arrays, activity traces and hardware parameters; its
analog core does not directly consume the measured multiplication LUT.

## 2026-09-30 — specification platform iteration 1

- Read the full outsourcing specification and created a requirement-to-evidence
  matrix, architecture boundary and decision log under `docs/`.
- Implemented the previously missing array-MAC branch with configurable column
  mapping, UTF-8/GBK and comma/semicolon import, missing-value policies, rectangular
  arrays, 1/2/4/8-bit validation and per-experiment grouping.
- Added linear and polynomial response fits, R²/RMSE/MAE/full-scale nonlinearity,
  fitted-range enforcement and an integrity-checked model bundle.
- Added `fit-mac-model` to the common CLI and ran it on a retained 256-row 32×32,
  8-bit theory dataset. The run produced R² 0.9998356 and 0.8212% full-scale
  nonlinearity under `outputs/acceptance/mac_branch/`.
- Added five focused tests for fit/predict/save/load, GBK/semicolon input, invalid
  bit width, extrapolation rejection and bundle tampering.
- Added a UI-neutral `PlatformService` plus the first real PySide6 workflow: MAC CSV
  selection, encoding/delimiter/missing-value controls, custom mapping, preview,
  background fitting, saved model, metric table, CSV/XLSX export, log view and recent
  path memory.
- Installed and pinned PySide6 6.8.3, explicitly loaded Microsoft YaHei for reliable
  Chinese rendering, executed the GUI workflow offscreen and retained its real
  screenshot at `outputs/acceptance/gui/mac_analysis.png`.
- Completed GUI log clear/export and full MAC import-setting persistence.
- Added an acceptance benchmark script. On the recorded Windows PC, importing and
  validating a 100,000-row, 4.09 MB MAC CSV took 0.227 seconds against the 3-second
  requirement; evidence is under `outputs/acceptance/performance/benchmark.json`.

## 2026-09-30 — specification platform iteration 2

- Added a typed model registry with task, dataset, input shape, metrics, checkpoint
  availability and license metadata. `cli.py list-models` exposes the registry.
- Implemented canonical CIFAR ResNet-20 under `3_models/Resnet20`, plus deterministic
  45k/5k/10k CIFAR-10 loaders, evaluation, bounded training and activation calibration.
- Ran a real CUDA smoke with one 128-image train batch and validation batch; saved a
  clearly marked smoke checkpoint and scales for all 22 Conv/Linear layers.
- Ran exact ideal and 01_NMOS differential inference on one CIFAR-10 image. Both paths
  completed through all layers; this random/smoke model is not an accuracy claim.
- Exported 60.8 MB of four-path traces and ran all 22 MAC layers through official
  NeuroSim. PPA: 24.925 mm², 104.883 us, 10.5408 uJ, 30.1662 TOPS/W and 9,534.45 FPS.
  BatchNorm, ReLU, residual addition and adaptive pooling are explicitly listed as
  excluded non-MAC digital operations in the trace manifest.
- Added a live GUI model/dataset catalog. It reports checkpoint readiness and refuses
  to present the ResNet smoke/random state as an installed production model.
- Full regression now passes 40 tests.

## 2026-09-30 - autonomous iteration round 1

- Generalized cell-error import with configurable columns, units, encoding,
  delimiter and missing-value policies. Native 1/2/4/8-bit acquisition grids are
  retained in fitted bundles; the inference adapter performs the required uint8
  interpolation at its boundary. Existing four-device behavior remains covered.
- Added real LeNet-5 single-sample inference to the GUI. A worker executes float,
  ideal differential and measured-device differential modes and reports labels,
  confidence and per-mode timing. JSON and screenshot evidence are retained under
  `outputs/acceptance/gui_inference/`.
- Removed personal runtime path defaults. Dataset, output, NeuroSim binary and
  toolchain locations now resolve through environment variables, optional user
  configuration and repository-relative defaults.
- Added a Windows environment setup script, third-party notices and a validated
  packaging exclusion manifest so generated data, model weights, toolchains and
  restricted upstream sources cannot enter the distributable by accident.
- Full regression passes 54 tests when `NEUROCIM_MNIST_ROOT` identifies the local
  MNIST root. Without that optional integration resource, 53 tests pass and the
  real-data GUI test is explicitly skipped.

## 2026-09-30 - autonomous iteration round 2

- Added a GUI worker and results page that exports signed differential LeNet-5
  traces, invokes the official NeuroSim executable and displays area, latency,
  energy, energy efficiency and throughput. The verified run completed all five
  layers and retained JSON, raw stdout, trace manifest and a readable screenshot.
- Trained ResNet-20 for 100 deterministic CIFAR-10 epochs on the local RTX 3060.
  The best validation accuracy was 91.30%; independent recomputation on all 10,000
  test images produced 89.98% accuracy.
- Saved best/latest checkpoints with optimizer, scheduler and history state,
  calibrated all 22 Conv/Linear layers, and promoted the verified best model to
  the stable registry checkpoint path.
- Exact signed differential evaluation on fixed test indices 0 through 7 reached
  87.5% with the ideal LUT and 75.0% with 01_NMOS, with 62.5% prediction agreement.
  This bounded mechanism check is retained separately from the full float result.
- Independent QA recomputed the full test metrics, calibration values and bounded
  differential predictions. Full regression now passes 58 tests.

## 2026-09-30 - autonomous iteration round 3

- Added a cancellable LeNet-5 error-aware training service and GUI page with
  configurable error model, lambda, epochs, batch size, device, dataset and output.
  The retained GUI run trained all 55,000 samples for one epoch, validated on 5,000
  samples and evaluated all 10,000 test samples before and after training.
- Device differential accuracy improved from 98.52% to 99.02%; float and ideal
  differential accuracy after training were 99.02% and 99.01%. Ten parameter
  tensors changed with a measured L2 delta of 0.605139.
- Added a formal binary CNN with sign weights, sign hidden activations and a clipped
  identity STE. Three deterministic MNIST epochs produced 98.21% on the complete
  10,000-image test set, exceeding the declared 97% usability threshold.
- BNN checkpoints contain model, optimizer, scheduler, history, global RNG and data
  loader generator states. Independent QA verified exact continuation behavior,
  SHA sidecars and the stable registry checkpoint.
- Fixed 32-sample exact signed differential mechanism evidence is explicitly kept
  separate from the full native result. Ideal and 01_NMOS both scored 100% on that
  bounded sample; this is not reported as a full-test accuracy statistic.
- The model catalog now exposes three installed checkpoints: LeNet-5, ResNet-20 and
  BNN. Full fast regression passes 69 tests with one guarded formal GUI training
  test skipped because its retained real run is already verified.

## 2026-09-30 - autonomous iteration round 4

- Added a cell multiplication-error GUI inside the existing device/MAC fitting
  area. It supports custom columns and units, 1/2/4/8-bit acquisition, UTF-8/GBK,
  comma/semicolon, missing-value policies, preview, background fitting and saved
  configuration.
- A real custom-column 2-bit CSV completed preview and fitting, produced five plots
  and a loadable 256x256 `HardwareErrorModel`. Missing columns, invalid bit width and
  missing numeric values were independently rejected.
- The retained 01_NMOS regression remains R2=0.9908762695039321 with 6.578497847717723%
  full-scale nonlinearity, demonstrating no fitting-core drift.
- Added evidence-linked Chinese user, technical, environment and test manuals plus
  a documentation checker. It verifies five files, 40 local links and 11 evidence
  files while stating license, packaging and unverified-platform limits explicitly.
- Independent QA passed 12 focused checks and the full fast regression now reports
  73 passed with one guarded formal GUI training test skipped.

## 2026-09-30 - autonomous iteration round 5

- Added a real LeNet-5 inference runtime cache keyed by dataset and resource path,
  size, modification time and device. Missing or changed resources evict entries;
  hit, miss and invalidation counters are exposed for verification.
- On CPU, independent QA measured 10.39 ms for a hot single sample and 60.84 ms for
  a hot batch of ten across float, ideal differential and 01_NMOS modes. Cold and
  hot labels and confidence values matched exactly.
- Added explicit array-row, array-column and 1/2/4/8-bit controls to the MAC GUI.
  The selection drives both preview and fitting, persists through QSettings and
  rejects invalid or empty selections.
- Strict linear synthetic data reached R2=1.0000 with effectively zero full-scale
  nonlinearity. A known quadratic disturbance matched independent closed-form OLS
  R2, RMSE and nonlinearity calculations within 1e-12.
- The complete regression, including BNN tests, passes 78 tests with one guarded
  formal GUI training test skipped.

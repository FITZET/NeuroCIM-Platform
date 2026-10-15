# Device-error differential CIM platform

## Documentation

- [用户手册](docs/USER_MANUAL.md)：数据导入、推理、误差感知训练与 PPA 操作。
- [环境安装](docs/ENVIRONMENT_SETUP.md)：Windows 11 / Python 3.11 已验证环境与路径配置。
- [技术手册](docs/TECHNICAL_MANUAL.md)：模块边界、误差模型、差分计算和 NeuroSim 接口。
- [测试报告](docs/TEST_REPORT.md)：当前验收结果、证据路径和未验证项目。
- [第三方声明](THIRD_PARTY_NOTICES.md)：NeuroSim 与 Python 依赖的许可说明。

## Portable setup and local resources

The verified runtime is **CPython 3.11.4 on Windows 11**. Create an isolated
environment in one step with `powershell -ExecutionPolicy Bypass -File
5_NeuroCIM_opt/scripts/setup_environment.ps1`. Dataset and generated-file
locations are resolved from command-line options first, then environment/user
configuration, then repository-relative defaults. Useful variables are
`NEUROCIM_DATA_ROOT`, `NEUROCIM_MNIST_ROOT`, `NEUROCIM_CIFAR10_ROOT`,
`NEUROCIM_OUTPUT_ROOT`, `NEUROCIM_NEUROSIM_BIN`, and
`NEUROCIM_TOOLCHAIN_ROOT`. The optional JSON configuration defaults to
`%APPDATA%/NeuroCIM/config.json` and can be overridden with `NEUROCIM_CONFIG`.

Datasets, trained weights, measurement files, `.tools/`, and `outputs/` are
local assets excluded from releases. See `packaging_manifest.json` and
`THIRD_PARTY_NOTICES.md`; in particular, the vendored NeuroSim core is
CC BY-NC 4.0 and cannot be used commercially under that license.

This directory implements measured multiplication-error import, signed differential
inference, error-aware training, and official NeuroSim PPA estimation. Source is
updated in place; generated artifacts stay under `outputs/` rather than copied into
versioned source directories.

## Implemented flow

```text
raw 256x256 device CSV
  -> 1_data_fitting/DataFittingService
  -> interaction LUT H[W, X] + versioned metadata and hashes
  -> exact signed differential Conv2d/Linear inference
  -> LeNet-5 error evaluation and error-aware training
  -> four-path NeuroSim traces and C++ PPA estimation
```

The fitter removes the two one-variable baselines from the measured current:

```text
I_interaction(X,W) = I(X,W) - I(X,0) - I(0,W) + I(0,0)
```

The exact inference path quantizes unsigned magnitudes to uint8, looks up every
product in `H[W, X]`, accumulates in int64, rescales per output channel, then adds
the floating bias. Signed values use four branches:

```text
Y = X+W+ - X+W- - X-W+ + X-W-
```

During error-aware training the forward result uses the same LUT arithmetic as
inference. Backpropagation uses the corresponding floating Conv2d/Linear derivative
as a straight-through estimator.

## Commands

Run from the repository root:

```powershell
python 5_NeuroCIM_opt/cli.py build-error-model 0_data/01_NMOS.csv `
  --input-kind raw `
  --output-dir 5_NeuroCIM_opt/outputs/error_models/01_NMOS

python 5_NeuroCIM_opt/cli.py smoke-lenet `
  --error-model 5_NeuroCIM_opt/outputs/error_models/01_NMOS/error_model.json `
  --samples 32 --device cuda

python 5_NeuroCIM_opt/cli.py train-eat `
  --error-model 5_NeuroCIM_opt/outputs/error_models/01_NMOS/error_model.json `
  --epochs 1 --device cuda `
  --output-dir 5_NeuroCIM_opt/outputs/eat/01_NMOS

powershell -NoProfile -ExecutionPolicy Bypass -File `
  5_NeuroCIM_opt/scripts/build_neurosim.ps1

python 5_NeuroCIM_opt/cli.py neurosim-eval `
  --error-model 5_NeuroCIM_opt/outputs/error_models/01_NMOS/error_model.json `
  --eat-checkpoint 5_NeuroCIM_opt/outputs/eat/01_NMOS/eat_checkpoint.pt `
  --output-dir 5_NeuroCIM_opt/outputs/neurosim/01_NMOS

python 5_NeuroCIM_opt/cli.py fit-mac-model path/to/array_mac.csv `
  --output-dir 5_NeuroCIM_opt/outputs/mac_models/example

python 5_NeuroCIM_opt/app.py

python 5_NeuroCIM_opt/cli.py list-models
```

Use `--max-train-batches 1 --max-validation-batches 1` for a bounded training smoke
test. Resume with `--resume <output-dir>/eat_checkpoint.pt`. Add
`--training-lut ideal` for the equal-budget ideal differential QAT control.

## Error-model bundle

`error_model.json` records schema version, source path and SHA-256, fitting method,
gain, `[W, X]` axis order, bit widths, rounding/clipping policy, error definition and
fit metadata. `multiplication_lut.npy` is hash-checked when loaded. Error strength is
interpolated as:

```text
H_lambda = round(X*W + lambda * (H - X*W))
```

`lambda=0` is ideal multiplication and `lambda=1` is the complete measured table.

## Official NeuroSim integration

The source under `vendor/neurosim` is the official `neurosim/NeuroSim` repository,
branch `2DInferenceV1.4`, pinned to commit
`8a88abf85844c0e1ba17cc771ea535fff6040456` (CC BY-NC 4.0). The build script uses a
single isolated MinGW environment at `.tools/mingw`. The adapter exports all four
non-negative signed paths and invokes the upstream C++ estimator. See
`UPSTREAM_STATUS.md` for provenance and the exact modeling boundary.

The verified 01_NMOS run produced 4.7236 mm^2 area, 22.1344 us pipelined latency,
230.426 nJ dynamic energy, and 19.2858 TOPS/W. Structured results are saved at
`outputs/neurosim/01_NMOS/neurosim_result.json`.

The second integration is CIFAR ResNet-20. A bounded training/calibration smoke and
exact differential forward have run successfully. Its 22 MAC layers also completed
official NeuroSim estimation; non-MAC BatchNorm, ReLU, residual additions and pooling
are excluded and recorded in the manifest. Reproduce it with
`scripts/evaluate_resnet20_neurosim.py` after supplying a checkpoint and scale file.

## Verification

```powershell
pytest -q 5_NeuroCIM_opt/tests
pytest -q 1_data_fitting/tests 2_model_basic/tests 3_models/LeNet_5/tests
```

Tests cover model integrity, lambda endpoints, LUT axis order, signed exact
arithmetic, convolution padding, STE gradients, NeuroSim trace dimensions and both
layer-by-layer and pipelined output parsing.

## Adding another network

Add model architecture, dataset and model-specific calibration glue in a new
directory under `3_models/`. Keep device-model loading and generic differential
layers in this package. The integration must provide per-layer signed-magnitude input
scales and should fail clearly for operators that cannot be mapped to CIM.

# NeuroSim upstream integration

Verified on 2026-09-30 against the repository identified by the user:

- Repository: https://github.com/neurosim/NeuroSim
- Branch: `2DInferenceV1.4`
- Commit: `8a88abf85844c0e1ba17cc771ea535fff6040456`
- Downloaded archive SHA-256: `102cac5f02c34592087f1a602bfe28a5545f5f0ee76d3173f946b1be7a0433e2`
- Upstream license: CC BY-NC 4.0

The single source checkout is under `vendor/neurosim`. The integration uses the
official PyTorch trace layout and the official C++ NeuroSim area, latency and energy
estimator. The reproducible upstream configuration change is recorded in
`patches/neurosim_2dinferencev1.4.patch`: conventional mapping is enabled for the
general expanded matrices used by LeNet-5 and the four signed branches.

The adapter exports two activation rows (`X+`, `X-`) and two weight columns (`W+`,
`W-`). NeuroSim therefore accounts for all four non-negative physical paths. The
signed result is combined by the local exact LUT inference engine as
`X+W+ - X+W- - X-W+ + X-W-`; this digital subtraction is outside the analog core.
The measured multiplication LUT controls numerical inference and error-aware
training. NeuroSim estimates PPA from the resulting trained network, bit widths,
matrix dimensions, activity traces and hardware configuration; it does not consume
the measured multiplication-current LUT itself.

The verified LeNet-5 run completed all five layers with exit code 0. Its structured
result is `outputs/neurosim/01_NMOS/neurosim_result.json`, and the raw simulator
output is retained beside it.

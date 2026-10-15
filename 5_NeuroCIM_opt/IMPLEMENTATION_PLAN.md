# NeuroSim error-aware platform implementation plan

## Structure rules

- Work directly in this directory; do not create version-suffixed source trees.
- Keep reusable platform code in `neurocim_opt/`, tests in `tests/`, build helpers in
  `scripts/`, one upstream tree in `vendor/neurosim`, and run artifacts in `outputs/`.
- Keep model-specific code under `../3_models/` and preserve source device data and
  original checkpoints.

## Completed acceptance phases

1. Environment and existing-code audit with baseline tests.
2. Raw CSV/LUT import, axis and baseline checks, versioned metadata and hashes.
3. Exact signed differential Linear/Conv2d and STE training path.
4. LeNet-5 calibration, ideal/device inference, validation-based error-aware training
   and checkpoint resume.
5. Official `neurosim/NeuroSim` adapter pinned to branch `2DInferenceV1.4`, commit
   `8a88abf85844c0e1ba17cc771ea535fff6040456`, with license and archive provenance.
6. Four-path trace export, reproducible C++ build, full five-layer PPA run and parsed
   results.
7. Full MNIST comparison among float, ideal differential, measured-device,
   error-aware trained and equal-budget ideal-QAT conditions.

Acceptance is backed by 28 passing tests, the full 10,000-image evaluation, one full
training epoch, the control experiment and a successful official NeuroSim run.

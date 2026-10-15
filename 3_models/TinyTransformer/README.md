# TinyTransformer / CIFAR-10

This platform integration uses the official SHI-Labs Compact Convolutional
Transformer **CCT-7/3x1** for 32x32 CIFAR-10 images. The upstream model source
is retained under `tiny_transformer/` and augmented with platform data and
signed-differential helpers.

- Upstream: https://github.com/SHI-Labs/Compact-Transformers
- Architecture: `cct_7_3x1_32`
- License: Apache-2.0 (see `LICENSE`)
- Checkpoint: official CIFAR-10 300-epoch pretrained weights

The JSON in `checkpoint/input_scales.json` contains calibration scales used by
the platform's 8-bit signed positive/negative weight differential simulation.

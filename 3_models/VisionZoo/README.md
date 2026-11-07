# NeuroCIM VisionZoo

This package adds four isolated model backends without changing the platform
registry or GUI:

| Model ID | Dataset | Definition | Bundled checkpoint |
|---|---|---|---|
| `resnet8-mnist` | MNIST | CIFAR-style `6n+2`, `n=1`; one input channel | Locally trained, reproducible |
| `mobilenetv2-cifar10` | CIFAR-10 | MobileNetV2 x1.0 from chenyaofo | Official upstream |
| `vgg11-bn-cifar10` | CIFAR-10 | VGG11-BN from chenyaofo | Official upstream |
| `squeezenet1_1-cifar10` | CIFAR-10 | Native-32 Fire/BN SqueezeNet, 10-class head | Locally trained, reproducible |

The MobileNetV2 and VGG sources are pinned to
`chenyaofo/pytorch-cifar-models@786c16252c0fc58ee9adac063f8337cc4a7a497a`
under BSD-3-Clause. The SqueezeNet Fire-module semantics and initialization
follow torchvision's BSD implementation, while its stem, pooling schedule and
BatchNorm are specialized for native 32x32 CIFAR-10 inputs. ImageNet weights
are intentionally not presented as CIFAR-10 weights. Exact repositories,
commits, published metrics,
filenames and SHA256 values are recorded in `provenance.json`. License texts
are retained in `LICENSE-chenyaofo.txt` and `LICENSE-torchvision.txt`.
Their CIFAR-10 preprocessing is also pinned to the upstream training config:
mean `(0.4914, 0.4822, 0.4465)` and standard deviation
`(0.2023, 0.1994, 0.2010)`.
On the complete local 10,000-image CIFAR-10 test set, the downloaded files
produced 94.05% (MobileNetV2) and 92.78% (VGG11-BN) on CUDA. These checks use
the exact hashes in `provenance.json`; upstream reports 93.79% and 92.79%.

Download or re-verify the two official CIFAR-10 checkpoints:

```powershell
python 3_models/VisionZoo/download_pretrained.py
```

The downloader writes through a `.part` file and promotes it only after the
full SHA256 matches. Runtime loading uses `torch.load(..., weights_only=True)`,
requires a tensor-only state dictionary and supports an expected SHA256.

Reproducible training keeps 5,000 examples from the official training split
for model selection and evaluates the official test split once after selecting
the best validation state. The published ResNet-8/MNIST checkpoint used 30 epochs;
the published SqueezeNet/CIFAR-10 checkpoint used 160 epochs:

```powershell
python 3_models/VisionZoo/train.py resnet8-mnist `
  --dataset-root data/mnist --device cuda

python 3_models/VisionZoo/train.py squeezenet1_1-cifar10 `
  --dataset-root data/cifar10 --device cuda
```

The package exports a common `create_model(model_id)` factory, deterministic
MNIST/CIFAR-10 loaders, `calibrate_input_scales`, and `differential_copy`.
The formal ResNet-8 checkpoint at `checkpoint/resnet8_mnist.pt` was selected by
validation accuracy after the documented 30-epoch seed-42 run and reaches
99.36% on the 10,000-image MNIST test set. The formal SqueezeNet checkpoint at
`checkpoint/squeezenet1_1_cifar10.pt` was selected by 92.40% validation
accuracy after the documented 160-epoch seed-42 run and reaches 91.68% on the
10,000-image CIFAR-10 test set. Each formal checkpoint has a matching
`*_input_scales.json` generated once from the seeded training calibration
loader; inference and error-aware training reuse these static scales.

Run the isolated tests with:

```powershell
python -m pytest -q 3_models/VisionZoo/tests
```

# Third-Party Notices

NeuroCIM platform source code combines original project code with separately
licensed dependencies and user-supplied assets. This notice is informational;
the license text shipped with each dependency controls.

## NeuroSim

The vendored hardware simulator under `vendor/neurosim` comes from
https://github.com/neurosim/neurosim, branch `2DInferenceV1.4`, pinned from
commit `8a88abf85844c0e1ba17cc771ea535fff6040456`. NeuroSim is licensed under
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
It may not be used for commercial purposes under that license. Its upstream
README and license material must remain with redistributed source.

## Python dependencies

PyTorch, torchvision, NumPy, pandas, Matplotlib, PySide6, openpyxl, pytest and
other packages installed from `requirements.txt` remain under their respective
licenses. Binary redistribution must retain the notices required by the exact
versions included in a release.

## Compact Transformers (CCT)

The TinyTransformer CIFAR-10 integration under `3_models/TinyTransformer` is
based on SHI-Labs Compact Transformers, copyright 2021 SHI Lab, licensed under
Apache License 2.0. The upstream project is
https://github.com/SHI-Labs/Compact-Transformers and its full license is retained
in `3_models/TinyTransformer/LICENSE`.

## nanoGPT and GPT-2

The GPT-2 Small integration under `3_models/GPT2` uses the PyTorch model
definition from Andrej Karpathy's nanoGPT, copyright 2022 Andrej Karpathy,
licensed under the MIT License. The source repository is
https://github.com/karpathy/nanoGPT and its license is retained in
`3_models/GPT2/LICENSE`. Pretrained GPT-2 weights originate from OpenAI and are
subject to the OpenAI GPT-2 modified MIT license.

## WikiText-2

GPT-2 Large evaluation bundles the `wikitext-2-raw-v1` configuration from
`Salesforce/wikitext`, pinned to revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. The pinned dataset card declares
CC BY-SA 3.0 and GNU Free Documentation License (GFDL). The dataset card,
notice, and both license texts are copied into each Windows release under
`licenses/GPT2_and_WikiText2`.

## VisionZoo image classifiers

The MobileNetV2 and VGG11-BN CIFAR-10 definitions and pretrained weights use
the BSD-3-Clause implementation published by chenyaofo/pytorch-cifar-models.
SqueezeNet 1.1 uses torchvision's BSD-3-Clause implementation. ResNet-8 is the
project's explicit 6n+2, n=1 MNIST implementation. The corresponding license
texts are retained as `LICENSE-chenyaofo.txt` and `LICENSE-torchvision.txt` in
`3_models/VisionZoo` and copied into every Windows release.

## Bundled datasets, device measurements, and model weights

The complete Windows package includes MNIST, CIFAR-10, and WikiText-2 for local research
and evaluation. MNIST originates from Yann LeCun's dataset distribution and
CIFAR-10 originates from the dataset published by Alex Krizhevsky, Vinod Nair,
and Geoffrey Hinton. Their original dataset terms and citations continue to
apply.

The package also includes project-provided device measurement CSV files and
trained checkpoints for LeNet-5, BNN, ResNet-20, TinyTransformer/CCT,
ResNet-8, MobileNetV2, VGG11-BN, SqueezeNet 1.1, and GPT-2. These assets are
provided for use with this research platform. The
licenses of the underlying model implementations and pretrained sources remain
in force.

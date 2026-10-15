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

## Bundled datasets, device measurements, and model weights

The complete Windows package includes MNIST and CIFAR-10 for local research
and evaluation. MNIST originates from Yann LeCun's dataset distribution and
CIFAR-10 originates from the dataset published by Alex Krizhevsky, Vinod Nair,
and Geoffrey Hinton. Their original dataset terms and citations continue to
apply.

The package also includes project-provided device measurement CSV files and
trained checkpoints for LeNet-5, BNN, ResNet-20, TinyTransformer/CCT, and
GPT-2. These assets are provided for use with this research platform. The
licenses of the underlying model implementations and pretrained sources remain
in force.

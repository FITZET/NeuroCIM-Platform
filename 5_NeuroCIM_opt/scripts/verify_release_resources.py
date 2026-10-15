from __future__ import annotations

import argparse
import pickle
import struct
from pathlib import Path


MODEL_FILES = (
    "3_models/LeNet_5/chekpoint/lenet5_float.pt",
    "3_models/LeNet_5/chekpoint/quantization.json",
    "3_models/BNN/checkpoint/bnn_mnist.pt",
    "3_models/Resnet20/checkpoint/resnet20_cifar10.pt",
    "3_models/Resnet20/output/formal/input_scales.json",
    "3_models/TinyTransformer/checkpoint/cct_7_3x1_32_cifar10.pth",
    "3_models/TinyTransformer/checkpoint/input_scales.json",
    "3_models/GPT2/checkpoint/gpt2_nanogpt.pt",
    "3_models/GPT2/checkpoint/input_scales.json",
    "3_models/GPT2/checkpoint/tokenizer/vocab.json",
    "3_models/GPT2/checkpoint/tokenizer/merges.txt",
)
MNIST_FILES = (
    ("train-images-idx3-ubyte", 2051, 60_000),
    ("train-labels-idx1-ubyte", 2049, 60_000),
    ("t10k-images-idx3-ubyte", 2051, 10_000),
    ("t10k-labels-idx1-ubyte", 2049, 10_000),
)
CIFAR_FILES = ("batches.meta", "test_batch", *(f"data_batch_{index}" for index in range(1, 6)))


def require_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing bundled resource: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_root", type=Path)
    args = parser.parse_args()
    root = args.release_root.resolve(strict=True)

    for relative in MODEL_FILES:
        require_file(root, relative)

    raw = root / "data/mnist/MNIST/raw"
    for name, magic, count in MNIST_FILES:
        path = require_file(root, str(Path("data/mnist/MNIST/raw") / name))
        with path.open("rb") as stream:
            actual_magic, actual_count = struct.unpack(">II", stream.read(8))
        if (actual_magic, actual_count) != (magic, count):
            raise ValueError(f"Invalid MNIST header in {path}: {(actual_magic, actual_count)}")

    cifar = root / "data/cifar10/cifar-10-batches-py"
    for name in CIFAR_FILES:
        require_file(root, str(Path("data/cifar10/cifar-10-batches-py") / name))
    with (cifar / "test_batch").open("rb") as stream:
        test_batch = pickle.load(stream, encoding="latin1")
    if len(test_batch["data"]) != 10_000 or len(test_batch["labels"]) != 10_000:
        raise ValueError("CIFAR-10 test batch does not contain 10,000 samples")

    csv_files = sorted((root / "0_data").glob("*.csv"))
    if len(csv_files) < 5:
        raise FileNotFoundError("Expected at least five bundled device measurement CSV files")

    print(
        "bundled resources passed: "
        f"{len(MODEL_FILES)} model artifacts, MNIST 70000, CIFAR-10 60000, "
        f"{len(csv_files)} device CSV files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

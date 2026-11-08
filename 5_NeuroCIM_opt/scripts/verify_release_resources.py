from __future__ import annotations

import argparse
import hashlib
import json
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
    "3_models/VisionZoo/checkpoint/resnet8_mnist.pt",
    "3_models/VisionZoo/checkpoint/resnet8-mnist_input_scales.json",
    "3_models/VisionZoo/checkpoint/cifar10_mobilenetv2_x1_0-fe6a5b48.pt",
    "3_models/VisionZoo/checkpoint/mobilenetv2-cifar10_input_scales.json",
    "3_models/VisionZoo/checkpoint/cifar10_vgg11_bn-eaeebf42.pt",
    "3_models/VisionZoo/checkpoint/vgg11-bn-cifar10_input_scales.json",
    "3_models/VisionZoo/checkpoint/squeezenet1_1_cifar10.pt",
    "3_models/VisionZoo/checkpoint/squeezenet1_1-cifar10_input_scales.json",
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
GPT2_LARGE_MANIFEST = Path(__file__).resolve().parents[1] / "packaging/gpt2-large-resources.json"
GPT2_LARGE_RUNTIME = {
    "path": "model.fp16.safetensors",
    "size": 1623601216,
    "sha256": "9008e91f3c248005e27aa0a258f334029cccdd32f700a9c459b143b3eb1eccc3",
}
VISION_ZOO_IDS = {
    "resnet8-mnist", "mobilenetv2-cifar10", "vgg11-bn-cifar10",
    "squeezenet1_1-cifar10",
}


def verify_cuda_runtime(root: Path) -> int:
    """Reject a GPU release that omitted CUDA runtime libraries."""
    internal = root / "NeuroCIM" / "_internal"
    if not internal.is_dir():
        raise FileNotFoundError(f"Frozen runtime directory is missing: {internal}")
    required_patterns = (
        "torch/lib/torch_cuda.dll",
        "torch/lib/c10_cuda.dll",
        "torch/lib/cudnn64_*.dll",
        "torch/lib/cublas64_12.dll",
        "torch/lib/cublasLt64_12.dll",
        "torch/lib/cusparse64_12.dll",
        "torch/lib/nvrtc64_*.dll",
    )
    matched = 0
    for pattern in required_patterns:
        candidates = tuple(internal.glob(pattern))
        if not candidates or any(path.stat().st_size == 0 for path in candidates):
            raise FileNotFoundError(f"Required CUDA runtime library is missing: {pattern}")
        matched += len(candidates)
    return matched


def require_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing bundled resource: {path}")
    return path


def require_resource(root: Path, relative: str) -> Path:
    """Require a non-empty file or a non-empty resource directory."""
    path = root / relative
    if path.is_file() and path.stat().st_size > 0:
        return path
    if path.is_dir() and any(path.iterdir()):
        return path
    raise FileNotFoundError(f"Missing bundled resource: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_gpt2_large(root: Path) -> int:
    manifest = json.loads(GPT2_LARGE_MANIFEST.read_text(encoding="utf-8"))
    resource_root = root / "3_models/GPT2/checkpoint/gpt2-large"
    checked = 0
    for item in manifest["model"]["files"]:
        if item["path"] == "model.safetensors":
            continue
        path = require_file(resource_root, item["path"])
        if path.stat().st_size != int(item["size"]):
            raise ValueError(f"GPT-2-large size mismatch: {path}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"GPT-2-large SHA256 mismatch: {path}")
        checked += 1
    runtime_path = require_file(resource_root, GPT2_LARGE_RUNTIME["path"])
    if runtime_path.stat().st_size != GPT2_LARGE_RUNTIME["size"]:
        raise ValueError(f"GPT-2-large FP16 size mismatch: {runtime_path}")
    if sha256_file(runtime_path) != GPT2_LARGE_RUNTIME["sha256"]:
        raise ValueError(f"GPT-2-large FP16 SHA256 mismatch: {runtime_path}")
    checked += 1
    dataset_root = resource_root / "wikitext-2-raw-v1"
    for item in manifest["dataset"]["files"]:
        path = require_file(dataset_root, item["raw_name"])
        if sha256_file(path) != item["raw_sha256"]:
            raise ValueError(f"WikiText-2 SHA256 mismatch: {path}")
        checked += 1
    return checked


def verify_vision_zoo(root: Path) -> int:
    resource_manifest = json.loads(
        require_file(root, "model_resource_manifest.json").read_text(encoding="utf-8-sig")
    )
    resource_rows = {row["model_id"]: row for row in resource_manifest.get("models", ())}
    missing_rows = sorted(VISION_ZOO_IDS - resource_rows.keys())
    if missing_rows:
        raise ValueError(f"VisionZoo models missing from resource manifest: {missing_rows}")
    for model_id in sorted(VISION_ZOO_IDS):
        row = resource_rows[model_id]
        checkpoint = require_file(root, row["checkpoint"])
        actual = sha256_file(checkpoint)
        if actual != row["sha256"]:
            raise ValueError(
                f"SHA256 mismatch for {model_id}: expected {row['sha256']}, got {actual}"
            )
        for auxiliary in row.get("auxiliary", ()):
            require_resource(root, auxiliary)
    return len(VISION_ZOO_IDS)


def verify_all_model_hashes(root: Path) -> int:
    """Apply the shipped manifest checksum contract to every model checkpoint."""
    resource_manifest = json.loads(
        require_file(root, "model_resource_manifest.json").read_text(encoding="utf-8-sig")
    )
    rows = resource_manifest.get("models", ())
    if len(rows) != 10:
        raise ValueError(f"Expected 10 model rows in resource manifest, got {len(rows)}")
    seen: set[str] = set()
    for row in rows:
        model_id = str(row["model_id"])
        if model_id in seen:
            raise ValueError(f"Duplicate model id in resource manifest: {model_id}")
        seen.add(model_id)
        checkpoint = require_file(root, row["checkpoint"])
        actual = sha256_file(checkpoint)
        if actual != row["sha256"]:
            raise ValueError(
                f"SHA256 mismatch for {model_id}: expected {row['sha256']}, got {actual}"
            )
        for auxiliary in row.get("auxiliary", ()):
            require_resource(root, auxiliary)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_root", type=Path)
    args = parser.parse_args()
    root = args.release_root.resolve(strict=True)
    cuda_library_count = verify_cuda_runtime(root)

    for relative in MODEL_FILES:
        require_file(root, relative)
    model_count = verify_all_model_hashes(root)
    gpt2_large_count = verify_gpt2_large(root)

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
        f"{model_count} verified model checkpoints plus "
        f"{gpt2_large_count} GPT-2-large/WikiText files, "
        "MNIST 70000, CIFAR-10 60000, "
        f"{len(csv_files)} device CSV files, "
        f"{cuda_library_count} CUDA runtime libraries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

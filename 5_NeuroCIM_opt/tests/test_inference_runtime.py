from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from neurocim_opt.application import PlatformService
from neurocim_opt.inference_runtime import (
    clear_lenet_inference_cache,
    get_lenet_inference_resources,
    lenet_inference_cache_info,
)


def test_lenet_loader_prefers_best_error_aware_state(tmp_path: Path) -> None:
    import torch

    model_root = ROOT.parent / "3_models/LeNet_5"
    sys.path.insert(0, str(model_root))
    from lenet5.engine import load_checkpoint
    from lenet5.model import LeNet5

    final = LeNet5().state_dict()
    best = {name: value.clone() for name, value in final.items()}
    best["fc3.bias"].fill_(7.0)
    checkpoint = tmp_path / "eat_checkpoint.pt"
    torch.save({"model_state_dict": final, "best_model_state_dict": best}, checkpoint)
    loaded = LeNet5()
    load_checkpoint(loaded, checkpoint)
    assert torch.equal(loaded.fc3.bias, best["fc3.bias"])
from neurocim_opt.paths import data_root


def _dataset() -> Path:
    root = data_root("mnist")
    if not (root / "MNIST/raw/t10k-images-idx3-ubyte").is_file():
        pytest.skip("set NEUROCIM_MNIST_ROOT to run real inference cache tests")
    return root


def test_cached_batch_predictions_match_and_cache_is_observable() -> None:
    dataset = _dataset()
    clear_lenet_inference_cache()
    cold = PlatformService.infer_lenet5_mnist_batch(
        PROJECT_ROOT, (0,), dataset_root=dataset, device="cpu"
    )
    hot = PlatformService.infer_lenet5_mnist_batch(
        PROJECT_ROOT, (0,), dataset_root=dataset, device="cpu"
    )
    batch = PlatformService.infer_lenet5_mnist_batch(
        PROJECT_ROOT, tuple(range(10)), dataset_root=dataset, device="cpu"
    )
    assert cold.cache_hit is False
    assert hot.cache_hit is True and batch.cache_hit is True
    assert [(p.predicted_labels, p.confidences) for p in cold.predictions] == [
        (p.predicted_labels, p.confidences) for p in hot.predictions
    ]
    assert all(len(p.predicted_labels) == 10 for p in batch.predictions)
    info = lenet_inference_cache_info()
    assert (info.hits, info.misses, info.entries) == (2, 1, 1)


def test_checkpoint_mtime_invalidates_and_missing_resource_evicts(tmp_path: Path) -> None:
    dataset = _dataset()
    model_root = PROJECT_ROOT / "3_models/LeNet_5"
    checkpoint = tmp_path / "lenet5_float.pt"
    shutil.copy2(model_root / "chekpoint/lenet5_float.pt", checkpoint)
    quantization = model_root / "chekpoint/quantization.json"
    error_model = ROOT / "outputs/error_models/01_NMOS/error_model.json"
    kwargs = dict(
        model_root=model_root,
        dataset_root=dataset,
        checkpoint_path=checkpoint,
        quantization_path=quantization,
        error_model_path=error_model,
        device_name="cpu",
    )
    clear_lenet_inference_cache()
    first, first_hit = get_lenet_inference_resources(**kwargs)
    old = checkpoint.stat()
    os.utime(checkpoint, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000_000))
    second, second_hit = get_lenet_inference_resources(**kwargs)
    assert first_hit is False and second_hit is False
    assert second is not first
    assert lenet_inference_cache_info().invalidations == 1
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint.*missing"):
        get_lenet_inference_resources(**kwargs)
    info = lenet_inference_cache_info()
    assert info.entries == 0 and info.invalidations == 2


def test_batch_rejects_empty_or_out_of_range_indices() -> None:
    with pytest.raises(ValueError, match="at least one"):
        PlatformService.infer_lenet5_mnist_batch(PROJECT_ROOT, ())
    with pytest.raises(IndexError, match=r"\[0, 9999\]"):
        PlatformService.infer_lenet5_mnist_batch(PROJECT_ROOT, (10_000,))

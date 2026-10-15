from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT / "3_models/TinyTransformer"))


def _resources() -> tuple[Path, Path, Path]:
    checkpoint = PROJECT / "3_models/TinyTransformer/checkpoint/cct_7_3x1_32_cifar10.pth"
    scales = PROJECT / "3_models/TinyTransformer/checkpoint/input_scales.json"
    error = ROOT / "outputs/error_models/03_AOS_1/error_model.json"
    if not all(path.is_file() for path in (checkpoint, scales, error)):
        pytest.skip("TinyTransformer integration resources are not installed")
    return checkpoint, scales, error


def test_official_checkpoint_and_forward() -> None:
    import torch
    from tiny_transformer import cct_7_3x1_32

    checkpoint, _scales, _error = _resources()
    model = cct_7_3x1_32()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_760_139
    assert model(torch.zeros(1, 3, 32, 32)).shape == (1, 10)


def test_real_signed_differential_single_inference() -> None:
    import torch
    from neurocim_opt.application import PlatformService

    checkpoint, scales, error = _resources()
    dataset = Path(r"E:\datasets")
    if not (dataset / "cifar-10-batches-py/test_batch").is_file():
        pytest.skip("CIFAR-10 is not installed")
    outcome = PlatformService.infer_registered_single(
        PROJECT, "tinytransformer-cifar10", 0, dataset_root=dataset,
        checkpoint_path=checkpoint, auxiliary_path=scales,
        error_model_path=error, device="cuda" if torch.cuda.is_available() else "cpu",
    )
    assert outcome.input_shape == (3, 32, 32)
    assert len(outcome.predictions) == 3
    assert all(len(prediction.top5) == 5 for prediction in outcome.predictions)
    assert outcome.predictions[0].predicted_label == outcome.true_label


def test_finetuning_converts_only_two_heads() -> None:
    import json
    import numpy as np
    import torch
    from tiny_transformer import cct_7_3x1_32
    from tiny_transformer.engine import differential_head_copy
    from neurocim_opt.layers import DifferentialLinear

    checkpoint, scales_path, _error = _resources()
    model = cct_7_3x1_32()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
    scales = json.loads(scales_path.read_text(encoding="utf-8"))
    codes = np.arange(256, dtype=np.int64)
    converted = differential_head_copy(model, codes[:, None] * codes[None, :], scales, ste=True)
    names = {name for name, layer in converted.named_modules() if isinstance(layer, DifferentialLinear)}
    assert names == {"classifier.attention_pool", "classifier.fc"}

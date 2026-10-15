from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurocim_opt.batch_inference import (
    BatchInferenceCancelled,
    BatchInferenceRequest,
    run_grouped_batch_inference,
)


def _request(tmp_path: Path, **changes) -> BatchInferenceRequest:
    raw = tmp_path / "data/MNIST/raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "t10k-images-idx3-ubyte").write_bytes(b"images")
    (raw / "t10k-labels-idx1-ubyte").write_bytes(b"labels")
    for name in ("checkpoint.pt", "quantization.json", "error_model.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    values = dict(
        project_root=tmp_path,
        dataset_root=tmp_path / "data",
        checkpoint_path=tmp_path / "checkpoint.pt",
        quantization_path=tmp_path / "quantization.json",
        error_model_path=tmp_path / "error_model.json",
        sample_count=5,
        group_size=2,
        inference_batch_size=2,
        output_path=tmp_path / "result.json",
    )
    values.update(changes)
    return BatchInferenceRequest(**values)


def _fake_resources(monkeypatch) -> None:
    import torch
    from neurocim_opt import inference_runtime

    labels = [0, 1, 0, 1, 1]
    dataset = [(torch.tensor([float(label)]), label) for label in labels]

    class Model:
        def __init__(self, wrong: set[int] = set()):
            self.offset = 0
            self.wrong = wrong

        def __call__(self, inputs):
            rows = []
            for value in inputs[:, 0].long().tolist():
                index = self.offset
                self.offset += 1
                predicted = 1 - value if index in self.wrong else value
                rows.append([5.0, 0.0] if predicted == 0 else [0.0, 5.0])
            return torch.tensor(rows)

    resources = inference_runtime.LeNetInferenceResources(
        dataset=dataset,
        quantization={},
        device_lut=None,
        float_model=Model(),
        ideal_model=Model({1}),
        device_model=Model({0, 4}),
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(
        inference_runtime, "get_registered_inference_resources", lambda **kwargs: (resources, True)
    )


def test_grouped_accuracy_progress_and_json(tmp_path: Path, monkeypatch) -> None:
    _fake_resources(monkeypatch)
    updates = []
    outcome = run_grouped_batch_inference(
        _request(tmp_path), progress=lambda done, total: updates.append((done, total))
    )
    assert outcome.modes["float"].accuracy == 1.0
    assert outcome.modes["ideal_differential"].accuracy == pytest.approx(0.8)
    assert outcome.modes["device_differential"].accuracy == pytest.approx(0.6)
    assert [group.sample_count for group in outcome.groups] == [2, 2, 1]
    assert [group.accuracy["device_differential"] for group in outcome.groups] == [0.5, 1.0, 0.0]
    assert updates == [(0, 5), (2, 5), (4, 5), (5, 5)]
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert payload["sample_count"] == 5
    assert payload["groups"][-1]["end_index"] == 4


def test_validation_and_cancellation(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="sample_count"):
        run_grouped_batch_inference(_request(tmp_path, sample_count=0))
    _fake_resources(monkeypatch)
    with pytest.raises(BatchInferenceCancelled, match="after 0"):
        run_grouped_batch_inference(_request(tmp_path), cancelled=lambda: True)


def test_rejects_unknown_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ResNet-20/CIFAR-10"):
        run_grouped_batch_inference(_request(tmp_path, model_id="unknown"))


def test_resnet_cifar_grouped_inference(tmp_path: Path, monkeypatch) -> None:
    from neurocim_opt.inference_runtime import cifar10_required_files

    request = _request(tmp_path, model_id="resnet20-cifar10")
    for path in cifar10_required_files(request.dataset_root):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    _fake_resources(monkeypatch)
    outcome = run_grouped_batch_inference(request)
    assert outcome.model_id == "resnet20-cifar10"
    assert outcome.dataset == "CIFAR-10/test"
    assert [group.sample_count for group in outcome.groups] == [2, 2, 1]


def test_bnn_mnist_grouped_inference_without_auxiliary_file(tmp_path: Path, monkeypatch) -> None:
    from neurocim_opt.inference_runtime import mnist_required_files

    request = _request(
        tmp_path, model_id="bnn-mnist", quantization_path=tmp_path / "not-required.json"
    )
    for path in mnist_required_files(request.dataset_root, include_training=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    _fake_resources(monkeypatch)
    outcome = run_grouped_batch_inference(request)
    assert outcome.model_id == "bnn-mnist"
    assert outcome.dataset == "MNIST/test"
    assert outcome.modes["device_differential"].accuracy == pytest.approx(0.6)

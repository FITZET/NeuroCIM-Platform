from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_gate_requires_all_vision_zoo_checkpoints() -> None:
    verifier = _load_script("verify_release_resources.py")
    required = {
        "3_models/VisionZoo/checkpoint/resnet8_mnist.pt",
        "3_models/VisionZoo/checkpoint/resnet8-mnist_input_scales.json",
        "3_models/VisionZoo/checkpoint/cifar10_mobilenetv2_x1_0-fe6a5b48.pt",
        "3_models/VisionZoo/checkpoint/mobilenetv2-cifar10_input_scales.json",
        "3_models/VisionZoo/checkpoint/cifar10_vgg11_bn-eaeebf42.pt",
        "3_models/VisionZoo/checkpoint/vgg11-bn-cifar10_input_scales.json",
        "3_models/VisionZoo/checkpoint/squeezenet1_1_cifar10.pt",
        "3_models/VisionZoo/checkpoint/squeezenet1_1-cifar10_input_scales.json",
    }
    assert required <= set(verifier.MODEL_FILES)
    assert "3_models/VisionZoo/checkpoint" in _load_script(
        "create_runtime_assets.py"
    ).DIRECTORIES


def test_vision_zoo_release_resources_match_provenance_and_manifest() -> None:
    provenance = json.loads(
        (PROJECT / "3_models/VisionZoo/provenance.json").read_text(encoding="utf-8")
    )["weights"]
    manifest = {
        item["model_id"]: item
        for item in json.loads((ROOT / "model_resource_manifest.json").read_text(encoding="utf-8"))["models"]
    }
    for model_id in (
        "resnet8-mnist", "mobilenetv2-cifar10", "vgg11-bn-cifar10",
        "squeezenet1_1-cifar10",
    ):
        path = PROJECT / "3_models/VisionZoo/checkpoint" / provenance[model_id]["filename"]
        assert path.is_file() and path.stat().st_size > 0
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        assert digest == provenance[model_id]["sha256"] == manifest[model_id]["sha256"]
        assert (PROJECT / manifest[model_id]["checkpoint"]).resolve() == path.resolve()


def test_pyinstaller_and_release_script_include_vision_zoo() -> None:
    spec = (ROOT / "packaging/neurocim.spec").read_text(encoding="utf-8")
    build = (ROOT / "scripts/build_release.ps1").read_text(encoding="utf-8")
    assert 'PROJECT / "3_models" / "VisionZoo"' in spec
    assert '"vision_zoo"' in spec
    assert "3_models\\VisionZoo\\checkpoint" in build
    assert "model_resource_manifest.json" in build
    assert "LICENSE-chenyaofo.txt" in build
    assert "LICENSE-torchvision.txt" in build


def test_release_verifier_rejects_vision_zoo_hash_mismatch(tmp_path: Path) -> None:
    verifier = _load_script("verify_release_resources.py")
    rows = []
    for model_id in sorted(verifier.VISION_ZOO_IDS):
        checkpoint = tmp_path / "models" / f"{model_id}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(model_id.encode("utf-8"))
        auxiliary = tmp_path / "models" / f"{model_id}.json"
        auxiliary.write_text("{}", encoding="utf-8")
        with checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        rows.append({
            "model_id": model_id,
            "checkpoint": checkpoint.relative_to(tmp_path).as_posix(),
            "auxiliary": [auxiliary.relative_to(tmp_path).as_posix()],
            "sha256": digest,
        })
    (tmp_path / "model_resource_manifest.json").write_text(
        json.dumps({"models": rows}), encoding="utf-8"
    )
    assert verifier.verify_vision_zoo(tmp_path) == 4
    Path(tmp_path / rows[0]["checkpoint"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verifier.verify_vision_zoo(tmp_path)


def test_release_verifier_checks_every_model_checkpoint(tmp_path: Path) -> None:
    verifier = _load_script("verify_release_resources.py")
    rows = []
    for index in range(10):
        checkpoint = tmp_path / "models" / f"model-{index}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{index}".encode())
        rows.append({
            "model_id": f"model-{index}",
            "checkpoint": checkpoint.relative_to(tmp_path).as_posix(),
            "auxiliary": [],
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        })
    (tmp_path / "model_resource_manifest.json").write_text(
        json.dumps({"models": rows}), encoding="utf-8"
    )
    assert verifier.verify_all_model_hashes(tmp_path) == 10
    (tmp_path / rows[-1]["checkpoint"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verifier.verify_all_model_hashes(tmp_path)

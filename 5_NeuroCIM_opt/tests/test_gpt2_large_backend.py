from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F


PLATFORM = Path(__file__).resolve().parents[1]
PROJECT = PLATFORM.parent
GPT_ROOT = PROJECT / "3_models/GPT2"
for path in (PLATFORM, GPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gpt2_cim import GPT, GPTConfig, get_gpt2_spec
from gpt2_cim.engine import differential_layers_for
from neurocim_opt.gpt2_evaluation import (
    WikiText2EvaluationRequest, WikiText2EvaluationService,
    load_wikitext2_raw, resolve_wikitext2_raw_file, sliding_window_nll,
)
from neurocim_opt.gpt2_resources import (
    GPT2_LARGE_SHA256, GPT2_LARGE_SIZE, inspect_gpt2_resources,
    inspect_safetensors_layout, load_checkpoint_into_model,
)
from neurocim_opt.gpt2_service import (
    GPT2GenerationRequest, GPT2GenerationService, _validate_generation_context,
)
from scripts.manage_gpt2_large_resources import assemble_parts, split_file
import scripts.verify_release_resources as release_resources


def test_pinned_gpt2_specs_do_not_construct_large_model() -> None:
    small = get_gpt2_spec("gpt2-small")
    large = get_gpt2_spec("gpt2-large")
    assert (small.n_layer, small.n_head, small.n_embd) == (12, 12, 768)
    assert (large.n_layer, large.n_head, large.n_embd) == (36, 20, 1280)
    assert large.huggingface_id == "openai-community/gpt2-large"
    assert differential_layers_for("gpt2-small") == (
        "transformer.h.11.attn.c_proj", "transformer.h.11.mlp.c_proj",
    )
    assert differential_layers_for("gpt2-large") == (
        "transformer.h.35.attn.c_proj", "transformer.h.35.mlp.c_proj",
    )
    with pytest.raises(ValueError, match="Unsupported"):
        get_gpt2_spec("gpt2-xl")


def test_resource_manifest_is_pinned_and_explicit_about_limited_injection() -> None:
    manifest = json.loads(
        (PLATFORM / "packaging/gpt2-large-resources.json").read_text(encoding="utf-8")
    )
    model = manifest["model"]
    checkpoint = next(item for item in model["files"] if item["path"] == "model.safetensors")
    assert len(model["revision"]) == 40
    assert checkpoint == {
        "path": "model.safetensors", "size": GPT2_LARGE_SIZE,
        "sha256": GPT2_LARGE_SHA256,
    }
    assert manifest["dataset"]["config"] == "wikitext-2-raw-v1"
    assert manifest["runtime"]["batch_size"] == 1
    assert "last Transformer block" in manifest["runtime"]["injection_scope"]


def test_missing_large_checkpoint_is_reported_not_claimed_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "neurocim_opt.gpt2_resources.available_ram_bytes", lambda: 16 * 1024**3,
    )
    report = inspect_gpt2_resources(
        "gpt2-large", tmp_path / "model.safetensors", device="cpu", verify_hash=True,
    )
    assert not report.ready
    assert not report.checkpoint_exists
    assert report.expected_size == GPT2_LARGE_SIZE
    assert report.required_ram_bytes == 8 * 1024**3
    assert report.required_cuda_bytes is None
    assert any("missing" in problem for problem in report.problems)


def test_large_generation_allows_runtime_scale_calibration(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"fixture")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    error_model = tmp_path / "error.json"
    error_model.write_text("{}", encoding="utf-8")
    missing_scales = tmp_path / "input_scales.json"
    request = GPT2GenerationRequest(
        project_root=PROJECT, prompt="test", checkpoint_path=checkpoint,
        tokenizer_path=tokenizer, input_scales_path=missing_scales,
        error_model_path=error_model, model_variant="gpt2-large",
    )
    resolved = GPT2GenerationService.validate(request)
    assert resolved[2] == missing_scales.resolve()
    with pytest.raises(FileNotFoundError, match="input scales"):
        GPT2GenerationService.validate(
            GPT2GenerationRequest(
                project_root=PROJECT, prompt="test", checkpoint_path=checkpoint,
                tokenizer_path=tokenizer, input_scales_path=missing_scales,
                error_model_path=error_model, model_variant="gpt2-small",
            )
        )


def test_large_differential_generation_has_conservative_context_guard() -> None:
    _validate_generation_context("gpt2-large", 64, 64)
    with pytest.raises(ValueError, match="int64 LUT"):
        _validate_generation_context("gpt2-large", 65, 64)
    _validate_generation_context("gpt2-small", 900, 124)


def test_wikitext_raw_loader_resolves_offline_split_names(tmp_path: Path) -> None:
    root = tmp_path / "wikitext-2-raw-v1"
    root.mkdir()
    expected = root / "wiki.valid.raw"
    expected.write_text("first\nsecond\n", encoding="utf-8")
    assert resolve_wikitext2_raw_file(root, "valid") == expected.resolve()
    path, text = load_wikitext2_raw(root, "validation")
    assert path == expected.resolve()
    assert text == "first\nsecond\n"
    with pytest.raises(FileNotFoundError, match="wiki.test.raw"):
        load_wikitext2_raw(root, "test")


class _TinyBigram(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("table", torch.tensor([
            [4.0, 1.0, 0.0, -1.0],
            [0.0, 4.0, 1.0, -1.0],
            [-1.0, 0.0, 4.0, 1.0],
            [1.0, -1.0, 0.0, 4.0],
        ]))

    def forward(self, inputs: torch.Tensor, _targets: torch.Tensor):
        return self.table[inputs], None


def test_sliding_window_nll_counts_each_target_exactly_once() -> None:
    model = _TinyBigram()
    tokens = [0, 1, 2, 3, 0, 2, 1, 3, 2, 0, 1]
    full_logits = model.table[torch.tensor(tokens[:-1])]
    expected = F.cross_entropy(
        full_logits, torch.tensor(tokens[1:]), reduction="sum",
    ).item()
    actual, count, windows = sliding_window_nll(
        model, tokens, device=torch.device("cpu"), context_length=5, stride=3,
    )
    assert actual == pytest.approx(expected, rel=1e-6)
    assert count == len(tokens) - 1
    assert windows == 3


def test_wikitext_service_reports_float_scope_and_reproducibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gpt2_cim
    import transformers
    import neurocim_opt.gpt2_evaluation as evaluation

    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixed checkpoint fixture")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    dataset = tmp_path / "wiki.test.raw"
    dataset.write_text("fixture corpus", encoding="utf-8")

    class _Tokenizer:
        @staticmethod
        def encode(_text: str) -> list[int]:
            return [0, 1, 2, 3, 0, 2, 1, 3]

    monkeypatch.setattr(
        transformers.GPT2Tokenizer, "from_pretrained", lambda _path: _Tokenizer(),
    )
    monkeypatch.setattr(
        gpt2_cim, "gpt2_for_variant", lambda _variant, **_kwargs: _TinyBigram(),
    )
    monkeypatch.setattr(evaluation, "load_checkpoint_into_model", lambda _model, _path: None)
    monkeypatch.setattr(evaluation, "inspect_gpt2_resources", lambda *_args, **_kwargs: SimpleNamespace(
        ready=True, problems=(), source_revision="fixture-revision",
        actual_sha256=None,
        differential_layers=("transformer.h.35.attn.c_proj", "transformer.h.35.mlp.c_proj"),
    ))
    outcome = WikiText2EvaluationService().run(WikiText2EvaluationRequest(
        project_root=PROJECT, model_variant="gpt2-large", checkpoint_path=checkpoint,
        tokenizer_path=tokenizer, dataset_path=dataset, split="test", device="cpu",
        context_length=5, stride=3, batch_size=1,
    ))
    assert outcome.predicted_token_count == 7
    assert outcome.window_count == 2
    assert outcome.loss > 0 and outcome.perplexity > 1
    assert outcome.injection_scope == "none (float baseline)"
    assert outcome.batch_size == 1
    assert outcome.checkpoint_sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert outcome.dataset_sha256 == hashlib.sha256(dataset.read_bytes()).hexdigest()


def test_large_safetensors_loader_transposes_conv1d_weights(tmp_path: Path) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    source = GPT(GPTConfig(
        block_size=8, vocab_size=16, n_layer=1, n_head=2, n_embd=8,
        dropout=0.0, bias=True,
    ))
    native = source.state_dict()
    hf_state = {}
    transposed = (
        "attn.c_attn.weight", "attn.c_proj.weight",
        "mlp.c_fc.weight", "mlp.c_proj.weight",
    )
    for name, value in native.items():
        if name.endswith(".attn.bias") or name == "lm_head.weight":
            continue
        source_name = name.removeprefix("transformer.")
        hf_state[source_name] = (
            value.t().contiguous() if name.endswith(transposed) else value.clone()
        )
    checkpoint = tmp_path / "tiny.safetensors"
    safetensors.save_file(hf_state, checkpoint)
    target = GPT(source.config)
    load_checkpoint_into_model(target, checkpoint)
    for name, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[name], value), name


def test_real_large_safetensors_header_maps_to_nanogpt_names() -> None:
    checkpoint = PROJECT / "3_models/GPT2/checkpoint/gpt2-large/model.safetensors"
    if not checkpoint.is_file() or checkpoint.stat().st_size != GPT2_LARGE_SIZE:
        pytest.skip("Pinned GPT-2-large checkpoint is not installed")
    mapped = set(inspect_safetensors_layout(checkpoint))
    assert "transformer.h.0.attn.c_attn.bias" in mapped
    assert "transformer.h.35.mlp.c_proj.weight" in mapped
    assert "transformer.wte.weight" in mapped
    assert "transformer.wpe.weight" in mapped
    assert "transformer.ln_f.weight" in mapped
    assert all(not name.startswith("h.") for name in mapped)


def test_split_and_reassemble_verifies_every_part(tmp_path: Path) -> None:
    source = tmp_path / "model.safetensors"
    source.write_bytes(bytes(range(251)) * 41)
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = split_file(source, tmp_path / "parts", part_size=1024)
    output = assemble_parts(manifest, tmp_path / "restored/model.safetensors")
    assert hashlib.sha256(output.read_bytes()).hexdigest() == expected
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    (manifest.parent / metadata["parts"][0]["name"]).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="Invalid GPT-2-large part"):
        assemble_parts(manifest, tmp_path / "broken/model.safetensors")


def test_release_verifier_requires_reassembled_large_model_and_raw_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "release"
    resource_root = root / "3_models/GPT2/checkpoint/gpt2-large"
    dataset_root = resource_root / "wikitext-2-raw-v1"
    dataset_root.mkdir(parents=True)
    model_bytes = b"reassembled model fixture"
    raw_bytes = b"offline WikiText fixture\n"
    (resource_root / "model.safetensors").write_bytes(model_bytes)
    (dataset_root / "wiki.test.raw").write_bytes(raw_bytes)
    manifest = {
        "model": {"files": [{
            "path": "model.safetensors", "size": len(model_bytes),
            "sha256": hashlib.sha256(model_bytes).hexdigest(),
        }]},
        "dataset": {"files": [{
            "raw_name": "wiki.test.raw", "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        }]},
    }
    manifest_path = tmp_path / "resources.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(release_resources, "GPT2_LARGE_MANIFEST", manifest_path)
    assert release_resources.verify_gpt2_large(root) == 2
    (resource_root / "model.safetensors").unlink()
    (resource_root / "model.safetensors.partial").write_bytes(model_bytes)
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        release_resources.verify_gpt2_large(root)

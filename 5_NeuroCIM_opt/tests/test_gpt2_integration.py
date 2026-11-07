from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT / "3_models/GPT2"))


def _resources() -> tuple[Path, Path, Path, Path]:
    checkpoint = PROJECT / "3_models/GPT2/checkpoint/gpt2_nanogpt.pt"
    tokenizer = PROJECT / "3_models/GPT2/checkpoint/tokenizer"
    scales = PROJECT / "3_models/GPT2/checkpoint/input_scales.json"
    error = ROOT / "outputs/error_models/03_UOS_1/error_model.json"
    if not checkpoint.is_file() or not tokenizer.is_dir() or not scales.is_file() or not error.is_file():
        pytest.skip("GPT-2 integration resources are not installed")
    return checkpoint, tokenizer, scales, error


def test_native_checkpoint_has_official_gpt2_shape() -> None:
    from gpt2_cim import gpt2_small

    checkpoint, _tokenizer, _scales, _error = _resources()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = gpt2_small()
    model.load_state_dict(payload["model_state_dict"])
    assert sum(parameter.numel() for parameter in model.parameters()) == 124_439_808
    assert model.config.n_layer == 12 and model.config.n_head == 12 and model.config.n_embd == 768


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real GPT-2 differential smoke requires CUDA")
def test_real_three_mode_generation() -> None:
    from neurocim_opt.gpt2_service import GPT2GenerationRequest, GPT2GenerationService

    checkpoint, tokenizer, scales, error = _resources()
    outcome = GPT2GenerationService().run(GPT2GenerationRequest(
        project_root=PROJECT, prompt="Computing in memory is",
        checkpoint_path=checkpoint, tokenizer_path=tokenizer,
        input_scales_path=scales, error_model_path=error,
        max_new_tokens=2, device="cuda",
    ))
    assert [mode.mode for mode in outcome.modes] == [
        "float", "ideal_differential", "device_differential",
    ]
    assert all(len(mode.generated_token_ids) == 2 for mode in outcome.modes)
    assert all(len(mode.final_top5) == 5 for mode in outcome.modes)
    assert outcome.differential_layers == (
        "transformer.h.11.attn.c_proj", "transformer.h.11.mlp.c_proj",
    )

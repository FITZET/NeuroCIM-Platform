from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable

import numpy as np
import torch

from .error_model import HardwareErrorModel


ProgressCallback = Callable[[int, str], None]
CancelCallback = Callable[[], bool]


class GPT2GenerationCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class GPT2GenerationRequest:
    project_root: Path
    prompt: str
    checkpoint_path: Path
    tokenizer_path: Path
    input_scales_path: Path
    error_model_path: Path
    max_new_tokens: int = 32
    error_strength: float = 1.0
    device: str = "auto"


@dataclass(frozen=True)
class GPT2ModeResult:
    mode: str
    text: str
    generated_token_ids: tuple[int, ...]
    elapsed_ms: float
    tokens_per_second: float
    final_top5: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class GPT2GenerationOutcome:
    prompt: str
    prompt_token_count: int
    device: str
    differential_layers: tuple[str, ...]
    modes: tuple[GPT2ModeResult, ...]
    total_elapsed_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


def _resolve_device(value: str) -> torch.device:
    if value not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but is unavailable")
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else
                        "cpu" if value == "auto" else value)


class GPT2GenerationService:
    """Deterministic GPT-2 text generation in float, ideal and device modes."""

    @staticmethod
    def validate(request: GPT2GenerationRequest) -> tuple[Path, Path, Path, Path]:
        if not request.prompt.strip():
            raise ValueError("Prompt must not be empty")
        if not 1 <= request.max_new_tokens <= 128:
            raise ValueError("max_new_tokens must be in [1, 128]")
        if not 0.0 <= request.error_strength <= 1.0:
            raise ValueError("error_strength must be in [0, 1]")
        paths = tuple(Path(path).expanduser().resolve() for path in (
            request.checkpoint_path, request.tokenizer_path,
            request.input_scales_path, request.error_model_path,
        ))
        purposes = ("GPT-2 checkpoint", "GPT-2 tokenizer directory",
                    "GPT-2 input scales", "device error model")
        missing = [f"{purpose}: {path}" for purpose, path in zip(purposes, paths)
                   if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing GPT-2 resources:\n" + "\n".join(missing))
        return paths

    def run(
        self, request: GPT2GenerationRequest, *, progress: ProgressCallback | None = None,
        cancelled: CancelCallback | None = None,
    ) -> GPT2GenerationOutcome:
        checkpoint, tokenizer_path, scales_path, error_path = self.validate(request)
        notify = progress or (lambda _percent, _message: None)
        root = Path(request.project_root).expanduser().resolve()
        model_root = root / "3_models/GPT2"
        if str(model_root) not in sys.path:
            sys.path.insert(0, str(model_root))
        from transformers import GPT2Tokenizer
        from gpt2_cim import gpt2_small
        from gpt2_cim.engine import DIFFERENTIAL_LAYERS, configure_differential_heads

        device = _resolve_device(request.device)
        tokenizer = GPT2Tokenizer.from_pretrained(str(tokenizer_path))
        prompt_ids = tokenizer.encode(request.prompt)
        if not prompt_ids:
            raise ValueError("Prompt produced no GPT-2 tokens")
        if len(prompt_ids) + request.max_new_tokens > 1024:
            raise ValueError("Prompt plus generated tokens exceeds GPT-2 context length 1024")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("best_model_state_dict", payload.get("model_state_dict", payload))
        scales = json.loads(scales_path.read_text(encoding="utf-8"))
        error_model = HardwareErrorModel.load(error_path)
        codes = np.arange(256, dtype=np.int64)
        ideal_lut = codes[:, None] * codes[None, :]
        device_lut = error_model.effective_lut(request.error_strength)
        started = perf_counter()
        results: list[GPT2ModeResult] = []
        modes = (("float", None), ("ideal_differential", ideal_lut),
                 ("device_differential", device_lut))
        for mode_index, (mode, lut) in enumerate(modes):
            if cancelled and cancelled():
                raise GPT2GenerationCancelled("GPT-2 generation cancelled")
            notify(round(mode_index * 100 / len(modes)), f"Preparing {mode}")
            model = gpt2_small()
            model.load_state_dict(state)
            if lut is not None:
                configure_differential_heads(model, lut, scales, ste=False)
            result = self._generate(
                model, tokenizer, prompt_ids, request.max_new_tokens, device,
                mode, mode_index, len(modes), notify, cancelled,
            )
            results.append(result)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        notify(100, "GPT-2 generation completed")
        return GPT2GenerationOutcome(
            request.prompt, len(prompt_ids), str(device), tuple(DIFFERENTIAL_LAYERS),
            tuple(results), (perf_counter() - started) * 1000.0,
        )

    @staticmethod
    def _generate(model, tokenizer, prompt_ids: list[int], max_new_tokens: int,
                  device: torch.device, mode: str, mode_index: int, mode_count: int,
                  notify: ProgressCallback, cancelled: CancelCallback | None) -> GPT2ModeResult:
        model.eval().to(device)
        tokens = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        generated: list[int] = []
        final_top5: tuple[tuple[str, float], ...] = ()
        started = perf_counter()
        with torch.inference_mode():
            for step in range(max_new_tokens):
                if cancelled and cancelled():
                    raise GPT2GenerationCancelled("GPT-2 generation cancelled")
                logits, _loss = model(tokens[:, -model.config.block_size:])
                probabilities = torch.softmax(logits[:, -1, :], dim=-1)
                values, indices = probabilities.topk(5, dim=-1)
                next_token = int(indices[0, 0])
                generated.append(next_token)
                tokens = torch.cat((tokens, torch.tensor([[next_token]], device=device)), dim=1)
                final_top5 = tuple(
                    (tokenizer.decode([int(index)]), float(value))
                    for index, value in zip(indices[0].cpu(), values[0].cpu())
                )
                fraction = (mode_index + (step + 1) / max_new_tokens) / mode_count
                notify(round(fraction * 100), f"{mode}: {step + 1}/{max_new_tokens} tokens")
        elapsed = (perf_counter() - started) * 1000.0
        return GPT2ModeResult(
            mode=mode, text=tokenizer.decode(prompt_ids + generated),
            generated_token_ids=tuple(generated), elapsed_ms=elapsed,
            tokens_per_second=max_new_tokens / max(elapsed / 1000.0, 1e-12),
            final_top5=final_top5,
        )

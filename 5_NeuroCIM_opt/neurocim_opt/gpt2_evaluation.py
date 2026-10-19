from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys
from typing import Callable

import torch
from torch.nn import functional as F

from .gpt2_resources import (
    WIKITEXT2_REVISION, inspect_gpt2_resources,
    load_checkpoint_into_model, sha256_file,
)


ProgressCallback = Callable[[int, str], None]
CancelCallback = Callable[[], bool]


class WikiText2EvaluationCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class WikiText2EvaluationRequest:
    project_root: Path
    model_variant: str
    checkpoint_path: Path
    tokenizer_path: Path
    dataset_path: Path
    split: str = "test"
    device: str = "auto"
    stride: int = 512
    context_length: int = 1024
    batch_size: int = 1
    max_eval_tokens: int | None = None
    expected_dataset_sha256: str | None = None


@dataclass(frozen=True)
class WikiText2EvaluationOutcome:
    model_variant: str
    split: str
    loss: float
    perplexity: float
    predicted_token_count: int
    window_count: int
    truncated: bool
    device: str
    batch_size: int
    stride: int
    context_length: int
    model_revision: str
    checkpoint_sha256: str
    dataset_revision: str
    dataset_sha256: str
    differential_layers: tuple[str, ...]
    injection_scope: str

    def as_dict(self) -> dict:
        return asdict(self)


_SPLIT_FILENAMES = {
    "train": ("wiki.train.raw", "train.txt"),
    "validation": ("wiki.valid.raw", "validation.txt", "valid.txt"),
    "test": ("wiki.test.raw", "test.txt"),
}


def resolve_wikitext2_raw_file(path: str | Path, split: str) -> Path:
    if split == "valid":
        split = "validation"
    if split not in _SPLIT_FILENAMES:
        raise ValueError("split must be train, validation, or test")
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        for name in _SPLIT_FILENAMES[split]:
            resolved = candidate / name
            if resolved.is_file():
                return resolved
    expected = ", ".join(_SPLIT_FILENAMES[split])
    raise FileNotFoundError(
        f"WikiText-2 raw {split} split is missing under {candidate}; expected {expected}"
    )


def load_wikitext2_raw(path: str | Path, split: str = "test") -> tuple[Path, str]:
    resolved = resolve_wikitext2_raw_file(path, split)
    return resolved, resolved.read_text(encoding="utf-8")


def sliding_window_nll(
    model: torch.nn.Module, token_ids: list[int] | tuple[int, ...], *,
    device: torch.device, context_length: int = 1024, stride: int = 512,
    progress: ProgressCallback | None = None, cancelled: CancelCallback | None = None,
) -> tuple[float, int, int]:
    """Return summed causal NLL, predicted-token count, and window count.

    Overlapping context is retained, while every target token contributes exactly
    once to the denominator.
    """
    if len(token_ids) < 2:
        raise ValueError("WikiText-2 evaluation requires at least two tokens")
    if context_length < 2:
        raise ValueError("context_length must be at least 2")
    if not 1 <= stride < context_length:
        raise ValueError("stride must be in [1, context_length)")
    notify = progress or (lambda _percent, _message: None)
    total_nll = 0.0
    predicted = 0
    windows = 0
    sequence_length = len(token_ids)
    previous_end = 1
    end = min(context_length, sequence_length)
    model.eval().to(device)
    with torch.inference_mode():
        while True:
            if cancelled and cancelled():
                raise WikiText2EvaluationCancelled("WikiText-2 evaluation cancelled")
            begin = max(0, end - context_length)
            window = torch.tensor(token_ids[begin:end], dtype=torch.long, device=device)
            inputs = window[:-1].unsqueeze(0)
            targets = window[1:].unsqueeze(0)
            result = model(inputs, targets)
            logits = result[0] if isinstance(result, tuple) else result
            target_indices = torch.arange(begin + 1, end, device=device)
            fresh = target_indices >= previous_end
            if fresh.any():
                selected_logits = logits[0, fresh]
                selected_targets = targets[0, fresh]
                total_nll += float(F.cross_entropy(
                    selected_logits, selected_targets, reduction="sum",
                ))
                predicted += int(fresh.sum())
            windows += 1
            previous_end = end
            notify(round(end * 100 / sequence_length), f"WikiText-2: {end}/{sequence_length} tokens")
            if end == sequence_length:
                break
            end = min(end + stride, sequence_length)
    return total_nll, predicted, windows


class WikiText2EvaluationService:
    """Offline WikiText-2 raw-v1 loss/perplexity evaluation, batch size one."""

    def run(
        self, request: WikiText2EvaluationRequest, *,
        progress: ProgressCallback | None = None, cancelled: CancelCallback | None = None,
    ) -> WikiText2EvaluationOutcome:
        if request.batch_size != 1:
            raise ValueError("GPT-2 WikiText-2 evaluation supports batch_size=1 only")
        if request.model_variant not in {"gpt2-small", "gpt2-large"}:
            raise ValueError("model_variant must be gpt2-small or gpt2-large")
        if request.context_length > 1024:
            raise ValueError("GPT-2 context_length cannot exceed 1024")
        if request.max_eval_tokens is not None and request.max_eval_tokens < 2:
            raise ValueError("max_eval_tokens must be at least 2")
        root = Path(request.project_root).expanduser().resolve()
        model_root = root / "3_models/GPT2"
        if str(model_root) not in sys.path:
            sys.path.insert(0, str(model_root))
        from gpt2_cim import gpt2_for_variant
        from transformers import GPT2Tokenizer

        device = torch.device(
            "cuda" if request.device == "auto" and torch.cuda.is_available() else
            "cpu" if request.device == "auto" else request.device
        )
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected but is unavailable")
        report = inspect_gpt2_resources(
            request.model_variant, request.checkpoint_path, device=device.type,
            verify_hash=request.model_variant == "gpt2-large",
        )
        if not report.ready:
            raise RuntimeError(
                f"{request.model_variant} resources are not ready:\n" +
                "\n".join(report.problems)
            )
        tokenizer_root = Path(request.tokenizer_path).expanduser().resolve()
        if not tokenizer_root.is_dir():
            raise FileNotFoundError(f"GPT-2 tokenizer directory is missing: {tokenizer_root}")
        dataset_file, text = load_wikitext2_raw(request.dataset_path, request.split)
        dataset_hash = sha256_file(dataset_file)
        if request.expected_dataset_sha256 and dataset_hash != request.expected_dataset_sha256:
            raise RuntimeError(
                f"WikiText-2 SHA256 mismatch: expected {request.expected_dataset_sha256}, "
                f"got {dataset_hash}"
            )
        tokenizer = GPT2Tokenizer.from_pretrained(str(tokenizer_root))
        token_ids = tokenizer.encode(text)
        truncated = request.max_eval_tokens is not None and len(token_ids) > request.max_eval_tokens
        if request.max_eval_tokens is not None:
            token_ids = token_ids[:request.max_eval_tokens]
        model_dtype = (
            torch.float16
            if request.model_variant == "gpt2-large" and device.type == "cuda"
            else None
        )
        model = gpt2_for_variant(request.model_variant, dtype=model_dtype)
        load_checkpoint_into_model(model, request.checkpoint_path)
        try:
            nll, count, windows = sliding_window_nll(
                model, token_ids, device=device, context_length=request.context_length,
                stride=request.stride, progress=progress, cancelled=cancelled,
            )
        finally:
            model.cpu()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if count <= 0:
            raise RuntimeError("WikiText-2 evaluation produced no predicted tokens")
        loss = nll / count
        perplexity = math.exp(loss)
        if not math.isfinite(loss) or not math.isfinite(perplexity) or perplexity <= 0:
            raise RuntimeError(
                f"WikiText-2 evaluation produced an invalid loss/perplexity: {loss}/{perplexity}"
            )
        return WikiText2EvaluationOutcome(
            model_variant=request.model_variant, split=request.split,
            loss=loss, perplexity=perplexity, predicted_token_count=count,
            window_count=windows, truncated=truncated, device=str(device),
            batch_size=1, stride=request.stride, context_length=request.context_length,
            model_revision=report.source_revision,
            checkpoint_sha256=(
                report.actual_sha256 or sha256_file(request.checkpoint_path)
            ),
            dataset_revision=WIKITEXT2_REVISION, dataset_sha256=dataset_hash,
            differential_layers=report.differential_layers,
            injection_scope="none (float baseline)",
        )

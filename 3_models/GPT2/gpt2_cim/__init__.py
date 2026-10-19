"""Pinned GPT-2 Small/Large architecture factories adapted from nanoGPT."""
from dataclasses import dataclass

import torch

from .model import GPT, GPTConfig


@dataclass(frozen=True)
class GPT2ModelSpec:
    variant: str
    huggingface_id: str
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int


GPT2_MODEL_SPECS = {
    "gpt2-small": GPT2ModelSpec(
        "gpt2-small", "openai-community/gpt2", 1024, 50257, 12, 12, 768,
    ),
    "gpt2-large": GPT2ModelSpec(
        "gpt2-large", "openai-community/gpt2-large", 1024, 50257, 36, 20, 1280,
    ),
}


def get_gpt2_spec(variant: str) -> GPT2ModelSpec:
    try:
        return GPT2_MODEL_SPECS[variant]
    except KeyError as exc:
        raise ValueError(f"Unsupported GPT-2 variant: {variant}") from exc


def gpt2_for_variant(variant: str, *, dtype: torch.dtype | None = None) -> GPT:
    spec = get_gpt2_spec(variant)
    previous_dtype = torch.get_default_dtype()
    try:
        if dtype is not None:
            torch.set_default_dtype(dtype)
        return GPT(GPTConfig(
            block_size=spec.block_size, vocab_size=spec.vocab_size,
            n_layer=spec.n_layer, n_head=spec.n_head, n_embd=spec.n_embd,
            dropout=0.0, bias=True,
        ))
    finally:
        torch.set_default_dtype(previous_dtype)


def gpt2_small(*, dtype: torch.dtype | None = None) -> GPT:
    return gpt2_for_variant("gpt2-small", dtype=dtype)


def gpt2_large(*, dtype: torch.dtype | None = None) -> GPT:
    return gpt2_for_variant("gpt2-large", dtype=dtype)


__all__ = [
    "GPT", "GPTConfig", "GPT2ModelSpec", "GPT2_MODEL_SPECS", "get_gpt2_spec",
    "gpt2_small", "gpt2_large", "gpt2_for_variant",
]

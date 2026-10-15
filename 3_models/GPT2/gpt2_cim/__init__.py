"""GPT-2 Small model adapted from nanoGPT."""
from .model import GPT, GPTConfig


def gpt2_small() -> GPT:
    return GPT(GPTConfig(
        block_size=1024, vocab_size=50257, n_layer=12,
        n_head=12, n_embd=768, dropout=0.0, bias=True,
    ))

__all__ = ["GPT", "GPTConfig", "gpt2_small"]

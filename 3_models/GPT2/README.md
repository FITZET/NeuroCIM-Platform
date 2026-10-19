# GPT-2 Small / nanoGPT

This integration uses the GPT-2 Small architecture from Andrej Karpathy's
nanoGPT and the original OpenAI GPT-2 pretrained weights distributed through
Hugging Face.

- Architecture source: https://github.com/karpathy/nanoGPT
- Upstream license: MIT (`LICENSE`)
- Model: GPT-2 Small, 12 layers, 12 attention heads, 768 hidden dimensions
- Parameters: 124,439,808
- Tokenizer: GPT-2 byte-level BPE

The platform runtime uses a local checkpoint and tokenizer under `checkpoint/`.
The generic nanoGPT training scripts are intentionally not copied; training,
error injection, artifacts and GUI execution are handled by NeuroCIM services.

## GPT-2 Large and WikiText-2

GPT-2 Large uses the same architecture with 36 layers, 20 attention heads and
1280 hidden dimensions. Its official `openai-community/gpt2-large` safetensors
file is pinned to revision `32b71b12589c2f8d625668d2335a01cac3249519`.
The 3.25 GB checkpoint is deliberately not kept in Git. Use
`5_NeuroCIM_opt/scripts/manage_gpt2_large_resources.py` to download, verify,
split or reassemble it from the tracked resource manifest. WikiText-2 raw-v1 is
pinned independently and evaluated offline with batch size one and sliding
causal NLL/perplexity.

NeuroCIM error injection is intentionally limited to the final Transformer
block's `attn.c_proj` and `mlp.c_proj` layers (block 11 for Small and block 35
for Large). Results must report this restricted scope; it is not a whole-model
CIM simulation.

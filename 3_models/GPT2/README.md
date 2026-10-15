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

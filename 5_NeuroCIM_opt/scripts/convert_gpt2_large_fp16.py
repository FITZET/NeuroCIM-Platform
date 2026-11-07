from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a verified FP16 GPT-2 safetensors runtime file")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        parser.error("source and output must be different files")
    source_hash = sha256(source)
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(source, framework="pt", device="cpu") as reader:
        keys = list(reader.keys())
        for index, name in enumerate(keys, 1):
            value = reader.get_tensor(name)
            if value.is_floating_point():
                value = value.to(torch.float16)
            tensors[name] = value.contiguous().clone()
            if index % 24 == 0 or index == len(keys):
                print(f"Converted {index}/{len(keys)} tensors", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    save_file(tensors, temporary, metadata={
        "source_sha256": source_hash,
        "conversion": "floating tensors converted to IEEE FP16",
    })
    temporary.replace(output)
    print(f"source_sha256={source_hash}")
    print(f"runtime_sha256={sha256(output)}")
    print(f"runtime_bytes={output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

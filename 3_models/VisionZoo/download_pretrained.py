from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import urllib.request


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify official CIFAR-10 weights")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoint")
    parser.add_argument(
        "--model", action="append", choices=("mobilenetv2-cifar10", "vgg11-bn-cifar10"),
        help="Download selected model; repeat the option. Defaults to both.",
    )
    args = parser.parse_args()
    from vision_zoo.checkpoint import sha256_file

    metadata = json.loads((ROOT / "provenance.json").read_text(encoding="utf-8"))
    selected = args.model or ["mobilenetv2-cifar10", "vgg11-bn-cifar10"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model_id in selected:
        record = metadata["weights"][model_id]
        target = args.output_dir / record["filename"]
        if target.is_file() and sha256_file(target) == record["sha256"]:
            print(f"Verified existing {target}")
            continue
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.unlink(missing_ok=True)
        try:
            request = urllib.request.Request(record["url"], headers={"User-Agent": "NeuroCIM/1"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            actual = sha256_file(temporary)
            if actual != record["sha256"]:
                raise ValueError(f"SHA256 mismatch for {model_id}: {actual}")
            temporary.replace(target)
            print(f"Downloaded and verified {target}")
        finally:
            temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

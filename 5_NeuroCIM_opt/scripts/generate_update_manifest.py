from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--flavor", required=True, choices=("CPU", "GPU"))
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.artifact_dir.resolve(strict=True)
    prefix = f"NeuroCIM-Setup-{args.flavor}-{args.version}"
    files = sorted(path for path in root.glob(prefix + "*") if path.is_file())
    if not files or not any(path.suffix.lower() == ".exe" for path in files):
        raise FileNotFoundError(f"No installer files matching {prefix} in {root}")
    payload = {
        "schema_version": 1,
        "version": args.version,
        "flavors": {
            args.flavor.lower(): {
                "files": [
                    {"asset": path.name, "size": path.stat().st_size, "sha256": sha256(path)}
                    for path in files
                ]
            }
        },
    }
    destination = root / "update-manifest.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

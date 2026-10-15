from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    merged: dict = {"schema_version": 1, "version": None, "flavors": {}}
    for path in args.inputs:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if merged["version"] is None:
            merged["version"] = value["version"]
        if value["version"] != merged["version"]:
            raise ValueError("Cannot merge manifests from different versions")
        overlap = set(merged["flavors"]).intersection(value["flavors"])
        if overlap:
            raise ValueError(f"Duplicate flavors: {sorted(overlap)}")
        merged["flavors"].update(value["flavors"])
    if set(merged["flavors"]) != {"cpu", "gpu"}:
        raise ValueError("Combined manifest must contain CPU and GPU flavors")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

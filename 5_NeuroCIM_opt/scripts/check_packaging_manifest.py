"""Validate that the release manifest excludes local and generated assets."""
from __future__ import annotations

import json
from pathlib import Path

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PLATFORM_ROOT / "packaging_manifest.json"
REQUIRED_EXCLUSIONS = {
    ".tools", ".venv", "outputs", "output", "data", "datasets",
    "checkpoint", "checkpoints", "chekpoint", "*.pt", "*.pth", "*.ckpt",
}


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    exclusions = "\n".join(manifest.get("exclude", []))
    missing = sorted(item for item in REQUIRED_EXCLUSIONS if item not in exclusions)
    if missing:
        raise SystemExit(f"Packaging manifest is missing exclusions: {', '.join(missing)}")
    if not (PLATFORM_ROOT / "THIRD_PARTY_NOTICES.md").is_file():
        raise SystemExit("Third-party notice is missing")
    print(f"Packaging manifest OK: {len(manifest['exclude'])} exclusion rules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

from manage_gpt2_large_resources import split_file, verify_resources


DIRECTORIES = (
    "3_models/LeNet_5/chekpoint",
    "3_models/BNN/checkpoint",
    "3_models/Resnet20/checkpoint",
    "3_models/Resnet20/output/formal",
    "3_models/TinyTransformer/checkpoint",
    "3_models/VisionZoo/checkpoint",
    "3_models/GPT2/checkpoint",
    "data/mnist",
    "data/cifar10",
)

FILES = (
    "5_NeuroCIM_opt/outputs/neurosim/bin/neurosim.exe",
    "5_NeuroCIM_opt/.tools/mingw/Library/mingw-w64/bin/libgcc_s_seh-1.dll",
    "5_NeuroCIM_opt/.tools/mingw/Library/mingw-w64/bin/libstdc++-6.dll",
    "5_NeuroCIM_opt/.tools/mingw/Library/mingw-w64/bin/libwinpthread-1.dll",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--github-part-size-mib", type=int, default=1900,
        help="Split archives above this size into verified GitHub release assets",
    )
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    destination = args.output.resolve()
    verify_resources(
        root / "5_NeuroCIM_opt/packaging/gpt2-large-resources.json",
        root / "3_models/GPT2/checkpoint/gpt2-large",
        include_dataset=True,
    )
    selected: list[Path] = []
    for relative in DIRECTORIES:
        directory = root / relative
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        selected.extend(path for path in directory.rglob("*") if path.is_file())
    for relative in FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        selected.append(path)
    selected = sorted(set(selected))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
    ) as archive:
        for index, path in enumerate(selected, 1):
            archive.write(path, path.relative_to(root))
            if index % 25 == 0 or index == len(selected):
                print(f"Archived {index}/{len(selected)}", flush=True)
        manifest = {
            "schema_version": 1,
            "file_count": len(selected),
            "purpose": "Authenticated NeuroCIM build resources",
        }
        archive.writestr("runtime-assets-manifest.json", json.dumps(manifest, indent=2))
    temporary.replace(destination)
    print(f"Created {destination} ({destination.stat().st_size} bytes)")
    part_size = args.github_part_size_mib * 1024 * 1024
    if destination.stat().st_size > part_size:
        parts_dir = destination.parent / f"{destination.name}.parts"
        parts_manifest = split_file(destination, parts_dir, part_size)
        destination.unlink()
        print(f"Archive exceeds a GitHub asset; upload files from {parts_dir}")
        print(f"Parts manifest: {parts_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

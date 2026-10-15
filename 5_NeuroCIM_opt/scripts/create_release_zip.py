from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a ZIP64 release archive.")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    destination = args.destination.resolve(strict=False)
    if not source.is_dir():
        raise NotADirectoryError(source)
    if destination == source or source in destination.parents:
        raise ValueError("The destination archive cannot be inside the source directory")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)

    files = sorted(path for path in source.rglob("*") if path.is_file())
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for index, path in enumerate(files, start=1):
                archive.write(path, Path(source.name) / path.relative_to(source))
                if index % 250 == 0 or index == len(files):
                    print(f"Archived {index}/{len(files)} files", flush=True)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Created {destination} ({destination.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

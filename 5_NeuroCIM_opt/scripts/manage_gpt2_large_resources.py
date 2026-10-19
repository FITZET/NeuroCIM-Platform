from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "5_NeuroCIM_opt/packaging/gpt2-large-resources.json"
DEFAULT_DESTINATION = ROOT / "3_models/GPT2/checkpoint/gpt2-large"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def download_resumable(url: str, target: Path, size: int, sha256: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == size and sha256_file(target) == sha256:
        return target
    partial = target.with_name(target.name + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset >= size:
        if offset == size and sha256_file(partial) == sha256:
            partial.replace(target)
            return target
        partial.unlink()
        offset = 0
    failures = 0
    while offset < size:
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={offset}-"} if offset else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset and response.status != 206:
                    partial.unlink(missing_ok=True)
                    offset = 0
                    continue
                with partial.open("ab" if offset else "wb") as output:
                    shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
            offset = partial.stat().st_size
            failures = 0
        except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as exc:
            failures += 1
            offset = partial.stat().st_size if partial.exists() else 0
            if failures >= 10:
                raise RuntimeError(
                    f"Download repeatedly failed for {target.name} at {offset}/{size} bytes"
                ) from exc
            print(
                f"Download interrupted at {offset}/{size}; retrying ({failures}/10)",
                flush=True,
            )
            time.sleep(min(failures * 2, 10))
    if partial.stat().st_size != size:
        raise RuntimeError(f"Incomplete download for {target.name}: {partial.stat().st_size}/{size}")
    actual = sha256_file(partial)
    if actual != sha256:
        raise RuntimeError(f"SHA256 mismatch for {target.name}: expected {sha256}, got {actual}")
    partial.replace(target)
    return target


def split_file(source: Path, output_dir: Path, part_size: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    with source.open("rb") as handle:
        index = 1
        while True:
            block = handle.read(part_size)
            if not block:
                break
            part = output_dir / f"{source.name}.part{index:03d}"
            part.write_bytes(block)
            parts.append({"name": part.name, "size": len(block), "sha256": sha256_file(part)})
            index += 1
    metadata = {
        "source_name": source.name, "source_size": source.stat().st_size,
        "source_sha256": sha256_file(source), "parts": parts,
    }
    manifest = output_dir / f"{source.name}.parts.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return manifest


def assemble_parts(parts_manifest: Path, output: Path) -> Path:
    metadata = json.loads(parts_manifest.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    with partial.open("wb") as target:
        for item in metadata["parts"]:
            part = parts_manifest.parent / item["name"]
            if part.stat().st_size != item["size"] or sha256_file(part) != item["sha256"]:
                raise RuntimeError(f"Invalid GPT-2-large part: {part}")
            with part.open("rb") as source:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    if partial.stat().st_size != metadata["source_size"]:
        raise RuntimeError("Assembled GPT-2-large file has the wrong size")
    actual = sha256_file(partial)
    if actual != metadata["source_sha256"]:
        raise RuntimeError(
            f"Assembled GPT-2-large SHA256 mismatch: expected {metadata['source_sha256']}, got {actual}"
        )
    partial.replace(output)
    return output


def download_model(manifest_path: Path, destination: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = manifest["model"]
    model_base = f"https://huggingface.co/{model['id']}/resolve/{model['revision']}"
    for item in model["files"]:
        print(f"Fetching {item['path']}")
        download_resumable(
            f"{model_base}/{item['path']}?download=true", destination / item["path"],
            int(item["size"]), item["sha256"],
        )


def download_dataset(manifest_path: Path, destination: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Dataset export requires pandas with a parquet engine (pyarrow)") from exc
    dataset = manifest["dataset"]
    dataset_base = f"https://huggingface.co/datasets/{dataset['id']}/resolve/{dataset['revision']}"
    dataset_root = destination / "wikitext-2-raw-v1"
    for item in dataset["files"]:
        parquet = download_resumable(
            f"{dataset_base}/{item['path']}?download=true",
            destination / item["path"], int(item["size"]), item["sha256"],
        )
        values = pd.read_parquet(parquet, columns=["text"])["text"]
        text = "\n".join(str(value).rstrip("\r\n") for value in values) + "\n"
        raw = dataset_root / item["raw_name"]
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(text, encoding="utf-8", newline="\n")
        actual = sha256_file(raw)
        if actual != item["raw_sha256"]:
            raise RuntimeError(
                f"Exported WikiText-2 SHA256 mismatch for {raw.name}: "
                f"expected {item['raw_sha256']}, got {actual}"
            )


def verify_resources(manifest_path: Path, destination: Path, include_dataset: bool) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = [(destination / item["path"], item["size"], item["sha256"])
              for item in manifest["model"]["files"]]
    if include_dataset:
        checks.extend(
            (destination / "wikitext-2-raw-v1" / item["raw_name"], None, item["raw_sha256"])
            for item in manifest["dataset"]["files"]
        )
    failures = []
    for path, size, expected in checks:
        if not path.is_file():
            failures.append(f"missing: {path}")
        elif size is not None and path.stat().st_size != size:
            failures.append(f"wrong size: {path}")
        elif sha256_file(path) != expected:
            failures.append(f"wrong SHA256: {path}")
    if failures:
        raise RuntimeError("Resource verification failed:\n" + "\n".join(failures))
    print(f"Verified {len(checks)} GPT-2-large/WikiText-2 resources")


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage pinned GPT-2-large offline resources")
    parser.add_argument(
        "action",
        choices=("download", "download-model", "download-dataset", "verify", "split", "assemble"),
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--include-dataset", action="store_true")
    parser.add_argument("--part-size-mib", type=int, default=1900)
    parser.add_argument("--parts-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "download":
        download_model(args.manifest, args.destination)
        if args.include_dataset:
            download_dataset(args.manifest, args.destination)
    elif args.action == "download-model":
        download_model(args.manifest, args.destination)
    elif args.action == "download-dataset":
        download_dataset(args.manifest, args.destination)
    elif args.action == "verify":
        verify_resources(args.manifest, args.destination, args.include_dataset)
    elif args.action == "split":
        source = args.destination / "model.safetensors"
        print(split_file(source, args.destination / "parts", args.part_size_mib * 1024 * 1024))
    else:
        if args.parts_manifest is None:
            parser.error("assemble requires --parts-manifest")
        output = args.output or (args.destination / "model.safetensors")
        print(assemble_parts(args.parts_manifest, output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

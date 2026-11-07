from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import sys

import torch


GPT2_LARGE_REVISION = "32b71b12589c2f8d625668d2335a01cac3249519"
GPT2_LARGE_FILENAME = "model.safetensors"
GPT2_LARGE_SIZE = 3_247_159_078
GPT2_LARGE_SHA256 = "5f47f3e12f91cd33b662ce7e433b6150ad5512b5884a2cee961b50e9c3bbebce"
GPT2_LARGE_RUNTIME_FILENAME = "model.fp16.safetensors"
GPT2_LARGE_RUNTIME_SIZE = 1_623_601_216
GPT2_LARGE_RUNTIME_SHA256 = "9008e91f3c248005e27aa0a258f334029cccdd32f700a9c459b143b3eb1eccc3"
WIKITEXT2_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


@dataclass(frozen=True)
class GPT2ResourceReport:
    model_variant: str
    checkpoint: str
    checkpoint_exists: bool
    checkpoint_size: int | None
    expected_size: int | None
    expected_sha256: str | None
    actual_sha256: str | None
    source_revision: str
    differential_layers: tuple[str, ...]
    available_ram_bytes: int | None
    available_commit_bytes: int | None
    available_cuda_bytes: int | None
    required_ram_bytes: int | None
    required_commit_bytes: int | None
    required_cuda_bytes: int | None
    ready: bool
    problems: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def available_ram_bytes() -> int | None:
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except (AttributeError, OSError):
            return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page_size * available_pages)
    except (AttributeError, OSError, ValueError):
        return None


def available_commit_bytes() -> int | None:
    """Return Windows commit headroom (physical memory plus page file)."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX(); status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPageFile)
    except (AttributeError, OSError):
        return None
    return None


def inspect_gpt2_resources(
    model_variant: str, checkpoint_path: str | Path, *, device: str = "cpu",
    verify_hash: bool = False,
) -> GPT2ResourceReport:
    from gpt2_cim.engine import differential_layers_for

    if model_variant not in {"gpt2-small", "gpt2-large"}:
        raise ValueError(f"Unsupported GPT-2 variant: {model_variant}")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    exists = checkpoint.is_file()
    size = checkpoint.stat().st_size if exists else None
    is_runtime_fp16 = (
        model_variant == "gpt2-large" and checkpoint.name == GPT2_LARGE_RUNTIME_FILENAME
    )
    expected_size = (
        GPT2_LARGE_RUNTIME_SIZE if is_runtime_fp16 else
        GPT2_LARGE_SIZE if model_variant == "gpt2-large" else None
    )
    expected_hash = (
        GPT2_LARGE_RUNTIME_SHA256 if is_runtime_fp16 else
        GPT2_LARGE_SHA256 if model_variant == "gpt2-large" else None
    )
    revision = GPT2_LARGE_REVISION if model_variant == "gpt2-large" else "local-gpt2-small"
    problems: list[str] = []
    if not exists:
        problems.append(f"checkpoint is missing: {checkpoint}")
    if expected_size is not None and size is not None and size != expected_size:
        problems.append(f"checkpoint size mismatch: expected {expected_size}, got {size}")
    actual_hash: str | None = None
    if verify_hash and exists and expected_hash is not None:
        actual_hash = sha256_file(checkpoint)
        if actual_hash != expected_hash:
            problems.append(
                f"checkpoint SHA256 mismatch: expected {expected_hash}, got {actual_hash}"
            )

    ram = available_ram_bytes()
    commit = available_commit_bytes()
    cuda_free: int | None = None
    required_ram = 8 * 1024**3 if model_variant == "gpt2-large" else None
    required_commit = (
        ((5 if is_runtime_fp16 else 10) if device == "cuda" else 10) * 1024**3
        if model_variant == "gpt2-large" else None
    )
    required_cuda = 6 * 1024**3 if model_variant == "gpt2-large" and device == "cuda" else None
    if model_variant == "gpt2-large":
        if ram is not None and required_ram is not None and ram < required_ram:
            problems.append(
                f"insufficient free RAM for GPT-2-large: {ram / 1024**3:.1f} GiB available; "
                "at least 8 GiB free is required"
            )
        if commit is not None and required_commit is not None and commit < required_commit:
            problems.append(
                f"insufficient Windows commit/page-file headroom for GPT-2-large: "
                f"{commit / 1024**3:.1f} GiB available; at least "
                f"{required_commit / 1024**3:.0f} GiB is required. "
                "Increase the Windows virtual-memory page file and retry"
            )
        if device == "cuda":
            if not torch.cuda.is_available():
                problems.append("CUDA was selected but is unavailable")
            else:
                cuda_free, _total = torch.cuda.mem_get_info()
                if required_cuda is not None and cuda_free < required_cuda:
                    problems.append(
                        f"insufficient free CUDA memory for GPT-2-large batch=1: "
                        f"{cuda_free / 1024**3:.1f} GiB available; at least 6 GiB free is required"
                    )
    return GPT2ResourceReport(
        model_variant=model_variant, checkpoint=str(checkpoint), checkpoint_exists=exists,
        checkpoint_size=size, expected_size=expected_size, expected_sha256=expected_hash,
        actual_sha256=actual_hash,
        source_revision=revision, differential_layers=differential_layers_for(model_variant),
        available_ram_bytes=ram, available_commit_bytes=commit,
        available_cuda_bytes=cuda_free, required_ram_bytes=required_ram,
        required_commit_bytes=required_commit, required_cuda_bytes=required_cuda,
        ready=not problems, problems=tuple(problems),
    )


_TRANSPOSED_SUFFIXES = (
    "attn.c_attn.weight", "attn.c_proj.weight",
    "mlp.c_fc.weight", "mlp.c_proj.weight",
)


def _native_tensor_name(source_name: str) -> str | None:
    """Map supported Hugging Face GPT-2 safetensors keys to nanoGPT names.

    The pinned GPT-2 Large artifact stores transformer tensors without the
    ``transformer.`` prefix. Attention mask buffers are intentionally rebuilt
    by nanoGPT and are therefore not copied from the checkpoint.
    """
    if source_name.endswith((".attn.bias", ".attn.masked_bias")):
        return None
    if source_name == "lm_head.weight":
        return source_name
    if source_name.startswith(("h.", "ln_f.", "wpe.", "wte.")):
        return "transformer." + source_name
    return source_name


def inspect_safetensors_layout(checkpoint: str | Path) -> tuple[str, ...]:
    """Return mapped parameter names without loading tensor payloads."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("Inspecting GPT-2 safetensors requires 'safetensors'") from exc
    mapped: list[str] = []
    with safe_open(Path(checkpoint), framework="pt", device="cpu") as source:
        for source_name in source.keys():
            target_name = _native_tensor_name(source_name)
            if target_name is not None:
                mapped.append(target_name)
    return tuple(mapped)


def load_checkpoint_into_model(model: torch.nn.Module, checkpoint: str | Path) -> None:
    """Load either the existing native .pt file or pinned Hugging Face safetensors.

    Safetensors are copied one tensor at a time into nanoGPT to avoid constructing a
    second full state dictionary for GPT-2-large.
    """
    path = Path(checkpoint)
    if path.suffix.lower() != ".safetensors":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("best_model_state_dict", payload.get("model_state_dict", payload))
        model.load_state_dict(state)
        return
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "Loading GPT-2-large model.safetensors requires the 'safetensors' package"
        ) from exc

    target = model.state_dict()
    loaded: set[str] = set()
    try:
        with torch.no_grad(), safe_open(path, framework="pt", device="cpu") as source:
            for source_name in source.keys():
                name = _native_tensor_name(source_name)
                if name is None:
                    continue
                if name not in target:
                    raise RuntimeError(
                        f"Unexpected GPT-2 checkpoint tensor: {source_name} (mapped to {name})"
                    )
                value = source.get_tensor(source_name)
                if any(name.endswith(suffix) for suffix in _TRANSPOSED_SUFFIXES):
                    value = value.t()
                if value.shape != target[name].shape:
                    raise RuntimeError(
                        f"GPT-2 tensor shape mismatch for {name}: "
                        f"expected {tuple(target[name].shape)}, got {tuple(value.shape)}"
                    )
                target[name].copy_(value)
                loaded.add(name)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1455 or "1455" in str(exc):
            raise RuntimeError(
                "Windows virtual memory/page file is too small to map the GPT-2-large "
                "checkpoint. Increase the system-managed page file, close memory-heavy "
                "applications, and retry."
            ) from exc
        raise
    # Hugging Face safetensors stores a tied embedding only once.
    if "transformer.wte.weight" in loaded:
        loaded.add("lm_head.weight")
    missing = {
        name for name in target
        if name not in loaded and not name.endswith(".attn.bias")
    }
    if missing:
        raise RuntimeError(f"GPT-2 checkpoint is missing tensors: {sorted(missing)}")

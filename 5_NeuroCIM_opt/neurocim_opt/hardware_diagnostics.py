"""Reusable NVIDIA/CUDA diagnostics for source and frozen Windows builds."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class GpuDeviceDiagnostic:
    index: int
    name: str
    total_memory_bytes: int


@dataclass(frozen=True)
class HardwareDiagnostics:
    torch_version: str
    compiled_cuda: str | None
    cuda_available: bool
    cuda_initialized: bool
    cuda_initialization_error: str | None
    gpu_devices: tuple[GpuDeviceDiagnostic, ...]
    torch_cuda_dll: str | None
    torch_cuda_dll_exists: bool
    nvidia_smi_path: str | None
    nvidia_smi_driver_version: str | None
    minimum_driver_version: str | None
    driver_compatible: bool | None
    nvidia_smi_output: str | None
    nvidia_smi_error: str | None
    build_ready: bool
    runtime_ready: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_cuda_computation_smoke() -> dict[str, Any]:
    """Execute real CUDA kernels from the current (including frozen) runtime."""
    import torch

    report = collect_hardware_diagnostics(torch_module=torch)
    result: dict[str, Any] = report.as_dict()
    result["real_cuda_smoke"] = "failed"
    if not report.runtime_ready:
        result["smoke_error"] = "; ".join(report.reasons) or "CUDA runtime is not ready"
        return result
    try:
        left = torch.tensor([[1.0, 2.0]], device="cuda")
        right = torch.tensor([[3.0], [4.0]], device="cuda")
        value = float((left @ right).item())
        layer = torch.nn.Conv2d(1, 2, kernel_size=3, padding=1).eval().cuda()
        with torch.inference_mode():
            output = layer(torch.ones(1, 1, 8, 8, device="cuda"))
        torch.cuda.synchronize()
        if value != 11.0 or output.shape != (1, 2, 8, 8):
            raise RuntimeError(
                f"unexpected CUDA result: matmul={value}, conv_shape={tuple(output.shape)}"
            )
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("CUDA convolution produced non-finite output")
        result["real_cuda_smoke"] = "passed"
        result["cuda_matmul_result"] = value
        result["cuda_conv_shape"] = list(output.shape)
    except Exception as exc:
        result["smoke_error"] = f"{type(exc).__name__}: {exc}"
    return result


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _windows_minimum_driver(compiled_cuda: str | None) -> str | None:
    """Return NVIDIA's Windows minor-version compatibility floor."""
    if not compiled_cuda:
        return None
    try:
        major = int(compiled_cuda.split(".", 1)[0])
    except (TypeError, ValueError):
        return None
    if major >= 12:
        return "527.41"
    if major == 11:
        return "452.39"
    return None


def _version_at_least(actual: str, required: str) -> bool:
    def parts(value: str) -> tuple[int, ...]:
        return tuple(int(piece) for piece in value.split("."))
    try:
        return parts(actual) >= parts(required)
    except (TypeError, ValueError):
        return False


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), capture_output=True, text=True, timeout=10, check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _cuda_dll_candidates(torch_module: Any) -> tuple[Path, ...]:
    candidates: list[Path] = []
    torch_file = getattr(torch_module, "__file__", None)
    if torch_file:
        candidates.append(Path(torch_file).resolve().parent / "lib" / "torch_cuda.dll")
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        candidates.append(Path(frozen_root).resolve() / "torch" / "lib" / "torch_cuda.dll")
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def collect_hardware_diagnostics(
    *, torch_module: Any | None = None, command_runner: CommandRunner | None = None,
    platform_name: str | None = None, nvidia_smi_path: str | None = None,
) -> HardwareDiagnostics:
    """Inspect the installed torch build, CUDA initialization, DLL and driver.

    Dependency injection arguments keep this function deterministic in tests and
    allow callers to reuse it in installers without changing global state.
    """
    if torch_module is None:
        import torch as torch_module
    runner = command_runner or _run_command
    platform = platform_name or os.name
    torch_version = str(getattr(torch_module, "__version__", "unknown"))
    compiled_cuda = getattr(getattr(torch_module, "version", None), "cuda", None)

    dll_candidates = _cuda_dll_candidates(torch_module)
    existing_dll = next((path for path in dll_candidates if path.is_file()), None)
    dll_required = platform == "nt"
    dll_ok = existing_dll is not None if dll_required else True
    reported_dll = existing_dll or (dll_candidates[0] if dll_candidates else None)

    smi_path = nvidia_smi_path or shutil.which("nvidia-smi")
    smi_output = smi_error = driver_version = None
    if smi_path:
        try:
            completed = runner([
                smi_path,
                "--query-gpu=driver_version,name,memory.total",
                "--format=csv,noheader,nounits",
            ])
            if completed.returncode == 0:
                smi_output = completed.stdout.strip() or None
                if smi_output:
                    driver_version = smi_output.splitlines()[0].split(",", 1)[0].strip()
            else:
                smi_error = (completed.stderr or completed.stdout or
                             f"nvidia-smi exited with code {completed.returncode}").strip()
        except (OSError, subprocess.SubprocessError) as exc:
            smi_error = f"{type(exc).__name__}: {exc}"
    else:
        smi_error = "nvidia-smi was not found"

    cuda_available = False
    initialized = False
    initialization_error = None
    devices: list[GpuDeviceDiagnostic] = []
    try:
        cuda_available = bool(torch_module.cuda.is_available())
        if cuda_available:
            torch_module.cuda.init()
            count = int(torch_module.cuda.device_count())
            for index in range(count):
                properties = torch_module.cuda.get_device_properties(index)
                devices.append(GpuDeviceDiagnostic(
                    index=index,
                    name=str(properties.name),
                    total_memory_bytes=int(properties.total_memory),
                ))
            initialized = count > 0
    except Exception as exc:  # CUDA raises several backend-specific exception types.
        initialization_error = f"{type(exc).__name__}: {exc}"

    minimum_driver = _windows_minimum_driver(str(compiled_cuda) if compiled_cuda else None)
    driver_compatible = None
    if platform == "nt" and minimum_driver and driver_version:
        driver_compatible = _version_at_least(driver_version, minimum_driver)

    reasons: list[str] = []
    if not compiled_cuda:
        reasons.append("Installed PyTorch is a CPU-only build (torch.version.cuda is empty).")
    if dll_required and not dll_ok:
        reasons.append(
            "CUDA PyTorch DLL is missing: "
            + (str(reported_dll) if reported_dll else "torch/lib/torch_cuda.dll")
        )
    if smi_error:
        reasons.append(f"NVIDIA driver query failed: {smi_error}")
    if driver_compatible is False:
        reasons.append(
            f"NVIDIA driver {driver_version} is older than the Windows minimum "
            f"{minimum_driver} required by CUDA {compiled_cuda}."
        )
    if initialization_error:
        reasons.append(f"PyTorch CUDA initialization failed: {initialization_error}")
    elif compiled_cuda and not cuda_available:
        reasons.append(
            "PyTorch was compiled with CUDA but torch.cuda.is_available() is false; "
            "check the NVIDIA driver, GPU visibility, and package flavor."
        )
    elif cuda_available and not devices:
        reasons.append("PyTorch reports CUDA available but no CUDA device was enumerated.")

    build_ready = bool(compiled_cuda) and dll_ok
    runtime_ready = (
        build_ready and cuda_available and initialized and initialization_error is None
        and driver_compatible is not False
    )
    return HardwareDiagnostics(
        torch_version=torch_version,
        compiled_cuda=str(compiled_cuda) if compiled_cuda else None,
        cuda_available=cuda_available,
        cuda_initialized=initialized,
        cuda_initialization_error=initialization_error,
        gpu_devices=tuple(devices),
        torch_cuda_dll=str(reported_dll) if reported_dll else None,
        torch_cuda_dll_exists=existing_dll is not None,
        nvidia_smi_path=smi_path,
        nvidia_smi_driver_version=driver_version,
        minimum_driver_version=minimum_driver,
        driver_compatible=driver_compatible,
        nvidia_smi_output=smi_output,
        nvidia_smi_error=smi_error,
        build_ready=build_ready,
        runtime_ready=runtime_ready,
        reasons=tuple(reasons),
    )


def format_hardware_diagnostics(report: HardwareDiagnostics) -> str:
    gpu_lines = [
        f"  GPU {item.index}: {item.name}, {item.total_memory_bytes / 2**30:.2f} GiB"
        for item in report.gpu_devices
    ] or ["  GPU: none initialized"]
    reason_lines = [f"  - {reason}" for reason in report.reasons] or ["  - none"]
    return "\n".join([
        f"PyTorch: {report.torch_version}",
        f"Compiled CUDA: {report.compiled_cuda or 'none (CPU build)'}",
        f"torch.cuda available/initialized: {report.cuda_available}/{report.cuda_initialized}",
        f"torch_cuda.dll: {report.torch_cuda_dll or 'not located'} "
        f"(exists={report.torch_cuda_dll_exists})",
        f"nvidia-smi: {report.nvidia_smi_path or 'not found'}",
        f"NVIDIA driver: {report.nvidia_smi_driver_version or 'unknown'}",
        f"Minimum NVIDIA driver: {report.minimum_driver_version or 'unknown'} "
        f"(compatible={report.driver_compatible})",
        *gpu_lines,
        f"GPU build ready: {report.build_ready}",
        f"GPU runtime ready: {report.runtime_ready}",
        "Reasons:",
        *reason_lines,
    ])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect NeuroCIM NVIDIA/CUDA readiness")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--require-build", action="store_true",
                        help="Fail unless this is a CUDA PyTorch build with required DLLs")
    parser.add_argument("--require-cuda", action="store_true",
                        help="Fail unless a real CUDA device initializes")
    args = parser.parse_args(argv)
    report = collect_hardware_diagnostics()
    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False)
          if args.json else format_hardware_diagnostics(report))
    if args.require_cuda and not report.runtime_ready:
        return 2
    if args.require_build and not report.build_ready:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

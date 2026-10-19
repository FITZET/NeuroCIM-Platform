from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import subprocess

from neurocim_opt.hardware_diagnostics import (
    collect_hardware_diagnostics, format_hardware_diagnostics, main,
)


class FakeCuda:
    def __init__(self, *, available: bool, error: Exception | None = None):
        self.available = available
        self.error = error

    def is_available(self):
        if self.error:
            raise self.error
        return self.available

    def init(self):
        if self.error:
            raise self.error

    def device_count(self):
        return 1 if self.available else 0

    def get_device_properties(self, _index):
        return SimpleNamespace(name="Test NVIDIA GPU", total_memory=12 * 2**30)


def fake_torch(root: Path, *, compiled_cuda, available=False, error=None):
    package = root / "torch"
    package.mkdir(parents=True, exist_ok=True)
    init = package / "__init__.py"
    init.write_text("", encoding="utf-8")
    return SimpleNamespace(
        __version__="2.4.0+test", __file__=str(init),
        version=SimpleNamespace(cuda=compiled_cuda),
        cuda=FakeCuda(available=available, error=error),
    )


def successful_smi(_command):
    return subprocess.CompletedProcess(
        [], 0, "555.99, Test NVIDIA GPU, 12288\n", "",
    )


def test_cpu_build_reports_explicit_reason(tmp_path: Path) -> None:
    report = collect_hardware_diagnostics(
        torch_module=fake_torch(tmp_path, compiled_cuda=None),
        command_runner=successful_smi, platform_name="nt", nvidia_smi_path="nvidia-smi.exe",
    )
    assert not report.build_ready and not report.runtime_ready
    assert report.nvidia_smi_driver_version == "555.99"
    assert any("CPU-only" in reason for reason in report.reasons)
    assert any("DLL is missing" in reason for reason in report.reasons)


def test_cuda_initialization_exception_is_preserved(tmp_path: Path) -> None:
    dll = tmp_path / "torch/lib/torch_cuda.dll"
    dll.parent.mkdir(parents=True)
    dll.write_bytes(b"dll")
    report = collect_hardware_diagnostics(
        torch_module=fake_torch(tmp_path, compiled_cuda="12.1", error=RuntimeError("driver old")),
        command_runner=successful_smi, platform_name="nt", nvidia_smi_path="nvidia-smi.exe",
    )
    assert report.build_ready and not report.runtime_ready
    assert report.cuda_initialization_error == "RuntimeError: driver old"
    assert any("initialization failed" in reason for reason in report.reasons)


def test_ready_report_contains_gpu_memory_driver_and_reusable_text(tmp_path: Path) -> None:
    dll = tmp_path / "torch/lib/torch_cuda.dll"
    dll.parent.mkdir(parents=True)
    dll.write_bytes(b"dll")
    report = collect_hardware_diagnostics(
        torch_module=fake_torch(tmp_path, compiled_cuda="12.1", available=True),
        command_runner=successful_smi, platform_name="nt", nvidia_smi_path="nvidia-smi.exe",
    )
    assert report.build_ready and report.runtime_ready
    assert report.minimum_driver_version == "527.41"
    assert report.driver_compatible is True
    assert report.torch_cuda_dll_exists
    assert report.gpu_devices[0].total_memory_bytes == 12 * 2**30
    rendered = format_hardware_diagnostics(report)
    assert "Test NVIDIA GPU" in rendered and "12.00 GiB" in rendered
    assert "NVIDIA driver: 555.99" in rendered


def test_old_windows_driver_blocks_cuda_runtime_gate(tmp_path: Path) -> None:
    dll = tmp_path / "torch/lib/torch_cuda.dll"
    dll.parent.mkdir(parents=True)
    dll.write_bytes(b"dll")

    def old_driver(_command):
        return subprocess.CompletedProcess([], 0, "526.98, Test NVIDIA GPU, 12288\n", "")

    report = collect_hardware_diagnostics(
        torch_module=fake_torch(tmp_path, compiled_cuda="12.1", available=True),
        command_runner=old_driver, platform_name="nt", nvidia_smi_path="nvidia-smi.exe",
    )
    assert report.build_ready and not report.runtime_ready
    assert report.driver_compatible is False
    assert any("older than" in reason for reason in report.reasons)


def test_cli_require_cuda_returns_failure_for_cpu_build(tmp_path: Path, monkeypatch) -> None:
    report = collect_hardware_diagnostics(
        torch_module=fake_torch(tmp_path, compiled_cuda=None),
        platform_name="nt", nvidia_smi_path=None,
    )
    monkeypatch.setattr(
        "neurocim_opt.hardware_diagnostics.collect_hardware_diagnostics", lambda: report,
    )
    assert main(["--require-cuda", "--json"]) == 2

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from PySide6.QtCore import QSettings


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_frozen_app.py EXE SCREENSHOT")
    executable = Path(sys.argv[1]).resolve()
    screenshot = Path(sys.argv[2]).resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["NEUROCIM_RELEASE_SMOKE_SCREENSHOT"] = str(screenshot)
    gpt_result = screenshot.with_name("release_gpt_smoke.txt")
    version_result = screenshot.with_name("release_version_smoke.txt")
    expected_version = os.environ.get("NEUROCIM_RELEASE_EXPECTED_VERSION")
    settings = QSettings("DRAM-CIM", "NeuroSimPlatform")
    saved_settings = {key: settings.value(key) for key in settings.allKeys()}
    settings.clear()
    settings.sync()
    try:
        completed = subprocess.run(
            [str(executable), "--release-smoke"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        gpt_environment = environment.copy()
        gpt_environment.pop("NEUROCIM_RELEASE_SMOKE_SCREENSHOT", None)
        gpt_environment["NEUROCIM_RELEASE_GPT_SMOKE"] = str(gpt_result)
        gpt_completed = subprocess.run(
            [str(executable), "--release-gpt-smoke"],
            env=gpt_environment,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        version_environment = environment.copy()
        version_environment.pop("NEUROCIM_RELEASE_SMOKE_SCREENSHOT", None)
        version_environment["NEUROCIM_RELEASE_VERSION_RESULT"] = str(version_result)
        version_completed = subprocess.run(
            [str(executable), "--release-version-smoke"],
            env=version_environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    finally:
        settings.clear()
        for key, value in saved_settings.items():
            settings.setValue(key, value)
        settings.sync()
    if completed.returncode != 0:
        raise RuntimeError(
            f"frozen smoke failed ({completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    if not screenshot.is_file() or screenshot.stat().st_size == 0:
        raise RuntimeError(f"frozen GUI did not create screenshot: {screenshot}")
    if gpt_completed.returncode != 0:
        details = gpt_result.read_text(encoding="utf-8") if gpt_result.is_file() else ""
        raise RuntimeError(
            f"frozen GPT-2 import smoke failed ({gpt_completed.returncode})\n{details}"
        )
    if not gpt_result.is_file() or not gpt_result.read_text(encoding="utf-8").startswith("ok:"):
        raise RuntimeError(f"frozen GPT-2 smoke did not create a success result: {gpt_result}")
    if version_completed.returncode != 0 or not version_result.is_file():
        raise RuntimeError("frozen application version smoke failed")
    frozen_version = version_result.read_text(encoding="utf-8").strip()
    if expected_version and frozen_version != expected_version:
        raise RuntimeError(
            f"frozen application version mismatch: expected {expected_version}, got {frozen_version}"
        )
    print(f"frozen smoke passed: {screenshot}")
    print(f"frozen GPT-2 tokenizer smoke passed: {gpt_result}")
    print(f"frozen application version: {frozen_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

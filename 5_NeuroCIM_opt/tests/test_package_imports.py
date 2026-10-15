from __future__ import annotations

import subprocess
import sys


def test_package_root_keeps_lightweight_tools_free_of_numpy() -> None:
    code = (
        "import sys; import neurocim_opt; "
        "assert 'numpy' not in sys.modules; "
        "from neurocim_opt import HardwareErrorModel; "
        "assert 'numpy' in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

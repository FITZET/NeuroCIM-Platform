from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM_ROOT))

from neurocim_opt import MacImportOptions, PlatformService


def main() -> int:
    target = PLATFORM_ROOT / "outputs" / "acceptance" / "performance"
    target.mkdir(parents=True, exist_ok=True)
    source = target / "mac_100k.csv"
    count = 100_000
    values = np.arange(count, dtype=np.float64) % 65536
    pd.DataFrame(
        {
            "array_size": ["64x64"] * count,
            "bit_width": np.full(count, 8),
            "ideal_mac": values,
            "measured_output": 0.999 * values + 0.000001 * values**2,
            "group": ["benchmark"] * count,
        }
    ).to_csv(source, index=False)
    service = PlatformService()
    started = time.perf_counter()
    preview = service.preview_mac(source, MacImportOptions({}), limit=20)
    import_seconds = time.perf_counter() - started
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "rows": preview.rows,
        "csv_bytes": source.stat().st_size,
        "import_and_validate_seconds": import_seconds,
        "target_seconds": 3.0,
        "passed": import_seconds <= 3.0,
    }
    (target / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

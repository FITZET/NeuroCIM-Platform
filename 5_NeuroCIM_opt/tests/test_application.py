from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neurocim_opt import MacImportOptions, PlatformService


def test_platform_service_preview_fit_and_export(tmp_path: Path) -> None:
    source = tmp_path / "mac.csv"
    pd.DataFrame(
        {
            "array_size": ["16x16"] * 4,
            "bit_width": [2] * 4,
            "ideal_mac": [0, 1, 2, 3],
            "measured_output": [0, 1.1, 2.4, 3.9],
            "group": ["a"] * 4,
        }
    ).to_csv(source, index=False)
    service = PlatformService()
    options = MacImportOptions({})
    preview = service.preview_mac(source, options, limit=2)
    assert preview.rows == 4
    assert len(preview.records) == 2
    outcome = service.fit_mac(source, tmp_path / "model", options)
    assert outcome.model_path.is_file()
    exported = service.export_table(list(outcome.summary), tmp_path / "summary.xlsx")
    assert exported.is_file()


def test_lenet_inference_reports_missing_resources(tmp_path: Path) -> None:
    service = PlatformService()
    try:
        service.infer_lenet5_mnist(tmp_path, 0)
    except FileNotFoundError as exc:
        message = str(exc)
        assert "Required inference resources are missing" in message
        assert "lenet5_float.pt" in message
        assert "error_model.json" in message
    else:
        raise AssertionError("missing inference resources were not rejected")


def test_ppa_reports_missing_resources(tmp_path: Path) -> None:
    service = PlatformService()
    try:
        service.evaluate_lenet5_ppa(
            ROOT.parent, tmp_path / "missing.pt", tmp_path / "missing.json", tmp_path / "out"
        )
    except FileNotFoundError as exc:
        message = str(exc)
        assert "Required NeuroSim resources are missing" in message
        assert "missing.pt" in message
        assert "missing.json" in message
    else:
        raise AssertionError("missing PPA resources were not rejected")

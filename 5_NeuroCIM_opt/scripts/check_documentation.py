"""Check delivery-document links and retained acceptance evidence."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    PLATFORM_ROOT / "README.md",
    PLATFORM_ROOT / "docs" / "USER_MANUAL.md",
    PLATFORM_ROOT / "docs" / "TECHNICAL_MANUAL.md",
    PLATFORM_ROOT / "docs" / "ENVIRONMENT_SETUP.md",
    PLATFORM_ROOT / "docs" / "TEST_REPORT.md",
)
EVIDENCE = (
    "outputs/automation/AUTOMATION_STATE.json",
    "outputs/error_models/01_NMOS/error_model.json",
    "outputs/acceptance/gui_cell_fit/acceptance.json",
    "outputs/acceptance/gui_cell_fit/cell_fitting_complete.png",
    "outputs/acceptance/performance/benchmark.json",
    "outputs/acceptance/gui/mac_analysis.png",
    "outputs/acceptance/gui_inference/sample_0_prediction.json",
    "outputs/acceptance/gui_training/gui_acceptance.json",
    "outputs/acceptance/gui_training/training_result.json",
    "outputs/acceptance/gui_ppa/gui_ppa_result.json",
    "outputs/neurosim/01_NMOS/neurosim_result.json",
)
LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


def _local_target(document: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().strip("<>").split("#", 1)[0]
    if not target or "://" in target or target.startswith("mailto:"):
        return None
    return (document.parent / unquote(target)).resolve()


def main() -> int:
    missing: list[str] = []
    checked_links = 0
    for document in DOCUMENTS:
        if not document.is_file():
            missing.append(str(document.relative_to(PLATFORM_ROOT)))
            continue
        text = document.read_text(encoding="utf-8")
        for raw_target in LINK.findall(text):
            target = _local_target(document, raw_target)
            if target is None:
                continue
            checked_links += 1
            if not target.exists():
                missing.append(
                    f"{document.relative_to(PLATFORM_ROOT)} -> {raw_target}"
                )
    for relative in EVIDENCE:
        if not (PLATFORM_ROOT / relative).is_file():
            missing.append(relative)
    if missing:
        raise SystemExit("Missing documentation targets:\n- " + "\n- ".join(missing))
    print(
        f"Documentation OK: {len(DOCUMENTS)} files, "
        f"{checked_links} local links, {len(EVIDENCE)} evidence files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

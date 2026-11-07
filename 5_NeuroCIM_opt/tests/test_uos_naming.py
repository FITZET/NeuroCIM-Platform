from __future__ import annotations

from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
TEXT_SUFFIXES = {".py", ".json", ".md", ".txt", ".csv", ".yaml", ".yml", ".toml", ".ps1"}
SCANNED_ROOTS = (
    WORKSPACE / "0_data",
    WORKSPACE / "1_data_fitting",
    WORKSPACE / "2_model_basic",
    WORKSPACE / "3_models",
    WORKSPACE / "5_NeuroCIM_opt" / "neurocim_opt",
    WORKSPACE / "5_NeuroCIM_opt" / "scripts",
    WORKSPACE / "5_NeuroCIM_opt" / "tests",
    WORKSPACE / "5_NeuroCIM_opt" / "docs",
    WORKSPACE / "5_NeuroCIM_opt" / "packaging",
)


def test_active_platform_uses_uos_naming_exclusively() -> None:
    forbidden = "a" + "os"
    forbidden_paths: list[str] = []
    forbidden_text: list[str] = []
    for root in SCANNED_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if "__pycache__" in path.parts:
                continue
            if forbidden in path.name.casefold():
                forbidden_paths.append(str(path.relative_to(WORKSPACE)))
            if not path.is_file() or path.suffix.casefold() not in TEXT_SUFFIXES:
                continue
            if path == Path(__file__) or (
                "GPT2" in path.parts and path.name in {"merges.txt", "tokenizer.json", "vocab.json"}
            ):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if forbidden in text.casefold():
                forbidden_text.append(str(path.relative_to(WORKSPACE)))
    assert forbidden_paths == []
    assert forbidden_text == []

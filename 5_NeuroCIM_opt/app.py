from pathlib import Path
import json
import sys

if __name__ == "__main__":
    platform_root = (
        Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parent
    )
    if "--gpu-smoke" in sys.argv:
        from neurocim_opt.hardware_diagnostics import run_cuda_computation_smoke

        index = sys.argv.index("--gpu-smoke")
        output_path = (
            Path(sys.argv[index + 1]).expanduser().resolve()
            if index + 1 < len(sys.argv) and not sys.argv[index + 1].startswith("--")
            else platform_root / "gpu_diagnostics.json"
        )
        result = run_cuda_computation_smoke()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        raise SystemExit(0 if result.get("real_cuda_smoke") == "passed" else 3)

    from neurocim_opt.gui import run_gui

    raise SystemExit(run_gui(platform_root))

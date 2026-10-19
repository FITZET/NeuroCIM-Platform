"""GPU release gate: build inspection on hosted runners, real CUDA smoke locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PLATFORM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM))

from neurocim_opt.hardware_diagnostics import collect_hardware_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-only", action="store_true",
        help="Verify the CUDA build and DLL without claiming real GPU execution",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = collect_hardware_diagnostics()
    result = report.as_dict()
    result["real_cuda_smoke"] = "not_executed" if args.build_only else "failed"

    if not report.build_ready:
        result["gate_error"] = "CUDA build/DLL prerequisite failed"
        exit_code = 2
    elif args.build_only:
        result["gate_note"] = (
            "Build prerequisites passed; real CUDA execution was not run. "
            "Run this script without --build-only on an NVIDIA/self-hosted runner before release."
        )
        exit_code = 0
    elif not report.runtime_ready:
        result["gate_error"] = "A real CUDA device could not be initialized"
        exit_code = 3
    else:
        import torch
        try:
            left = torch.tensor([[1.0, 2.0]], device="cuda")
            right = torch.tensor([[3.0], [4.0]], device="cuda")
            value = float((left @ right).item())
            torch.cuda.synchronize()
            if value != 11.0:
                raise RuntimeError(f"unexpected CUDA matmul result: {value}")
            project = PLATFORM.parent
            vision_root = project / "3_models" / "VisionZoo"
            sys.path.insert(0, str(vision_root))
            from vision_zoo.engine import differential_copy
            from vision_zoo.models import resnet8_mnist

            model = resnet8_mnist().eval().cuda()
            sample = torch.randn(1, 1, 28, 28, device="cuda")
            with torch.inference_mode():
                float_logits = model(sample)
            scales = {
                name: 1.0 / 255.0 for name, module in model.named_modules()
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
            }
            codes = torch.arange(256, dtype=torch.int64)
            ideal_lut = codes[:, None] * codes[None, :]
            differential = differential_copy(
                model.cpu(), ideal_lut, scales, ste=False,
            ).eval().cuda()
            with torch.inference_mode():
                differential_logits = differential(sample)
            torch.cuda.synchronize()
            if float_logits.shape != (1, 10) or differential_logits.shape != (1, 10):
                raise RuntimeError("ResNet-8 CUDA smoke returned an invalid output shape")
            if not torch.isfinite(float_logits).all() or not torch.isfinite(differential_logits).all():
                raise RuntimeError("ResNet-8 CUDA smoke returned non-finite logits")
            result["real_cuda_smoke"] = "passed"
            result["cuda_matmul_result"] = value
            result["formal_model_smoke"] = {
                "model_id": "resnet8-mnist",
                "float_output_shape": list(float_logits.shape),
                "ideal_differential_output_shape": list(differential_logits.shape),
                "device": str(float_logits.device),
            }
            exit_code = 0
        except Exception as exc:
            result["gate_error"] = f"Real CUDA tensor smoke failed: {type(exc).__name__}: {exc}"
            exit_code = 4

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

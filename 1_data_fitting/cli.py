from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_fitting import DataFittingService, FittingConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit DRAM-CIM device response data.")
    parser.add_argument("input", type=Path, help="Device or ideal multiplication CSV")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Result directory")
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    config = FittingConfig(
        generate_plots=not args.no_plots,
        image_format=args.format,
        plot_dpi=args.dpi,
    )
    service = DataFittingService()
    result = service.fit(
        args.input,
        output_dir=args.output,
        config=config,
        progress=lambda percent, message: print(f"[{percent:3d}%] {message}"),
    )
    print(json.dumps(result.summary(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

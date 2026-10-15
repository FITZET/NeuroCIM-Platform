from pathlib import Path
import sys

from neurocim_opt.gui import run_gui


if __name__ == "__main__":
    platform_root = (
        Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parent
    )
    raise SystemExit(run_gui(platform_root))

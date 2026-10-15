from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

PLATFORM_ROOT = Path(__file__).resolve().parents[3] / "5_NeuroCIM_opt"
if str(PLATFORM_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PLATFORM_ROOT))
from neurocim_opt.paths import data_root


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_root: Path = data_root("mnist")
    output_dir: Path = Path("outputs")
    batch_size: int = 128
    eval_batch_size: int = 256
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 0
    seed: int = 42
    calibration_batches: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def mnist_root(self) -> Path:
        # torchvision.datasets.MNIST appends its own "MNIST" directory.
        return self.dataset_root

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dataset_root"] = str(self.dataset_root)
        data["output_dir"] = str(self.output_dir)
        return data

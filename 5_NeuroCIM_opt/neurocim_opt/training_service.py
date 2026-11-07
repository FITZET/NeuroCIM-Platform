from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
from pathlib import Path
import math
import sys
from time import perf_counter
from typing import Callable

import torch
from torch import nn

from .paths import data_root


class TrainingCancelled(RuntimeError):
    """Raised after a cooperative cancellation request."""


@dataclass(frozen=True)
class TrainingRequest:
    # Keep the historical positional fields first for existing GUI callers.
    project_root: Path
    error_model_path: Path
    output_dir: Path
    error_strength: float = 1.0
    epochs: int = 1
    batch_size: int = 128
    device: str = "auto"
    dataset_root: Path | None = None
    learning_rate: float = 1e-4
    model_id: str = "lenet5-mnist"
    checkpoint_path: Path | None = None
    auxiliary_path: Path | None = None
    seed: int = 42
    # Explicit smoke-run limits. None always means a complete dataset pass.
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    sequence_length: int = 64
    gradient_accumulation_steps: int = 1
    adc_bits: int | None = None
    noise_sigma: float = 0.0
    noise_rho: float = 0.0


@dataclass(frozen=True)
class TrainingOutcome:
    checkpoint_path: str
    history_path: str
    result_path: str
    output_dir: str
    device: str
    elapsed_seconds: float
    history: tuple[dict, ...]
    evaluations: dict
    weight_update: dict

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainingResources:
    model_id: str
    dataset_name: str
    model_root: Path
    checkpoint: Path
    auxiliary: Path | None
    error_model: Path
    dataset_root: Path
    device: torch.device


ProgressCallback = Callable[[int, str, dict | None], None]
CancelCallback = Callable[[], bool]

TINYTRANSFORMER_TRAIN_BATCH_LIMIT = 128
TINYTRANSFORMER_EVAL_BATCH_LIMIT = 16
VISION_ZOO_DIFFERENTIAL_BATCH_LIMIT = 4
TINYTRANSFORMER_FINETUNE_MODULES = (
    "classifier.attention_pool",
    "classifier.fc",
)


_MODEL_ALIASES = {
    "lenet5": "lenet5-mnist", "lenet-5": "lenet5-mnist",
    "lenet5_mnist": "lenet5-mnist", "lenet5-mnist": "lenet5-mnist",
    "bnn": "bnn-mnist", "bnn_mnist": "bnn-mnist", "bnn-mnist": "bnn-mnist",
    "resnet20": "resnet20-cifar10", "resnet-20": "resnet20-cifar10",
    "resnet20_cifar10": "resnet20-cifar10", "resnet20-cifar10": "resnet20-cifar10",
    "tinytransformer": "tinytransformer-cifar10", "tiny-transformer": "tinytransformer-cifar10",
    "cct": "tinytransformer-cifar10", "cct-7": "tinytransformer-cifar10",
    "tinytransformer-cifar10": "tinytransformer-cifar10",
    "resnet8": "resnet8-mnist", "resnet-8": "resnet8-mnist",
    "resnet8_mnist": "resnet8-mnist", "resnet8-mnist": "resnet8-mnist",
    "mobilenetv2": "mobilenetv2-cifar10", "mobilenet-v2": "mobilenetv2-cifar10",
    "mobilenetv2-cifar10": "mobilenetv2-cifar10",
    "vgg11": "vgg11-bn-cifar10", "vgg11-bn": "vgg11-bn-cifar10",
    "vgg11-bn-cifar10": "vgg11-bn-cifar10",
    "squeezenet": "squeezenet1_1-cifar10", "squeezenet1.1": "squeezenet1_1-cifar10",
    "squeezenet1_1-cifar10": "squeezenet1_1-cifar10",
    "gpt2-small": "gpt2-small", "gpt-2-small": "gpt2-small",
}


def canonical_model_id(value: str) -> str:
    try:
        return _MODEL_ALIASES[value.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported training model {value!r}; choose a registered trainable model"
        ) from exc


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # The trusted bundled BNN checkpoint contains NumPy RNG state, which
        # some PyTorch safe loaders cannot decode.
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must contain a state dictionary: {path}")
    state = payload.get("best_model_state_dict", payload.get("model_state_dict", payload))
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError(f"Checkpoint has no valid model state dictionary: {path}")
    return state


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _dataset_present(dataset: Path, name: str) -> bool:
    if name == "wikitext2":
        return dataset.is_dir() and all(
            any((dataset / filename).is_file() for filename in candidates)
            for candidates in (
                ("wiki.train.raw", "train.txt"),
                ("wiki.valid.raw", "validation.txt", "valid.txt"),
                ("wiki.test.raw", "test.txt"),
            )
        )
    if name == "mnist":
        raw = dataset / "MNIST" / "raw"
        return raw.is_dir() and any(raw.glob("*-images-idx*-ubyte*"))
    folder = dataset / "cifar-10-batches-py"
    return folder.is_dir() and (folder / "test_batch").is_file()


def _normalise_dataset_root(dataset: Path, name: str) -> Path:
    """Accept either a torchvision root or a selected raw folder/file."""
    dataset = dataset.expanduser().resolve()
    if name == "cifar10":
        if dataset.is_file() and dataset.name in {"test_batch", "data_batch_1"}:
            return dataset.parent.parent
        if dataset.name == "cifar-10-batches-py":
            return dataset.parent
    if name == "mnist":
        if dataset.is_file() and "ubyte" in dataset.name:
            dataset = dataset.parent
        if dataset.name.lower() == "raw" and dataset.parent.name == "MNIST":
            return dataset.parent.parent
        if dataset.name == "MNIST":
            return dataset.parent
    return dataset


def _configure_tinytransformer_head_finetuning(model: nn.Module) -> tuple[str, ...]:
    """Freeze CCT's backbone and leave sequence-pooling/classification heads trainable."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = dict(model.named_modules())
    missing = [name for name in TINYTRANSFORMER_FINETUNE_MODULES if name not in modules]
    if missing:
        raise RuntimeError(f"TinyTransformer fine-tuning modules are missing: {missing}")
    for name in TINYTRANSFORMER_FINETUNE_MODULES:
        for parameter in modules[name].parameters():
            parameter.requires_grad_(True)
    return tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)


class ErrorAwareTrainingService:
    """Error-aware training for all registered platform classifiers.

    The selected fitted LUT drives an exact signed positive/negative
    differential forward pass. Floating surrogate gradients (STE) make that
    hardware-aware path trainable.
    """

    def validate(self, request: TrainingRequest) -> TrainingResources:
        if not 0.0 <= request.error_strength <= 1.0:
            raise ValueError("lambda must be in [0, 1]")
        if request.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if request.batch_size < 1:
            raise ValueError("batch size must be at least 1")
        if request.learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        if request.max_train_batches is not None and request.max_train_batches < 1:
            raise ValueError("max_train_batches must be at least 1")
        if request.max_eval_batches is not None and request.max_eval_batches < 1:
            raise ValueError("max_eval_batches must be at least 1")
        if not 2 <= request.sequence_length <= 1024:
            raise ValueError("sequence_length must be in [2, 1024]")
        if request.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if request.adc_bits is not None and not 2 <= int(request.adc_bits) <= 16:
            raise ValueError("adc_bits must be None or in [2, 16]")
        if request.noise_sigma < 0 or not 0 <= request.noise_rho <= 1:
            raise ValueError("noise_sigma must be non-negative and noise_rho must be in [0, 1]")
        if request.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if request.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected but is unavailable")
        device = torch.device(
            "cuda" if request.device == "auto" and torch.cuda.is_available() else
            "cpu" if request.device == "auto" else request.device
        )
        model_id = canonical_model_id(request.model_id)
        root = Path(request.project_root).expanduser().resolve()
        defaults = {
            "lenet5-mnist": ("mnist", root / "3_models/LeNet_5",
                              root / "3_models/LeNet_5/chekpoint/lenet5_float.pt",
                              root / "3_models/LeNet_5/chekpoint/quantization.json"),
            "bnn-mnist": ("mnist", root / "3_models/BNN",
                           root / "3_models/BNN/checkpoint/bnn_mnist.pt", None),
            "resnet20-cifar10": ("cifar10", root / "3_models/Resnet20",
                                  root / "3_models/Resnet20/checkpoint/resnet20_cifar10.pt",
                                  root / "3_models/Resnet20/output/formal/input_scales.json"),
            "tinytransformer-cifar10": (
                "cifar10", root / "3_models/TinyTransformer",
                root / "3_models/TinyTransformer/checkpoint/cct_7_3x1_32_cifar10.pth",
                root / "3_models/TinyTransformer/checkpoint/input_scales.json",
            ),
            "resnet8-mnist": (
                "mnist", root / "3_models/VisionZoo",
                root / "3_models/VisionZoo/checkpoint/resnet8_mnist.pt",
                root / "3_models/VisionZoo/checkpoint/resnet8-mnist_input_scales.json",
            ),
            "mobilenetv2-cifar10": (
                "cifar10", root / "3_models/VisionZoo",
                root / "3_models/VisionZoo/checkpoint/cifar10_mobilenetv2_x1_0-fe6a5b48.pt",
                root / "3_models/VisionZoo/checkpoint/mobilenetv2-cifar10_input_scales.json",
            ),
            "vgg11-bn-cifar10": (
                "cifar10", root / "3_models/VisionZoo",
                root / "3_models/VisionZoo/checkpoint/cifar10_vgg11_bn-eaeebf42.pt",
                root / "3_models/VisionZoo/checkpoint/vgg11-bn-cifar10_input_scales.json",
            ),
            "squeezenet1_1-cifar10": (
                "cifar10", root / "3_models/VisionZoo",
                root / "3_models/VisionZoo/checkpoint/squeezenet1_1_cifar10.pt",
                root / "3_models/VisionZoo/checkpoint/squeezenet1_1-cifar10_input_scales.json",
            ),
            "gpt2-small": (
                "wikitext2", root / "3_models/GPT2",
                root / "3_models/GPT2/checkpoint/gpt2_nanogpt.pt",
                root / "3_models/GPT2/checkpoint/input_scales.json",
            ),
        }
        dataset_name, model_root, default_checkpoint, default_auxiliary = defaults[model_id]
        checkpoint = Path(request.checkpoint_path or default_checkpoint).expanduser().resolve()
        auxiliary_value = request.auxiliary_path if request.auxiliary_path is not None else default_auxiliary
        auxiliary = Path(auxiliary_value).expanduser().resolve() if auxiliary_value else None
        error_model = Path(request.error_model_path).expanduser().resolve()
        required = [checkpoint, error_model]
        if model_id != "bnn-mnist":
            required.append(auxiliary)
        missing = [str(path or "<auxiliary parameter file>") for path in required if path is None or not path.is_file()]
        if not model_root.is_dir():
            missing.append(str(model_root))
        if missing:
            raise FileNotFoundError(
                f"Required training resources are missing for {model_id}:\n" + "\n".join(missing)
            )
        if model_id == "gpt2-small":
            bundled_wikitext = root / "3_models/GPT2/checkpoint/gpt2-large/wikitext-2-raw-v1"
            requested_dataset = Path(request.dataset_root) if request.dataset_root else bundled_wikitext
            # GPT text generation historically recorded its tokenizer directory as
            # dataset_root. Convert that inherited value to the bundled training corpus.
            if requested_dataset.name == "tokenizer":
                requested_dataset = bundled_wikitext
            dataset = requested_dataset.expanduser().resolve()
        else:
            dataset = _normalise_dataset_root(
                Path(request.dataset_root) if request.dataset_root else data_root(dataset_name),
                dataset_name,
            )
        if not _dataset_present(dataset, dataset_name):
            expected = (
                dataset / "wiki.train.raw" if dataset_name == "wikitext2" else
                dataset / ("MNIST/raw" if dataset_name == "mnist" else "cifar-10-batches-py/test_batch")
            )
            raise FileNotFoundError(
                f"{dataset_name.upper()} is unavailable under {dataset}; expected {expected}"
            )
        return TrainingResources(
            model_id, dataset_name, model_root, checkpoint, auxiliary,
            error_model, dataset, device,
        )

    @staticmethod
    def _cancelled(cancel: CancelCallback | None) -> None:
        if cancel is not None and cancel():
            raise TrainingCancelled("Training cancelled by user")

    @staticmethod
    def _add_import_root(path: Path) -> None:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    def _prepare(self, resources: TrainingResources, request: TrainingRequest):
        """Build loaders and model conversion functions for one registered model."""
        self._add_import_root(resources.model_root)
        state = _checkpoint_state(resources.checkpoint)
        from .runtime_memory import memory_aware_batch_size
        mib = 1024 ** 2
        # Loader batches are inexpensive host tensors. Keep them large for low
        # Python/DataLoader overhead; the CUDA execution micro-batch is tuned
        # independently from measured peaks and persists after an OOM backoff.
        host_bytes_per_sample = 4 * mib if resources.dataset_name == "cifar10" else mib
        train_batch_size, _memory_snapshot = memory_aware_batch_size(
            request.batch_size, "cpu",
            estimated_host_bytes_per_sample=host_bytes_per_sample,
        )
        eval_batch_size, _ = memory_aware_batch_size(
            min(request.batch_size, 256), "cpu",
            estimated_host_bytes_per_sample=host_bytes_per_sample,
        )

        if resources.model_id == "lenet5-mnist":
            from lenet5.config import ExperimentConfig
            from lenet5.data import create_mnist_train_val_test_loaders
            from lenet5.differential import configure_differential_model
            from lenet5.model import LeNet5
            from lenet5.quantization import load_quantization_parameters

            config = ExperimentConfig(
                dataset_root=resources.dataset_root, output_dir=Path(request.output_dir),
                batch_size=train_batch_size, eval_batch_size=eval_batch_size,
                epochs=request.epochs, learning_rate=request.learning_rate,
                device=str(resources.device), num_workers=0,
            )
            loaders = create_mnist_train_val_test_loaders(config)
            parameters = load_quantization_parameters(resources.auxiliary)

            def base():
                model = LeNet5()
                model.load_state_dict(state)
                return model

            def differential(model, lut, ste):
                return configure_differential_model(model, parameters, lut, ste=ste)

            return loaders, base, differential, differential, state

        if resources.model_id == "bnn-mnist":
            from bnn.data import create_loaders
            from bnn.differential import collect_input_scales
            from bnn.model import BinaryMNIST
            from .layers import adapt_module

            loaders = create_loaders(
                resources.dataset_root, train_batch_size, eval_batch_size,
                request.seed, num_workers=0,
            )

            def base():
                model = BinaryMNIST()
                model.load_state_dict(state)
                return model

            scales = collect_input_scales(
                base().frozen(), loaders[0], str(resources.device), batches=1
            )

            def differential(model, lut, ste):
                # The frozen deployment graph supplies actual +/-1 BNN weights;
                # converted MAC parameters remain trainable via layer STE.
                graph = model.frozen() if hasattr(model, "frozen") else model
                return adapt_module(
                    graph, lut=torch.as_tensor(lut), input_scales=scales, ste=ste
                )

            return loaders, base, differential, differential, state

        if resources.model_id in {
            "resnet8-mnist", "mobilenetv2-cifar10", "vgg11-bn-cifar10",
            "squeezenet1_1-cifar10",
        }:
            from vision_zoo.data import create_loaders
            from vision_zoo.engine import calibrate_input_scales, differential_copy
            from vision_zoo.models import create_model

            loaders = create_loaders(
                resources.model_id, resources.dataset_root,
                batch_size=train_batch_size, eval_batch_size=eval_batch_size,
                seed=request.seed, num_workers=0, download=False,
            )

            def base():
                model = create_model(resources.model_id)
                model.load_state_dict(state)
                return model

            if resources.auxiliary is not None:
                scales = json.loads(resources.auxiliary.read_text(encoding="utf-8"))
                if not isinstance(scales, dict) or not scales:
                    raise ValueError(
                        f"{resources.model_id} input scales must be a non-empty object: "
                        f"{resources.auxiliary}"
                    )
                scales = {str(key): float(value) for key, value in scales.items()}
            else:
                scales = calibrate_input_scales(
                    base(), loaders[0], device=str(resources.device), max_batches=1,
                )

            def differential(model, lut, ste):
                return differential_copy(model, lut, scales, ste=ste)

            return loaders, base, differential, differential, state

        if resources.model_id == "tinytransformer-cifar10":
            from tiny_transformer import cct_7_3x1_32 as model_factory
            from tiny_transformer.data import create_cifar10_loaders
            from tiny_transformer.engine import differential_copy as full_differential_copy
            from tiny_transformer.engine import differential_head_copy
        else:
            from resnet20.data import create_cifar10_loaders
            from resnet20.engine import differential_copy
            from resnet20.model import resnet20 as model_factory

        loaders = create_cifar10_loaders(
            resources.dataset_root, batch_size=train_batch_size,
            eval_batch_size=eval_batch_size, seed=request.seed,
            num_workers=0, download=False,
        )
        scales = json.loads(resources.auxiliary.read_text(encoding="utf-8"))
        if not isinstance(scales, dict) or not scales:
            raise ValueError(
                f"{resources.model_id} input scales must be a non-empty object: {resources.auxiliary}"
            )
        scales = {str(key): float(value) for key, value in scales.items()}

        def base():
            model = model_factory()
            model.load_state_dict(state)
            if resources.model_id == "tinytransformer-cifar10":
                _configure_tinytransformer_head_finetuning(model)
            return model

        if resources.model_id == "tinytransformer-cifar10":
            def training_differential(model, lut, ste):
                converted = differential_head_copy(model, lut, scales, ste=ste)
                _configure_tinytransformer_head_finetuning(converted)
                return converted

            def evaluation_differential(model, lut, ste):
                # Match single-sample and grouped inference: every Conv2d/Linear
                # executes the selected signed-differential LUT during evaluation.
                return full_differential_copy(model, lut, scales, ste=ste)
        else:
            def training_differential(model, lut, ste):
                return differential_copy(model, lut, scales, ste=ste)

            evaluation_differential = training_differential

        return loaders, base, training_differential, evaluation_differential, state

    def run(
        self, request: TrainingRequest, *, progress: ProgressCallback | None = None,
        cancel: CancelCallback | None = None,
    ) -> TrainingOutcome:
        notify = progress or (lambda _percent, _message, _metrics=None: None)
        resources = self.validate(request)
        self._cancelled(cancel)
        # Batch inference keeps three model variants in the shared runtime
        # cache.  Training rebuilds its own models, so retaining those objects
        # only consumes RAM/VRAM and can prevent CIFAR-10 from allocating its
        # raw training array on memory-constrained Windows machines.
        from .inference_runtime import clear_lenet_inference_cache
        clear_lenet_inference_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if resources.model_id == "gpt2-small":
            return self._run_gpt2_small(request, resources, notify, cancel)
        from .error_model import HardwareErrorModel
        from .runtime_memory import (
            empirical_cuda_batch_size, is_cuda_oom, release_cuda_memory,
            runtime_memory_snapshot,
        )

        started = perf_counter()
        output = Path(request.output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        (
            loaders, base, training_differential, evaluation_differential, initial_state,
        ) = self._prepare(resources, request)
        train_loader, validation_loader, test_loader = loaders
        planned_train_batches = len(train_loader)
        if request.max_train_batches is not None:
            planned_train_batches = min(planned_train_batches, request.max_train_batches)
        planned_eval_batches = len(test_loader)
        planned_validation_batches = len(validation_loader)
        if request.max_eval_batches is not None:
            planned_eval_batches = min(planned_eval_batches, request.max_eval_batches)
            planned_validation_batches = min(
                planned_validation_batches, request.max_eval_batches
            )
        notify(
            0,
            "训练计划："
            f"每轮训练 {planned_train_batches} batches（batch={train_loader.batch_size}）；"
            f"每种测试模式 {planned_eval_batches} batches（batch={test_loader.batch_size}），"
            "训练前后共 6 次测试；"
            f"每轮验证 {planned_validation_batches} batches",
            {
                "train_batches_per_epoch": planned_train_batches,
                "train_batch_size": train_loader.batch_size,
                "eval_batches_per_mode": planned_eval_batches,
                "eval_batch_size": test_loader.batch_size,
            },
        )
        error_model = HardwareErrorModel.load(resources.error_model)
        ideal_lut = error_model.effective_lut(0.0)
        device_lut = error_model.effective_lut(request.error_strength)
        from .layers import configure_model_nonidealities

        def ideal_nonidealities(model):
            return configure_model_nonidealities(model, adc_bits=request.adc_bits, noise_seed=request.seed)

        def device_nonidealities(model):
            return configure_model_nonidealities(
                model, adc_bits=request.adc_bits, noise_sigma=request.noise_sigma,
                noise_rho=request.noise_rho, noise_seed=request.seed,
            )

        initial_memory = runtime_memory_snapshot(resources.device)
        notify(
            1,
            "资源自适应："
            f"RAM 可用 {initial_memory['ram_available_bytes'] / 1024 ** 3:.2f} GiB；"
            + (
                f"显存可用 {initial_memory['free_bytes'] / 1024 ** 3:.2f} GiB，"
                "目标使用率 90%"
                if resources.device.type == "cuda" else "CPU 训练"
            ),
            None,
        )
        notify(1, f"Evaluating {resources.model_id} baseline", None)
        def evaluate_with_progress(
            model, loader, mode: str, label: str, start_percent: int, end_percent: int,
        ):
            def report(done: int, total: int) -> None:
                percent = start_percent + round((end_percent - start_percent) * done / max(total, 1))
                notify(percent, f"Evaluating {label}: {done}/{total} batches", None)
            return self._evaluate(
                model, loader, resources.device, mode, cancel, request.max_eval_batches,
                batch_progress=report,
            )

        before = {
            "float": evaluate_with_progress(base(), test_loader, "float", "baseline float", 1, 3),
            "ideal_differential": evaluate_with_progress(
                ideal_nonidealities(evaluation_differential(base(), ideal_lut, False)), test_loader,
                "ideal_differential",
                "baseline ideal differential", 3, 6,
            ),
            "device_differential": evaluate_with_progress(
                device_nonidealities(evaluation_differential(base(), device_lut, False)), test_loader,
                "device_differential",
                "baseline device differential", 6, 9,
            ),
        }

        model = device_nonidealities(training_differential(base(), device_lut, True)).to(resources.device)
        training_initial_state = deepcopy(model.state_dict())
        trainable_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        if not trainable_names:
            raise RuntimeError(f"{resources.model_id} has no trainable parameters")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=request.learning_rate,
        )
        history: list[dict] = []
        best_accuracy = -1.0
        best_state: dict[str, torch.Tensor] = {}
        per_epoch = len(train_loader)
        if request.max_train_batches is not None:
            per_epoch = min(per_epoch, request.max_train_batches)
        total_batches = per_epoch * request.epochs
        completed_batches = 0
        adaptive_train_micro_batch = max(1, int(train_loader.batch_size))
        minimum_train_micro_batch = adaptive_train_micro_batch
        maximum_train_micro_batch = adaptive_train_micro_batch
        training_oom_retries = 0
        for epoch in range(request.epochs):
            model.train().to(resources.device)
            total_loss = 0.0
            correct = samples = 0
            for batch_index, (inputs, targets) in enumerate(train_loader):
                if request.max_train_batches is not None and batch_index >= request.max_train_batches:
                    break
                self._cancelled(cancel)
                non_blocking = resources.device.type == "cuda"
                inputs = inputs.to(resources.device, non_blocking=non_blocking)
                targets = targets.to(resources.device, non_blocking=non_blocking)
                count = targets.numel()
                micro_batch = min(count, adaptive_train_micro_batch)
                while True:
                    baseline_allocated = 0
                    if resources.device.type == "cuda":
                        baseline_allocated = int(torch.cuda.memory_allocated(resources.device))
                        torch.cuda.reset_peak_memory_stats(resources.device)
                    optimizer.zero_grad(set_to_none=True)
                    batch_loss = 0.0
                    batch_correct = 0
                    try:
                        for micro_start in range(0, count, micro_batch):
                            micro_stop = min(micro_start + micro_batch, count)
                            micro_inputs = inputs[micro_start:micro_stop]
                            micro_targets = targets[micro_start:micro_stop]
                            logits = model(micro_inputs)
                            loss = criterion(logits, micro_targets)
                            if not torch.isfinite(loss):
                                raise RuntimeError("Non-finite loss in error-aware training")
                            micro_count = micro_targets.numel()
                            (loss * (micro_count / count)).backward()
                            batch_loss += float(loss.detach()) * micro_count
                            batch_correct += int((logits.argmax(1) == micro_targets).sum())
                        optimizer.step()
                        if resources.device.type == "cuda":
                            peak_allocated = int(torch.cuda.max_memory_allocated(resources.device))
                            tuned = empirical_cuda_batch_size(
                                int(train_loader.batch_size), micro_batch, resources.device,
                                baseline_allocated_bytes=baseline_allocated,
                                peak_allocated_bytes=peak_allocated,
                                processed_samples=micro_batch,
                            )
                            adaptive_train_micro_batch = tuned
                            maximum_train_micro_batch = max(maximum_train_micro_batch, tuned)
                        break
                    except BaseException as exc:
                        optimizer.zero_grad(set_to_none=True)
                        if (resources.device.type != "cuda" or not is_cuda_oom(exc)
                                or micro_batch <= 1):
                            raise
                        micro_batch = max(1, micro_batch // 2)
                        adaptive_train_micro_batch = micro_batch
                        minimum_train_micro_batch = min(minimum_train_micro_batch, micro_batch)
                        training_oom_retries += 1
                        release_cuda_memory()
                        notify(
                            10 + round(65 * completed_batches / max(total_batches, 1)),
                            f"显存动态调整：训练 micro-batch 降为 {micro_batch}", None,
                        )
                total_loss += batch_loss
                correct += batch_correct
                samples += count
                completed_batches += 1
                if completed_batches == 1 or completed_batches % 20 == 0:
                    notify(
                        10 + round(65 * completed_batches / max(total_batches, 1)),
                        f"Epoch {epoch + 1}/{request.epochs}: {completed_batches}/{total_batches} batches",
                        {"train_loss": total_loss / samples,
                         "train_accuracy": correct / samples, "samples": samples},
                    )
            if samples == 0:
                raise RuntimeError("No training samples were processed")
            validation = self._evaluate(
                model, validation_loader, resources.device, "device_differential", cancel,
                request.max_eval_batches,
                batch_progress=lambda done, total, current=epoch + 1: notify(
                    75, f"Epoch {current}/{request.epochs} validation: {done}/{total} batches", None
                ),
            )
            record = {
                "epoch": epoch + 1, "train_loss": total_loss / samples,
                "train_accuracy": correct / samples,
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "train_samples": samples, "validation_samples": validation["samples"],
            }
            history.append(record)
            notify(75, f"Epoch {epoch + 1} complete", record)
            if validation["accuracy"] > best_accuracy:
                best_accuracy = validation["accuracy"]
                best_state = deepcopy(model.state_dict())

        self._cancelled(cancel)
        notify(78, f"Evaluating trained {resources.model_id} weights", None)

        def trained(lut=None):
            trained_model = base()
            trained_model.load_state_dict(best_state, strict=False)
            if lut is not None:
                trained_model = evaluation_differential(trained_model, lut, False)
                trained_model.load_state_dict(best_state, strict=False)
            return trained_model

        after = {
            "float": evaluate_with_progress(
                trained(), test_loader, "float", "trained float", 78, 85,
            ),
            "ideal_differential": evaluate_with_progress(
                ideal_nonidealities(trained(ideal_lut)), test_loader, "ideal_differential",
                "trained ideal differential", 85, 92,
            ),
            "device_differential": evaluate_with_progress(
                device_nonidealities(trained(device_lut)), test_loader, "device_differential",
                "trained device differential", 92, 99,
            ),
        }
        final_parameters = {
            name: value for name, value in best_state.items()
            if name in training_initial_state and name in trainable_names
        }
        delta_sq = 0.0
        max_delta = 0.0
        changed = 0
        for name, initial in training_initial_state.items():
            if name not in final_parameters or name not in trainable_names or not torch.is_floating_point(initial):
                continue
            delta = final_parameters[name].detach().cpu() - initial.detach().cpu()
            delta_sq += float(torch.sum(delta.double().square()))
            maximum = float(delta.abs().max())
            max_delta = max(max_delta, maximum)
            changed += int(maximum > 0)
        weight_update = {
            "source_checkpoint_sha256": _state_digest(initial_state),
            "initial_sha256": _state_digest({
                name: value for name, value in training_initial_state.items()
                if name in trainable_names
            }),
            "trained_sha256": _state_digest(final_parameters),
            "l2_delta": delta_sq ** 0.5, "max_abs_delta": max_delta,
            "changed_parameter_tensors": changed,
        }

        checkpoint_path = output / "eat_checkpoint.pt"
        history_path = output / "history.json"
        result_path = output / "training_result.json"
        torch.save({
            "model_state_dict": model.state_dict(), "best_model_state_dict": best_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": {
                "training": "signed differential device-LUT forward with float STE backward",
                "model_id": resources.model_id, "dataset": resources.dataset_name,
                "lambda": request.error_strength,
                "error_model": str(resources.error_model), "history": history,
            },
        }, checkpoint_path)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        request_payload = asdict(request)
        for key, value in list(request_payload.items()):
            if isinstance(value, Path):
                request_payload[key] = str(value)
        payload = {
            "status": "completed", "model_id": resources.model_id,
            "dataset": resources.dataset_name,
            "formal_evaluation": request.max_eval_batches is None,
            "config": request_payload,
            "resolved_resources": {
                "checkpoint": str(resources.checkpoint),
                "auxiliary": str(resources.auxiliary) if resources.auxiliary else None,
                "error_model": str(resources.error_model),
                "dataset_root": str(resources.dataset_root),
            },
            "training_strategy": {
                "scope": (
                    "differential_sequence_pooling_and_classifier_head_training; "
                    "full_network_differential_evaluation"
                    if resources.model_id == "tinytransformer-cifar10" else "full_network"
                ),
                "trainable_parameter_names": sorted(trainable_names),
                "requested_batch_size": request.batch_size,
                "effective_train_batch_size": train_loader.batch_size,
                "effective_eval_batch_size": test_loader.batch_size,
                "adaptive_train_micro_batch_min": minimum_train_micro_batch,
                "adaptive_train_micro_batch_max": maximum_train_micro_batch,
                "cuda_oom_retries": training_oom_retries,
                "initial_memory": initial_memory,
            },
            "device": str(resources.device), "elapsed_seconds": perf_counter() - started,
            "history": history, "evaluations": {"before": before, "after": after},
            "weight_update": weight_update,
            "artifacts": {"checkpoint": str(checkpoint_path), "history": str(history_path)},
        }
        result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        notify(100, "Training and evaluation completed", history[-1])
        return TrainingOutcome(
            str(checkpoint_path), str(history_path), str(result_path), str(output),
            str(resources.device), payload["elapsed_seconds"], tuple(history),
            payload["evaluations"], weight_update,
        )

    def _run_gpt2_small(
        self, request: TrainingRequest, resources: TrainingResources,
        notify: ProgressCallback, cancel: CancelCallback | None,
    ) -> TrainingOutcome:
        """Fine-tune GPT-2 Small's two deployed LUT projection layers on WikiText-2."""
        from torch.utils.data import DataLoader, Dataset

        from .error_model import HardwareErrorModel
        from .gpt2_evaluation import load_wikitext2_raw
        from .gpt2_resources import load_checkpoint_into_model

        self._add_import_root(resources.model_root)
        from gpt2_cim import gpt2_for_variant
        from gpt2_cim.engine import configure_differential_heads, differential_layers_for
        from transformers import GPT2Tokenizer

        class CausalWindows(Dataset):
            def __init__(self, token_ids: list[int], length: int):
                self.tokens = torch.tensor(token_ids, dtype=torch.long)
                self.length = length
                self.windows = max(0, (self.tokens.numel() - 1) // length)

            def __len__(self) -> int:
                return self.windows

            def __getitem__(self, index: int):
                begin = index * self.length
                window = self.tokens[begin:begin + self.length + 1]
                return window[:-1], window[1:]

        started = perf_counter()
        output = Path(request.output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        tokenizer_root = resources.model_root / "checkpoint/tokenizer"
        if not tokenizer_root.is_dir():
            raise FileNotFoundError(f"GPT-2 tokenizer directory is missing: {tokenizer_root}")
        tokenizer = GPT2Tokenizer.from_pretrained(str(tokenizer_root))
        # The corpus is intentionally tokenized once and split into short model
        # windows afterwards; suppress the tokenizer's single-sequence warning.
        tokenizer.model_max_length = 2**31 - 1
        split_tokens: dict[str, list[int]] = {}
        for split in ("train", "validation", "test"):
            self._cancelled(cancel)
            _path, text = load_wikitext2_raw(resources.dataset_root, split)
            split_tokens[split] = tokenizer.encode(text)
            if len(split_tokens[split]) <= request.sequence_length:
                raise ValueError(
                    f"WikiText-2 {split} split has too few tokens for sequence_length="
                    f"{request.sequence_length}"
                )
        generator = torch.Generator().manual_seed(request.seed)
        train_loader = DataLoader(
            CausalWindows(split_tokens["train"], request.sequence_length),
            batch_size=min(request.batch_size, 2), shuffle=True, generator=generator,
            num_workers=0,
        )
        validation_loader = DataLoader(
            CausalWindows(split_tokens["validation"], request.sequence_length),
            batch_size=1, shuffle=False, num_workers=0,
        )
        test_loader = DataLoader(
            CausalWindows(split_tokens["test"], request.sequence_length),
            batch_size=1, shuffle=False, num_workers=0,
        )
        layer_names = differential_layers_for("gpt2-small")
        scales = json.loads(resources.auxiliary.read_text(encoding="utf-8"))
        if any(name not in scales for name in layer_names):
            raise ValueError(f"GPT-2 input scales are missing deployed layers: {layer_names}")
        error_model = HardwareErrorModel.load(resources.error_model)
        ideal_lut = error_model.effective_lut(0.0)
        device_lut = error_model.effective_lut(request.error_strength)

        def base_model():
            model = gpt2_for_variant("gpt2-small")
            load_checkpoint_into_model(model, resources.checkpoint)
            return model

        def apply_trainable_state(model: nn.Module, state: dict[str, torch.Tensor] | None) -> None:
            if not state:
                return
            parameters = dict(model.named_parameters())
            with torch.no_grad():
                for name, value in state.items():
                    parameters[name].copy_(value.to(parameters[name].device, parameters[name].dtype))

        def differential_model(lut, *, ste: bool, trained=None, device_noise: bool = False):
            model = base_model()
            apply_trainable_state(model, trained)
            configure_differential_heads(
                model, lut, scales, ste=ste, layer_names=layer_names,
            )
            from .layers import configure_model_nonidealities
            configure_model_nonidealities(
                model, adc_bits=request.adc_bits,
                noise_sigma=request.noise_sigma if device_noise else 0.0,
                noise_rho=request.noise_rho if device_noise else 0.0,
                noise_seed=request.seed,
            )
            return model

        def set_training_scope(model: nn.Module) -> tuple[str, ...]:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            modules = dict(model.named_modules())
            for name in layer_names:
                layer = modules.get(name)
                if layer is None:
                    raise RuntimeError(f"GPT-2 fine-tuning layer is missing: {name}")
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
            return tuple(
                name for name, parameter in model.named_parameters() if parameter.requires_grad
            )

        def evaluate(model: nn.Module, loader, mode: str, label: str,
                     start_percent: int, end_percent: int) -> dict:
            model.eval().to(resources.device)
            total_loss = 0.0
            correct = tokens = batches = 0
            limit = request.max_eval_batches
            total_batches = len(loader) if limit is None else min(len(loader), limit)
            with torch.inference_mode():
                for batch_index, (inputs, targets) in enumerate(loader):
                    if limit is not None and batch_index >= limit:
                        break
                    self._cancelled(cancel)
                    inputs, targets = inputs.to(resources.device), targets.to(resources.device)
                    logits, loss = model(inputs, targets)
                    count = targets.numel()
                    total_loss += float(loss) * count
                    correct += int((logits.argmax(-1) == targets).sum())
                    tokens += count
                    batches += 1
                    if batches == 1 or batches % 10 == 0 or batches == total_batches:
                        percent = start_percent + round(
                            (end_percent - start_percent) * batches / max(total_batches, 1)
                        )
                        notify(percent, f"Evaluating {label}: {batches}/{total_batches} batches", None)
            model.cpu()
            if resources.device.type == "cuda":
                torch.cuda.empty_cache()
            if tokens == 0:
                raise RuntimeError("GPT-2 evaluation processed no target tokens")
            average = total_loss / tokens
            return {
                "mode": mode, "loss": average,
                "perplexity": math.exp(min(average, 50.0)),
                "accuracy": correct / tokens, "token_accuracy": correct / tokens,
                "correct": correct, "samples": tokens, "batches": batches,
            }

        notify(1, "Evaluating GPT-2 Small baseline", None)
        baseline_float = base_model()
        before_float = evaluate(baseline_float, test_loader, "float", "baseline float", 1, 3)
        del baseline_float
        baseline_ideal = differential_model(ideal_lut, ste=False)
        before_ideal = evaluate(
            baseline_ideal, test_loader, "ideal_differential", "baseline ideal differential", 3, 6,
        )
        del baseline_ideal
        baseline_device = differential_model(device_lut, ste=False, device_noise=True)
        before_device = evaluate(
            baseline_device, test_loader, "device_differential", "baseline device differential", 6, 9,
        )
        del baseline_device
        before = {
            "float": before_float,
            "ideal_differential": before_ideal,
            "device_differential": before_device,
        }

        model = differential_model(device_lut, ste=True, device_noise=True).to(resources.device)
        trainable_names = set_training_scope(model)
        initial_trainable = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=request.learning_rate,
        )
        history: list[dict] = []
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] = {}
        per_epoch = len(train_loader)
        if request.max_train_batches is not None:
            per_epoch = min(per_epoch, request.max_train_batches)
        total_batches = per_epoch * request.epochs
        completed_batches = 0
        accumulation = request.gradient_accumulation_steps
        for epoch in range(request.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            correct = tokens = epoch_batches = 0
            for batch_index, (inputs, targets) in enumerate(train_loader):
                if request.max_train_batches is not None and batch_index >= request.max_train_batches:
                    break
                self._cancelled(cancel)
                inputs, targets = inputs.to(resources.device), targets.to(resources.device)
                logits, loss = model(inputs, targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss in GPT-2 error-aware training")
                (loss / accumulation).backward()
                epoch_batches += 1
                if epoch_batches % accumulation == 0 or epoch_batches == per_epoch:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                count = targets.numel()
                total_loss += float(loss.detach()) * count
                correct += int((logits.argmax(-1) == targets).sum())
                tokens += count
                completed_batches += 1
                if completed_batches == 1 or completed_batches % 10 == 0:
                    notify(
                        10 + round(60 * completed_batches / max(total_batches, 1)),
                        f"GPT-2 epoch {epoch + 1}/{request.epochs}: "
                        f"{completed_batches}/{total_batches} batches",
                        {"train_loss": total_loss / tokens,
                         "train_accuracy": correct / tokens, "samples": tokens},
                    )
            if tokens == 0:
                raise RuntimeError("No GPT-2 training tokens were processed")
            validation = evaluate(
                model, validation_loader, "device_differential", "GPT-2 validation", 70, 75,
            )
            record = {
                "epoch": epoch + 1, "train_loss": total_loss / tokens,
                "train_perplexity": math.exp(min(total_loss / tokens, 50.0)),
                "train_token_accuracy": correct / tokens,
                "validation_loss": validation["loss"],
                "validation_perplexity": validation["perplexity"],
                "validation_token_accuracy": validation["token_accuracy"],
                "train_tokens": tokens, "validation_tokens": validation["samples"],
            }
            history.append(record)
            notify(75, f"GPT-2 epoch {epoch + 1} complete", record)
            if validation["loss"] < best_loss:
                best_loss = validation["loss"]
                best_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.named_parameters()
                    if name in trainable_names
                }
        del model
        if resources.device.type == "cuda":
            torch.cuda.empty_cache()

        trained_float = base_model(); apply_trainable_state(trained_float, best_state)
        after_float = evaluate(trained_float, test_loader, "float", "trained float", 78, 85)
        del trained_float
        trained_ideal = differential_model(ideal_lut, ste=False, trained=best_state)
        after_ideal = evaluate(
            trained_ideal, test_loader, "ideal_differential", "trained ideal differential", 85, 92,
        )
        del trained_ideal
        trained_device = differential_model(
            device_lut, ste=False, trained=best_state, device_noise=True
        )
        after_device = evaluate(
            trained_device, test_loader, "device_differential", "trained device differential", 92, 99,
        )
        del trained_device
        after = {
            "float": after_float,
            "ideal_differential": after_ideal,
            "device_differential": after_device,
        }

        delta_sq = 0.0
        max_delta = 0.0
        changed = 0
        for name, initial in initial_trainable.items():
            delta = best_state[name] - initial
            delta_sq += float(delta.double().square().sum())
            maximum = float(delta.abs().max())
            max_delta = max(max_delta, maximum)
            changed += int(maximum > 0)
        weight_update = {
            "source_checkpoint_sha256": _state_digest(_checkpoint_state(resources.checkpoint)),
            "initial_sha256": _state_digest(initial_trainable),
            "trained_sha256": _state_digest(best_state),
            "l2_delta": delta_sq ** 0.5, "max_abs_delta": max_delta,
            "changed_parameter_tensors": changed,
            "trainable_parameter_count": sum(value.numel() for value in best_state.values()),
        }
        checkpoint_path = output / "gpt2_small_eat_checkpoint.pt"
        history_path = output / "history.json"
        result_path = output / "training_result.json"
        deployment = base_model(); apply_trainable_state(deployment, best_state)
        torch.save({
            "model_state_dict": deployment.state_dict(),
            "best_model_state_dict": deployment.state_dict(),
            "trainable_state_dict": best_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": {
                "training": "GPT-2 Small final projections; device-LUT STE",
                "model_id": resources.model_id, "dataset": resources.dataset_name,
                "lambda": request.error_strength, "differential_layers": layer_names,
                "history": history,
            },
        }, checkpoint_path)
        del deployment
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        request_payload = asdict(request)
        for key, value in list(request_payload.items()):
            if isinstance(value, Path):
                request_payload[key] = str(value)
        elapsed = perf_counter() - started
        payload = {
            "status": "completed", "model_id": resources.model_id,
            "dataset": resources.dataset_name,
            "formal_evaluation": request.max_eval_batches is None,
            "config": request_payload,
            "resolved_resources": {
                "checkpoint": str(resources.checkpoint), "auxiliary": str(resources.auxiliary),
                "error_model": str(resources.error_model),
                "dataset_root": str(resources.dataset_root), "tokenizer": str(tokenizer_root),
            },
            "training_strategy": {
                "scope": "last_transformer_block_attention_and_mlp_output_projections",
                "differential_layers": list(layer_names),
                "trainable_parameter_names": sorted(trainable_names),
                "trainable_parameter_count": weight_update["trainable_parameter_count"],
                "effective_train_batch_size": train_loader.batch_size,
                "gradient_accumulation_steps": accumulation,
                "sequence_length": request.sequence_length,
            },
            "device": str(resources.device), "elapsed_seconds": elapsed,
            "history": history, "evaluations": {"before": before, "after": after},
            "weight_update": weight_update,
            "artifacts": {"checkpoint": str(checkpoint_path), "history": str(history_path)},
        }
        result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        notify(100, "GPT-2 Small training and evaluation completed", history[-1])
        return TrainingOutcome(
            str(checkpoint_path), str(history_path), str(result_path), str(output),
            str(resources.device), elapsed, tuple(history), payload["evaluations"], weight_update,
        )

    def _evaluate(
        self, model, loader, device, mode: str, cancel: CancelCallback | None,
        max_batches: int | None = None,
        batch_progress: Callable[[int, int], None] | None = None,
    ) -> dict:
        from .runtime_memory import (
            empirical_cuda_batch_size, is_cuda_oom, release_cuda_memory,
        )

        model.eval().to(device)
        criterion = nn.CrossEntropyLoss(reduction="sum")
        loss = 0.0
        correct = samples = 0
        confusion = torch.zeros((10, 10), dtype=torch.int64)
        total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
        adaptive_micro_batch = max(1, int(loader.batch_size))
        minimum_micro_batch = adaptive_micro_batch
        maximum_micro_batch = adaptive_micro_batch
        oom_retries = 0
        with torch.inference_mode():
            for batch_index, (inputs, targets) in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                self._cancelled(cancel)
                cuda_execution = torch.device(device).type == "cuda"
                inputs = inputs.to(device, non_blocking=cuda_execution)
                targets = targets.to(device, non_blocking=cuda_execution)
                micro_batch = min(targets.numel(), adaptive_micro_batch)
                baseline_allocated = 0
                if cuda_execution:
                    baseline_allocated = int(torch.cuda.memory_allocated(device))
                    torch.cuda.reset_peak_memory_stats(device)
                micro_start = 0
                while micro_start < targets.numel():
                    micro_stop = min(micro_start + micro_batch, targets.numel())
                    micro_inputs = inputs[micro_start:micro_stop]
                    micro_targets = targets[micro_start:micro_stop]
                    try:
                        logits = model(micro_inputs)
                    except BaseException as exc:
                        if (torch.device(device).type != "cuda" or not is_cuda_oom(exc)
                                or micro_batch <= 1):
                            raise
                        micro_batch = max(1, micro_batch // 2)
                        adaptive_micro_batch = micro_batch
                        minimum_micro_batch = min(minimum_micro_batch, micro_batch)
                        oom_retries += 1
                        release_cuda_memory()
                        continue
                    loss += float(criterion(logits, micro_targets))
                    predictions = logits.argmax(1)
                    correct += int((predictions == micro_targets).sum())
                    indices = (micro_targets * 10 + predictions).detach().cpu()
                    confusion += torch.bincount(indices, minlength=100).reshape(10, 10)
                    samples += micro_targets.numel()
                    micro_start = micro_stop
                if cuda_execution:
                    peak_allocated = int(torch.cuda.max_memory_allocated(device))
                    adaptive_micro_batch = empirical_cuda_batch_size(
                        int(loader.batch_size), micro_batch, device,
                        baseline_allocated_bytes=baseline_allocated,
                        peak_allocated_bytes=peak_allocated,
                        processed_samples=micro_batch,
                    )
                    maximum_micro_batch = max(maximum_micro_batch, adaptive_micro_batch)
                completed = batch_index + 1
                if batch_progress and (completed == 1 or completed % 10 == 0 or completed == total_batches):
                    batch_progress(completed, total_batches)
        if samples == 0:
            raise RuntimeError("No evaluation samples were processed")
        class_totals = confusion.sum(1)
        return {
            "mode": mode, "loss": loss / samples, "accuracy": correct / samples,
            "correct": correct, "samples": samples,
            "adaptive_micro_batch_min": minimum_micro_batch,
            "adaptive_micro_batch_max": maximum_micro_batch,
            "cuda_oom_retries": oom_retries,
            "per_class_accuracy": (
                confusion.diag().float() / class_totals.clamp_min(1)
            ).tolist(),
            "confusion_matrix": confusion.tolist(),
        }

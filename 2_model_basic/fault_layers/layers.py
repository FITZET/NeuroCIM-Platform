from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol, TypeAlias, TypeVar, runtime_checkable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@runtime_checkable
class ErrorModelProtocol(Protocol):
    """Minimal interface accepted by a fault layer."""

    def __call__(self, output: Tensor) -> Tensor: ...


ErrorModel: TypeAlias = nn.Module | Callable[[Tensor], Tensor]
FaultLayerT = TypeVar("FaultLayerT", bound="FaultLayerMixin")


class FaultLayerMixin:
    """Shared output-feature-map error injection behavior.

    This mixin must be combined with ``torch.nn.Module``. The normal operator is
    always evaluated first. When enabled, ``error_model(normal_output)`` becomes
    the layer output. No quantization or LUT semantics are assumed here; those
    belong to the externally supplied error model.
    """

    fault_enable: bool
    strict_output: bool
    error_model: ErrorModel | None

    def _init_fault_layer(
        self,
        *,
        fault_enable: bool = False,
        error_model: ErrorModel | None = None,
        strict_output: bool = True,
        mac_chunk_size: int = 32,
    ) -> None:
        self.fault_enable = bool(fault_enable)
        self.strict_output = bool(strict_output)
        if mac_chunk_size <= 0:
            raise ValueError("mac_chunk_size must be positive")
        self.mac_chunk_size = int(mac_chunk_size)
        self.error_model = None
        # Non-persistent buffers follow layer device moves but do not break
        # compatibility with an original Conv2d/Linear state_dict.
        self.register_buffer("hardware_lut", torch.empty(0, dtype=torch.int64), persistent=False)
        self.register_buffer("input_scale", torch.tensor(1.0), persistent=False)
        self.register_buffer("input_zero_point", torch.tensor(0, dtype=torch.int64), persistent=False)
        self.register_buffer("weight_scale", torch.tensor([1.0]), persistent=False)
        self.register_buffer("weight_zero_point", torch.tensor([0], dtype=torch.int64), persistent=False)
        self.register_buffer(
            "_cached_quantized_weight", torch.empty(0, dtype=torch.int64), persistent=False
        )
        self._cached_weight_version = -1
        self._ideal_product_lut = False
        if error_model is not None:
            self.set_error_model(error_model)

    def enable_fault(self, enabled: bool = True) -> FaultLayerMixin:
        """Enable/disable injection and return ``self`` for controller use."""

        self.fault_enable = bool(enabled)
        return self

    def disable_fault(self) -> FaultLayerMixin:
        return self.enable_fault(False)

    def set_error_model(self, error_model: ErrorModel | None) -> FaultLayerMixin:
        """Attach a callable ``Tensor -> Tensor`` error model.

        ``nn.Module`` instances are registered as child modules, so their state,
        device and training mode participate in normal PyTorch behavior.
        """

        if error_model is not None and not callable(error_model):
            raise TypeError("error_model must be callable or None")

        # Removing first permits switching safely between nn.Module and a plain
        # callable; nn.Module otherwise reserves registered child names.
        if hasattr(self, "error_model"):
            delattr(self, "error_model")
        self.error_model = error_model
        return self

    def clear_error_model(self) -> FaultLayerMixin:
        return self.set_error_model(None)

    @property
    def fault_active(self) -> bool:
        return self.fault_enable and (self.error_model is not None or self.has_hardware_lut)

    @property
    def has_hardware_lut(self) -> bool:
        return self.hardware_lut.numel() == 256 * 256

    def configure_hardware_lut(
        self,
        lut: Tensor,
        *,
        input_scale: float,
        input_zero_point: int,
        weight_scale: float | Tensor,
        weight_zero_point: int | Tensor,
    ) -> FaultLayerMixin:
        """Configure per-multiplication uint8 affine LUT computation.

        ``lut`` must use the data-fitting export convention ``lut[W, X]``.
        Weight scale/zero-point may be scalar or per-output-channel vectors.
        """

        lut_tensor = torch.as_tensor(lut, device=self.weight.device)
        if lut_tensor.shape != (256, 256):
            raise ValueError(f"lut must have shape (256, 256), got {tuple(lut_tensor.shape)}")
        if not torch.isfinite(lut_tensor).all():
            raise ValueError("lut contains NaN or infinite values")
        if input_scale <= 0:
            raise ValueError("input_scale must be positive")
        if not 0 <= int(input_zero_point) <= 255:
            raise ValueError("input_zero_point must be in [0, 255]")

        ws = torch.as_tensor(weight_scale, dtype=torch.float64, device=self.weight.device).reshape(-1)
        wz = torch.as_tensor(weight_zero_point, dtype=torch.int64, device=self.weight.device).reshape(-1)
        output_channels = self.weight.shape[0]
        if ws.numel() not in (1, output_channels):
            raise ValueError("weight_scale must be scalar or have one value per output channel")
        if wz.numel() not in (1, output_channels):
            raise ValueError("weight_zero_point must be scalar or have one value per output channel")
        if torch.any(ws <= 0):
            raise ValueError("all weight_scale values must be positive")
        if torch.any((wz < 0) | (wz > 255)):
            raise ValueError("weight_zero_point values must be in [0, 255]")

        table = torch.round(lut_tensor).to(torch.int64)
        if (int(table.min()) >= torch.iinfo(torch.int32).min
                and int(table.max()) <= torch.iinfo(torch.int32).max):
            table = table.to(torch.int32)
        self.hardware_lut = table
        codes = torch.arange(256, dtype=torch.int64, device=self.weight.device)
        self._ideal_product_lut = bool(torch.equal(
            self.hardware_lut.to(torch.int64), codes[:, None] * codes[None, :]
        ))
        self.input_scale = torch.tensor(float(input_scale), dtype=torch.float64, device=self.weight.device)
        self.input_zero_point = torch.tensor(int(input_zero_point), dtype=torch.int64, device=self.weight.device)
        self.weight_scale = ws
        self.weight_zero_point = wz
        self._cached_weight_version = -1
        return self

    def clear_hardware_lut(self) -> FaultLayerMixin:
        self.hardware_lut = torch.empty(0, dtype=torch.int64, device=self.weight.device)
        self._ideal_product_lut = False
        self._cached_weight_version = -1
        return self

    def configure_hardware_lut_from_csv(
        self,
        path: str | Path,
        *,
        input_scale: float,
        input_zero_point: int,
        weight_scale: float | Tensor,
        weight_zero_point: int | Tensor,
    ) -> FaultLayerMixin:
        """Load the indexed CSV exported by ``1_data_fitting`` and configure it."""

        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        table = np.loadtxt(source, delimiter=",", skiprows=1, usecols=range(1, 257))
        return self.configure_hardware_lut(
            torch.from_numpy(table),
            input_scale=input_scale,
            input_zero_point=input_zero_point,
            weight_scale=weight_scale,
            weight_zero_point=weight_zero_point,
        )

    @staticmethod
    def _quantize_uint8(value: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        return torch.clamp(torch.round(value / scale + zero_point), 0, 255).to(torch.int64)

    def _lut_product(self, qx: Tensor, qw: Tensor, zx: Tensor, zw: Tensor) -> Tensor:
        """Affine zero-point corrected product; LUT indexing is [W, X]."""

        lut = self.hardware_lut
        return lut[qw, qx] - lut[zw, qx] - lut[qw, zx] + lut[zw, zx]

    def _quantized_weight(self, scale: Tensor, zero_point: Tensor) -> Tensor:
        """Cache quantized weights and invalidate after every optimizer update."""

        version = int(self.weight._version)
        if (self._cached_weight_version != version
                or self._cached_quantized_weight.shape != self.weight.shape):
            shape = [self.weight.shape[0]] + [1] * (self.weight.ndim - 1)
            self._cached_quantized_weight = self._quantize_uint8(
                self.weight.detach(),
                scale.to(self.weight.dtype).view(*shape),
                zero_point.view(*shape),
            )
            self._cached_weight_version = version
        return self._cached_quantized_weight

    @staticmethod
    def _ideal_code_mm(left: Tensor, right: Tensor) -> Tensor:
        # Imported lazily so the reusable base layer remains usable without the
        # full platform package in simple floating-point projects.
        from neurocim_opt.layers import _ideal_signed_code_mm
        return _ideal_signed_code_mm(left.to(torch.int16), right.to(torch.int16))

    def _inject_output_error(self, output: Tensor) -> Tensor:
        if not self.fault_enable or self.error_model is None:
            return output
        assert self.error_model is not None
        faulty_output = self.error_model(output)
        if not isinstance(faulty_output, Tensor):
            raise TypeError(
                "error_model must return torch.Tensor, "
                f"got {type(faulty_output).__name__}"
            )
        if self.strict_output and faulty_output.shape != output.shape:
            raise ValueError(
                "error_model changed output shape from "
                f"{tuple(output.shape)} to {tuple(faulty_output.shape)}"
            )
        return faulty_output

    def fault_extra_repr(self) -> str:
        model_name = None if self.error_model is None else type(self.error_model).__name__
        return (
            f"fault_enable={self.fault_enable}, error_model={model_name}, "
            f"hardware_lut={self.has_hardware_lut}, strict_output={self.strict_output}"
        )


class FaultConv2d(FaultLayerMixin, nn.Conv2d):
    """Drop-in ``nn.Conv2d`` with optional output error injection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        *,
        fault_enable: bool = False,
        error_model: ErrorModel | None = None,
        strict_output: bool = True,
    ) -> None:
        nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device=device,
            dtype=dtype,
        )
        self._init_fault_layer(
            fault_enable=fault_enable,
            error_model=error_model,
            strict_output=strict_output,
        )

    def forward(self, input: Tensor) -> Tensor:
        if self.fault_enable and self.has_hardware_lut:
            output = self._hardware_forward(input)
        else:
            output = nn.Conv2d.forward(self, input)
        return self._inject_output_error(output)

    def _hardware_forward(self, input: Tensor) -> Tensor:
        if isinstance(self.padding, str):
            raise NotImplementedError("LUT Conv2d currently requires numeric padding")
        if self.padding_mode != "zeros":
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding
        patches = F.unfold(
            input,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=padding,
            stride=self.stride,
        )
        batch, _, locations = patches.shape
        kernel_elements = self.weight[0].numel()
        out_per_group = self.out_channels // self.groups
        in_per_group_elements = kernel_elements
        accumulator = torch.zeros(
            (batch, self.out_channels, locations), dtype=torch.int64, device=input.device
        )

        zx = self.input_zero_point.to(input.device)
        qx_all = self._quantize_uint8(patches, self.input_scale.to(input.device), zx)
        ws = self.weight_scale.to(input.device)
        wz = self.weight_zero_point.to(input.device)
        if ws.numel() == 1:
            ws = ws.expand(self.out_channels)
        if wz.numel() == 1:
            wz = wz.expand(self.out_channels)
        qw_all = self._quantized_weight(ws, wz).reshape(self.out_channels, kernel_elements)

        for group in range(self.groups):
            out_start = group * out_per_group
            out_stop = out_start + out_per_group
            patch_start = group * in_per_group_elements
            patch_stop = patch_start + in_per_group_elements
            qx_group = qx_all[:, patch_start:patch_stop, :]
            qw_group = qw_all[out_start:out_stop]
            zw_group = wz[out_start:out_stop]
            group_acc = accumulator[:, out_start:out_stop, :]
            if self._ideal_product_lut:
                centered_x = (qx_group - zx).permute(0, 2, 1).reshape(
                    batch * locations, kernel_elements
                )
                centered_w = qw_group - zw_group.view(-1, 1)
                exact = self._ideal_code_mm(centered_x, centered_w)
                group_acc.copy_(exact.reshape(batch, locations, out_per_group).permute(0, 2, 1))
            else:
                for start in range(0, kernel_elements, self.mac_chunk_size):
                    stop = min(start + self.mac_chunk_size, kernel_elements)
                    qx = qx_group[:, start:stop, :].permute(0, 2, 1).unsqueeze(1)
                    qw = qw_group[:, start:stop].view(1, out_per_group, 1, stop - start)
                    products = self._lut_product(
                        qx, qw, zx, zw_group.view(1, out_per_group, 1, 1)
                    )
                    group_acc.add_(products.sum(dim=-1, dtype=torch.int64))

        output = accumulator.to(input.dtype) * (
            self.input_scale.to(input.device, input.dtype) * ws.to(input.device, input.dtype).view(1, -1, 1)
        )
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)
        height = (input.shape[-2] + 2 * padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        width = (input.shape[-1] + 2 * padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        return output.reshape(batch, self.out_channels, height, width)

    @classmethod
    def from_module(
        cls,
        module: nn.Conv2d,
        *,
        fault_enable: bool = False,
        error_model: ErrorModel | None = None,
        strict_output: bool = True,
    ) -> FaultConv2d:
        """Create an equivalent fault layer and preserve learned state."""

        if not isinstance(module, nn.Conv2d):
            raise TypeError(f"Expected nn.Conv2d, got {type(module).__name__}")
        converted = cls(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None,
            module.padding_mode,
            device=module.weight.device,
            dtype=module.weight.dtype,
            fault_enable=fault_enable,
            error_model=None,
            strict_output=strict_output,
        )
        converted.load_state_dict(module.state_dict(), strict=True)
        if error_model is not None:
            converted.set_error_model(error_model)
        converted.train(module.training)
        converted.weight.requires_grad_(module.weight.requires_grad)
        if converted.bias is not None and module.bias is not None:
            converted.bias.requires_grad_(module.bias.requires_grad)
        return converted

    def extra_repr(self) -> str:
        return f"{nn.Conv2d.extra_repr(self)}, {self.fault_extra_repr()}"


class FaultLinear(FaultLayerMixin, nn.Linear):
    """Drop-in ``nn.Linear`` with optional output error injection."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        *,
        fault_enable: bool = False,
        error_model: ErrorModel | None = None,
        strict_output: bool = True,
    ) -> None:
        nn.Linear.__init__(
            self,
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self._init_fault_layer(
            fault_enable=fault_enable,
            error_model=error_model,
            strict_output=strict_output,
        )

    def forward(self, input: Tensor) -> Tensor:
        if self.fault_enable and self.has_hardware_lut:
            output = self._hardware_forward(input)
        else:
            output = nn.Linear.forward(self, input)
        return self._inject_output_error(output)

    def _hardware_forward(self, input: Tensor) -> Tensor:
        original_shape = input.shape[:-1]
        flat_input = input.reshape(-1, self.in_features)
        zx = self.input_zero_point.to(input.device)
        qx = self._quantize_uint8(flat_input, self.input_scale.to(input.device), zx)
        ws = self.weight_scale.to(input.device)
        wz = self.weight_zero_point.to(input.device)
        if ws.numel() == 1:
            ws = ws.expand(self.out_features)
        if wz.numel() == 1:
            wz = wz.expand(self.out_features)
        qw = self._quantized_weight(ws, wz)
        if self._ideal_product_lut:
            accumulator = self._ideal_code_mm(qx - zx, qw - wz.view(-1, 1))
        else:
            accumulator = torch.zeros(
                (flat_input.shape[0], self.out_features), dtype=torch.int64, device=input.device
            )
            for start in range(0, self.in_features, self.mac_chunk_size):
                stop = min(start + self.mac_chunk_size, self.in_features)
                accumulator.add_(
                    self._lut_product(
                        qx[:, None, start:stop],
                        qw[None, :, start:stop],
                        zx,
                        wz.view(1, -1, 1),
                    ).sum(dim=-1, dtype=torch.int64)
                )
        output = accumulator.to(input.dtype) * (
            self.input_scale.to(input.device, input.dtype) * ws.to(input.device, input.dtype).view(1, -1)
        )
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*original_shape, self.out_features)

    @classmethod
    def from_module(
        cls,
        module: nn.Linear,
        *,
        fault_enable: bool = False,
        error_model: ErrorModel | None = None,
        strict_output: bool = True,
    ) -> FaultLinear:
        """Create an equivalent fault layer and preserve learned state."""

        if not isinstance(module, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(module).__name__}")
        converted = cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
            fault_enable=fault_enable,
            error_model=None,
            strict_output=strict_output,
        )
        converted.load_state_dict(module.state_dict(), strict=True)
        if error_model is not None:
            converted.set_error_model(error_model)
        converted.train(module.training)
        converted.weight.requires_grad_(module.weight.requires_grad)
        if converted.bias is not None and module.bias is not None:
            converted.bias.requires_grad_(module.bias.requires_grad)
        return converted

    def extra_repr(self) -> str:
        return f"{nn.Linear.extra_repr(self)}, {self.fault_extra_repr()}"

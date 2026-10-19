from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _as_channel_values(value: float | Tensor, channels: int, device: torch.device) -> Tensor:
    result = torch.as_tensor(value, dtype=torch.float64, device=device).reshape(-1)
    if result.numel() == 1:
        result = result.expand(channels)
    if result.numel() != channels or torch.any(result <= 0):
        raise ValueError(f"scale must be positive and scalar or length {channels}")
    return result


def _quantize_magnitude(value: Tensor, scale: Tensor) -> Tensor:
    # Quantize with the high-precision scale even when the model uses FP16.
    # Saturating non-finite magnitudes prevents integer conversion from producing
    # an invalid negative LUT index; infinity has the same ADC-code semantics
    # as any other value above the representable range.
    # Preserve the configured scale's precision so existing quantization
    # boundaries remain bit-for-bit compatible with FP32 execution.
    normalized = torch.abs(value) / scale.to(value.device)
    normalized = torch.nan_to_num(normalized, nan=0.0, posinf=255.0, neginf=0.0)
    return torch.clamp(torch.round(normalized), 0, 255).to(torch.int64)


def _signed_lut_product(lut: Tensor, x: Tensor, w: Tensor, x_scale: Tensor, w_scale: Tensor) -> Tensor:
    """Exact signed product from an interaction LUT whose axes are already zeroed."""

    qx = _quantize_magnitude(x, x_scale)
    qw = _quantize_magnitude(w, w_scale)
    magnitude = lut[qw, qx]
    safe_x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    safe_w = torch.nan_to_num(w, nan=0.0, posinf=1.0, neginf=-1.0)
    sign = torch.sign(safe_x).to(torch.int64) * torch.sign(safe_w).to(torch.int64)
    return magnitude * sign


def _ste(exact: Tensor, surrogate: Tensor) -> Tensor:
    """Use exact forward values and the floating operation's backward derivative."""

    return surrogate + (exact - surrogate).detach()


class DifferentialLayerMixin:
    """Shared configuration for signed differential CIM layers.

    The supplied table is an interaction-product table in `[W, X]` order. Its
    zero axes must already be zero, so signed branches do not repeat the device
    current baseline extraction performed by the fitting backend.
    """

    def _init_differential(
        self,
        lut: Tensor,
        *,
        input_scale: float,
        weight_scale: float | Tensor,
        ste: bool,
        mac_chunk_size: int,
    ) -> None:
        table = torch.as_tensor(lut, device=self.weight.device)
        if table.shape != (256, 256):
            raise ValueError(f"lut must have shape (256, 256), got {tuple(table.shape)}")
        if not torch.isfinite(table).all():
            raise ValueError("lut contains NaN or infinite values")
        table = torch.round(table).to(torch.int64)
        if torch.any(table[0] != 0) or torch.any(table[:, 0] != 0):
            raise ValueError("differential layers require an interaction LUT with zero X/W axes")
        if input_scale <= 0:
            raise ValueError("input_scale must be positive")
        if mac_chunk_size <= 0:
            raise ValueError("mac_chunk_size must be positive")
        self.register_buffer("hardware_lut", table, persistent=False)
        self.register_buffer(
            "input_scale", torch.tensor(float(input_scale), dtype=torch.float64), persistent=False
        )
        self.register_buffer(
            "weight_scale",
            _as_channel_values(weight_scale, self.weight.shape[0], self.weight.device),
            persistent=False,
        )
        self.ste = bool(ste)
        self.mac_chunk_size = int(mac_chunk_size)

    def _scale_for_output(self, dtype: torch.dtype, dimensions: int) -> Tensor:
        shape = [1, -1] + [1] * (dimensions - 2)
        return (
            self.input_scale.to(self.weight.device, dtype)
            * self.weight_scale.to(self.weight.device, dtype).view(*shape)
        )


class DifferentialLinear(DifferentialLayerMixin, nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        lut: Tensor,
        input_scale: float,
        weight_scale: float | Tensor,
        bias: bool = True,
        ste: bool = False,
        mac_chunk_size: int = 64,
        device=None,
        dtype=None,
    ) -> None:
        nn.Linear.__init__(self, in_features, out_features, bias, device=device, dtype=dtype)
        self._init_differential(
            lut,
            input_scale=input_scale,
            weight_scale=weight_scale,
            ste=ste,
            mac_chunk_size=mac_chunk_size,
        )

    @classmethod
    def from_module(
        cls,
        module: nn.Linear,
        *,
        lut: Tensor,
        input_scale: float,
        weight_scale: float | Tensor | None = None,
        ste: bool = False,
    ) -> "DifferentialLinear":
        if weight_scale is None:
            maximum = module.weight.detach().abs().reshape(module.out_features, -1).amax(dim=1)
            weight_scale = maximum.clamp_min(torch.finfo(torch.float32).eps) / 255.0
        converted = cls(
            module.in_features,
            module.out_features,
            lut=lut,
            input_scale=input_scale,
            weight_scale=weight_scale,
            bias=module.bias is not None,
            ste=ste,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        converted.load_state_dict(module.state_dict(), strict=False)
        converted.train(module.training)
        return converted

    def forward(self, input: Tensor) -> Tensor:
        flat = input.reshape(-1, self.in_features)
        accumulator = torch.zeros(
            (flat.shape[0], self.out_features), dtype=torch.int64, device=input.device
        )
        lut = self.hardware_lut.to(input.device)
        x_scale = self.input_scale.to(input.device).view(1, 1, 1)
        w_scale = self.weight_scale.to(input.device).view(1, -1, 1)
        for start in range(0, self.in_features, self.mac_chunk_size):
            stop = min(start + self.mac_chunk_size, self.in_features)
            accumulator.add_(
                _signed_lut_product(
                    lut,
                    flat[:, None, start:stop],
                    self.weight[None, :, start:stop],
                    x_scale,
                    w_scale,
                ).sum(dim=-1)
            )
        # int64 MAC totals can exceed the finite FP16 range before the small
        # quantization scales are applied.  Scale in FP32, then cast back, so
        # GPT-2-large FP16 inference does not create inf/NaN LUT indices in the
        # following differential layer.
        compute_dtype = (
            torch.float32 if input.dtype in (torch.float16, torch.bfloat16) else input.dtype
        )
        exact = accumulator.to(compute_dtype) * (
            self.input_scale.to(input.device, compute_dtype)
            * self.weight_scale.to(input.device, compute_dtype).view(1, -1)
        )
        if self.bias is not None:
            exact = exact + self.bias.to(compute_dtype)
        exact = exact.to(input.dtype)
        exact = exact.reshape(*input.shape[:-1], self.out_features)
        if not self.ste:
            return exact
        return _ste(exact, F.linear(input, self.weight, self.bias))


class DifferentialConv2d(DifferentialLayerMixin, nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        *,
        lut: Tensor,
        input_scale: float,
        weight_scale: float | Tensor,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        ste: bool = False,
        mac_chunk_size: int = 32,
        device=None,
        dtype=None,
    ) -> None:
        nn.Conv2d.__init__(
            self, in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device=device, dtype=dtype,
        )
        self._init_differential(
            lut,
            input_scale=input_scale,
            weight_scale=weight_scale,
            ste=ste,
            mac_chunk_size=mac_chunk_size,
        )

    @classmethod
    def from_module(
        cls,
        module: nn.Conv2d,
        *,
        lut: Tensor,
        input_scale: float,
        weight_scale: float | Tensor | None = None,
        ste: bool = False,
    ) -> "DifferentialConv2d":
        if isinstance(module.padding, str):
            raise NotImplementedError("differential Conv2d requires numeric padding")
        if weight_scale is None:
            maximum = module.weight.detach().abs().reshape(module.out_channels, -1).amax(dim=1)
            weight_scale = maximum.clamp_min(torch.finfo(torch.float32).eps) / 255.0
        converted = cls(
            module.in_channels, module.out_channels, module.kernel_size,
            lut=lut, input_scale=input_scale, weight_scale=weight_scale,
            stride=module.stride, padding=module.padding, dilation=module.dilation,
            groups=module.groups, bias=module.bias is not None,
            padding_mode=module.padding_mode, ste=ste,
            device=module.weight.device, dtype=module.weight.dtype,
        )
        converted.load_state_dict(module.state_dict(), strict=False)
        converted.train(module.training)
        return converted

    def forward(self, input: Tensor) -> Tensor:
        if self.padding_mode != "zeros":
            raise NotImplementedError("differential Conv2d currently supports zero padding only")
        patches = F.unfold(
            input, self.kernel_size, dilation=self.dilation,
            padding=self.padding, stride=self.stride,
        )
        batch, _, locations = patches.shape
        kernel_elements = self.weight[0].numel()
        in_group_elements = kernel_elements
        out_per_group = self.out_channels // self.groups
        accumulator = torch.zeros(
            (batch, self.out_channels, locations), dtype=torch.int64, device=input.device
        )
        lut = self.hardware_lut.to(input.device)
        x_scale = self.input_scale.to(input.device).view(1, 1, 1, 1)
        w_scale_all = self.weight_scale.to(input.device)
        weights = self.weight.reshape(self.out_channels, kernel_elements)
        for group in range(self.groups):
            out_start = group * out_per_group
            out_stop = out_start + out_per_group
            patch_start = group * in_group_elements
            patch_stop = patch_start + in_group_elements
            x_group = patches[:, patch_start:patch_stop, :]
            w_group = weights[out_start:out_stop]
            group_acc = accumulator[:, out_start:out_stop]
            group_scales = w_scale_all[out_start:out_stop].view(1, -1, 1, 1)
            for start in range(0, kernel_elements, self.mac_chunk_size):
                stop = min(start + self.mac_chunk_size, kernel_elements)
                x = x_group[:, start:stop, :].permute(0, 2, 1).unsqueeze(1)
                w = w_group[:, start:stop].view(1, out_per_group, 1, stop - start)
                group_acc.add_(
                    _signed_lut_product(lut, x, w, x_scale, group_scales).sum(dim=-1)
                )
        compute_dtype = (
            torch.float32 if input.dtype in (torch.float16, torch.bfloat16) else input.dtype
        )
        exact = accumulator.to(compute_dtype) * (
            self.input_scale.to(input.device, compute_dtype)
            * w_scale_all.to(input.device, compute_dtype).view(1, -1, 1)
        )
        if self.bias is not None:
            exact = exact + self.bias.to(compute_dtype).view(1, -1, 1)
        exact = exact.to(input.dtype)
        height = (
            input.shape[-2] + 2 * self.padding[0]
            - self.dilation[0] * (self.kernel_size[0] - 1) - 1
        ) // self.stride[0] + 1
        width = (
            input.shape[-1] + 2 * self.padding[1]
            - self.dilation[1] * (self.kernel_size[1] - 1) - 1
        ) // self.stride[1] + 1
        exact = exact.reshape(batch, self.out_channels, height, width)
        if not self.ste:
            return exact
        surrogate = F.conv2d(
            input, self.weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups,
        )
        return _ste(exact, surrogate)


def adapt_module(
    model: nn.Module,
    *,
    lut: Tensor,
    input_scales: dict[str, float],
    ste: bool = False,
) -> nn.Module:
    """Replace Conv2d/Linear children in place using fully qualified module names."""

    named = dict(model.named_modules())
    missing = {name for name, module in named.items() if isinstance(module, (nn.Conv2d, nn.Linear))} - set(input_scales)
    if missing:
        raise KeyError(f"Missing input scales for layers: {sorted(missing)}")
    for full_name, module in list(model.named_modules()):
        if not full_name or isinstance(module, (DifferentialConv2d, DifferentialLinear)):
            continue
        replacement: nn.Module | None = None
        if isinstance(module, nn.Conv2d):
            replacement = DifferentialConv2d.from_module(
                module, lut=lut, input_scale=input_scales[full_name], ste=ste
            )
        elif isinstance(module, nn.Linear):
            replacement = DifferentialLinear.from_module(
                module, lut=lut, input_scale=input_scales[full_name], ste=ste
            )
        if replacement is not None:
            parent_name, _, child_name = full_name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, replacement)
    return model

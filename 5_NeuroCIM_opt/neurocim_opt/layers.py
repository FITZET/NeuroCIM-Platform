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


def _quantize_signed_code(value: Tensor, scale: Tensor) -> Tensor:
    """Return the exact signed magnitude code used by the LUT implementation."""

    magnitude = _quantize_magnitude(value, scale)
    safe = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0)
    return (magnitude * torch.sign(safe).to(torch.int64)).to(torch.int16)


def _signed_lut_product_codes(lut: Tensor, qx: Tensor, qw: Tensor) -> Tensor:
    """Look up already-quantized signed codes without changing LUT semantics."""

    magnitude = lut[qw.abs().to(torch.int64), qx.abs().to(torch.int64)]
    return (magnitude * torch.sign(qx).to(magnitude.dtype)
            * torch.sign(qw).to(magnitude.dtype))


def _signed_lut_product(lut: Tensor, x: Tensor, w: Tensor, x_scale: Tensor, w_scale: Tensor) -> Tensor:
    """Exact signed product from an interaction LUT whose axes are already zeroed."""

    qx = _quantize_magnitude(x, x_scale)
    qw = _quantize_magnitude(w, w_scale)
    magnitude = lut[qw, qx]
    safe_x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    safe_w = torch.nan_to_num(w, nan=0.0, posinf=1.0, neginf=-1.0)
    sign = torch.sign(safe_x).to(torch.int64) * torch.sign(safe_w).to(torch.int64)
    return magnitude * sign


def _split_signed_code(code: Tensor) -> tuple[Tensor, Tensor]:
    """Split [-255, 255] exactly into low + 128 * high INT8 terms."""

    value = code.to(torch.int16)
    high = torch.sign(value) * (value.abs() // 128)
    low = value - high * 128
    return low.to(torch.int8), high.to(torch.int8)


def _int8_mm(left: Tensor, right: Tensor) -> Tensor:
    """Exact INT8 GEMM with padding for CUDA's alignment constraints."""

    rows, shared = left.shape
    if right.ndim != 2 or right.shape[0] != shared:
        raise ValueError("INT8 matrix dimensions do not match")
    columns = right.shape[1]
    if left.device.type != "cuda":
        return torch._int_mm(left.contiguous(), right.contiguous())

    # torch._int_mm uses the CUDA INT8 tensor-core path, which requires aligned
    # K/N dimensions and more than 16 rows. Zero padding is mathematically inert.
    padded_rows = max(24, ((rows + 7) // 8) * 8)
    padded_shared = ((shared + 7) // 8) * 8
    padded_columns = ((columns + 7) // 8) * 8
    if (padded_rows, padded_shared) != (rows, shared):
        left = F.pad(left, (0, padded_shared - shared, 0, padded_rows - rows))
    if (padded_shared, padded_columns) != (shared, columns):
        right = F.pad(right, (0, padded_columns - columns, 0, padded_shared - shared))
    return torch._int_mm(left.contiguous(), right.contiguous())[:rows, :columns]


def _ideal_signed_code_mm(left_codes: Tensor, right_codes: Tensor) -> Tensor:
    """Compute signed-code matrix multiplication with exact integer arithmetic.

    ``left_codes`` is [M, K], ``right_codes`` is [N, K]. Four INT8 GEMMs
    reconstruct the full signed 9-bit product without approximation.
    """

    if left_codes.shape[1] != right_codes.shape[1]:
        raise ValueError("signed-code matrix dimensions do not match")
    # The final result is bounded by K * 255^2. Keep an explicit guard even
    # though all currently supported layers are far below the INT32 limit.
    if left_codes.shape[1] * 255 * 255 > torch.iinfo(torch.int32).max:
        return left_codes.to(torch.int64) @ right_codes.to(torch.int64).transpose(0, 1)
    left_low, left_high = _split_signed_code(left_codes)
    right_low, right_high = _split_signed_code(right_codes)
    right_low_t = right_low.transpose(0, 1).contiguous()
    right_high_t = right_high.transpose(0, 1).contiguous()
    low_low = _int8_mm(left_low, right_low_t).to(torch.int64)
    low_high = _int8_mm(left_low, right_high_t).to(torch.int64)
    high_low = _int8_mm(left_high, right_low_t).to(torch.int64)
    high_high = _int8_mm(left_high, right_high_t).to(torch.int64)
    return low_low + 128 * (low_high + high_low) + 16384 * high_high


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
        if (int(table.min()) >= torch.iinfo(torch.int32).min
                and int(table.max()) <= torch.iinfo(torch.int32).max):
            table = table.to(torch.int32)
        self.register_buffer("hardware_lut", table, persistent=False)
        ideal_codes = torch.arange(256, dtype=torch.int64, device=table.device)
        self._ideal_product_lut = bool(torch.equal(
            table.to(torch.int64), ideal_codes[:, None] * ideal_codes[None, :]
        ))
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
        self.adc_bits: int | None = None
        self.noise_sigma = 0.0
        self.noise_rho = 0.0
        self.noise_seed = 0
        self.register_buffer(
            "_cached_signed_weight_codes", torch.empty(0, dtype=torch.int16, device=self.weight.device),
            persistent=False,
        )
        self._cached_weight_version = -1

    def _signed_weight_codes(self) -> Tensor:
        """Cache inference weights, refreshing automatically after optimizer updates."""

        version = int(self.weight._version)
        if (self._cached_weight_version != version
                or self._cached_signed_weight_codes.shape != self.weight.shape):
            scale_shape = [self.weight.shape[0]] + [1] * (self.weight.ndim - 1)
            self._cached_signed_weight_codes = _quantize_signed_code(
                self.weight.detach(), self.weight_scale.view(*scale_shape)
            )
            self._cached_weight_version = version
        return self._cached_signed_weight_codes

    def configure_nonidealities(
        self, *, adc_bits: int | None = None, noise_sigma: float = 0.0,
        noise_rho: float = 0.0, noise_seed: int = 0,
    ) -> None:
        if adc_bits is not None and not 2 <= int(adc_bits) <= 16:
            raise ValueError("adc_bits must be None or an integer in [2, 16]")
        if float(noise_sigma) < 0:
            raise ValueError("noise_sigma must be non-negative")
        if not 0 <= float(noise_rho) <= 1:
            raise ValueError("noise_rho must be in [0, 1]")
        self.adc_bits = None if adc_bits is None else int(adc_bits)
        self.noise_sigma = float(noise_sigma)
        self.noise_rho = float(noise_rho)
        self.noise_seed = int(noise_seed)

    def _apply_nonidealities(self, value: Tensor, full_scale: Tensor) -> Tensor:
        """Apply signed ADC quantization and reproducible normalized device noise."""
        scale = full_scale.to(value.device, value.dtype).clamp_min(torch.finfo(value.dtype).eps)
        while scale.ndim < value.ndim:
            scale = scale.unsqueeze(-1)
        if self.adc_bits is not None:
            qmax = float((1 << (self.adc_bits - 1)) - 1)
            value = torch.round(torch.clamp(value / scale, -1.0, 1.0) * qmax) * scale / qmax
        if self.noise_sigma > 0:
            generator = torch.Generator(device=value.device)
            generator.manual_seed(self.noise_seed)
            independent = torch.randn(value.shape, generator=generator, device=value.device, dtype=value.dtype)
            shared_shape = list(value.shape); shared_shape[1] = 1
            shared = torch.randn(shared_shape, generator=generator, device=value.device, dtype=value.dtype)
            noise = ((1.0 - self.noise_rho) ** 0.5 * independent
                     + self.noise_rho ** 0.5 * shared)
            value = value + self.noise_sigma * scale * noise
        return value

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
        qx_all = _quantize_signed_code(flat, self.input_scale.to(input.device))
        qw_all = self._signed_weight_codes().reshape(self.out_features, self.in_features)
        if self._ideal_product_lut:
            accumulator = _ideal_signed_code_mm(qx_all, qw_all)
        else:
            accumulator = torch.zeros(
                (flat.shape[0], self.out_features), dtype=torch.int64, device=input.device
            )
        lut = self.hardware_lut.to(input.device)
        if not self._ideal_product_lut:
            for start in range(0, self.in_features, self.mac_chunk_size):
                stop = min(start + self.mac_chunk_size, self.in_features)
                accumulator.add_(
                    _signed_lut_product_codes(
                        lut,
                        qx_all[:, None, start:stop],
                        qw_all[None, :, start:stop],
                    ).sum(dim=-1, dtype=torch.int64)
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
        lut_peak = self.hardware_lut.abs().amax().to(input.device, compute_dtype)
        full_scale = (self.input_scale.to(input.device, compute_dtype)
                      * self.weight_scale.to(input.device, compute_dtype)
                      * lut_peak * self.in_features).view(1, -1)
        exact = self._apply_nonidealities(exact, full_scale)
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
        w_scale_all = self.weight_scale.to(input.device)
        qpatches = _quantize_signed_code(
            patches, self.input_scale.to(input.device)
        )
        qweights = self._signed_weight_codes().reshape(self.out_channels, kernel_elements)
        for group in range(self.groups):
            out_start = group * out_per_group
            out_stop = out_start + out_per_group
            patch_start = group * in_group_elements
            patch_stop = patch_start + in_group_elements
            x_group = qpatches[:, patch_start:patch_stop, :]
            w_group = qweights[out_start:out_stop]
            group_acc = accumulator[:, out_start:out_stop]
            if self._ideal_product_lut:
                rows = x_group.permute(0, 2, 1).reshape(batch * locations, kernel_elements)
                exact_group = _ideal_signed_code_mm(rows, w_group)
                group_acc.copy_(exact_group.reshape(batch, locations, out_per_group).permute(0, 2, 1))
            else:
                for start in range(0, kernel_elements, self.mac_chunk_size):
                    stop = min(start + self.mac_chunk_size, kernel_elements)
                    x = x_group[:, start:stop, :].permute(0, 2, 1).unsqueeze(1)
                    w = w_group[:, start:stop].view(1, out_per_group, 1, stop - start)
                    group_acc.add_(
                        _signed_lut_product_codes(lut, x, w).sum(dim=-1, dtype=torch.int64)
                    )
        compute_dtype = (
            torch.float32 if input.dtype in (torch.float16, torch.bfloat16) else input.dtype
        )
        exact = accumulator.to(compute_dtype) * (
            self.input_scale.to(input.device, compute_dtype)
            * w_scale_all.to(input.device, compute_dtype).view(1, -1, 1)
        )
        lut_peak = self.hardware_lut.abs().amax().to(input.device, compute_dtype)
        full_scale = (self.input_scale.to(input.device, compute_dtype)
                      * w_scale_all.to(input.device, compute_dtype)
                      * lut_peak * kernel_elements).view(1, -1, 1)
        exact = self._apply_nonidealities(exact, full_scale)
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


def configure_model_nonidealities(
    model: nn.Module, *, adc_bits: int | None = None, noise_sigma: float = 0.0,
    noise_rho: float = 0.0, noise_seed: int = 0,
) -> nn.Module:
    """Configure every differential layer without rebuilding cached models."""
    modules = model.modules() if hasattr(model, "modules") else ()
    for offset, module in enumerate(modules):
        if isinstance(module, DifferentialLayerMixin):
            module.configure_nonidealities(
                adc_bits=adc_bits, noise_sigma=noise_sigma, noise_rho=noise_rho,
                noise_seed=noise_seed + offset,
            )
    return model

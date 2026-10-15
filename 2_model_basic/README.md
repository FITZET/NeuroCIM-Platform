# Fault layer modules

`fault_layers` currently provides two PyTorch-compatible replacement layers:

- `FaultConv2d`
- `FaultLinear`

Both compute the original layer result first and optionally pass its output
feature map through an externally supplied error model.

They also support a per-multiplication device-LUT path. The LUT uses the fitting
backend convention `lut[W, X]`; affine zero points are removed with four lookups
for every activation-weight multiplication before integer accumulation.

```python
from fault_layers import FaultConv2d

fault_conv = FaultConv2d.from_module(original_conv)
fault_conv.set_error_model(error_model)
fault_conv.enable_fault()
output = fault_conv(input_tensor)
```

```python
fault_conv.configure_hardware_lut_from_csv(
    "../1_data_fitting/output/01_NMOS/hardware_multiplication_lut.csv",
    input_scale=activation_scale,
    input_zero_point=activation_zero_point,
    weight_scale=weight_scales,          # scalar or per-output-channel
    weight_zero_point=weight_zero_points,
)
fault_conv.enable_fault()
faulty_output = fault_conv(input_tensor)
```

The error model contract is simply:

```python
faulty_output = error_model(normal_output)
```

An optional output error model may still be applied after either the native or
LUT computation. It may be an `nn.Module` or another callable. By default its
returned tensor must keep the original output shape. Automatic recursive model
replacement and global fault control remain outside this module's current scope.

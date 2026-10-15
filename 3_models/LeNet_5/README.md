# MNIST LeNet-5 backend framework

This framework uses `2_model_basic/fault_layers` for all convolution and fully
connected layers. It provides five workflows through one entry point:

```powershell
python 3_models/LeNet_5/main.py train --download
python 3_models/LeNet_5/main.py float-eval
python 3_models/LeNet_5/main.py calibrate
python 3_models/LeNet_5/main.py quant-eval
python 3_models/LeNet_5/main.py fault-eval --lut 1_data_fitting/output/01_NMOS/hardware_multiplication_lut.csv
```

Residual device-error strength is a GUI/CLI-facing parameter:

```powershell
python 3_models/LeNet_5/main.py fault-eval `
  --lut 1_data_fitting/output/02_CFET/hardware_multiplication_lut.csv `
  --lambda 0.25
```

The accepted range is `0 <= lambda <= 1`. The effective per-MAC table is

```text
L_eff = round(L_ideal + lambda * (L_device - L_ideal))
```

where `0` is ideal multiplication and `1` is the full measured device LUT.

Dataset location:

```text
<repository-parent>/data/mnist/MNIST
```

Set `NEUROCIM_MNIST_ROOT` or pass `--data-root` to use another location.
The value is the root passed to torchvision; torchvision creates the
dataset-specific `MNIST` child directory itself.

Inference modes:

- `float`: native floating-point Conv2d/Linear path
- `quantized`: calibrated uint8 affine path with the ideal multiplication LUT
- `fault`: identical quantization parameters with a measured device LUT

Keeping quantized and fault inference on the same implementation path ensures
their difference is caused by the multiplication LUT rather than two unrelated
quantization backends.

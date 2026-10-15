# 环境安装

## 已验证基线

- 操作系统：Windows 11，64 位；
- Python：CPython 3.11.4；
- 依赖版本：以 [`requirements.txt`](../requirements.txt) 为准；
- GUI：PySide6 6.8.3；
- 可选 GPU 路径：CUDA 可用时由 PyTorch 使用。

Windows 10 洁净机、安装器交付和连续 2 小时稳定运行尚未验证，不能由当前结果推断为已通过。

## 创建环境

从仓库根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\5_NeuroCIM_opt\scripts\setup_environment.ps1
```

脚本在 `5_NeuroCIM_opt/.venv` 创建 Python 3.11 环境并安装固定依赖。若 `py -3.11` 不可用，可显式传入 Python：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\5_NeuroCIM_opt\scripts\setup_environment.ps1 `
  -Python "C:\Python311\python.exe"
```

激活并检查：

```powershell
.\5_NeuroCIM_opt\.venv\Scripts\Activate.ps1
python --version
python .\5_NeuroCIM_opt\cli.py --help
python .\5_NeuroCIM_opt\cli.py list-models
```

## 配置外部资源

项目不打包数据集、测量 CSV、神经网络权重、生成结果或本地编译工具。推荐用环境变量配置：

```powershell
$env:NEUROCIM_DATA_ROOT = "D:\datasets"
$env:NEUROCIM_MNIST_ROOT = "D:\datasets"
$env:NEUROCIM_CIFAR10_ROOT = "D:\datasets"
$env:NEUROCIM_OUTPUT_ROOT = "D:\neurocim-results"
$env:NEUROCIM_NEUROSIM_BIN = "D:\tools\neurosim.exe"
$env:NEUROCIM_TOOLCHAIN_ROOT = "D:\tools\mingw"
```

路径解析规则和 JSON 配置示例见[用户手册](USER_MANUAL.md)。应用导入路径时不会隐式下载用户数据；命令行仅在显式使用 `--download` 时允许 torchvision 下载 MNIST。

## 构建官方 NeuroSim

仓库中的 `vendor/neurosim` 固定到官方 `2DInferenceV1.4` 分支提交 `8a88abf85844c0e1ba17cc771ea535fff6040456`。执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  .\5_NeuroCIM_opt\scripts\build_neurosim.ps1
```

构建脚本使用隔离的 `.tools/mingw`，默认输出 `5_NeuroCIM_opt/outputs/neurosim/bin/neurosim.exe`。`.tools`、二进制和 `outputs` 均属于本地产物，不进入发布包。

NeuroSim 采用 CC BY-NC 4.0，仅允许符合该许可的使用；再分发时必须保留上游许可材料。完整来源与边界见 [`UPSTREAM_STATUS.md`](../UPSTREAM_STATUS.md) 和[第三方声明](../THIRD_PARTY_NOTICES.md)。

## 开发与验收命令

```powershell
python -m pytest -q .\1_data_fitting\tests .\2_model_basic\tests `
  .\3_models\LeNet_5\tests .\3_models\Resnet20\tests `
  .\3_models\BNN\tests .\5_NeuroCIM_opt\tests

python .\5_NeuroCIM_opt\scripts\check_packaging_manifest.py
python .\5_NeuroCIM_opt\scripts\check_documentation.py
```

正式 GUI 训练测试依赖本地数据、权重和较长运行时间，因此缺少显式启用条件时会受保护跳过。最近一次完整快速回归为 78 项通过、1 项受保护跳过；详见[测试报告](TEST_REPORT.md)。

## 发布前检查

[`packaging_manifest.json`](../packaging_manifest.json)规定发布包排除 `.tools`、`.venv`、`outputs`、数据目录、权重、CSV/XLSX、可执行文件与动态库。执行清单检查后，还应在实际待发布目录再次核对，保留 `THIRD_PARTY_NOTICES.md` 和 NeuroSim 上游许可材料。

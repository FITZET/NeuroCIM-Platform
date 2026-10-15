# NeuroCIM 用户手册

## 1. 启动前准备

平台已在 Windows 11、CPython 3.11.4 环境验证。先按[环境安装](ENVIRONMENT_SETUP.md)创建虚拟环境，并由用户准备以下本地资源：

- 器件乘法测量 CSV 或已经生成的 LUT；
- MNIST/CIFAR-10 数据集；
- 神经网络权重和量化参数；
- 需要 PPA 评估时使用的 NeuroSim 可执行文件。

数据集、测量数据、权重、工具链和运行结果不会随发布包分发，排除规则见 [`packaging_manifest.json`](../packaging_manifest.json)。

在仓库根目录启动桌面程序：

```powershell
.\5_NeuroCIM_opt\.venv\Scripts\python.exe .\5_NeuroCIM_opt\app.py
```

程序包含“误差建模与阵列分析”“网络与数据集”“单样本推理”“批量推理”“误差感知训练”和“NeuroSim PPA”六个主页签。误差分析页内部包含“器件误差拟合”和“阵列 MAC 分析”两个子页。耗时任务在后台运行；运行期间请保留窗口与终端。

## 2. 配置本地路径

路径解析顺序为命令行参数、环境变量或用户配置、仓库相对默认值。常用环境变量：

```powershell
$env:NEUROCIM_MNIST_ROOT = "D:\datasets"
$env:NEUROCIM_CIFAR10_ROOT = "D:\datasets"
$env:NEUROCIM_OUTPUT_ROOT = "D:\neurocim-results"
$env:NEUROCIM_NEUROSIM_BIN = "D:\tools\neurosim.exe"
$env:NEUROCIM_TOOLCHAIN_ROOT = "D:\tools\mingw"
```

其中 MNIST 根目录下应有 `MNIST` 子目录。也可在 `%APPDATA%\NeuroCIM\config.json` 写入：

```json
{
  "datasets": {
    "mnist": "D:/datasets",
    "cifar10": "D:/datasets"
  },
  "output_root": "D:/neurocim-results",
  "neurosim_binary": "D:/tools/neurosim.exe",
  "toolchain_root": "D:/tools/mingw"
}
```

用 `NEUROCIM_CONFIG` 可指定其他配置文件。

## 3. 导入误差数据

### 3.1 单器件乘法表

在“Device and MAC fitting”主页签中打开“Device error fitting”子页：

1. 用“Select CSV”选择器件测量文件，并选择编码、分隔符和缺失值策略；
2. 按输入文件填写 `X`、`W`、`VD`、`VG`、`output` 五个字段映射；
3. 设置 1/2/4/8 位码宽，以及 VD、VG 和输出量的单位；
4. 选择输出目录，先点“Preview and validate”检查列、数值和完整网格；
5. 点“Fit and create error model”，等待后台拟合完成。

成功后页面显示交互面指标与各拟合模型指标，输出拟合摘要、五类图、硬件乘法 LUT，以及可直接用于推理和训练的 `error_model/error_model.json`。已验收的自定义列 2 位数据成功生成 256×256 推理 LUT，证据见 [`acceptance.json`](../outputs/acceptance/gui_cell_fit/acceptance.json) 和[完成截图](../outputs/acceptance/gui_cell_fit/cell_fitting_complete.png)。

从原始器件 CSV 拟合并保存误差模型：

```powershell
python .\5_NeuroCIM_opt\cli.py build-error-model .\0_data\01_NMOS.csv `
  --input-kind raw `
  --output-dir .\5_NeuroCIM_opt\outputs\error_models\01_NMOS
```

若输入已是 LUT，将 `--input-kind` 改为 `lut`。输出目录包含 `error_model.json` 和带哈希校验的 `multiplication_lut.npy`。原始数据不会被修改。

### 3.2 阵列 MAC 数据

在“阵列 MAC 分析”页选择 CSV，设置编码、分隔符、缺失值处理、字段映射和多项式阶数。先执行“预览并校验”，再执行“计算线性度并保存模型”。结果可导出 CSV/XLSX，日志可单独导出。

命令行等价操作：

```powershell
python .\5_NeuroCIM_opt\cli.py fit-mac-model D:\data\array_mac.csv `
  --map array_size=Array --map bit_width=Bits `
  --map ideal_mac=Ideal --map measured_output=Measured `
  --encoding utf-8 --delimiter comma --missing error --degree 2 `
  --output-dir .\5_NeuroCIM_opt\outputs\mac_models\example
```

输入支持 UTF-8/GBK、逗号/分号、1/2/4/8 位和矩形阵列。缺失值策略为 `error`、`drop` 或 `interpolate`。字段名不同必须显式映射，异常位宽、越界预测和损坏模型会明确报错。

## 4. 推理与训练

“网络与数据集”页是模型组合的唯一选择入口。勾选一个可用组合后，单样本推理、批量推理以及推理成功后的训练配置统一继承该选择，后续页面中的网络字段不可编辑。页面同时显示任务、数据集、输入形状、指标和权重就绪状态。当前登记了 LeNet-5、ResNet-20、BNN、面向 CIFAR-10 的 TinyTransformer（CCT-7/3x1），以及 GPT-2 Small。TinyTransformer 使用 SHI-Labs 官方 300 epoch 预训练权重，源码许可为 Apache-2.0；GPT-2 使用 nanoGPT 的 MIT 许可模型定义和本地 GPT-2 Small 权重。

“单样本推理”页根据模型页的勾选结果自动切换任务界面。选择 LeNet-5、BNN、ResNet-20 或 TinyTransformer 时，输入测试样本索引并查看图片、真实标签、三种计算模式的预测、Top-5 置信度和耗时；选择 GPT-2 时，页面自动切换为文本提示词、生成长度、三模式生成文本和最后一步 Top-5 token 概率。

“批量推理”页同样继承模型组合。图像分类任务显示每 N 个样本的分组准确率柱状图；GPT-2 任务接受每行一条的提示词列表，并显示各提示词的浮点生成、器件误差生成和 token 一致率。单条与批量 GPT-2 结果分别保存到 `outputs/gui_runs/<时间>/gpt2/` 和 `gpt2_batch/`。当前 GPT-2 误差映射范围是最后一个 Transformer block 的注意力输出投影和 MLP 输出投影；其余骨干及语言模型输出头保持浮点运算。

批量命令行验证示例：

```powershell
python .\5_NeuroCIM_opt\cli.py smoke-lenet `
  --error-model .\5_NeuroCIM_opt\outputs\error_models\01_NMOS\error_model.json `
  --dataset-root D:\datasets --samples 32 --device cpu
```

“误差感知训练”页选择误差模型、MNIST 根目录、输出目录、误差强度、轮数、批量大小和设备后开始训练。页面支持取消、进度显示和结果 JSON 导出。正式运行前先用小批次验证资源：

TinyTransformer 采用头部微调：卷积 tokenizer 和 Transformer blocks 作为冻结浮点特征提取骨干，仅序列池化头与最终分类头使用所选器件 LUT 做正负权重差分计算并更新。请求批量与实际批量会写入 `training_result.json` 的 `training_strategy`，评估过程中按 batch 持续显示进度。

```powershell
python .\5_NeuroCIM_opt\cli.py train-eat `
  --error-model .\5_NeuroCIM_opt\outputs\error_models\01_NMOS\error_model.json `
  --dataset-root D:\datasets --epochs 1 `
  --max-train-batches 1 --max-validation-batches 1 --eval-samples 32 `
  --device cpu --output-dir .\5_NeuroCIM_opt\outputs\eat\trial
```

正式训练去掉两个 `--max-*-batches` 限制。用 `--resume <checkpoint>` 续训；用 `--training-lut ideal` 运行同预算理想差分 QAT 对照。结果目录保存检查点、训练历史和评估 JSON。

## 5. NeuroSim PPA

先按[环境安装](ENVIRONMENT_SETUP.md)构建 NeuroSim。在“NeuroSim PPA”页选择 LeNet-5 检查点和误差模型，设置位宽、子阵列行数与并行行数后运行。页面显示面积、延迟、能耗、能效和吞吐率，并保存输入 traces、原始 stdout 和结构化 JSON。

命令行示例：

```powershell
python .\5_NeuroCIM_opt\cli.py neurosim-eval `
  --error-model .\5_NeuroCIM_opt\outputs\error_models\01_NMOS\error_model.json `
  --eat-checkpoint .\5_NeuroCIM_opt\outputs\eat\01_NMOS\eat_checkpoint.pt `
  --dataset-root D:\datasets `
  --output-dir .\5_NeuroCIM_opt\outputs\neurosim\01_NMOS
```

测量 LUT 决定数值推理和训练误差；NeuroSim 根据网络矩阵、活动 traces、位宽和硬件参数估计 PPA，并不直接读取测量 LUT。NeuroSim 使用 CC BY-NC 4.0，商业使用受限，详见[第三方声明](../THIRD_PARTY_NOTICES.md)。

## 6. 结果与故障定位

默认产物位于 `5_NeuroCIM_opt/outputs/`，也可用环境变量改到其他位置。JSON 保存配置和可复核指标；NeuroSim 同时保留原始输出与 trace manifest。

- 提示数据集不存在：设置正确的 `NEUROCIM_*_ROOT`，或传 `--dataset-root`。
- 提示权重或误差模型不存在：导入用户资源并选择实际文件。
- CUDA 不可用或显存不足：将设备改为 `cpu`，并先缩小批次。
- NeuroSim 资源缺失：先构建可执行文件并配置 `NEUROCIM_NEUROSIM_BIN`。
- CSV 校验失败：核对编码、分隔符、字段映射、缺失值策略与位宽。

当前测试范围与限制见[测试报告](TEST_REPORT.md)。

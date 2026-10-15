# 测试报告

报告日期：2026-09-30。

## 验收结论

最近一次多模块快速回归结果为 **78 项通过、1 项受保护跳过**。跳过项是需要显式启用、本地 MNIST/权重和正式运行时间的 GUI 完整误差感知训练测试；对应真实完整运行产物已保留。自动化状态记录见 [`AUTOMATION_STATE.json`](../outputs/automation/AUTOMATION_STATE.json)。

当前证据支持 Windows 11、CPython 3.11.4 开发环境中的功能验收。以下项目尚未验证：

- Windows 10 洁净机安装与运行；
- 可执行安装器的构建、安装、卸载和升级；
- 连续 2 小时稳定运行；
- 不同硬件和多个随机种子的统计结论。

## 功能与证据

| 范围 | 验收结果 | 主要证据 |
|---|---|---|
| 器件 LUT 拟合与完整性 | 四种已有器件回归通过；01_NMOS 交互面 R² 0.990876 | [`error_model.json`](../outputs/error_models/01_NMOS/error_model.json) |
| 器件误差拟合 GUI | 自定义列 2 位、16 行完整网格导入成功，生成 256×256 推理 LUT、摘要与五类图；缺列、3 位码宽和缺失数值反例均被拒绝 | [`acceptance.json`](../outputs/acceptance/gui_cell_fit/acceptance.json)、[完成截图](../outputs/acceptance/gui_cell_fit/cell_fitting_complete.png) |
| 阵列 MAC 导入 | 10 万行、4.09 MB 导入校验 0.227 s，目标 3 s，通过 | [`benchmark.json`](../outputs/acceptance/performance/benchmark.json) |
| MAC 拟合 GUI | 预览、拟合、保存、表格/日志导出完成 | [`mac_analysis.png`](../outputs/acceptance/gui/mac_analysis.png) |
| 单样本推理 GUI | MNIST 样本 0 的三种模式均预测 7；总耗时 548.9 ms | [`sample_0_prediction.json`](../outputs/acceptance/gui_inference/sample_0_prediction.json) |
| Inference runtime | Real CPU: hot single 10.39 ms; hot batch of 10 is 60.84 ms; cold timing is reported separately | [`inference_benchmark.json`](../outputs/acceptance/performance/inference_benchmark.json) |
| LeNet-5 完整测试 | 浮点 98.69%，器件训练前 98.52%，训练后 99.02% | [`training_result.json`](../outputs/acceptance/gui_training/training_result.json) |
| GUI 误差感知训练 | 55,000 训练、5,000 验证、10,000 测试；10 个参数张量变化 | [`gui_acceptance.json`](../outputs/acceptance/gui_training/gui_acceptance.json) |
| LeNet-5 NeuroSim GUI | 五层运行完成；面积 4.7236 mm²，延迟 22.1344 µs | [`gui_ppa_result.json`](../outputs/acceptance/gui_ppa/gui_ppa_result.json) |
| ResNet-20 | 100 epoch；完整 CIFAR-10 浮点测试 89.98%；22 个 MAC 层完成 PPA | [`PROGRESS.md`](../PROGRESS.md) |
| BNN | 三个确定性 epoch；完整 MNIST 测试 98.21% | [`AUTOMATION_STATE.json`](../outputs/automation/AUTOMATION_STATE.json) |
| TinyTransformer | 官方 CCT-7/3x1 权重；真实 CIFAR-10 单样本三模式推理、4 样本分组推理及 1 batch 误差感知训练均通过 | [`tinytransformer_smoke.json`](../outputs/inference/tinytransformer_smoke.json) |
| GPT-2 Small | 124,439,808 参数本地权重；真实 CUDA 浮点、理想差分和器件误差差分文本生成通过，器件误差可改变生成 token | [`gpt2_smoke.json`](../outputs/inference/gpt2_smoke.json) |

TinyTransformer 头部微调使用界面批量 128 做了显存与耗时回归：冻结浮点骨干，仅 `classifier.attention_pool` 与 `classifier.fc` 使用器件差分并更新 2,827 个参数。训练前后六次评估、一次反向传播和逐 batch 进度回调均在 12 GiB RTX 3060 上完成。
| 发布清单 | 20 条排除规则检查通过 | [`packaging_manifest.json`](../packaging_manifest.json) |

器件 GUI 回归同时重新拟合真实 `0_data/01_NMOS.csv`。交互面 R² 为 `0.9908762695039321`、满量程非线性为 `6.578497847717723%`，与既有基线一致；测试 `test_01_nmos_metric_regression_has_no_drift` 以 `1e-12` 绝对容差守护 R²，未发现数值漂移。

单样本、32 样本或 8 样本差分结果只用于机制验证，不作为完整测试集精度。训练提升来自单个种子工程验证，不构成统计显著性结论。

## NeuroSim 结果解释

正式 LeNet-5 命令行证据记录 4.7236 mm²、22.1344 µs、230.426 nJ 动态能耗和 19.2858 TOPS/W，见 [`neurosim_result.json`](../outputs/neurosim/01_NMOS/neurosim_result.json)。GUI PPA 验收采用保留的确定性零输入 trace，其结构化结果为 196.474 nJ 与 22.5887 TOPS/W。两组活动输入不同，不能混为同一运行。

测量误差 LUT 控制数值推理和误差感知训练；官方 NeuroSim 根据网络映射、位宽和活动 trace 估计 PPA，不直接读取测量 LUT。上游版本、补丁和建模边界见 [`UPSTREAM_STATUS.md`](../UPSTREAM_STATUS.md)。

## 复现检查

从仓库根目录执行快速回归：

```powershell
$env:NEUROCIM_MNIST_ROOT = "D:\datasets"
python -m pytest -q .\1_data_fitting\tests .\2_model_basic\tests `
  .\3_models\LeNet_5\tests .\3_models\Resnet20\tests `
  .\3_models\BNN\tests .\5_NeuroCIM_opt\tests
```

资源和时间条件不满足时，正式 GUI 训练测试应显示为受保护跳过。再执行：

```powershell
python .\5_NeuroCIM_opt\scripts\check_packaging_manifest.py
python .\5_NeuroCIM_opt\scripts\check_documentation.py
```

文档检查会验证本地 Markdown 链接和本报告引用的关键证据是否存在。完整训练无需在每次快速回归中重复，重新声明正式指标时必须重新执行并保留新的 JSON、日志和截图。

## 许可与交付限制

NeuroSim 源码来自官方 `neurosim/NeuroSim` 的 `2DInferenceV1.4` 分支、提交 `8a88abf85844c0e1ba17cc771ea535fff6040456`，许可为 CC BY-NC 4.0，不能据此用于商业用途。数据集、测量数据和模型权重均由用户提供，不进入发布包。详见[第三方声明](../THIRD_PARTY_NOTICES.md)和[环境安装](ENVIRONMENT_SETUP.md)。

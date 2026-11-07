# 报告证据核验清单

核验日期：2026-10-06  
核验范围：原始规格书、`5_NeuroCIM_opt/docs`、`5_NeuroCIM_opt/outputs`、`3_models/*/output`。  
使用原则：报告中的实验数值优先引用原始 JSON/CSV；Markdown 文档只用于解释背景和交叉核对。下文把“完整测试集”“小样本机制验证”“单批次 smoke”严格分开。

## 1. 规格书原文提炼

原始文件：`C:\Users\29075\Downloads\20260421-DRAM存算仿真平台外包开发规格书-zys(1).docx`。本次直接读取 DOCX 内 `word/document.xml`，并与 [`specification_extract.txt`](specification_extract.txt) 交叉核对。

### 1.1 项目目标与两条业务分支

- 目标是根据 DRAM 单元模拟乘法或阵列乘累加的输入输出线性度数据，外推神经网络推理精度，并由 GUI 完成数据导入、参数配置、精度计算和单数据实时推理。
- 分支 1：导入 Virtuoso/TCAD 等得到的单元多 bit 乘法 CSV，支持自定义列映射；外推可配置 M×N 阵列、1/2/4/8 bit 乘累加和线性度。
- 分支 2：导入阵列 n-bit MAC CSV，字段含阵列规模、位宽、输入映射和输出映射，同样要求自定义列映射和线性度拟合。
- 线性度指标至少包括非线性误差与 R²，计算结果要能供后续神经网络精度计算使用。

### 1.2 网络、数据与训练需求

- 规格书要求覆盖四种不同算力的主流 DNN、1 种 BNN、1 种 LLM；数据集要求覆盖 CIFAR-10、ImageNet 等四种主流数据集。
- 要求内置或自动匹配预训练模型和数据集，支持基于器件误差/线性度的二次训练，输出完整测试集推理准确率并支持 CSV/Excel 导出。
- 单样本界面需要展示预测标签、真实标签、置信度、推理耗时和关键参数。

### 1.3 GUI、兼容性与性能验收线

- GUI 要覆盖导入预览、有效性校验、阵列与位宽设置、网络/数据集选择、二次训练、精度展示、单样本推理、日志、参数记忆和异常提示。
- CSV 要支持 ANSI/UTF-8、逗号/分号、缺失值处理，且单次导入不少于 10 万行。
- 规格书性能阈值：启动时间 ≤10 s、界面响应 ≤0.5 s、10 万行 CSV 处理 ≤3 s、单网络/单数据集批量精度计算 ≤10 s、单样本推理 ≤1 s；连续运行 2 小时无泄漏、卡顿或闪退。
- 计算准确性要求包括：线性度结果相对理论值误差 ≤1%；更好线性度应对应合理的更高精度趋势；单样本结果应与标准预训练模型一致。

### 1.4 交付物和仍需谨慎陈述的合同项

- 规格书要求绿色版、安装版、完整源码、Git 历史、内置模型与数据集、测试数据、用户/技术/环境/测试文档和一键配置脚本。
- 还包括 30 个自然日周期、12 个月支持、7×24 服务、24 小时响应与 48 小时解决等合同义务。
- 现有 [`REQUIREMENTS_TRACEABILITY.md`](../REQUIREMENTS_TRACEABILITY.md) 已明确：安装器、Git 历史、连续 2 小时稳定性、完整网络/数据集覆盖和洁净 Windows 10 验证尚未完成，报告不得写成“规格书全部验收通过”。

## 2. 证据分级规则

| 分类 | 判定标准 | 报告中允许的表述 |
|---|---|---|
| 完整测试/正式训练 | 明确遍历官方测试集全量样本，或正式训练配置与完整测试结果均保存在原始产物中 | 可报告为“完整测试集准确率”或“正式训练结果”，仍需说明种子数 |
| 小样本机制验证 | 固定索引、几十/几百/一千样本，验证数值路径、GUI 或批量分组逻辑 | 只能称“限定样本验证/机制验证”，不可泛化为测试集准确率 |
| smoke | 仅 1 个或少量 batch、少量 prompt/样本，主要验证能跑通、反传或显存边界 | 只能称 smoke，不可用于模型精度结论 |
| PPA | 官方 NeuroSim 进程返回码为 0，原始 stdout、trace 和解析 JSON 均保留 | 可报告对应 trace 和硬件配置下的工程估计，不可称芯片实测 |
| 文档自述 | 只有 Markdown 汇总，缺少新的原始运行产物 | 可作说明或索引，不应独立支撑定量结论 |

## 3. 器件拟合与阵列 MAC 的原始证据

四个器件 CSV 均各含 65,536 个数据行，即完整 256×256 数字码网格。以下指标来自各自 `fitting_summary.json` 的 `interaction_metrics`：

| 数据 | R² | RMSE（A） | MAE（A） | 满量程非线性 | 原始证据 |
|---|---:|---:|---:|---:|---|
| 01_NMOS | 0.9908762695 | 1.54136×10⁻⁷ | 1.15245×10⁻⁷ | 6.57850% | [`01_NMOS/fitting_summary.json`](../../../1_data_fitting/output/01_NMOS/fitting_summary.json) |
| 02_CFET | 0.9281507793 | 2.25192×10⁻⁷ | 1.68309×10⁻⁷ | 18.17685% | [`02_CFET/fitting_summary.json`](../../../1_data_fitting/output/02_CFET/fitting_summary.json) |
| 03_UOS_1 | 0.9940709785 | 2.96212×10⁻⁸ | 1.98155×10⁻⁸ | 5.77810% | [`03_UOS_1/fitting_summary.json`](../../../1_data_fitting/output/03_UOS_1/fitting_summary.json) |
| 04_UOS_2 | 0.9846901626 | 4.87017×10⁻⁸ | 3.89651×10⁻⁸ | 8.95808% | [`04_UOS_2/fitting_summary.json`](../../../1_data_fitting/output/04_UOS_2/fitting_summary.json) |

可用于正文的补充证据：

- UOS1 的 GUI 重新拟合结果与上述基线一致：R²=0.9940709785、RMSE=2.96212×10⁻⁸ A、非线性=5.77810%；采集位宽 8 bit、网格范围 0–255、输出原始单位 uA，拟合前换算为 A。证据：[`workflow_acceptance/uos1_gui_fit/fitting_summary.json`](../../outputs/workflow_acceptance/uos1_gui_fit/fitting/fitting_summary.json)。
- 自定义列 2-bit GUI 用例有 16 行完整 4×4 网格，成功生成 256×256 推理 LUT 和 5 类图；缺列、3-bit 位宽和缺失数值均被拒绝。它是功能验收样例，不是器件精度代表值。证据：[`gui_cell_fit/acceptance.json`](../../outputs/acceptance/gui_cell_fit/acceptance.json)。
- 阵列 MAC 理论数据共 256 点、32×32、8 bit。线性拟合 R²=0.9998356259、RMSE=0.976916、满量程非线性=0.821209%；二次多项式拟合 R²=1.0。证据：[`mac_error_model.json`](../../outputs/acceptance/mac_branch/model/mac_error_model.json)。
- GUI 阵列选择 16×32、4 bit 的验证中，线性组 41 点 R²=1.0；已知二次扰动组 41 点的线性拟合 R²=0.9993017379、非线性=1.625%，二次拟合 R²=1.0。证据：[`array_config_metrics.json`](../../outputs/acceptance/gui_array_config/array_config_metrics.json)。
- 10 万行、4,086,271 字节 CSV 的导入与校验耗时 0.226845 s，低于 3 s 阈值；环境字段记录为 CPython 3.11.4、Windows 内核字符串 `Windows-10-10.0.26200-SP0`、Intel64 Family 6 Model 191。证据：[`benchmark.json`](../../outputs/acceptance/performance/benchmark.json)。该单次结果不能替代多硬件统计。

## 4. 完整测试集与正式训练结果

### 4.1 LeNet-5 / MNIST

完整测试集均为 10,000 张，种子 42；训练集/验证集为 55,000/5,000。当前报告应优先采用精确四路差分管线的以下证据：

| 条件 | 准确率 | 损失 | 证据 |
|---|---:|---:|---|
| 原始浮点 checkpoint | 98.69% | 0.043687 | [`01_NMOS_full_test.json`](../../outputs/smoke/01_NMOS_full_test.json) |
| 理想差分量化 | 98.70% | 0.043664 | 同上 |
| 01_NMOS 器件差分，训练前 | 98.52% | 0.046854 | 同上 |
| 器件感知训练 1 epoch 后，器件差分 | 99.02% | 0.030654（GUI 正式运行） | [`gui_training/training_result.json`](../../outputs/acceptance/gui_training/training_result.json) |
| 同一训练后，浮点 / 理想差分 | 99.02% / 99.01% | 0.030828 / 0.030894 | 同上 |
| 等预算理想 QAT 后，在器件/理想路径 | 98.95% / 99.07% | 0.032288 / 0.030117 | [`qat_control/evaluation.json`](../../outputs/qat_control/01_NMOS/evaluation.json) |

训练运行耗时 31.491 s（CUDA），10 个参数张量发生变化，参数 L2 改变量 0.605139；单 epoch 训练准确率 99.7818%，验证准确率 99.9%。这是单一随机种子的工程验证，不构成统计显著性结论。

注意：[`3_models/LeNet_5/output/inference_comparison.json`](../../../3_models/LeNet_5/output/inference_comparison.json) 是较早一套误差注入产物，给出 01_NMOS/02_CFET/03_UOS_1/04_UOS_2 的 97.77%/36.77%/10.10%/8.92%。其 01_NMOS 数值与当前精确四路差分结果 98.52% 不一致，因此不要把两套管线的数据合并成同一张“当前结果”表；如保留，应明确标为历史/旧管线结果。

### 4.2 ResNet-20 / CIFAR-10

- 正式训练 100 epoch，固定种子 42；最佳验证准确率 91.30%，完整 10,000 张测试集浮点准确率 89.98%，损失 0.412841。训练用时 1,638.605 s，最终评估用时 2.911 s，校准 22 个 Conv/Linear 层。证据：[`formal/report.json`](../../../3_models/Resnet20/output/formal/report.json)。
- 差分路径只在固定测试索引 0–7 上运行：理想差分 87.5%，01_NMOS 器件差分 75.0%，两者预测一致率 62.5%。这是 8 样本机制验证，不是完整测试集差分准确率。证据：[`formal/differential_evaluation.json`](../../../3_models/Resnet20/output/formal/differential_evaluation.json)。

### 4.3 BNN / MNIST

- 正式训练 3 epoch，种子 42，完整 10,000 张测试集准确率 98.21%，损失 0.244203；运行耗时 38.157 s。证据：[`formal/evaluation.json`](../../../3_models/BNN/output/formal/evaluation.json)。
- 差分验证仅 32 个固定样本：原生 96.875%，理想差分和 01_NMOS 器件差分均为 100%。文件本身标注 `bounded_mechanism_validation`，不得写成完整测试集成绩。证据：[`formal/differential_evaluation.json`](../../../3_models/BNN/output/formal/differential_evaluation.json)。

## 5. 小样本、批量机制验证与 smoke

| 模型/流程 | 样本规模 | 结果 | 正确分类 | 证据 |
|---|---:|---|---|---|
| LeNet-5 GUI 批量推理（UOS2） | 1,000 | 浮点 98.3%，理想 98.3%，器件 98.5%；总耗时 507.49 ms | 小样本/子集批量验证 | [`gui_batch_inference/batch_result.json`](../../outputs/gui_batch_inference/batch_result.json) |
| LeNet-5 CPU 后端 | 20 | 浮点/理想 100%，器件 95% | smoke | [`batch_inference/backend_smoke.json`](../../outputs/batch_inference/backend_smoke.json) |
| LeNet-5 UOS1 GUI 批量 | 23 | 浮点/理想 100%，器件 95.652% | 小样本 GUI 验证 | [`batch_inference_uos1/batch_result.json`](../../outputs/acceptance/batch_inference_uos1/batch_result.json) |
| ResNet-20 GUI 批量 | 23 | 三模式均 95.652%；总耗时 2,148.03 ms | 小样本机制验证 | [`resnet20_batch/batch_result.json`](../../outputs/workflow_acceptance/resnet20_batch/batch_result.json) |
| ResNet-20 误差训练 | 1 train batch + 1 eval batch，各 32 样本 | 前：96.875/96.875/93.75%；后：93.75/93.75/90.625%；65 个参数张量变化 | 单 batch smoke，不支持“训练提升”结论 | [`resnet20_training_smoke/training_result.json`](../../outputs/workflow_acceptance/resnet20_training_smoke/training_result.json) |
| TinyTransformer 推理 | 4 | 三模式均 100%；总耗时 3,688.57 ms | smoke | [`tinytransformer_smoke.json`](../../outputs/inference/tinytransformer_smoke.json) |
| TinyTransformer 冻结骨干、头部训练 | 128 train + 128 validation | 前后浮点/理想 96.09375%，器件 95.3125%；4 个张量变化 | 单 batch smoke | [`tinytransformer_head_only_smoke/training_result.json`](../../outputs/eat/tinytransformer_head_only_smoke/training_result.json) |

TinyTransformer 头部 smoke 的 `training_strategy.scope` 明确为 `frozen_float_backbone_with_differential_sequence_pooling_and_classifier_heads`，请求和实际 train/eval batch 均为 128。现有 [`TEST_REPORT.md`](../TEST_REPORT.md) 还记录该范围共有 2,827 个可训练参数、在 12 GiB RTX 3060 上完成；“2,827”来自文档汇总，不在上述 training_result JSON 的顶层字段中，正文引用时应同时指明这是实现/测试报告记录。

GPT-2 Small 证据如下：

- 模型 README 记录参数量 124,439,808；推理 smoke 使用 CUDA，只将最后一个 block 的 `transformer.h.11.attn.c_proj` 和 `transformer.h.11.mlp.c_proj` 映射为差分层。证据：[`GPT2/README.md`](../../../3_models/GPT2/README.md)、[`gpt2_smoke.json`](../../outputs/inference/gpt2_smoke.json)。
- 单 prompt `Computing in memory is`，输入 5 token、生成 8 token。浮点与理想差分生成 token 完全相同；器件差分从第 5 个生成 token 起发生变化。三模式耗时分别 166.83/200.13/176.02 ms；这是单提示词 smoke。
- GUI 批量仅 3 条 prompt、每条生成 8 token；文件直接记录 `average_device_token_agreement=0.0`。这只是 24 个生成 token 的小样本机制证据，不能解释为语言模型总体准确率、困惑度或完整语料评估。证据：[`gpt2_batch/batch_result.json`](../../outputs/gui_runs/20261006-155533/gpt2_batch/batch_result.json)。
- 当前没有 GPT-2 误差感知训练产物，也没有测试语料上的 NLL/perplexity 结果；报告只能说明推理链路已跑通。

## 6. 推理性能证据

[`inference_benchmark.json`](../../outputs/acceptance/performance/inference_benchmark.json) 是真实 LeNet-5/MNIST CPU 三模式推理：

- 冷启动单样本：资源准备 1,230.847 ms，总耗时 1,244.242 ms；冷启动超过规格书单样本 ≤1 s 阈值。
- 缓存热单样本：准备 1.574 ms，总耗时 12.364 ms；三种核心模式耗时合计约 10.653 ms。
- 缓存热 10 样本批量：准备 2.740 ms，总耗时 67.721 ms；三种核心模式耗时合计约 64.831 ms。
- 原始文件与 [`AUTOMATION_STATE.json`](../../outputs/automation/AUTOMATION_STATE.json) 的汇总数字略有差异：后者写 10.39 ms 和 60.84 ms。报告如给精确分项，应以原始 benchmark JSON 为准；若沿用状态文件数字，应标为状态汇总值。
- 该性能证据只覆盖缓存后的 LeNet-5 CPU 推理，不代表所有网络或完整测试集都在 10 s 内完成。

## 7. 官方 NeuroSim PPA 证据

以下均为官方 `neurosim/NeuroSim` 的 `2DInferenceV1.4` 分支、提交 `8a88abf85844c0e1ba17cc771ea535fff6040456`，配置为 8-bit 权重、8-bit 激活、128 行子阵列、128 并行行。它们是给定 trace 下的模型估计，不是硅后实测。

| 网络与 trace | MAC 层数 | 面积 | 延迟/图像 | 动态能耗/图像 | 泄漏能量 | 泄漏功耗 | 能效 | 吞吐 | FPS | 证据 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LeNet-5 正式活动 trace | 5 | 4.7236 mm² | 22.1344 µs | 230.426 nJ | 1.7741 nJ | 80.1516 µW | 19.2858 TOPS/W | 0.246827 TOPS | 45,178.6 | [`neurosim/01_NMOS/neurosim_result.json`](../../outputs/neurosim/01_NMOS/neurosim_result.json) |
| LeNet-5 GUI 确定性零输入 trace | 5 | 4.7236 mm² | 22.1344 µs | 196.474 nJ | 1.7741 nJ | 80.1516 µW | 22.5887 TOPS/W | 0.246827 TOPS | 45,178.6 | [`acceptance/gui_ppa/neurosim_result.json`](../../outputs/acceptance/gui_ppa/neurosim_result.json) |
| ResNet-20 活动 trace | 22 | 24.925 mm² | 104.883 µs | 10.5408 µJ | 40.4442 nJ | 385.613 µW | 30.1662 TOPS/W | 3.7129 TOPS | 9,534.45 | [`resnet20_01_NMOS/neurosim_result.json`](../../outputs/neurosim/resnet20_01_NMOS/neurosim_result.json) |

两个 LeNet-5 运行的动态能耗和能效不同，是活动输入 trace 不同导致，不能择一混写或取平均。测量误差 LUT 用于 Python 数值推理/训练；NeuroSim 根据网络矩阵、位宽、活动 trace 和阵列配置估计 PPA，不直接读取该测量 LUT。

## 8. 自动化、GUI 与验收边界

- 截至 2026-09-30 的状态文件记录 78 项通过、1 项受保护跳过，跳过的是需要显式启用和正式数据/运行时间的 GUI 完整误差感知训练测试。证据：[`AUTOMATION_STATE.json`](../../outputs/automation/AUTOMATION_STATE.json)。这是当日快照，不应表述为 2026-10-06 当前代码重新运行后的结果。
- GUI 单样本验收中 MNIST 样本 0 的真实标签为 7，浮点、理想差分和器件差分均预测 7。原始性能 benchmark 记录冷启动总耗时 1.244 s；[`TEST_REPORT.md`](../TEST_REPORT.md) 中的“总耗时 548.9 ms”来自另一 GUI 验收运行，报告需要注明具体产物，避免与 benchmark 混用。
- 当前证据支持 Windows 开发机上的功能验证，但没有洁净 Windows 10 矩阵、安装器安装/卸载、连续 2 小时稳定性或多随机种子统计。
- [`REQUIREMENTS_TRACEABILITY.md`](../REQUIREMENTS_TRACEABILITY.md) 对“完整平台”仍判为 partial，并明确安装器、Git 历史、两小时稳定性和完整交付为 blocked。最终技术报告应保留这些限制。

## 9. 建议正文直接采用的结论

1. 器件拟合表可采用四个 65,536 点完整网格的真实 R²/RMSE/MAE/非线性指标。
2. 分类模型的完整精度只采用：LeNet-5 的 10,000 张 MNIST 三模式与训练前后结果、ResNet-20 的 10,000 张 CIFAR-10 浮点结果、BNN 的 10,000 张 MNIST 原生结果。
3. ResNet-20 差分 8 样本、BNN 差分 32 样本、LeNet-5 1,000/23/20 样本批量结果均标为限定样本或机制验证。
4. TinyTransformer 的 4 样本推理和 1 batch 训练、GPT-2 的 1/3 prompt 生成全部标为 smoke；目前没有 TinyTransformer 完整 CIFAR-10 三模式精度，也没有 GPT-2 perplexity。
5. PPA 表明确写“官方 NeuroSim 工程估计”，同时给出模型、层数、trace 类型和 8/8/128/128 配置。
6. 不得声称规格书全部验收通过；只能说明核心链路已有证据，并列出模型/数据集覆盖、安装器、Windows 10 洁净机、两小时稳定性和多种子统计等缺口。

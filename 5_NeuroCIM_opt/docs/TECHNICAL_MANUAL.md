# 技术手册

## 系统边界

平台把器件数值误差与电路级 PPA 分开建模：

```text
测量 CSV/LUT -> 误差拟合与带哈希模型包 -> 差分推理/误差感知训练
模型与活动数据 -> 四路非负 trace -> 官方 NeuroSim -> PPA JSON
```

详细模块关系见[架构说明](ARCHITECTURE.md)。通用服务位于 `neurocim_opt`，模型与数据集适配位于 `3_models`，器件拟合位于 `1_data_fitting`。GUI 调用 `PlatformService` 和训练服务，数值算法不写在控件回调中。

## 器件误差模型

器件原始电流先移除两个单变量基线：

```text
I_interaction(X,W) = I(X,W) - I(X,0) - I(0,W) + I(0,0)
```

拟合输出是轴顺序为 `[W, X]` 的乘法 LUT。有效误差强度为：

```text
H_lambda = round(X*W + lambda * (H - X*W)), 0 <= lambda <= 1
```

`lambda=0` 表示理想整数乘法，`lambda=1` 表示完整测量表。模型 JSON 保存来源路径和 SHA-256、拟合方法、增益、轴顺序、位宽、舍入/裁剪规则与 LUT 哈希；加载时校验完整性。

通用 cell 导入支持自定义列名、单位、编码、分隔符和缺失值策略。1/2/4/8 位原生采样网格保留在拟合包中，只在构造 8 位推理 LUT 时插值。阵列 MAC 分支按阵列行列数、位宽和实验组拟合线性/多项式响应，并输出 R²、RMSE、MAE 与满量程非线性。

## 正负权重差分计算

输入和权重拆为非负分量：

```text
Y = X+W+ - X+W- - X-W+ + X-W-
```

Conv2d/Linear 将幅值量化为无符号 8 位码，对每个乘积查 LUT，以 int64 累加，按输出通道缩放后加入浮点偏置。误差感知训练前向使用同一精确 LUT 算术，反向使用对应浮点层导数作为 straight-through estimator。这样训练与推理共享误差定义。

## 模型接入

模型登记项描述模型 ID、显示名、任务、数据集、输入形状、指标、检查点路径和许可。新增网络时：

1. 在 `3_models/<model>` 添加架构、数据加载、训练与校准代码；
2. 为所有映射到 CIM 的 Conv/Linear 提供有符号幅值输入尺度；
3. 在模型注册表登记权重路径与许可；
4. 对不能映射的算子给出明确边界；
5. 增加最小的数值、保存/加载和端到端机制测试。

当前实现覆盖 LeNet-5/MNIST、ResNet-20/CIFAR-10、二值 CNN/MNIST、TinyTransformer/CIFAR-10 和 GPT-2 Small 文本生成。TinyTransformer 采用 SHI-Labs CCT-7/3x1，卷积 tokenizer、注意力 QKV/投影、MLP 和分类头中的 Conv2d/Linear 均进入差分 MAC 仿真；归一化、激活、softmax、序列池化等保留为数字操作。

GPT-2 采用 nanoGPT 的 12 层、12 头、768 隐藏维 GPT-2 Small 模型定义与 OpenAI 预训练参数，运行时只读取本地 checkpoint 和 byte-level BPE 分词器。为把单次推理控制在 12 GiB 显存内，当前 CIM 映射边界固定为最后一个 Transformer block 的 `attn.c_proj` 和 `mlp.c_proj`。两层使用校准输入尺度和 256×256 LUT 执行正负权重差分；词嵌入、前 11 个 block、最后一层归一化和 50,257 类语言模型头保留浮点。服务逐个装载浮点、理想差分和器件误差差分模型并在模式间释放显存，采用确定性贪心解码。

GUI 工作流将模型选择持久化在 `outputs/workflow/workflow_context.json` 的 `network_selection` 字段。模型页是唯一写入口；单样本和批量页面监听 `network_changed` 信号并按任务类型切换图像分类或文本生成界面。更换模型组合会使之前的推理和训练上下文失效，防止后续阶段沿用不匹配的数据集或 checkpoint。

## NeuroSim 适配

`neurocim_opt.neurosim_adapter` 导出 `X+`、`X-` 活动和 `W+`、`W-` 权重，四个非负物理路径都计入官方 NeuroSim。数字域执行四路有符号组合。NeuroSim 根据展开矩阵、活动、位宽、子阵列参数和网络配置计算面积、延迟、动态/泄漏能耗、能效与吞吐。

测量 LUT 不会直接送入 NeuroSim 模拟器，因此 PPA 结果不能解释为器件 LUT 的电路级复现。集成来源固定为官方分支与提交，补丁和边界记录于 [`UPSTREAM_STATUS.md`](../UPSTREAM_STATUS.md)。NeuroSim 为 CC BY-NC 4.0，商业用途不在该许可授权范围内。

## 目录和产物约定

- `neurocim_opt/`：通用误差模型、差分层、应用服务、GUI 与 NeuroSim 适配；
- `scripts/`：环境、构建、评估和发布检查；
- `docs/`：架构、决策、追踪与交付文档；
- `vendor/neurosim/`：固定版本上游源码；
- `outputs/`：所有生成结果和验收证据，禁止作为源代码版本目录；
- `tests/`：平台单元、集成和 GUI smoke 测试。

路径统一由 `neurocim_opt.paths` 解析。优先级为显式参数、`NEUROCIM_*` 环境变量、用户 JSON 配置、仓库相对默认值。模块导入阶段不要求外部资源已存在；真正执行时通过 `require_path` 给出可操作的缺失资源错误。

## 可复现性与发布

随机训练固定种子并保存模型、优化器、调度器、历史和必要 RNG 状态；误差模型与关键产物保存 SHA-256。正式指标必须区分完整测试集、受限样本机制检查和训练 smoke。

发布包按 [`packaging_manifest.json`](../packaging_manifest.json)排除数据、权重、工具链、输出、CSV/XLSX 与二进制。用户自行导入这些资源。许可说明见[第三方声明](../THIRD_PARTY_NOTICES.md)，验收状态见[测试报告](TEST_REPORT.md)。

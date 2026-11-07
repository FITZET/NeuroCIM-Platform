# NeuroCIM 平台源码审查材料

> 审查范围：`5_NeuroCIM_opt`、`3_models`，并追踪平台实际调用的 `1_data_fitting` 公共接口。本文只描述当前源码已经实现的行为。函数名可作为技术报告中的源码定位依据。

## 1. 结论与系统边界

平台由两条相互关联、但计算职责不同的链路组成：

1. **数值精度链路**：器件 CSV 经 `1_data_fitting` 去除行/列基线并生成 256×256 交互乘法 LUT；`HardwareErrorModel` 对 LUT 做版本化和完整性校验；`DifferentialLinear`、`DifferentialConv2d` 在 PyTorch 中用该 LUT 完成有符号 MAC；同一计算层用于单样本推理、批量推理和误差感知训练。
2. **PPA 链路**：`neurosim_adapter.py` 将网络展开为 `X+ / X-` 两个输入分支和 `W+ / W-` 两个权重分支，调用 vendored 官方 NeuroSim C++ 可执行文件，解析面积、延迟、能量、泄漏、吞吐和能效。**NeuroSim 内核不读取器件误差 LUT，也不负责四路模拟结果的带符号数字组合。**

因此，报告中应写成“基于 NeuroSim 的 PPA 内核 + 本项目的器件误差数值仿真/训练前端”，不能写成“NeuroSim 使用实测 LUT 完成器件误差电路仿真”。当前 GUI 的 PPA 用例仅为 **LeNet-5/MNIST**，五模型通用 PPA 尚未实现。

## 2. 器件 CSV 导入、拟合与误差模型

### 2.1 数据导入与校验

核心文件与函数：

- `1_data_fitting/data_fitting/loader.py::inspect_csv_columns`：仅读取表头，供 GUI 列映射控件使用。
- `loader.py::_read_csv`：自动依次尝试 `utf-8-sig`、`gbk`，以及逗号、分号；仅接受解析出多列的组合。
- `loader.py::load_dataset`：建立规范列 `X, W, VD, VG, output`，支持用户映射，也支持默认器件列 `VD_Digital, VG_Digital, VD, VG, Iout` 等。
- `loader.py::_parse_code`：支持十进制、二进制和自动识别；位宽只允许 1/2/4/8 bit，代码范围为 `[0, 2^b-1]`。
- `loader.py::_validate_grid`：检查重复 `(X,W)`、完整网格、NaN/Inf。`missing_policy` 可为 `error`、`drop`、`interpolate`；插值仅用于 `VD/VG/output`，`X/W` 不允许插值。
- `5_NeuroCIM_opt/neurocim_opt/application.py::PlatformService.preview_cell/fit_cell`：GUI 无关的导入与拟合入口。

实际关键逻辑可在报告中节选为：

```python
frame = pd.DataFrame({
    "X": _parse_code(raw[mapping["X"]], ..., code_max, forced_format, ...),
    "W": _parse_code(raw[mapping["W"]], ..., code_max, forced_format, ...),
    "output": pd.to_numeric(raw[mapping["output"]], errors="coerce"),
})
frame = frame.sort_values(["W", "X"], kind="stable").reset_index(drop=True)
_validate_grid(frame, levels, require_complete=config.missing_policy == "error")
```

### 2.2 分析拟合与可部署 LUT 的区别

`1_data_fitting/data_fitting/fitting.py` 生成多种分析拟合：

- `fit_code_voltage`：`VD = a0 + a1 X`，`VG = b0 + b1 W`；
- `fit_linear_product`：归一化输出对归一化理想乘积的一阶拟合；
- `fit_raw_product_output`：`output = intercept + slope·XW`；
- `fit_quadratic_surface`：六项二次响应面；
- `fit_full_surface`：`output = α VD·VG + β VD + γ VG + ρ`。

这些是诊断/解释模型，不直接用于网络推理。真正可部署的器件乘法表由 `build_interaction_lut` 生成。它先从实测输出矩阵 `I(W,X)` 提取交互项：

\[
I_{\mathrm{int}}(W,X)=I(W,X)-I(W,0)-I(0,X)+I(0,0).
\]

此式保证零输入或零权重轴为零，避免正负差分中重复累加器件静态基线。随后用最小二乘标量增益把交互电流映射到理想整数乘积：

\[
g=\frac{\sum_{W,X} I_{\mathrm{int}}(W,X)\,XW}
        {\sum_{W,X}I_{\mathrm{int}}^2(W,X)},\qquad
H(W,X)=\operatorname{clip}\!\left(\operatorname{round}(gI_{\mathrm{int}}),0,65535\right).
\]

源码对应：

```python
native_interaction = output_grid - output_grid[:, [0]] \
    - output_grid[[0], :] + output_grid[0, 0]
product = np.multiply.outer(np.arange(256), np.arange(256)).T.astype(float)
gain = np.sum(interaction * product) / np.sum(interaction**2)
lut = np.clip(np.rint(gain * interaction), 0, 65535).astype(np.uint16)
```

如果原始采集位宽小于 8 bit，代码会在归一化坐标中把交互曲面双线性地插值到 256×256；这属于表面重采样，不是新增测量点。

指标实现位于 `fitting.py::calculate_metrics`：

\[
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2},\quad
RMSE=\sqrt{\frac1N\sum_i(y_i-\hat y_i)^2},
\]

\[
MAE=\frac1N\sum_i|y_i-\hat y_i|,\quad
NL_{FS}=\frac{\max_i|y_i-\hat y_i|}{y_{\max}-y_{\min}}\times100\%.
\]

需说明退化情况：总方差为零时，代码对完美预测返回 `R²=1`，否则返回 `NaN`；满量程为零且误差也为零时非线性为零。

### 2.3 模型包、平滑变体与完整性

核心文件：

- `neurocim_opt/error_model.py::HardwareErrorModel`：LUT 约定为 `lut[W,X]`；严格要求 256×256、有限、整数、uint16 范围、零轴为零。
- `HardwareErrorModel.save/load`：保存 `multiplication_lut.npy` 和 `error_model.json`；加载时核对 SHA-256。
- `HardwareErrorModel.effective_lut`：用误差强度 `λ` 插值：

\[
H_\lambda(W,X)=\operatorname{round}\{XW+\lambda[H(W,X)-XW]\},\quad 0\le\lambda\le1.
\]

- `neurocim_opt/workflow.py::_smooth_interaction_lut`：由经验 LUT 另生成一阶和二阶平滑交互曲面。它拟合的是 `H=XW·g(X,W)`，从结构上保持零轴：

\[
g_1=a_0+a_1x+a_2w,
\]

\[
g_2=a_0+a_1x+a_2w+a_3x^2+a_4xw+a_5w^2,
\quad x=X/255,\;w=W/255.
\]

`ErrorModelRegistry.register_fitting_run` 会把分析拟合登记为 `selectable=False`，把经验交互 LUT、一阶平滑 LUT、二阶平滑 LUT 登记为 `deployable_error_model` 和 `selectable=True`。这是防止把回归曲线误当作推理 LUT 的关键约束。

### 2.4 已有 UOS1 真实证据

`outputs/demo/uos1_gui/fit_result/fitting/fitting_summary.json` 记录了 `03_UOS_1.csv` 的真实结果：数据为完整 8-bit 网格，映射为 `X=VD_Digital, W=VG_Digital, output=Iout`；交互模型 `R²=0.9940709785`、`RMSE=2.9621×10^-8 A`、`MAE=1.9816×10^-8 A`、`NL_FS=5.7781%`。报告可引用这些数值，但应注明它们是“交互电流与反标定理想乘积”的指标。原始输出二次曲面拟合的 `R²=0.9999486006`，不能与交互 LUT 指标混为一谈。

### 2.5 独立的阵列 MAC 回归模块

`neurocim_opt/mac_error_model.py` 是另一类数据模型。`load_mac_csv` 按 `(array_rows,array_cols,bit_width,group)` 规范化数据；`MacErrorModel.fit` 对 `ideal_mac → measured_output` 分组执行一次和指定 1/2/3 次多项式拟合；`predict` 默认禁止超出拟合区间外推。该模块用于阵列级分析，当前没有接入 `DifferentialLinear/Conv2d`，报告不应说它驱动网络误差推理。

## 3. 有符号差分 LUT 层

核心文件：`5_NeuroCIM_opt/neurocim_opt/layers.py`。

### 3.1 量化与乘法

`_quantize_magnitude` 采用逐元素绝对值、四舍五入和饱和：

\[
q_x=\operatorname{clip}(\operatorname{round}(|x|/s_x),0,255),\qquad
q_{w,j}=\operatorname{clip}(\operatorname{round}(|w_j|/s_{w,j}),0,255).
\]

权重尺度由每个输出通道最大绝对值确定：

\[
s_{w,j}=\frac{\max_k|w_{jk}|}{255};
\]

输入尺度为每层一个标量。`_signed_lut_product` 并未显式构造四个大张量，而是利用零轴交互 LUT 的等价形式：

\[
p_H(x,w)=\operatorname{sgn}(x)\operatorname{sgn}(w)H(q_w,q_x).
\]

```python
qx = _quantize_magnitude(x, x_scale)
qw = _quantize_magnitude(w, w_scale)
magnitude = lut[qw, qx]
sign = torch.sign(x).to(torch.int64) * torch.sign(w).to(torch.int64)
return magnitude * sign
```

若按物理分支展开，等价表达为：

\[
x=x^+-x^-,\quad w=w^+-w^-,
\]

\[
y=M(x^+,w^+)-M(x^+,w^-)-M(x^-,w^+)+M(x^-,w^-).
\]

这里必须强调 `M` 表示零轴已校正的交互乘法 LUT。Python 实现是符号乘以一次 LUT 查询，不是逐项调用四次 LUT；四路展开主要用于物理映射和 NeuroSim PPA trace。

### 3.2 线性层与卷积层

`DifferentialLinear.forward` 按 `mac_chunk_size=64` 分块，在 int64 累加后反量化：

\[
y_j=s_xs_{w,j}\sum_k p_H(x_k,w_{jk})+b_j.
\]

`DifferentialConv2d.forward` 用 `torch.nn.functional.unfold` 将卷积转为滑窗矩阵，支持 groups，但只支持数值 padding 和 `padding_mode="zeros"`。默认按 32 个核元素分块累加。偏置在反量化后加入；池化、BatchNorm、激活、残差、softmax 等仍由 PyTorch 浮点算子执行。

`adapt_module` 按全限定层名把模型中所有普通 `nn.Conv2d/nn.Linear` 原位替换为差分层，并要求 `input_scales` 覆盖所有待转换层。

### 3.3 STE

训练使用：

```python
def _ste(exact, surrogate):
    return surrogate + (exact - surrogate).detach()
```

所以前向值严格等于 LUT 路径 `exact`，反向梯度等于浮点 `F.linear` 或 `F.conv2d` 的梯度：

\[
y_{STE}=y_{float}+\operatorname{stopgrad}(y_H-y_{float}),
\]

\[
y_{STE}^{forward}=y_H,\qquad
\frac{\partial y_{STE}}{\partial \theta}=
\frac{\partial y_{float}}{\partial \theta}.
\]

这比笼统写“量化函数梯度近似为 1”更贴合当前代码。

## 4. 推理数据流

### 4.1 资源加载与缓存

`neurocim_opt/inference_runtime.py` 负责加载数据集、checkpoint、输入尺度和误差模型，并构造三种模型：浮点/原生、理想差分、器件差分。缓存键不仅包含路径，还包含文件大小和 `mtime_ns` 的 `ResourceSignature`，资源变化后不会静默复用旧权重或 LUT。

`PlatformService.infer_registered_single` 支持 LeNet-5、BNN、ResNet-20、TinyTransformer；对每种模式计算 softmax、Top-1、Top-5 及耗时。GPT-2 通过独立的 `GPT2GenerationService` 处理。

### 4.2 批量图像推理

`neurocim_opt/batch_inference.py::run_grouped_batch_inference` 对相同样本依次运行 `float`、`ideal_differential`、`device_differential`，记录逐样本正确性，再按 `group_size=N` 分组：

\[
Accuracy=\frac{N_{correct}}{N_{total}},\qquad
Accuracy_g=\frac{1}{N_g}\sum_{i\in g}\mathbf1(\hat y_i=y_i).
\]

`max_eval_batches` 或批量请求的 `sample_count` 限制属于机制验证时的抽样，不能标成完整测试集精度。

### 4.3 GPT-2 生成

`neurocim_opt/gpt2_service.py::GPT2GenerationService.run` 加载本地 GPT-2 tokenizer、checkpoint、输入尺度和器件模型，分别执行浮点、理想差分、器件差分三种模式。`_generate` 每步选择概率最高的 token，是贪心解码：

\[
\hat x_t=\arg\max_v P(v\mid x_{<t}).
\]

服务返回生成 token、最终一步 Top-5、耗时和 token/s。GUI 的 `GPT2BatchInferenceWidget._agreement` 才负责比较 token 序列一致率：

\[
Agreement=\frac1T\sum_{t=1}^{T}\mathbf1(\hat x_t^{candidate}=\hat x_t^{float}).
\]

当前生成服务不计算 NLL 或 perplexity；这两个字段虽列在模型注册元数据中，但报告不得宣称当前 GUI 已输出困惑度。

## 5. 误差感知训练

核心文件：`neurocim_opt/training_service.py::ErrorAwareTrainingService`。

支持 LeNet-5、BNN、ResNet-20、TinyTransformer；`canonical_model_id` 明确排除 GPT-2。验证阶段要求 `0≤λ≤1`、有效 epoch/batch/lr、数据集与模型资源存在。

训练流程为：

1. 用测试集分别评估训练前 float、ideal differential、device differential；
2. 用 `device_lut = effective_lut(λ)` 和 `ste=True` 构造训练模型；
3. 使用交叉熵和 Adam，仅优化 `requires_grad=True` 参数；
4. 每轮在验证集选最优权重；
5. 训练后再次执行三模式测试；
6. 保存 checkpoint、history、结果 JSON，并计算参数 SHA-256、L2 变化量、最大变化量。

目标可写为：

\[
W^*=\arg\min_W\mathbb E_{(x,y)}
\left[-\sum_c y_c\log\operatorname{softmax}(f_{H_\lambda}(x;W))_c\right].
\]

关键训练代码：

```python
model = differential(base(), device_lut, True).to(resources.device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    (p for p in model.parameters() if p.requires_grad), lr=request.learning_rate)
logits = model(inputs)       # 前向为器件 LUT
loss = criterion(logits, targets)
loss.backward()              # 反向为浮点 surrogate 梯度
optimizer.step()
```

`max_train_batches`、`max_eval_batches` 为显式 smoke-run 限制；`None` 才表示完整数据 pass。结果 JSON 的 `formal_evaluation` 仅由 `max_eval_batches is None` 判定，报告应同时列出训练批次数，避免把单 batch 训练称为正式训练。

TinyTransformer 是特例：`_configure_tinytransformer_head_finetuning` 冻结全部参数，只解冻 `classifier.attention_pool` 和 `classifier.fc`；训练时 `differential_head_copy` 也只转换这两个 Linear。这样避免在 Transformer 全骨干中构造代价极高的 LUT 广播中间量。推理路径则调用 `tiny_transformer.engine::differential_copy`，会转换模型中的全部 Conv2d/Linear；必须把“推理映射范围”和“训练微调范围”分别描述。

GPT-2 没有加入 `TrainingRequest` 的模型别名、数据加载器和训练分支，GUI 也在继承到 `gpt2-small` 时禁用训练。源码没有给出“理论上不能训练”的结论；准确说法是“当前版本未实现 GPT-2 误差感知训练”。

## 6. NeuroSim 集成

### 6.1 上游版本与补丁

`neurocim_opt/neurosim_adapter.py` 固定记录：

- repository：`https://github.com/neurosim/NeuroSim`
- branch：`2DInferenceV1.4`
- commit：`8a88abf85844c0e1ba17cc771ea535fff6040456`
- archive SHA-256：`102cac5f...7a0433e2`

`patches/neurosim_2dinferencev1.4.patch` 只把 `Param.cpp` 的 `novelMapping` 从 `true` 改为 `false`，使用 conventional mapping。原因记录为：扩展后的通用矩阵、5×5 卷积和全连接层不适用于该构建中的上游 novel mapping 路径。

### 6.2 四路 trace 导出

`export_differential_traces` 给所有 Conv2d/Linear 注册 forward pre-hook，用一次真实模型前向捕获每层输入；随后：

- `_expanded_differential_weight` 按输出通道量化权重幅值，拆为正权重/负权重，并在两个输入分支上复制，形成 2 倍输出、2 倍输入维度的扩展阵列；
- 输入拆为 `X+` 和 `X-`，沿通道维拼接；
- 卷积使用上游 `utee.hook.stretch_input` 做 im2col；
- `_write_unsigned_bitplanes` 以 MSB→LSB 写出 8 个激活 bit plane；
- 网络 CSV 每层记录 `[H,W,2Cin,Kh,Kw,2Cout,pool,stride]`，全连接写为 `[1,1,2Fin,1,1,2Fout,0,1]`。

trace manifest 明确声明：

```text
mapping: four non-negative paths: X+/X- rows by W+/W- columns
digital_combination: Y = X+W+ - X+W- - X-W+ + X-W-;
                     not modeled by NeuroSim core
```

适配器限制：当前 measured multiplication model 只允许 8-bit；卷积只支持 `groups=1`、`dilation=(1,1)`，且网络 CSV 只使用方形卷积核的第一个尺寸作为 `window_size`；非 MAC 运算只能记入 `excluded_non_mac_ops`，并未由适配器逐项建模。

### 6.3 执行与结果解析

`run_neurosim` 构造命令：

```text
neurosim.exe network.csv weight_bits activation_bits
             subarray_rows parallel_rows
             layer1_weight.csv layer1_input.csv ...
```

`NeuroSimConfig.validate` 只检查位宽为正、子阵列行数为正、`1≤parallel_rows≤subarray_rows`。当前 config 没有独立的 `subarray_cols` 字段。

`parse_neurosim_output` 用正则直接读取 C++ 输出的指标，而不是在 Python 中自行用 `A_total=ΣA_l` 等公式重算。可在报告中解释一般关系：

\[
P_{avg}=E/T,\quad Throughput=N_{op}/T,\quad
EnergyEfficiency=N_{op}/E,
\]

但最终表格必须标注为“NeuroSim 输出字段”，尤其不能把逐层 latency 简单相加来替代工具报告的 chip latency，因为上游同时区分 layer-by-layer 和 pipelined 口径。

### 6.4 当前 PPA 用例及真实结果

`PlatformService.evaluate_lenet5_ppa` 只实现 LeNet-5：加载 5 个 MAC 层及量化尺度，使用固定全零 `[1,1,32,32]` 输入导出 trace，池化位置为 `conv1/conv2`，然后运行官方二进制。加载器件误差 manifest 只是完整性和工作流一致性校验，LUT 内容没有传入 NeuroSim。

`outputs/neurosim/01_NMOS/neurosim_result.json` 的已验证运行返回码为 0，包含五层，配置 8/8 bit、128 行子阵列、128 并行行。主要结果：面积 `4,723,600 μm²`，读延迟 `22,134.4 ns/image`，读动态能 `230,426 pJ/image`，泄漏能 `1,774.1 pJ/image`，泄漏功率 `80.1516 μW`，能效 `19.2858 TOPS/W`，吞吐 `0.246827 TOPS`、`45,178.6 FPS`，计算密度 `0.052254 TOPS/mm²`。

限制必须同时呈现：该 trace 由全零输入生成，不能视为真实数据集平均活动率；PPA 仅覆盖 LeNet-5；数字四路加减不在 NeuroSim 内核；非 MAC 运算覆盖范围由上游模型和网络 CSV 语义决定；这些数值是工具模型估计，不是芯片实测。

## 7. 五个模型的真实实现与映射范围

### 7.1 LeNet-5 / MNIST

- 注册：`model_registry.py::builtin_registry`，输入 `(1,32,32)`。
- 结构：`3_models/LeNet_5/lenet5/model.py::LeNet5`；`Conv(1→6,5)`、平均池化、`Conv(6→16,5)`、平均池化、`FC 400→120→84→10`，Tanh 激活。
- 差分映射：`lenet5/differential.py::configure_differential_model` 通过 `adapt_module` 转换全部 2 个卷积和 3 个全连接层。
- 输入尺度：`symmetric_input_scales` 从已保存 affine calibration 参数恢复浮点最小/最大值，再取最大绝对值/255；差分计算不使用 affine zero-point。
- 数字边界：池化、Tanh、flatten、softmax 为 PyTorch 浮点。
- PPA：当前唯一接入 NeuroSim 的网络。

### 7.2 BNN / MNIST

- 结构：`3_models/BNN/bnn/model.py::BinaryMNIST`；四个二值卷积、两个二值全连接，隐藏阶段包含 BN、二值激活和两次最大池化。
- `_BinarySignSTE` 前向把零归为 `+1`，反向仅在 `|x|≤1` 传递梯度。
- 部署图：`BinaryMNIST.frozen` 把权重固化为 `±1` 的普通 Conv/Linear。
- 差分映射：`bnn/differential.py::differential_models` 在 frozen graph 上转换全部 Conv/Linear。输入尺度由 `collect_input_scales` 在校准数据的绝对最大值上计算。
- 原生模式在 UI 中名为 `native_bnn`，不应称为普通浮点 LeNet 式 baseline。
- 训练服务对 frozen deployment graph 使用差分层 STE；BN、偏置以外的 PyTorch 运算仍保持数字/浮点。

### 7.3 ResNet-20 / CIFAR-10

- 结构：`3_models/Resnet20/resnet20/model.py::ResNetCifar`；首层 3×3 Conv，三个 stage、每个 3 个 BasicBlock，通道 16/32/64；跨级 shortcut 使用 1×1 Conv；自适应平均池化后 FC。
- 差分映射：`resnet20/engine.py::differential_copy` 深拷贝后转换全部 Conv2d/Linear，包括 shortcut 1×1 Conv 和最终 FC。
- 输入尺度：`calibrate_input_scales` 用若干校准 batch 捕获每层输入绝对最大值/255；当前保存尺度直接从 JSON 加载。
- 数字边界：BatchNorm、ReLU、残差加法、adaptive average pooling、softmax 留在 PyTorch。
- 训练：统一训练服务把全网络标为训练范围，优化器为 Adam，而模型自身 `engine.py` 另有 SGD 训练辅助函数；报告描述平台 EAT 时应以 `training_service.py` 为准。

### 7.4 TinyTransformer（CCT-7/3×1）/ CIFAR-10

- 工厂：`tiny_transformer/cct.py::cct_7_3x1_32`；图像经卷积 tokenizer 变为 token 序列，再进入 `TransformerClassifier`。
- 注册输入 `(3,32,32)`，checkpoint 为 `cct_7_3x1_32_cifar10.pth`。
- 推理：`tiny_transformer/engine.py::differential_copy` 转换所有 Conv2d/Linear，因此包括 tokenizer、Transformer 内线性层和分类头。
- 训练：只转换并训练 `classifier.attention_pool` 与 `classifier.fc`；其余骨干冻结。
- LayerNorm、attention softmax、GELU、序列加权、位置编码等不经 LUT。
- 报告不能把“head-only EAT”误写为全模型误差感知训练。

### 7.5 GPT-2 Small / 自回归文本生成

- 结构：`3_models/GPT2/gpt2_cim/model.py`；配置 `12 layers / 12 heads / 768 embedding / context 1024`，token embedding 与 `lm_head` 权重共享。
- 自回归模型：

\[
P(x_1,\ldots,x_T)=\prod_{t=1}^{T}P(x_t\mid x_{<t}),\qquad
\mathcal L_{NLL}=-\frac1T\sum_t\log P(x_t\mid x_{<t}).
\]

- 差分映射严格只有 `gpt2_cim/engine.py::DIFFERENTIAL_LAYERS` 两层：最后一个 block 的 `transformer.h.11.attn.c_proj` 与 `transformer.h.11.mlp.c_proj`。
- `c_attn`、`mlp.c_fc`、前 11 个 block、embedding 和 `lm_head` 均保持原生浮点。
- tokenizer 由本地 Hugging Face `GPT2Tokenizer` 加载，属于 GPT-2 byte-level BPE 资源。
- 当前是贪心生成，最大新增 token 为 128；支持三模式、取消、进度、token/s、最终 Top-5。
- 当前没有 GPT-2 EAT，也没有实际 perplexity 评估流程。

## 8. GUI 与工作流状态

### 8.1 组件关系

`neurocim_opt/gui.py::MainWindow` 用 PySide6 `QTabWidget` 组织器件拟合/阵列分析、网络与数据集、单样本推理、批量推理、误差感知训练和 NeuroSim PPA；每页经 `_scrollable` 包装为 `QScrollArea`。

耗时任务使用 `QThread`：`CellFitThread`、`InferenceThread`、`BatchInferenceThread`、`TrainingThread`、`PpaThread`、`GPT2GenerationThread/GPT2BatchThread`。训练、图像批量推理和 GPT-2 支持协作式取消，进度通过 Qt Signal 回到 UI；PPA 线程当前没有取消接口。

### 8.2 唯一网络选择与动态页面

`ModelCatalogWidget` 从 `builtin_registry` 展示五个模型；`WorkflowCoordinator.select_network_model` 持久化选择并发射 `network_changed`。`UnifiedSingleInferenceWidget` 和 `UnifiedBatchInferenceWidget` 监听该信号：文本生成时切到 GPT-2 页面，图像分类时切到共享图像页面，并禁用其中的模型下拉框。因此工作流模式下“网络与数据集”页是模型组合的唯一入口。

### 8.3 状态继承与失效

`workflow.py::WorkflowContext` 以临时文件替换实现原子 JSON 写入，保存到 `outputs/workflow/workflow_context.json`：

- `select_network_model`：模型改变时清空 inference/training；
- `select_error_model`：误差模型选择后无条件清空 inference/training；
- `record_inference`：校验误差模型 id 和网络选择一致性，保存 checkpoint、辅助尺度、数据集、设备和结果，并清空旧 training；
- `training_inputs`：只有成功推理后才返回训练继承参数；
- `record_training`：只有推理仍成功时才记录训练结果。

训练页继承到 GPT-2 时明确禁用运行，因为 `training_service.py` 不支持该模型。误差模型下拉框只列 `selectable=True` 的可部署模型。

## 9. 报告中建议引用的核心代码片段

以下片段覆盖关键链路，适合放在各章，而不需要大段复制实现：

1. `fitting.py::build_interaction_lut`：四项基线消除、增益最小二乘、LUT clipping。
2. `error_model.py::effective_lut`：理想/器件 LUT 的 λ 插值。
3. `workflow.py::_smooth_interaction_lut`：`XW·g(X,W)` 约束拟合。
4. `layers.py::_signed_lut_product`：幅值量化、`lut[qw,qx]`、符号恢复。
5. `layers.py::DifferentialLinear.forward`：分块 int64 MAC 和反量化。
6. `layers.py::DifferentialConv2d.forward`：unfold、group、分块 MAC。
7. `layers.py::_ste`：器件前向/浮点反向。
8. `inference_runtime.py::get_registered_inference_resources`：四个分类模型的资源路由。
9. `application.py::infer_registered_single`：三模式、softmax、Top-1/Top-5。
10. `batch_inference.py::run_grouped_batch_inference`：分组准确率。
11. `training_service.py::ErrorAwareTrainingService.run`：训练前后三模式评估和 EAT。
12. `gpt2_service.py::_generate`：贪心自回归生成。
13. `neurosim_adapter.py::export_differential_traces`：四路非负 trace。
14. `neurosim_adapter.py::run_neurosim/parse_neurosim_output`：进程执行和指标解析。
15. `workflow.py::WorkflowContext` 与 `gui.py::WorkflowCoordinator`：跨页面状态与失效规则。

## 10. 必须如实写入的限制

- 器件误差是确定性的二维 `(X,W)` LUT；没有温度、电压、工艺角、时间漂移、阵列位置相关性或随机噪声维度。
- LUT 只表示去基线后的交互乘法；ADC/DAC、IR drop、读写噪声等没有在 Python 数值层中分别建模。
- 输入为逐层单标量 scale，权重为逐输出通道 scale；均采用最大绝对值校准，异常值可能压缩有效码域。
- 差分层计算成本高，虽有分块仍可能产生很大的广播中间张量；这也是大模型只映射少数层、TinyTransformer EAT 只训练头部的现实约束。
- GPT-2 只映射最后一个 block 的两个输出投影；不能据此声称完成 GPT-2 全网络 CIM 映射。
- GPT-2 没有 EAT、NLL/perplexity 评估；当前 token agreement 位于 GUI 批处理逻辑。
- NeuroSim PPA 仅有 LeNet-5 服务入口，使用全零样本 trace；器件 LUT 未传给 NeuroSim，数字差分组合不由 NeuroSim 计入。
- NeuroSim 输出指标由正则解析；Python 未独立复算或交叉验证各项工具公式。
- 当前 NeuroSim 补丁启用 conventional mapping；重新同步上游版本后需重新验证补丁和输出正则。
- 分析回归模型、阵列 MAC 回归模型和可部署交互 LUT 是三类不同对象，报告和图表必须分开命名。


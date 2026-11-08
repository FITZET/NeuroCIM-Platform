from pathlib import Path
import json
import sys
from datetime import date
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

PLATFORM=Path(__file__).resolve().parents[2]
ROOT=PLATFORM.parent
ASSETS=PLATFORM/'docs/report_assets'
sys.path.insert(0,str(ASSETS))
from report_builder import (ReportBuilder, validate_docx, math_text, math_concat,
    math_fraction, math_sub, math_sup, math_subsup, math_radical, math_nary,
    math_delimiter)

OUT=PLATFORM/'docs/基于NeuroSim的DRAM存算仿真平台开发技术报告.docx'
b=ReportBuilder('基于 NeuroSim 的 DRAM 存算仿真平台开发技术报告',header_text='基于 NeuroSim 的 DRAM 存算仿真平台开发技术报告')
b.add_cover('器件误差建模、差分推理、误差感知训练与 PPA 评估',[
 ('项目','DRAM 存算仿真平台'),('技术内核','官方 NeuroSim 2DInferenceV1.4'),
 ('工程路径',str(ROOT)),('报告日期','2026年10月6日'),('文档性质','依据当前源码与保留实验产物生成')])

def h(t,l=1): b.add_heading(t,l)
def p(t): return b.add_paragraph(t)
def bullets(items):
 for x in items: b.add_paragraph(x,style='List Bullet')
def cap(text): b.document.add_paragraph(text,style='Caption')
def fig(num,path,title,w=15.0):
 p(f'如图 {num} 所示，{title}。')
 b.add_image(path,width_cm=w); cap(f'图 {num} {title}')
def table(num,title,headers,rows):
 p(f'表 {num} 汇总了{title}。'); cap(f'表 {num} {title}'); return b.add_table(headers,rows)
def eq(parts,num): b.add_equation(parts,num)
def page(): b.document.add_page_break()
def snippet(title,rel,start,end):
 path=ROOT/rel; lines=path.read_text(encoding='utf-8').splitlines(); start=max(1,start); end=min(end,len(lines))
 code='\n'.join(f'{i:04d}  {lines[i-1]}' for i in range(start,end+1))
 p(f'核心代码：{title}（{rel}，第 {start}—{end} 行）。')
 b.add_code_block(code,language='Python')

# 摘要
h('摘要',1)
p('本报告面向 DRAM 存算融合的软件评估需求，说明一个以官方 NeuroSim 为硬件性能评估内核、以测量数据驱动乘法查找表为数值误差模型的桌面仿真平台。平台读取 Virtuoso、TCAD 或阵列级仿真导出的 CSV，完成编码、列映射、网格校验、交互电流提取、线性/非线性拟合和 256×256 乘法 LUT 构建；随后将 LUT 注入 PyTorch 的 Conv2d 与 Linear 层，以正负差分形式执行单样本推理、分组批量推理和误差感知训练。另一条链路把网络权重与真实活动展开为 X⁺、X⁻、W⁺、W⁻ 四条非负 trace，调用固定版本的官方 NeuroSim C++ 程序获得面积、延迟、动态能耗、泄漏功耗、吞吐和能效。')
p('平台目前登记 LeNet-5/MNIST、BNN/MNIST、ResNet-20/CIFAR-10、TinyTransformer/CIFAR-10 和 GPT-2 Small/自回归文本生成五种组合。图形界面把模型与数据集勾选设置为唯一配置入口，并将误差模型、推理结果和训练配置通过持久化工作流衔接。报告所列指标均来自当前仓库的 JSON、日志、截图或测试；未完成的规格书条款单独列出，不以计划替代实现。')
p('关键词：DRAM 存内计算；NeuroSim；器件非理想性；查找表；正负差分；误差感知训练；PPA；PyTorch')
page(); b.add_toc();

# Chapter 1
h('第1章 项目背景与需求分析',1)
h('1.1 项目背景',2)
p('传统处理器在存储器与计算单元之间频繁搬移权重和激活，数据传输能耗和延迟成为神经网络推理的重要约束。存内计算通过在存储阵列附近或阵列内部完成乘累加，减少数据搬移，但模拟器件的非线性、偏置、噪声和量化误差会改变网络数值行为。因此，平台必须同时回答两个问题：测量误差如何影响算法精度，以及给定网络映射在电路层面的面积、延迟和能效如何。NeuroSim 面向后一个问题，是公开的存算加速器层次化评估框架[1-3]。')
p('本工程采用“双链路”设计。器件误差链把测量电流转换为可重复的乘法 LUT，直接控制差分推理和训练前向；PPA 链把网络矩阵、量化位宽和活动数据交给官方 NeuroSim。两条链共享网络与活动信息，但当前官方模拟器不直接读取测量 LUT，因而不能把 PPA 输出解释为特定测量器件曲线的电路级复现。')
h('1.2 规格书目标',2)
p('规格书要求两个数据分支：单元多 bit 乘法结果，以及阵列 n-bit 乘累加结果；同时要求 CSV 自定义列映射、1/2/4/8 bit、典型阵列规格、线性度指标、网络精度评估、二次训练、单数据实时推理、数据导出、参数记忆、日志、异常处理和 Windows GUI。技术要求还包含十万行导入性能、单样本时延、批量计算时延、Windows 10/11、免安装/安装包以及持续运行等验收项。')
table('1-1','规格书要求与当前实现的对应关系',['范围','当前实现','结论'],[
 ('单元乘法数据','CSV 编码/分隔符识别、自定义映射、1/2/4/8 bit、拟合、LUT 与图表','已实现'),
 ('阵列 MAC 数据','阵列规模/位宽/分组拟合、R²/RMSE/MAE/NL、导出','已实现'),
 ('网络组合','LeNet-5、ResNet-20、TinyTransformer、BNN、GPT-2；MNIST/CIFAR-10/prompt_text','部分满足'),
 ('推理与训练','图像单样本、图像分组批量、GPT 文本单/批量、四类图像模型训练','已实现当前登记模型范围'),
 ('NeuroSim PPA','官方 2DInferenceV1.4，LeNet-5/ResNet-20 trace 与 PPA','已实现'),
 ('ImageNet/四类数据集/四种 DNN','当前未提供 ImageNet，DNN 和数据集数量未达到规格书原始目标','未完全满足'),
 ('安装器与绿色版','有环境脚本和源码运行方式，未见完成的安装器验收证据','未完成验收'),
 ('Windows 10/两小时稳定性','现有证据覆盖 Windows 11 开发环境，未完成指定稳定性实验','未完成验收')])
h('1.3 平台输入与输出',2)
bullets(['输入：器件或阵列 CSV、列映射、位宽与阵列参数、误差强度 λ、模型 checkpoint、数据集或文本提示词。','中间产物：拟合摘要、经验/一阶/二阶 LUT、误差模型注册表、输入尺度、NeuroSim trace、工作流上下文。','输出：拟合指标和可视化、三模式推理结果、分组准确率或 token 一致率、训练 checkpoint 与历史、NeuroSim PPA JSON 和日志。'])

# Chapter 2
page(); h('第2章 总体架构设计',1)
h('2.1 分层架构',2)
fig('2-1',ASSETS/'fig_2_1_system_architecture.png','平台总体架构及两条计算链的关系')
p('源代码按职责分离：1_data_fitting 提供器件网格读取与拟合；3_models 保存模型定义、数据加载、校准和 checkpoint；5_NeuroCIM_opt/neurocim_opt 提供通用误差模型、差分层、应用服务、训练、工作流、GUI 和 NeuroSim 适配；vendor/neurosim 保存固定上游源码。GUI 只编排服务，耗时计算通过 QThread 执行，避免把数值算法放入控件回调。')
h('2.2 业务流程与依赖关系',2)
fig('2-2',ASSETS/'fig_2_2_business_flow.png','从 CSV 导入到训练与 PPA 的端到端流程')
p('模型与数据集页面是网络组合的唯一选择入口。WorkflowContext 在 network_selection 中持久化 model_id、task 和 dataset_name；更换网络或误差模型会清除旧 inference 与 training 状态。这种失效策略防止 ResNet-20 的数据集路径、checkpoint 或误差选择被错误传递给 LeNet-5 或 GPT-2。成功的单样本或批量推理才会把当前资源传递给训练页面。')
snippet('模型选择导致下游状态失效','5_NeuroCIM_opt/neurocim_opt/workflow.py',317,340)
h('2.3 关键模块',2)
table('2-1','核心模块与职责',['模块','核心职责','主要输出'],[
 ('data_fitting.loader','CSV 探测、码值解析、列映射、完整网格校验','DeviceDataset'),('data_fitting.fitting','最小二乘、交互电流、LUT 和指标','FittingResult'),('error_model.py','LUT 元数据、SHA-256、λ 插值','HardwareErrorModel'),('layers.py','差分 Linear/Conv2d 与 STE','可推理/可训练模块'),('inference_runtime.py','模型与数据集资源缓存、三模式执行','SingleInferenceOutcome'),('batch_inference.py','分组批量评价','准确率与组统计'),('training_service.py','三模式训练前后评价、误差感知更新','checkpoint/history/result'),('neurosim_adapter.py','trace 导出、官方程序调用、指标解析','PPA JSON'),('gui.py/workflow.py','可视化、后台线程、状态继承','交互与持久化状态')])

# Chapter 3
page(); h('第3章 NeuroSim 内核集成方法',1)
h('3.1 上游版本与集成边界',2)
p('工程固定使用官方 neurosim/NeuroSim 仓库的 2DInferenceV1.4 分支、提交 8a88abf85844c0e1ba17cc771ea535fff6040456，源码位于 vendor/neurosim，许可为 CC BY-NC 4.0。补丁启用适合通用展开矩阵和四路有符号分支的 conventional mapping。适配器常量保存仓库 URL、分支、提交和归档 SHA-256，使结果能追溯到确定的内核版本。')
fig('3-1',ASSETS/'fig_3_1_neurosim_dual_chain.png','器件误差数值链与官方 NeuroSim PPA 链')
h('3.2 四路非负物理路径',2)
p('NeuroSim 的输入矩阵使用非负码。对有符号激活和权重分别拆分正、负分量，得到四个非负乘积。模拟器对四条物理路径计入阵列和外围电路开销，数字域按照符号组合结果。')
eq(['x = ',math_subsup('x','+',''),' − ',math_subsup('x','−',''),',   w = ',math_subsup('w','+',''),' − ',math_subsup('w','−','')],'3-1')
eq(['y = M(',math_subsup('x','+',''),',',math_subsup('w','+',''),') − M(',math_subsup('x','+',''),',',math_subsup('w','−',''),') − M(',math_subsup('x','−',''),',',math_subsup('w','+',''),') + M(',math_subsup('x','−',''),',',math_subsup('w','−',''),')'],'3-2')
snippet('差分权重矩阵展开','5_NeuroCIM_opt/neurocim_opt/neurosim_adapter.py',67,82)
h('3.3 Trace 导出与执行',2)
p('export_differential_traces 为每个 Conv2d/Linear 注册 forward-pre-hook，捕获调用方提供样本所产生的层输入活动。卷积输入使用官方 hook 工具展开，权重写成 NeuroSim 兼容矩阵，激活按 8 bit 位平面写出；网络描述文件记录层尺寸、卷积核、步幅、池化和 MAC 类型。NeuroSimConfig 校验权重/激活位宽、子阵列行数和并行行数。随后 run_neurosim 以网络 CSV、位宽、子阵列参数及每层权重/输入 trace 调用 C++ 可执行文件，并解析 stdout。当前 GUI/PPA 服务只实现 LeNet-5，并传入固定全零 [1,1,32,32] 样本以获得确定性 trace；因此该入口的活动率不是 MNIST 测试集平均活动率。仓库保留的 ResNet-20 PPA 是独立扩展运行证据，不代表当前 GUI 已提供五模型通用 PPA。')
snippet('捕获层输入并导出 trace','5_NeuroCIM_opt/neurocim_opt/neurosim_adapter.py',95,150)
h('3.4 PPA 指标解释',2)
p('结构化结果直接保留官方输出的芯片面积、读延迟、动态能耗、泄漏功耗、能效和吞吐。总量与派生指标可用下列关系解释，但正式数值以 NeuroSim 输出解析值为准。')
eq(['A_total = ',math_nary('∑',math_sub('A','l'),lower='l=1',upper='L')],'3-3')
eq(['T_total = ',math_nary('∑',math_sub('T','l'),lower='l=1',upper='L')],'3-4')
eq(['E_total = ',math_nary('∑',math_sub('E','l'),lower='l=1',upper='L')],'3-5')
eq(['P_avg = ',math_fraction('E_total','T_total')],'3-6')
eq(['Throughput = ',math_fraction('N_operation','T_total')],'3-7')
eq(['EnergyEfficiency = ',math_fraction('N_operation','E_total')],'3-8')
p('单位换算中，1 mm²=10⁶ μm²，1 μs=10³ ns，1 nJ=10³ pJ。TOPS/W 是每焦耳完成的万亿次运算；throughput_fps 则按一次网络推理所需延迟折算。层延迟不必简单相加等于芯片级关键路径，因此报告保留 NeuroSim 给出的芯片级聚合量。')

# Chapter 4
page(); h('第4章 器件数据导入与误差拟合',1)
h('4.1 CSV 导入与列映射',2)
p('loader.py 依次尝试指定或自动编码与分隔符，支持 UTF-8、UTF-8-SIG、GBK/ANSI，以及逗号和分号。inspect_csv_columns 只读取表头，GUI 再把 X、W、VD、VG、output 分别显示为下拉框；因此用户选择的是实际 CSV 列名，不需要把 gate、sense 等旧字段写死。码值解析支持 decimal、binary 和 auto，科学计数法仅适用于模拟量，不会被误判为数字码。')
p('导入后校验码值范围、重复坐标、缺失组合、NaN/Inf 和 1/2/4/8 bit。低位宽原生网格先保留，只在生成 8 bit 推理 LUT 时沿 X、W 两轴插值；missing 策略可选 error、drop 或 interpolate。')
snippet('CSV 探测与规范字段映射','1_data_fitting/data_fitting/loader.py',68,145)
h('4.2 交互电流提取',2)
p('测量输出同时包含只随输入端变化的基线、只随权重端变化的基线和真正的双变量交互。平台用四项差分消除两条单变量轴，使 X=0 或 W=0 时交互量严格为零，满足差分层 LUT 的零轴约束。')
eq(['I_int(X,W)=I(X,W)−I(X,0)−I(0,W)+I(0,0)'],'4-1')
eq(['y_ideal(X,W)=XW'],'4-2')
eq(['e(X,W)=y_device(X,W)−y_ideal(X,W)'],'4-3')
h('4.3 LUT 增益与误差强度',2)
p('build_interaction_lut 用最小二乘标量增益把交互电流投影到理想整数乘积空间。令 P 为全部 XW 构成的目标矩阵、I_int 为交互矩阵，增益 g 的闭式解与部署 LUT 为：')
eq(['g = ',math_fraction([math_nary('∑','I_int(X,W)XW',lower='X,W')],[math_nary('∑',math_sup('I_int(X,W)','2'),lower='X,W')])],'4-4')
eq(['H(X,W)=clip(round(gI_int(X,W)),0,65535)'],'4-5')
p('HardwareErrorModel 通过 λ 在理想乘法与完整器件 LUT 之间插值。λ=0 是理想整数乘法，λ=1 使用完整测量误差；中间值用于敏感性扫描。')
eq([math_sub('H','λ'),'=round(XW+λ(H−XW)),   0≤λ≤1'],'4-6')
snippet('交互电流、增益和 LUT 构建','1_data_fitting/data_fitting/fitting.py',138,182)
h('4.4 多种拟合结果',2)
p('一次拟合同时保留诊断模型和可部署误差模型。诊断项包括码值—电压标定、线性乘积、原始乘积—输出、二次归一化曲面和 VD/VG 全曲面；可部署项包括经验交互 LUT、低阶平滑交互面和二阶平滑交互面。后续页面的误差模型下拉框只列出 selectable=true 的部署项，防止把只用于解释的回归模型当作乘法表。')
eq(['H_smooth(X,W)=XW·g(',math_fraction('X','255'),',',math_fraction('W','255'),')'],'4-7')
eq(['g_1=a_0+a_1X_n+a_2W_n'],'4-8')
eq(['g_2=a_0+a_1X_n+a_2W_n+a_3',math_sup('X_n','2'),'+a_4X_nW_n+a_5',math_sup('W_n','2')],'4-9')
h('4.5 拟合指标',2)
eq([math_sup('R','2'),'=1−',math_fraction([math_nary('∑',math_sup([math_sub('y','i'),'−',math_sub('ŷ','i')],'2'),lower='i=1',upper='N')],[math_nary('∑',math_sup([math_sub('y','i'),'−ȳ'],'2'),lower='i=1',upper='N')])],'4-10')
eq(['RMSE=',math_radical(math_fraction(math_nary('∑',math_sup([math_sub('y','i'),'−',math_sub('ŷ','i')],'2'),lower='i=1',upper='N'),'N'))],'4-11')
eq(['MAE=',math_fraction(math_nary('∑','|y_i−ŷ_i|',lower='i=1',upper='N'),'N')],'4-12')
eq(['NL_FS=',math_fraction('max|y_i−ŷ_i|','y_max−y_min'),'×100%'],'4-13')
fig('4-1',ASSETS/'fig_4_1_fitting_pipeline.png','器件数据导入、拟合、LUT 构建和注册流程')
realfit=PLATFORM/'outputs/demo/uos1_gui/fit_result/fitting'
fig('4-2',realfit/'surface_overview.png','03_UOS_1 实际器件表面与拟合结果',14)
fig('4-3',realfit/'normalized_lut.png','03_UOS_1 归一化硬件乘法 LUT',14)
p('保留的 03_UOS_1 结果中，交互面 R²=0.9940709785、RMSE=2.9621×10⁻⁸ A、MAE=1.9816×10⁻⁸ A、满量程非线性=5.7781%。这些数值来自 fitting_summary.json，而非根据图形估读。')

# Chapter 5
page(); h('第5章 正负差分存内计算模型',1)
h('5.1 幅值量化与符号处理',2)
p('当前数值内核不显式创建四个巨大乘法张量，而是对输入和权重的绝对值进行无符号量化、查表，再乘 sign(x)sign(w)。逐元素上它与四路正负分解严格等价，同时显著降低 Python 张量数量。每个输出通道使用独立权重尺度，输入尺度由校准集统计。')
eq([math_sub('q','x'),'=clip(round(',math_fraction('|x|',math_sub('s','x')), '),0,255)'],'5-1')
eq([math_sub('q','w'),'=clip(round(',math_fraction('|w|',math_sub('s','w')), '),0,255)'],'5-2')
eq(['m(x,w)=sign(x)sign(w)L[',math_sub('q','w'),',',math_sub('q','x'),']'],'5-3')
eq(['ŷ=',math_sub('s','x'),math_sub('s','w'),math_nary('∑','m(x_k,w_k)',lower='k=1',upper='K'),'+b'],'5-4')
fig('5-1',ASSETS/'fig_5_1_differential_compute.png','有符号数值经幅值量化、LUT 查表和反量化完成 MAC')
snippet('有符号 LUT 乘法与 STE','5_NeuroCIM_opt/neurocim_opt/layers.py',13,32)
h('5.2 DifferentialLinear',2)
p('DifferentialLinear 将输入展平为二维矩阵，按 mac_chunk_size 分块，构造 [batch,out,chunk] 的局部广播而不是一次展开完整 K 维。每块查表后以 int64 累加，最后乘输入尺度和逐输出通道权重尺度并加偏置。int64 累加避免 8 bit 乘积在大扇入层中溢出。')
snippet('分块线性差分前向','5_NeuroCIM_opt/neurocim_opt/layers.py',142,176)
h('5.3 DifferentialConv2d',2)
p('卷积层先用 F.unfold 将滑动窗口转换为列，再按 group、输出通道和卷积核元素分块查表。输出高宽依据 PyTorch 卷积定义恢复。当前差分卷积支持数值 padding、零填充和 PyTorch groups；NeuroSim trace 适配器的边界更窄，只支持 groups=1、dilation=1。')
snippet('卷积窗口展开和差分累加','5_NeuroCIM_opt/neurocim_opt/layers.py',224,285)
h('5.4 直通估计器',2)
p('器件 LUT、round 和离散索引不可直接微分。训练时前向必须仍为 exact，反向则采用浮点 Linear/Conv2d surrogate。代码返回 surrogate+(exact−surrogate).detach()，数值等于 exact，而梯度等于 surrogate 的梯度。')
eq(['y_STE=y_sur+(y_exact−y_sur).detach()'],'5-5')
eq([math_fraction('∂y_STE','∂θ'),'=',math_fraction('∂y_sur','∂θ')],'5-6')

# Chapter 6
page(); h('第6章 神经网络模型集成',1)
fig('6-1',ASSETS/'fig_6_1_model_mapping.png','PyTorch 模型算子到差分数值域和 NeuroSim trace 的映射')
table('6-1','已登记模型、任务与差分边界',['模型组合','任务/数据','主要结构','差分映射','训练状态'],[
 ('LeNet-5 + MNIST','10类图像分类，1×28×28','2 Conv + 3 Linear','全部 Conv/Linear','支持误差感知训练'),
 ('BNN + MNIST','二值图像分类','二值 Conv/Linear + BN','冻结二值图中的 Conv/Linear','支持误差感知训练'),
 ('ResNet-20 + CIFAR-10','10类图像分类，3×32×32','20层残差网络，22个MAC层','全部 Conv/Linear','支持误差感知训练'),
 ('TinyTransformer + CIFAR-10','10类图像分类','CCT-7/3×1 tokenizer + Transformer','推理映射 Conv/Linear；训练仅池化头/分类头','支持头部微调'),
 ('GPT-2 Small + prompt_text','自回归文本生成','12层、12头、768隐藏维','最后 block 的 attn.c_proj 与 mlp.c_proj','当前仅推理')])
h('6.1 模型注册与资源约束',2)
p('model_registry.py 用 ModelSpec 统一描述 model_id、显示名、任务、数据集、factory、checkpoint、输入形状、指标和许可。图像推理组合只查询 task=image_classification；GPT-2 通过 text_generation 动态切换到文本界面。正式执行前检查 checkpoint、数据集文件和输入尺度，资源缺失时报告精确路径，不回退到随机权重。')
snippet('五类模型注册','5_NeuroCIM_opt/neurocim_opt/model_registry.py',72,123)
h('6.2 LeNet-5、BNN 与 ResNet-20',2)
p('LeNet-5 是完整链路基准，MNIST 原始 IDX 文件被划分为训练、验证和测试集；量化参数 JSON 为各层提供输入尺度。BNN 的隐藏权重与激活使用 sign，第一层输入和最终 logits 保持实值；部署前 frozen() 生成确定性二值图，再把 Conv/Linear 适配到 LUT。ResNet-20 使用 CIFAR-10 标准归一化，22 个 Conv/Linear 层经钩子校准输入尺度，残差相加、BatchNorm、ReLU 和池化留在数字域。')
h('6.3 TinyTransformer',2)
p('TinyTransformer 采用 SHI-Labs CCT-7/3×1 架构和 CIFAR-10 预训练权重。推理时 tokenizer 卷积、注意力投影、MLP 与分类相关 Linear 均可进入差分计算；LayerNorm、softmax、位置编码、随机深度和序列池化保留数字操作。误差感知训练冻结 tokenizer 和 Transformer blocks，仅开放 classifier.attention_pool 与 classifier.fc，共 2,827 个可训练参数。该策略避免训练时为所有 token、头和投影保留大规模查表中间量，解决 12 GiB GPU 上曾出现的显存溢出。')
h('6.4 GPT-2 Small',2)
p('GPT-2 使用 nanoGPT 的纯 PyTorch GPT-2 Small 定义与本地 OpenAI 预训练参数，共 12 层、12 个注意力头、隐藏维 768、上下文长度 1024，参数总数 124,439,808。byte-level BPE 把提示词编码为 token，模型在因果掩码下逐 token 给出下一个 token 的条件分布，界面采用确定性贪心解码。')
eq(['P(',math_sub('x','1'),',…,',math_sub('x','T'),')=',math_nary('∏','P(x_t|x_{<t})',lower='t=1',upper='T')],'6-1')
eq([math_sub('L','NLL'),'=−',math_fraction(math_nary('∑','log P(x_t|x_{<t})',lower='t=1',upper='T'),'T')],'6-2')
eq(['Perplexity=exp(',math_sub('L','NLL'),')'],'6-3')
eq(['TokenAgreement=',math_fraction(math_nary('∑','I(ŷ_t^device=ŷ_t^float)',lower='t=1',upper='T'),'T')],'6-4')
p('式（6-2）和式（6-3）给出语言模型标准评价定义，用于说明后续语料评估方案。当前 GPT2GenerationService 只输出生成 token、最后一步 Top-5、耗时和 token/s，批量 GUI 只计算 token 一致率；当前版本没有在测试语料上计算 NLL 或困惑度。')
p('全模型精确 LUT 会产生不可接受的中间张量，尤其 50,257 类语言模型头。当前只替换最后一个 block 的 transformer.h.11.attn.c_proj 与 transformer.h.11.mlp.c_proj，其余 11 个 block、词嵌入、最终 LayerNorm 和语言模型头保持浮点。服务逐模式加载模型并在模式之间清空 CUDA 缓存。')
snippet('GPT-2 差分层边界','3_models/GPT2/gpt2_cim/engine.py',1,88)
snippet('GPT-2 三模式贪心生成','5_NeuroCIM_opt/neurocim_opt/gpt2_service.py',115,174)

# Chapter 7
page(); h('第7章 单样本与批量误差推理',1)
h('7.1 三种计算模式',2)
p('常规图像模型和 GPT-2 使用 float、ideal_differential 和 device_differential 三种模式；BNN 的第一项按实际实现命名为 native_bnn。原生/浮点模式反映预训练网络，理想差分模式反映 8 bit LUT 量化和差分结构本身，器件差分模式在相同量化规则下使用 Hλ。三者并列可以把结构/量化误差与器件非线性影响分开。')
h('7.2 图像单样本推理',2)
p('模型页勾选图像组合后，单样本页面自动显示数据集根目录、只读模型资源、样本索引、误差方法和设备。后端读取真实测试样本，同时返回经过反归一化的图像数据、真实标签、每种模式的预测、置信度、Top-5 和耗时。MNIST 显示灰度图，CIFAR-10 显示 RGB 图。')
eq(['Confidence=max_c softmax(z)_c'],'7-1')
eq(['Top5=arg top-5_c softmax(z)_c'],'7-2')
h('7.3 图像批量推理',2)
p('批量页面按 sample_count 截取测试集，使用 inference_batch_size 分批执行，并按 group_size 形成连续样本区间。柱状图对每一组并列显示三种模式准确率，末组允许少于 N 个样本。')
eq(['Accuracy=',math_fraction('N_correct','N_total')],'7-3')
eq([math_sub('Accuracy','g'),'=',math_fraction(math_nary('∑','I(ŷ_i=y_i)',lower='i∈g'),'N_g')],'7-4')
fig('7-1',PLATFORM/'outputs/acceptance/batch_inference_uos1/batch_result_accuracy.png','实际批量推理分组准确率柱状图',14)
h('7.4 GPT-2 单条与批量推理',2)
p('勾选 GPT-2 后，同一个“单样本推理”标签切换为提示词输入和三模式文本生成，不再把模型名称与功能页面并列。批量页面接受每行一条提示词，展示浮点生成、器件生成、每条 token 一致率和三模式总耗时，并计算平均一致率。结果保存为 JSON，包含 token IDs、最后一步 Top-5 token 概率和差分层列表。')

# Chapter 8
page(); h('第8章 误差感知训练',1)
fig('8-1',ASSETS/'fig_8_1_eat_pipeline.png','误差感知训练的精确前向与 STE 反向流程')
h('8.1 优化目标',2)
p('训练服务先在测试集上评价预训练 float、理想差分和器件差分，然后构建 ste=True 的器件差分模型。每个训练 batch 使用器件 LUT 的精确前向、交叉熵和 Adam 更新；每个 epoch 在验证集上选择最佳状态，训练结束再评价三种模式。')
eq([math_sup('W','*'),'=arg min_W E_{(x,y)}[L(f_device(x;W,H_λ),y)]'],'8-1')
eq([math_sub('L','CE'),'=−',math_nary('∑','y_c log p_c',lower='c=1',upper='C')],'8-2')
eq([math_fraction('∂Q(x)','∂x'),'≈1   （有效量化区间）'],'8-3')
h('8.2 模型相关训练策略',2)
p('LeNet-5 与 ResNet-20 训练其差分 Conv/Linear 参数；BNN 先转为冻结二值部署图，再由层级 STE 更新可训练 MAC 参数；TinyTransformer 冻结浮点骨干，仅训练序列池化和分类头。训练请求包含 seed、epoch、batch、learning_rate、完整/受限 batch 标志和设备。max_train_batches/max_eval_batches 只用于机制 smoke，None 才表示完整遍历。')
snippet('误差感知训练的三模式基线与优化器','5_NeuroCIM_opt/neurocim_opt/training_service.py',330,405)
h('8.3 产物与恢复',2)
p('输出目录包含 eat_checkpoint.pt、history.json 和 training_result.json。checkpoint 保存模型、优化器、epoch、最佳准确率、请求配置和必要状态；结果记录训练前后六组评价、权重 L2 变化、变化参数张量和训练策略。GUI 提供进度、取消和 JSON 导出。GPT-2 当前不开放训练，因为平台尚未定义具有许可和固定切分的文本训练语料，也未实现内存可控的最后层语言建模微调。')

# Chapter 9
page(); h('第9章 GUI 与工作流管理',1)
fig('9-1',ASSETS/'fig_9_1_gui_workflow.png','统一模型勾选、误差选择和下游状态继承')
h('9.1 页面组织',2)
p('主窗口包含“误差建模与阵列分析”“网络与数据集”“单样本推理”“批量推理”“误差感知训练”和“NeuroSim PPA”六个顶层标签。误差建模内部再分器件误差拟合和阵列 MAC 分析。所有主页面置于可伸缩 QScrollArea，窗口高度按 Windows availableGeometry 设置，避免任务栏遮挡底部按钮。')
h('9.2 唯一模型组合入口',2)
p('模型页为每个已登记组合提供互斥勾选框。选择写入 WorkflowContext 并发出 network_changed；UnifiedSingleInferenceWidget 和 UnifiedBatchInferenceWidget 根据 task 在图像分类与 GPT 文本组件之间切换。页面内部保留的模型控件被禁用，只展示继承值，从交互层消除“不同页面各选一套模型”的不一致。')
snippet('统一单样本/批量页面动态切换','5_NeuroCIM_opt/neurocim_opt/gui.py',2507,2579)
h('9.3 误差模型注册和状态传递',2)
p('一次器件拟合完成后，ErrorModelRegistry 登记所有分析结果，但下拉框仅展示三种可部署 LUT。选择误差模型会写入 fitting 节点并清空 inference/training；成功推理记录 network_model、dataset_root、checkpoint、auxiliary、device 和 result，训练页面再读取 training_inputs。写文件采用临时文件替换，避免进程中断留下半个 JSON。')
snippet('推理资源写入训练上下文','5_NeuroCIM_opt/neurocim_opt/workflow.py',347,438)
h('9.4 后台任务与异常处理',2)
p('拟合、推理、批量、训练、GPT 和 PPA 都使用独立 QThread。信号传递进度、结果、失败堆栈和取消状态；GUI 线程只更新控件。资源验证在启动线程前执行，错误信息包含缺失文件用途和绝对路径。QSettings 保存最近路径、编码、分隔符、位宽和设备，工作流 JSON 保存跨页面业务状态。')

# Chapter 10
page(); h('第10章 实验验证与结果分析',1)
h('10.1 证据分级原则',2)
p('本章只引用仓库中可复核的产物，并区分完整测试集、正式训练、小样本机制验证和单 batch smoke。小样本结果证明误差确实进入计算图，但不能替代完整测试集精度；单种子训练提升也不构成统计显著性结论。')
h('10.2 器件拟合与导入性能',2)
table('10-1','器件拟合和数据处理证据',['项目','真实结果','证据'],[
 ('01_NMOS 交互面','R²=0.9908762695；NL=6.5784978477% FS','1_data_fitting/output/01_NMOS/fitting_summary.json'),
 ('03_UOS_1 交互面','R²=0.9940709785；NL=5.7780978235% FS','outputs/demo/uos1_gui/.../fitting_summary.json'),
 ('04_UOS_2 交互面','R²=0.9846901626','1_data_fitting/output/04_UOS_2/fitting_summary.json'),
 ('十万行阵列 MAC 导入','4.09 MB，0.227 s，低于 3 s 目标','outputs/acceptance/performance/benchmark.json'),
 ('GUI 反例','缺列、3 bit、缺失数值均被拒绝','outputs/acceptance/gui_cell_fit/acceptance.json')])
h('10.3 模型精度和训练',2)
table('10-2','网络实验结果及统计属性',['模型','结果','证据级别'],[
 ('LeNet-5/MNIST','完整10,000测试：浮点98.69%，器件训练前98.52%，训练后99.02%','完整测试集，单种子'),
 ('BNN/MNIST','3 epoch；完整10,000测试准确率98.21%','正式训练/完整测试集'),
 ('ResNet-20/CIFAR-10','100 epoch；完整10,000浮点测试89.98%；8样本器件差分75%','正式浮点+小样本差分'),
 ('TinyTransformer/CIFAR-10','4样本三模式均100%；1 batch头部训练通过','机制 smoke'),
 ('GPT-2 Small','8 token：浮点/理想一致；器件从第6 token开始分歧','真实 CUDA 机制验证')])
p('GPT-2 保留证据中，提示词“Computing in memory is”生成 8 个 token：浮点输出为“a very important part of the computer\'s”，器件差分输出为“a very important part of computing. It”。差异表明器件 LUT 已改变 logits 和贪心路径；该单提示词结果不能解释为通用语言质量提升或下降。')
h('10.4 NeuroSim PPA',2)
table('10-3','LeNet-5 官方 NeuroSim 正式运行指标',['指标','数值','单位'],[
 ('芯片面积','4.7236','mm²'),('读延迟','22.1344','μs/image'),('动态能耗','230.426','nJ/image'),('泄漏功耗','80.1516','μW'),('能效','19.2858','TOPS/W'),('吞吐','45,178.6','FPS'),('计算密度','0.052254','TOPS/mm²')])
fig('10-1',PLATFORM/'outputs/acceptance/gui_ppa/neurosim_ppa.png','GUI 保留的 NeuroSim PPA 可视化',14)
p('正式命令行运行和 GUI 验收运行使用的活动 trace 不同：前者动态能耗为230.426 nJ，GUI 保留的确定性零输入 trace 为196.474 nJ，因此不能把两者混为同一次测量。ResNet-20 另有22层 trace 和结构化 PPA 结果。')
h('10.5 自动化测试',2)
p('本次报告编制前，平台全量测试结果为83 passed、3 skipped；跳过项受显式资源或耗时条件保护。报告生成后另行执行 DOCX 包完整性、OMML、目录域、页码域、图片、路径和禁用占位词检查。现有证据支持 Windows 11、CPython 3.11.4 开发环境，尚不能外推到所有 Windows 10/11 机器。')

# Chapter 11
page(); h('第11章 局限性与后续改进',1)
h('11.1 模型与数据覆盖',2)
p('当前共有三种常规 DNN 图像结构（LeNet-5、ResNet-20、TinyTransformer）、一种 BNN 和一种 LLM，只覆盖 MNIST、CIFAR-10 与提示词文本。规格书提出的四种 DNN、四种主流数据集和 ImageNet 尚未全部落实。后续新增模型应同时提供合法 checkpoint、数据加载、尺度校准、差分边界、完整测试和 NeuroSim 映射证据。')
h('11.2 器件与电路非理想性',2)
p('当前 LUT 是 X、W 两维确定性均值模型，λ 只做理想—实测插值。它没有显式条件化温度、电源、工艺角、时间漂移、阵列位置，也没有统计描述器件间/周期间相关噪声。ADC/DAC 量化、IR drop、写入噪声、读出噪声和 sense amplifier 偏差可在未来作为独立、可组合且有标定来源的模型加入。')
h('11.3 NeuroSim 边界',2)
p('官方 NeuroSim 根据网络矩阵、位宽、活动和硬件配置估算 PPA；当前适配器不把测量电流 LUT 转换成 NeuroSim 器件参数。未来若要实现测量器件的电路级闭环，需要明确地把 I–V、导通态范围、变异和外围电路条件标定到 NeuroSim 支持的器件/电路参数，并用 SPICE 或芯片测量交叉验证。')
h('11.4 性能和训练',2)
p('精确查表计算强调可解释性而非 GPU kernel 峰值性能。大型 Transformer 全层展开会产生巨大中间量；当前 GPT-2 仅映射末 block 两个输出投影。可行改进包括自定义 CUDA/Triton 分块算子、稀疏/低秩适配、逐层流式执行，以及 LoRA 式误差感知微调。GPT-2 训练还需要确定具有许可的语料、token 切分、验证集、上下文长度和困惑度基线。')
h('11.5 产品化验收',2)
p('免安装版、安装/卸载程序、Windows 10 洁净机、CPU-only 8 GB 环境、两小时稳定运行和完整批量时延目标尚需独立验收。测试报告应为每项保留机器配置、命令、开始/结束时间、峰值内存和原始日志，避免仅凭开发机结果宣称通用兼容。')

# Chapter 12
page(); h('第12章 总结',1)
p('本平台建立了从测量数据到算法精度和硬件 PPA 的可追溯工程链路。器件侧通过基线消除、最小二乘增益、经验/平滑 LUT 和完整元数据，把仿真或测量 CSV 转为可复用误差模型；算法侧使用统一的 DifferentialLinear/DifferentialConv2d，以精确 LUT 前向和 STE 反向支持推理与误差感知训练；硬件侧通过四条非负差分 trace 调用固定版本的官方 NeuroSim，输出结构化 PPA。')
p('GUI 将误差拟合、模型组合、单样本、批量、训练和 PPA 组织为有依赖的流程。模型页的唯一勾选入口和工作流失效机制保证后续页面使用一致资源；图像与 GPT 文本任务共享误差选择和三模式比较，但使用与任务相符的输入、输出和指标。当前结果已经证明器件误差能改变分类和文本生成路径，也完成 LeNet-5/ResNet-20 的 NeuroSim 映射。')
p('工程价值在于把“器件数值误差影响”和“体系结构 PPA”放入同一可复核平台，同时保持两者边界清楚。后续工作的重点是扩大模型与数据集覆盖、增加统计器件模型、实现 GPT-2 可控微调，并完成安装包、Windows 10 与长期稳定性等产品化验收。')

# References
page(); h('参考文献',1)
refs=[
 '[1] NeuroSim. Official open-source repository. https://github.com/neurosim/NeuroSim',
 '[2] X. Peng et al. DNN+NeuroSim V2.0: An End-to-End Benchmarking Framework for Compute-in-Memory Accelerators for On-Chip Training. IEEE TCAD, 2021. DOI: 10.1109/TCAD.2020.3043731.',
 '[3] P.-Y. Chen et al. NeuroSim Simulator for Compute-in-Memory Hardware Accelerator: Validation and Benchmark. Frontiers in Artificial Intelligence, 2021.',
 '[4] PyTorch Documentation. Quantization and quantization-aware training. https://docs.pytorch.org/docs/stable/quantization.html',
 '[5] A. Radford et al. Language Models are Unsupervised Multitask Learners. OpenAI, 2019.',
 '[6] A. Karpathy. nanoGPT. https://github.com/karpathy/nanoGPT',
 '[7] SHI-Labs. Compact Transformers / CCT. https://github.com/SHI-Labs/Compact-Transformers',
 '[8] Y. LeCun et al. Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE, 1998.',
 '[9] K. He et al. Deep Residual Learning for Image Recognition. CVPR, 2016.',
 '[10] 本项目《DRAM存算仿真平台外包开发规格书》，2026。']
for r in refs: p(r)

# Appendices
page(); h('附录A 核心文件索引',1)
table('A-1','报告所引用的核心文件',['类别','相对路径'],[
 ('拟合','1_data_fitting/data_fitting/loader.py；fitting.py；service.py'),('通用误差层','5_NeuroCIM_opt/neurocim_opt/error_model.py；layers.py'),('推理','application.py；inference_runtime.py；batch_inference.py；gpt2_service.py'),('训练','training_service.py'),('NeuroSim','neurosim_adapter.py；vendor/neurosim；UPSTREAM_STATUS.md'),('工作流/GUI','workflow.py；model_registry.py；gui.py'),('模型','3_models/LeNet_5；BNN；Resnet20；TinyTransformer；GPT2'),('证据','5_NeuroCIM_opt/outputs；3_models/*/output')])
h('附录B 复现与验收命令',1)
b.add_code_block('''cd E:\\Prj_Python\\DRAM_CIM_Platform\\5_NeuroCIM_opt
python -m pytest -q
python docs\\report_assets\\generate_technical_report.py
python docs\\report_assets\\report_builder.py --validate "docs\\基于NeuroSim的DRAM存算仿真平台开发技术报告.docx"''',language='PowerShell')
h('附录C 公式变量说明',1)
table('C-1','主要符号及含义',['符号','含义','典型单位'],[
 ('X、W','非负输入码与权重码','code'),('x、w','有符号浮点激活与权重','模型数值'),('H、Hλ','完整及插值后的乘法 LUT','code²'),('sx、sw','输入尺度、逐输出通道权重尺度','模型数值/code'),('Iint','消除单变量基线后的交互电流','A'),('λ','器件误差强度','无量纲'),('A、T、E、P','面积、延迟、能耗、功耗','μm²/ns/pJ/μW'),('LCE、LNLL','分类交叉熵和语言模型负对数似然','无量纲')])
p('以下块级公式给出连续变量的数域，并作为 Word 块级公式结构验收样本。')
b.add_display_equation(['x,w,s_x,s_w ∈ ℝ'])

path=b.save(OUT)
counts=validate_docx(path,minimum_equations=25)
print(path)
print(json.dumps(counts,ensure_ascii=False,indent=2))

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','Arial Unicode MS']
plt.rcParams['axes.unicode_minus']=False
OUT=Path(__file__).resolve().parent
OUT.mkdir(parents=True,exist_ok=True)

def diagram(name,title,columns,arrows,footer=None):
    fig,ax=plt.subplots(figsize=(12,6.3),dpi=180)
    ax.set_xlim(0,12); ax.set_ylim(0,7); ax.axis('off')
    ax.text(6,6.65,title,ha='center',va='center',fontsize=17,fontweight='bold',color='#17365D')
    boxes={}
    for key,x,y,w,h,text,color in columns:
        p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.10',
                         facecolor=color,edgecolor='#365F91',linewidth=1.3)
        ax.add_patch(p); ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10,wrap=True)
        boxes[key]=(x,y,w,h)
    for a,b,label in arrows:
        x,y,w,h=boxes[a]; x2,y2,w2,h2=boxes[b]
        if abs((x+w/2)-(x2+w2/2)) > abs((y+h/2)-(y2+h2/2)):
            start=(x+w if x<x2 else x, y+h/2); end=(x2 if x<x2 else x2+w2,y2+h2/2)
        else:
            start=(x+w/2,y+h if y<y2 else y); end=(x2+w2/2,y2 if y<y2 else y2+h2)
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=1.5,color='#44546A'))
        if label: ax.text((start[0]+end[0])/2,(start[1]+end[1])/2+0.16,label,ha='center',fontsize=8,color='#555')
    if footer: ax.text(6,0.20,footer,ha='center',va='center',fontsize=9,color='#7F6000')
    fig.tight_layout(); fig.savefig(OUT/name,bbox_inches='tight'); plt.close(fig)

blue='#D9EAF7'; green='#E2F0D9'; orange='#FCE4D6'; purple='#E4DFEC'; gray='#EDEDED'; yellow='#FFF2CC'
diagram('fig_2_1_system_architecture.png','DRAM 存算仿真平台总体架构',[
 ('data',.4,4.3,2.1,1.2,'器件与阵列数据层\nCSV / LUT / MAC',blue),('fit',3.0,4.3,2.2,1.2,'误差建模层\n拟合、注册、溯源',green),('runtime',5.7,4.3,2.2,1.2,'数值计算层\n差分推理与训练',orange),('ppa',8.4,4.3,2.2,1.2,'NeuroSim 层\n映射与 PPA',purple),
 ('models',3.0,2.1,2.2,1.1,'模型资源层\n5 类模型组合',blue),('service',5.7,2.1,2.2,1.1,'服务层\nApplication / Training',green),('gui',8.4,2.1,2.2,1.1,'交互层\nPySide6 GUI / CLI',yellow)],
 [('data','fit','校验与拟合'),('fit','runtime','256×256 LUT'),('runtime','ppa','权重与活动 trace'),('models','service','模型/数据集'),('service','gui','结构化结果'),('models','runtime','Conv/Linear 适配')],
 '数值误差链与 PPA 链共享网络和活动信息，但器件 LUT 不直接输入 NeuroSim。')
diagram('fig_2_2_business_flow.png','端到端业务流程',[
 ('import',.3,4.2,1.7,1.0,'导入 CSV',blue),('map',2.25,4.2,1.7,1.0,'列映射与校验',green),('fit',4.2,4.2,1.7,1.0,'生成多种拟合',orange),('select',6.15,4.2,1.7,1.0,'选择误差模型',yellow),('infer',8.1,4.2,1.7,1.0,'单/批量推理',purple),('train',10.05,4.2,1.7,1.0,'误差感知训练',orange),
 ('model',3.0,2.0,2.4,1.0,'勾选模型+数据集',blue),('evidence',6.0,2.0,2.4,1.0,'JSON / 图表 / checkpoint',gray),('ppa',9.0,2.0,2.4,1.0,'NeuroSim PPA',purple)],
 [('import','map',''),('map','fit',''),('fit','select',''),('select','infer',''),('infer','train',''),('model','infer','统一继承'),('infer','evidence','结果'),('infer','ppa','trace')])
diagram('fig_3_1_neurosim_dual_chain.png','器件数值误差与 NeuroSim PPA 双链路',[
 ('csv',.5,4.5,2.1,1.0,'测量 CSV',blue),('lut',3.1,4.5,2.2,1.0,'交互电流 → LUT',green),('nn',5.8,4.5,2.2,1.0,'差分推理 / STE 训练',orange),('acc',8.5,4.5,2.7,1.0,'精度、损失、Token 一致率',yellow),
 ('model',.5,2.1,2.1,1.0,'网络 + 样本',blue),('trace',3.1,2.1,2.2,1.0,'X± / W± trace',green),('ns',5.8,2.1,2.2,1.0,'官方 NeuroSim C++',purple),('ppa',8.5,2.1,2.7,1.0,'面积、延迟、能耗、TOPS/W',yellow)],
 [('csv','lut',''),('lut','nn','器件乘法'),('nn','acc',''),('model','trace','量化与展开'),('trace','ns','四条非负路径'),('ns','ppa','')],
 '两条链路的输出不可互相替代：LUT 表征数值误差，NeuroSim 估算硬件 PPA。')
diagram('fig_4_1_fitting_pipeline.png','器件误差拟合与模型注册流程',[
 ('read',.5,4.3,1.9,1.1,'编码/分隔符识别',blue),('columns',2.75,4.3,1.9,1.1,'表头下拉映射',green),('grid',5.0,4.3,1.9,1.1,'位宽/网格校验',orange),('base',7.25,4.3,1.9,1.1,'四项基线消除',purple),('lut',9.5,4.3,1.9,1.1,'8-bit LUT',yellow),
 ('fits',2.3,2.0,2.5,1.1,'线性/二次/全曲面拟合',gray),('variants',5.2,2.0,2.5,1.1,'经验/一阶/二阶 LUT',green),('registry',8.1,2.0,2.5,1.1,'注册表与 SHA-256',blue)],
 [('read','columns',''),('columns','grid',''),('grid','base',''),('base','lut',''),('grid','fits','诊断'),('lut','variants','平滑'),('variants','registry','可部署')])
diagram('fig_5_1_differential_compute.png','正负差分 LUT 乘累加',[
 ('x',.6,4.3,1.8,1.0,'输入 x\n符号 + 幅值码',blue),('w',.6,2.4,1.8,1.0,'权重 w\n符号 + 幅值码',blue),('q',3.0,3.35,2.0,1.1,'|·| / scale\nround + clip',green),('lookup',5.7,3.35,2.0,1.1,'L[q_w,q_x]\n器件查表',orange),('sign',8.4,3.35,2.0,1.1,'sign(x)sign(w)\n有符号累加',purple),('scale',10.6,3.35,1.1,1.1,'反量化\n+bias',yellow)],
 [('x','q',''),('w','q',''),('q','lookup','uint8'),('lookup','sign','int64'),('sign','scale','s_x s_w')],
 '等价物理解释：X⁺W⁺ − X⁺W⁻ − X⁻W⁺ + X⁻W⁻。')
diagram('fig_6_1_model_mapping.png','模型算子到平台计算域的映射',[
 ('net',.5,4.4,2.2,1.0,'PyTorch 网络',blue),('mac',3.2,4.4,2.2,1.0,'Conv2d / Linear',green),('diff',5.9,4.4,2.2,1.0,'DifferentialConv2d\nDifferentialLinear',orange),('lut',8.6,4.4,2.2,1.0,'理想/器件 LUT',purple),
 ('digital',3.2,2.0,2.2,1.0,'BN / 激活 / Pool',gray),('cal',5.9,2.0,2.2,1.0,'输入尺度校准',yellow),('trace',8.6,2.0,2.2,1.0,'NeuroSim trace',blue)],
 [('net','mac','识别层'),('mac','diff','替换'),('diff','lut','查表'),('net','digital','保留数字域'),('cal','diff','s_x'),('mac','trace','权重/活动')])
diagram('fig_8_1_eat_pipeline.png','误差感知训练流程',[
 ('base',.4,4.3,2.0,1.0,'加载预训练权重',blue),('convert',2.85,4.3,2.0,1.0,'器件 LUT 差分层',green),('forward',5.3,4.3,2.0,1.0,'精确查表前向',orange),('loss',7.75,4.3,1.7,1.0,'交叉熵',yellow),('ste',9.9,4.3,1.7,1.0,'STE 反向',purple),
 ('eval',2.85,2.0,2.0,1.0,'训练前三模式评估',gray),('update',5.3,2.0,2.0,1.0,'Adam 参数更新',green),('save',7.75,2.0,2.0,1.0,'验证/最佳权重',blue),('after',10.1,2.0,1.6,1.0,'训练后三模式',gray)],
 [('base','convert',''),('convert','forward',''),('forward','loss',''),('loss','ste',''),('ste','update','梯度'),('update','forward','迭代'),('base','eval','基线'),('update','save',''),('save','after','')])
diagram('fig_9_1_gui_workflow.png','GUI 统一选择与状态继承',[
 ('fit',.5,4.3,2.0,1.0,'误差拟合页\n多种 LUT',green),('catalog',3.0,4.3,2.0,1.0,'模型与数据集页\n唯一勾选入口',blue),('single',5.5,4.3,1.8,1.0,'单样本推理',orange),('batch',7.75,4.3,1.8,1.0,'批量推理',orange),('train',10.0,4.3,1.5,1.0,'训练',purple),
 ('ctx',3.6,2.0,3.0,1.0,'workflow_context.json',gray),('registry',.5,2.0,2.4,1.0,'error_model_registry.json',gray),('dynamic',7.3,2.0,3.0,1.0,'图像 / GPT 动态界面',yellow)],
 [('fit','single','误差模型'),('catalog','single','模型组合'),('catalog','batch','模型组合'),('single','train','成功配置'),('batch','train','成功配置'),('fit','registry','注册'),('catalog','ctx','持久化'),('single','ctx','结果'),('single','dynamic','任务切换'),('batch','dynamic','任务切换')],
 '上游模型或误差模型变化会清除不再匹配的推理与训练上下文。')
print('generated',len(list(OUT.glob('fig_*.png'))),'figures')

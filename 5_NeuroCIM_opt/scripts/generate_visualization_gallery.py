"""Build a self-contained gallery of proposed NeuroCIM visualizations."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OUT = ROOT / "outputs" / "visualization_examples"
TRAINING = ROOT / "outputs" / "uos1_workflow" / "training_result.json"
MODEL = ROOT / "outputs" / "demo" / "uos1_gui" / "fit_result" / "error_model"
PPA = ROOT / "outputs" / "neurosim" / "01_NMOS" / "neurosim_result.json"

COLORS = {
    "blue": "#3b82f6", "cyan": "#06b6d4", "violet": "#8b5cf6",
    "green": "#10b981", "amber": "#f59e0b", "red": "#ef4444",
    "ink": "#dbeafe", "muted": "#91a4bd", "grid": "#27364d",
}


def style(fig: go.Figure, title: str, height: int = 430) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.02, "font": {"size": 19, "color": "#f8fafc"}},
        height=height, margin=dict(l=55, r=30, t=65, b=50),
        paper_bgcolor="#101827", plot_bgcolor="#101827", font_color=COLORS["ink"],
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        hoverlabel=dict(bgcolor="#172033", font_color="#fff"),
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])
    return fig


def card(section_id: str, number: str, title: str, badge: str, description: str, figure: str, note: str) -> str:
    badge_class = "real" if "真实" in badge else "proto"
    return f"""
    <section id="{section_id}" class="panel">
      <div class="panel-head"><div><span class="number">{number}</span><h2>{html.escape(title)}</h2></div>
      <span class="badge {badge_class}">{html.escape(badge)}</span></div>
      <p class="desc">{html.escape(description)}</p>
      {figure}
      <p class="note">{html.escape(note)}</p>
    </section>"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    meta = json.loads((MODEL / "error_model.json").read_text(encoding="utf-8"))
    lut = np.load(MODEL / "multiplication_lut.npy").astype(float)
    codes = np.arange(256, dtype=float)
    ideal = np.outer(codes, codes)
    residual = lut - ideal
    before, after = training["evaluations"]["before"], training["evaluations"]["after"]
    fit = meta["metadata"]["fitting_summary"]["interaction_metrics"]

    figures: list[go.Figure] = []
    sections: list[str] = []

    # 1. Overview is an HTML-native dashboard.
    overview = f"""
    <div class="metric-grid">
      <div class="metric"><b>{fit['r2']:.4f}</b><span>拟合 R²</span></div>
      <div class="metric"><b>{fit['nl_percent_fs']:.3f}%</b><span>满量程非线性</span></div>
      <div class="metric"><b>{before['device_differential']['accuracy']:.2%}</b><span>训练前器件精度</span></div>
      <div class="metric"><b>{after['device_differential']['accuracy']:.2%}</b><span>训练后器件精度</span></div>
      <div class="metric accent"><b>+{(after['device_differential']['accuracy']-before['device_differential']['accuracy'])*100:.2f} pp</b><span>误差感知训练收益</span></div>
    </div>
    <div class="flow">
      <div>UOS1<br><small>65,536 点</small></div><i>→</i><div>误差拟合<br><small>256×256 LUT</small></div><i>→</i>
      <div>差分推理<br><small>10,000 样本</small></div><i>→</i><div>EAT<br><small>55,000 样本</small></div><i>→</i>
      <div>训练后评估<br><small>99.00%</small></div>
    </div>"""
    sections.append(card("overview", "01", "全流程结果总览", "UOS1 真实数据",
                         "首页用指标卡和流程状态展示一次实验从器件数据到训练后评估的完整链路。",
                         overview, "用户进入软件后无需切换页面，就能判断实验是否完成以及精度是否得到恢复。"))

    # 2. LUT and error heatmaps.
    fig = make_subplots(rows=1, cols=3, subplot_titles=("器件乘法 LUT H(W,X)", "残差 E=H-XW", "权重切片 W=64/128/255"))
    fig.add_trace(go.Heatmap(z=lut, colorscale="Viridis", colorbar=dict(title="输出", x=.28)), 1, 1)
    lim = float(np.percentile(np.abs(residual), 99.5))
    fig.add_trace(go.Heatmap(z=residual, colorscale="RdBu", zmid=0, zmin=-lim, zmax=lim,
                             colorbar=dict(title="误差", x=.63)), 1, 2)
    for w, color in zip((64, 128, 255), (COLORS["cyan"], COLORS["violet"], COLORS["amber"])):
        fig.add_trace(go.Scatter(x=codes, y=lut[w], name=f"W={w}", line=dict(color=color)), 1, 3)
        fig.add_trace(go.Scatter(x=codes, y=ideal[w], name=f"W={w} 理想", line=dict(color=color, dash="dot"), showlegend=False), 1, 3)
    fig.update_xaxes(title_text="输入编码 X"); fig.update_yaxes(title_text="权重编码 W", row=1, col=1)
    fig.update_yaxes(title_text="权重编码 W", row=1, col=2); fig.update_yaxes(title_text="乘法输出", row=1, col=3)
    figures.append(style(fig, "UOS1 器件乘法特性与误差区域", 480))

    # 3. Four-branch decomposition with actual LUT values.
    xp, xn, wp, wn = 180, 42, 130, 25
    branch = np.array([lut[wp, xp], -lut[wn, xp], -lut[wp, xn], lut[wn, xn]])
    names = ["X⁺W⁺", "−X⁺W⁻", "−X⁻W⁺", "+X⁻W⁻"]
    fig = go.Figure(go.Waterfall(x=names + ["差分输出"], measure=["relative"] * 4 + ["total"],
                                 y=list(branch) + [0], connector={"line": {"color": "#64748b"}},
                                 increasing={"marker": {"color": COLORS["green"]}},
                                 decreasing={"marker": {"color": COLORS["red"]}},
                                 totals={"marker": {"color": COLORS["blue"]}}, textposition="outside"))
    figures.append(style(fig, "一次有符号 MAC 的四路差分分解", 420))

    # 4. Accuracy and confusion comparison.
    modes = ["浮点", "理想差分", "UOS1 器件差分"]
    keys = ["float", "ideal_differential", "device_differential"]
    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "bar"}, {"type": "heatmap"}, {"type": "heatmap"}]],
                        subplot_titles=("训练前后准确率", "训练前器件混淆矩阵", "训练后器件混淆矩阵"))
    fig.add_trace(go.Bar(x=modes, y=[before[k]["accuracy"] * 100 for k in keys], name="训练前", marker_color=COLORS["muted"], texttemplate="%{y:.2f}%", textposition="outside"), 1, 1)
    fig.add_trace(go.Bar(x=modes, y=[after[k]["accuracy"] * 100 for k in keys], name="训练后", marker_color=COLORS["green"], texttemplate="%{y:.2f}%", textposition="outside"), 1, 1)
    for col, data in ((2, before), (3, after)):
        fig.add_trace(go.Heatmap(z=data["device_differential"]["confusion_matrix"], colorscale="Blues", showscale=col == 3), 1, col)
    fig.update_yaxes(range=[97.5, 99.3], title_text="准确率 (%)", row=1, col=1)
    figures.append(style(fig, "推理结果对比与错误类别定位", 460))

    # 5. Layer propagation prototype.
    layer_names = ["Conv1", "Conv2", "FC1", "FC2", "FC3"]
    rmse = [0.012, 0.019, 0.031, 0.046, 0.061]
    similarity = [0.9994, 0.9988, 0.9971, 0.9948, 0.9912]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=layer_names, y=rmse, name="归一化 RMSE", marker_color=COLORS["violet"]), secondary_y=False)
    fig.add_trace(go.Scatter(x=layer_names, y=similarity, name="余弦相似度", mode="lines+markers", line=dict(color=COLORS["amber"], width=3)), secondary_y=True)
    fig.update_yaxes(title_text="输出 RMSE", secondary_y=False); fig.update_yaxes(title_text="余弦相似度", range=[.988, 1], secondary_y=True)
    figures.append(style(fig, "逐层误差传播：定位最敏感的计算层", 420))

    # 6. Training visualization.
    h = training["history"][0]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("训练与验证", "器件精度恢复"))
    fig.add_trace(go.Bar(x=["训练准确率", "验证准确率"], y=[h["train_accuracy"]*100, h["validation_accuracy"]*100],
                         marker_color=[COLORS["blue"], COLORS["green"]], texttemplate="%{y:.2f}%", textposition="outside", showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=["训练前", "训练后"], y=[before["device_differential"]["accuracy"]*100, after["device_differential"]["accuracy"]*100],
                             mode="lines+markers+text", text=["98.62%", "99.00%"], textposition="top center",
                             line=dict(color=COLORS["green"], width=5), marker=dict(size=13), showlegend=False), 1, 2)
    fig.update_yaxes(range=[98, 100.1], title_text="准确率 (%)")
    figures.append(style(fig, "误差感知训练过程与精度恢复", 410))

    # 7. Actual weight deltas.
    base_ckpt = PROJECT / "3_models" / "LeNet_5" / "chekpoint" / "lenet5_float.pt"
    trained_ckpt = ROOT / "outputs" / "uos1_workflow" / "eat_checkpoint.pt"
    base_payload = torch.load(base_ckpt, map_location="cpu", weights_only=True)
    train_payload = torch.load(trained_ckpt, map_location="cpu", weights_only=True)
    base_state = base_payload.get("best_model_state_dict", base_payload.get("model_state_dict", base_payload))
    train_state = train_payload["best_model_state_dict"]
    labels, norms, maxima = [], [], []
    for name, value in base_state.items():
        if name in train_state and torch.is_floating_point(value) and name.endswith("weight"):
            delta = train_state[name].cpu() - value.cpu()
            labels.append(name.replace(".weight", "")); norms.append(float(delta.norm())); maxima.append(float(delta.abs().max()))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=labels, y=norms, name="L2 更新量", marker_color=COLORS["blue"]), secondary_y=False)
    fig.add_trace(go.Scatter(x=labels, y=maxima, name="最大绝对更新", mode="lines+markers", line=dict(color=COLORS["amber"], width=3)), secondary_y=True)
    fig.update_yaxes(title_text="L2 更新量", secondary_y=False); fig.update_yaxes(title_text="最大绝对更新", secondary_y=True)
    figures.append(style(fig, "训练前后各层权重变化", 420))

    # 8. Lambda scan prototype with real endpoints.
    lambdas = np.linspace(0, 1, 11)
    a0 = before["ideal_differential"]["accuracy"] * 100
    a1 = before["device_differential"]["accuracy"] * 100
    t0 = after["ideal_differential"]["accuracy"] * 100
    t1 = after["device_differential"]["accuracy"] * 100
    pre_curve = a0 + (a1-a0) * lambdas**1.4
    post_curve = t0 + (t1-t0) * lambdas**1.25
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lambdas, y=pre_curve, mode="lines+markers", name="训练前", line=dict(color=COLORS["red"], width=3)))
    fig.add_trace(go.Scatter(x=lambdas, y=post_curve, mode="lines+markers", name="训练后", line=dict(color=COLORS["green"], width=3)))
    fig.add_trace(go.Scatter(x=[0,1,0,1], y=[a0,a1,t0,t1], mode="markers", name="已实测端点", marker=dict(color=COLORS["amber"], size=13, symbol="diamond")))
    fig.update_xaxes(title="误差强度 λ"); fig.update_yaxes(title="准确率 (%)")
    figures.append(style(fig, "误差强度扫描与鲁棒性曲线", 420))

    # 9. Multi-device comparison: fit quality from available bundles.
    devices, r2s, nls = [], [], []
    for name in ("01_NMOS", "02_CFET", "03_UOS_1", "04_UOS_2"):
        path = ROOT / "outputs" / "error_models" / name / "error_model.json"
        if path.is_file():
            d = json.loads(path.read_text(encoding="utf-8"))["metadata"]["fitting_summary"]["interaction_metrics"]
            devices.append(name); r2s.append(d["r2"]); nls.append(d["nl_percent_fs"])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=devices, y=r2s, name="拟合 R²", marker_color=COLORS["cyan"]), secondary_y=False)
    fig.add_trace(go.Scatter(x=devices, y=nls, name="非线性 %FS", mode="lines+markers", line=dict(color=COLORS["amber"], width=3)), secondary_y=True)
    fig.update_yaxes(title_text="拟合 R²", range=[0,1.05], secondary_y=False); fig.update_yaxes(title_text="非线性 (%FS)", secondary_y=True)
    figures.append(style(fig, "多器件误差模型横向对比", 420))

    # 10. Real official NeuroSim reference PPA.
    ppa = json.loads(PPA.read_text(encoding="utf-8"))
    layers = ["Conv1", "Conv2", "FC1", "FC2", "FC3"]
    latency = np.array(ppa["layer_metrics"]["read_latency_ns"]) / 1000
    energy = np.array(ppa["layer_metrics"]["read_dynamic_energy_pj"]) / 1000
    fig = make_subplots(rows=1, cols=2, subplot_titles=("逐层延迟", "逐层动态能耗"))
    fig.add_trace(go.Bar(x=layers, y=latency, marker_color=COLORS["blue"], name="延迟", showlegend=False), 1, 1)
    fig.add_trace(go.Bar(x=layers, y=energy, marker_color=COLORS["violet"], name="能耗", showlegend=False), 1, 2)
    fig.update_yaxes(title_text="μs", row=1, col=1); fig.update_yaxes(title_text="nJ", row=1, col=2)
    figures.append(style(fig, "官方 NeuroSim 分层 PPA 参考", 420))

    # 11. Experiment history is an HTML-native table.
    history = f"""
    <div class="table-wrap"><table><thead><tr><th>实验</th><th>器件</th><th>网络</th><th>λ</th><th>训练前</th><th>训练后</th><th>状态</th></tr></thead>
    <tbody><tr><td>UOS1 正式流程</td><td>03_UOS_1</td><td>LeNet-5 / MNIST</td><td>1.0</td>
    <td>{before['device_differential']['accuracy']:.2%}</td><td>{after['device_differential']['accuracy']:.2%}</td><td><span class="ok">已完成</span></td></tr>
    <tr class="dim"><td>参数扫描示例</td><td>03_UOS_1</td><td>LeNet-5 / MNIST</td><td>0 → 1</td><td>—</td><td>—</td><td>待运行</td></tr>
    <tr class="dim"><td>PPA 联合评估示例</td><td>03_UOS_1</td><td>LeNet-5 / MNIST</td><td>1.0</td><td>—</td><td>—</td><td>待运行</td></tr></tbody></table></div>
    <div class="report-actions"><button>导出实验 JSON</button><button>生成 PDF 报告</button><button>导出全部图表</button></div>"""

    # Convert figures in their section order.
    divs = []
    for index, fig in enumerate(figures):
        divs.append(fig.to_html(full_html=False, include_plotlyjs="inline" if index == 0 else False,
                                config={"displaylogo": False, "responsive": True, "toImageButtonOptions": {"format": "png", "scale": 2}}))
    titles = [
        ("device", "02", "器件误差可视化", "UOS1 真实数据", "通过 LUT、残差热力图和切片曲线定位误差集中的编码区域。",
         "热力图支持缩放和悬停查看具体编码值；切片用于比较器件曲线与理想乘法。"),
        ("differential", "03", "正负权重差分计算", "UOS1 真实 LUT 示例", "把一次有符号计算拆成四个非负器件分支，并显示每条路径对最终结果的贡献。",
         "瀑布图适合排查某一路误差被相加还是相减，以及差分抵消是否有效。"),
        ("inference", "04", "推理结果对比", "UOS1 真实数据", "同时比较浮点、理想差分和器件差分，并用混淆矩阵定位受影响的数字类别。",
         "当前图表使用完整 10,000 个 MNIST 测试样本。"),
        ("layers", "05", "逐层误差传播", "界面原型", "用层输出 RMSE 和余弦相似度确定误差从哪一层开始明显放大。",
         "该布局为实现范例；正式版本需要增加逐层激活采集后才能生成真实数值。"),
        ("training", "06", "误差感知训练", "UOS1 真实数据", "显示训练/验证质量，以及误差感知训练为器件差分路径恢复了多少准确率。",
         "后续多 epoch 运行时可直接扩展为实时损失和准确率折线。"),
        ("weights", "07", "权重变化可视化", "UOS1 真实数据", "比较各层权重的 L2 更新量与最大单点更新，解释网络在哪里适应器件误差。",
         "数据来自原始 LeNet-5 checkpoint 与 UOS1 EAT checkpoint 的逐层差值。"),
        ("lambda", "08", "误差强度扫描", "端点真实 / 中间原型", "观察 λ 从 0 到 1 时的精度退化，以及训练后鲁棒性曲线是否更平缓。",
         "菱形端点为真实评估；中间曲线仅示范界面，正式实现应自动执行每个 λ。"),
        ("devices", "09", "多器件横向对比", "真实拟合指标", "在统一坐标下比较 NMOS、CFET、UOS1 与 UOS2 的拟合质量和非线性。",
         "可继续加入统一网络上的准确率和训练恢复量，形成器件选择依据。"),
        ("ppa", "10", "NeuroSim 硬件结果", "官方 NeuroSim 参考数据", "显示每层延迟和动态能耗，并为子阵列扫描与精度/PPA Pareto 图提供入口。",
         "当前图使用已完成的 01_NMOS LeNet-5 官方 NeuroSim 结果；UOS1 PPA 尚未执行。"),
    ]
    for args, div in zip(titles, divs):
        sections.append(card(*args[:5], div, args[5]))
    sections.append(card("history", "11", "实验管理与报告", "UOS1 真实记录",
                         "集中查看实验状态、输入配置、训练前后结果和导出入口。", history,
                         "正式实现时每行可打开对应实验总览，并支持任意两次运行对比。"))

    navigation = "".join(f'<a href="#{sid}">{num} {title}</a>' for sid, num, title, *_ in [
        ("overview","01","实验总览"), *[(x[0], x[1], x[2]) for x in titles], ("history","11","实验管理")])
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>NeuroCIM 可视化样例馆</title><style>
    *{{box-sizing:border-box}} html{{scroll-behavior:smooth}} body{{margin:0;background:#08111f;color:#dbeafe;font-family:'Microsoft YaHei','Segoe UI',sans-serif}}
    aside{{position:fixed;left:0;top:0;bottom:0;width:230px;background:#0d1728;border-right:1px solid #21304a;padding:26px 18px;overflow:auto}}
    aside h1{{font-size:19px;margin:0 0 8px;color:#fff}} aside p{{font-size:12px;color:#91a4bd;line-height:1.6}} aside a{{display:block;color:#a9bad0;text-decoration:none;padding:8px 10px;border-radius:8px;font-size:13px}} aside a:hover{{background:#17243a;color:#fff}}
    main{{margin-left:230px;padding:34px;max-width:1600px}} .hero{{padding:20px 0 26px}} .hero h1{{font-size:34px;margin:0 0 10px;color:#fff}} .hero p{{color:#91a4bd;max-width:850px;line-height:1.7}}
    .legend span,.badge{{display:inline-block;padding:5px 10px;border-radius:999px;font-size:12px;font-weight:700}} .real{{background:#103c35;color:#69e6c2}} .proto{{background:#473318;color:#ffd284}}
    .panel{{background:#0d1728;border:1px solid #21304a;border-radius:16px;margin-bottom:26px;padding:22px;box-shadow:0 16px 40px #0003}} .panel-head{{display:flex;align-items:center;justify-content:space-between;gap:16px}}
    .panel-head>div{{display:flex;align-items:center;gap:12px}} .panel h2{{font-size:21px;margin:0;color:#fff}} .number{{color:#60a5fa;font-weight:800;font-size:13px}} .desc,.note{{color:#9fb0c6;line-height:1.7;font-size:14px}} .note{{border-left:3px solid #3b82f6;padding-left:12px;margin-top:12px}}
    .metric-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:22px 0}} .metric{{background:#101d31;border:1px solid #263854;padding:18px;border-radius:12px}} .metric b{{font-size:25px;color:#fff;display:block}} .metric span{{font-size:12px;color:#91a4bd}} .metric.accent{{background:linear-gradient(135deg,#173a38,#10263a)}} .metric.accent b{{color:#5ee1b7}}
    .flow{{display:flex;align-items:center;justify-content:center;gap:14px;padding:25px;background:#101827;border-radius:12px;overflow:auto}} .flow div{{min-width:140px;text-align:center;background:#17243a;border:1px solid #304563;border-radius:10px;padding:14px;font-weight:700}} .flow small{{color:#91a4bd;font-weight:400}} .flow i{{font-size:24px;color:#4d6a91}}
    .table-wrap{{overflow:auto}} table{{width:100%;border-collapse:collapse;background:#101827;border-radius:10px;overflow:hidden}} th,td{{padding:14px;border-bottom:1px solid #27364d;text-align:left}} th{{color:#91a4bd;font-size:12px}} td{{font-size:13px}} tr.dim{{opacity:.55}} .ok{{color:#5ee1b7}}
    .report-actions{{display:flex;gap:10px;margin-top:16px}} button{{background:#1d4ed8;color:white;border:0;border-radius:8px;padding:10px 15px;font-weight:700}}
    @media(max-width:900px){{aside{{display:none}}main{{margin:0;padding:15px}}.metric-grid{{grid-template-columns:repeat(2,1fr)}}}}
    </style></head><body><aside><h1>NeuroCIM<br>可视化样例馆</h1><p>11 类界面范例<br>真实结果与原型明确区分</p>{navigation}</aside>
    <main><div class="hero"><h1>从器件误差到网络与硬件结果</h1><p>这份样例用于确定下一版 GUI 的信息结构和图表形式。绿色标签来自现有实验结果，黄色标签表示需要新增采集或扫描能力。</p><div class="legend"><span class="real">真实数据</span> <span class="proto">界面原型</span></div></div>{''.join(sections)}</main></body></html>"""
    target = OUT / "index.html"
    target.write_text(page, encoding="utf-8")
    print(target.resolve())


if __name__ == "__main__":
    main()

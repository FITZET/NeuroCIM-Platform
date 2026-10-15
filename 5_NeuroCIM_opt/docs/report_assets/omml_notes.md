# Word/OMML 报告生成说明

`report_builder.py` 基于 `python-docx` 和其底层 `lxml` 节点创建 DOCX。独立公式写入 `m:oMathPara`，内部表达式写入 `m:oMath`；公式因此是 Word 原生可编辑对象，不是图片、普通文本或未转换的 LaTeX。

## 快速验证

在本目录运行：

```powershell
python .\report_builder.py
```

命令会生成 `omml_minimal_test.docx`，随后检查 ZIP 包完整性、`word/document.xml` 中的 OMML 标签、目录域、页码域、自动更新域设置、表格和媒体文件，并再次用 `python-docx` 打开文档。

也可检查已有文档：

```powershell
python .\report_builder.py --validate .\目标报告.docx
```

安装了 Microsoft Word 和 `pywin32` 的 Windows 环境可追加 `--word`，以只读方式让 Word 实际打开文档并检查公式、目录、表格、图片和页数：

```powershell
python .\report_builder.py --validate .\目标报告.docx --word
```

## 常用接口

```python
from report_builder import (
    ReportBuilder,
    math_fraction,
    math_nary,
    math_radical,
    math_sub,
    math_sup,
    validate_docx,
)

report = ReportBuilder("基于 NeuroSim 的技术报告")
report.add_cover("DRAM 存内计算仿真平台")
report.add_toc()
report.add_heading("1 数学模型", 1)

formula = [
    "RMSE = ",
    math_radical(
        math_fraction(
            math_nary("∑", math_sup([math_sub("y", "i"), " − ŷᵢ"], "2"),
                      lower="i=1", upper="N"),
            "N",
        )
    ),
]
report.add_equation(formula, "1-1")
report.add_display_equation("H(X,W) = XW · g(X,W)")
report.add_code_block("y = x * w + error[x, w]", language="Python")
report.add_table(["项目", "值"], [("位宽", 8)], caption="配置参数")
report.add_image("figure.png", caption="系统架构", width_cm=15)
report.save("report.docx")
validate_docx("report.docx")
```

结构化构造器包括：

- `math_text`、`math_concat`：数学文本与组合表达式；
- `math_fraction`：分式；
- `math_sub`、`math_sup`、`math_subsup`：上下标；
- `math_radical`：平方根或带次数根式；
- `math_nary`：求和、连乘、积分等大型运算符；
- `math_delimiter`：可伸缩括号；
- `math_matrix`：矩阵；
- `omath`、`omath_para`：行内与独立公式包装。

`add_equation(parts, number)` 使用 Word 稳定支持的无边框三列表格排出“居中公式 + 右侧编号”，内部是原生 `m:oMath`。`add_display_equation(parts)` 写入包含 `m:oMathPara` 和 `m:oMath` 的独立公式段。两者都可在 Word 中双击编辑。

字符串参数会作为 OMML 数学运行写入。复杂公式应优先使用结构化构造器，避免把 LaTeX 源码直接交给 `add_equation`；本模块没有把 LaTeX 解析为 OMML 的隐式转换逻辑。

## Word 行为与验收边界

目录和页码使用 Word 域代码，并在 `word/settings.xml` 设置 `w:updateFields`。Microsoft Word 首次打开时通常会更新域；若目录仍显示占位文字，可全选文档后按 `F9` 更新。

自动验证确认 OOXML 结构和包关系有效，但不能代替 Word 的最终视觉检查。交付前仍应在 Microsoft Word 中检查分页、图表清晰度、公式换行、目录页码，并随机双击至少 10 个关键公式确认可编辑。

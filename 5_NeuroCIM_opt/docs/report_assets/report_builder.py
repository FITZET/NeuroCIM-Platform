"""Reusable Word report builder with native, editable OMML equations.

The module intentionally does not convert LaTeX.  Equations are created as
Office Math Markup Language nodes, so Word treats them as equations rather
than pictures or ordinary text.  Run this file directly to build and validate
``omml_minimal_test.docx`` next to the script.
"""

from __future__ import annotations

import argparse
import copy
import struct
import tempfile
import zipfile
import zlib
from pathlib import Path
from typing import Iterable, Sequence, Union

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from lxml import etree


MathPart = Union[str, OxmlElement]


def _set_attr(element: OxmlElement, name: str, value: str) -> None:
    element.set(qn(name), value)


def _clone(part: OxmlElement) -> OxmlElement:
    return copy.deepcopy(part)


def math_text(text: str, *, normal: bool = False) -> OxmlElement:
    """Return an OMML math run containing editable Unicode math text."""
    run = OxmlElement("m:r")
    if normal:
        run_pr = OxmlElement("m:rPr")
        nor = OxmlElement("m:nor")
        run_pr.append(nor)
        run.append(run_pr)
    node = OxmlElement("m:t")
    if text[:1].isspace() or text[-1:].isspace() or "  " in text:
        node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    node.text = text
    run.append(node)
    return run


def _parts(value: MathPart | Iterable[MathPart]) -> list[OxmlElement]:
    if isinstance(value, str):
        return [math_text(value)]
    # Registered WordprocessingML nodes are BaseOxmlElement subclasses, while
    # unregistered OMML tags are plain lxml _Element instances.
    if isinstance(value, etree._Element):
        return [_clone(value)]
    result: list[OxmlElement] = []
    for part in value:
        result.extend(_parts(part))
    return result


def math_concat(*parts: MathPart) -> OxmlElement:
    """Return an ``m:e`` expression containing parts in sequence."""
    expr = OxmlElement("m:e")
    for part in parts:
        for node in _parts(part):
            expr.append(node)
    return expr


def _argument(tag: str, value: MathPart | Iterable[MathPart]) -> OxmlElement:
    arg = OxmlElement(tag)
    for node in _parts(value):
        arg.append(node)
    return arg


def math_fraction(numerator: MathPart, denominator: MathPart) -> OxmlElement:
    frac = OxmlElement("m:f")
    frac.append(_argument("m:num", numerator))
    frac.append(_argument("m:den", denominator))
    return frac


def math_sub(base: MathPart, subscript: MathPart) -> OxmlElement:
    node = OxmlElement("m:sSub")
    node.append(_argument("m:e", base))
    node.append(_argument("m:sub", subscript))
    return node


def math_sup(base: MathPart, superscript: MathPart) -> OxmlElement:
    node = OxmlElement("m:sSup")
    node.append(_argument("m:e", base))
    node.append(_argument("m:sup", superscript))
    return node


def math_subsup(base: MathPart, subscript: MathPart, superscript: MathPart) -> OxmlElement:
    node = OxmlElement("m:sSubSup")
    node.append(_argument("m:e", base))
    node.append(_argument("m:sub", subscript))
    node.append(_argument("m:sup", superscript))
    return node


def math_radical(radicand: MathPart, degree: MathPart | None = None) -> OxmlElement:
    node = OxmlElement("m:rad")
    props = OxmlElement("m:radPr")
    if degree is None:
        hide = OxmlElement("m:degHide")
        _set_attr(hide, "m:val", "1")
        props.append(hide)
    node.append(props)
    node.append(_argument("m:deg", "" if degree is None else degree))
    node.append(_argument("m:e", radicand))
    return node


def math_nary(
    operator: str,
    expression: MathPart,
    *,
    lower: MathPart | None = None,
    upper: MathPart | None = None,
) -> OxmlElement:
    """Create a summation/product/integral-style n-ary expression."""
    node = OxmlElement("m:nary")
    props = OxmlElement("m:naryPr")
    char = OxmlElement("m:chr")
    _set_attr(char, "m:val", operator)
    props.append(char)
    location = OxmlElement("m:limLoc")
    _set_attr(location, "m:val", "undOvr")
    props.append(location)
    if lower is None:
        hidden = OxmlElement("m:subHide")
        _set_attr(hidden, "m:val", "1")
        props.append(hidden)
    if upper is None:
        hidden = OxmlElement("m:supHide")
        _set_attr(hidden, "m:val", "1")
        props.append(hidden)
    node.append(props)
    node.append(_argument("m:sub", "" if lower is None else lower))
    node.append(_argument("m:sup", "" if upper is None else upper))
    node.append(_argument("m:e", expression))
    return node


def math_delimiter(
    expression: MathPart | Iterable[MathPart], begin: str = "(", end: str = ")"
) -> OxmlElement:
    node = OxmlElement("m:d")
    props = OxmlElement("m:dPr")
    begin_char = OxmlElement("m:begChr")
    _set_attr(begin_char, "m:val", begin)
    end_char = OxmlElement("m:endChr")
    _set_attr(end_char, "m:val", end)
    props.extend([begin_char, end_char])
    node.append(props)
    node.append(_argument("m:e", expression))
    return node


def math_matrix(rows: Sequence[Sequence[MathPart]]) -> OxmlElement:
    if not rows or not rows[0]:
        raise ValueError("matrix must have at least one row and one column")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("all matrix rows must have the same number of columns")
    matrix = OxmlElement("m:m")
    for row in rows:
        row_node = OxmlElement("m:mr")
        for cell in row:
            row_node.append(_argument("m:e", cell))
        matrix.append(row_node)
    return matrix


def omath(parts: MathPart | Iterable[MathPart]) -> OxmlElement:
    """Wrap math parts in an inline ``m:oMath`` element."""
    node = OxmlElement("m:oMath")
    for part in _parts(parts):
        node.append(part)
    return node


def omath_para(parts: MathPart | Iterable[MathPart]) -> OxmlElement:
    """Wrap math parts in a centered display ``m:oMathPara`` element."""
    para = OxmlElement("m:oMathPara")
    props = OxmlElement("m:oMathParaPr")
    justification = OxmlElement("m:jc")
    _set_attr(justification, "m:val", "center")
    props.append(justification)
    para.append(props)
    para.append(omath(parts))
    return para


def _set_cell_margins(cell, top=60, start=80, bottom=60, end=80) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        element = tc_mar.find(qn(f"w:{side}"))
        if element is None:
            element = OxmlElement(f"w:{side}")
            tc_mar.append(element)
        _set_attr(element, "w:w", str(value))
        _set_attr(element, "w:type", "dxa")


def _set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    _set_attr(repeat, "w:val", "true")
    tr_pr.append(repeat)


def _field(paragraph, instruction: str, placeholder: str = "") -> None:
    """Append an updateable Word complex field to a paragraph."""
    def field_run(child: OxmlElement) -> OxmlElement:
        run = OxmlElement("w:r")
        run.append(child)
        return run

    begin = OxmlElement("w:fldChar")
    _set_attr(begin, "w:fldCharType", "begin")
    _set_attr(begin, "w:dirty", "true")
    instr = OxmlElement("w:instrText")
    instr.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    _set_attr(separate, "w:fldCharType", "separate")
    end = OxmlElement("w:fldChar")
    _set_attr(end, "w:fldCharType", "end")
    for element in (begin, instr, separate):
        paragraph._p.append(field_run(element))
    if placeholder:
        paragraph.add_run(placeholder)
    paragraph._p.append(field_run(end))


class ReportBuilder:
    """Small opinionated builder for a Chinese technical Word report."""

    def __init__(self, title: str = "技术报告", *, header_text: str = "") -> None:
        self.document = Document()
        self.title = title
        self.header_text = header_text or title
        self._figure_number = 0
        self._table_number = 0
        self._configure_page()
        self._configure_styles()
        self._configure_fields()
        self._configure_header_footer()

    def _configure_page(self) -> None:
        section = self.document.sections[0]
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(2.5)
        section.header_distance = Cm(1.5)
        section.footer_distance = Cm(1.5)

    @staticmethod
    def _font(style, east_asia: str, latin: str, size: float, *, bold=False) -> None:
        font = style.font
        font.name = latin
        font.size = Pt(size)
        font.bold = bold
        font.color.rgb = RGBColor(0, 0, 0)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), east_asia)
        style._element.rPr.rFonts.set(qn("w:ascii"), latin)
        style._element.rPr.rFonts.set(qn("w:hAnsi"), latin)

    def _configure_styles(self) -> None:
        styles = self.document.styles
        normal = styles["Normal"]
        self._font(normal, "宋体", "Times New Roman", 12)
        normal.paragraph_format.line_spacing = 1.5
        normal.paragraph_format.first_line_indent = Pt(24)
        normal.paragraph_format.space_after = Pt(0)

        heading_specs = {
            "Title": ("黑体", "Times New Roman", 22, True, WD_ALIGN_PARAGRAPH.CENTER),
            "Heading 1": ("黑体", "Times New Roman", 18, True, WD_ALIGN_PARAGRAPH.LEFT),
            "Heading 2": ("黑体", "Times New Roman", 16, True, WD_ALIGN_PARAGRAPH.LEFT),
            "Heading 3": ("黑体", "Times New Roman", 14, True, WD_ALIGN_PARAGRAPH.LEFT),
            "Caption": ("宋体", "Times New Roman", 10.5, False, WD_ALIGN_PARAGRAPH.CENTER),
        }
        for name, (cn, latin, size, bold, align) in heading_specs.items():
            style = styles[name]
            self._font(style, cn, latin, size, bold=bold)
            style.paragraph_format.alignment = align
            style.paragraph_format.first_line_indent = Pt(0)
            style.paragraph_format.keep_with_next = name.startswith("Heading")

        if "Code Block" not in styles:
            code = styles.add_style("Code Block", WD_STYLE_TYPE.PARAGRAPH)
        else:
            code = styles["Code Block"]
        self._font(code, "等线", "Consolas", 9)
        code.paragraph_format.left_indent = Cm(0.3)
        code.paragraph_format.right_indent = Cm(0.3)
        code.paragraph_format.first_line_indent = Pt(0)
        code.paragraph_format.line_spacing = 1.0
        code.paragraph_format.space_before = Pt(2)
        code.paragraph_format.space_after = Pt(2)

    def _configure_fields(self) -> None:
        settings = self.document.settings._element
        update = settings.find(qn("w:updateFields"))
        if update is None:
            update = OxmlElement("w:updateFields")
            settings.append(update)
        _set_attr(update, "w:val", "true")

    def _configure_header_footer(self) -> None:
        for section in self.document.sections:
            header = section.header.paragraphs[0]
            header.text = self.header_text
            header.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in header.runs:
                run.font.name = "宋体"
                run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
                run.font.size = Pt(9)
            footer = section.footer.paragraphs[0]
            footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _field(footer, " PAGE ", "1")

    def add_cover(self, subtitle: str = "", metadata: Sequence[tuple[str, str]] = ()) -> None:
        for _ in range(4):
            self.document.add_paragraph()
        self.document.add_heading(self.title, 0)
        if subtitle:
            paragraph = self.document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph.add_run(subtitle)
            run.bold = True
            run.font.size = Pt(16)
            run.font.name = "黑体"
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        for _ in range(5):
            self.document.add_paragraph()
        for key, value in metadata:
            paragraph = self.document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.first_line_indent = Pt(0)
            paragraph.add_run(f"{key}：{value}")
        self.document.add_page_break()

    def add_toc(self, title: str = "目录", levels: str = "1-3") -> None:
        self.document.add_heading(title, level=1)
        paragraph = self.document.add_paragraph()
        paragraph.paragraph_format.first_line_indent = Pt(0)
        _field(paragraph, f' TOC \\o "{levels}" \\h \\z \\u ', "请在 Word 中更新目录")
        self.document.add_page_break()

    def add_heading(self, text: str, level: int = 1):
        return self.document.add_heading(text, level=level)

    def add_paragraph(self, text: str = "", *, style: str | None = None):
        return self.document.add_paragraph(text, style=style)

    def add_inline_equation(self, paragraph, parts: MathPart | Iterable[MathPart]) -> OxmlElement:
        node = omath(parts)
        paragraph._p.append(node)
        return node

    def add_equation(
        self,
        parts: MathPart | Iterable[MathPart],
        number: str | None = None,
    ):
        """Add a centered native equation and optional same-line number.

        Numbered equations use an invisible three-column table, the standard
        robust Word layout.  The formula itself remains an editable m:oMath.
        Use :meth:`add_display_equation` when an m:oMathPara block is needed.
        """
        table = self.document.add_table(rows=1, cols=3)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        table.columns[0].width = Cm(2.0)
        table.columns[1].width = Cm(12.0)
        table.columns[2].width = Cm(2.0)
        table_el = table._tbl
        table_pr = table_el.tblPr
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            border = OxmlElement(f"w:{edge}")
            _set_attr(border, "w:val", "nil")
            borders.append(border)
        table_pr.append(borders)

        blank_cell, formula_cell, number_cell = table.rows[0].cells
        blank_cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        formula_cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        number_cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        _set_cell_margins(blank_cell, top=30, start=0, bottom=30, end=0)
        _set_cell_margins(formula_cell, top=30, start=0, bottom=30, end=0)
        _set_cell_margins(number_cell, top=30, start=0, bottom=30, end=0)
        blank_cell.paragraphs[0].paragraph_format.first_line_indent = Pt(0)
        formula_p = formula_cell.paragraphs[0]
        formula_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        formula_p.paragraph_format.first_line_indent = Pt(0)
        formula_p._p.append(omath(parts))
        number_p = number_cell.paragraphs[0]
        number_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        number_p.paragraph_format.first_line_indent = Pt(0)
        if number:
            number_p.add_run(number if number.startswith("（") else f"（{number}）")
        return table

    def add_display_equation(self, parts: MathPart | Iterable[MathPart]) -> OxmlElement:
        """Add a Word-compatible paragraph containing ``m:oMathPara``."""
        node = omath_para(parts)
        paragraph = self.document.add_paragraph()
        paragraph.paragraph_format.first_line_indent = Pt(0)
        paragraph._p.append(node)
        return node

    def add_code_block(self, code: str, *, language: str = ""):
        table = self.document.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.cell(0, 0)
        shading = OxmlElement("w:shd")
        _set_attr(shading, "w:fill", "F2F2F2")
        cell._tc.get_or_add_tcPr().append(shading)
        _set_cell_margins(cell, top=90, start=120, bottom=90, end=120)
        paragraph = cell.paragraphs[0]
        paragraph.style = self.document.styles["Code Block"]
        paragraph.paragraph_format.keep_together = True
        if language:
            label = paragraph.add_run(f"[{language}]\n")
            label.bold = True
            label.font.color.rgb = RGBColor(89, 89, 89)
        paragraph.add_run(code.rstrip())
        return table

    def add_image(
        self,
        path: str | Path,
        *,
        caption: str | None = None,
        width_cm: float = 14.0,
    ):
        paragraph = self.document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.first_line_indent = Pt(0)
        paragraph.add_run().add_picture(str(path), width=Cm(width_cm))
        if caption:
            self._figure_number += 1
            self.document.add_paragraph(
                f"图 {self._figure_number} {caption}", style="Caption"
            )
        return paragraph

    def add_table(
        self,
        headers: Sequence[str],
        rows: Iterable[Sequence[object]],
        *,
        caption: str | None = None,
        repeat_header: bool = True,
    ):
        if caption:
            self._table_number += 1
            self.document.add_paragraph(
                f"表 {self._table_number} {caption}", style="Caption"
            )
        table = self.document.add_table(rows=1, cols=len(headers))
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for index, value in enumerate(headers):
            cell = table.rows[0].cells[index]
            cell.text = str(value)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                paragraph.paragraph_format.first_line_indent = Pt(0)
                for run in paragraph.runs:
                    run.bold = True
        if repeat_header:
            _set_repeat_table_header(table.rows[0])
        for values in rows:
            if len(values) != len(headers):
                raise ValueError("table row length does not match headers")
            cells = table.add_row().cells
            for index, value in enumerate(values):
                cells[index].text = str(value)
                cells[index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                for paragraph in cells[index].paragraphs:
                    paragraph.paragraph_format.first_line_indent = Pt(0)
        return table

    def add_section_break(self, start=WD_SECTION_START.NEW_PAGE):
        return self.document.add_section(start)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.document.save(output)
        return output


def validate_docx(path: str | Path, *, minimum_equations: int = 1) -> dict[str, int | bool]:
    """Perform structural checks without requiring Microsoft Word."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    with zipfile.ZipFile(target) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"corrupt DOCX member: {bad}")
        document_xml = archive.read("word/document.xml")
        settings_xml = archive.read("word/settings.xml")
        footer_names = [name for name in archive.namelist() if name.startswith("word/footer")]
        footer_xml = b"".join(archive.read(name) for name in footer_names)
        media_count = sum(name.startswith("word/media/") for name in archive.namelist())
    document_root = etree.fromstring(document_xml)
    math_ns = "http://schemas.openxmlformats.org/officeDocument/2006/math"
    word_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    counts = {
        "oMathPara": len(document_root.findall(f".//{{{math_ns}}}oMathPara")),
        "oMath": len(document_root.findall(f".//{{{math_ns}}}oMath")),
        "fractions": len(document_root.findall(f".//{{{math_ns}}}f")),
        "scripts": sum(
            len(document_root.findall(f".//{{{math_ns}}}{tag}"))
            for tag in ("sSub", "sSup", "sSubSup")
        ),
        "tocField": document_xml.count(b" TOC "),
        "pageField": footer_xml.count(b" PAGE "),
        "tables": len(document_root.findall(f".//{{{word_ns}}}tbl")),
        "media": media_count,
        "updateFields": settings_xml.count(b"<w:updateFields"),
        "zipValid": True,
    }
    failures = []
    if counts["oMath"] < minimum_equations:
        failures.append(f"expected at least {minimum_equations} native math objects")
    if counts["oMathPara"] < 1:
        failures.append("missing block-level m:oMathPara equation")
    if counts["tocField"] < 1:
        failures.append("missing TOC field")
    if counts["pageField"] < 1:
        failures.append("missing PAGE field")
    if counts["updateFields"] < 1:
        failures.append("missing updateFields setting")
    if failures:
        raise ValueError("; ".join(failures))
    # Reopen with python-docx to catch package/relationship errors.
    Document(target)
    return counts


def validate_with_word(path: str | Path) -> dict[str, int | bool]:
    """Open a DOCX read-only in local Microsoft Word and return object counts.

    This Windows-only optional check catches schema-placement errors that ZIP,
    XML, and python-docx checks cannot detect.  It never saves the document.
    """
    target = Path(path).resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        raise RuntimeError("Microsoft Word validation requires pywin32") from exc

    pythoncom.CoInitialize()
    word = None
    document = None
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        document = word.Documents.Open(
            str(target), ConfirmConversions=False, ReadOnly=True, AddToRecentFiles=False
        )
        return {
            "openedByWord": True,
            "wordEquations": int(document.OMaths.Count),
            "wordTables": int(document.Tables.Count),
            "wordTOCs": int(document.TablesOfContents.Count),
            "wordInlineShapes": int(document.InlineShapes.Count),
            "wordPages": int(document.ComputeStatistics(2)),
        }
    finally:
        if document is not None:
            document.Close(False)
        if word is not None:
            word.Quit()
        pythoncom.CoUninitialize()


def _write_test_png(path: Path, width: int = 320, height: int = 100) -> None:
    """Create a dependency-free RGB PNG used only by the self-test."""
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend((35 + x * 80 // width, 92 + y * 80 // height, 150))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def build_minimal_test(output: str | Path) -> Path:
    builder = ReportBuilder("OMML 技术报告生成测试", header_text="Word 报告基础设施测试")
    builder.add_cover("python-docx + lxml", [("生成方式", "自动化构建")])
    builder.add_toc()
    builder.add_heading("1 功能验证", 1)
    paragraph = builder.add_paragraph("行内公式示例：")
    builder.add_inline_equation(paragraph, [math_sub("y", "i"), " = ", math_sup("x", "2")])
    paragraph.add_run("，该公式位于正文数学区域。")

    mse = [
        "RMSE = ",
        math_radical(
            math_fraction(
                [math_nary("∑", math_sup([math_sub("y", "i"), " − ", math_sub("ŷ", "i")], "2"), lower="i=1", upper="N")],
                "N",
            )
        ),
    ]
    builder.add_equation(mse, "1-1")
    builder.add_equation(
        [
            "y = ",
            math_matrix([["1", "0"], ["0", "1"]]),
            math_delimiter([math_sub("x", "1"), ", ", math_sub("x", "2")], "[", "]"),
        ],
        "1-2",
    )
    builder.add_display_equation(["H(X,W) = XW · g(X,W)"])
    builder.add_code_block(
        "def device_output(x, w, error):\n    return x * w + error[x, w]",
        language="Python",
    )
    builder.add_table(
        ["能力", "验证结果"],
        [("OMML 公式", "可编辑"), ("目录与页码", "域代码"), ("表格", "表头重复")],
        caption="基础能力验证",
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = Path(temp_dir) / "test.png"
        _write_test_png(image_path)
        builder.add_image(image_path, caption="嵌入图片验证", width_cm=10)
        return builder.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("omml_minimal_test.docx"),
    )
    parser.add_argument("--validate", type=Path, help="validate an existing DOCX")
    parser.add_argument(
        "--word",
        action="store_true",
        help="also open the document read-only in local Microsoft Word",
    )
    args = parser.parse_args()
    target = args.validate if args.validate else build_minimal_test(args.output)
    counts = validate_docx(target, minimum_equations=4 if not args.validate else 1)
    print(f"validated: {target}")
    for key, value in counts.items():
        print(f"  {key}: {value}")
    if args.word:
        for key, value in validate_with_word(target).items():
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

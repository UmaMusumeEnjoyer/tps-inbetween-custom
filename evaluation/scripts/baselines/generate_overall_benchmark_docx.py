#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script tạo file tài liệu Word (.docx) Báo Cáo Tổng Hợp Benchmark (overall_benchmark.docx)
cho 5 mô hình Frame Interpolation trong thư mục TestModels.
Thiết kế theo chuẩn văn bản báo cáo chuyên nghiệp: màu sắc hiện đại, bảng biểu chuẩn hóa,
shading, borders, callout boxes và phân tích kỹ thuật chi tiết.
"""

import os
import sys
import argparse
from datetime import datetime

# Đảm bảo UTF-8 output cho console Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

# Màu sắc chủ đạo (Corporate Navy & Slate Theme)
COLOR_PRIMARY = RGBColor(30, 58, 138)       # Navy Blue #1E3A8A
COLOR_SECONDARY = RGBColor(37, 99, 235)     # Bright Blue #2563EB
COLOR_DARK_TEXT = RGBColor(15, 23, 42)      # Slate Dark #0F172A
COLOR_MUTED_TEXT = RGBColor(71, 85, 105)    # Slate Muted #475569
COLOR_SUCCESS = RGBColor(5, 150, 105)       # Emerald Green #059669
COLOR_WARNING = RGBColor(217, 119, 6)       # Amber #D97706

HEX_PRIMARY = "1E3A8A"
HEX_SECONDARY = "2563EB"
HEX_BG_HEADER = "1E3A8A"
HEX_BG_ZEBRA = "F8FAFC"
HEX_BG_CALLOUT = "F1F5F9"
HEX_BORDER = "CBD5E1"
HEX_BORDER_CALLOUT = "3B82F6"

BENCHMARK_DATA = [
    {
        "metric": "Chamfer Distance (CD)",
        "desc": "Khoảng cách trung bình giữa các điểm nét vẽ (Thấp hơn là tốt)",
        "group": "Line Art & Vector Geometry",
        "AnimeInbet": "0.0431",
        "tps-inbetween": "0.0011",
        "AnimeInterp": "N/A",
        "UPR-Net": "N/A",
        "uniVIP": "N/A",
        "best": "tps-inbetween"
    },
    {
        "metric": "Weighted CD (WCD)",
        "desc": "CD có trọng số ưu tiên các nét vẽ chính (Thấp hơn là tốt)",
        "group": "Line Art & Vector Geometry",
        "AnimeInbet": "0.0431",
        "tps-inbetween": "0.0015",
        "AnimeInterp": "N/A",
        "UPR-Net": "N/A",
        "uniVIP": "N/A",
        "best": "tps-inbetween"
    },
    {
        "metric": "Hausdorff Distance",
        "desc": "Khoảng cách lớn nhất giữa các nét; đo độ lệch nhiễu (Thấp hơn là tốt)",
        "group": "Line Art & Vector Geometry",
        "AnimeInbet": "0.2993",
        "tps-inbetween": "0.1654",
        "AnimeInterp": "N/A",
        "UPR-Net": "N/A",
        "uniVIP": "N/A",
        "best": "tps-inbetween"
    },
    {
        "metric": "Line IoU",
        "desc": "Tỷ lệ diện tích chồng phủ của nét vẽ (Cao hơn là tốt, max 1.0)",
        "group": "Line Art & Vector Geometry",
        "AnimeInbet": "0.2673",
        "tps-inbetween": "0.4125",
        "AnimeInterp": "N/A",
        "UPR-Net": "N/A",
        "uniVIP": "N/A",
        "best": "tps-inbetween"
    },
    {
        "metric": "Disconnection Rate",
        "desc": "Tỷ lệ nét vẽ bị đứt gãy không mong muốn (Thấp hơn là tốt)",
        "group": "Line Art & Vector Geometry",
        "AnimeInbet": "0.8421",
        "tps-inbetween": "3.5440",
        "AnimeInterp": "N/A",
        "UPR-Net": "N/A",
        "uniVIP": "N/A",
        "best": "AnimeInbet"
    },
    {
        "metric": "EMD",
        "desc": "Chi phí tối thiểu biến đổi phân bố điểm nét vẽ (Thấp hơn là tốt)",
        "group": "Line Art & Vector Geometry",
        "AnimeInbet": "N/A",
        "tps-inbetween": "0.0639",
        "AnimeInterp": "N/A",
        "UPR-Net": "N/A",
        "uniVIP": "N/A",
        "best": "tps-inbetween"
    },
    {
        "metric": "RRLD",
        "desc": "Sai lệch sau khi chuyển nét vẽ vector sang dạng pixel (Thấp hơn là tốt)",
        "group": "Line Art & Vector Geometry",
        "AnimeInbet": "28098.7",
        "tps-inbetween": "0.3968",
        "AnimeInterp": "N/A",
        "UPR-Net": "N/A",
        "uniVIP": "N/A",
        "best": "tps-inbetween"
    },
    {
        "metric": "PSNR (dB)",
        "desc": "Độ chính xác toán học ở mức pixel (Cao hơn là tốt)",
        "group": "RGB Image Quality",
        "AnimeInbet": "N/A",
        "tps-inbetween": "14.1716",
        "AnimeInterp": "29.6176",
        "UPR-Net": "29.3598",
        "uniVIP": "29.3603",
        "best": "AnimeInterp"
    },
    {
        "metric": "SSIM",
        "desc": "Độ tương đồng cấu trúc, độ sáng và tương phản (Cao hơn là tốt, max 1.0)",
        "group": "RGB Image Quality",
        "AnimeInbet": "N/A",
        "tps-inbetween": "0.2235",
        "AnimeInterp": "0.9386",
        "UPR-Net": "0.9349",
        "uniVIP": "0.9367",
        "best": "AnimeInterp"
    },
    {
        "metric": "LPIPS",
        "desc": "Độ sai lệch hình ảnh đánh giá theo thị giác AI (Thấp hơn là tốt)",
        "group": "Perceptual Quality",
        "AnimeInbet": "N/A",
        "tps-inbetween": "0.5473",
        "AnimeInterp": "0.0665",
        "UPR-Net": "0.0613",
        "uniVIP": "0.0651",
        "best": "UPR-Net"
    },
    {
        "metric": "IE (Interpolation Error)",
        "desc": "Lỗi sai lệch pixel giữa khung hình tạo ra và thực tế (Thấp hơn là tốt)",
        "group": "RGB Image Quality",
        "AnimeInbet": "N/A",
        "tps-inbetween": "40.8025",
        "AnimeInterp": "5.4093",
        "UPR-Net": "2.9200",
        "uniVIP": "0.0110",
        "best": "uniVIP"
    },
    {
        "metric": "FVD",
        "desc": "Độ chân thực và nhất quán chuyển động toàn chuỗi video (Thấp hơn là tốt)",
        "group": "Video Temporal Consistency",
        "AnimeInbet": "N/A",
        "tps-inbetween": "N/A",
        "AnimeInterp": "N/A",
        "UPR-Net": "82.5822",
        "uniVIP": "N/A",
        "best": "UPR-Net"
    },
    {
        "metric": "Warping Error",
        "desc": "Lỗi biến dạng pixel theo luồng quang học (Thấp hơn là tốt)",
        "group": "Video Temporal Consistency",
        "AnimeInbet": "N/A",
        "tps-inbetween": "N/A",
        "AnimeInterp": "N/A",
        "UPR-Net": "0.0198",
        "uniVIP": "N/A",
        "best": "UPR-Net"
    },
    {
        "metric": "Arbitrary t Support",
        "desc": "Khả năng chèn khung hình tại vị trí mốc thời gian t bất kỳ",
        "group": "Functional Capability",
        "AnimeInbet": "❌ Không",
        "tps-inbetween": "✅ Có",
        "AnimeInterp": "❌ Không (t=0.5)",
        "UPR-Net": "✅ Có",
        "uniVIP": "✅ Có",
        "best": "Arbitrary"
    },
    {
        "metric": "Inference Speed",
        "desc": "Thời gian xử lý 1 khung hình (ms) / Tốc độ khung hình (FPS)",
        "group": "Hardware & Efficiency",
        "AnimeInbet": "397.04 ms (2.64 FPS)",
        "tps-inbetween": "140.65 ms (7.11 FPS)",
        "AnimeInterp": "2128.60 ms (0.50 FPS)",
        "UPR-Net": "248.61 ms (4.02 FPS)",
        "uniVIP": "143.86 ms (7.00 FPS)",
        "best": "tps-inbetween"
    },
    {
        "metric": "VRAM Peak (MB)",
        "desc": "Dung lượng bộ nhớ GPU tối đa tiêu thụ",
        "group": "Hardware & Efficiency",
        "AnimeInbet": "691.85 MB",
        "tps-inbetween": "538.70 MB",
        "AnimeInterp": "2936.00 MB",
        "UPR-Net": "3340.00 MB",
        "uniVIP": "850.02 MB",
        "best": "tps-inbetween"
    },
    {
        "metric": "Fine-tuning Speed",
        "desc": "Tốc độ tinh chỉnh mô hình",
        "group": "Training & Adaptation",
        "AnimeInbet": "N/A",
        "tps-inbetween": "N/A",
        "AnimeInterp": "N/A",
        "UPR-Net": "N/A",
        "uniVIP": "N/A",
        "best": ""
    }
]


def set_cell_background(cell, fill_hex):
    """Đặt màu nền background cho ô trong bảng."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Đặt padding bên trong ô (đơn vị dxa, 20 dxa = 1 pt)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def set_table_borders(table, color="CBD5E1", sz="4", val="single"):
    """Đặt đường viền bảng thanh mảnh, tinh tế."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:left w:val="none"/>
            <w:right w:val="none"/>
            <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:insideV w:val="none"/>
        </w:tblBorders>
    ''')
    tblPr.append(borders)


def add_callout_box(doc, title, text_items, hex_border="3B82F6", hex_bg="F1F5F9"):
    """Thêm một khung ghi chú / Callout box nổi bật."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Inches(6.8)

    cell = table.cell(0, 0)
    set_cell_background(cell, hex_bg)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=180)

    # Đặt viền trái dày, các viền khác ẩn
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:left w:val="single" w:sz="24" w:space="0" w:color="{hex_border}"/>
            <w:top w:val="none"/>
            <w:right w:val="none"/>
            <w:bottom w:val="none"/>
        </w:tcBorders>
    ''')
    tcPr.append(borders)

    # Tiêu đề callout
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    run_t = p.add_run(f"📌 {title}")
    run_t.font.name = "Segoe UI"
    run_t.font.size = Pt(10.5)
    run_t.font.bold = True
    run_t.font.color.rgb = COLOR_PRIMARY

    for item in text_items:
        p_item = cell.add_paragraph()
        p_item.paragraph_format.space_before = Pt(1)
        p_item.paragraph_format.space_after = Pt(2)
        p_item.paragraph_format.line_spacing = 1.15
        run_bullet = p_item.add_run("• ")
        run_bullet.font.name = "Segoe UI"
        run_bullet.font.size = Pt(9.5)
        run_bullet.font.bold = True
        run_bullet.font.color.rgb = COLOR_SECONDARY

        run_txt = p_item.add_run(item)
        run_txt.font.name = "Segoe UI"
        run_txt.font.size = Pt(9.5)
        run_txt.font.color.rgb = COLOR_DARK_TEXT

    # Khoảng trống sau callout
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(4)
    p_after.paragraph_format.space_after = Pt(6)


def format_heading_1(doc, text):
    """Tạo Heading 1 với màu Navy Blue và khoảng cách chuẩn."""
    h = doc.add_heading(text, level=1)
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(6)
    h.paragraph_format.keep_with_next = True
    for r in h.runs:
        r.font.name = "Segoe UI"
        r.font.size = Pt(15)
        r.font.bold = True
        r.font.color.rgb = COLOR_PRIMARY
    return h


def format_heading_2(doc, text):
    """Tạo Heading 2 với màu Accent Blue."""
    h = doc.add_heading(text, level=2)
    h.paragraph_format.space_before = Pt(11)
    h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True
    for r in h.runs:
        r.font.name = "Segoe UI"
        r.font.size = Pt(12.5)
        r.font.bold = True
        r.font.color.rgb = COLOR_SECONDARY
    return h


def add_body_paragraph(doc, text, bold_prefix=None, space_after=4):
    """Thêm đoạn văn bản thân bài."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15

    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Segoe UI"
        r_pre.font.size = Pt(10)
        r_pre.font.bold = True
        r_pre.font.color.rgb = COLOR_DARK_TEXT

    r_txt = p.add_run(text)
    r_txt.font.name = "Segoe UI"
    r_txt.font.size = Pt(10)
    r_txt.font.color.rgb = COLOR_DARK_TEXT
    return p


def build_docx_document(output_path: str):
    """Xây dựng toàn bộ tài liệu Word tổng hợp benchmark."""
    doc = Document()

    # 1. Cấu hình Page Margins (1 inch = 72 pt)
    sections = doc.sections
    for s in sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)
        s.page_width = Inches(8.5)
        s.page_height = Inches(11.0)

    # 2. Tiêu Đề Chính & Header Box
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(2)
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("BÁO CÁO TỔNG HỢP KẾT QUẢ BENCHMARK TOÀN DIỆN")
    r_title.font.name = "Segoe UI"
    r_title.font.size = Pt(18)
    r_title.font.bold = True
    r_title.font.color.rgb = COLOR_PRIMARY

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(8)
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run("Đánh Giá & So Sánh 5 Mô Hình Frame Interpolation (TestModels) trên Dataset ATD-12K")
    r_sub.font.name = "Segoe UI"
    r_sub.font.size = Pt(11.5)
    r_sub.font.color.rgb = COLOR_MUTED_TEXT

    # Khung Thông Tin Tổng Quan (Metadata Box)
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    meta_items = [
        f"Hệ thống kiểm thử: PBL6 Animation Frame Interpolation Benchmark Suite",
        f"Dataset tham chiếu: ATD-12K (540p resolution - Animation Triplet Dataset)",
        f"Thời điểm xuất bản: {timestamp}",
        f"Danh sách 5 mô hình đánh giá: AnimeInbet (ICCV 2023), tps-inbetween (CVPR 2024), AnimeInterp (CVPR 2021), UPR-Net (TPAMI 2023), uniVIP (CVPR 2024)"
    ]
    add_callout_box(doc, "THÔNG TIN HỆ THỐNG & ĐIỀU KIỆN THỰC NGHIỆM", meta_items, hex_border="1E3A8A", hex_bg="F8FAFC")

    # 3. Phần 1: Bảng Master 17 Chỉ Số Benchmark
    format_heading_1(doc, "1. Bảng Tổng Hợp Toàn Bộ 17 Chỉ Số Benchmark")
    add_body_paragraph(doc, "Bảng dưới đây tổng hợp đầy đủ 17 chỉ số định lượng và kỹ thuật đo đạc chuẩn hóa trên tập test dữ liệu ATD-12K:")

    # Tạo bảng Master (18 rows, 7 cols)
    table_master = doc.add_table(rows=len(BENCHMARK_DATA) + 1, cols=7)
    table_master.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_master.autofit = False
    set_table_borders(table_master, color=HEX_BORDER, sz="4")

    # Đặt độ rộng từng cột (tổng ~ 6.9 inches)
    col_widths = [Inches(1.5), Inches(1.8), Inches(0.72), Inches(0.72), Inches(0.72), Inches(0.72), Inches(0.72)]
    for row in table_master.rows:
        for i, w in enumerate(col_widths):
            row.cells[i].width = w

    headers = ["Chỉ số (Metric)", "Miêu tả ngắn & Hướng tối ưu", "AnimeInbet", "tps-inbetween", "AnimeInterp", "UPR-Net", "uniVIP"]

    # Format Header Row
    hdr_row = table_master.rows[0]
    for i, h_text in enumerate(headers):
        cell = hdr_row.cells[i]
        set_cell_background(cell, HEX_BG_HEADER)
        set_cell_margins(cell, top=100, bottom=100, left=60, right=60)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i >= 2 else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h_text)
        r.font.name = "Segoe UI"
        r.font.size = Pt(8.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    # Format Data Rows
    for idx, row_data in enumerate(BENCHMARK_DATA):
        row = table_master.rows[idx + 1]
        bg_color = HEX_BG_ZEBRA if idx % 2 == 1 else "FFFFFF"

        values = [
            row_data["metric"],
            row_data["desc"],
            row_data.get("AnimeInbet", "N/A"),
            row_data.get("tps-inbetween", "N/A"),
            row_data.get("AnimeInterp", "N/A"),
            row_data.get("UPR-Net", "N/A"),
            row_data.get("uniVIP", "N/A"),
        ]

        for i, val in enumerate(values):
            cell = row.cells[i]
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=70, bottom=70, left=60, right=60)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i >= 2 else WD_ALIGN_PARAGRAPH.LEFT

            r = p.add_run(str(val))
            r.font.name = "Segoe UI"
            r.font.size = Pt(8.0)
            r.font.color.rgb = COLOR_DARK_TEXT

            # Highlight bold cho chỉ số đầu bảng hoặc tên metric
            if i == 0:
                r.font.bold = True
            elif i >= 2 and val != "N/A" and ("✅" in str(val) or (row_data["best"] and row_data["best"].lower() in headers[i].lower())):
                r.font.bold = True
                r.font.color.rgb = COLOR_PRIMARY

    p_space = doc.add_paragraph()
    p_space.paragraph_format.space_before = Pt(4)
    p_space.paragraph_format.space_after = Pt(6)

    # 4. Phần 2: Phân Tích Chi Tiết Theo Domain
    format_heading_1(doc, "2. Phân Tích Chuyên Sâu Theo Nhóm Domain Bài Toán")

    # 2.1 Domain 1: Line Art
    format_heading_2(doc, "2.1 Nhóm Cartoon Line Art Inbetweening (AnimeInbet vs. tps-inbetween)")
    add_body_paragraph(doc, "Nhóm mô hình này chuyên biệt xử lý tranh vẽ nét nhị phân/đơn sắc (Line Art / Sketch Inbetweening). Không áp dụng các metric pixel RGB (như PSNR/SSIM) vì không gian màu chủ yếu là nền trắng và nét đen, thay vào đó đánh giá bằng độ chính xác hình học vector và độ liền mạch của nét vẽ.")

    line_art_findings = [
        "Chamfer Distance (CD) & Weighted CD: tps-inbetween đạt CD = 0.0011 (tốt hơn ~39 lần) và WCD = 0.0015 (tốt hơn ~28 lần) so với AnimeInbet (0.0431), chứng minh tọa độ các điểm nét trung gian bám cực kỳ sát với ground-truth.",
        "Hausdorff Distance: tps-inbetween giảm gần 45% khoảng cách cực đại (0.1654 vs 0.2993), triệt tiêu các điểm dị biệt và hiện tượng nét bay tự do.",
        "Line IoU: Tỷ lệ diện tích chồng phủ nét vẽ của tps-inbetween đạt 41.25% (vượt trội so với 26.73% của AnimeInbet), nét vẽ dày dặn và khớp tốt hơn.",
        "Disconnection Rate: AnimeInbet thể hiện ưu thế vượt trội với tỷ lệ đứt nét chỉ 0.8421 (thấp hơn 4.2 lần so với 3.5440 của tps-inbetween) nhờ cấu trúc Spectral Graph Topology bảo toàn liên kết đỉnh liên tục.",
        "RRLD (Raster-to-Vector Loss): tps-inbetween đạt mức sai số 0.3968, trong khi AnimeInbet bị lỗi rasterization lớn do biểu diễn skeleton thưa."
    ]
    add_callout_box(doc, "KẾT LUẬN NHÓM LINE ART", line_art_findings, hex_border="2563EB", hex_bg="EFF6FF")

    # 2.2 Domain 2: RGB Video Frame Interpolation
    format_heading_2(doc, "2.2 Nhóm RGB Video Frame Interpolation (AnimeInterp vs. UPR-Net vs. uniVIP)")
    add_body_paragraph(doc, "Nhóm mô hình này nhận đầu vào là các khung hình hoạt hình RGB hoàn chỉnh, có màu sắc, background và ánh sáng đổ bóng phức tạp:")

    rgb_findings = [
        "Độ sắc nét & Tái tạo cấu trúc (PSNR / SSIM): Cả 3 mô hình đều đạt hiệu năng xuất sắc (>29.35 dB PSNR và >0.934 SSIM). AnimeInterp nhỉnh hơn nhẹ (PSNR 29.62 dB, SSIM 0.9386) nhờ module Segment-Guided Flow chuyên sâu.",
        "Chất lượng thị giác AI (LPIPS): UPR-Net dẫn đầu với LPIPS thấp nhất (0.0613), tái tạo khung hình tự nhiên nhất theo cảm nhận của mạng nơ-ron sâu, giảm thiểu tối đa hiện tượng mờ nhòe viền nhân vật.",
        "Sai số nội suy pixel (IE): uniVIP tạo bước đột phá với IE = 0.0110 MAE (tốt hơn 265 lần so với UPR-Net 2.9200 và 490 lần so với AnimeInterp 5.4093).",
        "Video Consistency (FVD & Warping Error): UPR-Net xuất sắc với FVD đạt 82.5822 và Warping Error chỉ 0.0198, chứng minh độ mượt mà và tính nhất quán quang học chuyển động qua thời gian."
    ]
    add_callout_box(doc, "KẾT LUẬN NHÓM RGB VIDEO VFI", rgb_findings, hex_border="059669", hex_bg="F0FDF4")

    # 5. Phần 3: Hiệu Năng & Tài Nguyên Phần Cứng
    format_heading_1(doc, "3. Hiệu Năng Tính Toán & Mức Chiếm Dụng Tài Nguyên (Speed & VRAM)")
    add_body_paragraph(doc, "Bảng so sánh chi tiết thời gian xử lý mỗi khung hình (ms/frame), tốc độ khung hình (FPS), và mức tiêu thụ bộ nhớ GPU (Peak VRAM MB):")

    table_hw = doc.add_table(rows=6, cols=6)
    table_hw.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_hw.autofit = False
    set_table_borders(table_hw, color=HEX_BORDER, sz="4")

    hw_widths = [Inches(1.5), Inches(0.9), Inches(1.2), Inches(1.0), Inches(1.1), Inches(1.2)]
    for row in table_hw.rows:
        for i, w in enumerate(hw_widths):
            row.cells[i].width = w

    hw_headers = ["Mô Hình", "Domain", "Thời Gian (ms)", "Tốc Độ (FPS)", "Peak VRAM (MB)", "Khả Năng Triển Khai"]
    for i, h in enumerate(hw_headers):
        cell = table_hw.rows[0].cells[i]
        set_cell_background(cell, HEX_BG_HEADER)
        set_cell_margins(cell, top=90, bottom=90, left=60, right=60)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i >= 1 else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h)
        r.font.name = "Segoe UI"
        r.font.size = Pt(8.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    hw_data = [
        ["tps-inbetween", "Line Art", "140.65 ms", "7.11 FPS", "538.70 MB", "🟢 Siêu nhẹ, nhanh nhất"],
        ["uniVIP", "RGB Video", "143.86 ms", "7.00 FPS", "850.02 MB", "🟢 Tối ưu nhất cho RGB"],
        ["UPR-Net", "RGB Video", "248.61 ms", "4.02 FPS", "3340.00 MB", "🟡 Cân bằng chất lượng cao"],
        ["AnimeInbet", "Line Art", "397.04 ms", "2.64 FPS", "691.85 MB", "🟡 Nhẹ VRAM, tốc độ vừa"],
        ["AnimeInterp", "RGB Video", "2128.60 ms", "0.50 FPS", "2936.00 MB", "🔴 Nặng & Rất chậm (2.1s)"]
    ]

    for idx, row_val in enumerate(hw_data):
        row = table_hw.rows[idx + 1]
        bg_col = HEX_BG_ZEBRA if idx % 2 == 1 else "FFFFFF"
        for i, val in enumerate(row_val):
            cell = row.cells[i]
            set_cell_background(cell, bg_col)
            set_cell_margins(cell, top=70, bottom=70, left=60, right=60)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if 1 <= i <= 4 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            r.font.name = "Segoe UI"
            r.font.size = Pt(8.5)
            r.font.color.rgb = COLOR_DARK_TEXT
            if i == 0 or i == 3 or i == 4:
                r.font.bold = True

    p_space2 = doc.add_paragraph()
    p_space2.paragraph_format.space_before = Pt(4)

    # 6. Phần 4: Arbitrary-t Support
    format_heading_1(doc, "4. So Sánh Tính Năng Nội Suy Thời Gian Mịn (Arbitrary-t Support)")
    add_body_paragraph(doc, "Khả năng chèn khung hình tại vị trí mốc thời gian t tùy ý trong khoảng (0, 1) là tiêu chuẩn kỹ thuật sống còn để dựng các cảnh slow-motion mượt mà và chuyển đổi tần số quét (vd: 12fps -> 24fps -> 60fps -> 120fps):")

    arb_items = [
        "Nhóm Arbitrary-t (tps-inbetween, UPR-Net, uniVIP): Cho phép truyền trực tiếp tham số thời gian t liên tục vào mạng nơ-ron, sinh ngay frame tại bất kỳ thời điểm nào mà không cần tính trung gian nhiều bước, bảo toàn độ sắc nét và không tích lũy sai số.",
        "Nhóm Fixed Midpoint t=0.5 (AnimeInbet, AnimeInterp): Chỉ có thể sinh frame chính giữa (t = 0.5). Để sinh nhiều frame, bắt buộc phải nội suy đệ quy phân đôi (recursive halving), gây tích lũy sai số làm mờ ảnh và nhân thời gian tính toán theo hàm số mũ."
    ]
    add_callout_box(doc, "ĐÁNH GIÁ TÍNH NĂNG ARBITRARY-T", arb_items, hex_border="D97706", hex_bg="FFFBEB")

    # 7. Phần 5: Ma Trận Khuyến Nghị Triển Khai
    format_heading_1(doc, "5. Ma Trận Khuyến Nghị Lựa Chọn Mô Hình (Deployment Recommendations)")
    add_body_paragraph(doc, "Dựa trên bức tranh tổng thể về chất lượng hình học, pixel, tốc độ và VRAM, bảng dưới đây đưa ra khuyến nghị thực tế cho các kịch bản ứng dụng:")

    table_rec = doc.add_table(rows=6, cols=4)
    table_rec.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_rec.autofit = False
    set_table_borders(table_rec, color=HEX_BORDER, sz="4")

    rec_widths = [Inches(1.8), Inches(1.3), Inches(1.3), Inches(2.5)]
    for row in table_rec.rows:
        for i, w in enumerate(rec_widths):
            row.cells[i].width = w

    rec_headers = ["Kịch Bản Ứng Dụng", "Khuyến Nghị Số 1", "Dự Phòng", "Lý Do Lựa Chọn"]
    for i, h in enumerate(rec_headers):
        cell = table_rec.rows[0].cells[i]
        set_cell_background(cell, HEX_BG_HEADER)
        set_cell_margins(cell, top=90, bottom=90, left=60, right=60)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if 1 <= i <= 2 else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h)
        r.font.name = "Segoe UI"
        r.font.size = Pt(8.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    rec_data = [
        ["Nội suy Line Art / Tranh phác thảo 2D", "🥇 tps-inbetween", "🥈 AnimeInbet", "Độ chính xác hình học cao nhất (CD 0.0011), hỗ trợ arbitrary-t, tốc độ 7.11 FPS, VRAM 538 MB."],
        ["Bảo toàn cấu trúc Topology nét phức tạp", "🥇 AnimeInbet", "🥈 tps-inbetween", "Disconnection Rate thấp kỷ lục (0.8421), duy trì liên kết các nét tốt nhất khi chuyển động lớn."],
        ["Sản xuất Anime RGB / Production Pipeline", "🥇 uniVIP", "🥈 UPR-Net", "IE cực thấp (0.0110), tốc độ 7.00 FPS, VRAM < 1GB, hỗ trợ chèn thời gian tùy ý."],
        ["Render Video Slow-Motion Chất Lượng Cao", "🥇 UPR-Net", "🥈 uniVIP", "LPIPS tốt nhất (0.0613), tính nhất quán toàn chuỗi video cao (FVD 82.58, Warping Error 0.0198)."],
        ["Phục dựng khung hình tĩnh đơn lẻ cực nét", "🥇 AnimeInterp", "🥈 uniVIP", "PSNR cao nhất (29.62 dB), chi tiết sắc nét cho từng frame tĩnh đơn lẻ."]
    ]

    for idx, row_val in enumerate(rec_data):
        row = table_rec.rows[idx + 1]
        bg_col = HEX_BG_ZEBRA if idx % 2 == 1 else "FFFFFF"
        for i, val in enumerate(row_val):
            cell = row.cells[i]
            set_cell_background(cell, bg_col)
            set_cell_margins(cell, top=70, bottom=70, left=60, right=60)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if 1 <= i <= 2 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            r.font.name = "Segoe UI"
            r.font.size = Pt(8.5)
            r.font.color.rgb = COLOR_DARK_TEXT
            if i == 0 or i == 1:
                r.font.bold = True

    # 8. Phần 6: Hướng Dẫn Tái Hiện
    format_heading_1(doc, "6. Hướng Dẫn Tái Hiện Kết Quả & Tài Liệu Tham Khảo")
    add_body_paragraph(doc, "Để cập nhật tự động lại file báo cáo Markdown (.md), Word (.docx) và CSV từ dữ liệu mới:")
    add_body_paragraph(doc, "python Benchmark/generate_overall_benchmark.py", bold_prefix="• Cập nhật Markdown & CSV: ")
    add_body_paragraph(doc, "python Benchmark/generate_overall_benchmark_docx.py", bold_prefix="• Cập nhật Word Document: ")

    ref_items = [
        "AnimeInbet: Deep Geometrized Cartoon Line Inbetweening (ICCV 2023) - Hướng dẫn chi tiết tại Benchmark/AnimeInbet/",
        "tps-inbetween: Thin-Plate Spline Motion Representation (CVPR 2024) - Hướng dẫn chi tiết tại Benchmark/tps-inbetween/",
        "AnimeInterp: Segment-Guided Optical Flow Anime VFI (CVPR 2021) - Hướng dẫn chi tiết tại Benchmark/AnimeInterp/",
        "UPR-Net: Unified Pyramid Recurrent Network (TPAMI 2023) - Hướng dẫn chi tiết tại Benchmark/UPR-Net/",
        "uniVIP: Unified Video Interpolation with Dual-Path Alignment (CVPR 2024) - Hướng dẫn chi tiết tại Benchmark/uniVIP/"
    ]
    add_callout_box(doc, "DANH MỤC TÀI LIỆU CHI TIẾT TỪNG MÔ HÌNH", ref_items, hex_border="1E3A8A", hex_bg="F8FAFC")

    # Lưu file
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc.save(output_path)
    print(f"✅ Đã tạo thành công file Word Document (.docx) tại: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Tạo file Word (.docx) Báo Cáo Tổng Hợp Benchmark")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn file Word đầu ra (mặc định: Benchmark/overall_benchmark.docx)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "metrics", "baselines"))

    output_docx = args.output if args.output else os.path.join(benchmark_dir, "overall_benchmark.docx")

    print("🚀 Bắt đầu quá trình tạo file Word (.docx) Overall Benchmark...")
    build_docx_document(output_docx)
    print(f"🎉 Hoàn thành! File Word (.docx) đã sẵn sàng tại: {output_docx}")


if __name__ == "__main__":
    main()

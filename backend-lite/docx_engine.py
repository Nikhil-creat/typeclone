"""
docx_engine.py
==============
Word document (.docx) text extraction and in-place style-preserving
editing via python-docx. Unlike PDF/image, a .docx has no fixed pixel
layout — text reflows — so there is no bounding box to tap on. Regions
are addressed by (paragraph_index, run_index) instead, and the editor
UI lists them rather than overlaying a canvas.

Style captured per run: bold, italic, underline, font name, font size
(points), and RGB color (when explicitly set — inherited/theme colors
read as None and are left untouched on edit).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

from docx import Document
from docx.shared import Pt, RGBColor


@dataclass
class DocxRun:
    region_id: str          # "{paragraph_index}:{run_index}"
    paragraph_index: int
    run_index: int
    text: str
    bold: bool
    italic: bool
    underline: bool
    font_name: Optional[str]
    font_size_pt: Optional[float]
    color_hex: Optional[str]


def _run_color_hex(run) -> Optional[str]:
    try:
        color = run.font.color
        if color and color.type is not None and color.rgb is not None:
            return f"#{color.rgb}"
    except Exception:
        pass
    return None


def extract_docx(docx_bytes: bytes) -> list[DocxRun]:
    doc = Document(io.BytesIO(docx_bytes))
    runs: list[DocxRun] = []

    for p_idx, paragraph in enumerate(doc.paragraphs):
        for r_idx, run in enumerate(paragraph.runs):
            text = run.text
            if not text.strip():
                continue
            runs.append(DocxRun(
                region_id=f"{p_idx}:{r_idx}",
                paragraph_index=p_idx,
                run_index=r_idx,
                text=text,
                bold=bool(run.bold),
                italic=bool(run.italic),
                underline=bool(run.underline),
                font_name=run.font.name,
                font_size_pt=run.font.size.pt if run.font.size else None,
                color_hex=_run_color_hex(run),
            ))
    return runs


def edit_docx(docx_bytes: bytes, paragraph_index: int, run_index: int, new_text: str) -> bytes:
    doc = Document(io.BytesIO(docx_bytes))

    if paragraph_index < 0 or paragraph_index >= len(doc.paragraphs):
        raise IndexError(f"paragraph_index {paragraph_index} out of range")
    paragraph = doc.paragraphs[paragraph_index]

    if run_index < 0 or run_index >= len(paragraph.runs):
        raise IndexError(f"run_index {run_index} out of range")
    run = paragraph.runs[run_index]

    # Replacing run.text in place preserves every style attribute python-docx
    # tracks on that run (bold/italic/underline/font/size/color) automatically
    # — no need to re-apply them manually.
    run.text = new_text

    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()

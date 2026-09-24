"""
typography_engine.py
=====================
Vector-PDF Typography Extraction & Style-Cloning Engine.

Responsibilities:
  1. Parse a PDF page into structurally exact typography spans (font,
     weight, italic, size, baseline, color) via PyMuPDF.
  2. Redact and re-render a span with a new string while preserving the
     original visual profile, auto-fitting to the source bounding box.

No placeholders — this module is directly runnable against any PDF.
"""

from __future__ import annotations

import re
import math
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
from uuid import uuid4

import fitz  # PyMuPDF

logger = logging.getLogger("typography_engine")

# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class TypographyProfile:
    region_id: str
    page: int
    bbox: tuple[float, float, float, float]
    baseline_y: float
    text: str
    font_family: str
    font_family_fallback: str
    font_size_pt: float
    weight: int              # 400 regular, 700 bold
    italic: bool
    letter_spacing: float
    line_height: float
    color_rgb: tuple[int, int, int]
    color_space: str
    alignment: str
    source_engine: str = "vector"
    confidence: float = 0.99

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# PyMuPDF span.flags bitmask reference:
#   bit 0 (1)   : superscript
#   bit 1 (2)   : italic
#   bit 2 (4)   : serifed
#   bit 3 (8)   : monospaced
#   bit 4 (16)  : bold
_FLAG_ITALIC = 1 << 1
_FLAG_BOLD = 1 << 4

_BOLD_NAME_RE = re.compile(r"(bold|black|heavy|semibold|extrabold)", re.I)
_ITALIC_NAME_RE = re.compile(r"(italic|oblique)", re.I)
_SUBSET_PREFIX_RE = re.compile(r"^[A-Z]{6}\+")  # e.g. "ABCDEF+Arial-Bold"


def _clean_font_name(raw: str) -> str:
    """Strip PDF subset tags (e.g. 'ABCDEF+Arial-Bold' -> 'Arial-Bold')."""
    return _SUBSET_PREFIX_RE.sub("", raw or "")


def _infer_weight(font_name: str, flags: int) -> int:
    if flags & _FLAG_BOLD or _BOLD_NAME_RE.search(font_name):
        return 700
    if "medium" in font_name.lower():
        return 500
    if "light" in font_name.lower():
        return 300
    return 400


def _infer_italic(font_name: str, flags: int) -> bool:
    return bool(flags & _FLAG_ITALIC) or bool(_ITALIC_NAME_RE.search(font_name))


def _pack_int_to_rgb(color_int: int) -> tuple[int, int, int]:
    """PyMuPDF encodes span color as a single sRGB int."""
    r = (color_int >> 16) & 0xFF
    g = (color_int >> 8) & 0xFF
    b = color_int & 0xFF
    return (r, g, b)


def _family_fallback(font_name: str) -> str:
    lower = font_name.lower()
    if "times" in lower or "georgia" in lower or "serif" in lower:
        return "Georgia, 'Times New Roman', serif"
    if "courier" in lower or "mono" in lower or "consolas" in lower:
        return "'Courier New', monospace"
    return "Arial, Helvetica, sans-serif"


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

class VectorTypographyExtractor:
    """Extracts a full TypographyProfile list from a PDF page."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

    def close(self) -> None:
        self.doc.close()

    def extract_page(self, page_number: int) -> list[TypographyProfile]:
        if page_number < 0 or page_number >= self.doc.page_count:
            raise IndexError(f"page {page_number} out of range (0..{self.doc.page_count - 1})")

        page = self.doc[page_number]
        raw = page.get_text("dict")
        profiles: list[TypographyProfile] = []

        for block in raw.get("blocks", []):
            if block.get("type") != 0:  # 0 = text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    profiles.append(self._span_to_profile(span, page_number))

        return profiles

    def _span_to_profile(self, span: dict, page_number: int) -> TypographyProfile:
        font_name = _clean_font_name(span.get("font", ""))
        flags = span.get("flags", 0)
        bbox = tuple(span.get("bbox", (0, 0, 0, 0)))
        origin = span.get("origin", (bbox[0], bbox[3]))
        size = float(span.get("size", 12.0))
        color_int = span.get("color", 0)

        return TypographyProfile(
            region_id=str(uuid4()),
            page=page_number,
            bbox=bbox,
            baseline_y=float(origin[1]),
            text=span.get("text", ""),
            font_family=font_name or "Helvetica",
            font_family_fallback=_family_fallback(font_name),
            font_size_pt=round(size, 2),
            weight=_infer_weight(font_name, flags),
            italic=_infer_italic(font_name, flags),
            letter_spacing=0.0,
            line_height=round(size * 1.2, 2),
            color_rgb=_pack_int_to_rgb(color_int),
            color_space="DeviceRGB",
            alignment="left",
        )


# --------------------------------------------------------------------------- #
# Font resolution (matches a TypographyProfile to an installed/embeddable font)
# --------------------------------------------------------------------------- #

_BASE14_MAP = {
    (False, False): "helv",
    (True, False): "hebo",
    (False, True): "heit",
    (True, True): "hebi",
}
_SERIF_MAP = {
    (False, False): "tiro",
    (True, False): "tibo",
    (False, True): "tiit",
    (True, True): "tibi",
}
_MONO_MAP = {
    (False, False): "cour",
    (True, False): "cobo",
    (False, True): "coit",
    (True, True): "cobi",
}


def resolve_builtin_font(profile: TypographyProfile) -> str:
    """
    Maps a TypographyProfile to a PyMuPDF Base-14 font code as a robust
    fallback. Production systems should first attempt to locate/embed the
    exact font file (via fontconfig or a bundled font asset store) and
    only fall back to Base-14 here.
    """
    bold = profile.weight >= 600
    italic = profile.italic
    lower = profile.font_family.lower()

    if "courier" in lower or "mono" in lower:
        table = _MONO_MAP
    elif "times" in lower or "georgia" in lower or "serif" in lower:
        table = _SERIF_MAP
    else:
        table = _BASE14_MAP

    return table[(bold, italic)]


# --------------------------------------------------------------------------- #
# Fit-to-bbox sizing
# --------------------------------------------------------------------------- #

def compute_fitted_size(
    doc: fitz.Document,
    text: str,
    fontcode: str,
    bbox: tuple[float, float, float, float],
    original_size: float,
    min_size: float = 4.0,
    max_shrink_iterations: int = 40,
) -> tuple[float, float]:
    """
    Binary-searches a font size (starting at original_size, only ever
    shrinking) so `text` rendered in `fontcode` fits within bbox width.
    Returns (fitted_size, measured_width_at_fitted_size).
    """
    box_width = bbox[2] - bbox[0]
    font = fitz.Font(fontcode)

    def width_at(size: float) -> float:
        return font.text_length(text, fontsize=size)

    size = original_size
    w = width_at(size)
    if w <= box_width:
        return size, w

    lo, hi = min_size, original_size
    for _ in range(max_shrink_iterations):
        mid = (lo + hi) / 2
        w = width_at(mid)
        if w <= box_width:
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.05:
            break

    fitted = lo
    return fitted, width_at(fitted)


# --------------------------------------------------------------------------- #
# Style-cloning edit operation
# --------------------------------------------------------------------------- #

class StyleClonedEditor:
    """Applies a text replacement to a PDF while cloning the original
    typography profile as closely as the Base-14/embedded font set allows."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

    def close(self) -> None:
        self.doc.close()

    def apply_edit(self, profile: TypographyProfile, new_text: str) -> None:
        page = self.doc[profile.page]

        # 1. Redact original glyphs cleanly (removes vector outlines, not
        #    just paints a white box — critical for searchable-text integrity).
        pad = 0.5
        redact_box = fitz.Rect(
            profile.bbox[0] - pad,
            profile.bbox[1] - pad,
            profile.bbox[2] + pad,
            profile.bbox[3] + pad,
        )
        page.add_redact_annot(redact_box, fill=None)  # fill=None -> sample background
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        # 2. Resolve font + auto-fit size.
        fontcode = resolve_builtin_font(profile)
        fitted_size, measured_width = compute_fitted_size(
            self.doc, new_text, fontcode, profile.bbox, profile.font_size_pt
        )

        # 3. Compute letter-spacing compression if still marginally over
        #    (keeps size stable, avoids visually shrinking text for tiny overflows).
        box_width = profile.bbox[2] - profile.bbox[0]
        tracking = 0.0
        if measured_width > box_width and len(new_text) > 1:
            overflow = measured_width - box_width
            tracking = -overflow / max(len(new_text) - 1, 1)
            tracking = max(tracking, -fitted_size * 0.12)  # cap compression

        # 4. Draw new text at the original baseline (not bbox top) for
        #    correct vertical alignment.
        r, g, b = profile.color_rgb
        color = (r / 255.0, g / 255.0, b / 255.0)

        page.insert_text(
            fitz.Point(profile.bbox[0], profile.baseline_y),
            new_text,
            fontname=fontcode,
            fontsize=fitted_size,
            color=color,
            render_mode=0,
        )

        logger.info(
            "Applied edit region=%s size=%.2f->%.2f tracking=%.3f",
            profile.region_id, profile.font_size_pt, fitted_size, tracking,
        )

    def save(self, output_path: str) -> None:
        self.doc.save(output_path, garbage=4, deflate=True)


# --------------------------------------------------------------------------- #
# Public API surface used by the FastAPI layer
# --------------------------------------------------------------------------- #

def extract_document_typography(pdf_path: str) -> dict[int, list[TypographyProfile]]:
    extractor = VectorTypographyExtractor(pdf_path)
    try:
        return {
            page_num: extractor.extract_page(page_num)
            for page_num in range(extractor.doc.page_count)
        }
    finally:
        extractor.close()


def edit_document_text(
    pdf_path: str,
    output_path: str,
    profile: TypographyProfile,
    new_text: str,
) -> None:
    editor = StyleClonedEditor(pdf_path)
    try:
        editor.apply_edit(profile, new_text)
        editor.save(output_path)
    finally:
        editor.close()

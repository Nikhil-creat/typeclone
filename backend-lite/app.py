"""
app.py — TypeClone Lite
========================
A real, working, single-service version of the PDF typography editor.

No Celery, no Redis, no Postgres, no MinIO, no PaddleOCR/PyTorch — just
FastAPI + PyMuPDF, synchronous, runs on Render's free tier.

Flow:
  1. POST /extract  — upload a PDF, get back every text region with its
     real font, size, weight, italic, color, and bounding box.
  2. POST /edit      — upload the same PDF + a region_id + new text,
     get back a *new* PDF with that text replaced, cloned to the
     original typography, auto-fit to the original box.

This is the same core engine (typography extraction + style-cloned
redraw) as the full architecture, just without the async task queue —
because a single request is fast enough for one page at a time.
"""

from __future__ import annotations

import io
import re
import uuid
from typing import Optional

import fitz  # PyMuPDF
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont

import ai_vision

app = FastAPI(title="TypeClone Lite — Live PDF Typography Editor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# In-memory session store: uploaded PDF bytes keyed by a short-lived doc_id.
# Fine for a single-instance free-tier deployment; swap for S3/Redis if you
# outgrow it.
# --------------------------------------------------------------------------- #
_DOCUMENTS: dict[str, bytes] = {}

_FLAG_ITALIC = 1 << 1
_FLAG_BOLD = 1 << 4
_BOLD_RE = re.compile(r"(bold|black|heavy|semibold|extrabold)", re.I)
_ITALIC_RE = re.compile(r"(italic|oblique)", re.I)
_SUBSET_RE = re.compile(r"^[A-Z]{6}\+")


class TypographyRegion(BaseModel):
    region_id: str
    page: int
    bbox: tuple[float, float, float, float]
    baseline_y: float
    text: str
    font_family: str
    font_size_pt: float
    weight: int
    italic: bool
    color_rgb: tuple[int, int, int]


class ExtractResponse(BaseModel):
    document_id: str
    page_count: int
    regions: list[TypographyRegion]


def _clean_font(name: str) -> str:
    return _SUBSET_RE.sub("", name or "")


def _weight(font_name: str, flags: int) -> int:
    if flags & _FLAG_BOLD or _BOLD_RE.search(font_name):
        return 700
    return 400


def _italic(font_name: str, flags: int) -> bool:
    return bool(flags & _FLAG_ITALIC) or bool(_ITALIC_RE.search(font_name))


def _rgb(color_int: int) -> tuple[int, int, int]:
    return ((color_int >> 16) & 0xFF, (color_int >> 8) & 0xFF, color_int & 0xFF)


@app.post("/extract", response_model=ExtractResponse)
async def extract(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "only .pdf files are supported")

    pdf_bytes = await file.read()
    document_id = str(uuid.uuid4())
    _DOCUMENTS[document_id] = pdf_bytes

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    regions: list[TypographyRegion] = []

    for page_number in range(doc.page_count):
        page = doc[page_number]
        raw = page.get_text("dict")
        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    font_name = _clean_font(span.get("font", ""))
                    flags = span.get("flags", 0)
                    origin = span.get("origin", (0, 0))
                    regions.append(TypographyRegion(
                        region_id=str(uuid.uuid4()),
                        page=page_number,
                        bbox=tuple(span.get("bbox", (0, 0, 0, 0))),
                        baseline_y=float(origin[1]),
                        text=span.get("text", ""),
                        font_family=font_name or "Helvetica",
                        font_size_pt=round(float(span.get("size", 12.0)), 2),
                        weight=_weight(font_name, flags),
                        italic=_italic(font_name, flags),
                        color_rgb=_rgb(span.get("color", 0)),
                    ))

    page_count = doc.page_count
    doc.close()

    if len(_DOCUMENTS) > 200:  # basic memory guard on free tier
        oldest = next(iter(_DOCUMENTS))
        _DOCUMENTS.pop(oldest, None)

    return ExtractResponse(document_id=document_id, page_count=page_count, regions=regions)


_BASE14 = {
    (False, False): "helv", (True, False): "hebo",
    (False, True): "heit", (True, True): "hebi",
}


def _fit_size(text: str, fontcode: str, box_width: float, start_size: float, min_size: float = 4.0) -> float:
    font = fitz.Font(fontcode)
    if font.text_length(text, fontsize=start_size) <= box_width:
        return start_size
    lo, hi = min_size, start_size
    for _ in range(30):
        mid = (lo + hi) / 2
        if font.text_length(text, fontsize=mid) <= box_width:
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.05:
            break
    return lo


@app.post("/edit")
async def edit(
    document_id: str = Form(...),
    region_json: str = Form(...),  # JSON-encoded TypographyRegion
    new_text: str = Form(...),
):
    if document_id not in _DOCUMENTS:
        raise HTTPException(404, "document not found or session expired — re-upload and extract again")

    import json
    region = TypographyRegion(**json.loads(region_json))

    pdf_bytes = _DOCUMENTS[document_id]
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[region.page]

    pad = 0.5
    redact_box = fitz.Rect(
        region.bbox[0] - pad, region.bbox[1] - pad,
        region.bbox[2] + pad, region.bbox[3] + pad,
    )
    page.add_redact_annot(redact_box, fill=None)
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

    fontcode = _BASE14[(region.weight >= 600, region.italic)]
    box_width = region.bbox[2] - region.bbox[0]
    fitted_size = _fit_size(new_text, fontcode, box_width, region.font_size_pt)

    r, g, b = region.color_rgb
    page.insert_text(
        fitz.Point(region.bbox[0], region.baseline_y),
        new_text,
        fontname=fontcode,
        fontsize=fitted_size,
        color=(r / 255.0, g / 255.0, b / 255.0),
    )

    out_buffer = io.BytesIO()
    doc.save(out_buffer, garbage=4, deflate=True)
    doc.close()
    out_buffer.seek(0)

    return StreamingResponse(
        out_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=edited.pdf"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "documents_in_memory": len(_DOCUMENTS), "gemini_configured": bool(ai_vision.GEMINI_API_KEY)}


# --------------------------------------------------------------------------- #
# Raster (PNG) support via Gemini — replaces the PaddleOCR+PyTorch pipeline
# --------------------------------------------------------------------------- #

_IMAGES: dict[str, bytes] = {}

_FONT_CANDIDATES = {
    (False, False): [
        "assets/fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ],
    (True, False): [
        "assets/fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ],
    (False, True): [
        "assets/fonts/DejaVuSans-Oblique.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
    ],
    (True, True): [
        "assets/fonts/DejaVuSans-BoldOblique.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf",
    ],
}


def _resolve_font_path(bold: bool, italic: bool) -> str:
    import os
    for path in _FONT_CANDIDATES[(bold, italic)]:
        if os.path.exists(path):
            return path
    raise HTTPException(
        500,
        "No TTF font found. Add DejaVuSans*.ttf files under backend-lite/assets/fonts/ "
        "in your repo (see README-LITE.md) — none of the system font paths existed either.",
    )


def _fit_pil_font(text: str, font_path: str, box_w: int, box_h: int, start_size: int, min_size: int = 8) -> ImageFont.FreeTypeFont:
    size = start_size
    while size > min_size:
        font = ImageFont.truetype(font_path, size)
        bbox = font.getbbox(text)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= box_w and h <= box_h:
            return font
        size -= 1
    return ImageFont.truetype(font_path, min_size)


class ImageRegion(BaseModel):
    region_id: str
    bbox: tuple[int, int, int, int]  # pixel x0,y0,x1,y1
    text: str
    bold: bool
    italic: bool
    color_hex: str
    font_size_px: int


class ExtractImageResponse(BaseModel):
    document_id: str
    width: int
    height: int
    regions: list[ImageRegion]


@app.post("/extract-image", response_model=ExtractImageResponse)
async def extract_image(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    if not (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
        raise HTTPException(400, "only .png/.jpg/.jpeg files are supported")

    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = img.size

    try:
        raw_regions = ai_vision.analyze_image(image_bytes, mime_type=file.content_type or "image/png")
    except ai_vision.GeminiNotConfigured as e:
        raise HTTPException(503, str(e))
    except ai_vision.GeminiRequestFailed as e:
        raise HTTPException(502, f"Gemini analysis failed: {e}")

    document_id = str(uuid.uuid4())
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    _IMAGES[document_id] = buf.getvalue()

    regions: list[ImageRegion] = []
    for r in raw_regions:
        try:
            x0, y0, x1, y1 = ai_vision.denormalize_bbox(r["bbox"], width, height)
            regions.append(ImageRegion(
                region_id=str(uuid.uuid4()),
                bbox=(x0, y0, x1, y1),
                text=r.get("text", ""),
                bold=bool(r.get("bold", False)),
                italic=bool(r.get("italic", False)),
                color_hex=r.get("color_hex", "#000000"),
                font_size_px=max(int((y1 - y0) * 0.85), 8),
            ))
        except (KeyError, TypeError, ValueError):
            continue  # skip malformed entries rather than failing the whole request

    if len(_IMAGES) > 100:
        oldest = next(iter(_IMAGES))
        _IMAGES.pop(oldest, None)

    return ExtractImageResponse(document_id=document_id, width=width, height=height, regions=regions)


@app.post("/edit-image")
async def edit_image(
    document_id: str = Form(...),
    region_json: str = Form(...),
    new_text: str = Form(...),
):
    if document_id not in _IMAGES:
        raise HTTPException(404, "image not found or session expired — re-upload and extract again")

    import json
    region = ImageRegion(**json.loads(region_json))

    img = Image.open(io.BytesIO(_IMAGES[document_id])).convert("RGB")
    x0, y0, x1, y1 = region.bbox
    box_w, box_h = x1 - x0, y1 - y0

    # Sample background color from just outside the box (simple flat-fill
    # erase — no OpenCV inpainting needed, keeps this deployable on free tier).
    sample_points = [
        (max(x0 - 3, 0), y0), (min(x1 + 3, img.width - 1), y0),
        (x0, max(y0 - 3, 0)), (x0, min(y1 + 3, img.height - 1)),
    ]
    samples = [img.getpixel(p) for p in sample_points]
    bg_color = tuple(sum(c[i] for c in samples) // len(samples) for i in range(3))

    draw = ImageDraw.Draw(img)
    draw.rectangle([x0, y0, x1, y1], fill=bg_color)

    font_path = _resolve_font_path(region.bold, region.italic)
    font = _fit_pil_font(new_text, font_path, box_w, box_h, region.font_size_px)
    text_color = ai_vision.hex_to_rgb(region.color_hex)
    draw.text((x0, y0), new_text, font=font, fill=text_color)

    out = io.BytesIO()
    img.save(out, format="PNG")
    out.seek(0)

    return StreamingResponse(
        out,
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=edited.png"},
    )

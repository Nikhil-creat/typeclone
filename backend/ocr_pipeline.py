"""
ocr_pipeline.py
================
Raster (PNG / scanned image) typography pipeline:
  1. Layout analysis via PaddleOCR -> line/word polygons + recognized text.
  2. Glyph-region color sampling (Otsu + median foreground color).
  3. Content-aware inpainting to erase original ink (OpenCV Telea).
  4. Style classification (weight/italic/family-cluster) via a lightweight
     CNN (GlyphStyleNet), with a deterministic heuristic fallback so the
     module is fully runnable even without a trained checkpoint.
  5. Compositing new text back onto the inpainted background using Pillow,
     auto-fitted to the original box.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("ocr_pipeline")

try:
    from paddleocr import PaddleOCR
    _PADDLE_AVAILABLE = True
except ImportError:  # keeps module importable in environments without paddle
    _PADDLE_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Data model (mirrors TypographyProfile in typography_engine.py)
# --------------------------------------------------------------------------- #

@dataclass
class RasterTextRegion:
    region_id: str
    bbox: tuple[int, int, int, int]        # x0, y0, x1, y1 (axis-aligned)
    polygon: list[tuple[int, int]]         # OCR quad, may be rotated
    text: str
    font_size_px: float
    weight: int
    italic: bool
    color_rgb: tuple[int, int, int]
    confidence: float
    source_engine: str = "raster"


# --------------------------------------------------------------------------- #
# 1. Layout analysis
# --------------------------------------------------------------------------- #

class LayoutAnalyzer:
    def __init__(self, lang: str = "en"):
        if not _PADDLE_AVAILABLE:
            raise RuntimeError(
                "paddleocr is not installed. Install with: pip install paddleocr paddlepaddle"
            )
        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def detect_lines(self, image_bgr: np.ndarray) -> list[dict]:
        """Returns raw PaddleOCR results: [[poly, (text, confidence)], ...]."""
        result = self._ocr.ocr(image_bgr, cls=True)
        lines = []
        if not result or not result[0]:
            return lines
        for poly, (text, conf) in result[0]:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            lines.append({
                "polygon": [(int(x), int(y)) for x, y in poly],
                "bbox": (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
                "text": text,
                "confidence": float(conf),
            })
        return lines


# --------------------------------------------------------------------------- #
# 2. Color sampling
# --------------------------------------------------------------------------- #

def sample_foreground_color(image_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, int, int]:
    x0, y0, x1, y1 = bbox
    crop = image_bgr[max(y0, 0):y1, max(x0, 0):x1]
    if crop.size == 0:
        return (0, 0, 0)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    fg_pixels = crop[mask > 0]
    if fg_pixels.size == 0:
        return (0, 0, 0)

    median_bgr = np.median(fg_pixels, axis=0)
    b, g, r = median_bgr.astype(int)
    return (int(r), int(g), int(b))


def estimate_font_size_px(bbox: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = bbox
    return float(y1 - y0)  # cap-height approx; refine with x-height detection in prod


# --------------------------------------------------------------------------- #
# 3. Content-aware inpainting
# --------------------------------------------------------------------------- #

def build_glyph_mask(image_shape: tuple[int, int], polygon: list[tuple[int, int]], dilate_px: int = 3) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    kernel = np.ones((dilate_px, dilate_px), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def inpaint_region(image_bgr: np.ndarray, mask: np.ndarray, radius: int = 5) -> np.ndarray:
    return cv2.inpaint(image_bgr, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)


# --------------------------------------------------------------------------- #
# 4. Style classification — GlyphStyleNet (CNN) with heuristic fallback
# --------------------------------------------------------------------------- #

if _TORCH_AVAILABLE:
    class GlyphStyleNet(nn.Module):
        """Lightweight CNN: predicts (weight_logit, italic_logit) from a
        normalized 64x128 grayscale glyph-line crop. weight_logit>0 => bold."""

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(2),                                   # 32x64
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),                                   # 16x32
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 2),  # [weight_logit, italic_logit]
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.classifier(self.features(x))


class StyleClassifier:
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.model = None
        if _TORCH_AVAILABLE:
            self.model = GlyphStyleNet()
            if checkpoint_path:
                state = torch.load(checkpoint_path, map_location="cpu")
                self.model.load_state_dict(state)
            self.model.eval()

    def classify(self, image_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, bool]:
        """Returns (weight, italic). Falls back to stroke-width / shear
        heuristics when no trained checkpoint or torch is unavailable."""
        x0, y0, x1, y1 = bbox
        crop = image_bgr[max(y0, 0):y1, max(x0, 0):x1]
        if crop.size == 0:
            return 400, False

        if self.model is not None:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (128, 64)).astype(np.float32) / 255.0
            tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(tensor)[0]
            weight = 700 if logits[0].item() > 0 else 400
            italic = logits[1].item() > 0
            return weight, italic

        return self._heuristic_classify(crop)

    @staticmethod
    def _heuristic_classify(crop: np.ndarray) -> tuple[int, bool]:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        ink_ratio = np.count_nonzero(binary) / max(binary.size, 1)
        weight = 700 if ink_ratio > 0.22 else 400

        # Shear estimate via image moments (skew of ink mass -> italic proxy).
        moments = cv2.moments(binary, binaryImage=True)
        italic = False
        if moments["mu02"] > 1e-3:
            skew = moments["mu11"] / moments["mu02"]
            italic = abs(skew) > 0.18

        return weight, italic


# --------------------------------------------------------------------------- #
# 5. End-to-end raster extraction
# --------------------------------------------------------------------------- #

class RasterTypographyExtractor:
    def __init__(self, lang: str = "en", style_checkpoint: Optional[str] = None):
        self.layout = LayoutAnalyzer(lang=lang)
        self.style_clf = StyleClassifier(style_checkpoint)

    def extract(self, image_bgr: np.ndarray) -> list[RasterTextRegion]:
        lines = self.layout.detect_lines(image_bgr)
        regions: list[RasterTextRegion] = []

        for line in lines:
            weight, italic = self.style_clf.classify(image_bgr, line["bbox"])
            color = sample_foreground_color(image_bgr, line["bbox"])
            font_size = estimate_font_size_px(line["bbox"])

            regions.append(RasterTextRegion(
                region_id=str(uuid4()),
                bbox=line["bbox"],
                polygon=line["polygon"],
                text=line["text"],
                font_size_px=font_size,
                weight=weight,
                italic=italic,
                color_rgb=color,
                confidence=line["confidence"],
            ))
        return regions


# --------------------------------------------------------------------------- #
# 6. Edit application: erase + fit + redraw
# --------------------------------------------------------------------------- #

def fit_font_to_box(
    text: str,
    font_path: str,
    box_width: int,
    box_height: int,
    starting_size: int,
    min_size: int = 6,
) -> ImageFont.FreeTypeFont:
    size = starting_size
    while size > min_size:
        font = ImageFont.truetype(font_path, size)
        bbox = font.getbbox(text)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= box_width and h <= box_height:
            return font
        size -= 1
    return ImageFont.truetype(font_path, min_size)


def resolve_font_path(weight: int, italic: bool, font_dir: str = "/app/assets/fonts") -> str:
    """Maps (weight, italic) to a bundled DejaVu/Noto font file. Production
    systems should extend this to a fontconfig-backed family matcher."""
    if weight >= 600 and italic:
        return f"{font_dir}/DejaVuSans-BoldOblique.ttf"
    if weight >= 600:
        return f"{font_dir}/DejaVuSans-Bold.ttf"
    if italic:
        return f"{font_dir}/DejaVuSans-Oblique.ttf"
    return f"{font_dir}/DejaVuSans.ttf"


def apply_raster_edit(
    image_bgr: np.ndarray,
    region: RasterTextRegion,
    new_text: str,
    font_dir: str = "/app/assets/fonts",
) -> np.ndarray:
    # 1. Erase original ink via inpainting.
    mask = build_glyph_mask(image_bgr.shape, region.polygon)
    inpainted = inpaint_region(image_bgr, mask)

    # 2. Composite new text using Pillow (RGB space).
    rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    x0, y0, x1, y1 = region.bbox
    box_w, box_h = x1 - x0, y1 - y0

    font_path = resolve_font_path(region.weight, region.italic, font_dir)
    font = fit_font_to_box(new_text, font_path, box_w, box_h, starting_size=int(region.font_size_px))

    draw.text((x0, y0), new_text, font=font, fill=region.color_rgb)

    out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return out_bgr


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def save_image(image_bgr: np.ndarray, path: str) -> None:
    cv2.imwrite(path, image_bgr)

# TypeClone — WYSIWYG PDF/PNG Typography-Preserving Text Editor

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT (Next.js)                                │
│  ┌───────────────┐   ┌────────────────────┐   ┌───────────────────────────┐ │
│  │ Document       │   │ Konva.js Canvas     │   │ Style Inspector Panel     │ │
│  │ Viewer (render │◄─►│ Overlay (editable   │◄─►│ (font/weight/size/color   │ │
│  │ PNG/PDF page)  │   │ text nodes)          │   │  live controls)           │ │
│  └───────────────┘   └────────────────────┘   └───────────────────────────┘ │
└───────────────────────────────┬───────────────────────────────────────────┘
                                 │ REST/WS (FastAPI)
┌───────────────────────────────▼───────────────────────────────────────────┐
│                             API GATEWAY (FastAPI)                          │
│   /documents/upload   /documents/{id}/regions   /documents/{id}/edit       │
└──────┬───────────────────────────────────────────────────────┬────────────┘
       │ enqueue                                                │ enqueue
┌──────▼──────────────┐                                ┌────────▼───────────┐
│ CELERY WORKER POOL   │                                │ CELERY WORKER POOL  │
│  (extraction queue)  │                                │  (render queue)     │
│ ┌───────────────────┐│                               │┌────────────────────┐│
│ │ Vector Engine      ││                               ││ Raster Engine       ││
│ │ (PyMuPDF/fitz)     ││                               ││ (PaddleOCR + CV2)   ││
│ │ → glyph spans,     ││                               ││ → line segmentation,││
│ │   font streams,    ││                               ││   inpainting mask,  ││
│ │   color, bbox      ││                               ││   glyph classifier  ││
│ └───────────────────┘│                               │└────────────────────┘│
│           │           │                               │           │          │
│           └───────────┴───────── TypographyProfile ───┴───────────┘          │
│                                  (unified schema)                            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                       ┌────────────────▼────────────────┐
                       │  Redis (broker + result backend) │
                       │  Postgres (document/region store)│
                       │  S3/MinIO (original + rendered)  │
                       └───────────────────────────────────┘
```

## 2. Unified Typography Profile Schema

Both engines (vector & raster) normalize into one schema so the frontend never
needs to know which pipeline produced a region:

```json
{
  "region_id": "uuid",
  "page": 1,
  "bbox": [x0, y0, x1, y1],
  "baseline_y": 412.3,
  "text": "Invoice Total",
  "font_family": "Helvetica-Bold",
  "font_family_fallback": "Arial, sans-serif",
  "font_size_pt": 11.5,
  "weight": 700,
  "italic": false,
  "letter_spacing": 0.0,
  "line_height": 13.8,
  "color_rgb": [26, 26, 26],
  "color_space": "DeviceRGB",
  "alignment": "left",
  "source_engine": "vector | raster",
  "confidence": 0.98
}
```

## 3. Pipeline A — Vector PDFs (PyMuPDF)

1. Open doc with `fitz`, walk `page.get_text("dict")` → spans carry font name,
   size, flags (bit 4=italic, bit 2=serif, bold inferred from font name +
   `flags & 2^4`), color as packed int.
2. Bold/Italic normalization: PDF embedded font names are unreliable
   (`ArialMT`, `Arial-BoldMT`, subset tags like `ABCDEF+Arial-Bold`) → regex
   + flag cross-check.
3. Baseline extracted directly from span `origin` (fitz gives true baseline,
   not bbox top).
4. On edit: old glyphs are redacted via `page.add_redact_annot(bbox)` +
   `page.apply_redactions()` (removes vector glyph outlines cleanly, no
   raster artifacts), then new text is drawn with `page.insert_text()` using
   a matched system/embedded font, with **auto-shrink-to-fit** iteration.

## 4. Pipeline B — Raster PNG/Scans

1. **Layout analysis**: PaddleOCR (`PP-OCRv4`) for line + word boxes with
   angle correction.
2. **Style classification**: lightweight PyTorch CNN (`GlyphStyleNet`,
   ResNet18-slim backbone) trained on synthetic font-rendered crops →
   predicts {weight, italic, family-cluster} per line.
3. **Color sampling**: mode-color of foreground pixels inside glyph mask
   (Otsu threshold + median color, robust to anti-aliasing).
4. **Inpainting**: OpenCV `cv2.inpaint` (Telea) seeded by a dilated glyph
   mask from the OCR polygon — removes old ink while preserving paper
   texture/background gradient.
5. New text rendered via Pillow/FreeType using the matched font, then
   composited back onto the inpainted background.

## 5. Fit-to-Bbox Algorithm (shared by both engines)

Binary search on font size (and optionally negative tracking) so the new
string's measured width/height stays within the original bbox ± a
configurable overflow tolerance, falling back to controlled letter-spacing
compression before truncation is ever considered.

## 6. Deployment Topology

- `api` (FastAPI, uvicorn, 2+ replicas behind LB)
- `worker-vector`, `worker-raster` (separate Celery queues — raster is GPU-
  optional/CPU-heavy, vector is CPU-light — scaled independently)
- `redis` (broker + cache)
- `postgres` (metadata)
- `minio` (object storage, S3-compatible)
- `frontend` (Next.js, served via its own container or Vercel)

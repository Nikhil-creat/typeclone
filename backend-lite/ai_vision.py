"""
ai_vision.py
============
Uses the Gemini API (free tier) as a drop-in replacement for the heavy
local PaddleOCR + PyTorch raster pipeline. One multimodal request returns
every text region's content, position, and approximate style — no GPU,
no multi-GB model downloads, works on a free Render instance.

Trade-off vs. the local pipeline: bounding boxes and style guesses are
model-inferred, not pixel-measured, so they're approximate rather than
exact. Good enough for "click a line, replace it" editing; if you need
pixel-precise raster typography later, swap this module out for the
PaddleOCR pipeline in the full architecture and keep this same call
signature.

Requires: GEMINI_API_KEY environment variable (free at
https://aistudio.google.com/apikey).
"""

from __future__ import annotations

import base64
import json
import os
import re

import requests

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

_PROMPT = """Analyze this image and find every distinct piece of text (line or short phrase).

For each one, return an object with:
- "text": the exact text content
- "bbox": [ymin, xmin, ymax, xmax] as integers normalized to a 0-1000 scale (Gemini's standard object-detection coordinate format)
- "bold": true or false
- "italic": true or false
- "color_hex": your best estimate of the text's color as a hex string like "#1a1a1a"

Respond with ONLY a raw JSON array of these objects. No markdown fences, no explanation, no extra text."""


class GeminiNotConfigured(RuntimeError):
    pass


class GeminiRequestFailed(RuntimeError):
    pass


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def analyze_image(image_bytes: bytes, mime_type: str = "image/png") -> list[dict]:
    """Sends the image to Gemini and returns a list of detected text regions
    with normalized bbox (0-1000 scale, [ymin,xmin,ymax,xmax]), bold, italic,
    and color_hex fields. Raises GeminiNotConfigured / GeminiRequestFailed on
    problems so the caller can surface a clear error to the client."""

    if not GEMINI_API_KEY:
        raise GeminiNotConfigured(
            "GEMINI_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/apikey and set it as an env var."
        )

    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "contents": [{
            "parts": [
                {"text": _PROMPT},
                {"inline_data": {"mime_type": mime_type, "data": b64_image}},
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }

    try:
        resp = requests.post(
            GEMINI_URL,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=60,
        )
    except requests.RequestException as e:
        raise GeminiRequestFailed(f"could not reach Gemini API: {e}") from e

    if resp.status_code != 200:
        raise GeminiRequestFailed(f"Gemini API returned {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    try:
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise GeminiRequestFailed(f"unexpected Gemini response shape: {data}") from e

    cleaned = _strip_code_fences(raw_text)
    try:
        regions = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise GeminiRequestFailed(f"Gemini did not return valid JSON: {cleaned[:500]}") from e

    if not isinstance(regions, list):
        raise GeminiRequestFailed(f"expected a JSON array, got: {type(regions)}")

    return regions


def denormalize_bbox(bbox_1000: list[int], img_width: int, img_height: int) -> tuple[int, int, int, int]:
    """Converts Gemini's [ymin, xmin, ymax, xmax] on a 0-1000 scale into
    real pixel coordinates (x0, y0, x1, y1) for the given image size."""
    ymin, xmin, ymax, xmax = bbox_1000
    x0 = int(xmin / 1000 * img_width)
    y0 = int(ymin / 1000 * img_height)
    x1 = int(xmax / 1000 * img_width)
    y1 = int(ymax / 1000 * img_height)
    return x0, y0, x1, y1


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = (hex_color or "#000000").lstrip("#")
    if len(h) != 6:
        return (0, 0, 0)
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore
    except ValueError:
        return (0, 0, 0)

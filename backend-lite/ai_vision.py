"""
ai_vision.py
============
Multi-provider vision-LLM text/style detection for raster images.
Default provider: Groq (meta-llama/llama-4-scout-17b-16e-instruct) —
fast, free-tier friendly, OpenAI-compatible API. Gemini remains available
as a fallback provider for anyone who prefers it.

Two ways to supply credentials:
  1. Backend env var (GROQ_API_KEY / GEMINI_API_KEY) — most secure, key
     never leaves the server.
  2. Per-request override via the api_key argument, sourced from a request
     header on the client's own deployment — convenience for personal/solo
     use where you control both ends. If you use this path, understand the
     key is visible in your browser's network tab and localStorage; don't
     share the page publicly with the key baked in.

Both paths normalize to the same output schema regardless of provider:
list of {text, bbox (0-1000 normalized [ymin,xmin,ymax,xmax]), bold,
italic, color_hex}.
"""

from __future__ import annotations

import base64
import json
import os
import re

import requests

DEFAULT_PROVIDER = os.environ.get("AI_PROVIDER", "groq")  # "groq" | "gemini"

GROQ_API_KEY_ENV = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

GEMINI_API_KEY_ENV = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

_PROMPT = """Analyze this image and find every distinct piece of text (line or short phrase).

For each one, return an object with:
- "text": the exact text content
- "bbox": [ymin, xmin, ymax, xmax] as integers normalized to a 0-1000 scale (top-left origin, 0-1000 covers the full image in each dimension)
- "bold": true or false
- "italic": true or false
- "color_hex": your best estimate of the text's color as a hex string like "#1a1a1a"

Respond with ONLY a raw JSON array of these objects. No markdown fences, no explanation, no extra text. If you cannot find any text, return an empty array []."""


class AIVisionNotConfigured(RuntimeError):
    pass


class AIVisionRequestFailed(RuntimeError):
    pass


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _parse_json_array(raw_text: str, provider: str) -> list[dict]:
    cleaned = _strip_code_fences(raw_text)
    # Some models wrap the array in prose despite instructions — grab the
    # first [...] block as a fallback.
    if not cleaned.startswith("["):
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
    try:
        regions = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise AIVisionRequestFailed(f"{provider} did not return valid JSON: {cleaned[:500]}") from e
    if not isinstance(regions, list):
        raise AIVisionRequestFailed(f"expected a JSON array from {provider}, got: {type(regions)}")
    return regions


def _analyze_with_groq(image_bytes: bytes, mime_type: str, api_key: str) -> list[dict]:
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:{mime_type};base64,{b64_image}"

    payload = {
        "model": GROQ_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }],
        "temperature": 0.1,
        "max_completion_tokens": 4096,
    }

    try:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
    except requests.RequestException as e:
        raise AIVisionRequestFailed(f"could not reach Groq API: {e}") from e

    if resp.status_code != 200:
        raise AIVisionRequestFailed(f"Groq API returned {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    try:
        raw_text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise AIVisionRequestFailed(f"unexpected Groq response shape: {data}") from e

    return _parse_json_array(raw_text, "Groq")


def _analyze_with_gemini(image_bytes: bytes, mime_type: str, api_key: str) -> list[dict]:
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "contents": [{
            "parts": [
                {"text": _PROMPT},
                {"inline_data": {"mime_type": mime_type, "data": b64_image}},
            ]
        }],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
    }

    try:
        resp = requests.post(GEMINI_URL, params={"key": api_key}, json=payload, timeout=60)
    except requests.RequestException as e:
        raise AIVisionRequestFailed(f"could not reach Gemini API: {e}") from e

    if resp.status_code != 200:
        raise AIVisionRequestFailed(f"Gemini API returned {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    try:
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise AIVisionRequestFailed(f"unexpected Gemini response shape: {data}") from e

    return _parse_json_array(raw_text, "Gemini")


def analyze_image(
    image_bytes: bytes,
    mime_type: str = "image/png",
    provider: str | None = None,
    api_key: str | None = None,
) -> list[dict]:
    """Sends the image to the chosen vision LLM (Groq by default) and
    returns detected text regions. `provider`/`api_key` override the
    server's default provider/env-var key for this one request — pass
    None to use server defaults."""

    provider = (provider or DEFAULT_PROVIDER).lower().strip()

    if provider == "groq":
        key = api_key or GROQ_API_KEY_ENV
        if not key:
            raise AIVisionNotConfigured(
                "No Groq API key available. Get a free key at "
                "https://console.groq.com/keys and either set GROQ_API_KEY "
                "on the backend, or save it in the editor's API key field."
            )
        return _analyze_with_groq(image_bytes, mime_type, key)

    if provider == "gemini":
        key = api_key or GEMINI_API_KEY_ENV
        if not key:
            raise AIVisionNotConfigured(
                "No Gemini API key available. Get a free key at "
                "https://aistudio.google.com/apikey and either set GEMINI_API_KEY "
                "on the backend, or save it in the editor's API key field."
            )
        return _analyze_with_gemini(image_bytes, mime_type, key)

    raise AIVisionNotConfigured(f"Unknown provider '{provider}' — use 'groq' or 'gemini'.")


def denormalize_bbox(bbox_1000: list[int], img_width: int, img_height: int) -> tuple[int, int, int, int]:
    """Converts a [ymin, xmin, ymax, xmax] box on a 0-1000 scale into real
    pixel coordinates (x0, y0, x1, y1) for the given image size."""
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

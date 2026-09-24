# TypeClone Lite — Setup Notes

## Two document types, two engines

| Mode | Engine | Precision | Needs |
|---|---|---|---|
| PDF | PyMuPDF (local) | Exact — reads real embedded font data | nothing extra |
| Image (PNG/JPG) | Gemini API | Approximate — AI-estimated from pixels | `GEMINI_API_KEY` |

Image mode replaces the heavy PaddleOCR + PyTorch pipeline from the full
architecture with a single Gemini API call — free, no GPU, deployable on
Render's free tier. The trade-off: Gemini's bounding boxes and style
guesses (bold/italic/color) are inferred by the model, not measured
pixel-by-pixel, so treat them as "close enough to click and replace,"
not pixel-perfect.

## Getting a free Gemini API key

1. Go to **https://aistudio.google.com/apikey**
2. Sign in with any Google account.
3. Click **Create API key** → copy it.

## Setting it on Render

1. Your Render service → **Environment** tab.
2. Add environment variable: `GEMINI_API_KEY` = *(paste your key)*.
3. Save — Render redeploys automatically.

(If you used the included `render.yaml` blueprint, Render will prompt
you for this value during setup instead.)

## Fonts required for image-mode editing

Image mode redraws replacement text using Pillow, which needs a real
`.ttf` font file (not Pillow's built-in bitmap font, which can't be
resized or styled). The app looks for fonts in this order:

1. `backend-lite/assets/fonts/DejaVuSans*.ttf` in your repo
2. Common system paths (`/usr/share/fonts/truetype/dejavu/...`,
   `/usr/share/fonts/truetype/liberation/...`) — present on some hosts,
   not guaranteed on Render's Python runtime.

**If image editing fails with a "No TTF font found" error:**

1. Download the DejaVu font family (free, open license) from
   `https://dejavu-fonts.github.io/` on your phone.
2. You need these 4 files: `DejaVuSans.ttf`, `DejaVuSans-Bold.ttf`,
   `DejaVuSans-Oblique.ttf`, `DejaVuSans-BoldOblique.ttf`.
3. In your repo, create `backend-lite/assets/fonts/` and put the 4 files
   there.
4. Commit and push — Render redeploys with the fonts included.

PDF mode doesn't need this — PyMuPDF has built-in Base-14 fonts (Helvetica
family) it draws with directly, no font files required.

## Known limitations of this "lite" build

- **Erase method for images** is a flat background-color fill (sampled
  from just outside the text box), not true content-aware inpainting —
  fine for solid/simple backgrounds, visibly imperfect on gradients,
  textures, or photos behind text.
- **In-memory sessions**: uploaded documents live in server RAM and are
  lost on redeploy/restart, and evicted after ~100-200 concurrent
  documents. Fine for personal/demo use; not meant for production
  traffic.
- **No undo/redo, no collaboration, no task queue** — those exist in the
  full architecture (`/backend` + `docker-compose.yml`) if you later
  deploy that version to a real container host.

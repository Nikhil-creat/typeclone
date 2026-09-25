# TypeClone Lite — Setup Notes

## Three document types, three engines

| Mode | Engine | Precision | Editing UX | Needs |
|---|---|---|---|---|
| PDF | PyMuPDF (local) | Exact — reads real embedded font data | Tap directly on the rendered page | nothing extra |
| Image (PNG/JPG) | Groq or Gemini vision LLM | Approximate — AI-estimated from pixels | Tap directly on the uploaded image | an API key (see below) |
| Word (.docx) | python-docx (local) | Exact — reads real run-level styling | List of text runs (docx has no fixed layout to tap on) | nothing extra |

## Getting a free Groq API key (default provider)

1. Go to **https://console.groq.com/keys**
2. Sign in (free).
3. **Create API Key** → copy it.

Gemini remains available as an alternate provider (switch in the dropdown in
the editor) — get a free key at **https://aistudio.google.com/apikey** if
you prefer it.

## Two ways to supply the key — pick one

**Option A — Save it in the browser (convenient, for personal use)**
Paste the key into the editor's "AI provider & API key" box and hit **Save
API Key**. It's stored in that browser's `localStorage` and sent as a
request header (`X-Ai-Api-Key`) directly to *your own* backend on every
image-mode request. Nothing touches a third-party server. The trade-off:
anyone with access to that browser, or anyone you share a screen-recording
with, could see it in devtools. Fine for your own use; don't do this on a
shared/public device.

**Option B — Set it on the backend (more secure, for sharing the tool)**
Render dashboard → your service → **Environment** tab → add `GROQ_API_KEY`
(or `GEMINI_API_KEY`) → save. The key never appears in the browser at all.
Leave the in-page key field empty — the backend falls back to its own env
var automatically. Use this if you're going to share the editor's link with
anyone else.

You can also set `AI_PROVIDER=groq` (or `gemini`) as a backend env var to
change the server-side default; the in-page dropdown overrides it per
request either way.

## Fonts required for image-mode editing

Image mode redraws replacement text using Pillow, which needs a real
`.ttf` font file. The app looks for fonts in this order:

1. `backend-lite/assets/fonts/DejaVuSans*.ttf` in your repo
2. Common system paths (`/usr/share/fonts/truetype/dejavu/...`,
   `/usr/share/fonts/truetype/liberation/...`)

**If image editing fails with "No TTF font found":**
Download the DejaVu family from `https://dejavu-fonts.github.io/` (4 files:
`DejaVuSans.ttf`, `-Bold.ttf`, `-Oblique.ttf`, `-BoldOblique.ttf`), place
them in `backend-lite/assets/fonts/` in your repo, commit, push.

PDF and DOCX modes don't need this — PyMuPDF and Word both carry their own
font handling.

## How tap-to-edit works

- **PDF**: the backend renders each page to a PNG and sends it to the
  browser alongside each text region's position (as a percentage of the
  page, so it lines up regardless of screen size). Tap a highlighted box →
  a small popover opens with the current text → Save sends the edit, the
  backend redraws that region, and the browser automatically re-analyzes
  the result so the canvas reflects your change immediately.
- **Image**: identical flow, using the AI provider's detected regions
  instead of embedded font data.
- **Word (.docx)**: has no fixed pixel layout (text reflows), so there's
  no canvas — instead you get a tappable list of text runs, each carrying
  its real bold/italic/font/size/color from the document.
- Every edit updates an in-memory "current file" in the browser tab; use
  **Download edited file** whenever you want to save your progress. Edits
  are not auto-downloaded on every tap — only final export is.

## Known limitations of this "lite" build

- **Erase method for images** is a flat background-color fill, not true
  content-aware inpainting — fine on solid backgrounds, visibly imperfect
  on gradients/photos behind text.
- **In-memory sessions**: uploaded documents live in server RAM and are
  lost on redeploy/restart. Fine for personal/demo use.
- **PDF canvas renders one page image per page at extract time** — large,
  many-page PDFs will be slower to analyze and heavier to transfer than a
  single-page document. Fine for typical documents (letters, invoices,
  forms); reconsider for 100+ page files.
- **No undo/redo, no collaboration, no task queue** — those exist in the
  full architecture (`/backend` + `docker-compose.yml`) if you later
  deploy that version to a real container host.

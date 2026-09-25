# TypeClone

> **DESIGNED &amp; DEVELOPED BY NIKHIL CHARY SRIRAMOJU**
> [GitHub](https://github.com/Nikhil-creat) · [LinkedIn](https://in.linkedin.com/in/nikhil-chary-sriramoju-95041b38a) · [sriramojunikhil66@gmail.com](mailto:sriramojunikhil66@gmail.com)

**Live:** [Docs & Demo](https://nikhil-creat.github.io/typeclone/) · [Live Editor](https://nikhil-creat.github.io/typeclone/app.html)

Enterprise-grade PDF &amp; PNG typography-preserving text editor. Detects,
extracts, and clones font weight, style, baseline, and color across vector
PDFs (PyMuPDF) and raster scans (PaddleOCR + OpenCV), then lets you edit
text inline in a Konva.js canvas with pixel-faithful re-rendering.

Full architecture breakdown: [`ARCHITECTURE.md`](./ARCHITECTURE.md)

## Repository layout

```
backend/            FastAPI gateway, Celery tasks, CV/OCR/PDF engines
  typography_engine.py   Vector (PDF) extraction + style-cloned redraw
  ocr_pipeline.py         Raster (PNG) OCR + inpainting + CNN style classifier
  font_matcher.py         Visual-metrics font similarity matching
  edit_history.py         Non-destructive undo/redo ledger
  main.py                 API routes, Celery tasks, WebSocket live channel
frontend/           Konva.js React hook for the WYSIWYG canvas overlay
docker/              Dockerfile.backend (multi-stage, FreeType/Cairo/OpenCV)
docker-compose.yml   Full local stack: api, workers, redis, postgres, minio
docs/                Static docs/demo site — this is what GitHub Pages serves
```

## Important: what GitHub Pages can and can't host

GitHub Pages only serves **static files** (HTML/CSS/JS). It cannot run the
FastAPI backend, Celery workers, PaddleOCR, or PyTorch — those need a real
container host. This repo is split accordingly:

- **`/docs`** → a static documentation + interactive demo page. This is what
  you'll publish to GitHub Pages (instructions below).
- **`/backend` + `docker-compose.yml`** → the actual engine. Deploy this to
  any Docker-capable host: Render, Railway, Fly.io, a VPS, or your own
  Kubernetes cluster. Point the frontend's `NEXT_PUBLIC_API_URL` at that
  host once it's live.

## Run the full engine locally

```bash
docker compose up --build
# API      → http://localhost:8000
# Frontend → http://localhost:3000
# MinIO console → http://localhost:9001
```

---

## Uploading this to GitHub

1. **Extract the zip** you downloaded, into any folder on your machine.
2. **Create a new repository** on GitHub (github.com → the `+` icon top
   right → *New repository*). Name it, e.g., `typeclone`. Leave it empty —
   don't initialize with a README, .gitignore, or license (this repo
   already has them).
3. Open a terminal **inside the extracted folder** and run:

   ```bash
   git init
   git add .
   git commit -m "Initial commit: TypeClone typography engine"
   git branch -M main
   git remote add origin https://github.com/<your-username>/typeclone.git
   git push -u origin main
   ```

   (If you don't have `git` installed, or aren't signed in, GitHub Desktop
   is the easiest alternative: File → Add Local Repository → select the
   extracted folder → Publish repository.)

## Publishing the docs site to GitHub Pages

This repo ships a workflow at `.github/workflows/deploy-pages.yml` that
auto-publishes the `/docs` folder whenever you push to `main`.

1. In your new GitHub repo, go to **Settings → Pages**.
2. Under **Build and deployment → Source**, choose **GitHub Actions**
   (not "Deploy from a branch").
3. Push any commit that touches `docs/` (your initial push already does) —
   this triggers the **Deploy Docs to GitHub Pages** workflow. Watch it run
   under the **Actions** tab.
4. Once it finishes (green check), your site is live at:

   ```
   https://<your-username>.github.io/typeclone/
   ```

5. Before or after publishing, open `docs/index.html` and replace the two
   `OWNER/REPO` placeholders in the architecture link with your actual
   `<your-username>/typeclone`, then commit and push — the workflow will
   redeploy automatically.

## Deploying the actual backend engine

Pages only gives you the demo/docs site above. To make the editor
functional end-to-end, deploy `docker-compose.yml` to a container host:

- **Render / Railway**: point a new "Web Service" at this repo, set the
  Dockerfile path to `docker/Dockerfile.backend`, and add a managed Redis
  + Postgres addon (swap the `REDIS_URL`/`DATABASE_URL` env vars
  accordingly).
- **A VPS (DigitalOcean, Hetzner, etc.)**: `git clone` the repo on the
  server and run `docker compose up -d --build` directly — this is the
  closest to the local setup and needs no code changes.
- **Fly.io**: `fly launch` from the repo root, pointing at
  `docker/Dockerfile.backend`; add a `fly.toml` with the exposed port
  `8000` and attach Fly's managed Redis.

Once the API is live at a public URL, update `frontend/useTypographyCanvas.ts`'s
consumer (your Next.js app) to point `NEXT_PUBLIC_API_URL` at it, and
redeploy the frontend (Vercel is the simplest option for a Next.js app).

## Author

**DESIGNED AND DEVELOPED BY NIKHIL CHARY SRIRAMOJU**

- GitHub: [github.com/Nikhil-creat](https://github.com/Nikhil-creat)
- LinkedIn: [in.linkedin.com/in/nikhil-chary-sriramoju-95041b38a](https://in.linkedin.com/in/nikhil-chary-sriramoju-95041b38a)
- Email: [sriramojunikhil66@gmail.com](mailto:sriramojunikhil66@gmail.com)

## License

MIT — see [`LICENSE`](./LICENSE).

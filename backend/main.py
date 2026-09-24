"""
main.py — FastAPI Gateway
=========================
Exposes document upload, region extraction, and text-edit endpoints.
Heavy CV/OCR/PDF work is dispatched to Celery workers on dedicated queues
(vector-extract, raster-extract, render) so the API stays responsive.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import uuid
from typing import Optional

from celery import Celery
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from typography_engine import (
    TypographyProfile,
    edit_document_text,
    extract_document_typography,
)
from ocr_pipeline import (
    RasterTypographyExtractor,
    apply_raster_edit,
    load_image,
    save_image,
)
from edit_history import history_registry
from font_matcher import FontLibrary

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
STORAGE_DIR = os.environ.get("STORAGE_DIR", "/app/storage")
FONT_DIR = os.environ.get("FONT_DIR", "/app/assets/fonts")
os.makedirs(STORAGE_DIR, exist_ok=True)

_font_library: Optional[FontLibrary] = None


def get_font_library() -> FontLibrary:
    global _font_library
    if _font_library is None:
        _font_library = FontLibrary(FONT_DIR)
    return _font_library

celery_app = Celery("typeclone", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.task_routes = {
    "tasks.extract_vector": {"queue": "vector-extract"},
    "tasks.extract_raster": {"queue": "raster-extract"},
    "tasks.render_edit": {"queue": "render"},
}

class ConnectionManager:
    """Tracks active WebSocket clients per document_id and broadcasts
    live task-status / collaborative-edit events to all of them — powers
    real-time multi-user editing and progress indicators without polling."""

    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, document_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.setdefault(document_id, []).append(ws)

    def disconnect(self, document_id: str, ws: WebSocket) -> None:
        conns = self._connections.get(document_id, [])
        if ws in conns:
            conns.remove(ws)

    async def broadcast(self, document_id: str, message: dict) -> None:
        for ws in list(self._connections.get(document_id, [])):
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(document_id, ws)


ws_manager = ConnectionManager()

app = FastAPI(title="TypeClone — Typography-Preserving Document Editor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    doc_type: str  # "pdf" | "png"


class ExtractionTaskResponse(BaseModel):
    task_id: str
    document_id: str


class EditRequest(BaseModel):
    document_id: str
    doc_type: str
    region: dict            # serialized TypographyProfile / RasterTextRegion
    new_text: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None


class UndoRedoResponse(BaseModel):
    op_id: Optional[str] = None
    region_id: Optional[str] = None
    text: Optional[str] = None
    can_undo: bool = False
    can_redo: bool = False


class FontMatchRequest(BaseModel):
    weight: int
    italic: bool
    x_height_hint: float = 0.72
    top_k: int = 3


class FontMatchResponse(BaseModel):
    matches: list[dict]


# --------------------------------------------------------------------------- #
# Celery tasks
# --------------------------------------------------------------------------- #

@celery_app.task(name="tasks.extract_vector")
def task_extract_vector(document_path: str) -> dict:
    profiles_by_page = extract_document_typography(document_path)
    return {
        str(page): [p.to_dict() for p in profiles]
        for page, profiles in profiles_by_page.items()
    }


@celery_app.task(name="tasks.extract_raster")
def task_extract_raster(image_path: str) -> dict:
    extractor = RasterTypographyExtractor()
    image = load_image(image_path)
    regions = extractor.extract(image)
    return {"regions": [r.__dict__ for r in regions]}


@celery_app.task(name="tasks.render_edit")
def task_render_edit(document_id: str, doc_type: str, region: dict, new_text: str) -> dict:
    src_path = os.path.join(STORAGE_DIR, f"{document_id}.{doc_type}")
    out_id = str(uuid.uuid4())
    out_path = os.path.join(STORAGE_DIR, f"{out_id}.{doc_type}")

    if doc_type == "pdf":
        profile = TypographyProfile(**region)
        edit_document_text(src_path, out_path, profile, new_text)
    elif doc_type == "png":
        from ocr_pipeline import RasterTextRegion
        raster_region = RasterTextRegion(**region)
        image = load_image(src_path)
        edited = apply_raster_edit(image, raster_region, new_text)
        save_image(edited, out_path)
    else:
        raise ValueError(f"unsupported doc_type: {doc_type}")

    return {"output_document_id": out_id, "path": out_path}


# --------------------------------------------------------------------------- #
# API routes
# --------------------------------------------------------------------------- #

@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    ext = (file.filename or "").split(".")[-1].lower()
    if ext not in ("pdf", "png"):
        raise HTTPException(400, "only .pdf and .png are supported")

    document_id = str(uuid.uuid4())
    dest_path = os.path.join(STORAGE_DIR, f"{document_id}.{ext}")

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return UploadResponse(document_id=document_id, filename=file.filename, doc_type=ext)


@app.post("/documents/{document_id}/extract", response_model=ExtractionTaskResponse)
async def extract_typography(document_id: str, doc_type: str):
    src_path = os.path.join(STORAGE_DIR, f"{document_id}.{doc_type}")
    if not os.path.exists(src_path):
        raise HTTPException(404, "document not found")

    if doc_type == "pdf":
        async_result = task_extract_vector.delay(src_path)
    elif doc_type == "png":
        async_result = task_extract_raster.delay(src_path)
    else:
        raise HTTPException(400, "doc_type must be 'pdf' or 'png'")

    return ExtractionTaskResponse(task_id=async_result.id, document_id=document_id)


@app.post("/documents/edit", response_model=ExtractionTaskResponse)
async def edit_text_region(req: EditRequest, background_tasks: BackgroundTasks):
    src_path = os.path.join(STORAGE_DIR, f"{req.document_id}.{req.doc_type}")
    if not os.path.exists(src_path):
        raise HTTPException(404, "document not found")

    # Record into the non-destructive edit ledger before dispatching render.
    old_text = req.region.get("text", "")
    region_id = req.region.get("region_id", "")
    history = history_registry.get(req.document_id)
    history.record(region_id=region_id, old_text=old_text, new_text=req.new_text)

    async_result = task_render_edit.delay(
        req.document_id, req.doc_type, req.region, req.new_text
    )
    await ws_manager.broadcast(req.document_id, {
        "event": "edit_queued",
        "task_id": async_result.id,
        "region_id": region_id,
    })
    background_tasks.add_task(_poll_and_broadcast_task, req.document_id, async_result.id)
    return ExtractionTaskResponse(task_id=async_result.id, document_id=req.document_id)


@app.post("/documents/{document_id}/undo", response_model=UndoRedoResponse)
async def undo_edit(document_id: str):
    history = history_registry.get(document_id)
    op = history.undo()
    if op is None:
        return UndoRedoResponse(can_undo=False, can_redo=history.can_redo())
    await ws_manager.broadcast(document_id, {"event": "undo", "region_id": op.region_id, "text": op.old_text})
    return UndoRedoResponse(
        op_id=op.op_id, region_id=op.region_id, text=op.old_text,
        can_undo=history.can_undo(), can_redo=history.can_redo(),
    )


@app.post("/documents/{document_id}/redo", response_model=UndoRedoResponse)
async def redo_edit(document_id: str):
    history = history_registry.get(document_id)
    op = history.redo()
    if op is None:
        return UndoRedoResponse(can_undo=history.can_undo(), can_redo=False)
    await ws_manager.broadcast(document_id, {"event": "redo", "region_id": op.region_id, "text": op.new_text})
    return UndoRedoResponse(
        op_id=op.op_id, region_id=op.region_id, text=op.new_text,
        can_undo=history.can_undo(), can_redo=history.can_redo(),
    )


@app.get("/documents/{document_id}/timeline")
async def get_timeline(document_id: str):
    history = history_registry.get(document_id)
    return {"ops": [op.__dict__ for op in history.timeline()]}


@app.post("/fonts/match", response_model=FontMatchResponse)
async def match_font(req: FontMatchRequest):
    library = get_font_library()
    results = library.nearest_by_profile(
        weight=req.weight, italic=req.italic, x_height_hint=req.x_height_hint, top_k=req.top_k
    )
    return FontMatchResponse(matches=[{"family": name, "similarity": sim} for name, sim in results])


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    result = celery_app.AsyncResult(task_id)
    return TaskStatusResponse(
        task_id=task_id,
        status=result.status,
        result=result.result if result.ready() and result.successful() else None,
    )


@app.websocket("/ws/documents/{document_id}")
async def document_ws(websocket: WebSocket, document_id: str):
    """Live channel for a document: pushes render-task progress, undo/redo
    events, and (in a multi-user deployment) peer cursor/selection updates
    from other collaborators editing the same document."""
    await ws_manager.connect(document_id, websocket)
    try:
        while True:
            # Clients may send lightweight presence pings / cursor positions;
            # we simply rebroadcast them to other collaborators on the doc.
            msg = await websocket.receive_json()
            await ws_manager.broadcast(document_id, {"event": "peer_update", **msg})
    except WebSocketDisconnect:
        ws_manager.disconnect(document_id, websocket)


async def _poll_and_broadcast_task(document_id: str, task_id: str) -> None:
    """Background poller: watches a Celery AsyncResult and pushes a single
    completion/failure event over the document's WebSocket channel, so the
    client never has to poll /tasks/{task_id} manually."""
    result = celery_app.AsyncResult(task_id)
    for _ in range(600):  # up to ~5 min at 0.5s interval
        if result.ready():
            await ws_manager.broadcast(document_id, {
                "event": "render_complete" if result.successful() else "render_failed",
                "task_id": task_id,
                "result": result.result if result.successful() else str(result.result),
            })
            return
        await asyncio.sleep(0.5)


@app.get("/health")
async def health():
    return {"status": "ok"}

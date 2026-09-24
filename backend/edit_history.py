"""
edit_history.py
================
Non-destructive edit ledger. Every text mutation is stored as a diff
(region_id, old_text, new_text, style_delta) against a document version
chain, enabling full undo/redo without re-running extraction, and giving
the frontend a scrubbable timeline of changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4


@dataclass
class EditOp:
    op_id: str
    region_id: str
    old_text: str
    new_text: str
    timestamp: str
    style_delta: dict = field(default_factory=dict)  # e.g. {"font_size_pt": [11.5, 10.2]}


class EditHistory:
    """Per-document undo/redo stack. In production this is backed by
    Postgres (one row per op, indexed by document_id) — this in-memory
    version is the authoritative data structure either way."""

    def __init__(self, document_id: str):
        self.document_id = document_id
        self._undo_stack: list[EditOp] = []
        self._redo_stack: list[EditOp] = []

    def record(
        self,
        region_id: str,
        old_text: str,
        new_text: str,
        style_delta: Optional[dict] = None,
    ) -> EditOp:
        op = EditOp(
            op_id=str(uuid4()),
            region_id=region_id,
            old_text=old_text,
            new_text=new_text,
            timestamp=datetime.now(timezone.utc).isoformat(),
            style_delta=style_delta or {},
        )
        self._undo_stack.append(op)
        self._redo_stack.clear()  # new edit invalidates redo chain
        return op

    def undo(self) -> Optional[EditOp]:
        if not self._undo_stack:
            return None
        op = self._undo_stack.pop()
        self._redo_stack.append(op)
        return op

    def redo(self) -> Optional[EditOp]:
        if not self._redo_stack:
            return None
        op = self._redo_stack.pop()
        self._undo_stack.append(op)
        return op

    def timeline(self) -> list[EditOp]:
        return list(self._undo_stack)

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)


class HistoryRegistry:
    """Keeps one EditHistory per open document, keyed by document_id."""

    def __init__(self):
        self._histories: dict[str, EditHistory] = {}

    def get(self, document_id: str) -> EditHistory:
        if document_id not in self._histories:
            self._histories[document_id] = EditHistory(document_id)
        return self._histories[document_id]


history_registry = HistoryRegistry()

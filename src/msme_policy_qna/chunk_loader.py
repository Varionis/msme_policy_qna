from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from .config import CHUNKS_PATH


def _join_list(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def load_chunk_payload(path: Path = CHUNKS_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def chunk_to_document(chunk: dict[str, Any]) -> Document:
    metadata = {
        "chunk_id": chunk.get("chunk_id", ""),
        "source_type": chunk.get("source_type", ""),
        "role": chunk.get("role") or "",
        "document_id": chunk.get("document_id", ""),
        "scheme_id": chunk.get("scheme_id", ""),
        "section_title": chunk.get("section_title") or "",
        "clause_number": chunk.get("clause_number") or "",
        "page_numbers": _join_list(chunk.get("page_numbers")),
        "paragraph_orders": _join_list(chunk.get("paragraph_orders")),
        "table_index": "" if chunk.get("table_index") is None else str(chunk.get("table_index")),
    }
    return Document(page_content=chunk.get("text", ""), metadata=metadata)


def load_documents(path: Path = CHUNKS_PATH) -> list[Document]:
    payload = load_chunk_payload(path)
    chunks = payload.get("chunks", [])
    return [chunk_to_document(chunk) for chunk in chunks if chunk.get("text")]


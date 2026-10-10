from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma

from .chunk_loader import load_documents
from .config import CHUNKS_PATH, COLLECTION_NAME, VECTOR_DB_DIR
from .embeddings import get_embeddings


def get_vector_store(persist_directory: Path = VECTOR_DB_DIR) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_directory),
    )


def build_vector_store(
    chunks_path: Path = CHUNKS_PATH,
    persist_directory: Path = VECTOR_DB_DIR,
    reset_collection: bool = True,
) -> Chroma:
    documents = load_documents(chunks_path)
    vector_store = get_vector_store(persist_directory)

    if reset_collection:
        existing = vector_store.get(include=[])
        existing_ids = existing.get("ids", [])
        if existing_ids:
            vector_store.delete(ids=existing_ids)

    ids = [doc.metadata["chunk_id"] for doc in documents]
    vector_store.add_documents(documents=documents, ids=ids)
    return vector_store


def vector_store_has_documents(persist_directory: Path = VECTOR_DB_DIR) -> bool:
    vector_store = get_vector_store(persist_directory)
    return bool(vector_store.get(include=[], limit=1).get("ids"))


def ensure_vector_store(
    chunks_path: Path = CHUNKS_PATH,
    persist_directory: Path = VECTOR_DB_DIR,
) -> Chroma:
    vector_store = get_vector_store(persist_directory)
    if not vector_store.get(include=[], limit=1).get("ids"):
        return build_vector_store(
            chunks_path=chunks_path,
            persist_directory=persist_directory,
            reset_collection=False,
        )
    return vector_store


def main() -> None:
    documents = load_documents(CHUNKS_PATH)
    build_vector_store()
    print(f"Indexed {len(documents)} chunks into {VECTOR_DB_DIR}")


if __name__ == "__main__":
    main()

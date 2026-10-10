from pathlib import Path

from src.msme_policy_qna.chunk_loader import load_documents


def test_load_documents_preserves_metadata():
    documents = load_documents(Path("data/chunks_sample.json"))

    assert len(documents) == 133
    assert documents[0].page_content
    assert documents[0].metadata["chunk_id"].startswith("chunk_")
    assert "page_numbers" in documents[0].metadata


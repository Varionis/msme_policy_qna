from __future__ import annotations

import os
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CHUNKS_PATH = DATA_DIR / "chunks_sample.json"
EVAL_QUESTIONS_PATH = DATA_DIR / "eval_questions.json"
DEFAULT_VECTOR_DB_DIR = Path(tempfile.gettempdir()) / "msme_policy_qna" / "vector_db" / "chroma"
VECTOR_DB_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", DEFAULT_VECTOR_DB_DIR))
COLLECTION_NAME = "msme_policy_chunks"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RETRIEVAL_K = 5
SOURCE_PDF_URL = "https://www.dcmsme.gov.in/schemes/New-Guidelines.pdf"

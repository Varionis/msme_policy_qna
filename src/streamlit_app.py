from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.msme_policy_qna.config import DEFAULT_RETRIEVAL_K, SOURCE_PDF_URL, VECTOR_DB_DIR
from src.msme_policy_qna.rag_chain import answer_question
from src.msme_policy_qna.vector_store import build_vector_store, ensure_vector_store, vector_store_has_documents


st.set_page_config(page_title="MSME Policy Q&A", page_icon="MSME", layout="wide")

st.title("MSME Policy Q&A")

if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = False

if not st.session_state.vector_store_ready:
    if vector_store_has_documents():
        st.session_state.vector_store_ready = True
    else:
        with st.spinner("Preparing the policy search index. This runs once on startup..."):
            ensure_vector_store()
        st.session_state.vector_store_ready = True

with st.sidebar:
    st.header("Source")
    st.markdown(f"Source PDF: [{SOURCE_PDF_URL}]({SOURCE_PDF_URL})")
    st.caption("New Guidelines of Micro & Small Enterprises Cluster Development Programme (MSE-CDP).")

    st.header("API Key")
    openai_api_key = st.text_input(
        "OpenAI API key",
        type="password",
        help="Used only for this session. The public app does not read a repo .env key.",
    )

    st.header("Index")
    st.caption(f"Vector store: {VECTOR_DB_DIR}")
    if st.button("Rebuild vector store", use_container_width=True):
        with st.spinner("Embedding chunks and rebuilding Chroma..."):
            build_vector_store()
            st.session_state.vector_store_ready = True
        st.success("Vector store rebuilt.")

    k = st.slider("Retrieved chunks", min_value=3, max_value=10, value=DEFAULT_RETRIEVAL_K)

question = st.text_area(
    "Question",
    value="What is the GoI grant pattern for Common Facility Centres?",
    height=110,
)

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Enter a question first.")
    elif not openai_api_key.strip():
        st.warning("Add your OpenAI API key in the sidebar to generate an answer.")
    else:
        with st.spinner("Retrieving policy chunks and drafting an answer..."):
            result = answer_question(
                question.strip(),
                k=k,
                openai_api_key=openai_api_key.strip(),
                allow_env=False,
            )

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources")
        for citation in result["citations"]:
            st.markdown(
                f"- **{citation['label']}**  \n"
                f"  `{citation['chunk_id']}`"
            )

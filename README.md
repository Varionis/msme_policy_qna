# MSME Policy Q&A RAG POC

This repository is a first-pass RAG proof of concept for answering questions from one sample MSME policy chunk file: `data/chunks_sample.json`.

## Vector DB Recommendation

Use **Chroma** for this POC. Chroma has a dedicated LangChain integration, and LangChain documents Chroma as a vector store integration through the `langchain-chroma` package.

Why Chroma:

- Free and open-source.
- Runs locally with no hosted service account.
- Persists embeddings to disk for repeatable local demos.
- Has a maintained LangChain integration through `langchain-chroma`.

The default persistent store path is `vector_db/chroma`.

References:

- LangChain Chroma package reference: https://reference.langchain.com/python/langchain-chroma
- Chroma LangChain integration docs: https://docs.trychroma.com/integrations/langchain

## Project Layout

```text
data/
  chunks_sample.json
  eval_questions.json
src/
  msme_policy_qna/
    chunk_loader.py
    config.py
    embeddings.py
    evaluate.py
    rag_chain.py
    vector_store.py
  streamlit_app.py
test/
  test_chunk_loader.py
  test_citations.py
requirements.txt
README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional environment variables:

```bash
set OPENAI_API_KEY=your_key
set OPENAI_MODEL=gpt-4o-mini
```

For a free local generation path, install Ollama separately, pull a chat model, and set:

```bash
set OLLAMA_MODEL=llama3.1
```

If neither OpenAI nor Ollama is configured, the app still retrieves relevant chunks and returns an extractive preview, but the final generated answer will be limited.

## Build the Vector Store

```bash
python -m src.msme_policy_qna.vector_store
```

This reads `data/chunks_sample.json`, embeds each chunk, and stores vectors in `vector_db/chroma`.

## Ask a Question from the CLI

```bash
python -m src.msme_policy_qna.rag_chain "What is the GoI grant for Common Facility Centres?"
```

## Run the Streamlit UI

```bash
streamlit run src/streamlit_app.py
```

The Streamlit app is public-deployment oriented: it asks each user for their own OpenAI API key in the sidebar and does not read the repo `.env` key. The CLI and evaluation scripts can still use `.env` for local development.

The sidebar also links to the source PDF used by the POC: https://www.dcmsme.gov.in/schemes/New-Guidelines.pdf

For Streamlit Community Cloud, select Python 3.12 in the app's Advanced settings. Chroma is pinned in `requirements.txt` to the pre-1.x local persistent client line for this POC.

## Evaluate the POC Question Set

The starter evaluation set has 18 questions covering funding, eligibility, exceptions, procedures, timelines, monitoring, and unanswerable questions.

```bash
python -m src.msme_policy_qna.evaluate
```

The script prints JSON lines with the question, retrieved citations, answer, and whether the question is expected to be answerable from the document.

## Tests

```bash
pytest
```

## Notes for the Next Increment

- Add ground-truth expected answers for the evaluation set.
- Add retrieval metrics such as top-k citation hit rate.
- Add answer-quality scoring with a human rubric or model-assisted grading.
- Add ingestion for multiple source documents and scheme-level filtering.
- Add source PDF links if the production corpus provides stable document URLs.

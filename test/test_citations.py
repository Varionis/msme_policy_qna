from langchain_core.documents import Document

from src.msme_policy_qna.rag_chain import citation_from_document


def test_citation_label_includes_page_clause_and_section():
    document = Document(
        page_content="Example",
        metadata={
            "chunk_id": "chunk_1",
            "page_numbers": "4, 5",
            "clause_number": "9",
            "section_title": "Monitoring and Evaluation",
        },
    )

    citation = citation_from_document(document)

    assert citation.label == "p. 4, 5; clause 9; Monitoring and Evaluation"

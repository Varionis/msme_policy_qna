from __future__ import annotations

import json

from dotenv import load_dotenv

from .config import EVAL_QUESTIONS_PATH, ROOT_DIR
from .rag_chain import answer_question


def load_eval_questions() -> list[dict]:
    with EVAL_QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")

    for item in load_eval_questions():
        result = answer_question(item["question"])
        print(
            json.dumps(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "expected_answerable": item["expected_answerable"],
                    "question": item["question"],
                    "answer": result["answer"],
                    "citations": result["citations"],
                },
                ensure_ascii=True,
            )
        )


if __name__ == "__main__":
    main()

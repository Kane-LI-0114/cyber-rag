from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from cyber_rag.config import EmbeddingConfig, GenerationConfig, RetrievalConfig
from cyber_rag.evaluation.datasets import load_evaluation_examples
from cyber_rag.generation.chain import answer_with_retrieval, answer_without_retrieval


def _get_field(example, name: str, default=None):
    if isinstance(example, Mapping):
        return example.get(name, default)
    return getattr(example, name, default)


def _is_multiple_choice(example) -> bool:
    answers = _get_field(example, "answers")
    solution = _get_field(example, "solution")
    return isinstance(answers, dict) and len(answers) > 0 and solution is not None


def _format_mcq_question(question: str, answers: dict[str, str]) -> str:
    lines = [question, "", "Choices:"]
    for key, value in answers.items():
        lines.append(f"{key}. {value}")
    lines.append("")
    lines.append("Return only the single best option letter.")
    return "\n".join(lines)


def _extract_choice(answer_text: str, valid_choices: set[str]) -> str:
    text = answer_text.strip().upper()

    if text in valid_choices:
        return text

    match = re.search(r"\b([A-Z])\b", text)
    if match and match.group(1) in valid_choices:
        return match.group(1)

    for choice in valid_choices:
        if f"ANSWER: {choice}" in text or f"OPTION {choice}" in text:
            return choice

    return text


def run_evaluation(
    dataset_path: str | Path,
    index_path: str | Path,
    embedding_config: EmbeddingConfig | None = None,
    retrieval_config: RetrievalConfig | None = None,
    generation_config: GenerationConfig | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    examples = load_evaluation_examples(dataset_path)
    if limit is not None:
        examples = examples[:limit]

    rows: list[dict] = []
    for example in examples:
        question = _get_field(example, "question")

        if _is_multiple_choice(example):
            answers = _get_field(example, "answers")
            solution = str(_get_field(example, "solution")).strip().upper()

            baseline = answer_without_retrieval(
                question=question,
                answer_options=answers,
                generation_config=generation_config,
            )
            rag_result = answer_with_retrieval(
                question=question,
                answer_options=answers,
                index_path=index_path,
                embedding_config=embedding_config,
                retrieval_config=retrieval_config,
                generation_config=generation_config,
            )

            baseline_choice = _extract_choice(baseline.answer, set(answers.keys()))
            rag_choice = _extract_choice(rag_result.answer, set(answers.keys()))

            rows.append(
                {
                    "question": question,
                    "question_type": "multiple_choice",
                    "choices": answers,
                    "reference_answer": solution,
                    "reference_answer_text": answers.get(solution),
                    "baseline_answer": baseline_choice,
                    "baseline_answer_raw": baseline.answer,
                    "baseline_correct": baseline_choice == solution,
                    "rag_answer": rag_choice,
                    "rag_answer_raw": rag_result.answer,
                    "rag_correct": rag_choice == solution,
                    "retrieved_chunks": len(rag_result.sources),
                }
            )
        else:
            reference_answer = _get_field(example, "answer")

            baseline = answer_without_retrieval(
                question=question,
                generation_config=generation_config,
            )
            rag_result = answer_with_retrieval(
                question=question,
                index_path=index_path,
                embedding_config=embedding_config,
                retrieval_config=retrieval_config,
                generation_config=generation_config,
            )

            rows.append(
                {
                    "question": question,
                    "question_type": "short_answer",
                    "reference_answer": reference_answer,
                    "baseline_answer": baseline.answer,
                    "rag_answer": rag_result.answer,
                    "retrieved_chunks": len(rag_result.sources),
                }
            )

    return pd.DataFrame(rows)
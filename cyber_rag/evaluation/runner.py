from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from cyber_rag.config import EmbeddingConfig, GenerationConfig, RetrievalConfig
from cyber_rag.evaluation.datasets import load_evaluation_examples
from cyber_rag.evaluation.ragas_metrics import run_ragas_short_answer_batches
from cyber_rag.evaluation.text_metrics import (
    cosine_similarity_texts,
    exact_match,
    token_f1,
)
from cyber_rag.generation.chain import answer_with_retrieval, answer_without_retrieval
from cyber_rag.generation.local_llm import  answer_with_retrieval_local, answer_without_retrieval_local
from cyber_rag.indexing.faiss_store import create_embeddings
from cyber_rag.retrieval.retriever import retrieve_documents


def _get_field(example, name: str, default=None):
    if isinstance(example, Mapping):
        return example.get(name, default)
    return getattr(example, name, default)


def _is_multiple_choice(example) -> bool:
    answers = _get_field(example, "answers")
    solution = _get_field(example, "solution")
    return isinstance(answers, dict) and len(answers) > 0 and solution is not None


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
    skip_ragas: bool = False,
) -> pd.DataFrame:
    examples = load_evaluation_examples(dataset_path)

    if limit is not None:
        examples = examples[:limit]

    embedder = create_embeddings(embedding_config)

    rows: list[dict] = []
    sa_row_indices: list[int] = []
    sa_questions: list[str] = []
    sa_refs: list[str] = []
    sa_baseline: list[str] = []
    sa_rag: list[str] = []
    sa_contexts: list[list[str]] = []

    for example in examples:
        question = _get_field(example, "question")

        if _is_multiple_choice(example):
            answers = _get_field(example, "answers")
            solution = str(_get_field(example, "solution")).strip().upper()

            baseline = answer_without_retrieval_local(
                question=question,
                answer_options=answers,
                generation_config=generation_config,
            )

            rag_result = answer_with_retrieval_local(
                question=question,
                answer_options=answers,
                index_path=index_path,
                embedding_config=embedding_config,
                retrieval_config=retrieval_config,
                generation_config=generation_config,
            )

            baseline_choice = _extract_choice(baseline.answer, set(answers.keys()))
            rag_choice = _extract_choice(rag_result.answer, set(answers.keys()))

            baseline_correct = baseline_choice == solution
            rag_correct = rag_choice == solution

            rows.append(
                {
                    "question": question,
                    "question_type": "multiple_choice",
                    "choices": answers,
                    "reference_answer": solution,
                    "reference_answer_text": answers.get(solution),
                    "baseline_answer": baseline_choice,
                    "baseline_answer_raw": baseline.answer,
                    "baseline_correct": baseline_correct,
                    "rag_answer": rag_choice,
                    "rag_answer_raw": rag_result.answer,
                    "rag_correct": rag_correct,
                    "retrieved_chunks": len(rag_result.sources),
                }
            )
        else:
            reference_answer = _get_field(example, "answer")
            ref_str = None if reference_answer is None else str(reference_answer)

            # Full chunk text for RAGAS faithfulness / context_precision (same k as retrieval config).
            # answer_with_retrieval below performs a second retrieval internally for generation.
            rag_docs = retrieve_documents(
                query=question,
                index_path=index_path,
                embedding_config=embedding_config,
                retrieval_config=retrieval_config,
            )
            rag_contexts = [d.page_content for d in rag_docs]

            baseline = answer_without_retrieval_local(
                question=question,
                generation_config=generation_config,
            )

            rag_result = answer_with_retrieval_local(
                question=question,
                index_path=index_path,
                embedding_config=embedding_config,
                retrieval_config=retrieval_config,
                generation_config=generation_config,
            )

            row: dict = {
                "question": question,
                "question_type": "short_answer",
                "reference_answer": reference_answer,
                "baseline_answer": baseline.answer,
                "rag_answer": rag_result.answer,
                "retrieved_chunks": len(rag_result.sources),
                "baseline_em": exact_match(baseline.answer, ref_str),
                "baseline_token_f1": token_f1(baseline.answer, ref_str),
                "baseline_cosine_similarity": cosine_similarity_texts(
                    baseline.answer, ref_str, embedder
                ),
                "rag_em": exact_match(rag_result.answer, ref_str),
                "rag_token_f1": token_f1(rag_result.answer, ref_str),
                "rag_cosine_similarity": cosine_similarity_texts(
                    rag_result.answer, ref_str, embedder
                ),
                "baseline_ragas_answer_relevancy": math.nan,
                "baseline_ragas_faithfulness": math.nan,
                "baseline_ragas_context_precision": math.nan,
                "rag_ragas_faithfulness": math.nan,
                "rag_ragas_answer_relevancy": math.nan,
                "rag_ragas_context_precision": math.nan,
            }

            sa_row_indices.append(len(rows))
            sa_questions.append(question)
            sa_refs.append(ref_str or "")
            sa_baseline.append(str(baseline.answer))
            sa_rag.append(str(rag_result.answer))
            sa_contexts.append(rag_contexts)

            rows.append(row)

    if sa_questions and not skip_ragas:
        (
            b_ar,
            r_faith,
            r_ar,
            r_cp,
        ) = run_ragas_short_answer_batches(
            questions=sa_questions,
            baseline_answers=sa_baseline,
            rag_answers=sa_rag,
            rag_contexts=sa_contexts,
            ground_truths=sa_refs,
            generation_config=generation_config,
            embedding_config=embedding_config,
        )
        for j, row_idx in enumerate(sa_row_indices):
            rows[row_idx]["baseline_ragas_answer_relevancy"] = b_ar[j]
            rows[row_idx]["rag_ragas_faithfulness"] = r_faith[j]
            rows[row_idx]["rag_ragas_answer_relevancy"] = r_ar[j]
            rows[row_idx]["rag_ragas_context_precision"] = r_cp[j]

    return pd.DataFrame(rows)

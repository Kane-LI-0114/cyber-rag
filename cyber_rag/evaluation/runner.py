from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from cyber_rag.config import EmbeddingConfig, GenerationConfig, RetrievalConfig
from cyber_rag.evaluation.datasets import load_evaluation_examples
from cyber_rag.evaluation.llm_judge import judge_answer_accuracy
from cyber_rag.generation.chain import answer_with_retrieval, answer_without_retrieval

def _get_field(example, name: str, default=None):
    if isinstance(example, Mapping):
        return example.get(name, default)
    return getattr(example, name, default)


def _is_multiple_choice(example) -> bool:
    answers = _get_field(example, "answers")
    solution = _get_field(example, "solution")
    return isinstance(answers, dict) and len(answers) > 0 and solution is not None


def _skip_row_mcq(
    question: str,
    answers: dict,
    solution: str,
    reason: str,
) -> dict:
    return {
        "question": question,
        "question_type": "multiple_choice",
        "choices": answers,
        "reference_answer": solution,
        "reference_answer_text": answers.get(solution),
        "baseline_answer": "",
        "baseline_answer_raw": "",
        "baseline_correct": False,
        "rag_answer": "",
        "rag_answer_raw": "",
        "rag_correct": False,
        "retrieved_chunks": 0,
        "eval_skipped": True,
        "skip_reason": reason,
    }


def _skip_row_short(
    question: str,
    reference_answer,
    reason: str,
) -> dict:
    return {
        "question": question,
        "question_type": "short_answer",
        "reference_answer": reference_answer,
        "baseline_answer": "",
        "baseline_judge_accuracy": float("nan"),
        "rag_judge_accuracy": float("nan"),
        "baseline_correct": False,
        "baseline_judge_reason": reason,
        "rag_answer": "",
        "rag_correct": False,
        "rag_judge_reason": reason,
        "retrieved_chunks": 0,
        "eval_skipped": True,
        "skip_reason": reason,
    }


def _extract_choice(answer_text: str, valid_choices: set[str]) -> str:
    print(f"  [DEBUG _extract_choice] Extracting from raw text: {answer_text!r}")
    print(f"  [DEBUG _extract_choice] Valid choices: {valid_choices}")
    
    text = answer_text.strip().upper()

    if text in valid_choices:
        print(f"  [DEBUG _extract_choice] Exact match found: {text}")
        return text

    match = re.search(r"\b([A-Z])\b", text)
    if match and match.group(1) in valid_choices:
        print(f"  [DEBUG _extract_choice] Regex match found: {match.group(1)}")
        return match.group(1)

    for choice in valid_choices:
        if f"ANSWER: {choice}" in text or f"OPTION {choice}" in text:
            print(f"  [DEBUG _extract_choice] Substring match found: {choice}")
            return choice

    print(f"  [DEBUG _extract_choice] No clear match found. Returning sanitized raw: {text!r}")
    return text


def run_evaluation(
    dataset_path: str | Path,
    index_path: str | Path,
    embedding_config: EmbeddingConfig | None = None,
    retrieval_config: RetrievalConfig | None = None,
    generation_config: GenerationConfig | None = None,
    judge_generation_config: GenerationConfig | None = None,
    judge_threshold: float = 0.5,
    limit: int | None = None,
) -> pd.DataFrame:
    
    print(f"[DEBUG] Loading evaluation examples from: {dataset_path}")
    examples = load_evaluation_examples(dataset_path)
    
    if limit is not None:
        examples = examples[:limit]
        print(f"[DEBUG] Applied limit. Evaluating {limit} examples.")
    else:
        print(f"[DEBUG] Evaluating all {len(examples)} examples.")

    rows: list[dict] = []
    for i, example in enumerate(examples):
        print(f"\n{'='*60}")
        print(f"[DEBUG] Processing Example {i+1} / {len(examples)}")
        
        question = _get_field(example, "question")
        print(f"[DEBUG] Question: {question}")

        if _is_multiple_choice(example):
            print("[DEBUG] Type Detected: Multiple Choice")
            answers = _get_field(example, "answers")
            solution = str(_get_field(example, "solution")).strip().upper()

            print(f"[DEBUG] Expected Solution: {solution}")

            try:
                print("[DEBUG] Running baseline (without retrieval)...")
                baseline = answer_without_retrieval(
                    question=question,
                    answer_options=answers,
                    generation_config=generation_config,
                )

                print("[DEBUG] Running RAG (with retrieval)...")
                rag_result = answer_with_retrieval(
                    question=question,
                    answer_options=answers,
                    index_path=index_path,
                    embedding_config=embedding_config,
                    retrieval_config=retrieval_config,
                    generation_config=generation_config,
                )

                print("[DEBUG] Extracting Baseline choice...")
                baseline_choice = _extract_choice(baseline.answer, set(answers.keys()))

                print("[DEBUG] Extracting RAG choice...")
                rag_choice = _extract_choice(rag_result.answer, set(answers.keys()))

                baseline_correct = baseline_choice == solution
                rag_correct = rag_choice == solution

                print(
                    f"[DEBUG] Baseline Correct? {baseline_correct} | RAG Correct? {rag_correct}"
                )
                print(f"[DEBUG] RAG Retrieved Chunks: {len(rag_result.sources)}")

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
                        "eval_skipped": False,
                    }
                )
            except Exception as e:
                print(f"[WARN] Skipping MCQ example ({i + 1}): {e}")
                rows.append(_skip_row_mcq(question, answers, solution, str(e)))
        else:
            print("[DEBUG] Type Detected: Short Answer")
            reference_answer = _get_field(example, "answer")
            print(f"[DEBUG] Expected Answer: {reference_answer}")

            try:
                print("[DEBUG] Running baseline (without retrieval)...")
                baseline = answer_without_retrieval(
                    question=question,
                    generation_config=generation_config,
                )

                print("[DEBUG] Running RAG (with retrieval)...")
                rag_result = answer_with_retrieval(
                    question=question,
                    index_path=index_path,
                    embedding_config=embedding_config,
                    retrieval_config=retrieval_config,
                    generation_config=generation_config,
                )

                print(f"[DEBUG] Baseline output: {baseline.answer!r}")
                print(f"[DEBUG] RAG output: {rag_result.answer!r}")
                print(f"[DEBUG] RAG Retrieved Chunks: {len(rag_result.sources)}")

                print(
                    "[DEBUG] Judging baseline short-answer accuracy with judge model..."
                )
                baseline_judge_accuracy, baseline_judge_reason = judge_answer_accuracy(
                    question=question,
                    reference_answer=str(reference_answer),
                    candidate_answer=str(baseline.answer),
                    judge_config=judge_generation_config,
                )
                baseline_correct = baseline_judge_accuracy >= judge_threshold
                print(
                    f"[DEBUG] Baseline judge accuracy: {baseline_judge_accuracy:.3f} "
                    f"(correct>={judge_threshold}: {baseline_correct})"
                )

                print("[DEBUG] Judging RAG short-answer accuracy with judge model...")
                rag_judge_accuracy, rag_judge_reason = judge_answer_accuracy(
                    question=question,
                    reference_answer=str(reference_answer),
                    candidate_answer=str(rag_result.answer),
                    judge_config=judge_generation_config,
                )
                rag_correct = rag_judge_accuracy >= judge_threshold
                print(
                    f"[DEBUG] RAG judge accuracy: {rag_judge_accuracy:.3f} "
                    f"(correct>={judge_threshold}: {rag_correct})"
                )

                rows.append(
                    {
                        "question": question,
                        "question_type": "short_answer",
                        "reference_answer": reference_answer,
                        "baseline_answer": baseline.answer,
                        "baseline_judge_accuracy": baseline_judge_accuracy,
                        "rag_judge_accuracy": rag_judge_accuracy,
                        "baseline_correct": baseline_correct,
                        "baseline_judge_reason": baseline_judge_reason,
                        "rag_answer": rag_result.answer,
                        "rag_correct": rag_correct,
                        "rag_judge_reason": rag_judge_reason,
                        "retrieved_chunks": len(rag_result.sources),
                        "eval_skipped": False,
                    }
                )
            except Exception as e:
                print(f"[WARN] Skipping short-answer example ({i + 1}): {e}")
                rows.append(_skip_row_short(question, reference_answer, str(e)))

    print(f"\n[DEBUG] Evaluation complete. Processed {len(rows)} rows.")
    return pd.DataFrame(rows)
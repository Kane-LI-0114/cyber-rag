from __future__ import annotations

import logging
import math

from datasets import Dataset

from cyber_rag.config import EmbeddingConfig, GenerationConfig
from cyber_rag.generation.chain import _build_llm
from cyber_rag.indexing.faiss_store import create_embeddings

logger = logging.getLogger(__name__)


def _as_float(x: object) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan
    return math.nan if math.isnan(v) else v


def run_ragas_short_answer_batches(
    questions: list[str],
    baseline_answers: list[str],
    rag_answers: list[str],
    rag_contexts: list[list[str]],
    ground_truths: list[str],
    generation_config: GenerationConfig | None,
    embedding_config: EmbeddingConfig | None,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Run RAGAS metrics in batch for short-answer evaluation.

    Baseline (no retrieval): ``answer_relevancy`` only; there is no retrieved context, so
    **faithfulness** and **context_precision** are not defined for the baseline path (use NaN in the runner).

    RAG: ``faithfulness``, ``answer_relevancy``, ``context_precision``.

    Returns
    -------
    baseline_answer_relevancy, rag_faithfulness, rag_answer_relevancy, rag_context_precision
    """
    n = len(questions)
    nan = [math.nan] * n

    if n == 0:
        return nan, nan, nan, nan

    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    gc = generation_config or GenerationConfig()
    ec = embedding_config or EmbeddingConfig()

    llm = _build_llm(gc)
    embeddings = create_embeddings(ec)

    baseline_ar = nan
    rag_faith = nan
    rag_ar = nan
    rag_ctxprec = nan

    # --- Baseline: answer relevancy only (no retrieved contexts)
    ds_baseline = Dataset.from_dict(
        {
            "question": questions,
            "answer": baseline_answers,
            "ground_truth": ground_truths,
        }
    )
    try:
        res_b = evaluate(
            ds_baseline,
            metrics=[answer_relevancy],
            llm=llm,
            embeddings=embeddings,
            show_progress=False,
            raise_exceptions=False,
        )
        baseline_ar = [_as_float(x) for x in res_b["answer_relevancy"]]
    except Exception:
        logger.exception("RAGAS baseline answer_relevancy batch failed; filling with NaN.")

    # --- RAG: faithfulness, answer_relevancy, context_precision
    ds_rag = Dataset.from_dict(
        {
            "question": questions,
            "answer": rag_answers,
            "contexts": rag_contexts,
            "ground_truth": ground_truths,
        }
    )
    try:
        res_r = evaluate(
            ds_rag,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=llm,
            embeddings=embeddings,
            show_progress=False,
            raise_exceptions=False,
        )
        rag_faith = [_as_float(x) for x in res_r["faithfulness"]]
        rag_ar = [_as_float(x) for x in res_r["answer_relevancy"]]
        rag_ctxprec = [_as_float(x) for x in res_r["context_precision"]]
    except Exception:
        logger.exception(
            "RAGAS RAG batch (faithfulness, answer_relevancy, context_precision) failed; filling with NaN."
        )

    return (baseline_ar, rag_faith, rag_ar, rag_ctxprec)

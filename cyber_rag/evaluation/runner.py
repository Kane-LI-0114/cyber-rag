from __future__ import annotations

from pathlib import Path

import pandas as pd

from cyber_rag.config import EmbeddingConfig, GenerationConfig, RetrievalConfig
from cyber_rag.evaluation.datasets import load_evaluation_examples
from cyber_rag.generation.chain import answer_with_retrieval, answer_without_retrieval


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
        baseline = answer_without_retrieval(
            question=example.question,
            generation_config=generation_config,
        )
        rag_result = answer_with_retrieval(
            question=example.question,
            index_path=index_path,
            embedding_config=embedding_config,
            retrieval_config=retrieval_config,
            generation_config=generation_config,
        )
        rows.append(
            {
                "question": example.question,
                "reference_answer": example.answer,
                "baseline_answer": baseline.answer,
                "rag_answer": rag_result.answer,
                "retrieved_chunks": len(rag_result.sources),
            }
        )

    return pd.DataFrame(rows)

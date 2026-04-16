from __future__ import annotations

import argparse
from pathlib import Path

from cyber_rag.config import DEFAULT_EVAL_PATH, DEFAULT_INDEX_DIR, EmbeddingConfig, GenerationConfig, RetrievalConfig
from cyber_rag.evaluation.runner import run_evaluation
from cyber_rag.evaluation.summarize_output import append_eval_summary_to_overall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CyberRAG evaluation batch.")
    parser.add_argument("dataset", help="Path to a JSONL or CSV evaluation dataset.")
    parser.add_argument(
        "--index-path",
        default=str(DEFAULT_INDEX_DIR),
        help="Path to the local FAISS index directory.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    parser.add_argument("--k", type=int, default=4, help="Number of retrieved chunks.")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name used to load the vector store.",
    )
    parser.add_argument("--model", default=None, help="LLM name for answer generation (defaults to .env config).")
    parser.add_argument(
        "--judge-model",
        default=None,
        help="LLM name for short-answer judging (defaults to --model or .env config).",
    )
    parser.add_argument(
        "--judge-threshold",
        type=float,
        default=0.5,
        help="Short-answer judge: baseline_correct/rag_correct are True when score >= this (default 0.5).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_EVAL_PATH),
        help="CSV path for writing evaluation outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = run_evaluation(
        dataset_path=args.dataset,
        index_path=args.index_path,
        embedding_config=EmbeddingConfig(model_name=args.embedding_model),
        retrieval_config=RetrievalConfig(k=args.k),
        generation_config=GenerationConfig(model_name=args.model),
        judge_generation_config=GenerationConfig(
            model_name=args.judge_model or args.model
        ),
        judge_threshold=args.judge_threshold,
        limit=args.limit,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"Saved {len(frame)} evaluation rows to {output_path}")

    overall = append_eval_summary_to_overall(output_path, frame)
    print(f"Appended summary row to {overall}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from cyber_rag.config import DEFAULT_INDEX_DIR, EmbeddingConfig, GenerationConfig, RetrievalConfig
from cyber_rag.generation.chain import answer_with_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single CyberRAG query.")
    parser.add_argument("query", help="User question to send through the retrieval pipeline.")
    parser.add_argument(
        "--index-path",
        default=str(DEFAULT_INDEX_DIR),
        help="Path to the local FAISS index directory.",
    )
    parser.add_argument("--k", type=int, default=4, help="Number of retrieved chunks.")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name used to load the vector store.",
    )
    parser.add_argument("--model", default=None, help="LLM name for answer generation (defaults to .env config).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = answer_with_retrieval(
        question=args.query,
        index_path=args.index_path,
        embedding_config=EmbeddingConfig(model_name=args.embedding_model),
        retrieval_config=RetrievalConfig(k=args.k),
        generation_config=GenerationConfig(model_name=args.model),
    )
    print(result.answer)
    print("\nSources:")
    print(
        json.dumps(
            [asdict(source) for source in result.sources],
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

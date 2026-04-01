from __future__ import annotations

import argparse
import json
from pathlib import Path

from cyber_rag.config import ChunkingConfig, DEFAULT_INDEX_DIR, EmbeddingConfig, RAW_DATA_DIR, ensure_project_directories
from cyber_rag.indexing.faiss_store import build_and_save_index
from cyber_rag.ingest.loaders import load_sources
from cyber_rag.logging_utils import configure_logging
from cyber_rag.processing.chunking import split_documents
from cyber_rag.processing.normalize import normalize_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local CyberRAG retrieval index.")
    parser.add_argument(
        "sources",
        nargs="*",
        default=[str(RAW_DATA_DIR)],
        help="Files, folders, or URLs to ingest into the index.",
    )
    parser.add_argument(
        "--index-path",
        default=str(DEFAULT_INDEX_DIR),
        help="Destination directory for the FAISS index.",
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name used for local indexing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logging()
    ensure_project_directories()

    raw_documents = load_sources(args.sources)
    if not raw_documents:
        raise SystemExit("No supported source documents were found.")

    normalized_documents = normalize_documents(raw_documents)
    chunks = split_documents(
        normalized_documents,
        ChunkingConfig(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap),
    )
    build_and_save_index(
        documents=chunks,
        index_path=args.index_path,
        config=EmbeddingConfig(model_name=args.embedding_model),
    )

    manifest = {
        "source_inputs": args.sources,
        "raw_document_count": len(raw_documents),
        "normalized_document_count": len(normalized_documents),
        "chunk_count": len(chunks),
        "embedding_model": args.embedding_model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
    }
    manifest_path = Path(args.index_path) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Index built successfully at %s", args.index_path)
    logger.info("Manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()

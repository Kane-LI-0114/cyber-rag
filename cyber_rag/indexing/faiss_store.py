from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from cyber_rag.config import EmbeddingConfig


def create_embeddings(config: EmbeddingConfig | None = None) -> HuggingFaceEmbeddings:
    embedding_config = config or EmbeddingConfig()
    return HuggingFaceEmbeddings(
        model_name=embedding_config.model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": embedding_config.normalize_embeddings},
    )


def build_and_save_index(
    documents: list[Document],
    index_path: str | Path,
    config: EmbeddingConfig | None = None,
) -> FAISS:
    path = Path(index_path)
    path.mkdir(parents=True, exist_ok=True)
    vector_store = FAISS.from_documents(documents, create_embeddings(config))
    vector_store.save_local(str(path))
    return vector_store


def load_index(index_path: str | Path, config: EmbeddingConfig | None = None) -> FAISS:
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Index path does not exist: {path}")
    return FAISS.load_local(
        str(path),
        create_embeddings(config),
        allow_dangerous_deserialization=True,
    )

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from cyber_rag.config import EmbeddingConfig, RetrievalConfig
from cyber_rag.indexing.faiss_store import load_index


def build_retriever(
    index_path: str | Path,
    embedding_config: EmbeddingConfig | None = None,
    retrieval_config: RetrievalConfig | None = None,
):
    retrieval = retrieval_config or RetrievalConfig()
    vector_store = load_index(index_path=index_path, config=embedding_config)
    return vector_store.as_retriever(
        search_type=retrieval.search_type,
        search_kwargs={"k": retrieval.k},
    )


def retrieve_documents(
    query: str,
    index_path: str | Path,
    embedding_config: EmbeddingConfig | None = None,
    retrieval_config: RetrievalConfig | None = None,
) -> list[Document]:
    retriever = build_retriever(
        index_path=index_path,
        embedding_config=embedding_config,
        retrieval_config=retrieval_config,
    )
    return retriever.invoke(query)

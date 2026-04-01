from __future__ import annotations

from langchain_core.documents import Document

from cyber_rag.config import ChunkingConfig
from cyber_rag.processing.chunking import split_documents
from cyber_rag.processing.normalize import normalize_documents


def test_chunking_preserves_source_metadata() -> None:
    documents = [
        Document(
            page_content="Intro\n" + ("evidence " * 250),
            metadata={"source": "data/raw/example.txt", "source_path": "data/raw/example.txt", "page": 1},
        )
    ]

    normalized = normalize_documents(documents)
    chunks = split_documents(normalized, ChunkingConfig(chunk_size=250, chunk_overlap=50))

    assert len(chunks) >= 2
    for index, chunk in enumerate(chunks):
        assert chunk.metadata["source_path"] == "data/raw/example.txt"
        assert chunk.metadata["chunk_index"] == index
        assert chunk.metadata["chunk_id"]
        assert chunk.metadata["security_domain"]

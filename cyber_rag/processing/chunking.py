from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cyber_rag.config import ChunkingConfig


def _build_chunk_id(metadata: dict, chunk_index: int, content: str) -> str:
    source = metadata.get("source_path") or metadata.get("source") or "unknown"
    stem = Path(str(source)).stem or "source"
    page = metadata.get("page", "na")
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
    return f"{stem}:{page}:{chunk_index}:{digest}"


def split_documents(
    documents: list[Document],
    config: ChunkingConfig | None = None,
) -> list[Document]:
    chunking = config or ChunkingConfig()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunking.chunk_size,
        chunk_overlap=chunking.chunk_overlap,
        separators=chunking.separators,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata)
        metadata["chunk_index"] = index
        metadata["chunk_id"] = _build_chunk_id(metadata, index, chunk.page_content)
        metadata.setdefault("start_index", metadata.get("start_index", 0))
        chunk.metadata = metadata
    return chunks

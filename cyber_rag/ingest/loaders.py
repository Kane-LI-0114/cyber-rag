from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

os.environ.setdefault("USER_AGENT", "CyberRAG/0.1")

from langchain_community.document_loaders import (
    BSHTMLLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".html", ".htm"}


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _load_local_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        documents = PyPDFLoader(str(path)).load()
    elif suffix in {".md", ".txt"}:
        documents = TextLoader(str(path), encoding="utf-8").load()
    elif suffix in {".html", ".htm"}:
        documents = BSHTMLLoader(str(path)).load()
    else:
        raise ValueError(f"Unsupported file type: {path}")

    for index, document in enumerate(documents):
        document.metadata.setdefault("source", str(path))
        document.metadata["source_path"] = str(path.resolve())
        document.metadata["source_name"] = path.name
        document.metadata["file_extension"] = suffix
        document.metadata["record_index"] = index
    return documents


def _load_web_source(url: str) -> list[Document]:
    documents = WebBaseLoader(web_paths=(url,)).load()
    for index, document in enumerate(documents):
        document.metadata.setdefault("source", url)
        document.metadata["source_url"] = url
        document.metadata["source_name"] = urlparse(url).netloc
        document.metadata["record_index"] = index
    return documents


def _expand_inputs(inputs: Iterable[str | Path]) -> list[str | Path]:
    expanded: list[str | Path] = []
    for value in inputs:
        if isinstance(value, Path):
            path = value
        else:
            if _is_url(value):
                expanded.append(value)
                continue
            path = Path(value)

        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                    expanded.append(child)
        elif path.is_file():
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                logger.warning("Skipping unsupported file: %s", path)
                continue
            expanded.append(path)
        else:
            logger.warning("Skipping missing input: %s", path)
    return expanded


def load_sources(inputs: Iterable[str | Path]) -> list[Document]:
    documents: list[Document] = []
    for value in _expand_inputs(inputs):
        if isinstance(value, str):
            documents.extend(_load_web_source(value))
        else:
            documents.extend(_load_local_file(value))
    return documents

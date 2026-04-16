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

from cyber_rag.config import DotsOcrConfig

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".html", ".htm"}


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _load_pdf_with_dots_ocr(path: Path) -> tuple[list[Document], dict[str, object] | None]:
    """Try to parse a PDF using DotsOCR.

    Returns:
        A tuple of (documents, structured_layout_data).
        On failure, returns ([], None) so the caller can fall back.
    """
    from cyber_rag.ingest.dots_ocr import parse_pdf_with_dots_ocr

    config = DotsOcrConfig()
    if not config.is_configured:
        logger.info("DotsOCR not configured, falling back to PyPDFLoader")
        return [], None

    try:
        documents, structured_data = parse_pdf_with_dots_ocr(str(path), config)
        logger.info(
            "DotsOCR parsed %s: %d pages, %d layout blocks",
            path.name,
            len(documents),
            len(structured_data.get("blocks", [])),
        )
        return documents, structured_data
    except Exception as exc:
        logger.warning("DotsOCR failed for %s: %s", path, exc)
        return [], None


def _load_pdf_with_pypdf(path: Path) -> list[Document]:
    """Fallback: parse a PDF using PyPDFLoader (no structured layout)."""
    logger.info("Using PyPDFLoader fallback for %s", path.name)
    documents = PyPDFLoader(str(path)).load()
    for index, document in enumerate(documents):
        document.metadata.setdefault("source", str(path))
        document.metadata["source_path"] = str(path.resolve())
        document.metadata["source_name"] = path.name
        document.metadata["file_extension"] = ".pdf"
        document.metadata["record_index"] = index
        document.metadata["parser"] = "pypdf"
    return documents


def _load_local_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        # Try DotsOCR first, fall back to PyPDFLoader
        documents, structured_data = _load_pdf_with_dots_ocr(path)
        if documents:
            # Enrich metadata with source provenance
            for index, document in enumerate(documents):
                document.metadata.setdefault("source", str(path))
                document.metadata["source_path"] = str(path.resolve())
                document.metadata["source_name"] = path.name
                document.metadata["file_extension"] = suffix
                document.metadata["record_index"] = index
                # Attach structured layout data to the first page's metadata
                # so downstream processing can access it
                if structured_data is not None and index == 0:
                    document.metadata["dots_ocr_structured"] = structured_data
            return documents
        # DotsOCR failed or not configured — use PyPDFLoader
        return _load_pdf_with_pypdf(path)
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

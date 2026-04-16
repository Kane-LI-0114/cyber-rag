from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document

_SPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINE_RE = re.compile(r"\n{3,}")
_DOMAIN_HINTS = {
    "cryptography": ("crypto", "cipher", "rsa", "aes", "hash"),
    "reverse-engineering": ("reverse", "disassembly", "binary", "assembly"),
    "network-security": ("network", "http", "tcp", "udp", "packet", "rfc"),
    "forensics": ("forensics", "memory dump", "artifact", "disk image"),
    "web-security": ("xss", "csrf", "sql injection", "cookie", "session"),
}


def _clean_text(text: str) -> str:
    lines = [_SPACE_RE.sub(" ", line).rstrip() for line in text.splitlines()]
    return _BLANK_LINE_RE.sub("\n\n", "\n".join(lines)).strip()


def _infer_security_domain(text: str, source: str) -> str:
    haystack = f"{source}\n{text}".lower()
    for domain, hints in _DOMAIN_HINTS.items():
        if any(hint in haystack for hint in hints):
            return domain
    return "general-cybersecurity"


def normalize_documents(documents: list[Document]) -> list[Document]:
    normalized: list[Document] = []
    for document in documents:
        cleaned = _clean_text(document.page_content)
        if not cleaned:
            continue

        metadata = dict(document.metadata)
        source_path = metadata.get("source_path") or metadata.get("source") or "unknown"
        metadata.setdefault("title", Path(str(source_path)).stem)

        # Preserve section from DotsOCR structured parsing if available
        metadata.setdefault("section", None)
        metadata.setdefault("page", metadata.get("page"))
        metadata.setdefault("security_domain", _infer_security_domain(cleaned, str(source_path)))

        # Preserve DotsOCR layout metadata for downstream chunking
        # (category, bbox, parser, layout_blocks, etc.)
        # These are already in metadata from the loader, so we just
        # make sure they survive the normalization step.

        normalized.append(Document(page_content=cleaned, metadata=metadata))
    return normalized

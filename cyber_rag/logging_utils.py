from __future__ import annotations

import logging

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=_DEFAULT_FORMAT)
    return logging.getLogger("cyber_rag")

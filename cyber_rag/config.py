from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
INDEXES_DIR = ARTIFACTS_DIR / "indexes"
EVALS_DIR = ARTIFACTS_DIR / "evals"
DEFAULT_INDEX_DIR = INDEXES_DIR / "default"
DEFAULT_EVAL_PATH = EVALS_DIR / "latest.csv"

load_dotenv(ENV_PATH)


def _read_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


@dataclass(slots=True)
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )


@dataclass(slots=True)
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize_embeddings: bool = True


@dataclass(slots=True)
class RetrievalConfig:
    k: int = 4
    search_type: str = "similarity"


@dataclass(slots=True)
class GenerationConfig:
    provider: str = field(
        default_factory=lambda: _read_env("CYBER_RAG_LLM_PROVIDER") or "azure"
    )
    model_name: str | None = None
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None
    api_version: str | None = None

    def __post_init__(self) -> None:
        if self.provider == "azure":
            self.model_name = self.model_name or _read_env("CYBER_RAG_AZURE_MODEL_NAME")
            self.api_key = self.api_key or _read_env("CYBER_RAG_AZURE_API_KEY")
            self.base_url = self.base_url or _read_env("CYBER_RAG_AZURE_BASE_URL")
            self.api_version = self.api_version or _read_env("CYBER_RAG_AZURE_API_VERSION")
        elif self.provider == "oneapi":
            self.model_name = self.model_name or _read_env("CYBER_RAG_ONEAPI_MODEL_NAME")
            self.api_key = self.api_key or _read_env("CYBER_RAG_ONEAPI_API_KEY")
            self.base_url = self.base_url or _read_env("CYBER_RAG_ONEAPI_BASE_URL")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
def ensure_project_directories() -> None:
    for path in (RAW_DATA_DIR, INDEXES_DIR, EVALS_DIR):
        path.mkdir(parents=True, exist_ok=True)

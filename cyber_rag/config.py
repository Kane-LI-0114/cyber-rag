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

    @property
    def is_configured(self) -> bool:
        """Check if the current provider has all required credentials."""
        return bool(self.api_key and self.base_url and self.model_name)


def get_current_provider() -> str:
    """Get the currently configured LLM provider."""
    return _read_env("CYBER_RAG_LLM_PROVIDER") or "azure"


def print_config_status() -> None:
    """Print the current configuration status for debugging."""
    provider = get_current_provider()
    config = GenerationConfig()
    print(f"Current Provider: {provider.upper()}")
    print(f"Model: {config.model_name or 'NOT SET'}")
    print(f"Base URL: {config.base_url or 'NOT SET'}")
    print(f"API Key: {'***' + config.api_key[-8:] if config.api_key else 'NOT SET'}")
    print(f"Configured: {'Yes' if config.is_configured else 'No'}")


@dataclass(slots=True)
class DotsOcrConfig:
    """Configuration for DotsOCR PDF parsing via OneAPI.

    Falls back to the project's OneAPI credentials when dedicated
    DotsOCR environment variables are not set.
    """

    endpoint: str = ""
    api_key: str | None = None
    model_name: str = "DotsOCR"
    temperature: float = 0.0
    max_tokens: int = 7000
    top_p: float = 1.0
    timeout: int = 120
    max_workers: int = 10
    max_retries: int = 2
    retry_delay: float = 1.0
    page_scale: float = 2.0

    def __post_init__(self) -> None:
        # Resolve endpoint: dedicated var > OneAPI base_url + path > empty
        ep = _read_env("CYBER_RAG_DOTS_OCR_ENDPOINT")
        if not ep:
            base = _read_env("CYBER_RAG_ONEAPI_BASE_URL")
            if base:
                ep = f"{base.rstrip('/')}/chat/completions"
        if ep:
            if not ep.endswith("/chat/completions"):
                ep = f"{ep.rstrip('/')}/chat/completions"
            self.endpoint = ep

        # Resolve API key: dedicated var > OneAPI key
        if not self.api_key:
            self.api_key = _read_env("CYBER_RAG_DOTS_OCR_API_KEY") or _read_env(
                "CYBER_RAG_ONEAPI_API_KEY"
            )

        # Resolve model name
        env_model = _read_env("CYBER_RAG_DOTS_OCR_MODEL")
        if env_model:
            self.model_name = env_model

    @property
    def is_configured(self) -> bool:
        """Check if DotsOCR has all required credentials."""
        return bool(self.api_key and self.endpoint and self.model_name)


# The DotsOCR prompt used for structured layout extraction
DOTS_OCR_PROMPT = """
Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as Markdown.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""


def ensure_project_directories() -> None:
    for path in (RAW_DATA_DIR, INDEXES_DIR, EVALS_DIR):
        path.mkdir(parents=True, exist_ok=True)

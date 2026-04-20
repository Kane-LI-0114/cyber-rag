"""CyberRAG Web API server with streaming LLM responses."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Query, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from langchain_core.prompts import ChatPromptTemplate

from cyber_rag.config import (
    DEFAULT_INDEX_DIR,
    ROOT_DIR,
    EmbeddingConfig,
    GenerationConfig,
    RetrievalConfig,
)
from cyber_rag.generation.chain import (
    _build_llm,
    _format_context,
    _to_chunk_references,
    _SYSTEM_PROMPT,
    _HUMAN_PROMPT,
    _BASELINE_PROMPT,
    _MC_HUMAN_PROMPT,
    _MC_BASELINE_PROMPT,
    _prompt_question_for_llm,
)
from cyber_rag.retrieval.retriever import retrieve_documents
from cyber_rag.evaluation.datasets import load_evaluation_examples
from cyber_rag.schemas import ChunkReference

app = FastAPI(title="CyberRAG Web UI")
logger = logging.getLogger(__name__)


class _NoCacheMiddleware(BaseHTTPMiddleware):
    """Prevent browser caching of HTML pages during development."""
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.endswith(".html"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


app.add_middleware(_NoCacheMiddleware)

# Serve static files
_STATIC_DIR = Path(__file__).parent / "static"


def _sources_to_dicts(sources: list[ChunkReference]) -> list[dict]:
    return [
        {
            "chunk_id": s.chunk_id,
            "source": s.source,
            "title": s.title,
            "page": s.page,
            "excerpt": s.excerpt,
        }
        for s in sources
    ]


def _candidate_eval_dataset_dirs() -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path | None) -> None:
        if path is None:
            return
        resolved = path.expanduser().resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(resolved)

    for env_name in ("CYBER_RAG_EVAL_DATASETS_DIR", "CYBER_RAG_DATASETS_DIR"):
        raw = os.getenv(env_name)
        if raw:
            add(Path(raw))

    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        add(base / "eval_datasets")

    add(ROOT_DIR / "eval_datasets")
    add(Path(__file__).resolve().parents[2] / "eval_datasets")
    return candidates


def _list_dataset_files(datasets_dir: Path) -> list[Path]:
    files = list(datasets_dir.glob("*.jsonl")) + list(datasets_dir.glob("*.csv"))
    return sorted(files, key=lambda path: path.name.lower())


def _resolve_eval_datasets_dir() -> Path | None:
    first_existing_dir: Path | None = None

    for candidate in _candidate_eval_dataset_dirs():
        if not candidate.exists() or not candidate.is_dir():
            continue
        if first_existing_dir is None:
            first_existing_dir = candidate
        if _list_dataset_files(candidate):
            return candidate

    if first_existing_dir is None:
        logger.warning(
            "No eval_datasets directory found. Searched: %s",
            ", ".join(str(path) for path in _candidate_eval_dataset_dirs()),
        )
    return first_existing_dir


def _resolve_dataset_path(filename: str) -> Path | None:
    requested = Path(filename)
    if requested.name != filename or requested.is_absolute() or ".." in requested.parts:
        return None

    datasets_dir = _resolve_eval_datasets_dir()
    if datasets_dir is None:
        return None

    path = datasets_dir / filename
    if path.exists() and path.is_file():
        return path
    return None


async def _stream_llm(llm, messages):
    """Stream tokens from an LLM, yielding SSE-formatted chunks."""
    collected = []
    try:
        async for chunk in llm.astream(messages):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                collected.append(token)
                data = json.dumps({"type": "token", "content": token}, ensure_ascii=False)
                yield f"data: {data}\n\n"
    except Exception as e:
        data = json.dumps({"type": "error", "content": str(e)}, ensure_ascii=False)
        yield f"data: {data}\n\n"
        return

    full_text = "".join(collected)
    data = json.dumps({"type": "done", "content": full_text}, ensure_ascii=False)
    yield f"data: {data}\n\n"


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/models")
async def list_models():
    """List available LLM models from configured providers."""
    models = []
    for env_prefix, provider_label in [
        ("CYBER_RAG_AZURE", "Azure"),
        ("CYBER_RAG_ONEAPI", "OneAPI"),
        ("CYBER_RAG_HUGGINGFACE", "HuggingFace"),
    ]:
        api_key = os.getenv(f"{env_prefix}_API_KEY", "")
        model_name = os.getenv(f"{env_prefix}_MODEL_NAME", "")
        if api_key and model_name:
            models.append({
                "provider": provider_label.lower(),
                "model": model_name,
                "label": f"{model_name} ({provider_label})",
            })
    return JSONResponse(models)


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload an evaluation dataset file (.jsonl or .csv)."""
    filename = file.filename or ""
    if not (filename.endswith(".jsonl") or filename.endswith(".csv")):
        return JSONResponse(
            {"detail": "Only .jsonl and .csv files are accepted"},
            status_code=400,
        )

    # Save to a temp location under eval_datasets
    datasets_dir = _resolve_eval_datasets_dir()
    if datasets_dir is None:
        # Create eval_datasets in project root
        datasets_dir = ROOT_DIR / "eval_datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)

    dest = datasets_dir / filename
    content = await file.read()
    dest.write_bytes(content)

    # Count questions
    examples = load_evaluation_examples(dest)
    return JSONResponse({"filename": filename, "count": len(examples)})


@app.get("/api/datasets/{filename}/questions")
async def get_dataset_questions(filename: str):
    """Get questions from a specific dataset."""
    path = _resolve_dataset_path(filename)
    if path is None:
        return JSONResponse({"error": "Dataset not found"}, status_code=404)
    examples = load_evaluation_examples(path)
    questions = []
    for ex in examples[:50]:  # limit for UI
        q = {"question": ex.question}
        if ex.answers:
            q["type"] = "multiple_choice"
            q["choices"] = ex.answers
            q["solution"] = ex.solution
        else:
            q["type"] = "short_answer"
            q["answer"] = ex.answer
        questions.append(q)
    return JSONResponse(questions)


@app.get("/api/query/stream")
async def stream_query(
    question: str = Query(...),
    index_path: str = Query(default=str(DEFAULT_INDEX_DIR)),
    k: int = Query(default=4),
    model: str | None = Query(default=None),
):
    """Stream a RAG query response with sources."""
    embedding_config = EmbeddingConfig()
    retrieval_config = RetrievalConfig(k=k)
    generation_config = GenerationConfig(model_name=model)

    # Step 1: Retrieve documents (non-streaming)
    documents = retrieve_documents(
        query=question,
        index_path=index_path,
        embedding_config=embedding_config,
        retrieval_config=retrieval_config,
    )
    sources = _to_chunk_references(documents)
    context = _format_context(documents)

    # Build LLM with streaming
    llm = _build_llm(generation_config)

    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_PROMPT), ("human", _HUMAN_PROMPT)]
    )
    formatted_question = _prompt_question_for_llm(question)
    messages = prompt.format_messages(context=context, question=formatted_question)

    async def generate() -> AsyncGenerator[str, None]:
        # Send sources first
        sources_data = json.dumps(
            {"type": "sources", "content": _sources_to_dicts(sources)},
            ensure_ascii=False,
        )
        yield f"data: {sources_data}\n\n"

        # Stream LLM tokens
        async for sse in _stream_llm(llm, messages):
            yield sse

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/eval/stream")
async def stream_eval(
    question: str = Query(...),
    index_path: str = Query(default=str(DEFAULT_INDEX_DIR)),
    k: int = Query(default=4),
    model: str | None = Query(default=None),
    answer_options: str | None = Query(default=None),
    reference_answer: str | None = Query(default=None),
):
    """Stream an evaluation: baseline then RAG, each with sources."""
    embedding_config = EmbeddingConfig()
    retrieval_config = RetrievalConfig(k=k)
    generation_config = GenerationConfig(model_name=model)

    # Parse answer_options if provided (JSON string)
    options_dict = None
    if answer_options:
        try:
            options_dict = json.loads(answer_options)
        except json.JSONDecodeError:
            pass

    is_mcq = bool(options_dict)
    llm = _build_llm(generation_config)

    # ---- BASELINE ----
    baseline_prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_PROMPT), ("human", _MC_BASELINE_PROMPT if is_mcq else _BASELINE_PROMPT)]
    )
    formatted_question = _prompt_question_for_llm(question, options_dict)
    baseline_messages = baseline_prompt.format_messages(question=formatted_question)

    # ---- RAG ----
    documents = retrieve_documents(
        query=question,
        index_path=index_path,
        embedding_config=embedding_config,
        retrieval_config=retrieval_config,
    )
    sources = _to_chunk_references(documents)
    context = _format_context(documents)

    rag_prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_PROMPT), ("human", _MC_HUMAN_PROMPT if is_mcq else _HUMAN_PROMPT)]
    )
    rag_messages = rag_prompt.format_messages(context=context, question=formatted_question)

    async def generate() -> AsyncGenerator[str, None]:
        # Send reference info
        ref_data = {
            "type": "reference",
            "content": {
                "question": question,
                "reference_answer": reference_answer,
                "question_type": "multiple_choice" if is_mcq else "short_answer",
                "choices": options_dict,
            },
        }
        yield f"data: {json.dumps(ref_data, ensure_ascii=False)}\n\n"

        # Baseline phase
        phase_data = json.dumps({"type": "phase", "content": "baseline"}, ensure_ascii=False)
        yield f"data: {phase_data}\n\n"

        async for sse in _stream_llm(llm, baseline_messages):
            yield sse

        # RAG phase
        phase_data = json.dumps({"type": "phase", "content": "rag"}, ensure_ascii=False)
        yield f"data: {phase_data}\n\n"

        # Send RAG sources
        sources_data = json.dumps(
            {"type": "sources", "content": _sources_to_dicts(sources)},
            ensure_ascii=False,
        )
        yield f"data: {sources_data}\n\n"

        async for sse in _stream_llm(llm, rag_messages):
            yield sse

        # Done
        done_data = json.dumps({"type": "complete"}, ensure_ascii=False)
        yield f"data: {done_data}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# Mount static files last (catch-all)
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

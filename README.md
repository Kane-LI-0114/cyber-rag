# CyberRAG

CyberRAG is an original course project for evaluating retrieval-augmented generation on cybersecurity knowledge questions and CTF-style tasks.

## Project goal

The project studies whether domain-specific retrieval improves LLM performance on cybersecurity benchmarks such as CyberMetric, CyberQA, and CTFNow. The current repository now includes a runnable first-pass framework for ingestion, chunking, indexing, retrieval, answer generation, and batch evaluation.

## Core design choices

- **LangChain-first orchestration** for loaders, text splitting, embeddings, retrievers, prompt assembly, and model integration.
- **Structure-aware document processing** that preserves page, section, source, and security-domain metadata before chunking.
- **Local-first retrieval indexing** built on FAISS for reproducible experiments.
- **Evidence-centric answer generation** so outputs can be traced back to retrieved chunks.
- **Evaluation-driven development** because the main deliverable is reproducible evidence, not only an interactive demo.
- **Original project framing**: repository files should describe the pipeline as this project's own design and should not name outside systems.

## Current framework layout

```text
cyber_rag/
├── cli/                  # Package-native CLI entrypoints
├── config.py             # Shared paths and runtime configs
├── schemas.py            # Answer, evidence, and evaluation dataclasses
├── ingest/               # Source loading for local files and web pages
├── processing/           # Normalization and recursive chunking
├── indexing/             # Local FAISS index build/load
├── retrieval/            # Retriever construction and query-time fetch
├── generation/           # Baseline and retrieval-grounded answer flows
└── evaluation/           # Dataset loading and batch evaluation

scripts/
├── build_index.py        # Thin compatibility wrapper for index building
├── run_query.py          # Thin compatibility wrapper for single-query runs
└── run_eval.py           # Thin compatibility wrapper for batch evaluation

tests/
└── test_chunking.py      # Metadata-preservation smoke test

pyproject.toml            # Standard package metadata and console scripts
```

## End-to-end pipeline

1. **Corpus ingestion**  
   Load cybersecurity references from `data/raw/`, specific files, folders, or URLs.

2. **Parsing and normalization**  
   Convert each source into normalized LangChain documents with provenance metadata such as source path, title, page, and security-domain hints.

3. **Structure-aware chunking**  
   Apply recursive chunk splitting with overlap while preserving metadata and assigning stable chunk IDs.

4. **Embedding and indexing**  
   Embed normalized chunks with a local sentence-transformer model and save the FAISS index plus a build manifest.

5. **Retrieval and generation**  
   Retrieve evidence, assemble a source-delimited prompt, and generate either baseline or retrieval-grounded answers.

6. **Evaluation**  
   Run batch comparisons over JSONL or CSV datasets and export a results table for later analysis.

## Environment setup

```bash
conda activate cyber-rag
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Runtime credentials

### Quick Start

```bash
# 1. Copy the template
cp .env.example .env

# 2. Edit .env with your API credentials
nano .env

# 3. Check configuration status
python -m cyber_rag.cli.check_config
```

### One-Key Provider Switching

The project supports switching between **Azure OpenAI** and **OneAPI** (OpenAI-compatible) providers:

```bash
# In .env, simply change this line:
CYBER_RAG_LLM_PROVIDER=azure    # Use Azure OpenAI
# or
CYBER_RAG_LLM_PROVIDER=oneapi   # Use OneAPI/OpenAI-compatible API

# No code changes needed - the system automatically loads the correct credentials
```

### Configuration Structure

```bash
# Provider Selection
CYBER_RAG_LLM_PROVIDER=azure        # or 'oneapi'

# Azure OpenAI Configuration (when provider=azure)
CYBER_RAG_AZURE_API_KEY=your_key
CYBER_RAG_AZURE_BASE_URL=https://xxx.openai.azure.com/openai
CYBER_RAG_AZURE_MODEL_NAME=gpt-4o-mini
CYBER_RAG_AZURE_API_VERSION=2024-10-21

# OneAPI Configuration (when provider=oneapi)
CYBER_RAG_ONEAPI_API_KEY=your_key
CYBER_RAG_ONEAPI_BASE_URL=https://api.example.com/v1
CYBER_RAG_ONEAPI_MODEL_NAME=deepseek-v3
```

Additional runtime defaults are configured in `cyber_rag/config.py`, including:
- chunking defaults (`chunk_size`, `chunk_overlap`)
- embedding model and normalization settings
- retrieval defaults (`k`, `search_type`)
- generation defaults (provider, model name, temperature)

## Development commands

### Text corpus and retrieval workflow

- Build a FAISS index from local text sources under `data/raw/`:
  ```bash
  python -m cyber_rag.cli.build_index data/raw --index-path artifacts/indexes/default
  ```
- Run a single retrieval-grounded question against an existing index:
  ```bash
  python -m cyber_rag.cli.run_query "What is the difference between symmetric and asymmetric encryption?"
  ```
- Run batch evaluation (baseline vs RAG) on a dataset and write CSV output:
  ```bash
  python -m cyber_rag.cli.run_eval path/to/eval.jsonl --index-path artifacts/indexes/default --output artifacts/evals/latest.csv
  ```

### Other development operations

- Run static checks for style and common issues:
  ```bash
  ruff check cyber_rag scripts tests
  ```
- Run the full unit test suite:
  ```bash
  pytest
  ```
- Run only the chunking metadata smoke test:
  ```bash
  pytest tests/test_chunking.py
  ```

## Supported starter inputs

The current ingestion layer supports:
- PDF
- Markdown
- Plain text
- HTML
- HTTP/HTTPS web pages

## Current repository status

This is a first-pass implementation scaffold. It is intentionally modular so later work can extend parsing depth, add richer retrieval strategies, support more model providers, and plug in fuller metric computation without rewriting the entire pipeline.

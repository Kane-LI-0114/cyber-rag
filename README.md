# CyberRAG

CyberRAG is an original course project for evaluating retrieval-augmented generation on cybersecurity knowledge questions and CTF-style tasks.

## Project goal

The project studies whether domain-specific retrieval improves LLM performance on cybersecurity benchmarks such as CyberMetric, CyberQA, and CTFNow. The current repository includes a complete framework for ingestion, chunking, indexing, retrieval, answer generation, and batch evaluation with support for both short-answer and multiple-choice questions.

## Core design choices

- **LangChain-first orchestration** for loaders, text splitting, embeddings, retrievers, prompt assembly, and model integration.
- **Structure-aware document processing** that preserves page, section, source, and security-domain metadata before chunking.
- **Vision-based PDF parsing** using DotsOCR for structured layout extraction (titles, sections, tables, formulas).
- **Local-first retrieval indexing** built on FAISS for reproducible experiments.
- **Evidence-centric answer generation** so outputs can be traced back to retrieved chunks.
- **Evaluation-driven development** because the main deliverable is reproducible evidence, not only an interactive demo.
- **Original project framing**: repository files should describe the pipeline as this project's own design and should not name outside systems.

## Current framework layout

```text
cyber_rag/
├── cli/                  # Package-native CLI entrypoints
│   ├── build_index.py    # Build FAISS index from documents
│   ├── run_query.py      # Single query with retrieval
│   ├── run_eval.py       # Batch evaluation (baseline vs RAG)
│   └── check_config.py   # Verify API configuration
├── config.py             # Shared paths and runtime configs
├── schemas.py            # Answer, evidence, and evaluation dataclasses
├── ingest/               # Source loading with structured PDF parsing
│   ├── loaders.py        # Main ingestion entry point
│   └── dots_ocr.py       # Vision-based PDF layout extraction
├── processing/           # Normalization and recursive chunking
│   ├── normalize.py      # Text cleaning and metadata enrichment
│   └── chunking.py       # Recursive splitting with chunk IDs
├── indexing/             # Local FAISS index build/load
│   └── faiss_store.py    # Embedding and vector store management
├── retrieval/            # Retriever construction and query-time fetch
│   └── retriever.py      # FAISS-based document retrieval
├── generation/           # Baseline and retrieval-grounded answer flows
│   └── chain.py          # LLM prompting and answer generation
└── evaluation/           # Dataset loading and batch evaluation
    ├── datasets.py       # JSONL/CSV evaluation data loading
    └── runner.py         # Baseline vs RAG comparison runner

scripts/
├── build_index.py        # Thin compatibility wrapper for index building
├── run_query.py          # Thin compatibility wrapper for single-query runs
├── run_eval.py           # Thin compatibility wrapper for batch evaluation
└── analyze_eval.py       # Evaluation results analysis script

tests/
└── test_chunking.py      # Metadata-preservation smoke test

pyproject.toml            # Standard package metadata and console scripts
```

## End-to-end pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CyberRAG Pipeline Overview                          │
└─────────────────────────────────────────────────────────────────────────────┘

1. CORPUS INGESTION
   ┌─────────────────────────────────────────────────────────────────────┐
   │  Inputs: PDF, Markdown, Text, HTML, URLs                            │
   │                                                                     │
   │  PDF Processing (DotsOCR):                                          │
   │    • Render pages to images                                         │
   │    • Vision-LM extracts structured layout (titles, sections, tables)│
   │    • Fallback to PyPDFLoader if DotsOCR unavailable                 │
   │                                                                     │
   │  Other Formats:                                                     │
   │    • TextLoader for .md, .txt                                       │
   │    • BSHTMLLoader for .html                                         │
   │    • WebBaseLoader for URLs                                         │
   └─────────────────────────────────────────────────────────────────────┘
                                    ↓
2. NORMALIZATION
   ┌─────────────────────────────────────────────────────────────────────┐
   │  • Clean whitespace and normalize text                              │
   │  • Enrich metadata: title, page, section, security_domain           │
   │  • Preserve DotsOCR layout blocks for downstream use                │
   └─────────────────────────────────────────────────────────────────────┘
                                    ↓
3. CHUNKING
   ┌─────────────────────────────────────────────────────────────────────┐
   │  • RecursiveCharacterTextSplitter with configurable separators      │
   │  • Preserve all metadata through splits                             │
   │  • Generate stable chunk IDs: {source}:{page}:{index}:{hash}        │
   └─────────────────────────────────────────────────────────────────────┘
                                    ↓
4. INDEXING
   ┌─────────────────────────────────────────────────────────────────────┐
   │  • HuggingFace sentence-transformer embeddings                      │
   │  • FAISS vector store for local, reproducible retrieval             │
   │  • Save index manifest with build parameters                        │
   └─────────────────────────────────────────────────────────────────────┘
                                    ↓
5. RETRIEVAL & GENERATION
   ┌─────────────────────────────────────────────────────────────────────┐
   │  Retrieval:                                                         │
   │    • Similarity search over FAISS index                             │
   │    • Return k most relevant chunks with full metadata               │
   │                                                                     │
   │  Generation (Two Modes):                                            │
   │    • Baseline: Direct LLM answering without retrieval               │
   │    • RAG: Context-augmented answering with retrieved evidence       │
   │                                                                     │
   │  Supported Question Types:                                          │
   │    • Short answer (free-form text)                                  │
   │    • Multiple choice (A/B/C/D selection with automatic extraction)  │
   └─────────────────────────────────────────────────────────────────────┘
```

## Environment setup

```bash
conda activate cyber-rag
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Runtime credentials

## Quick commands

A convenient `run.sh` script is provided for common operations:

```bash
# Environment setup
./run.sh setup              # Install/update dependencies
./run.sh check              # Check configuration status

# Index building
./run.sh build-index        # Build FAISS index (default settings)
./run.sh build-index-custom 500 100  # Custom chunk_size and overlap

# Query and evaluation
./run.sh query "What is XSS?"        # Single retrieval-grounded question
./run.sh eval CyberMetric-80-v1.jsonl  # Batch evaluation

# Evaluation analysis
./run.sh analyze                       # Quick summary of latest results
./run.sh analyze -v                    # Detailed text report
./run.sh analyze --report report.txt  # Save text report to file
./run.sh analyze --json summary.json   # Save JSON summary
./run.sh analyze -e rag_regressed      # Export RAG-regressed cases to CSV

# Testing
./run.sh test               # Run all tests
./run.sh test-chunking      # Run chunking tests only
./run.sh lint               # Code style check

# Help
./run.sh help               # Show all available commands
```

Alternatively, use the module-based commands directly:

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
# Note: base_url should NOT include /openai suffix - the SDK appends it automatically.
# For Azure API Management gateways, set base_url to the gateway root (e.g., https://xxx.azure-api.net).
CYBER_RAG_AZURE_API_KEY=your_key
CYBER_RAG_AZURE_BASE_URL=https://xxx.openai.azure.com
CYBER_RAG_AZURE_MODEL_NAME=gpt-4o-mini
CYBER_RAG_AZURE_API_VERSION=2024-10-21

# OneAPI Configuration (when provider=oneapi)
CYBER_RAG_ONEAPI_API_KEY=your_key
CYBER_RAG_ONEAPI_BASE_URL=https://api.example.com/v1
CYBER_RAG_ONEAPI_MODEL_NAME=deepseek-v3

# DotsOCR Configuration (optional, for enhanced PDF parsing)
# Falls back to OneAPI credentials if not explicitly set
CYBER_RAG_DOTS_OCR_ENDPOINT=https://api.example.com/v1/chat/completions
CYBER_RAG_DOTS_OCR_API_KEY=your_key
CYBER_RAG_DOTS_OCR_MODEL=DotsOCR
```

Additional runtime defaults are configured in `cyber_rag/config.py`, including:
- chunking defaults (`chunk_size`, `chunk_overlap`)
- embedding model and normalization settings
- retrieval defaults (`k`, `search_type`)
- generation defaults (provider, model name, temperature)
- DotsOCR settings (timeout, max_workers, page_scale)

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
- **PDF** - With optional DotsOCR for structured layout extraction
- **Markdown** - Preserved with formatting
- **Plain text** - UTF-8 encoded
- **HTML** - Parsed with BeautifulSoup
- **HTTP/HTTPS web pages** - Fetched and parsed at runtime

## DotsOCR PDF Parsing

The system includes an optional vision-based PDF parsing module (`dots_ocr.py`) that uses a vision-language model to extract structured layout information:

### Features
- **Layout Categories**: Title, Section-header, Text, Table, Formula, List-item, Caption, Picture, etc.
- **Bounding Box Extraction**: Precise coordinates for each element
- **Reading Order Preservation**: Elements sorted by human reading order
- **Formula Handling**: LaTeX formatting for mathematical content
- **Table Extraction**: Markdown table formatting
- **Concurrent Processing**: Multi-threaded page processing for speed

### Configuration
DotsOCR automatically uses your OneAPI credentials if dedicated environment variables are not set:

```bash
# Option 1: Use OneAPI credentials (recommended)
CYBER_RAG_ONEAPI_API_KEY=your_key
CYBER_RAG_ONEAPI_BASE_URL=https://api.example.com/v1

# Option 2: Dedicated DotsOCR credentials
CYBER_RAG_DOTS_OCR_ENDPOINT=https://api.example.com/v1/chat/completions
CYBER_RAG_DOTS_OCR_API_KEY=your_key
CYBER_RAG_DOTS_OCR_MODEL=DotsOCR
```

### Fallback Behavior
If DotsOCR is not configured or fails, the system automatically falls back to PyPDFLoader for standard text extraction.

## Evaluation Dataset Format

The system supports two evaluation dataset formats:

### Multiple Choice (JSONL)
```json
{
  "question": "What is the primary purpose of a firewall?",
  "answers": {
    "A": "To encrypt data",
    "B": "To filter network traffic",
    "C": "To store passwords"
  },
  "solution": "B"
}
```

### Short Answer (JSONL)
```json
{
  "question": "Explain the difference between symmetric and asymmetric encryption.",
  "answer": "Symmetric encryption uses the same key for encryption and decryption..."
}
```

### CSV Format
```csv
question,answer
"What is XSS?","Cross-site scripting is a security vulnerability..."
```

## Current repository status

This is a functional implementation for cybersecurity RAG evaluation. The modular architecture supports:
- Extending parsing depth with additional layout analyzers
- Adding hybrid retrieval strategies (sparse + dense)
- Supporting additional model providers
- Implementing richer evaluation metrics
- Adding reranking and query expansion

All without rewriting the core pipeline.

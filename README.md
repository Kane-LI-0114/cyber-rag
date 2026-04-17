# CyberRAG

CyberRAG is an original course project for evaluating retrieval-augmented generation on cybersecurity knowledge questions and CTF-style tasks.

## Project goal

The project studies whether domain-specific retrieval improves LLM performance on cybersecurity benchmarks such as CyberMetric, CyberQA, and CTFNow. The current repository includes a complete framework for ingestion, chunking, indexing, retrieval, answer generation, and batch evaluation with support for both short-answer and multiple-choice questions.

## Core design choices

- **LangChain-first orchestration** for loaders, text splitting, embeddings, retrievers, prompt assembly, and model integration.
- **Structure-aware document processing** that preserves page, section, source, and security-domain metadata before chunking.
- **Multi-engine PDF parsing**: Vision-based DotsOCR for scanned PDFs and PyMuPDF for embedded text PDFs with automatic detection.
- **Multi-provider LLM support**: Cloud APIs (Azure OpenAI, OneAPI, HuggingFace Inference API) and local HuggingFace models (Mistral-7B).
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
├── logging_utils.py      # Unified logging configuration
├── ingest/               # Source loading with structured PDF parsing
│   ├── loaders.py        # Main ingestion entry point
│   ├── dots_ocr.py       # Vision-based PDF layout extraction (DotsOCR)
│   └── pymupdf_parser.py # Fast PDF parsing with embedded text detection
├── processing/           # Normalization and recursive chunking
│   ├── normalize.py      # Text cleaning and metadata enrichment
│   └── chunking.py       # Recursive splitting with chunk IDs
├── indexing/             # Local FAISS index build/load
│   └── faiss_store.py    # Embedding and vector store management
├── retrieval/            # Retriever construction and query-time fetch
│   └── retriever.py      # FAISS-based document retrieval
├── generation/           # Baseline and retrieval-grounded answer flows
│   ├── chain.py          # LLM prompting and answer generation (Azure/OneAPI/HuggingFace)
│   └── local_llm.py      # Local HuggingFace LLM support (Mistral-7B)
└── evaluation/           # Dataset loading and batch evaluation
    ├── datasets.py       # JSONL/CSV evaluation data loading
    └── runner.py         # Baseline vs RAG comparison runner

scripts/
├── build_index.py        # Thin compatibility wrapper for index building
├── run_query.py          # Thin compatibility wrapper for single-query runs
├── run_eval.py           # Thin compatibility wrapper for batch evaluation
└── analyze_eval.py       # Evaluation results analysis and reporting

eval_datasets/            # Evaluation datasets folder
├── CyberMetric-01-v1.jsonl
├── CyberMetric-01-v2.jsonl
├── CyberMetric-80-v1.jsonl
├── CyberMetric-500-v1.jsonl
├── SecQA.jsonl
├── ctfknow_multiple_choice.jsonl
├── ctfknow_short_answer.jsonl
└── test.jsonl

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
./run.sh eval CM-80                   # Batch evaluation (CyberMetric-80)
./run.sh eval CM-500                  # Batch evaluation (CyberMetric-500)
./run.sh eval SecQA                   # Batch evaluation (SecQA dataset)
./run.sh eval CTF-MC                  # Batch evaluation (CTFKnow multiple choice)
./run.sh eval CTF-SA                  # Batch evaluation (CTFKnow short answer)
./run.sh eval eval_datasets/test.jsonl  # Direct path also works

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

### Evaluation Datasets

All evaluation datasets are organized in the `eval_datasets/` folder:

| Alias | Dataset | Description |
|-------|---------|-------------|
| `CM-01-v1` | CyberMetric-01-v1.jsonl | 1 question (test) |
| `CM-01-v2` | CyberMetric-01-v2.jsonl | 1 question (variant) |
| `CM-80` | CyberMetric-80-v1.jsonl | 80 questions |
| `CM-500` | CyberMetric-500-v1.jsonl | 500 questions |
| `SecQA` | SecQA.jsonl | Security QA dataset |
| `CTF-MC` | ctfknow_multiple_choice.jsonl | CTF multiple choice |
| `CTF-SA` | ctfknow_short_answer.jsonl | CTF short answer |
| `test` | test.jsonl | Test dataset |

Use `./run.sh eval <alias>` for quick evaluation, or provide full path like `./run.sh eval eval_datasets/CyberMetric-80-v1.jsonl`.

### One-Key Provider Switching

The project supports switching between **Azure OpenAI**, **OneAPI** (OpenAI-compatible), **HuggingFace Inference API**, and **Local HuggingFace** models:

```bash
# In .env, simply change this line:
CYBER_RAG_LLM_PROVIDER=azure         # Use Azure OpenAI
# or
CYBER_RAG_LLM_PROVIDER=oneapi        # Use OneAPI/OpenAI-compatible API
# or
CYBER_RAG_LLM_PROVIDER=huggingface   # Use HuggingFace Inference API
# or
CYBER_RAG_LLM_PROVIDER=local         # Use local HuggingFace model (Mistral-7B)

# No code changes needed - the system automatically loads the correct credentials
```

**Local LLM Support** (`generation/local_llm.py`):
- Automatic device detection (CUDA/MPS/CPU)
- HuggingFace transformers integration (Mistral-7B-Instruct)
- Configurable generation parameters (temperature, top_p, repetition_penalty)
- Same API as cloud functions: `answer_with_retrieval_local()` and `answer_without_retrieval_local()`

### Configuration Structure

```bash
# Provider Selection
CYBER_RAG_LLM_PROVIDER=azure        # or 'oneapi' or 'huggingface'

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

# HuggingFace Configuration (when provider=huggingface)
# Get your token from: https://huggingface.co/settings/tokens
CYBER_RAG_HUGGINGFACE_API_KEY=hf_your_token_here
CYBER_RAG_HUGGINGFACE_BASE_URL=https://router.huggingface.co/v1
CYBER_RAG_HUGGINGFACE_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2:featherless-ai

# Judge Model Configuration (for short-answer evaluation)
# Falls back to main LLM provider if not explicitly set
# Recommended: use a cheaper model for judging (e.g., gpt-4o-mini for answer, gpt-4o for judge)
# Provider selection
CYBER_RAG_JUDGE_LLM_PROVIDER=azure        # or 'oneapi' or 'huggingface'

# Judge Azure OpenAI
CYBER_RAG_JUDGE_AZURE_API_KEY=your_key
CYBER_RAG_JUDGE_AZURE_BASE_URL=https://xxx.openai.azure.com
CYBER_RAG_JUDGE_AZURE_MODEL_NAME=gpt-4o-mini
CYBER_RAG_JUDGE_AZURE_API_VERSION=2024-10-21

# Judge OneAPI
CYBER_RAG_JUDGE_ONEAPI_API_KEY=your_key
CYBER_RAG_JUDGE_ONEAPI_BASE_URL=https://api.example.com/v1
CYBER_RAG_JUDGE_ONEAPI_MODEL_NAME=gpt-4o-mini

# Judge HuggingFace
CYBER_RAG_JUDGE_HUGGINGFACE_API_KEY=hf_your_token_here
CYBER_RAG_JUDGE_HUGGINGFACE_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2

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
  python -m cyber_rag.cli.run_eval eval_datasets/CyberMetric-80-v1.jsonl --index-path artifacts/indexes/default
  ```

  By default, evaluation files are automatically named with timestamps in the format `artifacts/evals/eval_YYYYMMDD_HHMMSS.csv`. To specify a custom output path:
  ```bash
  python -m cyber_rag.cli.run_eval eval_datasets/CyberMetric-80-v1.jsonl --output artifacts/evals/my_experiment.csv
  ```
  For short-answer datasets, an LLM judge scores each answer in **[0, 1]** as
  `baseline_judge_accuracy` and `rag_judge_accuracy`. Optional columns
  `baseline_correct` / `rag_correct` are True when the score is at least
  `--judge-threshold` (default `0.5`). Example with a separate judge model:
  ```bash
  python -m cyber_rag.cli.run_eval eval_datasets/ctfknow_short_answer.jsonl \
    --model deepseek-v3 \
    --judge-model gpt-4o-mini \
    --judge-threshold 0.5
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

## PDF Parsing

The system includes two complementary PDF parsing engines:

### DotsOCR Vision-based Parsing (`dots_ocr.py`)

A vision-language model-based parser for scanned PDFs and complex layouts:

**Features**:
- **Layout Categories**: Title, Section-header, Text, Table, Formula, List-item, Caption, Picture, etc.
- **Bounding Box Extraction**: Precise coordinates for each element
- **Reading Order Preservation**: Elements sorted by human reading order
- **Formula Handling**: LaTeX formatting for mathematical content
- **Table Extraction**: Markdown table formatting
- **Concurrent Processing**: Multi-threaded page processing for speed

**Configuration**:
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

### PyMuPDF Parser (`pymupdf_parser.py`)

A fast parser for PDFs with embedded text, featuring intelligent fallback logic:

**Features**:
- **Embedded Text Detection**: Automatically detects whether a PDF contains extractable embedded text
- **Sampling-based Detection**: Efficiently checks PDFs by sampling pages (15% for long documents, all pages for short ones)
- **Font Metadata Analysis**: Classifies text spans based on font size and style (Title, Section-header, List-item, Text)
- **Table Detection**: Uses PyMuPDF's built-in table detection with markdown export
- **DotsOCR-compatible Output**: Structured layout blocks aligned with DotsOCR's format for seamless downstream processing

**Fallback Logic**:
The loader automatically selects the appropriate parser:
1. Use PyMuPDF if PDF contains embedded text (fast path)
2. Fall back to DotsOCR for scanned/image-based PDFs
3. Final fallback to PyPDFLoader if both fail

This multi-tier approach ensures optimal parsing speed while maintaining high quality output.

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

## Evaluation Analysis

The system includes a comprehensive evaluation analysis script (`scripts/analyze_eval.py`) for deeper insights into baseline vs RAG performance:

### Analysis Dimensions

The analysis module provides **9 major analysis dimensions**:

#### 1. Basic Accuracy Metrics
- Overall and per-question-type (MCQ/Short Answer) accuracy
- Baseline vs RAG comparison
- Skipped/error tracking

#### 2. Result Categorization
Classifies questions into four categories:
- **Both Correct**: Baseline and RAG both answered correctly
- **RAG Improved**: Baseline failed, RAG succeeded
- **RAG Regressed**: Baseline succeeded, RAG failed
- **Both Wrong**: Neither method succeeded

#### 3. Retrieval Quality Analysis
- Chunk count distribution (mean, median, std, quartiles)
- Zero-chunk case detection
- Chunk count correlation with accuracy
- Source diversity analysis (if sources are tracked)

#### 4. Question Difficulty Profiling
- **Easy**: Both Correct rate (questions both systems can answer)
- **Medium**: RAG Improved only (questions requiring retrieval)
- **Hard**: Both Wrong rate (questions neither system can solve)
- **Retrieval Trap**: RAG Regressed (questions where retrieval hurts)
- Overall difficulty score (0-100 normalized)

#### 5. Answer Quality Deep Analysis
**MCQ Analysis**:
- Option distribution (A/B/C/D) for baseline and RAG
- Reference answer distribution
- Model bias detection (>10% deviation from expected 25%)
- Answer change patterns

**Short Answer Analysis**:
- Answer length statistics (mean, median, std)
- Judge score distribution (0-1)
- Boundary sample detection (scores near 0.5 threshold)

#### 6. Error Pattern Mining
- RAG Regressed root cause analysis
  - Chunk count correlation
  - High-chunk-but-still-wrong detection (possible noise/conflict)
- Both Wrong analysis (corpus coverage vs reasoning issues)
- Answer change effectiveness (net effect of RAG changes)

#### 7. Statistical Significance Testing
- **McNemar's Test**: Chi-square with continuity correction
- P-value calculation (α = 0.05, 0.01)
- **Effect Size**: Cohen's h interpretation
- **95% Confidence Interval**: For proportion difference
- Significance interpretation

#### 8. Cross-Dimensional Analysis
- Question Type × Result Classification matrix
- Chunk Count × Accuracy correlation
- Answer Length × Judge Score correlation
- Judge score distribution buckets (Very Low → Excellent)

#### 9. Sample Case Export
- Export specific error categories to CSV for targeted analysis

### Usage

```bash
# Quick summary (recommended first step)
./run.sh analyze -q

# Full comprehensive report (recommended)
./run.sh analyze -v

# Quick summary of latest results
./run.sh analyze

# Save text report to file
./run.sh analyze --report report.txt

# Save JSON summary (for programmatic analysis)
./run.sh analyze --json summary.json

# Export RAG-regressed cases for targeted analysis
./run.sh analyze -e rag_regressed --export-path regressed_cases.csv

# Export all both-wrong cases for corpus improvement
./run.sh analyze -e both_wrong --export-path hard_cases.csv
```

### Python API

```python
from scripts.analyze_eval import (
    # Data loading
    load_evaluation_data,
    
    # Core analysis functions
    calculate_accuracy_metrics,
    categorize_results,
    analyze_retrieval_quality,
    analyze_question_difficulty,
    analyze_answer_quality,
    mine_error_patterns,
    calculate_statistical_significance,
    analyze_cross_dimensions,
    
    # Report generation
    generate_comprehensive_report,
    generate_json_summary,
    export_error_cases,
    
    # Utility
    get_error_cases,
)

# Load and analyze
df = load_evaluation_data("artifacts/evals/eval_20260417.csv")
metrics = calculate_accuracy_metrics(df)
stats = calculate_statistical_significance(df)

# Generate full report
report = generate_comprehensive_report(
    df, metrics, categories,
    retrieval_quality, difficulty,
    answer_quality, error_patterns,
    stats, cross_analysis
)
```

## Current repository status

This is a functional implementation for cybersecurity RAG evaluation. The modular architecture supports:
- Extending parsing depth with additional layout analyzers
- Adding hybrid retrieval strategies (sparse + dense)
- Supporting additional model providers (Azure, OneAPI, HuggingFace Inference API)
- Implementing richer evaluation metrics
- Adding reranking and query expansion

All without rewriting the core pipeline.

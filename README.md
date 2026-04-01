# CyberRAG

CyberRAG is an original course project for evaluating retrieval-augmented generation on cybersecurity knowledge questions and CTF-style tasks.

## Project goal

The project studies whether domain-specific retrieval improves LLM performance on cybersecurity benchmarks such as CyberMetric, CyberQA, and CTFNow. The intended system combines a structured document-processing pipeline, LangChain-based orchestration, and an evaluation layer that compares retrieval-enabled runs against no-retrieval baselines.

## Core design choices

- **LangChain-first orchestration** for loaders, text splitting, embeddings, retrievers, prompt assembly, and model integration.
- **Structure-aware document processing** that preserves page, section, source, and security-domain metadata before chunking.
- **Evidence-centric retrieval** so answers can be traced back to specific chunks instead of opaque context blobs.
- **Evaluation-driven development** because the main deliverable is reproducible evidence, not only an interactive demo.
- **Original project framing**: repository files should describe the pipeline as this project's own design and should not name outside systems.

## Planned pipeline

1. **Corpus ingestion**  
   Load cybersecurity references from PDFs, web pages, markdown files, and related technical documents.

2. **Parsing and normalization**  
   Convert each source into a canonical internal document object with provenance metadata such as title, page, section, source path, and security domain.

3. **Structure-aware chunking**  
   Preserve headings, tables, lists, and other meaningful boundaries before applying recursive chunk splitting with overlap.

4. **Embedding and indexing**  
   Embed normalized chunks and store them in a local retrieval index for reproducible experiments.

5. **Retrieval and generation**  
   Use LangChain to assemble prompts from retrieved evidence, then generate answers with explicit source grounding.

6. **Evaluation**  
   Compare baseline and RAG-enabled runs using answer quality, retrieval quality, and RAG-focused metrics.

## Environment setup

```bash
conda create -n cyber-rag python=3.11 -y
conda activate cyber-rag
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Optional environment variables

If you use hosted model providers or tracing, set the corresponding variables before running experiments.

```bash
export OPENAI_API_KEY=your_api_key
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=your_langsmith_api_key
```

## Development commands

```bash
ruff check .
pytest
pytest tests/test_*.py -k <keyword>
python -c "import langchain, faiss, pypdf, ragas; print('env ok')"
```

## Current repository status

The repository is still in the bootstrap stage. It currently contains project proposal material and initialization documents, but the main Python package, indexing jobs, retrieval flows, and evaluation scripts still need to be implemented.

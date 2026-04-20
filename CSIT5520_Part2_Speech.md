# CyberRAG — Part II: System Implementation

## Slide 1: System Architecture Overview

CyberRAG follows a four-stage pipeline. **INGEST** parses PDFs via PyMuPDF and DotsOCR. **INDEX** converts documents into searchable embeddings. **RETRIEVE** matches user queries against the index to return relevant chunks. **GENERATE** produces grounded, citation-traceable answers from retrieved evidence.

## Slide 2: Document Ingestion & Knowledge Base

The retrieval corpus contains 53 domain-specific PDFs covering cryptography, network security, forensics, and related topics. Parsing preserves section headers, table boundaries, and page provenance. Chunking applies recursive splitting with configurable separators: it tries paragraph breaks first, then lines, sentences, and word boundaries only when necessary. Key metadata — source file, page number, sequence index, and content hash — is embedded in each chunk ID, enabling full traceability back to the original document. Overlapping windows of 200 characters between adjacent chunks prevent key information from being split at chunk boundaries.

## Slide 3: Dense Retrieval

Documents are embedded using a SentenceTransformer model (all-MiniLM-L6-v2, 384-dim) and stored in a FAISS FLAT index for fast cosine similarity search. Retrieval accepts a top-K parameter to control the number of returned chunks, with a default of K equals 4. The index is built entirely offline with zero cloud dependency, and any domain-specific knowledge base can be swapped in by simply rebuilding the index without retraining the embedding model. Each chunk carries its source citation so that any generated answer can be traced back to the original document.

## Slide 4: Generation Layer

The system implements RAG-Sequence, concatenating retrieved chunks as a flat context window for single-pass generation. Two answer paths run side-by-side: baseline mode relies on parametric memory only, while RAG mode injects retrieved evidence. The evidence-as-evidence design treats retrieved passages as data, never instructions, preventing prompt injection attacks. Every chunk carries source citation enabling full traceability.

## Slide 5: Evaluation Framework

The system compares baseline versus RAG across CyberMetric MCQ and CTFKnow datasets, totaling 80+ questions in cryptography, network security, and forensics. Evaluation uses exact match for MCQ and LLM-as-judge scoring with threshold 0.5 for short answers. Results are categorized into: both correct, RAG improved, RAG regressed, and both wrong. Analysis covers task-level recall, RAG harm rate, knowledge gap coverage, and F1 score.

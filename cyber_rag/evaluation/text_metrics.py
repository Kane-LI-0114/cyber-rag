from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np


_WS_RE = re.compile(r"\s+")


def normalize_answer_text(text: str) -> str:
    """Lowercased, collapsed whitespace; remove most punctuation for robust matching."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def exact_match(prediction: str | None, reference: str | None) -> float:
    if reference is None or prediction is None:
        return math.nan
    return 1.0 if normalize_answer_text(str(prediction)) == normalize_answer_text(str(reference)) else 0.0


def token_f1(prediction: str | None, reference: str | None) -> float:
    """Token-level F1 over whitespace tokens after normalization (SQuAD-style overlap)."""
    if reference is None or prediction is None:
        return math.nan
    pred_tokens = normalize_answer_text(str(prediction)).split()
    ref_tokens = normalize_answer_text(str(reference)).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    overlap = sum((pred_counter & ref_counter).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def cosine_similarity_texts(
    prediction: str | None,
    reference: str | None,
    embedder,
) -> float:
    """Cosine similarity between sentence embeddings of prediction vs reference."""
    if reference is None or prediction is None:
        return math.nan
    a = str(reference).strip()
    b = str(prediction).strip()
    if not a or not b:
        return math.nan
    va = np.asarray(embedder.embed_query(a), dtype=float)
    vb = np.asarray(embedder.embed_query(b), dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return math.nan
    return float(np.dot(va, vb) / denom)

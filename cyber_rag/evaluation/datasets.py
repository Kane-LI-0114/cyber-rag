from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cyber_rag.schemas import EvaluationExample


def load_evaluation_examples(path: str | Path) -> list[EvaluationExample]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if dataset_path.suffix.lower() == ".jsonl":
        examples: list[EvaluationExample] = []
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)

                # Fields with dedicated schema slots
                _KNOWN = {"question", "answer", "answers", "solution"}

                examples.append(
                    EvaluationExample(
                        question=payload["question"],
                        answer=payload.get("answer"),
                        answers=payload.get("answers"),       # ← explicit MCQ choices
                        solution=payload.get("solution"),     # ← explicit MCQ key
                        metadata={
                            key: value
                            for key, value in payload.items()
                            if key not in _KNOWN
                        },
                    )
                )
        return examples

    if dataset_path.suffix.lower() == ".csv":
        frame = pd.read_csv(dataset_path)
        return [
            EvaluationExample(
                question=str(row["question"]),
                answer=None if pd.isna(row.get("answer")) else str(row.get("answer")),
                answers=None,    # CSV format does not support MCQ choices
                solution=None,
                metadata={
                    column: row[column]
                    for column in frame.columns
                    if column not in {"question", "answer"}
                },
            )
            for _, row in frame.iterrows()
        ]

    raise ValueError("Supported evaluation formats are .jsonl and .csv")
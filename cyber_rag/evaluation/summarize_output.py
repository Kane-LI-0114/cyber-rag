from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from cyber_rag.config import EVALS_DIR, ROOT_DIR, GenerationConfig


def _skipped_mask(df: pd.DataFrame) -> pd.Series:
    if "eval_skipped" not in df.columns:
        return pd.Series(False, index=df.index)
    s = df["eval_skipped"]
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin(("true", "1", "yes"))


def _completed_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[~_skipped_mask(df)]


def summarize_eval_dataframe(df: pd.DataFrame) -> dict:
    """Per-file metrics: MCQ accuracies; short-answer mean judge scores."""
    total_rows = len(df)
    skipped = int(_skipped_mask(df).sum()) if "eval_skipped" in df.columns else 0
    done = _completed_rows(df)

    mcq = done[done["question_type"] == "multiple_choice"]
    short = done[done["question_type"] == "short_answer"]

    mcq_n = len(mcq)
    short_n = len(short)

    row: dict = {
        "total_rows": total_rows,
        "evaluated_count": len(done),
        "skipped_count": skipped,
        "mcq_count": mcq_n,
        "mcq_baseline_accuracy": math.nan,
        "mcq_rag_accuracy": math.nan,
        "short_answer_count": short_n,
        "short_baseline_mean_accuracy": math.nan,
        "short_rag_mean_accuracy": math.nan,
    }

    if mcq_n > 0:
        row["mcq_baseline_accuracy"] = float(mcq["baseline_correct"].astype(float).mean())
        row["mcq_rag_accuracy"] = float(mcq["rag_correct"].astype(float).mean())

    if short_n > 0:
        if "baseline_judge_accuracy" in short.columns:
            row["short_baseline_mean_accuracy"] = float(
                short["baseline_judge_accuracy"].mean()
            )
        else:
            row["short_baseline_mean_accuracy"] = float(
                short["baseline_correct"].astype(float).mean()
            )
        if "rag_judge_accuracy" in short.columns:
            row["short_rag_mean_accuracy"] = float(short["rag_judge_accuracy"].mean())
        else:
            row["short_rag_mean_accuracy"] = float(
                short["rag_correct"].astype(float).mean()
            )

    row["summary_note"] = (
        f"说明: 本文件共 {total_rows} 条结果，有效评测 {len(done)} 题"
        f"（选择题 {mcq_n}，简答题 {short_n}），跳过 {skipped} 题。"
    )
    return row


def _relative_output_path(output_path: Path) -> str:
    try:
        return str(output_path.resolve().relative_to(ROOT_DIR.resolve()))
    except ValueError:
        return str(output_path.resolve())


def append_eval_summary_to_overall(
    output_path: Path,
    frame: pd.DataFrame,
    overall_path: Path | None = None,
    *,
    answer_provider: str | None = None,
    baseline_answer_model: str | None = None,
    rag_answer_model: str | None = None,
    judge_provider: str | None = None,
    judge_model: str | None = None,
) -> Path:
    """Append one summary row for this eval CSV to ``overall.csv``."""
    overall = overall_path or (EVALS_DIR / "overall.csv")
    overall.parent.mkdir(parents=True, exist_ok=True)

    summary = summarize_eval_dataframe(frame)
    summary["output_file"] = _relative_output_path(Path(output_path))
    gen = GenerationConfig()
    summary["answer_provider"] = answer_provider or gen.provider or ""
    summary["baseline_answer_model"] = baseline_answer_model or gen.model_name or ""
    summary["rag_answer_model"] = rag_answer_model or gen.model_name or ""
    summary["judge_provider"] = judge_provider or ""
    summary["judge_model"] = judge_model or ""

    columns_order = [
        "output_file",
        "answer_provider",
        "baseline_answer_model",
        "rag_answer_model",
        "judge_provider",
        "judge_model",
        "total_rows",
        "evaluated_count",
        "skipped_count",
        "mcq_count",
        "mcq_baseline_accuracy",
        "mcq_rag_accuracy",
        "short_answer_count",
        "short_baseline_mean_accuracy",
        "short_rag_mean_accuracy",
        "summary_note",
    ]
    new_row = pd.DataFrame([{k: summary[k] for k in columns_order}])

    if overall.exists():
        existing = pd.read_csv(overall)
        for col in columns_order:
            if col not in existing.columns:
                existing[col] = pd.NA
        extra_cols = [c for c in existing.columns if c not in columns_order]
        existing = existing[columns_order + extra_cols]
        for col in extra_cols:
            new_row[col] = pd.NA
        new_row = new_row[columns_order + extra_cols]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    combined.to_csv(overall, index=False, encoding="utf-8-sig")

    parts: list[str] = []
    if mcq_n := summary["mcq_count"]:
        parts.append(
            f"MCQ (n={mcq_n}): baseline={summary['mcq_baseline_accuracy']:.2%}, "
            f"RAG={summary['mcq_rag_accuracy']:.2%}"
        )
    if short_n := summary["short_answer_count"]:
        parts.append(
            f"Short (n={short_n}): mean baseline={summary['short_baseline_mean_accuracy']:.3f}, "
            f"mean RAG={summary['short_rag_mean_accuracy']:.3f}"
        )
    if parts:
        print("  Summary: " + " | ".join(parts))
    print(f"  {summary['summary_note']}")

    return overall

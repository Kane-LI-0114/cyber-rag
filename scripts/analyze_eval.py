#!/usr/bin/env python3
"""
Evaluation Results Analysis Script for CyberRAG

Analyzes the evaluation CSV output to compare baseline vs RAG performance,
identifies improvement areas, and generates detailed reports.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd


def load_evaluation_data(csv_path: str) -> pd.DataFrame:
    """Load evaluation results from CSV file."""
    df = pd.read_csv(csv_path)
    return df


def calculate_accuracy_metrics(df: pd.DataFrame) -> dict:
    """Calculate accuracy metrics for baseline and RAG."""
    total = len(df)
    baseline_correct = df["baseline_correct"].sum()
    rag_correct = df["rag_correct"].sum()

    baseline_acc = baseline_correct / total if total > 0 else 0
    rag_acc = rag_correct / total if total > 0 else 0

    # Separate by question type
    mcq_df = df[df["question_type"] == "multiple_choice"]
    short_df = df[df["question_type"] == "short_answer"]

    return {
        "total_questions": total,
        "baseline_correct": int(baseline_correct),
        "rag_correct": int(rag_correct),
        "baseline_accuracy": baseline_acc,
        "rag_accuracy": rag_acc,
        "improvement": rag_acc - baseline_acc,
        "by_type": {
            "multiple_choice": {
                "total": len(mcq_df),
                "baseline_correct": int(mcq_df["baseline_correct"].sum()),
                "rag_correct": int(mcq_df["rag_correct"].sum()),
                "baseline_accuracy": (
                    mcq_df["baseline_correct"].sum() / len(mcq_df)
                    if len(mcq_df) > 0
                    else 0
                ),
                "rag_accuracy": (
                    mcq_df["rag_correct"].sum() / len(mcq_df)
                    if len(mcq_df) > 0
                    else 0
                ),
            },
            "short_answer": {
                "total": len(short_df),
                "baseline_correct": int(short_df["baseline_correct"].sum()),
                "rag_correct": int(short_df["rag_correct"].sum()),
                "baseline_accuracy": (
                    short_df["baseline_correct"].sum() / len(short_df)
                    if len(short_df) > 0
                    else 0
                ),
                "rag_accuracy": (
                    short_df["rag_correct"].sum() / len(short_df)
                    if len(short_df) > 0
                    else 0
                ),
            },
        },
    }


def categorize_results(df: pd.DataFrame) -> dict:
    """Categorize questions by baseline vs RAG performance."""
    both_correct = df[
        (df["baseline_correct"] == True) & (df["rag_correct"] == True)
    ]
    both_wrong = df[
        (df["baseline_correct"] == False) & (df["rag_correct"] == False)
    ]
    rag_improved = df[
        (df["baseline_correct"] == False) & (df["rag_correct"] == True)
    ]
    rag_regressed = df[
        (df["baseline_correct"] == True) & (df["rag_correct"] == False)
    ]

    return {
        "both_correct": {
            "count": len(both_correct),
            "percentage": len(both_correct) / len(df) * 100,
            "indices": both_correct.index.tolist(),
        },
        "both_wrong": {
            "count": len(both_wrong),
            "percentage": len(both_wrong) / len(df) * 100,
            "indices": both_wrong.index.tolist(),
        },
        "rag_improved": {
            "count": len(rag_improved),
            "percentage": len(rag_improved) / len(df) * 100,
            "indices": rag_improved.index.tolist(),
        },
        "rag_regressed": {
            "count": len(rag_regressed),
            "percentage": len(rag_regressed) / len(df) * 100,
            "indices": rag_regressed.index.tolist(),
        },
    }


def get_error_cases(df: pd.DataFrame, case_type: str) -> pd.DataFrame:
    """Get specific error case types for detailed analysis."""
    if case_type == "rag_improved":
        return df[
            (df["baseline_correct"] == False) & (df["rag_correct"] == True)
        ]
    elif case_type == "rag_regressed":
        return df[
            (df["baseline_correct"] == True) & (df["rag_correct"] == False)
        ]
    elif case_type == "both_wrong":
        return df[
            (df["baseline_correct"] == False) & (df["rag_correct"] == False)
        ]
    elif case_type == "both_correct":
        return df[
            (df["baseline_correct"] == True) & (df["rag_correct"] == True)
        ]
    return pd.DataFrame()


def format_question_details(row: pd.Series) -> str:
    """Format a single question for display."""
    q_type = row["question_type"]
    lines = []
    lines.append(f"Question: {row['question']}")
    lines.append(f"Type: {q_type}")

    if q_type == "multiple_choice":
        try:
            choices = eval(row["choices"])
            for key, value in choices.items():
                marker = "✓" if key == row["reference_answer"] else " "
                lines.append(f"  [{marker}] {key}: {value}")
        except Exception:
            pass

    lines.append(f"Reference Answer: {row['reference_answer_text']}")
    lines.append(f"Baseline Answer: {row['baseline_answer']} {'✓' if row['baseline_correct'] else '✗'}")
    lines.append(f"RAG Answer: {row['rag_answer']} {'✓' if row['rag_correct'] else '✗'}")
    lines.append(f"Retrieved Chunks: {row['retrieved_chunks']}")

    return "\n".join(lines)


def generate_text_report(
    df: pd.DataFrame,
    metrics: dict,
    categories: dict,
    output_path: Optional[str] = None,
) -> str:
    """Generate a detailed text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("CyberRAG Evaluation Analysis Report")
    lines.append("=" * 80)
    lines.append("")

    # Overall Metrics
    lines.append("## Overall Performance Metrics")
    lines.append("-" * 40)
    lines.append(f"Total Questions: {metrics['total_questions']}")
    lines.append(f"Baseline Accuracy: {metrics['baseline_accuracy']:.2%}")
    lines.append(f"RAG Accuracy: {metrics['rag_accuracy']:.2%}")
    lines.append(f"Improvement: {metrics['improvement']:+.2%}")
    lines.append("")

    # By Question Type
    lines.append("## Performance by Question Type")
    lines.append("-" * 40)
    for qtype, data in metrics["by_type"].items():
        if data["total"] > 0:
            lines.append(f"\n### {qtype.replace('_', ' ').title()}")
            lines.append(f"  Total: {data['total']}")
            lines.append(
                f"  Baseline: {data['baseline_correct']}/{data['total']} ({data['baseline_accuracy']:.2%})"
            )
            lines.append(
                f"  RAG: {data['rag_correct']}/{data['total']} ({data['rag_accuracy']:.2%})"
            )

    lines.append("")

    # Result Categorization
    lines.append("## Result Categorization")
    lines.append("-" * 40)
    lines.append(
        f"Both Correct (Baseline + RAG): {categories['both_correct']['count']} ({categories['both_correct']['percentage']:.1f}%)"
    )
    lines.append(
        f"RAG Improved (Baseline failed, RAG correct): {categories['rag_improved']['count']} ({categories['rag_improved']['percentage']:.1f}%)"
    )
    lines.append(
        f"RAG Regressed (Baseline correct, RAG failed): {categories['rag_regressed']['count']} ({categories['rag_regressed']['percentage']:.1f}%)"
    )
    lines.append(
        f"Both Wrong (Neither succeeded): {categories['both_wrong']['count']} ({categories['both_wrong']['percentage']:.1f}%)"
    )
    lines.append("")

    # RAG Improved Cases
    if categories["rag_improved"]["count"] > 0:
        lines.append("## RAG Improved Cases (Sample)")
        lines.append("-" * 40)
        improved_df = get_error_cases(df, "rag_improved")
        for i, (_, row) in enumerate(improved_df.head(5).iterrows()):
            lines.append(f"\n### Case {i + 1}")
            lines.append(format_question_details(row))

    # RAG Regressed Cases
    if categories["rag_regressed"]["count"] > 0:
        lines.append("\n## RAG Regressed Cases (Sample)")
        lines.append("-" * 40)
        regressed_df = get_error_cases(df, "rag_regressed")
        for i, (_, row) in enumerate(regressed_df.head(5).iterrows()):
            lines.append(f"\n### Case {i + 1}")
            lines.append(format_question_details(row))

    # Both Wrong Cases
    if categories["both_wrong"]["count"] > 0:
        lines.append("\n## Both Wrong Cases (Need Attention)")
        lines.append("-" * 40)
        lines.append(
            f"Total: {categories['both_wrong']['count']} questions failed by both methods."
        )
        lines.append("Possible reasons:")
        lines.append("  - Missing knowledge in the document corpus")
        lines.append("  - Questions require reasoning beyond retrieved facts")
        lines.append("  - Retrieval quality needs improvement")
        both_wrong_df = get_error_cases(df, "both_wrong")
        for i, (_, row) in enumerate(both_wrong_df.head(3).iterrows()):
            lines.append(f"\n### Case {i + 1}")
            lines.append(format_question_details(row))

    lines.append("")
    lines.append("=" * 80)
    lines.append("End of Report")
    lines.append("=" * 80)

    report = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(report)
        print(f"Report saved to: {output_path}")

    return report


def generate_json_summary(
    metrics: dict, categories: dict, output_path: Optional[str] = None
) -> dict:
    """Generate a JSON summary for further processing."""
    summary = {
        "metrics": metrics,
        "categories": categories,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"JSON summary saved to: {output_path}")

    return summary


def export_error_cases(
    df: pd.DataFrame,
    case_type: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Export specific error case types to CSV."""
    error_df = get_error_cases(df, case_type)

    if len(error_df) == 0:
        print(f"No cases found for type: {case_type}")
        return error_df

    if output_path:
        error_df.to_csv(output_path, index=False)
        print(f"Exported {len(error_df)} cases to: {output_path}")

    return error_df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CyberRAG evaluation results"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="artifacts/evals/latest.csv",
        help="Path to evaluation CSV file (default: artifacts/evals/latest.csv)",
    )
    parser.add_argument(
        "--report",
        "-r",
        metavar="PATH",
        help="Output path for text report",
    )
    parser.add_argument(
        "--json",
        "-j",
        metavar="PATH",
        help="Output path for JSON summary",
    )
    parser.add_argument(
        "--export",
        "-e",
        choices=["rag_improved", "rag_regressed", "both_wrong", "both_correct"],
        metavar="TYPE",
        help="Export specific case type to CSV",
    )
    parser.add_argument(
        "--export-path",
        metavar="PATH",
        help="Output path for exported cases",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed analysis to console",
    )

    args = parser.parse_args()

    # Load data
    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        return 1

    print(f"Loading evaluation data from: {csv_path}")
    df = load_evaluation_data(str(csv_path))

    # Calculate metrics
    metrics = calculate_accuracy_metrics(df)
    categories = categorize_results(df)

    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"Total Questions: {metrics['total_questions']}")
    print(f"Baseline Accuracy: {metrics['baseline_accuracy']:.2%}")
    print(f"RAG Accuracy: {metrics['rag_accuracy']:.2%}")
    print(f"Improvement: {metrics['improvement']:+.2%}")
    print(f"\nRAG Improved: {categories['rag_improved']['count']}")
    print(f"RAG Regressed: {categories['rag_regressed']['count']}")
    print(f"Both Wrong: {categories['both_wrong']['count']}")

    # Generate outputs
    if args.report or args.verbose:
        report = generate_text_report(df, metrics, categories, args.report)
        if args.verbose:
            print("\n" + report)

    if args.json:
        generate_json_summary(metrics, categories, args.json)

    if args.export:
        export_path = args.export_path or f"{args.export}_cases.csv"
        export_error_cases(df, args.export, export_path)

    return 0


if __name__ == "__main__":
    exit(main())

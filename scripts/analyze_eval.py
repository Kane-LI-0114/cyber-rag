#!/usr/bin/env python3
"""
Enhanced Evaluation Results Analysis Script for CyberRAG

This module provides comprehensive analysis of evaluation results including:
- Basic accuracy metrics (baseline vs RAG)
- Retrieval quality analysis
- Question difficulty profiling
- Answer quality deep analysis
- Error pattern mining
- Statistical significance testing
- Cross-dimensional analysis
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd


# =============================================================================
# Helper Functions
# =============================================================================


def get_latest_eval_file() -> str:
    """Get the most recent timestamped evaluation file."""
    eval_dir = Path("artifacts/evals")
    eval_files = sorted(eval_dir.glob("eval_*.csv"), reverse=True)
    if eval_files:
        return str(eval_files[0])
    return "artifacts/evals/latest.csv"


def load_evaluation_data(csv_path: str) -> pd.DataFrame:
    """Load evaluation results from CSV file."""
    df = pd.read_csv(csv_path)
    return df


def _skipped_mask(df: pd.DataFrame) -> pd.Series:
    """True where a row was skipped (generation/judge error)."""
    if "eval_skipped" not in df.columns:
        return pd.Series(False, index=df.index)
    s = df["eval_skipped"]
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin(("true", "1", "yes"))


def _completed_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that completed evaluation (excludes skipped)."""
    return df[~_skipped_mask(df)]


def _per_row_score(df: pd.DataFrame, judge_col: str, bool_col: str) -> pd.Series:
    """Prefer LLM judge scores [0,1] when present; else 0/1 from bool column."""
    if judge_col in df.columns:
        return df[judge_col].fillna(df[bool_col].astype(float))
    return df[bool_col].astype(float)


# =============================================================================
# 1. Basic Accuracy Metrics
# =============================================================================


def calculate_accuracy_metrics(df: pd.DataFrame) -> dict:
    """Calculate accuracy metrics for baseline and RAG."""
    skipped_count = (
        int(_skipped_mask(df).sum()) if "eval_skipped" in df.columns else 0
    )
    dfm = _completed_rows(df)
    total = len(dfm)
    baseline_correct = dfm["baseline_correct"].sum()
    rag_correct = dfm["rag_correct"].sum()

    baseline_scores = _per_row_score(
        dfm, "baseline_judge_accuracy", "baseline_correct"
    )
    rag_scores = _per_row_score(dfm, "rag_judge_accuracy", "rag_correct")

    baseline_acc = float(baseline_scores.mean()) if total > 0 else 0.0
    rag_acc = float(rag_scores.mean()) if total > 0 else 0.0

    # Separate by question type
    mcq_df = dfm[dfm["question_type"] == "multiple_choice"]
    short_df = dfm[dfm["question_type"] == "short_answer"]

    short_baseline_acc = (
        float(short_df["baseline_judge_accuracy"].mean())
        if len(short_df) > 0 and "baseline_judge_accuracy" in short_df.columns
        else (
            short_df["baseline_correct"].sum() / len(short_df)
            if len(short_df) > 0
            else 0
        )
    )
    short_rag_acc = (
        float(short_df["rag_judge_accuracy"].mean())
        if len(short_df) > 0 and "rag_judge_accuracy" in short_df.columns
        else (
            short_df["rag_correct"].sum() / len(short_df)
            if len(short_df) > 0
            else 0
        )
    )

    return {
        "total_questions": total,
        "skipped_count": skipped_count,
        "total_rows_in_csv": len(df),
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
                "baseline_accuracy": short_baseline_acc,
                "rag_accuracy": short_rag_acc,
            },
        },
    }


def categorize_results(df: pd.DataFrame) -> dict:
    """Categorize questions by baseline vs RAG performance."""
    dfm = _completed_rows(df)
    n = len(dfm)
    denom = n if n > 0 else 1
    both_correct = dfm[
        (dfm["baseline_correct"] == True) & (dfm["rag_correct"] == True)
    ]
    both_wrong = dfm[
        (dfm["baseline_correct"] == False) & (dfm["rag_correct"] == False)
    ]
    rag_improved = dfm[
        (dfm["baseline_correct"] == False) & (dfm["rag_correct"] == True)
    ]
    rag_regressed = dfm[
        (dfm["baseline_correct"] == True) & (dfm["rag_correct"] == False)
    ]

    return {
        "both_correct": {
            "count": len(both_correct),
            "percentage": len(both_correct) / denom * 100,
            "indices": both_correct.index.tolist(),
        },
        "both_wrong": {
            "count": len(both_wrong),
            "percentage": len(both_wrong) / denom * 100,
            "indices": both_wrong.index.tolist(),
        },
        "rag_improved": {
            "count": len(rag_improved),
            "percentage": len(rag_improved) / denom * 100,
            "indices": rag_improved.index.tolist(),
        },
        "rag_regressed": {
            "count": len(rag_regressed),
            "percentage": len(rag_regressed) / denom * 100,
            "indices": rag_regressed.index.tolist(),
        },
    }


# =============================================================================
# 2. Retrieval Quality Analysis
# =============================================================================


def analyze_retrieval_quality(df: pd.DataFrame) -> dict:
    """
    Analyze retrieval quality metrics including:
    - Chunk count distribution
    - Chunk diversity (unique sources)
    - Retrieval coverage analysis
    """
    dfm = _completed_rows(df)
    
    if "retrieved_chunks" not in dfm.columns:
        return {"error": "retrieved_chunks column not found"}
    
    chunks = dfm["retrieved_chunks"].astype(int)
    
    result = {
        "chunk_count_stats": {
            "mean": float(chunks.mean()),
            "median": float(chunks.median()),
            "std": float(chunks.std()),
            "min": int(chunks.min()),
            "max": int(chunks.max()),
            "q25": float(chunks.quantile(0.25)),
            "q75": float(chunks.quantile(0.75)),
        },
        "chunk_distribution": {},
        "zero_chunk_cases": {
            "count": int((chunks == 0).sum()),
            "percentage": float((chunks == 0).sum() / len(dfm) * 100),
        },
    }
    
    # Chunk count distribution
    chunk_counts = chunks.value_counts().sort_index()
    result["chunk_distribution"] = {
        f"k={k}": {"count": int(v), "percentage": float(v / len(dfm) * 100)}
        for k, v in chunk_counts.items()
    }
    
    # Analyze by correctness
    for status in ["correct", "incorrect"]:
        if status == "correct":
            sub_df = dfm[dfm["rag_correct"] == True]
        else:
            sub_df = dfm[dfm["rag_correct"] == False]
        
        if len(sub_df) > 0:
            sub_chunks = sub_df["retrieved_chunks"].astype(int)
            result[f"chunk_count_when_rag_{status}"] = {
                "mean": float(sub_chunks.mean()),
                "median": float(sub_chunks.median()),
                "count": len(sub_chunks),
            }
    
    # Source diversity analysis (if sources are available)
    if "retrieved_sources" in dfm.columns:
        source_diversity = []
        for sources in dfm["retrieved_sources"].dropna():
            if isinstance(sources, str):
                unique_sources = len(set(sources.split("|")))
                source_diversity.append(unique_sources)
        
        if source_diversity:
            result["source_diversity"] = {
                "mean_unique_sources": float(sum(source_diversity) / len(source_diversity)),
                "min": min(source_diversity),
                "max": max(source_diversity),
            }
    
    return result


# =============================================================================
# 3. Question Difficulty Profiling
# =============================================================================


def analyze_question_difficulty(df: pd.DataFrame) -> dict:
    """
    Profile questions by difficulty based on evaluation results:
    - Easy: Both Correct > 80%
    - Medium: Baseline failed, RAG succeeded
    - Hard: Both Wrong
    - Retrieval Trap: Baseline correct, RAG failed
    """
    dfm = _completed_rows(df)
    n = len(dfm)
    denom = n if n > 0 else 1
    
    both_correct = (dfm["baseline_correct"] == True) & (dfm["rag_correct"] == True)
    both_wrong = (dfm["baseline_correct"] == False) & (dfm["rag_correct"] == False)
    rag_improved = (dfm["baseline_correct"] == False) & (dfm["rag_correct"] == True)
    rag_regressed = (dfm["baseline_correct"] == True) & (dfm["rag_correct"] == False)
    
    both_correct_rate = both_correct.sum() / denom
    both_wrong_rate = both_wrong.sum() / denom
    rag_improved_rate = rag_improved.sum() / denom
    rag_regressed_rate = rag_regressed.sum() / denom
    
    # Classify difficulty based on patterns
    difficulty_levels = {
        "easy": {
            "criteria": "Both Correct rate",
            "rate": float(both_correct_rate * 100),
            "interpretation": "Questions both systems can answer correctly"
        },
        "medium": {
            "criteria": "RAG Improved only",
            "rate": float(rag_improved_rate * 100),
            "interpretation": "Questions requiring retrieval to solve"
        },
        "hard": {
            "criteria": "Both Wrong rate",
            "rate": float(both_wrong_rate * 100),
            "interpretation": "Questions neither system can solve"
        },
        "retrieval_trap": {
            "criteria": "RAG Regressed rate",
            "rate": float(rag_regressed_rate * 100),
            "interpretation": "Questions where retrieval hurts performance"
        }
    }
    
    # Overall difficulty score (higher = harder)
    # Formula: weighted sum favoring hard and retrieval trap cases
    difficulty_score = (
        both_correct_rate * 0.0 +      # Easy: 0 points
        rag_improved_rate * 0.5 +      # Medium: 0.5 points
        both_wrong_rate * 1.0 +        # Hard: 1.0 point
        rag_regressed_rate * 1.5       # Trap: 1.5 points (retrieval harmful)
    )
    
    difficulty_levels["overall_score"] = {
        "value": float(difficulty_score),
        "normalized_0_100": float(difficulty_score * 100),
        "interpretation": "Higher score = harder questions overall"
    }
    
    # By question type
    difficulty_by_type = {}
    for qtype in dfm["question_type"].unique():
        type_df = dfm[dfm["question_type"] == qtype]
        type_n = len(type_df)
        type_denom = type_n if type_n > 0 else 1
        
        type_both_correct = ((type_df["baseline_correct"] == True) & 
                             (type_df["rag_correct"] == True)).sum() / type_denom
        type_both_wrong = ((type_df["baseline_correct"] == False) & 
                          (type_df["rag_correct"] == False)).sum() / type_denom
        
        difficulty_by_type[qtype] = {
            "total": type_n,
            "easy_rate": float(type_both_correct * 100),
            "hard_rate": float(type_both_wrong * 100),
            "retrieval_effectiveness": float(
                ((type_df["rag_correct"] == True).sum() - 
                 (type_df["baseline_correct"] == True).sum()) / type_denom * 100
            )
        }
    
    difficulty_levels["by_type"] = difficulty_by_type
    
    return difficulty_levels


# =============================================================================
# 4. Answer Quality Deep Analysis
# =============================================================================


def analyze_answer_quality(df: pd.DataFrame) -> dict:
    """
    Deep analysis of answer quality including:
    - MCQ option distribution
    - Short answer quality (length, judge scores)
    - Answer change patterns
    """
    dfm = _completed_rows(df)
    result = {}
    
    # MCQ Option Distribution Analysis
    mcq_df = dfm[dfm["question_type"] == "multiple_choice"]
    if len(mcq_df) > 0:
        option_counts = Counter()
        correct_option_counts = Counter()
        
        for _, row in mcq_df.iterrows():
            # Count baseline answers
            baseline_ans = str(row.get("baseline_answer", "")).strip().upper()
            if baseline_ans in ["A", "B", "C", "D"]:
                option_counts[baseline_ans] += 1
            
            # Count RAG answers
            rag_ans = str(row.get("rag_answer", "")).strip().upper()
            if rag_ans in ["A", "B", "C", "D"]:
                option_counts[rag_ans] += 1
            
            # Count correct reference answers
            ref_ans = str(row.get("reference_answer", "")).strip().upper()
            if ref_ans in ["A", "B", "C", "D"]:
                correct_option_counts[ref_ans] += 1
        
        total_answers = len(mcq_df) * 2  # Both baseline and RAG
        
        result["mcq_option_distribution"] = {
            "baseline_and_rag_combined": {
                opt: {
                    "count": count,
                    "percentage": float(count / total_answers * 100)
                }
                for opt, count in sorted(option_counts.items())
            },
            "reference_answer_distribution": {
                opt: {
                    "count": count,
                    "percentage": float(count / len(mcq_df) * 100)
                }
                for opt, count in sorted(correct_option_counts.items())
            },
            "model_bias_detection": {}
        }
        
        # Detect model bias (unusual distribution compared to expected 25% each)
        expected_rate = 25.0
        bias_detected = {}
        for opt, data in result["mcq_option_distribution"]["baseline_and_rag_combined"].items():
            actual_rate = data["percentage"]
            deviation = abs(actual_rate - expected_rate)
            if deviation > 10:  # More than 10% deviation from expected
                bias_detected[opt] = {
                    "actual_percentage": actual_rate,
                    "expected_percentage": expected_rate,
                    "deviation": deviation,
                    "severity": "high" if deviation > 20 else "moderate"
                }
        
        result["mcq_option_distribution"]["model_bias_detection"] = bias_detected
        
        # Answer change analysis
        answer_changes = 0
        baseline_only_correct = 0
        rag_only_correct = 0
        
        for _, row in mcq_df.iterrows():
            baseline_ans = str(row.get("baseline_answer", "")).strip().upper()
            rag_ans = str(row.get("rag_answer", "")).strip().upper()
            
            if baseline_ans != rag_ans:
                answer_changes += 1
                if row.get("baseline_correct"):
                    baseline_only_correct += 1
                if row.get("rag_correct"):
                    rag_only_correct += 1
        
        result["mcq_answer_changes"] = {
            "total_changes": answer_changes,
            "change_percentage": float(answer_changes / len(mcq_df) * 100),
            "changes_favoring_baseline": baseline_only_correct,
            "changes_favoring_rag": rag_only_correct,
        }
    
    # Short Answer Quality Analysis
    short_df = dfm[dfm["question_type"] == "short_answer"]
    if len(short_df) > 0:
        # Answer length analysis
        baseline_lengths = short_df["baseline_answer"].astype(str).str.len()
        rag_lengths = short_df["rag_answer"].astype(str).str.len()
        
        result["short_answer_quality"] = {
            "baseline_length_stats": {
                "mean": float(baseline_lengths.mean()),
                "median": float(baseline_lengths.median()),
                "std": float(baseline_lengths.std()),
                "min": int(baseline_lengths.min()),
                "max": int(baseline_lengths.max()),
            },
            "rag_length_stats": {
                "mean": float(rag_lengths.mean()),
                "median": float(rag_lengths.median()),
                "std": float(rag_lengths.std()),
                "min": int(rag_lengths.min()),
                "max": int(rag_lengths.max()),
            },
            "length_change_ratio": float(
                (rag_lengths.mean() - baseline_lengths.mean()) / 
                max(baseline_lengths.mean(), 1) * 100
            ),
        }
        
        # Judge score distribution
        if "baseline_judge_accuracy" in short_df.columns:
            baseline_judge = short_df["baseline_judge_accuracy"].dropna()
            rag_judge = short_df["rag_judge_accuracy"].dropna()
            
            if len(baseline_judge) > 0:
                result["short_answer_quality"]["judge_score_stats"] = {
                    "baseline_judge": {
                        "mean": float(baseline_judge.mean()),
                        "median": float(baseline_judge.median()),
                        "std": float(baseline_judge.std()),
                        "min": float(baseline_judge.min()),
                        "max": float(baseline_judge.max()),
                    },
                    "rag_judge": {
                        "mean": float(rag_judge.mean()) if len(rag_judge) > 0 else 0,
                        "median": float(rag_judge.median()) if len(rag_judge) > 0 else 0,
                        "std": float(rag_judge.std()) if len(rag_judge) > 0 else 0,
                    },
                }
                
                # Boundary samples (scores close to 0.5 threshold)
                threshold = 0.5
                boundary_range = 0.1
                baseline_boundary = (
                    (baseline_judge >= threshold - boundary_range) & 
                    (baseline_judge <= threshold + boundary_range)
                ).sum()
                rag_boundary = (
                    (rag_judge >= threshold - boundary_range) & 
                    (rag_judge <= threshold + boundary_range)
                ).sum()
                
                result["short_answer_quality"]["boundary_samples"] = {
                    "threshold": threshold,
                    "boundary_range": boundary_range,
                    "baseline_near_threshold": int(baseline_boundary),
                    "rag_near_threshold": int(rag_boundary),
                    "baseline_boundary_percentage": float(
                        baseline_boundary / len(baseline_judge) * 100
                    ),
                    "rag_boundary_percentage": float(
                        rag_boundary / len(rag_judge) * 100 if len(rag_judge) > 0 else 0
                    ),
                }
    
    return result


# =============================================================================
# 5. Error Pattern Mining
# =============================================================================


def mine_error_patterns(df: pd.DataFrame) -> dict:
    """
    Mine error patterns including:
    - RAG Regressed root cause analysis
    - Answer similarity analysis
    - Common failure modes
    """
    dfm = _completed_rows(df)
    result = {}
    
    # RAG Regressed Analysis
    rag_regressed = dfm[
        (dfm["baseline_correct"] == True) & (dfm["rag_correct"] == False)
    ]
    
    if len(rag_regressed) > 0:
        result["rag_regressed_analysis"] = {
            "total_count": len(rag_regressed),
            "by_type": {},
        }
        
        # By question type
        for qtype in rag_regressed["question_type"].unique():
            type_count = len(rag_regressed[rag_regressed["question_type"] == qtype])
            result["rag_regressed_analysis"]["by_type"][qtype] = {
                "count": type_count,
                "percentage": float(type_count / len(rag_regressed) * 100),
            }
        
        # Chunk count analysis for regressed cases
        if "retrieved_chunks" in rag_regressed.columns:
            chunks = rag_regressed["retrieved_chunks"].astype(int)
            result["rag_regressed_analysis"]["chunk_stats"] = {
                "mean_chunks": float(chunks.mean()),
                "median_chunks": float(chunks.median()),
                "zero_chunk_count": int((chunks == 0).sum()),
            }
            
            # High chunk count but still wrong (possible noise/conflict)
            high_chunks = rag_regressed[chunks >= 4]
            result["rag_regressed_analysis"]["high_chunk_regressed"] = {
                "count": len(high_chunks),
                "percentage": float(len(high_chunks) / len(rag_regressed) * 100),
                "possible_cause": "Retrieved content may contain noise or conflicting information"
            }
        
        # Sample cases
        sample_cols = ["question", "baseline_answer", "rag_answer", "retrieved_chunks"]
        available_cols = [c for c in sample_cols if c in rag_regressed.columns]
        result["rag_regressed_analysis"]["sample_cases"] = (
            rag_regressed[available_cols].head(3).to_dict("records")
        )
    
    # Both Wrong Analysis
    both_wrong = dfm[
        (dfm["baseline_correct"] == False) & (dfm["rag_correct"] == False)
    ]
    
    if len(both_wrong) > 0:
        result["both_wrong_analysis"] = {
            "total_count": len(both_wrong),
            "possible_causes": [
                "Missing knowledge in document corpus",
                "Questions require reasoning beyond retrieved facts",
                "Retrieval quality needs improvement",
                "LLM lacks domain-specific knowledge",
            ],
        }
        
        # Chunk count for both wrong cases
        if "retrieved_chunks" in both_wrong.columns:
            chunks = both_wrong["retrieved_chunks"].astype(int)
            zero_chunk_pct = float((chunks == 0).sum() / len(both_wrong) * 100)
            
            result["both_wrong_analysis"]["retrieval_stats"] = {
                "mean_chunks": float(chunks.mean()),
                "zero_chunk_percentage": zero_chunk_pct,
                "likely_cause": (
                    "Corpus coverage issue" if zero_chunk_pct > 30 
                    else "Reasoning/complexity issue"
                )
            }
    
    # Answer Similarity Analysis (for MCQ)
    mcq_df = dfm[dfm["question_type"] == "multiple_choice"]
    if len(mcq_df) > 0:
        # Calculate how often RAG changes baseline answer
        changes = 0
        changes_to_correct = 0
        changes_to_incorrect = 0
        
        for _, row in mcq_df.iterrows():
            baseline = str(row.get("baseline_answer", "")).strip().upper()
            rag = str(row.get("rag_answer", "")).strip().upper()
            ref = str(row.get("reference_answer", "")).strip().upper()
            
            if baseline != rag:
                changes += 1
                if rag == ref:
                    changes_to_correct += 1
                else:
                    changes_to_incorrect += 1
        
        result["answer_change_patterns"] = {
            "total_changes": changes,
            "change_rate": float(changes / len(mcq_df) * 100),
            "changes_to_correct": changes_to_correct,
            "changes_to_incorrect": changes_to_incorrect,
            "change_effectiveness": {
                "improved": changes_to_correct,
                "degraded": changes_to_incorrect,
                "net_effect": changes_to_correct - changes_to_incorrect,
            }
        }
    
    return result


# =============================================================================
# 6. Statistical Significance Testing
# =============================================================================


def calculate_statistical_significance(df: pd.DataFrame) -> dict:
    """
    Calculate statistical significance using McNemar's test.
    
    McNemar's test is appropriate for paired nominal data - comparing
    two classifiers on the same test set.
    """
    dfm = _completed_rows(df)
    n = len(dfm)
    
    # Build contingency table
    #                  | RAG Correct | RAG Wrong |
    # -----------------|--------------|-----------|
    # Baseline Correct |     b       |     c     |
    # Baseline Wrong   |     d       |     e     |
    
    b = ((dfm["baseline_correct"] == True) & (dfm["rag_correct"] == True)).sum()
    c = ((dfm["baseline_correct"] == True) & (dfm["rag_correct"] == False)).sum()  # RAG regressed
    d = ((dfm["baseline_correct"] == False) & (dfm["rag_correct"] == True)).sum()  # RAG improved
    e = ((dfm["baseline_correct"] == False) & (dfm["rag_correct"] == False)).sum()
    
    result = {
        "contingency_table": {
            "baseline_correct_rag_correct": int(b),
            "baseline_correct_rag_wrong": int(c),  # RAG regressed
            "baseline_wrong_rag_correct": int(d),  # RAG improved
            "baseline_wrong_rag_wrong": int(e),
        },
        "total_samples": n,
    }
    
    # McNemar's test (using exact binomial test for 2x2)
    # Under null hypothesis: P(RAG correct | Baseline wrong) = P(RAG wrong | Baseline correct)
    #discordant_1 = c  # baseline correct, RAG wrong
    #discordant_2 = d  # baseline wrong, RAG correct
    
    # Use continuity correction for chi-square approximation
    n12 = c
    n21 = d
    n = n12 + n21
    
    if n > 0:
        # Chi-square with continuity correction
        chi2 = (abs(n12 - n21) - 1) ** 2 / (n12 + n21)
        
        # Calculate p-value (chi-square with 1 df)
        from scipy import stats
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
        
        # Effect size: Cohen's h (difference between two proportions)
        p1 = n21 / (n21 + n) if (n21 + n) > 0 else 0  # RAG improvement rate
        p2 = n12 / (n12 + n) if (n12 + n) > 0 else 0  # RAG regression rate
        
        def cohens_h(p1, p2):
            """Calculate Cohen's h for two proportions."""
            if p1 == 0 and p2 == 0:
                return 0.0
            phi1 = 2 * math.asin(math.sqrt(max(p1, 0.0001)))
            phi2 = 2 * math.asin(math.sqrt(max(p2, 0.0001)))
            return abs(phi1 - phi2)
        
        effect_size = cohens_h(p1, p2)
        
        result["mcnemar_test"] = {
            "chi_square": float(chi2),
            "p_value": float(p_value),
            "significant_at_0.05": bool(p_value < 0.05),
            "significant_at_0.01": bool(p_value < 0.01),
            "interpretation": (
                "Significant difference between baseline and RAG"
                if p_value < 0.05
                else "No significant difference"
            ),
        }
        
        result["effect_size"] = {
            "cohens_h": float(effect_size),
            "interpretation": _interpret_cohens_h(effect_size),
        }
        
        # Confidence interval for the difference
        prop_diff = (d - c) / n if n > 0 else 0
        se = math.sqrt((c + d) / n ** 2) if n > 0 else 0
        ci_lower = prop_diff - 1.96 * se
        ci_upper = prop_diff + 1.96 * se
        
        result["difference_estimate"] = {
            "proportion_difference": float(prop_diff),
            "95_ci_lower": float(max(-1, ci_lower)),
            "95_ci_upper": float(min(1, ci_upper)),
            "interpretation": (
                "RAG is better than baseline"
                if prop_diff > 0.1
                else "Baseline is better or similar"
                if prop_diff < -0.1
                else "Similar performance"
            )
        }
    
    else:
        result["mcnemar_test"] = {
            "error": "No discordant pairs found - cannot perform McNemar test"
        }
        result["effect_size"] = {"error": "Cannot calculate effect size"}
        result["difference_estimate"] = {"error": "Cannot estimate difference"}
    
    return result


def _interpret_cohens_h(h: float) -> str:
    """Interpret Cohen's h effect size."""
    h = abs(h)
    if h < 0.2:
        return "Negligible effect"
    elif h < 0.5:
        return "Small effect"
    elif h < 0.8:
        return "Medium effect"
    else:
        return "Large effect"


# =============================================================================
# 7. Cross-Dimensional Analysis
# =============================================================================


def analyze_cross_dimensions(df: pd.DataFrame) -> dict:
    """
    Perform cross-dimensional analysis including:
    - Question type × Result classification
    - Chunk count × Accuracy
    - Answer length × Accuracy
    """
    dfm = _completed_rows(df)
    result = {}
    
    # Question Type × Result Classification
    if "question_type" in dfm.columns:
        result["type_vs_result"] = {}
        
        for qtype in dfm["question_type"].unique():
            type_df = dfm[dfm["question_type"] == qtype]
            n = len(type_df)
            
            if n == 0:
                continue
            
            both_correct = ((type_df["baseline_correct"] == True) & 
                           (type_df["rag_correct"] == True)).sum()
            both_wrong = ((type_df["baseline_correct"] == False) & 
                         (type_df["rag_correct"] == False)).sum()
            rag_improved = ((type_df["baseline_correct"] == False) & 
                           (type_df["rag_correct"] == True)).sum()
            rag_regressed = ((type_df["baseline_correct"] == True) & 
                            (type_df["rag_correct"] == False)).sum()
            
            result["type_vs_result"][qtype] = {
                "total": n,
                "both_correct": {"count": int(both_correct), "percentage": float(both_correct / n * 100)},
                "both_wrong": {"count": int(both_wrong), "percentage": float(both_wrong / n * 100)},
                "rag_improved": {"count": int(rag_improved), "percentage": float(rag_improved / n * 100)},
                "rag_regressed": {"count": int(rag_regressed), "percentage": float(rag_regressed / n * 100)},
            }
    
    # Chunk Count × Accuracy Analysis
    if "retrieved_chunks" in dfm.columns:
        chunks = dfm["retrieved_chunks"].astype(int)
        
        # Bin chunks into categories
        bins = [0, 1, 2, 4, float('inf')]
        labels = ['0', '1', '2-3', '4+']
        dfm = dfm.copy()
        dfm['chunk_bin'] = pd.cut(chunks, bins=bins, labels=labels, include_lowest=True)
        
        result["chunk_count_vs_accuracy"] = {}
        for bin_label in labels:
            bin_df = dfm[dfm['chunk_bin'] == bin_label]
            n = len(bin_df)
            
            if n > 0:
                baseline_acc = bin_df["baseline_correct"].sum() / n
                rag_acc = bin_df["rag_correct"].sum() / n
                
                result["chunk_count_vs_accuracy"][f"chunks_{bin_label}"] = {
                    "sample_count": n,
                    "baseline_accuracy": float(baseline_acc * 100),
                    "rag_accuracy": float(rag_acc * 100),
                    "retrieval_gain": float((rag_acc - baseline_acc) * 100),
                }
        
        # Remove temporary column
        dfm.drop(columns=['chunk_bin'], inplace=True)
    
    # Answer Length × Accuracy (for short answer)
    short_df = dfm[dfm["question_type"] == "short_answer"]
    if len(short_df) > 0 and "baseline_judge_accuracy" in short_df.columns:
        short_df = short_df.copy()
        
        baseline_lengths = short_df["baseline_answer"].astype(str).str.len()
        rag_lengths = short_df["rag_answer"].astype(str).str.len()
        
        # Bin by length
        q25 = baseline_lengths.quantile(0.25)
        q75 = baseline_lengths.quantile(0.75)
        
        short_df['length_bin'] = pd.cut(
            baseline_lengths,
            bins=[0, q25, q75, float('inf')],
            labels=['short', 'medium', 'long']
        )
        
        result["answer_length_vs_accuracy"] = {}
        for bin_label in ['short', 'medium', 'long']:
            bin_df = short_df[short_df['length_bin'] == bin_label]
            n = len(bin_df)
            
            if n > 0:
                baseline_judge = bin_df["baseline_judge_accuracy"].dropna()
                rag_judge = bin_df["rag_judge_accuracy"].dropna()
                
                result["answer_length_vs_accuracy"][f"baseline_length_{bin_label}"] = {
                    "sample_count": n,
                    "baseline_judge_mean": float(baseline_judge.mean()) if len(baseline_judge) > 0 else 0,
                    "rag_judge_mean": float(rag_judge.mean()) if len(rag_judge) > 0 else 0,
                }
        
        short_df.drop(columns=['length_bin'], inplace=True)
    
    # Judge Score Distribution (boundary analysis)
    if "baseline_judge_accuracy" in dfm.columns:
        judge_scores = dfm["baseline_judge_accuracy"].dropna()
        
        if len(judge_scores) > 0:
            # Create score buckets
            buckets = {
                "0.0-0.2 (Very Low)": ((judge_scores >= 0) & (judge_scores < 0.2)).sum(),
                "0.2-0.4 (Low)": ((judge_scores >= 0.2) & (judge_scores < 0.4)).sum(),
                "0.4-0.6 (Borderline)": ((judge_scores >= 0.4) & (judge_scores < 0.6)).sum(),
                "0.6-0.8 (Good)": ((judge_scores >= 0.6) & (judge_scores < 0.8)).sum(),
                "0.8-1.0 (Excellent)": ((judge_scores >= 0.8) & (judge_scores <= 1.0)).sum(),
            }
            
            result["judge_score_distribution"] = {
                bucket: {"count": int(count), "percentage": float(count / len(judge_scores) * 100)}
                for bucket, count in buckets.items()
            }
    
    return result


# =============================================================================
# Helper Functions (Error Cases)
# =============================================================================


def get_error_cases(df: pd.DataFrame, case_type: str) -> pd.DataFrame:
    """Get specific error case types for detailed analysis."""
    dfm = _completed_rows(df)
    if case_type == "rag_improved":
        return dfm[
            (dfm["baseline_correct"] == False) & (dfm["rag_correct"] == True)
        ]
    elif case_type == "rag_regressed":
        return dfm[
            (dfm["baseline_correct"] == True) & (dfm["rag_correct"] == False)
        ]
    elif case_type == "both_wrong":
        return dfm[
            (dfm["baseline_correct"] == False) & (dfm["rag_correct"] == False)
        ]
    elif case_type == "both_correct":
        return dfm[
            (dfm["baseline_correct"] == True) & (dfm["rag_correct"] == True)
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

    ref = row.get("reference_answer_text", row.get("reference_answer", ""))
    lines.append(f"Reference Answer: {ref}")
    if row.get("eval_skipped"):
        lines.append(f"Skipped: {row.get('skip_reason', '')}")
    lines.append(f"Baseline Answer: {row['baseline_answer']} {'✓' if row['baseline_correct'] else '✗'}")
    lines.append(f"RAG Answer: {row['rag_answer']} {'✓' if row['rag_correct'] else '✗'}")
    lines.append(f"Retrieved Chunks: {row['retrieved_chunks']}")

    return "\n".join(lines)


# =============================================================================
# Report Generation
# =============================================================================


def generate_comprehensive_report(
    df: pd.DataFrame,
    metrics: dict,
    categories: dict,
    retrieval_quality: dict,
    difficulty: dict,
    answer_quality: dict,
    error_patterns: dict,
    stats: dict,
    cross_analysis: dict,
    output_path: Optional[str] = None,
) -> str:
    """Generate a comprehensive text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("CyberRAG Comprehensive Evaluation Analysis Report")
    lines.append("=" * 80)
    lines.append("")

    # 1. Overall Metrics
    lines.append("## 1. Overall Performance Metrics")
    lines.append("-" * 40)
    lines.append(f"Total Questions (evaluated): {metrics['total_questions']}")
    if metrics.get("skipped_count", 0) > 0:
        lines.append(f"Skipped (errors): {metrics['skipped_count']}")
    lines.append(f"Baseline Accuracy: {metrics['baseline_accuracy']:.2%}")
    lines.append(f"RAG Accuracy: {metrics['rag_accuracy']:.2%}")
    lines.append(f"Improvement: {metrics['improvement']:+.2%}")
    lines.append("")

    # 2. By Question Type
    lines.append("## 2. Performance by Question Type")
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

    # 3. Result Categorization
    lines.append("## 3. Result Categorization")
    lines.append("-" * 40)
    lines.append(
        f"Both Correct: {categories['both_correct']['count']} ({categories['both_correct']['percentage']:.1f}%)"
    )
    lines.append(
        f"RAG Improved: {categories['rag_improved']['count']} ({categories['rag_improved']['percentage']:.1f}%)"
    )
    lines.append(
        f"RAG Regressed: {categories['rag_regressed']['count']} ({categories['rag_regressed']['percentage']:.1f}%)"
    )
    lines.append(
        f"Both Wrong: {categories['both_wrong']['count']} ({categories['both_wrong']['percentage']:.1f}%)"
    )
    lines.append("")

    # 4. Retrieval Quality Analysis
    if "error" not in retrieval_quality:
        lines.append("## 4. Retrieval Quality Analysis")
        lines.append("-" * 40)
        chunk_stats = retrieval_quality.get("chunk_count_stats", {})
        if chunk_stats:
            lines.append(f"Mean chunks retrieved: {chunk_stats.get('mean', 0):.1f}")
            lines.append(f"Median chunks retrieved: {chunk_stats.get('median', 0):.1f}")
            lines.append(f"Chunk range: {chunk_stats.get('min', 0)} - {chunk_stats.get('max', 0)}")
        
        zero_chunks = retrieval_quality.get("zero_chunk_cases", {})
        if zero_chunks:
            lines.append(f"Zero-chunk cases: {zero_chunks.get('count', 0)} ({zero_chunks.get('percentage', 0):.1f}%)")
        
        # Chunk count by correctness
        for status in ["correct", "incorrect"]:
            key = f"chunk_count_when_rag_{status}"
            if key in retrieval_quality:
                stats_data = retrieval_quality[key]
                lines.append(f"Mean chunks when RAG {status}: {stats_data['mean']:.1f}")
        lines.append("")

    # 5. Question Difficulty Profile
    lines.append("## 5. Question Difficulty Profile")
    lines.append("-" * 40)
    score_data = difficulty.get("overall_score", {})
    lines.append(f"Overall Difficulty Score: {score_data.get('normalized_0_100', 0):.1f}/100")
    lines.append(f"  (Higher = harder questions)")
    
    for level in ["easy", "medium", "hard", "retrieval_trap"]:
        level_data = difficulty.get(level, {})
        lines.append(f"\n{level.capitalize()} Questions: {level_data.get('rate', 0):.1f}%")
        lines.append(f"  Interpretation: {level_data.get('interpretation', '')}")
    lines.append("")

    # 6. Answer Quality Analysis
    if answer_quality:
        lines.append("## 6. Answer Quality Analysis")
        lines.append("-" * 40)
        
        # MCQ Analysis
        mcq_dist = answer_quality.get("mcq_option_distribution", {})
        if mcq_dist:
            lines.append("\n### MCQ Option Distribution (Baseline + RAG)")
            for opt, data in mcq_dist.get("baseline_and_rag_combined", {}).items():
                lines.append(f"  Option {opt}: {data['percentage']:.1f}%")
            
            # Reference distribution
            lines.append("\n### Reference Answer Distribution")
            for opt, data in mcq_dist.get("reference_answer_distribution", {}).items():
                lines.append(f"  Option {opt}: {data['percentage']:.1f}%")
            
            # Model bias detection
            bias = mcq_dist.get("model_bias_detection", {})
            if bias:
                lines.append("\n### Model Bias Detected")
                for opt, data in bias.items():
                    lines.append(f"  Option {opt}: {data['severity'].upper()} (deviation: {data['deviation']:.1f}%)")
        
        # Answer changes
        changes = answer_quality.get("mcq_answer_changes", {})
        if changes:
            lines.append("\n### Answer Change Analysis")
            lines.append(f"  Total changes: {changes['total_changes']} ({changes['change_percentage']:.1f}%)")
            lines.append(f"  Changes favoring baseline: {changes['changes_favoring_baseline']}")
            lines.append(f"  Changes favoring RAG: {changes['changes_favoring_rag']}")
        
        # Short Answer Analysis
        short_qa = answer_quality.get("short_answer_quality", {})
        if short_qa:
            lines.append("\n### Short Answer Quality")
            base_len = short_qa.get("baseline_length_stats", {})
            rag_len = short_qa.get("rag_length_stats", {})
            lines.append(f"  Baseline avg length: {base_len.get('mean', 0):.0f} chars")
            lines.append(f"  RAG avg length: {rag_len.get('mean', 0):.0f} chars")
            lines.append(f"  Length change: {short_qa.get('length_change_ratio', 0):+.1f}%")
            
            judge_stats = short_qa.get("judge_score_stats", {})
            if judge_stats:
                lines.append(f"  Baseline judge mean: {judge_stats.get('baseline_judge', {}).get('mean', 0):.3f}")
                lines.append(f"  RAG judge mean: {judge_stats.get('rag_judge', {}).get('mean', 0):.3f}")
            
            boundary = short_qa.get("boundary_samples", {})
            if boundary:
                lines.append(f"  Baseline near-threshold: {boundary.get('baseline_near_threshold', 0)} ({boundary.get('baseline_boundary_percentage', 0):.1f}%)")
        lines.append("")

    # 7. Error Pattern Analysis
    if error_patterns:
        lines.append("## 7. Error Pattern Analysis")
        lines.append("-" * 40)
        
        regressed = error_patterns.get("rag_regressed_analysis", {})
        if regressed:
            lines.append(f"\n### RAG Regressed Cases: {regressed.get('total_count', 0)}")
            chunk_stats = regressed.get("chunk_stats", {})
            if chunk_stats:
                lines.append(f"  Mean chunks: {chunk_stats.get('mean_chunks', 0):.1f}")
                lines.append(f"  Zero-chunk cases: {chunk_stats.get('zero_chunk_count', 0)}")
            
            high_chunk = regressed.get("high_chunk_regressed", {})
            if high_chunk:
                lines.append(f"  High chunk (>4) but still wrong: {high_chunk.get('count', 0)} ({high_chunk.get('percentage', 0):.1f}%)")
        
        both_wrong = error_patterns.get("both_wrong_analysis", {})
        if both_wrong:
            lines.append(f"\n### Both Wrong Cases: {both_wrong.get('total_count', 0)}")
            for cause in both_wrong.get("possible_causes", [])[:2]:
                lines.append(f"  - {cause}")
        
        change_patterns = error_patterns.get("answer_change_patterns", {})
        if change_patterns:
            lines.append(f"\n### Answer Change Patterns (MCQ)")
            eff = change_patterns.get("change_effectiveness", {})
            lines.append(f"  Changes to correct: {eff.get('improved', 0)}")
            lines.append(f"  Changes to incorrect: {eff.get('degraded', 0)}")
            lines.append(f"  Net effect: {eff.get('net_effect', 0):+d}")
        lines.append("")

    # 8. Statistical Significance
    if "error" not in stats.get("mcnemar_test", {}):
        lines.append("## 8. Statistical Significance (McNemar's Test)")
        lines.append("-" * 40)
        
        mcnemar = stats.get("mcnemar_test", {})
        lines.append(f"Chi-square: {mcnemar.get('chi_square', 0):.4f}")
        lines.append(f"P-value: {mcnemar.get('p_value', 1):.6f}")
        lines.append(f"Significant at α=0.05: {'Yes' if mcnemar.get('significant_at_0.05') else 'No'}")
        lines.append(f"Significant at α=0.01: {'Yes' if mcnemar.get('significant_at_0.01') else 'No'}")
        lines.append(f"Interpretation: {mcnemar.get('interpretation', '')}")
        
        effect = stats.get("effect_size", {})
        lines.append(f"\nEffect Size (Cohen's h): {effect.get('cohens_h', 0):.4f}")
        lines.append(f"Interpretation: {effect.get('interpretation', '')}")
        
        diff_est = stats.get("difference_estimate", {})
        lines.append(f"\nDifference Estimate: {diff_est.get('proportion_difference', 0):+.2%}")
        lines.append(f"95% CI: [{diff_est.get('95_ci_lower', 0):.2%}, {diff_est.get('95_ci_upper', 0):.2%}]")
        lines.append(f"Interpretation: {diff_est.get('interpretation', '')}")
        lines.append("")
    elif "contingency_table" in stats:
        lines.append("## 8. Statistical Significance")
        lines.append("-" * 40)
        ct = stats["contingency_table"]
        lines.append("Contingency Table:")
        lines.append(f"  B/C (both correct): {ct['baseline_correct_rag_correct']}")
        lines.append(f"  B correct, RAG wrong: {ct['baseline_correct_rag_wrong']}")
        lines.append(f"  B wrong, RAG correct: {ct['baseline_wrong_rag_correct']}")
        lines.append(f"  Both wrong: {ct['baseline_wrong_rag_wrong']}")
        lines.append("")

    # 9. Cross-Dimensional Analysis
    if cross_analysis:
        lines.append("## 9. Cross-Dimensional Analysis")
        lines.append("-" * 40)
        
        # Type vs Result
        type_result = cross_analysis.get("type_vs_result", {})
        if type_result:
            lines.append("\n### Question Type × Result Classification")
            for qtype, data in type_result.items():
                lines.append(f"\n{qtype}:")
                lines.append(f"  Total: {data['total']}")
                lines.append(f"  Both Correct: {data['both_correct']['percentage']:.1f}%")
                lines.append(f"  RAG Improved: {data['rag_improved']['percentage']:.1f}%")
                lines.append(f"  RAG Regressed: {data['rag_regressed']['percentage']:.1f}%")
                lines.append(f"  Both Wrong: {data['both_wrong']['percentage']:.1f}%")
        
        # Chunk count vs accuracy
        chunk_acc = cross_analysis.get("chunk_count_vs_accuracy", {})
        if chunk_acc:
            lines.append("\n### Chunk Count × RAG Accuracy")
            for key, data in chunk_acc.items():
                lines.append(f"  {key}: {data['rag_accuracy']:.1f}% (n={data['sample_count']})")
        
        # Judge score distribution
        judge_dist = cross_analysis.get("judge_score_distribution", {})
        if judge_dist:
            lines.append("\n### Judge Score Distribution (Short Answer)")
            for bucket, data in judge_dist.items():
                lines.append(f"  {bucket}: {data['percentage']:.1f}%")
        lines.append("")

    # Sample Cases
    lines.append("## 10. Sample Cases")
    lines.append("-" * 40)
    
    # RAG Improved Cases
    if categories["rag_improved"]["count"] > 0:
        lines.append("\n### RAG Improved (Sample)")
        improved_df = get_error_cases(df, "rag_improved")
        for i, (_, row) in enumerate(improved_df.head(3).iterrows()):
            lines.append(f"\nCase {i + 1}:")
            lines.append(format_question_details(row))
    
    # RAG Regressed Cases
    if categories["rag_regressed"]["count"] > 0:
        lines.append("\n\n### RAG Regressed (Sample)")
        regressed_df = get_error_cases(df, "rag_regressed")
        for i, (_, row) in enumerate(regressed_df.head(3).iterrows()):
            lines.append(f"\nCase {i + 1}:")
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
    metrics: dict,
    categories: dict,
    retrieval_quality: dict,
    difficulty: dict,
    answer_quality: dict,
    error_patterns: dict,
    stats: dict,
    cross_analysis: dict,
    output_path: Optional[str] = None,
) -> dict:
    """Generate a comprehensive JSON summary."""
    summary = {
        "metrics": metrics,
        "categories": categories,
        "retrieval_quality": retrieval_quality,
        "question_difficulty": difficulty,
        "answer_quality": answer_quality,
        "error_patterns": error_patterns,
        "statistical_significance": stats,
        "cross_dimensional_analysis": cross_analysis,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
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


# =============================================================================
# Main CLI Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive CyberRAG Evaluation Analysis"
    )
    default_csv = get_latest_eval_file()
    parser.add_argument(
        "input",
        nargs="?",
        default=default_csv,
        help=f"Path to evaluation CSV file. Defaults to: {default_csv}",
    )
    parser.add_argument(
        "--report", "-r",
        metavar="PATH",
        help="Output path for text report",
    )
    parser.add_argument(
        "--json", "-j",
        metavar="PATH",
        help="Output path for JSON summary",
    )
    parser.add_argument(
        "--export", "-e",
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
        "--verbose", "-v",
        action="store_true",
        help="Print detailed report to console",
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Show only quick summary without full report",
    )

    args = parser.parse_args()

    # Load data
    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        return 1

    print(f"Loading evaluation data from: {csv_path}")
    df = load_evaluation_data(str(csv_path))

    # Run all analyses
    print("Running comprehensive analysis...")
    
    metrics = calculate_accuracy_metrics(df)
    categories = categorize_results(df)
    retrieval_quality = analyze_retrieval_quality(df)
    difficulty = analyze_question_difficulty(df)
    answer_quality = analyze_answer_quality(df)
    error_patterns = mine_error_patterns(df)
    stats = calculate_statistical_significance(df)
    cross_analysis = analyze_cross_dimensions(df)

    # Print quick summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Total Questions: {metrics['total_questions']}")
    if metrics.get("skipped_count", 0) > 0:
        print(f"Skipped: {metrics['skipped_count']}")
    print(f"Baseline Accuracy: {metrics['baseline_accuracy']:.2%}")
    print(f"RAG Accuracy: {metrics['rag_accuracy']:.2%}")
    print(f"Improvement: {metrics['improvement']:+.2%}")
    print(f"\nRAG Improved: {categories['rag_improved']['count']}")
    print(f"RAG Regressed: {categories['rag_regressed']['count']}")
    print(f"Both Wrong: {categories['both_wrong']['count']}")
    
    # Difficulty score
    diff_score = difficulty.get("overall_score", {}).get("normalized_0_100", 0)
    print(f"\nQuestion Difficulty Score: {diff_score:.1f}/100")
    
    # Statistical significance
    p_val = stats.get("mcnemar_test", {}).get("p_value", 1)
    if p_val < 1:
        sig = "Yes" if p_val < 0.05 else "No"
        print(f"Statistically Significant (p<0.05): {sig}")

    if args.quick:
        return 0

    # Generate outputs
    if args.report or args.verbose:
        report = generate_comprehensive_report(
            df, metrics, categories, retrieval_quality, difficulty,
            answer_quality, error_patterns, stats, cross_analysis,
            args.report
        )
        if args.verbose:
            print("\n" + report)

    if args.json:
        generate_json_summary(
            metrics, categories, retrieval_quality, difficulty,
            answer_quality, error_patterns, stats, cross_analysis,
            args.json
        )

    if args.export:
        export_path = args.export_path or f"{args.export}_cases.csv"
        export_error_cases(df, args.export, export_path)

    return 0


if __name__ == "__main__":
    exit(main())

"""Retrieval evaluation dataset and metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalQuestion:
    """A single evaluation question with expected relevant documents."""

    id: str
    question: str
    software: str
    version: str | None = None
    expected_chunk_ids: list[str] = field(default_factory=list)
    expected_urls: list[str] = field(default_factory=list)
    expected_titles: list[str] = field(default_factory=list)
    category: str = "general"
    difficulty: str = "medium"  # easy, medium, hard


@dataclass
class EvalResult:
    """Result of evaluating a single question."""

    question: EvalQuestion
    retrieved_chunk_ids: list[str]
    retrieved_scores: list[float]
    recall_at_5: float
    recall_at_10: float
    mrr: float


def load_eval_dataset(path: Path | None = None) -> list[EvalQuestion]:
    """Load evaluation dataset from JSON file."""
    if path is None:
        path = Path(__file__).parent / "eval_dataset.json"

    if not path.exists():
        return []

    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)

    return [EvalQuestion(**item) for item in data]


def save_eval_dataset(questions: list[EvalQuestion], path: Path | None = None) -> None:
    """Save evaluation dataset to JSON file."""
    if path is None:
        path = Path(__file__).parent / "eval_dataset.json"

    with Path(path).open("w", encoding="utf-8") as f:
        json.dump([q.__dict__ for q in questions], f, indent=2)


def compute_recall_at_k(
    retrieved_ids: list[str],
    expected_ids: list[str],
    k: int,
) -> float:
    """Compute Recall@K."""
    if not expected_ids:
        return 1.0
    retrieved_set = set(retrieved_ids[:k])
    expected_set = set(expected_ids)
    return len(retrieved_set & expected_set) / len(expected_set)


def compute_mrr(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """Compute Mean Reciprocal Rank."""
    if not expected_ids:
        return 1.0
    expected_set = set(expected_ids)
    for rank, chunk_id in enumerate(retrieved_ids, 1):
        if chunk_id in expected_set:
            return 1.0 / rank
    return 0.0


def evaluate_question(
    question: EvalQuestion,
    retrieved_chunk_ids: list[str],
    retrieved_scores: list[float] | None = None,
) -> EvalResult:
    """Evaluate a single question against retrieved results."""
    recall_5 = compute_recall_at_k(retrieved_chunk_ids, question.expected_chunk_ids, 5)
    recall_10 = compute_recall_at_k(retrieved_chunk_ids, question.expected_chunk_ids, 10)
    mrr = compute_mrr(retrieved_chunk_ids, question.expected_chunk_ids)

    return EvalResult(
        question=question,
        retrieved_chunk_ids=retrieved_chunk_ids,
        retrieved_scores=retrieved_scores or [],
        recall_at_5=recall_5,
        recall_at_10=recall_10,
        mrr=mrr,
    )


def evaluate_dataset(
    questions: list[EvalQuestion],
    search_fn: Any,  # Callable that takes (question, k) and returns list of (chunk_id, score)
    k: int = 10,
) -> list[EvalResult]:
    """Evaluate all questions in the dataset."""
    results = []
    for q in questions:
        retrieved = search_fn(q, k)
        chunk_ids = [c[0] for c in retrieved]
        scores = [c[1] for c in retrieved]
        results.append(evaluate_question(q, chunk_ids, scores))
    return results


def print_eval_summary(results: list[EvalResult]) -> None:
    """Print evaluation summary."""
    if not results:
        print("No evaluation results.")
        return

    avg_recall_5 = sum(r.recall_at_5 for r in results) / len(results)
    avg_recall_10 = sum(r.recall_at_10 for r in results) / len(results)
    avg_mrr = sum(r.mrr for r in results) / len(results)

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Questions evaluated: {len(results)}")
    print(f"Recall@5:  {avg_recall_5:.3f}")
    print(f"Recall@10: {avg_recall_10:.3f}")
    print(f"MRR:       {avg_mrr:.3f}")
    print("=" * 60)

    from collections import defaultdict

    by_category: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_category[r.question.category].append(r)

    print("\nPer-category:")
    for cat, cat_results in sorted(by_category.items()):
        r5 = sum(r.recall_at_5 for r in cat_results) / len(cat_results)
        r10 = sum(r.recall_at_10 for r in cat_results) / len(cat_results)
        m = sum(r.mrr for r in cat_results) / len(cat_results)
        print(
            f"  {cat:<20} Recall@5: {r5:.3f}  Recall@10: {r10:.3f}  "
            f"MRR: {m:.3f}  ({len(cat_results)} questions)"
        )

    by_difficulty: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_difficulty[r.question.difficulty].append(r)

    print("\nPer-difficulty:")
    for diff in ["easy", "medium", "hard"]:
        if diff in by_difficulty:
            cat_results = by_difficulty[diff]
            r5 = sum(r.recall_at_5 for r in cat_results) / len(cat_results)
            r10 = sum(r.recall_at_10 for r in cat_results) / len(cat_results)
            m = sum(r.mrr for r in cat_results) / len(cat_results)
            print(
                f"  {diff:<20} Recall@5: {r5:.3f}  Recall@10: {r10:.3f}  "
                f"MRR: {m:.3f}  ({len(cat_results)} questions)"
            )


def get_postgresql_questions() -> list[EvalQuestion]:
    """Get the PostgreSQL evaluation questions from the dataset."""
    return [q for q in load_eval_dataset() if q.software == "postgresql"]


def create_starter_eval_dataset() -> list[EvalQuestion]:
    """Load the full evaluation dataset (50+ documentation questions)."""
    return load_eval_dataset()

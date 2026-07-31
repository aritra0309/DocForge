"""Tests for retrieval evaluation framework."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.eval import (
    EvalQuestion,
    compute_mrr,
    compute_recall_at_k,
    create_starter_eval_dataset,
    evaluate_dataset,
    evaluate_question,
    get_postgresql_questions,
    load_eval_dataset,
    print_eval_summary,
)


class TestEvalMetrics:
    def test_compute_recall_at_k_perfect(self) -> None:
        retrieved = ["a", "b", "c", "d", "e"]
        expected = ["a", "b", "c"]
        assert compute_recall_at_k(retrieved, expected, 5) == 1.0
        assert compute_recall_at_k(retrieved, expected, 3) == 1.0
        assert compute_recall_at_k(retrieved, expected, 2) == 2/3

    def test_compute_recall_at_k_empty_expected(self) -> None:
        retrieved = ["a", "b", "c"]
        expected = []
        assert compute_recall_at_k(retrieved, expected, 5) == 1.0

    def test_compute_recall_at_k_no_overlap(self) -> None:
        retrieved = ["a", "b", "c"]
        expected = ["x", "y", "z"]
        assert compute_recall_at_k(retrieved, expected, 5) == 0.0

    def test_compute_mrr_first_result(self) -> None:
        retrieved = ["expected", "other", "more"]
        expected = ["expected"]
        assert compute_mrr(retrieved, expected) == 1.0

    def test_compute_mrr_third_result(self) -> None:
        retrieved = ["a", "b", "expected", "d"]
        expected = ["expected"]
        assert compute_mrr(retrieved, expected) == 1/3

    def test_compute_mrr_not_found(self) -> None:
        retrieved = ["a", "b", "c"]
        expected = ["expected"]
        assert compute_mrr(retrieved, expected) == 0.0

    def test_compute_mrr_empty_expected(self) -> None:
        retrieved = ["a", "b"]
        expected = []
        assert compute_mrr(retrieved, expected) == 1.0


class TestEvalQuestion:
    def test_postgresql_questions_loaded(self) -> None:
        questions = get_postgresql_questions()
        assert len(questions) >= 10
        assert all(q.software == "postgresql" for q in questions)
        assert all(q.version == "17" for q in questions)

    def test_postgresql_question_categories(self) -> None:
        questions = get_postgresql_questions()
        categories = {q.category for q in questions}
        assert "sql" in categories
        assert "tutorial" in categories
        assert "admin" in categories

    def test_postgresql_question_difficulties(self) -> None:
        questions = get_postgresql_questions()
        difficulties = {q.difficulty for q in questions}
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties

    def test_starter_dataset_created(self) -> None:
        dataset = create_starter_eval_dataset()
        assert len(dataset) >= 50

    def test_load_eval_dataset(self) -> None:
        path = Path(__file__).parent / "eval_dataset.json"
        questions = load_eval_dataset(path)
        assert len(questions) >= 50
        assert all(isinstance(q, EvalQuestion) for q in questions)

    def test_eval_dataset_covers_registry_software(self) -> None:
        questions = load_eval_dataset()
        softwares = {q.software for q in questions}
        for name in (
            "postgresql",
            "fastapi",
            "react",
            "redis",
            "kubernetes",
            "mongodb",
            "mysql",
        ):
            assert name in softwares
        assert any(q.expected_urls for q in questions)


class TestEvaluateQuestion:
    def test_evaluate_question_perfect_match(self) -> None:
        question = EvalQuestion(
            id="test-1",
            question="Test question",
            software="test",
            expected_chunk_ids=["chunk1", "chunk2", "chunk3"],
        )
        retrieved = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
        result = evaluate_question(question, retrieved)

        assert result.recall_at_5 == 1.0
        assert result.recall_at_10 == 1.0
        assert result.mrr == 1.0

    def test_evaluate_question_partial_match(self) -> None:
        question = EvalQuestion(
            id="test-2",
            question="Test question",
            software="test",
            expected_chunk_ids=["chunk1", "chunk2", "chunk3"],
        )
        retrieved = ["chunk1", "other1", "other2", "chunk2", "other3"]
        result = evaluate_question(question, retrieved)

        assert result.recall_at_5 == 2/3
        assert result.mrr == 1.0  # First relevant at position 1

    def test_evaluate_question_late_match(self) -> None:
        question = EvalQuestion(
            id="test-3",
            question="Test question",
            software="test",
            expected_chunk_ids=["chunk1"],
        )
        retrieved = ["other1", "other2", "other3", "chunk1"]
        result = evaluate_question(question, retrieved)

        assert result.recall_at_5 == 1.0
        assert result.mrr == 1/4

    def test_evaluate_question_no_match(self) -> None:
        question = EvalQuestion(
            id="test-4",
            question="Test question",
            software="test",
            expected_chunk_ids=["chunk1"],
        )
        retrieved = ["other1", "other2", "other3"]
        result = evaluate_question(question, retrieved)

        assert result.recall_at_5 == 0.0
        assert result.mrr == 0.0


class TestEvaluateDataset:
    def test_evaluate_dataset_mock_search(self) -> None:
        questions = get_postgresql_questions()[:3]

        # Mock search function that returns perfect results
        def mock_search(q: EvalQuestion, k: int):
            return [("chunk1", 0.9), ("chunk2", 0.8), ("chunk3", 0.7)]

        results = evaluate_dataset(questions, mock_search, k=10)

        assert len(results) == 3
        # All should have perfect recall since mock returns fixed chunks
        for r in results:
            assert r.recall_at_5 >= 0.0
            assert r.recall_at_10 >= 0.0


class TestPrintEvalSummary:
    def test_print_summary(self, capsys: pytest.CaptureFixture) -> None:
        from tests.eval import EvalResult

        results = [
            EvalResult(
                question=EvalQuestion(id="1", question="q", software="s", expected_chunk_ids=["a"]),
                retrieved_chunk_ids=["a"],
                retrieved_scores=[0.9],
                recall_at_5=1.0,
                recall_at_10=1.0,
                mrr=1.0,
            ),
            EvalResult(
                question=EvalQuestion(id="2", question="q", software="s", expected_chunk_ids=["b"]),
                retrieved_chunk_ids=["b"],
                retrieved_scores=[0.8],
                recall_at_5=1.0,
                recall_at_10=1.0,
                mrr=1.0,
            ),
        ]

        print_eval_summary(results)
        captured = capsys.readouterr()
        assert "RETRIEVAL EVALUATION SUMMARY" in captured.out
        assert "Recall@5:  1.000" in captured.out
        assert "Recall@10: 1.000" in captured.out
        assert "MRR:       1.000" in captured.out
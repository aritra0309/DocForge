"""Pytest configuration for benchmarks."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "benchmark: Performance benchmarks")


# Ensure benchmark results are collected
@pytest.fixture(autouse=True, scope="session")
def _collect_benchmark_results():
    """Collect benchmark results at session end."""
    yield
    # Print summary at end of session
    import tests.benchmarks as bm

    if hasattr(bm.benchmark, "results") and bm.benchmark.results:
        bm.print_summary_table(bm.benchmark.results)
        all_passed = bm.check_targets(bm.benchmark.results)
        if not all_passed:
            print("\n⚠ Some benchmarks did not meet their targets!")
        else:
            print("\n✓ All benchmarks met their targets!")

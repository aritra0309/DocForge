"""Benchmark configuration and utilities."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkResult:
    name: str
    duration_seconds: float
    items_processed: int
    items_per_second: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name}: {self.items_per_second:.2f} items/sec ({self.items_processed} items in {self.duration_seconds:.3f}s)"


@contextmanager
def benchmark(name: str, items_processed: int = 1):
    """Context manager to time a benchmark."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        result = BenchmarkResult(
            name=name,
            duration_seconds=duration,
            items_processed=items_processed,
            items_per_second=items_processed / duration if duration > 0 else 0,
        )
        print(result)
        # Store result for later collection
        if not hasattr(benchmark, "results"):
            benchmark.results = []
        benchmark.results.append(result)


def run_benchmarks() -> list[BenchmarkResult]:
    """Run all registered benchmarks and return results."""
    if hasattr(benchmark, "results"):
        results = benchmark.results
        benchmark.results = []
        return results
    return []


def print_summary_table(results: list[BenchmarkResult]) -> None:
    """Print a formatted summary table of benchmark results."""
    if not results:
        print("No benchmark results to display.")
        return

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Benchmark':<40} {'Items':>10} {'Time (s)':>12} {'Items/sec':>12}")
    print("-" * 80)
    for r in results:
        print(f"{r.name:<40} {r.items_processed:>10} {r.duration_seconds:>12.3f} {r.items_per_second:>12.2f}")
    print("=" * 80)


def check_targets(results: list[BenchmarkResult]) -> bool:
    """Check if all benchmarks meet their performance targets."""
    targets = {
        "extraction": 100,  # pages/sec
        "chunking": 1000,  # chunks/sec
        "embedding": 50,  # chunks/sec (local)
        "search_p50": 0.05,  # 50ms
        "search_p99": 0.100,  # 100ms
        "crawling_cached": 500,  # pages/sec
    }

    all_passed = True
    print("\nTARGET CHECK:")
    print("-" * 80)

    for r in results:
        target = None
        for key, value in targets.items():
            if key in r.name.lower():
                target = value
                break

        if target is not None:
            if "search" in r.name.lower():
                # For search, we check latency (lower is better)
                passed = r.duration_seconds <= target
                status = "PASS" if passed else "FAIL"
                print(f"  {r.name:<40} target: {target*1000:.0f}ms, actual: {r.duration_seconds*1000:.1f}ms [{status}]")
            else:
                # For throughput, higher is better
                passed = r.items_per_second >= target
                status = "PASS" if passed else "FAIL"
                print(f"  {r.name:<40} target: {target:.0f}/sec, actual: {r.items_per_second:.1f}/sec [{status}]")

            if not passed:
                all_passed = False

    return all_passed
.PHONY: help lint format typecheck test test-unit test-integration bench eval build docs-serve docs-build clean install dev-install

help:
	@echo "DocForge - Available targets:"
	@echo "  make lint        - Run ruff linting and formatting checks"
	@echo "  make format      - Auto-fix formatting with ruff"
	@echo "  make typecheck   - Run mypy static type checking"
	@echo "  make test        - Run unit + integration tests with coverage (>=80%)"
	@echo "  make test-unit   - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make bench       - Run performance benchmarks"
	@echo "  make eval        - Run retrieval evaluation suite"
	@echo "  make build       - Build wheel and sdist"
	@echo "  make docs-serve  - Serve documentation locally"
	@echo "  make docs-build  - Build documentation site"
	@echo "  make install     - Install package in editable mode"
	@echo "  make dev-install - Install with dev dependencies"
	@echo "  make clean       - Remove build artifacts"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff check --fix src tests
	ruff format src tests

typecheck:
	mypy src/docforge tests

test:
	pytest -v \
		--ignore=tests/benchmarks \
		--ignore=tests/eval \
		--ignore=tests/integration/test_crawl_real_docs.py \
		-m "not slow and not real_network and not benchmark" \
		--cov=docforge \
		--cov-report=term-missing:skip-covered \
		--cov-fail-under=80

test-unit:
	pytest -v tests/unit -m "not slow and not real_network"

test-integration:
	pytest -v tests/integration -m "integration and not slow and not real_network" \
		--ignore=tests/integration/test_crawl_real_docs.py

bench:
	pytest -v -m "benchmark" tests/benchmarks

eval:
	pytest -v tests/eval

build:
	python -m build

docs-serve:
	mkdocs serve

docs-build:
	mkdocs build --strict

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache coverage.xml htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

check: lint typecheck test

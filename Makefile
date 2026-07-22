.PHONY: help lint format typecheck test build clean install dev-install

help:
	@echo "DocForge - Available targets:"
	@echo "  make lint        - Run ruff linting and formatting checks"
	@echo "  make format      - Auto-fix formatting with ruff"
	@echo "  make typecheck   - Run mypy static type checking"
	@echo "  make test        - Run pytest test suite"
	@echo "  make build       - Build wheel and sdist"
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
	mypy src/docforge

test:
	pytest -v || [ $$? -eq 5 ]

test-unit:
	pytest -v -m "unit" --ignore=tests/integration --ignore=tests/benchmarks --ignore=tests/eval

test-integration:
	pytest -v -m "integration"

bench:
	pytest -v -m "benchmark" tests/benchmarks

build:
	python -m build

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache coverage.xml htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

check: lint typecheck test
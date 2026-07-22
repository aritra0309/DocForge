# Contributing to DocForge

Thank you for your interest in contributing to DocForge!

## Development Setup

```bash
# Clone the repo
git clone https://github.com/docforge/docforge
cd docforge

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install with dev dependencies
make dev-install

# Setup pre-commit hooks
pre-commit install
```

## Branch Naming

- `feat/short-description` — new features
- `fix/short-description` — bug fixes
- `docs/short-description` — documentation changes
- `refactor/short-description` — code refactoring
- `test/short-description` — test additions/fixes

## PR Checklist

- [ ] Code passes `make lint` (ruff check + format)
- [ ] Code passes `make typecheck` (mypy)
- [ ] Tests pass (`make test`)
- [ ] New code has corresponding tests
- [ ] Docstrings on all public functions/classes
- [ ] CHANGELOG.md updated (if applicable)

## Adding Software to the Registry

Create a YAML file in `registry/software/`:

```yaml
name: my-software
display_name: My Software
documentation:
  base_url: "https://docs.example.com/"
  version_pattern: "https://docs.example.com/{version}/"
  sitemap_url: "https://docs.example.com/sitemap.xml"
  versions:
    strategy: "url_enumeration"
    known_versions: ["3.0", "2.1", "2.0"]
    latest: "3.0"
  content_selectors:
    main_content: "#content"
    navigation: ".sidebar"
  url_filters:
    include: ["/docs/{version}/**"]
    exclude: []
```

Validate your entry:

```bash
python -c "from docforge.discovery.registry import RegistryLoader; r = RegistryLoader(); print(r.lookup('my-software'))"
```

## Running Tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests
make test-integration

# Benchmarks
make bench

# With coverage
pytest --cov=docforge --cov-report=html
```

## Code Style

- Python 3.11+ required
- Follow existing patterns in the codebase
- Use type hints on all functions
- Keep functions focused and small
- Write docstrings for public APIs
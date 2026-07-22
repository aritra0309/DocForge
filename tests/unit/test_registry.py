"""Unit tests for the software registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from docforge.discovery.registry import Registry, RegistryEntry, load_registry

# json.dumps alias for convenience
j = json.dumps


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry_dir(tmp_path: Path) -> Path:
    """Create a temporary registry directory with a few YAML entries."""
    rdir = tmp_path / "registry"
    sw_dir = rdir / "software"
    sw_dir.mkdir(parents=True)

    # Write schema
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["name", "display_name", "documentation"],
        "properties": {
            "name": {"type": "string"},
            "display_name": {"type": "string"},
            "documentation": {
                "type": "object",
                "required": ["base_url"],
                "properties": {
                    "base_url": {"type": "string", "format": "uri"},
                    "versions": {
                        "type": "object",
                        "properties": {
                            "strategy": {"type": "string"},
                            "known_versions": {"type": "array", "items": {"type": "string"}},
                            "latest": {"type": "string"},
                        },
                    },
                },
            },
        },
    }
    (rdir / "schema.json").write_text(j(schema))

    # PostgreSQL entry
    pg = {
        "name": "postgresql",
        "display_name": "PostgreSQL",
        "documentation": {
            "base_url": "https://www.postgresql.org/docs/",
            "versions": {
                "strategy": "url_enumeration",
                "known_versions": ["17", "16", "15"],
                "latest": "17",
            },
        },
    }
    (sw_dir / "postgresql.yaml").write_text(yaml.dump(pg))

    # FastAPI entry
    fa = {
        "name": "fastapi",
        "display_name": "FastAPI",
        "documentation": {
            "base_url": "https://fastapi.tiangolo.com/",
            "versions": {
                "strategy": "sitemap",
                "known_versions": ["latest"],
                "latest": "latest",
            },
        },
    }
    (sw_dir / "fastapi.yaml").write_text(yaml.dump(fa))

    return rdir


# ---------------------------------------------------------------------------
# RegistryEntry
# ---------------------------------------------------------------------------


class TestRegistryEntry:
    def test_basic_properties(self) -> None:
        entry = RegistryEntry(
            {
                "name": "test",
                "display_name": "Test Software",
                "documentation": {
                    "base_url": "https://example.com/docs/",
                    "versions": {
                        "strategy": "explicit",
                        "known_versions": ["2", "1"],
                        "latest": "2",
                    },
                    "content_selectors": {"main_content": "#content"},
                    "url_filters": {"include": ["/**"], "exclude": []},
                    "page_type_hints": {"tutorial_paths": ["/tutorial/**"]},
                },
            }
        )
        assert entry.name == "test"
        assert entry.display_name == "Test Software"
        assert entry.base_url == "https://example.com/docs/"
        assert entry.known_versions == ["2", "1"]
        assert entry.latest_version == "2"
        assert entry.version_strategy == "explicit"
        assert entry.content_selectors == {"main_content": "#content"}
        assert entry.url_filters == {"include": ["/**"], "exclude": []}
        assert entry.page_type_hints == {"tutorial_paths": ["/tutorial/**"]}

    def test_minimal_entry(self) -> None:
        entry = RegistryEntry(
            {
                "name": "minimal",
                "display_name": "Minimal",
                "documentation": {"base_url": "https://minimal.example.com/"},
            }
        )
        assert entry.base_url == "https://minimal.example.com/"
        assert entry.known_versions == []
        assert entry.latest_version is None
        assert entry.content_selectors == {}
        assert entry.url_filters == {}
        assert entry.page_type_hints == {}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_lookup_found(self) -> None:
        entries = [
            RegistryEntry(
                {
                    "name": "postgresql",
                    "display_name": "PostgreSQL",
                    "documentation": {"base_url": "https://www.postgresql.org/docs/"},
                }
            )
        ]
        reg = Registry(entries)
        result = reg.lookup("postgresql")
        assert result is not None
        assert result.name == "postgresql"

    def test_lookup_not_found(self) -> None:
        reg = Registry([])
        assert reg.lookup("nonexistent") is None

    def test_names(self) -> None:
        entries = [
            RegistryEntry(
                {"name": "b", "display_name": "B", "documentation": {"base_url": "https://b.com/"}}
            ),
            RegistryEntry(
                {"name": "a", "display_name": "A", "documentation": {"base_url": "https://a.com/"}}
            ),
        ]
        reg = Registry(entries)
        assert reg.names == ["a", "b"]

    def test_contains(self) -> None:
        reg = Registry(
            [
                RegistryEntry(
                    {
                        "name": "exists",
                        "display_name": "Exists",
                        "documentation": {"base_url": "https://exists.com/"},
                    }
                )
            ]
        )
        assert "exists" in reg
        assert "missing" not in reg

    def test_len(self) -> None:
        entries = [
            RegistryEntry(
                {"name": "a", "display_name": "A", "documentation": {"base_url": "https://a.com/"}}
            ),
            RegistryEntry(
                {"name": "b", "display_name": "B", "documentation": {"base_url": "https://b.com/"}}
            ),
        ]
        assert len(Registry(entries)) == 2

    def test_iteration(self) -> None:
        entries = [
            RegistryEntry(
                {"name": "a", "display_name": "A", "documentation": {"base_url": "https://a.com/"}}
            ),
            RegistryEntry(
                {"name": "b", "display_name": "B", "documentation": {"base_url": "https://b.com/"}}
            ),
        ]
        reg = Registry(entries)
        names = [e.name for e in reg]
        assert names == ["a", "b"]


# ---------------------------------------------------------------------------
# load_registry
# ---------------------------------------------------------------------------


class TestLoadRegistry:
    def test_loads_entries(self, registry_dir: Path) -> None:
        reg = load_registry(registry_dir)
        assert len(reg) == 2

    def test_lookup_postgresql(self, registry_dir: Path) -> None:
        reg = load_registry(registry_dir)
        entry = reg.lookup("postgresql")
        assert entry is not None
        assert entry.display_name == "PostgreSQL"
        assert entry.base_url == "https://www.postgresql.org/docs/"

    def test_lookup_fastapi(self, registry_dir: Path) -> None:
        reg = load_registry(registry_dir)
        entry = reg.lookup("fastapi")
        assert entry is not None
        assert entry.display_name == "FastAPI"

    def test_lookup_unknown_returns_none(self, registry_dir: Path) -> None:
        reg = load_registry(registry_dir)
        assert reg.lookup("unknown") is None

    def test_raises_on_missing_directory(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_registry("/nonexistent/path")

    def test_skips_empty_yaml(self, tmp_path: Path) -> None:
        rdir = tmp_path / "registry"
        sw_dir = rdir / "software"
        sw_dir.mkdir(parents=True)
        (rdir / "schema.json").write_text("{}")
        # Empty YAML file
        (sw_dir / "empty.yaml").write_text("")
        reg = load_registry(rdir)
        assert len(reg) == 0

    def test_schema_validation_rejects_bad_entry(self, tmp_path: Path) -> None:
        rdir = tmp_path / "registry"
        sw_dir = rdir / "software"
        sw_dir.mkdir(parents=True)
        schema = {
            "type": "object",
            "required": ["name", "display_name", "documentation"],
            "properties": {
                "name": {"type": "string"},
                "display_name": {"type": "string"},
                "documentation": {
                    "type": "object",
                    "required": ["base_url"],
                    "properties": {"base_url": {"type": "string"}},
                },
            },
        }
        (rdir / "schema.json").write_text(j(schema))
        bad = {"name": "bad", "display_name": "Bad", "documentation": "not-an-object"}
        (sw_dir / "bad.yaml").write_text(yaml.dump(bad))
        with pytest.raises(ValueError):
            load_registry(rdir)

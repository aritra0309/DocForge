"""Curated software registry — maps software names to documentation URLs."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import jsonschema
import yaml


class RegistryEntry:
    """A single entry in the software registry."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.name: str = data["name"]
        self.display_name: str = data["display_name"]
        self.documentation: dict[str, Any] = data["documentation"]

    @property
    def base_url(self) -> str:
        return cast(str, self.documentation["base_url"])

    @property
    def version_pattern(self) -> str | None:
        return cast("str | None", self.documentation.get("version_pattern"))

    @property
    def sitemap_url(self) -> str | None:
        return cast("str | None", self.documentation.get("sitemap_url"))

    @property
    def versions(self) -> dict[str, Any]:
        return cast("dict[str, Any]", self.documentation.get("versions", {}))

    @property
    def known_versions(self) -> list[str]:
        return cast("list[str]", self.versions.get("known_versions", []))

    @property
    def latest_version(self) -> str | None:
        return self.versions.get("latest")

    @property
    def version_strategy(self) -> str | None:
        return self.versions.get("strategy")

    @property
    def content_selectors(self) -> dict[str, str]:
        return cast("dict[str, str]", self.documentation.get("content_selectors", {}))

    @property
    def url_filters(self) -> dict[str, list[str]]:
        return cast("dict[str, list[str]]", self.documentation.get("url_filters", {}))

    @property
    def page_type_hints(self) -> dict[str, list[str]]:
        return cast("dict[str, list[str]]", self.documentation.get("page_type_hints", {}))


class Registry:
    """Indexed registry of all known software documentation sources."""

    def __init__(self, entries: list[RegistryEntry]) -> None:
        self._by_name: dict[str, RegistryEntry] = {e.name: e for e in entries}

    @property
    def names(self) -> list[str]:
        return sorted(self._by_name)

    def lookup(self, name: str) -> RegistryEntry | None:
        """Look up a software by canonical name.

        Returns:
            The matching ``RegistryEntry``, or ``None`` if not found.
        """
        return self._by_name.get(name)

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __iter__(self) -> Iterator[RegistryEntry]:
        return iter(self._by_name.values())  # type: ignore[return-value]


def _find_registry_dir() -> Path:
    """Locate the registry directory relative to the package root."""
    pkg_root = Path(__file__).resolve().parent.parent.parent
    candidate = pkg_root / "registry"
    if candidate.is_dir():
        return candidate
    return Path.cwd() / "registry"


def load_registry(registry_dir: str | Path | None = None) -> Registry:
    """Load all YAML registry entries from the given directory.

    Args:
        registry_dir: Path to the ``registry/`` directory. If ``None``,
            walks up from the package location to find it.

    Returns:
        A ``Registry`` instance containing all validated entries.

    Raises:
        FileNotFoundError: If the registry directory does not exist.
        ValueError: If any YAML file fails validation.
    """
    if registry_dir is None:
        registry_dir = _find_registry_dir()

    path = Path(registry_dir)
    software_dir = path / "software"

    if not software_dir.is_dir():
        msg = f"Registry directory not found: {software_dir}"
        raise FileNotFoundError(msg)

    schema_path = path / "schema.json"
    schema: dict[str, Any] | None = None
    if schema_path.is_file():
        with schema_path.open() as f:
            schema = json.load(f)

    entries: list[RegistryEntry] = []
    for yaml_file in sorted(software_dir.glob("*.yaml")):
        with yaml_file.open() as f:
            data = yaml.safe_load(f)
        if data is None:
            continue
        if schema:
            _validate_against_schema(data, schema, yaml_file.name)
        entries.append(RegistryEntry(data))

    return Registry(entries)


_ERR_SCHEMA_VALIDATION = "Registry entry '{}' failed schema validation: {}"
_ERR_NAME_EMPTY = "Registry entry '{}': 'name' must be a non-empty string"
_ERR_MISSING_BASE_URL = "Registry entry '{}': missing 'documentation.base_url'"


def _validate_against_schema(data: dict[str, Any], schema: dict[str, Any], filename: str) -> None:
    """Validate a single registry entry against the JSON schema."""
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        raise ValueError(_ERR_SCHEMA_VALIDATION.format(filename, e)) from e


_ERR_BASIC_MISSING_KEYS = "Registry entry '{}' missing required keys: {}"


def _basic_validate(data: dict[str, Any], filename: str) -> None:
    """Minimal structural validation when ``jsonschema`` is not installed."""
    required = {"name", "display_name", "documentation"}
    missing = required - set(data)
    if missing:
        msg = _ERR_BASIC_MISSING_KEYS.format(filename, ", ".join(sorted(missing)))
        raise ValueError(msg)
    if not isinstance(data["name"], str) or not data["name"]:
        raise ValueError(_ERR_NAME_EMPTY.format(filename))
    doc = data["documentation"]
    if "base_url" not in doc:
        raise ValueError(_ERR_MISSING_BASE_URL.format(filename))


__all__ = [
    "Registry",
    "RegistryEntry",
    "load_registry",
]

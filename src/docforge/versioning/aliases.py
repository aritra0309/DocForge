from __future__ import annotations

import json

from docforge.storage.metadata_store import MetadataStore
from docforge.versioning.comparator import sort_versions


def resolve_latest(
    metadata_store: MetadataStore,
    software: str,
    version: str,
) -> str:
    """Resolve the ``"latest"`` alias to the actual version string.

    Args:
        metadata_store: The metadata store to query.
        software: Software identifier.
        version: Version string (may be ``"latest"``).

    Returns:
        The resolved version string. If ``version`` is not ``"latest"``,
        returns it unchanged.

    Raises:
        ValueError: If ``"latest"`` is requested but no versions are indexed.
    """
    if version != "latest":
        return version

    versions = metadata_store.list_versions(software)
    if not versions:
        msg = f"No indexed versions found for '{software}' to resolve 'latest'"
        raise ValueError(msg)

    sorted_versions = sort_versions([v["version"] for v in versions])
    return sorted_versions[-1]


def set_latest_version_in_store(
    metadata_store: MetadataStore,
    software: str,
    version: str,
) -> None:
    """Record the given version as the latest in the software entry.

    This stores the latest version string in the ``indexed_software``
    table's ``config_snapshot`` field so it can be retrieved without
    iterating all indexed versions.

    Args:
        metadata_store: The metadata store.
        software: Software identifier.
        version: Version string to mark as latest.
    """
    entry = metadata_store.get_software(software)
    config: dict[str, str] = {}
    if entry and entry.get("config_snapshot"):
        try:
            config = json.loads(entry["config_snapshot"])
        except (json.JSONDecodeError, TypeError):
            config = {}
    config["latest_version"] = version
    metadata_store.upsert_software(
        software=software,
        display_name=(entry or {}).get("display_name", software),
        config_snapshot=json.dumps(config),
    )

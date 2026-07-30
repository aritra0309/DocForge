"""Version management — support multiple documentation versions coexisting."""

from docforge.versioning.aliases import (
    resolve_latest,
    set_latest_version_in_store,
)
from docforge.versioning.comparator import VersionComparator, compare_versions
from docforge.versioning.manager import VersionInfo, VersionManager

__all__ = [
    "VersionComparator",
    "VersionInfo",
    "VersionManager",
    "compare_versions",
    "resolve_latest",
    "set_latest_version_in_store",
]

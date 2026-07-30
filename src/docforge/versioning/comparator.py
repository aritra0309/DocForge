from __future__ import annotations

import re
from functools import total_ordering

_RE_SEMVER = re.compile(r"^(\d+)(?:\.(\d+)(?:\.(\d+))?)?([a-zA-Z0-9._-]*)?$")


@total_ordering
class VersionComparator:
    """Supports ordering of version strings like ``"17"``, ``"7.2.0"``, ``"3.1.4"``.

    Handles:
    - Plain numeric versions (``"17" > "16" > "15"``)
    - Semantic versions (``"7.2.0" > "7.1.0"``)
    - Mixed segment counts (``"17.0" > "16"``)
    """

    def __init__(self, version: str) -> None:
        self._raw = version
        m = _RE_SEMVER.match(version.strip())
        if m:
            parts = [int(p) if p else 0 for p in m.groups()[:3]]
            self._parts = parts
            self._suffix = (m.group(4) or "").lower()
        else:
            self._parts = []
            self._suffix = version.strip().lower()

    @property
    def raw(self) -> str:
        return self._raw

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VersionComparator):
            return NotImplemented
        return self._parts == other._parts and self._suffix == other._suffix

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, VersionComparator):
            return NotImplemented
        sp = self._parts
        op = other._parts
        max_len = max(len(sp), len(op))
        sp_padded = sp + [0] * (max_len - len(sp))
        op_padded = op + [0] * (max_len - len(op))
        if sp_padded != op_padded:
            return sp_padded < op_padded
        if (not self._suffix) != (not other._suffix):
            return bool(self._suffix)
        return self._suffix < other._suffix

    def __hash__(self) -> int:
        return hash((tuple(self._parts), self._suffix))

    def __repr__(self) -> str:
        return f"VersionComparator({self._raw!r})"


def compare_versions(a: str, b: str) -> int:
    """Compare two version strings.

    Returns:
        ``-1`` if ``a < b``, ``0`` if ``a == b``, ``1`` if ``a > b``.
    """
    va = VersionComparator(a)
    vb = VersionComparator(b)
    if va < vb:
        return -1
    if va > vb:
        return 1
    return 0


def sort_versions(versions: list[str]) -> list[str]:
    """Sort a list of version strings in ascending (oldest-first) order."""
    return sorted(versions, key=VersionComparator)

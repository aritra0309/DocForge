"""DocForge — Automatically discover, crawl, version, chunk, and index software documentation."""

from docforge._version import __version__
from docforge.api import DocForge

__all__ = ["DocForge", "__version__"]

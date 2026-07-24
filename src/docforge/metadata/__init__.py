"""Metadata generator — attaches rich metadata to every chunk."""

from docforge.metadata.breadcrumbs import (
    extract_breadcrumb_from_html,
    extract_breadcrumb_from_url,
)
from docforge.metadata.generator import MetadataGenerator
from docforge.metadata.hasher import compute_content_hash

__all__ = [
    "MetadataGenerator",
    "compute_content_hash",
    "extract_breadcrumb_from_html",
    "extract_breadcrumb_from_url",
]

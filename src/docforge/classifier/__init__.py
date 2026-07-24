"""Page classifier — determines the semantic type of each documentation page."""

from docforge.classifier.engine import ClassificationEngine
from docforge.classifier.rules import RuleBasedScorer
from docforge.classifier.taxonomy import (
    HEADING_PATTERN_SIGNALS,
    META_SIGNALS,
    TITLE_KEYWORD_SIGNALS,
    URL_PATH_SIGNALS,
)

__all__ = [
    "HEADING_PATTERN_SIGNALS",
    "META_SIGNALS",
    "TITLE_KEYWORD_SIGNALS",
    "URL_PATH_SIGNALS",
    "ClassificationEngine",
    "RuleBasedScorer",
]

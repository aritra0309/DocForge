"""Classification orchestrator — combines registry hints with rule-based scoring."""

from __future__ import annotations

from fnmatch import fnmatch

from docforge.classifier.rules import RuleBasedScorer
from docforge.core.interfaces import PageClassifier
from docforge.core.models import ClassifiedPage, ExtractedPage, PageType


class ClassificationEngine(PageClassifier):
    """Classifies pages using registry path hints, then rule-based scoring.

    1. Check registry ``page_type_hints`` path patterns → confidence 1.0.
    2. Run rule-based scorer → return if confidence ≥ threshold.
    3. Fall back to ``PageType.UNKNOWN``.
    """

    def __init__(
        self,
        page_type_hints: dict[str, list[str]] | None = None,
        confidence_threshold: float = 0.70,
    ) -> None:
        self.page_type_hints = page_type_hints or {}
        self.scorer = RuleBasedScorer(confidence_threshold=confidence_threshold)

    def classify(self, page: ExtractedPage) -> ClassifiedPage:
        page_type, confidence = self._resolve_type(page)
        return ClassifiedPage(
            url=page.url,
            title=page.title,
            markdown=page.markdown,
            headings=page.headings,
            code_blocks=page.code_blocks,
            breadcrumb=page.breadcrumb,
            raw_metadata=page.raw_metadata,
            page_type=page_type,
            confidence=confidence,
        )

    def _resolve_type(self, page: ExtractedPage) -> tuple[PageType, float]:
        hint_result = self._check_hints(page.url)
        if hint_result is not None:
            return (hint_result, 1.0)
        return self.scorer.classify(page)

    def _check_hints(self, url: str) -> PageType | None:
        hint_map: dict[str, PageType] = {
            "tutorial_paths": PageType.TUTORIAL,
            "reference_paths": PageType.API_REFERENCE,
            "guide_paths": PageType.GUIDE,
            "concept_paths": PageType.CONCEPTS,
            "example_paths": PageType.EXAMPLES,
            "faq_paths": PageType.FAQ,
            "getting_started_paths": PageType.GETTING_STARTED,
            "api_paths": PageType.API_REFERENCE,
            "function_ref_paths": PageType.FUNCTION_REFERENCE,
            "release_notes_paths": PageType.RELEASE_NOTES,
            "troubleshooting_paths": PageType.TROUBLESHOOTING,
            "migration_paths": PageType.MIGRATION,
            "configuration_paths": PageType.CONFIGURATION,
        }
        for hint_key, page_type in hint_map.items():
            patterns = self.page_type_hints.get(hint_key, [])
            for pattern in patterns:
                normalized = pattern.replace("{version}", "*")
                if fnmatch(url, normalized):
                    return page_type
        return None


__all__ = ["ClassificationEngine"]

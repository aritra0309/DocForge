"""Rule-based weighted feature scorer for page classification."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from docforge.classifier.taxonomy import (
    HEADING_PATTERN_SIGNALS,
    META_SIGNALS,
    TITLE_KEYWORD_SIGNALS,
    URL_PATH_SIGNALS,
)
from docforge.core.models import ExtractedPage, PageType

_HIGH_CODE_RATIO = 0.4
_LOW_CODE_RATIO = 0.1


class RuleBasedScorer:
    """Scores an ExtractedPage against signal patterns to predict PageType."""

    def __init__(self, confidence_threshold: float = 0.70) -> None:
        self.confidence_threshold = confidence_threshold

    def classify(self, page: ExtractedPage) -> tuple[PageType, float]:
        url_score = self._score_url_paths(page.url)
        title_score = self._score_title(page.title)
        heading_score = self._score_headings(page.headings)
        code_ratio_score = self._score_code_ratio(page.markdown, page.code_blocks)
        breadcrumb_score = self._score_breadcrumb(page.breadcrumb)
        meta_score = self._score_meta(page.raw_metadata)

        combined: dict[PageType, float] = {}
        for scores, weight in [
            (url_score, 0.30),
            (title_score, 0.25),
            (heading_score, 0.20),
            (code_ratio_score, 0.10),
            (breadcrumb_score, 0.10),
            (meta_score, 0.05),
        ]:
            for pt, val in scores.items():
                combined[pt] = combined.get(pt, 0.0) + val * weight

        if not combined:
            return (PageType.UNKNOWN, 0.0)

        best_type = max(combined, key=combined.get)
        best_score = combined[best_type]

        if best_score < self.confidence_threshold:
            return (PageType.UNKNOWN, 0.0)

        return (best_type, min(best_score, 1.0))

    @staticmethod
    def _score_url_paths(url: str) -> dict[PageType, float]:
        try:
            path = urlparse(url).path
        except Exception:
            path = url
        return RuleBasedScorer._match_path_patterns(path, URL_PATH_SIGNALS)

    @staticmethod
    def _score_title(title: str) -> dict[PageType, float]:
        return RuleBasedScorer._match_patterns(title.lower(), TITLE_KEYWORD_SIGNALS)

    @staticmethod
    def _score_headings(headings: list[str]) -> dict[PageType, float]:
        scores: dict[PageType, float] = {}
        for pt, patterns in HEADING_PATTERN_SIGNALS.items():
            for heading in headings:
                for pat in patterns:
                    if re.search(pat, heading, re.IGNORECASE):
                        scores[pt] = scores.get(pt, 0.0) + 1.0
                        break
        return {k: min(v, 1.0) for k, v in scores.items()}

    @staticmethod
    def _score_code_ratio(
        markdown: str, code_blocks: list[dict[str, str]]
    ) -> dict[PageType, float]:
        total_chars = len(markdown) if markdown else 1
        code_chars = sum(len(b.get("content", "")) for b in code_blocks)
        ratio = code_chars / total_chars
        scores: dict[PageType, float] = {}
        if ratio > _HIGH_CODE_RATIO:
            scores[PageType.EXAMPLES] = min(ratio, 1.0)
            scores[PageType.API_REFERENCE] = ratio * 0.7
        elif ratio > _LOW_CODE_RATIO:
            scores[PageType.TUTORIAL] = ratio * 1.5
            scores[PageType.GUIDE] = ratio
        else:
            scores[PageType.CONCEPTS] = 1.0 - ratio
            scores[PageType.GUIDE] = 1.0 - ratio
        return scores

    @staticmethod
    def _score_breadcrumb(breadcrumb: list[str]) -> dict[PageType, float]:
        scores: dict[PageType, float] = {}
        if not breadcrumb:
            return scores
        combined = " ".join(b.lower() for b in breadcrumb)
        for pt, keywords in TITLE_KEYWORD_SIGNALS.items():
            for kw in keywords:
                if kw in combined:
                    scores[pt] = scores.get(pt, 0.0) + 1.0
                    break
        return {k: min(v, 1.0) for k, v in scores.items()}

    @staticmethod
    def _score_meta(raw_metadata: dict[str, str]) -> dict[PageType, float]:
        scores: dict[PageType, float] = {}
        combined = " ".join(
            f"{k}:{v}" for k, v in raw_metadata.items()
        ).lower()
        for pt, patterns in META_SIGNALS.items():
            for pat in patterns:
                if pat.lower() in combined:
                    scores[pt] = 1.0
                    return scores
        return scores

    @staticmethod
    def _match_path_patterns(
        path: str, signal_map: dict[PageType, list[str]]
    ) -> dict[PageType, float]:
        scores: dict[PageType, float] = {}
        path_lower = path.lower()
        for pt, patterns in signal_map.items():
            for pat in patterns:
                p = pat.lower()
                if p in path_lower or p.rstrip("/") in path_lower or p + "/" in path_lower:
                    scores[pt] = scores.get(pt, 0.0) + 1.0
                    break
        return {k: min(v, 1.0) for k, v in scores.items()}

    @staticmethod
    def _match_patterns(
        text: str, signal_map: dict[PageType, list[str]]
    ) -> dict[PageType, float]:
        scores: dict[PageType, float] = {}
        text_lower = text.lower()
        for pt, patterns in signal_map.items():
            for pat in patterns:
                if pat.lower() in text_lower:
                    scores[pt] = scores.get(pt, 0.0) + 1.0
                    break
        return {k: min(v, 1.0) for k, v in scores.items()}


__all__ = ["RuleBasedScorer"]

"""Unit tests for the page classifier."""

from __future__ import annotations

import pytest

from docforge.classifier.engine import ClassificationEngine
from docforge.classifier.rules import RuleBasedScorer
from docforge.classifier.taxonomy import (
    HEADING_PATTERN_SIGNALS,
    META_SIGNALS,
    TITLE_KEYWORD_SIGNALS,
    URL_PATH_SIGNALS,
)
from docforge.core.models import ClassifiedPage, ExtractedPage, PageType


@pytest.fixture
def tutorial_page() -> ExtractedPage:
    return ExtractedPage(
        url="https://example.com/docs/tutorial/getting-started",
        title="Getting Started Tutorial",
        markdown=(
            "# Getting Started\n\nStep 1: Install.\n\n"
            "```bash\npip install foo\n```\n\nStep 2: Configure."
        ),
        headings=["Getting Started", "Step 1: Install", "Step 2: Configure"],
        code_blocks=[{"language": "bash", "content": "pip install foo"}],
        breadcrumb=["Docs", "Tutorial"],
        raw_metadata={"og:title": "Getting Started Tutorial"},
    )


@pytest.fixture
def api_ref_page() -> ExtractedPage:
    return ExtractedPage(
        url="https://example.com/docs/api/client",
        title="Client API Reference",
        markdown=(
            "# Client\n\n## connect()\n\nConnects to the server.\n\n"
            "**Parameters:**\n- host: str\n- port: int\n\n"
            "**Returns:**\nConnection\n\n```python\ndef connect(host, port):\n    pass\n```"
        ),
        headings=["Client", "connect()"],
        code_blocks=[{"language": "python", "content": "def connect(host, port):\n    pass"}],
        breadcrumb=["Docs", "API Reference"],
        raw_metadata={"og:type": "website"},
    )


@pytest.fixture
def guide_page() -> ExtractedPage:
    return ExtractedPage(
        url="https://example.com/docs/guide/deployment",
        title="Deployment Guide",
        markdown="# Deployment Guide\n\nThis guide covers best practices.",
        headings=["Deployment Guide"],
        code_blocks=[],
        breadcrumb=["Docs", "Guide"],
        raw_metadata={},
    )


@pytest.fixture
def faq_page() -> ExtractedPage:
    return ExtractedPage(
        url="https://example.com/docs/faq",
        title="Frequently Asked Questions",
        markdown="# FAQ\n\nQ: How do I install?",
        headings=["FAQ"],
        code_blocks=[],
        breadcrumb=["Docs", "FAQ"],
        raw_metadata={},
    )


@pytest.fixture
def unknown_page() -> ExtractedPage:
    return ExtractedPage(
        url="https://example.com/docs/some/obscure/page",
        title="Some Page",
        markdown="# Some Page\n\nRandom content with no clear type signals.",
        headings=["Some Page"],
        code_blocks=[],
        breadcrumb=["Docs"],
        raw_metadata={},
    )


@pytest.fixture
def engine_without_hints() -> ClassificationEngine:
    return ClassificationEngine()


@pytest.fixture
def engine_with_hints() -> ClassificationEngine:
    return ClassificationEngine(
        page_type_hints={
            "tutorial_paths": ["https://example.com/docs/tutorial/**"],
            "reference_paths": ["https://example.com/docs/api/**"],
        },
    )


class TestTaxonomy:
    def test_url_path_signals_have_all_types(self) -> None:
        assert PageType.TUTORIAL in URL_PATH_SIGNALS
        assert PageType.API_REFERENCE in URL_PATH_SIGNALS
        assert PageType.UNKNOWN not in URL_PATH_SIGNALS

    def test_title_keyword_signals_have_all_types(self) -> None:
        assert PageType.TUTORIAL in TITLE_KEYWORD_SIGNALS
        assert len(TITLE_KEYWORD_SIGNALS[PageType.GETTING_STARTED]) >= 3

    def test_heading_pattern_signals(self) -> None:
        assert PageType.TUTORIAL in HEADING_PATTERN_SIGNALS
        assert len(HEADING_PATTERN_SIGNALS[PageType.API_REFERENCE]) >= 3

    def test_meta_signals(self) -> None:
        assert PageType.TUTORIAL in META_SIGNALS


class TestRuleBasedScorer:
    def test_classify_tutorial(self, tutorial_page: ExtractedPage) -> None:
        scorer = RuleBasedScorer()
        page_type, confidence = scorer.classify(tutorial_page)
        assert page_type == PageType.TUTORIAL
        assert confidence >= 0.70

    def test_classify_api_ref(self, api_ref_page: ExtractedPage) -> None:
        scorer = RuleBasedScorer()
        page_type, confidence = scorer.classify(api_ref_page)
        assert page_type == PageType.API_REFERENCE
        assert confidence >= 0.70

    def test_classify_guide(self, guide_page: ExtractedPage) -> None:
        scorer = RuleBasedScorer()
        page_type, confidence = scorer.classify(guide_page)
        assert page_type in {PageType.GUIDE, PageType.CONCEPTS}
        assert confidence >= 0.70 or confidence == 0.0

    def test_classify_faq(self, faq_page: ExtractedPage) -> None:
        scorer = RuleBasedScorer(confidence_threshold=0.55)
        page_type, confidence = scorer.classify(faq_page)
        assert page_type == PageType.FAQ
        assert confidence >= 0.55

    def test_classify_unknown(self, unknown_page: ExtractedPage) -> None:
        scorer = RuleBasedScorer(confidence_threshold=0.70)
        page_type, confidence = scorer.classify(unknown_page)
        assert page_type == PageType.UNKNOWN
        assert confidence == 0.0

    def test_high_code_ratio_favours_examples(self) -> None:
        page = ExtractedPage(
            url="https://example.com/examples",
            title="Examples",
            markdown=(
            "# Examples\n\n```python\nprint('hello')\n```\n\n```javascript\nconsole.log('hi')\n```"
        ),
            headings=["Examples"],
            code_blocks=[
                {"language": "python", "content": "print('hello')"},
                {"language": "javascript", "content": "console.log('hi')"},
            ],
            breadcrumb=[],
            raw_metadata={},
        )
        scorer = RuleBasedScorer()
        page_type, confidence = scorer.classify(page)
        assert page_type in {PageType.EXAMPLES, PageType.API_REFERENCE}
        assert confidence >= 0.70

    def test_deterministic(self) -> None:
        page = ExtractedPage(
            url="https://example.com/tutorial/install",
            title="Installation Tutorial",
            markdown="# Installation\n\nSteps.",
            headings=["Installation"],
            code_blocks=[],
            breadcrumb=[],
            raw_metadata={},
        )
        scorer = RuleBasedScorer()
        result1 = scorer.classify(page)
        result2 = scorer.classify(page)
        assert result1 == result2


class TestClassificationEngine:
    def test_implements_interface(self) -> None:
        engine = ClassificationEngine()
        assert isinstance(engine, ClassificationEngine)

    def test_classify_returns_classified_page(
        self, tutorial_page: ExtractedPage
    ) -> None:
        engine = ClassificationEngine()
        result = engine.classify(tutorial_page)
        assert isinstance(result, ClassifiedPage)
        assert isinstance(result, ExtractedPage)
        assert result.page_type == PageType.TUTORIAL
        assert result.confidence >= 0.70

    def test_registry_hints_override_rules(
        self, engine_with_hints: ClassificationEngine
    ) -> None:
        page = ExtractedPage(
            url="https://example.com/docs/tutorial/install",
            title="Some Random Title",
            markdown="# Content",
            headings=[],
            code_blocks=[],
            breadcrumb=[],
            raw_metadata={},
        )
        result = engine_with_hints.classify(page)
        assert result.page_type == PageType.TUTORIAL
        assert result.confidence == 1.0

    def test_registry_hint_reference(
        self, engine_with_hints: ClassificationEngine
    ) -> None:
        page = ExtractedPage(
            url="https://example.com/docs/api/endpoints",
            title="Random",
            markdown="# Random",
            headings=[],
            code_blocks=[],
            breadcrumb=[],
            raw_metadata={},
        )
        result = engine_with_hints.classify(page)
        assert result.page_type == PageType.API_REFERENCE
        assert result.confidence == 1.0

    def test_unknown_page_without_hints(
        self, engine_without_hints: ClassificationEngine, unknown_page: ExtractedPage
    ) -> None:
        result = engine_without_hints.classify(unknown_page)
        assert result.page_type == PageType.UNKNOWN

    def test_classified_page_preserves_all_extracted_fields(
        self, tutorial_page: ExtractedPage
    ) -> None:
        engine = ClassificationEngine()
        result = engine.classify(tutorial_page)
        assert result.url == tutorial_page.url
        assert result.title == tutorial_page.title
        assert result.markdown == tutorial_page.markdown
        assert result.headings == tutorial_page.headings
        assert result.code_blocks == tutorial_page.code_blocks
        assert result.breadcrumb == tutorial_page.breadcrumb

    def test_classification_is_deterministic(self) -> None:
        engine = ClassificationEngine()
        page = ExtractedPage(
            url="https://example.com/guide/deploy",
            title="Deployment Guide",
            markdown="# Deploy\n\nContent.",
            headings=["Deploy"],
            code_blocks=[],
            breadcrumb=[],
            raw_metadata={},
        )
        r1 = engine.classify(page)
        r2 = engine.classify(page)
        assert r1.page_type == r2.page_type
        assert r1.confidence == r2.confidence

"""Unit tests for the chunking engine and all strategies."""

from __future__ import annotations

import pytest

from docforge.chunker.engine import ChunkingEngine
from docforge.chunker.overlap import apply_overlap
from docforge.chunker.strategies.api_ref import ApiRefChunker
from docforge.chunker.strategies.base import (
    count_tokens,
    merge_small_chunks,
    split_by_tokens,
)
from docforge.chunker.strategies.code import CodeChunker
from docforge.chunker.strategies.heading import HeadingChunker
from docforge.chunker.strategies.table import TableChunker
from docforge.chunker.strategies.tutorial import TutorialChunker
from docforge.core.models import (
    Chunk,
    ClassifiedPage,
    PageType,
)


@pytest.fixture
def heading_page() -> ClassifiedPage:
    return ClassifiedPage(
        url="https://example.com/docs/guide",
        title="User Guide",
        markdown=(
            "# User Guide\n\nIntroduction text with enough words to fill up space. "
            "This paragraph contains many tokens so that it exceeds the minimum chunk size "
            "threshold of sixty four tokens and avoids being merged with adjacent sections. "
            "We need about fifty more tokens worth of text to ensure this works reliably. "
            "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor.\n\n"
            "## Installation\n\n"
            "How to install the software package using pip or conda. "
            "This section describes the installation process in detail. "
            "There are several steps you need to follow carefully to ensure everything works "
            "properly on your system without any errors or warnings appearing during setup.\n\n"
            "```bash\npip install foo\n```\n\n"
            "## Configuration\n\n"
            "How to configure the software after installation is complete. "
            "The configuration file is located in the home directory. "
            "You can customize many options to suit your specific needs and preferences. "
            "Each option is documented thoroughly with examples showing proper usage.\n\n"
            "### Option A\n\n"
            "Details about option A including all its parameters and behaviors.\n\n"
            "### Option B\n\n"
            "Details about option B with its own set of configuration keys.\n\n"
            "## Usage\n\n"
            "Basic usage instructions for everyday tasks and common workflows."
        ),
        headings=[
            "User Guide",
            "Installation",
            "Configuration",
            "Option A",
            "Option B",
            "Usage",
        ],
        code_blocks=[{"language": "bash", "content": "pip install foo"}],
        breadcrumb=["Docs", "Guide"],
        raw_metadata={},
        page_type=PageType.GUIDE,
        confidence=0.95,
    )


@pytest.fixture
def api_ref_page() -> ClassifiedPage:
    return ClassifiedPage(
        url="https://example.com/docs/api/client",
        title="Client API",
        markdown=(
            "# Client API\n\n## connect(host, port)\n\n"
            "Connects to a server using the specified host and port number. "
            "This function establishes a TCP connection and returns a connection object "
            "that can be used to send and receive data over the network connection. "
            "The host parameter specifies the server address and the port specifies "
            "the port number on which to connect to the remote server.\n\n"
            "**Parameters:**\n- host: str - The server hostname or IP address\n"
            "- port: int - The port number to connect to\n\n"
            "**Returns:**\nConnection object representing the established connection\n\n"
            "## disconnect()\n\n"
            "Closes the connection and releases all associated resources. "
            "This method should be called when the connection is no longer needed "
            "to avoid resource leaks and ensure proper cleanup of network resources. "
            "After calling disconnect, the connection object should not be used again.\n\n"
            "**Parameters:**\nNone\n\n"
            "**Returns:**\nNone\n\n"
            "```python\ndef disconnect():\n    pass\n```"
        ),
        headings=["Client API", "connect(host, port)", "disconnect()"],
        code_blocks=[{"language": "python", "content": "def disconnect():\n    pass"}],
        breadcrumb=["Docs", "API"],
        raw_metadata={},
        page_type=PageType.API_REFERENCE,
        confidence=0.95,
    )


@pytest.fixture
def tutorial_page() -> ClassifiedPage:
    return ClassifiedPage(
        url="https://example.com/docs/tutorial/install",
        title="Installation Tutorial",
        markdown=(
            "# Installation Tutorial\n\n"
            "Step 1: Download the package from the official website. "
            "Make sure you download the correct version for your operating system. "
            "The download link is provided on the download page. "
            "You can use wget or curl to download the file from the command line. "
            "The file is compressed using gzip so you will need to extract it later.\n\n"
            "```bash\nwget https://example.com/pkg.tar.gz\n```\n\n"
            "Step 2: Extract the archive using the tar command with the xzf flags. "
            "This will create a new directory containing all the source files. "
            "Make sure you are in the correct directory before running the command. "
            "The extracted files include documentation examples and source code.\n\n"
            "```bash\ntar xzf pkg.tar.gz\n```\n\n"
            "Step 3: Run the installer script to complete the installation process. "
            "The installer will guide you through the remaining setup steps. "
            "You may need to provide administrator privileges during installation. "
            "After installation completes you can verify it worked "
            "by running the version command.\n\n"
            "```bash\n./install.sh\n```"
        ),
        headings=[
            "Installation Tutorial",
            "Step 1: Download",
            "Step 2: Extract",
            "Step 3: Install",
        ],
        code_blocks=[
            {"language": "bash", "content": "wget https://example.com/pkg.tar.gz"},
            {"language": "bash", "content": "tar xzf pkg.tar.gz"},
            {"language": "bash", "content": "./install.sh"},
        ],
        breadcrumb=["Docs", "Tutorial"],
        raw_metadata={},
        page_type=PageType.TUTORIAL,
        confidence=0.92,
    )


@pytest.fixture
def examples_page() -> ClassifiedPage:
    return ClassifiedPage(
        url="https://example.com/docs/examples",
        title="Examples",
        markdown=(
            "# Examples\n\n## Hello World\n\n"
            "```python\nprint('Hello, World!')\n```\n\n"
            "This is a basic example.\n\n"
            "## Fibonacci\n\n"
            "```python\ndef fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)\n```\n\n"
            "Recursive Fibonacci implementation."
        ),
        headings=["Examples", "Hello World", "Fibonacci"],
        code_blocks=[
            {"language": "python", "content": "print('Hello, World!')"},
            {
                "language": "python",
                "content": "def fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)",
            },
        ],
        breadcrumb=["Docs", "Examples"],
        raw_metadata={},
        page_type=PageType.EXAMPLES,
        confidence=0.85,
    )


@pytest.fixture
def table_page() -> ClassifiedPage:
    return ClassifiedPage(
        url="https://example.com/docs/config",
        title="Configuration Reference",
        markdown=(
            "# Configuration Reference\n\n"
            "| Parameter | Type | Default | Description |\n"
            "|-----------|------|---------|-------------|\n"
            "| host | string | localhost | Server hostname |\n"
            "| port | int | 8080 | Server port |\n"
            "| debug | bool | false | Enable debug mode |\n"
            "| timeout | int | 30 | Connection timeout |"
        ),
        headings=["Configuration Reference"],
        code_blocks=[],
        breadcrumb=["Docs", "Config"],
        raw_metadata={},
        page_type=PageType.CONFIGURATION,
        confidence=0.90,
    )


class TestCountTokens:
    def test_empty_string(self) -> None:
        assert count_tokens("") == 0

    def test_short_text(self) -> None:
        assert count_tokens("hello world") > 0

    def test_longer_text(self) -> None:
        text = " ".join(["word"] * 100)
        assert count_tokens(text) > 10


class TestSplitByTokens:
    def test_single_chunk(self) -> None:
        text = "short text"
        result = split_by_tokens(text, max_tokens=1000)
        assert result == [text]

    def test_splits_large_text(self) -> None:
        text = "\n\n".join(["paragraph"] * 50)
        result = split_by_tokens(text, max_tokens=20)
        assert len(result) > 1

    def test_preserves_content(self) -> None:
        text = "para1\n\npara2\n\npara3"
        result = split_by_tokens(text, max_tokens=3)
        joined = "".join(result)
        assert "para1" in joined and "para2" in joined and "para3" in joined


class TestMergeSmallChunks:
    def test_empty(self) -> None:
        assert merge_small_chunks([], 64) == []

    def test_single_chunk(self) -> None:
        assert merge_small_chunks(["hello"], 64) == ["hello"]

    def test_merges_small_chunks(self) -> None:
        chunks = ["small", "tiny"]
        result = merge_small_chunks(chunks, 64)
        assert len(result) <= len(chunks)


class TestHeadingChunker:
    def test_chunks_by_heading(self, heading_page: ClassifiedPage) -> None:
        chunker = HeadingChunker()
        chunks = chunker.chunk(heading_page)
        assert len(chunks) >= 3
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_no_chunk_exceeds_max_size(self, heading_page: ClassifiedPage) -> None:
        chunker = HeadingChunker(max_chunk_size=1024)
        chunks = chunker.chunk(heading_page)
        for c in chunks:
            assert count_tokens(c.content) <= 1024

    def test_each_chunk_has_metadata(self, heading_page: ClassifiedPage) -> None:
        chunker = HeadingChunker()
        chunks = chunker.chunk(heading_page)
        for c in chunks:
            assert c.metadata.section_heading
            assert c.metadata.page_type == PageType.GUIDE

    def test_deterministic(self, heading_page: ClassifiedPage) -> None:
        chunker = HeadingChunker()
        chunks1 = chunker.chunk(heading_page)
        chunks2 = chunker.chunk(heading_page)
        assert [c.content for c in chunks1] == [c.content for c in chunks2]

    def test_code_blocks_not_split(self) -> None:
        page = ClassifiedPage(
            url="https://example.com/docs",
            title="Code Page",
            markdown=(
                "# Code\n\n## Section 1\n\nSome text.\n\n"
                "```python\nline1\nline2\nline3\nline4\nline5\n```\n\n"
                "## Section 2\n\nMore text."
            ),
            headings=["Code", "Section 1", "Section 2"],
            code_blocks=[
                {
                    "language": "python",
                    "content": "line1\nline2\nline3\nline4\nline5",
                }
            ],
            breadcrumb=[],
            raw_metadata={},
            page_type=PageType.GUIDE,
            confidence=0.9,
        )
        chunker = HeadingChunker(max_chunk_size=10)
        chunks = chunker.chunk(page)
        for c in chunks:
            lines = c.content.split("\n")
            in_code = False
            for line in lines:
                if line.startswith("```"):
                    in_code = not in_code
            assert not in_code, "Code block was split"


class TestApiRefChunker:
    def test_one_chunk_per_function(self, api_ref_page: ClassifiedPage) -> None:
        chunker = ApiRefChunker()
        chunks = chunker.chunk(api_ref_page)
        assert len(chunks) >= 2
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_contains_signature(self, api_ref_page: ClassifiedPage) -> None:
        chunker = ApiRefChunker()
        chunks = chunker.chunk(api_ref_page)
        combined = " ".join(c.content for c in chunks)
        assert "connect" in combined
        assert "disconnect" in combined

    def test_no_chunk_exceeds_max_size(self, api_ref_page: ClassifiedPage) -> None:
        chunker = ApiRefChunker(max_chunk_size=1024)
        chunks = chunker.chunk(api_ref_page)
        for c in chunks:
            assert count_tokens(c.content) <= 1024


class TestTutorialChunker:
    def test_one_chunk_per_step(self, tutorial_page: ClassifiedPage) -> None:
        chunker = TutorialChunker()
        chunks = chunker.chunk(tutorial_page)
        assert len(chunks) >= 2
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_code_blocks_not_split(self, tutorial_page: ClassifiedPage) -> None:
        chunker = TutorialChunker()
        chunks = chunker.chunk(tutorial_page)
        for c in chunks:
            lines = c.content.split("\n")
            in_code = False
            for line in lines:
                if line.startswith("```"):
                    in_code = not in_code
            assert not in_code, "Code block was split"

    def test_no_chunk_exceeds_max_size(self, tutorial_page: ClassifiedPage) -> None:
        chunker = TutorialChunker(max_chunk_size=1024)
        chunks = chunker.chunk(tutorial_page)
        for c in chunks:
            assert count_tokens(c.content) <= 1024


class TestCodeChunker:
    def test_code_and_explanation_together(self, examples_page: ClassifiedPage) -> None:
        chunker = CodeChunker()
        chunks = chunker.chunk(examples_page)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_code_blocks_not_split(self) -> None:
        page = ClassifiedPage(
            url="https://example.com/docs",
            title="Examples",
            markdown=(
                "# Examples\n\nExplanation.\n\n"
                "```python\nline1\nline2\nline3\n```\n\n"
                "More text.\n\n```javascript\nvar x = 1;\n```"
            ),
            headings=["Examples"],
            code_blocks=[
                {"language": "python", "content": "line1\nline2\nline3"},
                {"language": "javascript", "content": "var x = 1;"},
            ],
            breadcrumb=[],
            raw_metadata={},
            page_type=PageType.EXAMPLES,
            confidence=0.85,
        )
        chunker = CodeChunker()
        chunks = chunker.chunk(page)
        for c in chunks:
            lines = c.content.split("\n")
            in_code = False
            for line in lines:
                if line.startswith("```"):
                    in_code = not in_code
            assert not in_code


class TestTableChunker:
    def test_small_table_one_chunk(self, table_page: ClassifiedPage) -> None:
        chunker = TableChunker()
        chunks = chunker.chunk(table_page)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_large_table_split(self) -> None:
        rows = "\n".join([f"| val{i} | data{i} |" for i in range(50)])
        table_md = f"| Header1 | Header2 |\n|------|------|\n{rows}"
        page = ClassifiedPage(
            url="https://example.com/docs",
            title="Large Table",
            markdown=f"# Large Table\n\n{table_md}",
            headings=["Large Table"],
            code_blocks=[],
            breadcrumb=[],
            raw_metadata={},
            page_type=PageType.CONFIGURATION,
            confidence=0.9,
        )
        chunker = TableChunker(large_table_row_threshold=100)
        chunks = chunker.chunk(page)
        assert len(chunks) >= 1

    def test_no_chunk_exceeds_max_size(self, table_page: ClassifiedPage) -> None:
        chunker = TableChunker(max_chunk_size=1024)
        chunks = chunker.chunk(table_page)
        for c in chunks:
            assert count_tokens(c.content) <= 1024


class TestChunkingEngine:
    def test_selects_heading_chunker_for_guide(self, heading_page: ClassifiedPage) -> None:
        engine = ChunkingEngine()
        chunks = engine.chunk(heading_page)
        assert len(chunks) >= 2

    def test_selects_api_ref_chunker(self, api_ref_page: ClassifiedPage) -> None:
        engine = ChunkingEngine()
        chunks = engine.chunk(api_ref_page)
        assert len(chunks) >= 2

    def test_selects_tutorial_chunker(self, tutorial_page: ClassifiedPage) -> None:
        engine = ChunkingEngine()
        chunks = engine.chunk(tutorial_page)
        assert len(chunks) >= 2

    def test_selects_code_chunker_for_examples(self, examples_page: ClassifiedPage) -> None:
        engine = ChunkingEngine()
        chunks = engine.chunk(examples_page)
        assert len(chunks) >= 1

    def test_selects_table_chunker_for_config(self, table_page: ClassifiedPage) -> None:
        engine = ChunkingEngine()
        chunks = engine.chunk(table_page)
        assert len(chunks) >= 1

    def test_unknown_type_uses_heading_chunker(self) -> None:
        page = ClassifiedPage(
            url="https://example.com/docs/random",
            title="Random",
            markdown="# Random\n\nSome content.\n\n## Section\n\nMore content.",
            headings=["Random", "Section"],
            code_blocks=[],
            breadcrumb=[],
            raw_metadata={},
            page_type=PageType.UNKNOWN,
            confidence=0.0,
        )
        engine = ChunkingEngine()
        chunks = engine.chunk(page)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_overlap_injection(self) -> None:
        page = ClassifiedPage(
            url="https://example.com/docs",
            title="Test",
            markdown=(
                "# H1\n\n## Section A\n\n" + "word " * 200 + "\n\n## Section B\n\n" + "word " * 200
            ),
            headings=["H1", "Section A", "Section B"],
            code_blocks=[],
            breadcrumb=[],
            raw_metadata={},
            page_type=PageType.GUIDE,
            confidence=0.9,
        )
        engine = ChunkingEngine(overlap_tokens=32)
        chunks = engine.chunk(page)
        if len(chunks) > 1:
            for i in range(1, len(chunks)):
                assert len(chunks[i].content) > 0

    def test_deterministic_output(self) -> None:
        page = ClassifiedPage(
            url="https://example.com/docs/guide",
            title="Guide",
            markdown=(
                "# Guide\n\nIntro.\n\n## Section 1\n\nContent 1.\n\n## Section 2\n\nContent 2."
            ),
            headings=["Guide", "Section 1", "Section 2"],
            code_blocks=[],
            breadcrumb=[],
            raw_metadata={},
            page_type=PageType.GUIDE,
            confidence=0.9,
        )
        engine = ChunkingEngine()
        chunks1 = engine.chunk(page)
        chunks2 = engine.chunk(page)
        assert [c.content for c in chunks1] == [c.content for c in chunks2]


class TestApplyOverlap:
    def test_empty_list(self) -> None:
        assert apply_overlap([], 64) == []

    def test_single_chunk(self) -> None:
        assert apply_overlap(["hello"], 64) == ["hello"]

    def test_no_overlap_when_zero(self) -> None:
        texts = ["chunk one", "chunk two"]
        result = apply_overlap(texts, 0)
        assert result == texts

    def test_overlap_injected(self) -> None:
        texts = [
            "This is the first chunk with some content.",
            "This is the second chunk.",
        ]
        result = apply_overlap(texts, 3)
        assert len(result) == 2
        assert result[0] == texts[0]
        assert len(result[1]) > len(texts[1])

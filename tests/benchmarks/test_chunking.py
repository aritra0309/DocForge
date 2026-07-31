"""Chunking benchmark tests."""

from __future__ import annotations

import pytest

from docforge.chunker.engine import ChunkingEngine
from docforge.core.models import ClassifiedPage, PageType


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


@pytest.mark.benchmark
def test_chunking_benchmark_heading_chunker(heading_page: ClassifiedPage) -> None:
    """Benchmark HeadingChunker throughput."""
    engine = ChunkingEngine()

    # Warm up
    for _ in range(10):
        engine.chunk(heading_page)

    # Benchmark
    iterations = 1000
    import time
    start = time.perf_counter()
    for _ in range(iterations):
        engine.chunk(heading_page)
    elapsed = time.perf_counter() - start

    chunks_per_sec = iterations / elapsed * 5  # ~5 chunks per page
    assert chunks_per_sec >= 1000, f"Chunking throughput {chunks_per_sec:.1f} chunks/sec below target 1000"

    from tests.benchmarks import benchmark
    with benchmark("chunking_heading_chunker", int(iterations * 5)):
        pass


@pytest.mark.benchmark
def test_chunking_benchmark_api_ref_chunker(api_ref_page: ClassifiedPage) -> None:
    """Benchmark ApiRefChunker throughput."""
    engine = ChunkingEngine()

    # Warm up
    for _ in range(10):
        engine.chunk(api_ref_page)

    # Benchmark
    iterations = 1000
    import time
    start = time.perf_counter()
    for _ in range(iterations):
        engine.chunk(api_ref_page)
    elapsed = time.perf_counter() - start

    chunks_per_sec = iterations / elapsed * 3  # ~3 chunks per page
    assert chunks_per_sec >= 1000, f"Chunking throughput {chunks_per_sec:.1f} chunks/sec below target 1000"

    from tests.benchmarks import benchmark
    with benchmark("chunking_api_ref_chunker", int(iterations * 3)):
        pass


@pytest.mark.benchmark
def test_chunking_benchmark_tutorial_chunker(tutorial_page: ClassifiedPage) -> None:
    """Benchmark TutorialChunker throughput."""
    engine = ChunkingEngine()

    # Warm up
    for _ in range(10):
        engine.chunk(tutorial_page)

    # Benchmark
    iterations = 1000
    import time
    start = time.perf_counter()
    for _ in range(iterations):
        engine.chunk(tutorial_page)
    elapsed = time.perf_counter() - start

    chunks_per_sec = iterations / elapsed * 4  # ~4 chunks per page
    assert chunks_per_sec >= 1000, f"Chunking throughput {chunks_per_sec:.1f} chunks/sec below target 1000"

    from tests.benchmarks import benchmark
    with benchmark("chunking_tutorial_chunker", int(iterations * 4)):
        pass


@pytest.mark.benchmark
def test_chunking_benchmark_mixed_pages(heading_page: ClassifiedPage, api_ref_page: ClassifiedPage, tutorial_page: ClassifiedPage) -> None:
    """Benchmark chunking on mixed page types."""
    engine = ChunkingEngine()
    pages = [heading_page, api_ref_page, tutorial_page]

    # Warm up
    for _ in range(5):
        for page in pages:
            engine.chunk(page)

    # Benchmark
    iterations = 500
    import time
    start = time.perf_counter()
    for _ in range(iterations):
        for page in pages:
            engine.chunk(page)
    elapsed = time.perf_counter() - start

    total_chunks = iterations * 12  # ~4 chunks per page * 3 pages
    chunks_per_sec = total_chunks / elapsed
    assert chunks_per_sec >= 1000, f"Chunking throughput {chunks_per_sec:.1f} chunks/sec below target 1000"

    from tests.benchmarks import benchmark
    with benchmark("chunking_mixed_pages", total_chunks):
        pass
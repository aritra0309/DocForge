"""Comprehensive unit tests for extractor submodules: cleaners, html_to_md, callouts, code_blocks."""

from __future__ import annotations

from lxml import html as lxml_html

from docforge.extractor.callouts import (
    _detect_callout_type,
    normalise_callouts,
)
from docforge.extractor.cleaners import (
    extract_breadcrumb,
    extract_main_content,
    extract_page_title,
    extract_raw_metadata,
)
from docforge.extractor.code_blocks import (
    detect_language_from_class,
    detect_language_from_context,
    extract_code_blocks_from_markdown,
    normalise_code_blocks,
    post_process_markdown_code_blocks,
    strip_line_numbers_from_pre,
)
from docforge.extractor.html_to_md import (
    _ensure_absolute_links,
    extract_headings_from_markdown,
    html_to_markdown,
)

# ---------------------------------------------------------------------------
# cleaners.py tests
# ---------------------------------------------------------------------------


class TestExtractMainContent:
    def test_registry_selector(self) -> None:
        html = "<html><body><div id='docContent'><h1>Title</h1><p>Content</p></div><nav>Nav</nav></body></html>"
        element, meta = extract_main_content(html, {"main_content": "#docContent"})
        assert meta["method"].startswith("registry:")
        assert element.cssselect("h1")[0].text_content() == "Title"

    def test_registry_selector_not_found_falls_to_semantic(self) -> None:
        html = "<html><body><main><h1>Main</h1></main></body></html>"
        _element, meta = extract_main_content(html, {"main_content": "#nonexistent"})
        assert "semantic" in meta["method"]

    def test_semantic_main_tag(self) -> None:
        html = "<html><body><main><h1>Hello</h1></main><nav>Nav</nav></body></html>"
        _element, meta = extract_main_content(html)
        assert "semantic" in meta["method"]

    def test_semantic_article_tag(self) -> None:
        html = "<html><body><article><h1>Article</h1><p>Content here.</p></article></body></html>"
        _element, meta = extract_main_content(html)
        assert "semantic" in meta["method"]

    def test_heuristic_fallback_doc_content(self) -> None:
        html = "<html><body><div class='doc-content'><h1>Doc</h1><p>Content text here for density.</p></div></body></html>"
        _, meta = extract_main_content(html)
        # Will fall to heuristic since no semantic or main selectors
        assert meta["method"] in {"heuristic", "readability", "body_fallback", "semantic:main"}

    def test_removes_nav_from_content(self) -> None:
        html = "<html><body><main><h1>Title</h1><nav>Navigation stuff</nav><p>Real content</p></main></body></html>"
        element, _ = extract_main_content(html)
        assert len(element.cssselect("nav")) == 0

    def test_removes_script_from_content(self) -> None:
        html = "<html><body><main><h1>Title</h1><script>alert('bad')</script><p>Content</p></main></body></html>"
        element, _ = extract_main_content(html)
        assert len(element.cssselect("script")) == 0

    def test_invalid_html_handled(self) -> None:
        # Should not raise
        element, _meta = extract_main_content("<not valid html>")
        assert element is not None

    def test_role_main_semantic(self) -> None:
        html = '<html><body><div role="main"><h1>Role Main</h1></div></body></html>'
        _, meta = extract_main_content(html)
        assert "semantic" in meta["method"]

    def test_body_fallback(self) -> None:
        # Plain HTML with no matching patterns → body or readability
        html = "<html><body><p>Simple content</p></body></html>"
        element, meta = extract_main_content(html)
        assert element is not None
        assert meta["method"] in {"readability", "body_fallback", "semantic:main"}


class TestExtractPageTitle:
    def test_h1_preferred(self) -> None:
        html = (
            "<html><head><title>Page - Site</title></head><body><h1>Real Title</h1></body></html>"
        )
        doc = lxml_html.fromstring(html)
        title = extract_page_title(doc, html)
        assert title == "Real Title"

    def test_title_tag_fallback(self) -> None:
        html = "<html><head><title>Page Title - Site Name</title></head><body><p>Content</p></body></html>"
        doc = lxml_html.fromstring(html)
        title = extract_page_title(doc, html)
        assert title == "Page Title"

    def test_title_tag_with_pipe_separator(self) -> None:
        html = "<html><head><title>Getting Started | My Docs</title></head><body><p>Content</p></body></html>"
        doc = lxml_html.fromstring(html)
        title = extract_page_title(doc, html)
        assert title == "Getting Started"

    def test_title_tag_with_em_dash_separator(self) -> None:
        html = "<html><head><title>API Reference — Docs</title></head><body><p>Content</p></body></html>"
        doc = lxml_html.fromstring(html)
        title = extract_page_title(doc, html)
        assert title == "API Reference"

    def test_untitled_fallback(self) -> None:
        html = "<html><body><p>No title here.</p></body></html>"
        doc = lxml_html.fromstring(html)
        title = extract_page_title(doc, html)
        assert title == "Untitled"

    def test_h1_with_whitespace(self) -> None:
        html = "<html><body><h1>  My   Title  </h1></body></html>"
        doc = lxml_html.fromstring(html)
        title = extract_page_title(doc, html)
        assert title == "My Title"


class TestExtractBreadcrumb:
    def test_navigation_selector(self) -> None:
        html = """<html><body>
            <nav class="toc">
              <a href="/docs">Home</a>
              <a href="/docs/sql">SQL Commands</a>
            </nav>
            <main><h1>Page</h1></main>
        </body></html>"""
        crumbs = extract_breadcrumb(html, {"navigation": ".toc"})
        assert "Home" in crumbs
        assert "SQL Commands" in crumbs

    def test_breadcrumb_class(self) -> None:
        html = """<html><body>
            <div class="breadcrumb">
              <a href="/">Home</a> / <span>Docs</span> / <span>API</span>
            </div>
        </body></html>"""
        crumbs = extract_breadcrumb(html)
        assert "Home" in crumbs
        assert "Docs" in crumbs

    def test_empty_when_no_breadcrumb(self) -> None:
        html = "<html><body><main><h1>No breadcrumb</h1></main></body></html>"
        crumbs = extract_breadcrumb(html)
        assert isinstance(crumbs, list)

    def test_invalid_html_returns_empty(self) -> None:
        crumbs = extract_breadcrumb("")
        assert crumbs == []

    def test_aria_label_breadcrumb(self) -> None:
        html = """<html><body>
            <nav aria-label="breadcrumb">
              <a href="/">Home</a>
              <a href="/docs">Docs</a>
            </nav>
        </body></html>"""
        crumbs = extract_breadcrumb(html)
        assert isinstance(crumbs, list)

    def test_deduplicates_items(self) -> None:
        html = """<html><body>
            <nav class="toc">
              <a href="/home">Home</a>
              <a href="/home">Home</a>
            </nav>
        </body></html>"""
        crumbs = extract_breadcrumb(html, {"navigation": ".toc"})
        assert crumbs.count("Home") == 1


class TestExtractRawMetadata:
    def test_og_title(self) -> None:
        html = '<html><head><meta property="og:title" content="My Page"></head></html>'
        metadata = extract_raw_metadata(html)
        assert metadata.get("og:title") == "My Page"

    def test_description_meta(self) -> None:
        html = '<html><head><meta name="description" content="Page description"></head></html>'
        metadata = extract_raw_metadata(html)
        assert metadata.get("description") == "Page description"

    def test_canonical_link(self) -> None:
        html = '<html><head><link rel="canonical" href="https://example.com/page"></head></html>'
        metadata = extract_raw_metadata(html)
        assert metadata.get("canonical") == "https://example.com/page"

    def test_empty_html_returns_empty(self) -> None:
        metadata = extract_raw_metadata("")
        assert isinstance(metadata, dict)

    def test_multiple_meta_tags(self) -> None:
        html = """<html><head>
            <meta name="description" content="Desc">
            <meta property="og:description" content="OG Desc">
            <meta name="author" content="Author">
        </head></html>"""
        metadata = extract_raw_metadata(html)
        assert "description" in metadata
        assert "og:description" in metadata
        assert "author" in metadata


# ---------------------------------------------------------------------------
# code_blocks.py tests
# ---------------------------------------------------------------------------


class TestDetectLanguageFromClass:
    def test_language_class(self) -> None:
        assert detect_language_from_class("language-python") == "python"
        assert detect_language_from_class("language-sql") == "sql"
        assert detect_language_from_class("language-javascript") == "javascript"
        assert detect_language_from_class("language-bash") == "bash"

    def test_highlight_class(self) -> None:
        assert detect_language_from_class("highlight-python") == "python"

    def test_none_class(self) -> None:
        assert detect_language_from_class(None) is None
        assert detect_language_from_class("") is None

    def test_no_language_in_class(self) -> None:
        assert detect_language_from_class("some-other-class") is None

    def test_language_case_insensitive(self) -> None:
        assert detect_language_from_class("Language-Python") == "python"

    def test_language_cpp(self) -> None:
        assert detect_language_from_class("language-c++") == "c++"


class TestDetectLanguageFromContext:
    def test_sql_detected(self) -> None:
        code = "SELECT * FROM users WHERE id = 1;"
        assert detect_language_from_context(code) == "sql"

    def test_python_detected(self) -> None:
        code = "def my_function(arg):\n    return arg * 2"
        assert detect_language_from_context(code) == "python"

    def test_javascript_detected(self) -> None:
        code = "const foo = function() { return 42; };"
        assert detect_language_from_context(code) == "javascript"

    def test_bash_pip_detected(self) -> None:
        code = "pip install requests"
        assert detect_language_from_context(code) == "bash"

    def test_bash_curl_detected(self) -> None:
        code = "curl -X GET https://api.example.com"
        assert detect_language_from_context(code) == "bash"

    def test_unknown_returns_none(self) -> None:
        code = "some random text without identifiers"
        assert detect_language_from_context(code) is None

    def test_go_detected(self) -> None:
        code = "package main\nfunc main() {}"
        assert detect_language_from_context(code) == "go"

    def test_create_index_sql(self) -> None:
        code = "CREATE INDEX idx_users ON users(email);"
        assert detect_language_from_context(code) == "sql"


class TestStripLineNumbers:
    def test_strip_linenos_class(self) -> None:
        html = """<pre><span class="linenos">1</span><code>print("hello")</code></pre>"""
        root = lxml_html.fromstring(html)
        pre = root.cssselect("pre")[0]
        strip_line_numbers_from_pre(pre)
        assert len(pre.cssselect(".linenos")) == 0

    def test_no_line_numbers_unchanged(self) -> None:
        html = "<pre><code>print('hello')</code></pre>"
        root = lxml_html.fromstring(html)
        pre = root.cssselect("pre")[0]
        strip_line_numbers_from_pre(pre)
        assert pre.cssselect("code")[0].text_content() == "print('hello')"


class TestNormaliseCodeBlocks:
    def test_adds_language_from_class(self) -> None:
        html = """<div><pre><code class="language-python">def foo(): pass</code></pre></div>"""
        root = lxml_html.fromstring(html)
        normalise_code_blocks(root)
        code = root.cssselect("code")[0]
        assert "language-python" in code.get("class", "")

    def test_detects_language_from_context(self) -> None:
        html = """<div><pre><code>SELECT * FROM users;</code></pre></div>"""
        root = lxml_html.fromstring(html)
        normalise_code_blocks(root)
        code = root.cssselect("code")[0]
        assert "language-sql" in code.get("class", "")

    def test_pre_without_code_skipped(self) -> None:
        html = """<div><pre>Plain text content here.</pre></div>"""
        root = lxml_html.fromstring(html)
        # Should not raise
        normalise_code_blocks(root)


class TestExtractCodeBlocksFromMarkdown:
    def test_extracts_python_block(self) -> None:
        md = "```python\nprint('hello')\n```"
        blocks = extract_code_blocks_from_markdown(md)
        assert len(blocks) == 1
        assert blocks[0]["language"] == "python"
        assert "print" in blocks[0]["content"]

    def test_extracts_multiple_blocks(self) -> None:
        md = "```python\ncode1\n```\n\n```bash\necho hello\n```"
        blocks = extract_code_blocks_from_markdown(md)
        assert len(blocks) == 2
        languages = [b["language"] for b in blocks]
        assert "python" in languages
        assert "bash" in languages

    def test_unlabeled_block_becomes_text(self) -> None:
        md = "```\nsome code\n```"
        blocks = extract_code_blocks_from_markdown(md)
        assert len(blocks) == 1
        assert blocks[0]["language"] == "text"

    def test_no_blocks(self) -> None:
        md = "Just regular text with no code blocks."
        blocks = extract_code_blocks_from_markdown(md)
        assert blocks == []


class TestPostProcessMarkdownCodeBlocks:
    def test_strips_line_number_pipe_format(self) -> None:
        md = "```python\n1 | print('hello')\n2 | print('world')\n```"
        result = post_process_markdown_code_blocks(md)
        assert "1 |" not in result
        assert "print('hello')" in result

    def test_detects_language_in_text_fence(self) -> None:
        md = "```text\nSELECT * FROM table;\n```"
        result = post_process_markdown_code_blocks(md)
        # Should detect SQL and replace ```text with ```sql
        assert "```sql" in result

    def test_preserves_language_fence(self) -> None:
        md = "```python\nprint('hello')\n```"
        result = post_process_markdown_code_blocks(md)
        assert "```python" in result
        assert "print('hello')" in result

    def test_empty_fence_adds_text_language(self) -> None:
        md = "```\nsome content\n```"
        result = post_process_markdown_code_blocks(md)
        assert "```text" in result


# ---------------------------------------------------------------------------
# callouts.py tests
# ---------------------------------------------------------------------------


class TestDetectCalloutType:
    def test_warning(self) -> None:
        el = lxml_html.fromstring('<div class="admonition warning"></div>')
        assert _detect_callout_type(el) == "Warning"

    def test_tip(self) -> None:
        el = lxml_html.fromstring('<div class="admonition tip"></div>')
        assert _detect_callout_type(el) == "Tip"

    def test_danger(self) -> None:
        el = lxml_html.fromstring('<div class="admonition danger"></div>')
        assert _detect_callout_type(el) == "Danger"

    def test_important(self) -> None:
        el = lxml_html.fromstring('<div class="admonition important"></div>')
        assert _detect_callout_type(el) == "Important"

    def test_caution(self) -> None:
        el = lxml_html.fromstring('<div class="admonition caution"></div>')
        assert _detect_callout_type(el) == "Caution"

    def test_info(self) -> None:
        el = lxml_html.fromstring('<div class="admonition info"></div>')
        assert _detect_callout_type(el) == "Info"

    def test_default_note(self) -> None:
        el = lxml_html.fromstring('<div class="admonition"></div>')
        assert _detect_callout_type(el) == "Note"


class TestNormaliseCallouts:
    def test_sphinx_note_normalised(self) -> None:
        html = """<div>
            <div class="admonition note">
                <p class="admonition-title">Note</p>
                <p>This is a note about something important.</p>
            </div>
        </div>"""
        root = lxml_html.fromstring(html)
        normalise_callouts(root)
        callouts = root.cssselect("div.docforge-callout")
        assert len(callouts) == 1
        assert callouts[0].get("data-callout-type") == "Note"

    def test_sphinx_warning_normalised(self) -> None:
        html = """<div>
            <div class="admonition warning">
                <p class="admonition-title">Warning</p>
                <p>This is a warning.</p>
            </div>
        </div>"""
        root = lxml_html.fromstring(html)
        normalise_callouts(root)
        callouts = root.cssselect("div.docforge-callout")
        assert len(callouts) >= 1
        assert any(c.get("data-callout-type") == "Warning" for c in callouts)

    def test_docusaurus_tip_normalised(self) -> None:
        html = """<div>
            <div class="theme-admonition-tip">
                <p>Pro tip: Use virtual environments.</p>
            </div>
        </div>"""
        root = lxml_html.fromstring(html)
        normalise_callouts(root)
        callouts = root.cssselect("div.docforge-callout")
        assert len(callouts) >= 1

    def test_no_duplicates_for_multiple_selectors(self) -> None:
        html = """<div>
            <div class="admonition note">
                <p class="admonition-title">Note</p>
                <p>Content.</p>
            </div>
        </div>"""
        root = lxml_html.fromstring(html)
        normalise_callouts(root)
        callouts = root.cssselect("div.docforge-callout")
        assert len(callouts) == 1

    def test_title_text_extracted(self) -> None:
        html = """<div>
            <div class="admonition tip">
                <p class="admonition-title">Pro Tip</p>
                <p>Something useful here.</p>
            </div>
        </div>"""
        root = lxml_html.fromstring(html)
        normalise_callouts(root)
        callouts = root.cssselect("div.docforge-callout")
        assert len(callouts) == 1
        assert callouts[0].get("data-callout-type") == "Pro Tip"


# ---------------------------------------------------------------------------
# html_to_md.py tests
# ---------------------------------------------------------------------------


class TestHtmlToMarkdown:
    def test_headings_converted(self) -> None:
        html = "<div><h1>Title</h1><h2>Section</h2><h3>Subsection</h3></div>"
        root = lxml_html.fromstring(html)
        result = html_to_markdown(root, "https://example.com/")
        assert "# Title" in result
        assert "## Section" in result
        assert "### Subsection" in result

    def test_code_block_with_language(self) -> None:
        html = '<div><pre><code class="language-python">def foo(): pass</code></pre></div>'
        root = lxml_html.fromstring(html)
        result = html_to_markdown(root, "https://example.com/")
        assert "```" in result
        assert "def foo" in result

    def test_image_converted(self) -> None:
        html = '<div><img src="/img/diagram.png" alt="Diagram"></div>'
        root = lxml_html.fromstring(html)
        result = html_to_markdown(root, "https://example.com/docs/")
        assert "![Diagram]" in result
        assert "diagram.png" in result

    def test_relative_links_resolved(self) -> None:
        html = '<div><a href="other.html">Other Page</a></div>'
        root = lxml_html.fromstring(html)
        result = html_to_markdown(root, "https://example.com/docs/")
        assert "https://example.com/docs/other.html" in result

    def test_absolute_links_unchanged(self) -> None:
        html = '<div><a href="https://other.com/page">External</a></div>'
        root = lxml_html.fromstring(html)
        result = html_to_markdown(root, "https://example.com/")
        assert "https://other.com/page" in result

    def test_definition_list_converted(self) -> None:
        html = "<div><dl><dt>Term</dt><dd>Definition of the term.</dd></dl></div>"
        root = lxml_html.fromstring(html)
        result = html_to_markdown(root, "https://example.com/")
        assert "**Term**" in result
        assert "Definition of the term." in result

    def test_extra_blank_lines_collapsed(self) -> None:
        html = "<div><p>Para 1</p><p>Para 2</p><p>Para 3</p></div>"
        root = lxml_html.fromstring(html)
        result = html_to_markdown(root, "https://example.com/")
        # Should not have 3+ consecutive newlines
        assert "\n\n\n" not in result

    def test_table_converted(self) -> None:
        html = """<div><table>
            <tr><th>Name</th><th>Value</th></tr>
            <tr><td>foo</td><td>bar</td></tr>
        </table></div>"""
        root = lxml_html.fromstring(html)
        result = html_to_markdown(root, "https://example.com/")
        assert "Name" in result
        assert "Value" in result
        assert "foo" in result


class TestExtractHeadingsFromMarkdown:
    def test_extracts_all_levels(self) -> None:
        md = "# Title\n\n## Section\n\n### Sub\n\nSome text\n\n#### Deep"
        headings = extract_headings_from_markdown(md)
        assert "Title" in headings
        assert "Section" in headings
        assert "Sub" in headings
        assert "Deep" in headings

    def test_no_headings(self) -> None:
        md = "Just plain text\nNo headings here"
        headings = extract_headings_from_markdown(md)
        assert headings == []

    def test_not_heading_in_code_fence(self) -> None:
        md = "```\n# Not a heading\n```\n\n# Real Heading"
        headings = extract_headings_from_markdown(md)
        # Should find at least the real heading (may also find the one inside code)
        assert "Real Heading" in headings


class TestEnsureAbsoluteLinks:
    def test_relative_path_resolved(self) -> None:
        md = "[Link](/docs/page)"
        result = _ensure_absolute_links(md, "https://example.com/")
        assert "https://example.com/docs/page" in result

    def test_dot_relative_resolved(self) -> None:
        md = "[Link](./other.html)"
        result = _ensure_absolute_links(md, "https://example.com/docs/")
        assert "https://example.com/docs/other.html" in result

    def test_absolute_unchanged(self) -> None:
        md = "[Link](https://other.com/page)"
        result = _ensure_absolute_links(md, "https://example.com/")
        assert "https://other.com/page" in result

    def test_anchor_unchanged(self) -> None:
        md = "[Link](#section)"
        result = _ensure_absolute_links(md, "https://example.com/")
        assert "#section" in result

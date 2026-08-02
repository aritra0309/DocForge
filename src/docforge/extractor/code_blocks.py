"""Code block post-processing utilities."""

from __future__ import annotations

import re

from lxml import html as lxml_html

LANGUAGE_CLASS_RE = re.compile(r"language-([\w+#.-]+)", re.IGNORECASE)
HIGHLIGHT_CLASS_RE = re.compile(r"highlight(?:-(\w+))?", re.IGNORECASE)
LINE_NUMBER_SELECTORS: tuple[str, ...] = (
    ".linenos",
    ".line-numbers",
    ".gutter",
    ".highlight .lineno",
    "span.linenos",
)

CONTEXT_LANGUAGE_HINTS: tuple[tuple[str, str], ...] = (
    (r"\bSELECT\b|\bINSERT\b|\bCREATE TABLE\b|\bCREATE INDEX\b", "sql"),
    (r"\bdef \w+\(", "python"),
    (r"\bfunction\b|\bconst\b|\bimport\b", "javascript"),
    (r"\bpackage main\b|\bfunc \w+\(", "go"),
    (r"\bcurl\b|\bwget\b|\bpip install\b|\bnpm install\b", "bash"),
    (r"\bkubectl\b|\bdocker\b", "bash"),
)


def detect_language_from_class(class_attr: str | None) -> str | None:
    """Extract language from a code element's class attribute."""
    if not class_attr:
        return None
    match = LANGUAGE_CLASS_RE.search(class_attr)
    if match:
        return match.group(1).lower()
    match = HIGHLIGHT_CLASS_RE.search(class_attr)
    if match and match.group(1):
        return match.group(1).lower()
    return None


def detect_language_from_context(code_text: str) -> str | None:
    """Guess programming language from code content heuristics."""
    for pattern, language in CONTEXT_LANGUAGE_HINTS:
        if re.search(pattern, code_text, re.IGNORECASE):
            return language
    return None


def strip_line_numbers_from_pre(pre: lxml_html.HtmlElement) -> None:
    """Remove injected line-number elements from a code block."""
    for selector in LINE_NUMBER_SELECTORS:
        for element in pre.cssselect(selector):
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)


def normalise_code_blocks(root: lxml_html.HtmlElement) -> None:
    """Normalise pre/code blocks: strip line numbers and annotate language."""
    for pre in root.cssselect("pre"):
        strip_line_numbers_from_pre(pre)
        code_el = pre.find(".//code")
        if code_el is None:
            continue

        language = detect_language_from_class(code_el.get("class"))
        code_text = code_el.text_content()
        if not language:
            language = detect_language_from_context(code_text)

        if language:
            existing = code_el.get("class", "")
            if f"language-{language}" not in existing:
                code_el.set("class", f"{existing} language-{language}".strip())


def extract_code_blocks_from_markdown(markdown: str) -> list[dict[str, str]]:
    """Parse fenced code blocks from markdown output."""
    blocks: list[dict[str, str]] = []
    pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    for match in pattern.finditer(markdown):
        language = match.group(1) or "text"
        content = match.group(2).strip("\n")
        blocks.append({"language": language, "content": content})
    return blocks


def post_process_markdown_code_blocks(markdown: str) -> str:
    """Clean code fences: remove line-number prefixes and fix empty language tags."""
    lines = markdown.split("\n")
    output: list[str] = []
    in_fence = False
    fence_lang = ""

    for line in lines:
        if line.startswith("```"):
            if not in_fence:
                in_fence = True
                fence_lang = line[3:].strip()
                if not fence_lang:
                    fence_lang = "text"
                output.append(f"```{fence_lang}")
            else:
                in_fence = False
                fence_lang = ""
                output.append("```")
            continue

        if in_fence:
            cleaned = re.sub(r"^\s*\d+\s*\|\s?", "", line)
            cleaned = re.sub(r"^\s*\d+\s{2,}", "", cleaned)
            output.append(cleaned)
        else:
            output.append(line)

    processed = "\n".join(_normalise_fence_spacing(output))

    def replace_text_fence(match: re.Match[str]) -> str:
        content = match.group(1)
        detected = detect_language_from_context(content)
        if detected:
            return f"```{detected}\n{content}```"
        return match.group(0)

    return re.sub(r"```text\n(.*?)```", replace_text_fence, processed, flags=re.DOTALL)


def _normalise_fence_spacing(lines: list[str]) -> list[str]:
    """Apply stable Markdown spacing around fenced code blocks."""
    normalised: list[str] = []
    in_fence = False
    after_closing_fence = False
    for line in lines:
        if line.startswith("```"):
            if in_fence:
                while normalised and not normalised[-1]:
                    normalised.pop()
                normalised.append(line)
                in_fence = False
                after_closing_fence = True
            else:
                if normalised and normalised[-1]:
                    normalised.append("")
                while len(normalised) > 1 and not normalised[-1] and not normalised[-2]:
                    normalised.pop()
                normalised.append(line)
                in_fence = True
                after_closing_fence = False
            continue

        if after_closing_fence and line and normalised and normalised[-1]:
            normalised.append("")
        if not in_fence and line.startswith("#") and normalised and normalised[-1]:
            normalised.append("")
        normalised.append(line)
        if line:
            after_closing_fence = False

    return normalised

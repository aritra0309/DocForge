"""Helpers for the 20-page fixture documentation site used in integration tests."""

from __future__ import annotations

from pathlib import Path

import respx

FIXTURES_HTML = Path(__file__).resolve().parent / "html"
FIXTURE_SITE_BASE = "https://docs.fixture.test"
FIXTURE_SITE_PAGE_COUNT = 20


def fixture_site_page_count() -> int:
    return FIXTURE_SITE_PAGE_COUNT


def load_fixture_html(name: str) -> str:
    return (FIXTURES_HTML / name).read_text(encoding="utf-8")


def mock_fixture_site(
    *,
    base: str = FIXTURE_SITE_BASE,
    index_paths: tuple[str, ...] = ("/docs/1.0", "/docs/index.html"),
    last_modified: str | None = None,
    robots_status: int = 404,
    robots_body: str = "",
    private_status: int = 404,
) -> dict[str, object]:
    """Register respx routes for the full 20-page fixture site.

    Returns a dict of named routes for call-count assertions.
    """
    headers: dict[str, str] = {}
    if last_modified:
        headers["Last-Modified"] = last_modified

    index_html = load_fixture_html("fixture_site_index.html")
    routes: dict[str, object] = {}

    if robots_status == 200:
        routes["robots"] = respx.get(f"{base}/robots.txt").respond(
            status_code=200, text=robots_body
        )
    else:
        routes["robots"] = respx.get(f"{base}/robots.txt").respond(status_code=robots_status)

    for path in index_paths:
        key = "index" if path.endswith(("index.html", "1.0")) else path
        routes[key] = respx.get(f"{base}{path}").respond(200, text=index_html, headers=headers)

    for i in range(1, FIXTURE_SITE_PAGE_COUNT + 1):
        html = load_fixture_html(f"fixture_site_page{i}.html")
        routes[f"page{i}"] = respx.get(f"{base}/docs/page{i}.html").respond(
            200, text=html, headers=headers
        )

    routes["private"] = respx.get(f"{base}/docs/private/secret.html").respond(
        status_code=private_status,
        text="<html>secret</html>" if private_status == 200 else "",
    )
    routes["blog"] = respx.get(f"{base}/blog/post.html").respond(status_code=404)
    return routes


def fixture_sitemap_xml(
    *, page_count: int = FIXTURE_SITE_PAGE_COUNT, changed_page: int | None = None
) -> str:
    """Build a sitemap covering the fixture site pages."""
    entries: list[str] = [
        _sitemap_entry(f"{FIXTURE_SITE_BASE}/docs/1.0", "2025-01-01"),
    ]
    for i in range(1, page_count + 1):
        lastmod = "2025-02-01" if changed_page == i else "2025-01-01"
        entries.append(_sitemap_entry(f"{FIXTURE_SITE_BASE}/docs/page{i}.html", lastmod))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "\n".join(entries)
        + "\n</urlset>"
    )


def _sitemap_entry(loc: str, lastmod: str) -> str:
    return f"  <url><loc>{loc}</loc><lastmod>{lastmod}</lastmod></url>"

"""Table extraction and GFM normalisation utilities."""

from __future__ import annotations

from lxml import html as lxml_html


def _cell_text(cell: lxml_html.HtmlElement) -> str:
    return " ".join(cell.text_content().split())


def _expand_row_cells(cells: list[lxml_html.HtmlElement]) -> list[str]:
    """Expand colspan into repeated cell values for GFM rendering."""
    expanded: list[str] = []
    for cell in cells:
        text = _cell_text(cell)
        colspan = int(cell.get("colspan", "1") or "1")
        expanded.extend([text] * max(1, colspan))
    return expanded


def flatten_table(table: lxml_html.HtmlElement) -> lxml_html.HtmlElement:
    """Flatten colspan/rowspan into a simple GFM-compatible table structure."""
    new_table = lxml_html.Element("table")

    rows = table.cssselect("tr")
    if not rows:
        return table

    header_cells = rows[0].cssselect("th, td")
    if header_cells:
        thead = lxml_html.Element("thead")
        header_row = lxml_html.Element("tr")
        for text in _expand_row_cells(list(header_cells)):
            th = lxml_html.Element("th")
            th.text = text
            header_row.append(th)
        thead.append(header_row)
        new_table.append(thead)

    tbody = lxml_html.Element("tbody")
    start_idx = 1 if header_cells else 0
    for row in rows[start_idx:]:
        cells = row.cssselect("th, td")
        if not cells:
            continue
        new_row = lxml_html.Element("tr")
        for text in _expand_row_cells(list(cells)):
            td = lxml_html.Element("td")
            td.text = text
            new_row.append(td)
        tbody.append(new_row)

    if len(tbody):
        new_table.append(tbody)

    return new_table if len(new_table) else table


def normalise_tables(root: lxml_html.HtmlElement) -> None:
    """Flatten all tables in the content tree for GFM compatibility."""
    for table in root.cssselect("table"):
        flattened = flatten_table(table)
        parent = table.getparent()
        if parent is not None:
            parent.replace(table, flattened)


def is_renderable_gfm_table(table: lxml_html.HtmlElement) -> bool:
    """Return True if the table has at least one row with cells."""
    for row in table.cssselect("tr"):
        if row.cssselect("th, td"):
            return True
    return False

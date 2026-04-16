"""PyMuPDF PDF parsing module for CyberRAG.

This module provides fast PDF parsing using PyMuPDF when a PDF contains
extractable embedded text. It offers two key functions:

1. `detect_embedded_text()` - Sampling-based detection of embedded text in PDFs
2. `parse_pdf_with_pymupdf()` - PyMuPDF-based parsing with output format
   aligned to DotsOCR's structured layout blocks

The output format mirrors DotsOCR's LayoutBlock structure to ensure
downstream chunking and processing code remains compatible with both parsers.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Layout categories aligned with DotsOCR's LayoutCategory
LAYOUT_TITLE = "Title"
LAYOUT_SECTION_HEADER = "Section-header"
LAYOUT_TEXT = "Text"
LAYOUT_LIST_ITEM = "List-item"
LAYOUT_TABLE = "Table"
LAYOUT_FORMULA = "Formula"
LAYOUT_PAGE_HEADER = "Page-header"
LAYOUT_PAGE_FOOTER = "Page-footer"
LAYOUT_CAPTION = "Caption"
LAYOUT_FOOTNOTE = "Footnote"
LAYOUT_PICTURE = "Picture"


def detect_embedded_text(
    pdf_path: str,
    sample_ratio: float = 0.15,
    min_sample_pages: int = 3,
    max_sample_pages: int = 15,
    char_threshold: int = 50,
) -> bool:
    """Detect whether a PDF contains extractable embedded text.

    Uses a sampling strategy:
    - Short documents (≤15 pages): check all pages
    - Long documents: uniformly sample 15% of pages, first and last pages are always checked
    - Decision: if more than half of sampled pages have ≥ char_threshold characters → True

    Args:
        pdf_path: Path to the PDF file.
        sample_ratio: Ratio of pages to sample for long documents (default 0.15 = 15%).
        min_sample_pages: Minimum number of pages to sample.
        max_sample_pages: Maximum number of pages to sample.
        char_threshold: Minimum character count to consider a page as having text.

    Returns:
        True if the PDF likely contains embedded text, False otherwise.
    """
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
    except Exception as exc:
        logger.warning("Failed to open PDF %s: %s", pdf_path, exc)
        return False

    if page_count == 0:
        return False

    # Determine which pages to sample
    if page_count <= 15:
        pages_to_check = list(range(page_count))
    else:
        # Always include first and last pages
        sampled = {0, page_count - 1}

        # Sample additional pages uniformly
        num_to_sample = min(max_sample_pages, max(min_sample_pages, int(page_count * sample_ratio)))
        remaining_slots = num_to_sample - len(sampled)

        if remaining_slots > 0:
            # Generate uniform samples between first and last (exclusive)
            interior_pages = list(range(1, page_count - 1))
            if interior_pages:
                step = len(interior_pages) / remaining_slots
                for i in range(remaining_slots):
                    idx = int(i * step)
                    if idx < len(interior_pages):
                        sampled.add(interior_pages[idx])

        pages_to_check = sorted(sampled)

    # Check each sampled page for embedded text
    text_pages = 0
    try:
        doc = fitz.open(pdf_path)
        for page_num in pages_to_check:
            if page_num >= len(doc):
                continue
            page = doc[page_num]
            text = page.get_text()
            if len(text.strip()) >= char_threshold:
                text_pages += 1
        doc.close()
    except Exception as exc:
        logger.warning("Error checking pages in %s: %s", pdf_path, exc)
        return False

    # Decision: more than half of sampled pages should have text
    threshold = len(pages_to_check) / 2
    has_embedded = text_pages > threshold

    logger.info(
        "Embedded text detection for %s: %d/%d pages with text (threshold=%.1f) → %s",
        Path(pdf_path).name,
        text_pages,
        len(pages_to_check),
        threshold,
        "detected" if has_embedded else "not detected",
    )

    return has_embedded


def _extract_page_spans(page: fitz.Page) -> list[dict[str, Any]]:
    """Extract text spans from a PyMuPDF page with font metadata.

    Each span contains: font_name, font_size, bold, text, color, bbox

    Returns:
        List of span dictionaries sorted by vertical position (top-to-bottom, left-to-right).
    """
    # Get text with font information using 'dict' option
    blocks = page.get_text("dict")["blocks"]

    spans: list[dict[str, Any]] = []
    for block in blocks:
        if block.get("type") != 0:  # Skip image blocks
            continue

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                spans.append({
                    "font_name": span.get("font", ""),
                    "font_size": span.get("size", 0),
                    "bold": "Bold" in span.get("font", "") or span.get("flags", 0) & 16 != 0,
                    "text": span.get("text", ""),
                    "color": span.get("color", 0),
                    "bbox": span.get("bbox", []),
                })

    # Sort by vertical position (y first, then x)
    spans.sort(key=lambda s: (s["bbox"][1] if s["bbox"] else 0, s["bbox"][0] if s["bbox"] else 0))
    return spans


def _classify_span_category(span: dict[str, Any]) -> str:
    """Classify a text span into a layout category based on font metadata.

    Uses font size and bold attributes along with content heuristics.

    Returns:
        One of: Title, Section-header, List-item, Text, Table, etc.
    """
    font_size = span.get("font_size", 0)
    bold = span.get("bold", False)
    text = span.get("text", "").strip()

    # Title: large font size (≥15) or explicitly bold + large
    if font_size >= 15 or (bold and font_size >= 12):
        return LAYOUT_TITLE

    # Section header: medium-large font or bold
    if font_size >= 12 or bold:
        return LAYOUT_SECTION_HEADER

    # List item: check for list prefixes
    if re.match(r"^\s*[\d\w][\.\)\:]+\s+", text) or re.match(r"^\s*[-•\*]\s+", text):
        return LAYOUT_LIST_ITEM

    # Check for special section header patterns
    if re.match(r"^(第[一二三四五六七八九十百千\d]+条|Article\s+\d+|Section\s+\d+)", text, re.IGNORECASE):
        return LAYOUT_SECTION_HEADER

    # Page number pattern (footer)
    if re.match(r"^\s*[-–—]?\s*\d+\s*[-–—]?$", text) and len(text) < 10:
        return LAYOUT_PAGE_FOOTER

    return LAYOUT_TEXT


def _group_spans_into_blocks(
    spans: list[dict[str, Any]],
    page_width: float,
    page_height: float,
) -> list[dict[str, Any]]:
    """Group consecutive spans of the same category into layout blocks.

    Args:
        spans: List of text spans with metadata.
        page_width: Width of the page (for bbox normalization).
        page_height: Height of the page.

    Returns:
        List of grouped block dictionaries with: category, text, bbox.
    """
    if not spans:
        return []

    blocks: list[dict[str, Any]] = []
    current_category: str | None = None
    current_text_parts: list[str] = []
    current_bbox: list[float] = []

    def flush_block():
        nonlocal current_category, current_text_parts, current_bbox
        if current_category and current_text_parts:
            text = "".join(current_text_parts).strip()
            if text:
                blocks.append({
                    "category": current_category,
                    "text": text,
                    "bbox": current_bbox[:] if current_bbox else [0, 0, page_width, page_height],
                })
        current_category = None
        current_text_parts = []
        current_bbox = []

    for span in spans:
        category = _classify_span_category(span)
        text = span.get("text", "")

        if not text.strip():
            continue

        # Start a new block on category change or for structural elements
        if current_category is None:
            current_category = category
        elif category != current_category and category in {
            LAYOUT_TITLE, LAYOUT_SECTION_HEADER, LAYOUT_TABLE
        }:
            flush_block()
            current_category = category

        current_text_parts.append(text)

        # Update bbox (union of all span bboxes)
        bbox = span.get("bbox", [])
        if len(bbox) == 4:
            if not current_bbox:
                current_bbox = list(bbox)
            else:
                current_bbox[0] = min(current_bbox[0], bbox[0])  # x1
                current_bbox[1] = min(current_bbox[1], bbox[1])  # y1
                current_bbox[2] = max(current_bbox[2], bbox[2])  # x2
                current_bbox[3] = max(current_bbox[3], bbox[3])  # y2

    flush_block()
    return blocks


def _extract_tables_from_page(page: fitz.Page) -> list[dict[str, Any]]:
    """Extract tables from a page using PyMuPDF's table detection.

    Returns:
        List of table blocks with bbox and markdown-formatted content.
    """
    tables: list[dict[str, Any]] = []

    try:
        tablefinder = page.find_tables()
        if tablefinder.tables:
            for table in tablefinder.tables:
                bbox = table.bbox
                extracted = table.extract()
                if not extracted:
                    continue

                # Convert to markdown format
                rows = []
                for row in extracted:
                    cells = [str(cell).strip() if cell else "" for cell in row]
                    rows.append("| " + " | ".join(cells) + " |")

                if rows:
                    markdown = "\n".join(rows)
                    tables.append({
                        "category": LAYOUT_TABLE,
                        "text": markdown,
                        "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                    })
    except Exception as exc:
        logger.debug("Table extraction failed for page: %s", exc)

    return tables


def parse_pdf_with_pymupdf(pdf_path: str) -> tuple[list[Document], dict[str, Any]]:
    """Parse a PDF using PyMuPDF, output format aligned to DotsOCR structure.

    This function extracts text from PDFs with embedded text and converts
    it to structured layout blocks that mirror DotsOCR's output format.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A tuple of:
        - List of LangChain Document objects (one per page)
        - A dict containing structured layout data compatible with DotsOCR format
    """
    logger.info("Parsing PDF with PyMuPDF: %s", pdf_path)

    doc = fitz.open(pdf_path)
    page_count = len(doc)

    documents: list[Document] = []
    all_blocks: list[dict[str, Any]] = []
    current_section: str | None = None

    for page_num in range(page_count):
        page = doc[page_num]
        page_width = page.rect.width
        page_height = page.rect.height

        # Extract spans with font metadata
        spans = _extract_page_spans(page)

        # Group spans into layout blocks
        blocks = _group_spans_into_blocks(spans, page_width, page_height)

        # Extract and merge tables
        tables = _extract_tables_from_page(page)
        if tables:
            # Insert tables into appropriate positions based on bbox
            for table in tables:
                table_y = table["bbox"][1]
                inserted = False
                for i, block in enumerate(blocks):
                    if block["bbox"][1] > table_y:
                        blocks.insert(i, table)
                        inserted = True
                        break
                if not inserted:
                    blocks.append(table)

        # Build page blocks with order information
        page_blocks_dicts: list[dict[str, Any]] = []
        for order, block in enumerate(blocks, start=1):
            block_dict = {
                "page_number": page_num + 1,
                "order_in_page": order,
                "category": block["category"],
                "category_raw": block["category"],
                "bbox": block["bbox"],
                "text": block["text"],
            }
            page_blocks_dicts.append(block_dict)
            all_blocks.append(block_dict)

            # Track section headers
            if block["category"] in {LAYOUT_TITLE, LAYOUT_SECTION_HEADER}:
                current_section = block["text"].strip()

        # Build cleaned text for the page
        text_parts: list[str] = []
        for block in blocks:
            category = block["category"]
            text = block["text"].strip()

            if not text:
                continue

            if category == LAYOUT_TITLE:
                text_parts.append(f"\n## {text}\n")
            elif category == LAYOUT_SECTION_HEADER:
                text_parts.append(f"\n### {text}\n")
            elif category == LAYOUT_TABLE:
                text_parts.append(f"\n{table}\n")
            else:
                text_parts.append(text)

        cleaned_text = "\n\n".join(text_parts)

        metadata = {
            "source": pdf_path,
            "source_path": pdf_path,
            "page": page_num + 1,
            "total_pages": page_count,
            "parser": "pymupdf",
            "section": current_section,
            "layout_blocks": page_blocks_dicts,
            "num_elements": len(blocks),
        }

        documents.append(Document(page_content=cleaned_text, metadata=metadata))

    doc.close()

    # Build structured output compatible with DotsOCR format
    structured_data: dict[str, Any] = {
        "source": pdf_path,
        "total_pages": page_count,
        "pages_parsed": page_count,
        "parser": "pymupdf",
        "blocks": all_blocks,
    }

    logger.info(
        "PyMuPDF parsing complete: %d pages, %d total blocks",
        page_count,
        len(all_blocks),
    )

    return documents, structured_data

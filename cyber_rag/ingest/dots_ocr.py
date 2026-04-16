"""DotsOCR PDF parsing module for CyberRAG.

Uses the DotsOCR vision-language model (via OneAPI) to perform structured
layout analysis on PDF pages.  Each page is rendered to a PNG image, sent
to the OCR API, and the returned JSON is parsed into structured layout
blocks that carry category, bbox, and text information.

The structured output is then converted to LangChain ``Document`` objects
whose metadata preserves the layout provenance (page number, category,
bbox, section headers, etc.) so that downstream chunking can leverage
structural boundaries.

If the DotsOCR API fails for any reason, the caller should fall back to
the traditional PyPDFLoader.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import fitz  # PyMuPDF
import requests
from langchain_core.documents import Document

from cyber_rag.config import DOTS_OCR_PROMPT, DotsOcrConfig

logger = logging.getLogger(__name__)


# ============================================================
# Data models for structured OCR output
# ============================================================


class LayoutCategory(str, Enum):
    """Standard layout categories returned by DotsOCR."""

    CAPTION = "Caption"
    FOOTNOTE = "Footnote"
    FORMULA = "Formula"
    LIST_ITEM = "List-item"
    PAGE_FOOTER = "Page-footer"
    PAGE_HEADER = "Page-header"
    PICTURE = "Picture"
    SECTION_HEADER = "Section-header"
    TABLE = "Table"
    TEXT = "Text"
    TITLE = "Title"

    @classmethod
    def from_raw(cls, value: str | None) -> LayoutCategory | None:
        if value is None:
            return None
        normalized = str(value).strip()
        if not normalized:
            return None
        for category in cls:
            if category.value == normalized:
                return category
        return None


@dataclass(slots=True)
class BBox:
    """Axis-aligned bounding box [x1, y1, x2, y2]."""

    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_sequence(cls, value: Any) -> BBox | None:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return None
        if len(value) < 4:
            return None
        try:
            return cls(
                x1=float(value[0]),
                y1=float(value[1]),
                x2=float(value[2]),
                y2=float(value[3]),
            )
        except (TypeError, ValueError):
            return None

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(slots=True)
class LayoutBlock:
    """A single layout element extracted from a PDF page."""

    page_number: int  # 1-based
    order_in_page: int
    category: LayoutCategory | None
    category_raw: str
    bbox: BBox | None
    text: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "order_in_page": self.order_in_page,
            "category": self.category.value if self.category else None,
            "category_raw": self.category_raw,
            "bbox": self.bbox.to_list() if self.bbox else None,
            "text": self.text,
        }


@dataclass(slots=True)
class StructuredPageResult:
    """Parsed result for a single PDF page."""

    page_number: int  # 0-based for internal, stored as-is
    raw_content: str
    elements: list[LayoutBlock]
    cleaned_text: str


# ============================================================
# JSON extraction helpers (adapted from parser markdown_cleaner)
# ============================================================


def _extract_json_payload(raw_text: str) -> Any:
    """Extract JSON payload from DotsOCR response text."""
    stripped = (raw_text or "").strip()
    if not stripped:
        return None

    # Direct JSON
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Code-block wrapped JSON
    code_block_match = re.fullmatch(
        r"```(?:json)?\s*([\s\S]*?)\s*```",
        stripped,
        flags=re.IGNORECASE,
    )
    if code_block_match:
        inner = code_block_match.group(1).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass

    # Find first JSON structure in text
    start_candidates = [
        i for i in (stripped.find("["), stripped.find("{")) if i >= 0
    ]
    if not start_candidates:
        return None

    decoder = json.JSONDecoder()
    for start in sorted(start_candidates):
        try:
            payload, _ = decoder.raw_decode(stripped[start:])
            return payload
        except json.JSONDecodeError:
            continue
    return None


def _extract_elements_from_raw(raw_text: str) -> list[dict[str, Any]]:
    """Extract layout elements from DotsOCR raw response text."""
    payload = _extract_json_payload(raw_text)
    if payload is None:
        # Fallback: loose recovery of bbox/category/text objects
        return _extract_json_like_elements(raw_text)

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):
        for key in ("layout", "elements", "items", "result", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]

    return []


def _extract_json_like_elements(raw_text: str) -> list[dict[str, Any]]:
    """Loose recovery of DotsOCR elements when strict JSON parsing fails."""
    text = (raw_text or "").strip()
    if not text:
        return []

    starts = [match.start() for match in re.finditer(r'\{\s*"bbox"\s*:', text)]
    if not starts:
        return []

    starts.append(len(text))
    elements: list[dict[str, Any]] = []

    for index, start in enumerate(starts[:-1]):
        end = starts[index + 1]
        chunk = text[start:end].strip().lstrip(",").rstrip(", \n\r\t]")
        if not chunk:
            continue

        last_brace = chunk.rfind("}")
        if last_brace >= 0:
            chunk = chunk[: last_brace + 1]

        bbox_match = re.search(r'"bbox"\s*:\s*\[([^\]]+)\]', chunk, flags=re.DOTALL)
        category_match = re.search(r'"category"\s*:\s*"([^"]+)"', chunk)
        if not bbox_match or not category_match:
            continue

        bbox_numbers = re.findall(r"-?\d+(?:\.\d+)?", bbox_match.group(1))
        if len(bbox_numbers) < 4:
            continue

        element: dict[str, Any] = {
            "bbox": [float(n) for n in bbox_numbers[:4]],
            "category": category_match.group(1),
        }

        text_key = re.search(r'"text"\s*:\s*"', chunk)
        if text_key:
            text_start = text_key.end()
            text_end = chunk.rfind('"')
            if text_end >= text_start:
                raw_value = chunk[text_start:text_end]
                element["text"] = _decode_json_like_string(raw_value)

        elements.append(element)
    return elements


def _decode_json_like_string(value: str) -> str:
    """Decode common escape sequences in JSON-like strings."""
    if not value:
        return ""

    def _unicode_replacer(match: re.Match[str]) -> str:
        try:
            return chr(int(match.group(1), 16))
        except ValueError:
            return match.group(0)

    decoded = re.sub(r"\\u([0-9a-fA-F]{4})", _unicode_replacer, value)
    decoded = decoded.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
    decoded = decoded.replace('\\"', '"').replace("\\/", "/")
    decoded = decoded.replace("\\\\", "\\")
    return decoded


# ============================================================
# OCR API interaction
# ============================================================


def _pdf_page_to_base64(doc: fitz.Document, page_num: int, scale: float) -> str:
    """Render a PDF page to a base64 PNG data URL."""
    page = doc.load_page(page_num)
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    encoded = base64.b64encode(img_data).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _build_payload(image_data_url: str, config: DotsOcrConfig) -> dict[str, Any]:
    """Build the API request payload for DotsOCR."""
    return {
        "model": config.model_name,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "top_p": config.top_p,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                    {
                        "type": "text",
                        "text": DOTS_OCR_PROMPT,
                    },
                ],
            }
        ],
    }


def _normalize_message_content(content: Any) -> str:
    """Normalize API response content to string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _call_dots_ocr_api(image_data_url: str, config: DotsOcrConfig) -> str:
    """Call the DotsOCR API for a single page image.

    Returns the raw text content from the API response.
    Raises RuntimeError on failure.
    """
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    payload = _build_payload(image_data_url, config)

    last_error: Exception | None = None
    for attempt in range(1, config.max_retries + 1):
        try:
            resp = requests.post(
                config.endpoint,
                json=payload,
                headers=headers,
                timeout=config.timeout,
            )

            if resp.status_code != 200:
                raise RuntimeError(
                    f"DotsOCR API returned status {resp.status_code}: {resp.text[:500]}"
                )

            data = resp.json()

            if "error" in data and data["error"].get("message"):
                raise RuntimeError(f"DotsOCR API error: {data['error']['message']}")

            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError("DotsOCR API returned no choices")

            content = _normalize_message_content(
                choices[0].get("message", {}).get("content")
            )
            if not content:
                finish_reason = choices[0].get("finish_reason", "")
                raise RuntimeError(
                    f"DotsOCR API returned empty content (finish_reason={finish_reason})"
                )

            return content

        except Exception as exc:
            last_error = exc
            if attempt < config.max_retries:
                logger.warning(
                    "DotsOCR request failed (attempt %s/%s): %s, retrying in %ss...",
                    attempt,
                    config.max_retries,
                    exc,
                    config.retry_delay,
                )
                time.sleep(config.retry_delay)

    raise RuntimeError(
        f"DotsOCR request failed after {config.max_retries} attempts: {last_error}"
    )


# ============================================================
# Page-level processing
# ============================================================


def _parse_page_elements(
    raw_content: str, page_number: int
) -> tuple[list[LayoutBlock], str]:
    """Parse DotsOCR raw response into structured layout blocks and cleaned text.

    Args:
        raw_content: The raw API response text.
        page_number: 1-based page number.

    Returns:
        A tuple of (layout blocks, cleaned markdown text).
    """
    raw_elements = _extract_elements_from_raw(raw_content)
    blocks: list[LayoutBlock] = []
    text_parts: list[str] = []

    for order, element in enumerate(raw_elements, start=1):
        if not isinstance(element, dict):
            continue

        category_raw = str(
            element.get("category") or element.get("type") or ""
        ).strip()
        category = LayoutCategory.from_raw(category_raw)
        bbox = BBox.from_sequence(element.get("bbox"))

        # Skip picture text (as per DotsOCR prompt rules)
        if category == LayoutCategory.PICTURE:
            text = None
        else:
            raw_text = element.get("text")
            text = str(raw_text) if raw_text is not None else None

        block = LayoutBlock(
            page_number=page_number,
            order_in_page=order,
            category=category,
            category_raw=category_raw,
            bbox=bbox,
            text=text,
        )
        blocks.append(block)

        # Build cleaned text from structured elements
        if category == LayoutCategory.PICTURE:
            # Omit pictures from text output
            continue
        elif category == LayoutCategory.FORMULA:
            formula_text = (text or "").strip()
            if formula_text:
                text_parts.append(f"\n$$\n{formula_text}\n$$\n")
        elif category in {LayoutCategory.TITLE, LayoutCategory.SECTION_HEADER}:
            heading_text = (text or "").strip()
            if heading_text:
                text_parts.append(f"\n## {heading_text}\n")
        elif text and text.strip():
            text_parts.append(text.strip())

    cleaned = "\n\n".join(text_parts)
    return blocks, cleaned


def _process_single_page(
    page_num: int,
    doc: fitz.Document,
    config: DotsOcrConfig,
) -> StructuredPageResult:
    """Process a single PDF page through DotsOCR."""
    logger.debug("Processing page %s via DotsOCR...", page_num + 1)
    image_data_url = _pdf_page_to_base64(doc, page_num, config.page_scale)
    raw_content = _call_dots_ocr_api(image_data_url, config)
    elements, cleaned_text = _parse_page_elements(raw_content, page_num + 1)

    logger.debug(
        "Page %s done: %d elements, %d chars",
        page_num + 1,
        len(elements),
        len(cleaned_text),
    )

    return StructuredPageResult(
        page_number=page_num,
        raw_content=raw_content,
        elements=elements,
        cleaned_text=cleaned_text,
    )


# ============================================================
# Public API
# ============================================================


def parse_pdf_with_dots_ocr(
    pdf_path: str,
    config: DotsOcrConfig | None = None,
) -> tuple[list[Document], dict[str, Any]]:
    """Parse a PDF file using DotsOCR, returning structured Documents and layout data.

    Args:
        pdf_path: Path to the PDF file.
        config: DotsOCR configuration. If None, reads from environment.

    Returns:
        A tuple of:
        - List of LangChain Document objects (one per page), with metadata
          containing structured layout information (category, bbox, section
          headers, etc.).
        - A dict containing the full structured layout data for downstream use.

    Raises:
        RuntimeError: If DotsOCR fails to parse any page.
        ValueError: If DotsOCR is not configured.
    """
    cfg = config or DotsOcrConfig()
    if not cfg.is_configured:
        raise ValueError(
            "DotsOCR is not configured. Set CYBER_RAG_DOTS_OCR_* or "
            "CYBER_RAG_ONEAPI_* environment variables."
        )

    doc = fitz.open(pdf_path)
    page_count = len(doc)
    logger.info("DotsOCR: parsing %s (%d pages)", pdf_path, page_count)

    page_results: dict[int, StructuredPageResult] = {}

    try:
        if page_count == 1:
            result = _process_single_page(0, doc, cfg)
            page_results[result.page_number] = result
        else:
            with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
                futures = {
                    executor.submit(_process_single_page, pn, doc, cfg): pn
                    for pn in range(page_count)
                }
                for future in as_completed(futures):
                    pn = futures[future]
                    try:
                        result = future.result()
                        page_results[result.page_number] = result
                    except Exception as exc:
                        logger.error("Page %s DotsOCR failed: %s", pn + 1, exc)
    finally:
        doc.close()

    if not page_results:
        raise RuntimeError("DotsOCR failed to parse any page")

    # Convert to LangChain Documents with rich metadata
    documents: list[Document] = []
    all_blocks: list[dict[str, Any]] = []
    current_section: str | None = None

    for page_num in sorted(page_results.keys()):
        result = page_results[page_num]

        # Track section headers for downstream chunking hints
        page_blocks_dicts: list[dict[str, Any]] = []
        for block in result.elements:
            block_dict = block.to_dict()
            block_dict["source"] = pdf_path
            page_blocks_dicts.append(block_dict)
            all_blocks.append(block_dict)

            # Update current section from section headers / titles
            if block.category in (
                LayoutCategory.SECTION_HEADER,
                LayoutCategory.TITLE,
            ) and block.text:
                current_section = block.text.strip()

        metadata: dict[str, Any] = {
            "source": pdf_path,
            "source_path": pdf_path,
            "page": page_num + 1,
            "total_pages": page_count,
            "parser": "dots_ocr",
            "section": current_section,
            "layout_blocks": page_blocks_dicts,
            "num_elements": len(result.elements),
        }

        documents.append(
            Document(page_content=result.cleaned_text, metadata=metadata)
        )

    # Build full structured output for downstream use
    structured_data: dict[str, Any] = {
        "source": pdf_path,
        "total_pages": page_count,
        "pages_parsed": len(page_results),
        "parser": "dots_ocr",
        "blocks": all_blocks,
    }

    logger.info(
        "DotsOCR complete: %d pages, %d total blocks, %d chars",
        len(page_results),
        len(all_blocks),
        sum(len(r.cleaned_text) for r in page_results.values()),
    )

    return documents, structured_data

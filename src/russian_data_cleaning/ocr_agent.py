from __future__ import annotations

import os
import re
import statistics
import subprocess
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from html import unescape
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import fitz

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None

from .page_commander import CommanderConfig, PageCommander
from .state_models import PageProcessingState, PageState
from .state_machine import transition
from .vendor_clients import qwen_ocr_via_dashscope


WHITESPACE_RE = re.compile(r"\s+")
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
SUSPICIOUS_MOJIBAKE_RE = re.compile(r"[\u0180-\u024F\u0250-\u02AF]")
MIXED_SCRIPT_TOKEN_RE = re.compile(r"\b(?=\w*[А-Яа-яЁё])(?=\w*[A-Za-z])[A-Za-zА-Яа-яЁё]{4,}\b")
CYRILLIC_INTRAWORD_PUNCT_RE = re.compile(r"\b[А-Яа-яЁё]{2,}[)!(/]{1}[А-Яа-яЁё]{2,}\b")
CYRILLIC_WITH_LATIN_FRAGMENT_RE = re.compile(r"\b[А-Яа-яЁё]{2,}[A-Za-z]{1,3}[А-Яа-яЁё]{2,}\b")
EXTRACT_ARTIFACT_BANNER_RE = re.compile(r"(?:[~_]{3,}|-{8,}|~\{4,}|[A-Za-zА-Яа-яЁё]~~)")
PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
FOOTNOTE_START_RE = re.compile(r"^\s*(?:\(?\d{1,3}\)|\[\d{1,3}\]|[*†‡])\s*")
NUMBERED_REFERENCE_LINE_RE = re.compile(r"^\s*\d{1,3}[.)]?\s+")
REFERENCE_LINE_CUE_RE = re.compile(
    r"(?i)(?:https?://|www\.|//|ibid\.|op\. cit\.|цит\. по:|там же|см\.:|см\.|references?|bibliograph|works cited|doi:|ргали\.|ф\.\s*\d+|оп\.\s*\d+|ед\.\s*хр\.|л\.\s*\d+)"
)
HEADING_START_RE = re.compile(r"(?i)^(?:глава|chapter|часть|part|section|раздел|статья|article)\b")
TOC_HEADING_RE = re.compile(r"(?mi)^\s*(?:содержание|оглавление|contents|table of contents|index)\s*$")
TOC_ENTRY_RE = re.compile(r"(?m)^(?:[A-ZА-ЯЁ0-9][^\n]{3,140}?)(?:\.{2,}|\s{2,}|\s)\d{1,4}\s*$")
PUBLISHER_META_RE = re.compile(
    r"(?i)\b(?:издательство|copyright|all rights reserved|published by|library of congress|isbn|научное издание|подписано в печать|тираж\b|cover design|typeset by|printed in)\b"
)
GLOSSARY_RE = re.compile(r"(?mi)^\s*(?:список сокращений|список терминов(?: и сокращений)?|глоссарий|glossary|abbreviations)\s*$")
LOWER_START_RE = re.compile(r"^[a-zа-яё]")
BODYISH_LINE_RE = re.compile(r"^(?:[-*•]\s+|\(?\d+[.)]?\s+|[a-zа-яё])")


@dataclass
class OCRAgentConfig:
    backend: str = "auto"
    language: str = "rus+eng"
    tesseract_psm: int = 3
    render_scale: float = 2.0
    extract_char_threshold: int = 120
    gemini_model: str = "gemini-2.5-flash"
    qwen_model: str = "qwen-vl-ocr-latest"
    skip_ocr_pages: bool = False
    adaptive_skip_low_yield_ocr: bool = True
    adaptive_ocr_min_attempts: int = 20
    adaptive_ocr_min_success_ratio: float = 0.10
    adaptive_ocr_empty_streak_limit: int = 12
    force_ocr_body_pages: bool = False


@dataclass
class OCRPageResult:
    page_number: int
    page_index: int
    route_hint: str
    source: str
    selected_text: str
    body_text: str
    notes_text: str
    reference_text: str
    page_type: str
    layout_status: str
    extracted_text: str
    ocr_text: str
    extracted_char_count: int
    ocr_char_count: int
    width: float
    height: float
    sanitized_image_path: str | None
    layout_sanitize_backend: str | None
    blocks: list[dict[str, Any]]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OCRDocumentResult:
    relative_path: str
    page_count: int
    route_hint: str
    backend: str
    pages: list[OCRPageResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "page_count": self.page_count,
            "route_hint": self.route_hint,
            "backend": self.backend,
            "pages": [page.to_dict() for page in self.pages],
        }


class OCRAgent:
    def __init__(self, config: OCRAgentConfig | None = None) -> None:
        self.config = config or OCRAgentConfig()
        self.commander = PageCommander(
            CommanderConfig(
                ocr_base_render_scale=self.config.render_scale,
                ocr_high_render_scale=max(self.config.render_scale, 2.6),
            )
        )

    def run(
        self,
        page_state: PageState,
        *,
        page_result: OCRPageResult | dict[str, Any] | None = None,
        document_path: str | Path | None = None,
        route_hint: str = "auto",
    ) -> PageState:
        if page_result is None:
            if document_path is None:
                raise ValueError("document_path is required when OCRAgent.run has no page_result.")
            document = self.process_document(document_path, pages=[page_state.page_num], route_hint=route_hint)
            if not document.get("pages"):
                raise ValueError(f"No OCR page returned for page {page_state.page_num}.")
            page_payload = dict(document["pages"][0])
        else:
            page_payload = page_result.to_dict() if hasattr(page_result, "to_dict") else dict(page_result)

        raw_text = (page_payload.get("body_text") or page_payload.get("selected_text") or "").strip()
        page_state.raw_text = raw_text
        page_state.layout_blocks = list(page_payload.get("blocks", []))
        page_state.page_type = page_payload.get("page_type")
        page_state.ocr_mode = str(page_payload.get("source") or "")
        page_state.stage_payloads["ocr"] = page_payload
        page_state.record_provenance(
            agent="OCRAgent",
            input_fields=["source_path", "page_num"],
            output_fields=["raw_text", "layout_blocks", "page_type", "ocr_mode", "stage_payloads.ocr"],
            note=f"source={page_state.ocr_mode}",
        )

        source = str(page_payload.get("source") or "")
        target_state = (
            PageProcessingState.OCR_DONE
            if source in {"ocr", "extract_fallback"}
            else PageProcessingState.EXTRACTED
        )
        transition(page_state, target_state, agent="OCRAgent", note=f"source={source}")
        return page_state

    def get_page_numbers(self, document_path: str | Path) -> list[int]:
        document_path = Path(document_path)
        suffix = document_path.suffix.lower()
        if suffix == ".epub":
            items = self._extract_epub_spine_items(document_path)
            return list(range(1, len(items) + 1))
        if suffix == ".pdf":
            with fitz.open(document_path) as doc:
                return list(range(1, len(doc) + 1))
        raise ValueError(f"Unsupported document type: {document_path.suffix}")

    def iterate_document_pages(
        self,
        document_path: str | Path,
        pages: list[int] | None = None,
        route_hint: str = "auto",
        sanitized_page_map: dict[int, str] | None = None,
        sanitized_layout_map: dict[int, dict[str, Any]] | None = None,
    ):
        document_path = Path(document_path)
        suffix = document_path.suffix.lower()
        if suffix == ".epub":
            yield from self._iterate_epub_pages(document_path, pages=pages, route_hint=route_hint)
            return
        if suffix == ".pdf":
            yield from self._iterate_pdf_pages(
                document_path,
                pages=pages,
                route_hint=route_hint,
                sanitized_page_map=sanitized_page_map,
                sanitized_layout_map=sanitized_layout_map,
            )
            return
        raise ValueError(f"Unsupported document type: {document_path.suffix}")

    def process_document(
        self,
        document_path: str | Path,
        pages: list[int] | None = None,
        route_hint: str = "auto",
        sanitized_page_map: dict[int, str] | None = None,
        sanitized_layout_map: dict[int, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        document_path = Path(document_path)
        suffix = document_path.suffix.lower()
        if suffix == ".epub":
            return self.process_epub(document_path, pages=pages, route_hint=route_hint)
        if suffix == ".pdf":
            return self.process_pdf(
                document_path,
                pages=pages,
                route_hint=route_hint,
                sanitized_page_map=sanitized_page_map,
                sanitized_layout_map=sanitized_layout_map,
            )
        raise ValueError(f"Unsupported document type: {document_path.suffix}")

    def process_pdf(
        self,
        pdf_path: str | Path,
        pages: list[int] | None = None,
        route_hint: str = "auto",
        sanitized_page_map: dict[int, str] | None = None,
        sanitized_layout_map: dict[int, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        pdf_path = Path(pdf_path)
        with fitz.open(pdf_path) as doc:
            page_count = len(doc)
        page_results = list(
            self._iterate_pdf_pages(
                pdf_path,
                pages=pages,
                route_hint=route_hint,
                sanitized_page_map=sanitized_page_map,
                sanitized_layout_map=sanitized_layout_map,
            )
        )

        result = OCRDocumentResult(
            relative_path=pdf_path.as_posix(),
            page_count=page_count,
            route_hint=route_hint,
            backend=self.config.backend,
            pages=page_results,
        )
        return result.to_dict()

    def process_epub(
        self,
        epub_path: str | Path,
        pages: list[int] | None = None,
        route_hint: str = "epub_extract_then_clean",
    ) -> dict[str, Any]:
        epub_path = Path(epub_path)
        spine_items = self._extract_epub_spine_items(epub_path)
        page_results = list(self._iterate_epub_pages(epub_path, pages=pages, route_hint=route_hint, spine_items=spine_items))

        result = OCRDocumentResult(
            relative_path=epub_path.as_posix(),
            page_count=len(spine_items),
            route_hint=route_hint,
            backend="epub_extract",
            pages=page_results,
        )
        return result.to_dict()

    def _iterate_pdf_pages(
        self,
        pdf_path: Path,
        pages: list[int] | None = None,
        route_hint: str = "auto",
        sanitized_page_map: dict[int, str] | None = None,
        sanitized_layout_map: dict[int, dict[str, Any]] | None = None,
    ):
        with fitz.open(pdf_path) as doc:
            page_numbers = self._normalize_pages(len(doc), pages)
            ocr_attempts = 0
            ocr_successes = 0
            empty_ocr_streak = 0
            adaptive_ocr_disabled = False
            for page_number in page_numbers:
                page = doc[page_number - 1]
                page_result = self._process_pdf_page(
                    page,
                    page_number=page_number,
                    route_hint=route_hint,
                    force_skip_ocr=adaptive_ocr_disabled,
                    sanitized_image_path=sanitized_page_map.get(page_number) if sanitized_page_map else None,
                    layout_sanitize_backend="paddle" if sanitized_page_map and page_number in sanitized_page_map else None,
                    sanitized_layout_payload=sanitized_layout_map.get(page_number) if sanitized_layout_map else None,
                )
                ocr_attempted = page_result.source in {"ocr", "extract_fallback"}
                if ocr_attempted:
                    ocr_attempts += 1
                    if page_result.ocr_char_count > 0:
                        ocr_successes += 1
                        empty_ocr_streak = 0
                    else:
                        empty_ocr_streak += 1

                    if (
                        self.config.adaptive_skip_low_yield_ocr
                        and not self.config.force_ocr_body_pages
                        and not adaptive_ocr_disabled
                        and route_hint != "pdf_ocr_then_clean"
                        and self.config.backend in {"qwen", "gemini", "google_documentai", "auto"}
                    ):
                        success_ratio = (ocr_successes / ocr_attempts) if ocr_attempts else 0.0
                        if (
                            ocr_attempts >= self.config.adaptive_ocr_min_attempts
                            and (
                                success_ratio < self.config.adaptive_ocr_min_success_ratio
                                or empty_ocr_streak >= self.config.adaptive_ocr_empty_streak_limit
                            )
                        ):
                            adaptive_ocr_disabled = True
                            page_result.notes.append(
                                f"adaptive_ocr_disable_triggered:attempts={ocr_attempts},successes={ocr_successes},streak={empty_ocr_streak}"
                            )
                yield page_result

    def _process_pdf_page(
        self,
        page: fitz.Page,
        page_number: int,
        route_hint: str,
        force_skip_ocr: bool = False,
        sanitized_image_path: str | Path | None = None,
        layout_sanitize_backend: str | None = None,
        sanitized_layout_payload: dict[str, Any] | None = None,
    ) -> OCRPageResult:
        extracted_text = self._normalize_text(page.get_text("text"))
        extracted_blocks = self._extract_pdf_text_blocks(page)
        notes: list[str] = []
        if sanitized_layout_payload is not None:
            filtered_text, filtered_blocks, filter_notes = self._build_sanitized_extract_view(
                extracted_blocks=extracted_blocks,
                page=page,
                layout_payload=sanitized_layout_payload,
            )
            notes.extend(filter_notes)
            if filtered_text.strip():
                extracted_text = filtered_text
                extracted_blocks = filtered_blocks
            else:
                notes.append("layout_extract_filter_empty_used_raw_extract")
        extracted_char_count = self._char_count(extracted_text)
        if self._looks_mojibake(extracted_text):
            notes.append("extract_text_looks_mojibake")
        sanitizer_enabled = sanitized_image_path is not None
        if sanitizer_enabled:
            notes.append(f"layout_sanitized={layout_sanitize_backend or 'external'}")
        pre_ocr_skip_reason = None if sanitizer_enabled else self._preclassify_skip_reason(extracted_text)
        ocr_plan = self.commander.plan_ocr_page(
            route_hint=route_hint,
            extracted_text=extracted_text,
            extracted_char_count=extracted_char_count,
            pre_ocr_skip_reason=pre_ocr_skip_reason,
            looks_mojibake=self._looks_mojibake(extracted_text),
            looks_low_quality_extract=self._looks_extract_low_quality(extracted_text),
            extracted_blocks=extracted_blocks,
            backend=self.config.backend,
        )
        bypass_sanitized_ocr = sanitizer_enabled and ocr_plan.source == "extract"
        if sanitizer_enabled:
            source = "extract" if bypass_sanitized_ocr else "ocr"
            notes.append(f"commander_ocr=sanitized:{ocr_plan.difficulty}:{ocr_plan.reason}")
            if bypass_sanitized_ocr:
                notes.append("sanitized_extract_bypass_used")
        else:
            source = self._select_source(
                route_hint,
                extracted_text,
                extracted_char_count,
                pre_ocr_skip_reason,
                commander_plan=ocr_plan,
            )
            notes.append(f"commander_ocr={ocr_plan.source}:{ocr_plan.difficulty}:{ocr_plan.reason}")

        ocr_text = ""
        ocr_char_count = 0

        if source == "ocr_skipped_nonbody":
            selected_text = extracted_text
            notes.append(f"ocr_skipped_{pre_ocr_skip_reason}")
        elif source == "extract":
            selected_text = extracted_text
            if not selected_text:
                notes.append("empty_extract")
        else:
            if self.config.skip_ocr_pages or force_skip_ocr:
                selected_text = ""
                source = "ocr_skipped"
                if force_skip_ocr and extracted_text:
                    selected_text = extracted_text
                    source = "ocr_disabled_low_yield"
                    notes.append("ocr_disabled_due_low_yield_used_extract")
                else:
                    notes.append("ocr_skipped_by_policy")
            else:
                ocr_text = self._run_ocr(page, render_scale=ocr_plan.render_scale, image_path=sanitized_image_path)
                ocr_char_count = self._char_count(ocr_text)
                selected_text = ocr_text
                if source == "ocr" and not ocr_text and extracted_text:
                    selected_text = extracted_text
                    source = "extract_fallback"
                    if sanitizer_enabled:
                        notes.append("sanitized_ocr_empty_used_extract_fallback")
                    else:
                        notes.append("ocr_empty_used_extract_fallback")
                elif source == "ocr" and not ocr_text:
                    notes.append("ocr_empty")

        if extracted_char_count >= self.config.extract_char_threshold and source.startswith("ocr"):
            notes.append("extract_text_available")
        if extracted_char_count < self.config.extract_char_threshold and source == "extract":
            notes.append("extract_text_sparse")
        if sanitizer_enabled and not bypass_sanitized_ocr:
            layout = self._segment_text_only_layout(selected_text)
        else:
            layout = self._segment_page_layout(page, selected_text, precomputed_blocks=extracted_blocks)
        if layout["layout_status"] != "text_fallback":
            notes.append(f"layout_{layout['layout_status']}")
        if layout["page_type"] != "body_only":
            notes.append(f"page_type={layout['page_type']}")

        return OCRPageResult(
            page_number=page_number,
            page_index=page_number - 1,
            route_hint=route_hint,
            source=source,
            selected_text=selected_text,
            body_text=layout["body_text"],
            notes_text=layout["notes_text"],
            reference_text=layout["reference_text"],
            page_type=layout["page_type"],
            layout_status=layout["layout_status"],
            extracted_text=extracted_text,
            ocr_text=ocr_text,
            extracted_char_count=extracted_char_count,
            ocr_char_count=ocr_char_count,
            width=page.rect.width,
            height=page.rect.height,
            sanitized_image_path=str(sanitized_image_path) if sanitized_image_path is not None else None,
            layout_sanitize_backend=layout_sanitize_backend,
            blocks=layout["blocks"],
            notes=notes,
        )

    def _build_sanitized_extract_view(
        self,
        *,
        extracted_blocks: list[dict[str, Any]],
        page: fitz.Page,
        layout_payload: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]], list[str]]:
        regions = layout_payload.get("regions", []) if isinstance(layout_payload, dict) else []
        if not regions:
            return "", extracted_blocks, []

        page_width = float(layout_payload.get("width") or 0.0)
        page_height = float(layout_payload.get("height") or 0.0)
        if page_width <= 0 or page_height <= 0 or page.rect.width <= 0 or page.rect.height <= 0:
            return "", extracted_blocks, ["layout_extract_filter_missing_dimensions"]

        scale_x = page_width / float(page.rect.width)
        scale_y = page_height / float(page.rect.height)
        keep_regions = [
            self._normalize_bbox(region.get("bbox"))
            for region in regions
            if region.get("action") == "keep"
        ]
        mask_regions = [
            self._normalize_bbox(region.get("bbox"))
            for region in regions
            if region.get("action") == "mask"
        ]
        if not keep_regions and not mask_regions:
            return "", extracted_blocks, []

        filtered_blocks: list[dict[str, Any]] = []
        removed = 0
        for block in extracted_blocks:
            scaled_bbox = self._scale_block_bbox(block["bbox"], scale_x=scale_x, scale_y=scale_y)
            keep_ratio = self._max_overlap_ratio(scaled_bbox, keep_regions)
            mask_ratio = self._max_overlap_ratio(scaled_bbox, mask_regions)
            center_in_keep = self._center_in_any_bbox(scaled_bbox, keep_regions)
            center_in_mask = self._center_in_any_bbox(scaled_bbox, mask_regions)

            keep_block = False
            if center_in_keep or keep_ratio >= 0.25:
                keep_block = True
            elif center_in_mask and keep_ratio < 0.05:
                keep_block = False
            elif mask_ratio >= 0.72 and keep_ratio < 0.10:
                keep_block = False
            else:
                keep_block = True

            if keep_block:
                filtered_blocks.append(block)
            else:
                removed += 1

        filtered_text = self._normalize_text("\n".join((block.get("text") or "").strip() for block in filtered_blocks if (block.get("text") or "").strip()))
        notes = [f"layout_extract_filter_applied:kept={len(filtered_blocks)},removed={removed}"]
        return filtered_text, filtered_blocks, notes

    @staticmethod
    def _normalize_bbox(bbox: Any) -> list[float]:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return [0.0, 0.0, 0.0, 0.0]
        return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]

    @staticmethod
    def _scale_block_bbox(bbox: list[float], *, scale_x: float, scale_y: float) -> list[float]:
        return [
            float(bbox[0]) * scale_x,
            float(bbox[1]) * scale_y,
            float(bbox[2]) * scale_x,
            float(bbox[3]) * scale_y,
        ]

    @staticmethod
    def _max_overlap_ratio(bbox: list[float], candidates: list[list[float]]) -> float:
        if not candidates:
            return 0.0
        x1, y1, x2, y2 = bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area <= 0:
            return 0.0
        best = 0.0
        for candidate in candidates:
            cx1, cy1, cx2, cy2 = candidate
            ix1 = max(x1, cx1)
            iy1 = max(y1, cy1)
            ix2 = min(x2, cx2)
            iy2 = min(y2, cy2)
            intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if intersection <= 0:
                continue
            best = max(best, intersection / area)
        return best

    @staticmethod
    def _center_in_any_bbox(bbox: list[float], candidates: list[list[float]]) -> bool:
        if not candidates:
            return False
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        for x1, y1, x2, y2 in candidates:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return True
        return False

    def _iterate_epub_pages(
        self,
        epub_path: Path,
        pages: list[int] | None = None,
        route_hint: str = "epub_extract_then_clean",
        spine_items: list[dict[str, Any]] | None = None,
    ):
        items = spine_items or self._extract_epub_spine_items(epub_path)
        page_numbers = self._normalize_pages(len(items), pages)
        for page_number in page_numbers:
            text = self._normalize_text(items[page_number - 1]["text"])
            notes: list[str] = []
            pre_ocr_skip_reason = self._preclassify_skip_reason(text)
            source = "epub_extract"
            if pre_ocr_skip_reason:
                source = "epub_skip_nonbody"
                notes.append(f"epub_skipped_{pre_ocr_skip_reason}")

            layout = self._segment_text_only_layout(text)
            if layout["layout_status"] != "text_fallback":
                notes.append(f"layout_{layout['layout_status']}")
            if layout["page_type"] != "body_only":
                notes.append(f"page_type={layout['page_type']}")

            yield OCRPageResult(
                page_number=page_number,
                page_index=page_number - 1,
                route_hint=route_hint,
                source=source,
                selected_text=text,
                body_text=layout["body_text"],
                notes_text=layout["notes_text"],
                reference_text=layout["reference_text"],
                page_type=layout["page_type"],
                layout_status=layout["layout_status"],
                extracted_text=text,
                ocr_text="",
                extracted_char_count=self._char_count(text),
                ocr_char_count=0,
                width=0.0,
                height=0.0,
                blocks=layout["blocks"],
                notes=notes,
            )

    def _normalize_pages(self, page_count: int, pages: list[int] | None) -> list[int]:
        if pages is None:
            return list(range(1, page_count + 1))
        normalized = sorted({page for page in pages if 1 <= page <= page_count})
        return normalized

    def _select_source(
        self,
        route_hint: str,
        extracted_text: str,
        extracted_char_count: int,
        pre_ocr_skip_reason: str | None = None,
        commander_plan=None,
    ) -> str:
        if commander_plan is not None:
            return commander_plan.source
        looks_mojibake = self._looks_mojibake(extracted_text)
        looks_low_quality_extract = self._looks_extract_low_quality(extracted_text)
        if pre_ocr_skip_reason and extracted_text:
            return "ocr_skipped_nonbody"
        if self.config.force_ocr_body_pages and self.config.backend != "extract_only":
            return "ocr"
        if self.config.backend == "extract_only":
            return "extract"
        if self.config.backend == "tesseract":
            return "ocr"
        if self.config.backend in {"gemini", "google_documentai"}:
            return "ocr"
        if self.config.backend == "qwen":
            if route_hint == "pdf_ocr_then_clean":
                return "ocr"
            if looks_mojibake or looks_low_quality_extract:
                return "ocr"
            if route_hint == "pdf_mixed_extract_plus_ocr":
                return "extract" if extracted_char_count >= self.config.extract_char_threshold else "ocr"
            return "extract" if extracted_char_count >= self.config.extract_char_threshold else "ocr"
        if route_hint == "pdf_extract_then_clean":
            if looks_mojibake or looks_low_quality_extract:
                return "ocr"
            return "extract"
        if route_hint == "pdf_mixed_extract_plus_ocr":
            if looks_mojibake or looks_low_quality_extract:
                return "ocr"
            if extracted_char_count >= self.config.extract_char_threshold:
                return "extract"
            return "ocr"
        if route_hint == "pdf_ocr_then_clean":
            return "ocr"
        if looks_mojibake or looks_low_quality_extract:
            return "ocr"
        if extracted_char_count >= self.config.extract_char_threshold:
            return "extract"
        return "ocr"

    def _preclassify_skip_reason(self, extracted_text: str) -> str | None:
        if not extracted_text:
            return None
        text = extracted_text.strip()
        if self._looks_like_toc_page(text):
            return "toc_index_page"
        if self._looks_like_reference_only_page(text):
            return "reference_only_page"
        if self._looks_like_glossary_page(text):
            return "glossary_page"
        if self._looks_like_publisher_meta_page(text):
            compact = WHITESPACE_RE.sub("", text)
            nonempty_lines = [line for line in text.splitlines() if line.strip()]
            if len(compact) <= 600 and len(nonempty_lines) <= 25:
                return "publisher_meta_page"
        return None

    def _run_ocr(self, page: fitz.Page, render_scale: float | None = None, image_path: str | Path | None = None) -> str:
        if self.config.backend == "gemini":
            return self._run_gemini_ocr(page, render_scale=render_scale, image_path=image_path)
        if self.config.backend == "qwen":
            return self._run_qwen_ocr(page, render_scale=render_scale, image_path=image_path)
        if self.config.backend.startswith("google"):
            return self._run_google_ocr(page, render_scale=render_scale, image_path=image_path)
        return self._run_tesseract(page, render_scale=render_scale, image_path=image_path)

    def _run_qwen_ocr(self, page: fitz.Page, render_scale: float | None = None, image_path: str | Path | None = None) -> str:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("Qwen OCR backend selected but DASHSCOPE_API_KEY is not set.")
        png_bytes = self._load_png_bytes(page, render_scale=render_scale, image_path=image_path)
        try:
            text = qwen_ocr_via_dashscope(
                api_key=api_key,
                png_bytes=png_bytes,
                model=self.config.qwen_model,
                task="multi_lan",
            )
            return self._normalize_text(text)
        except Exception:
            return ""

    def _run_google_ocr(self, page: fitz.Page, render_scale: float | None = None, image_path: str | Path | None = None) -> str:
        if not (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_API_KEY")):
            raise RuntimeError(
                "Google OCR backend selected but no GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_API_KEY is set."
            )
        raise NotImplementedError(
            "Google OCR backend is a placeholder in the terminal-first version. "
            "Set credentials later and wire your preferred Google API client here."
        )

    def _run_gemini_ocr(self, page: fitz.Page, render_scale: float | None = None, image_path: str | Path | None = None) -> str:
        client = self._get_gemini_client()
        png_bytes = self._load_png_bytes(page, render_scale=render_scale, image_path=image_path)
        prompt = (
            "You are an OCR transcriber and layout reader for Russian and mixed-language PDF pages. "
            "Extract the visible text from the page image. Preserve paragraph and line breaks where possible. "
            "Pay attention to the difference between main body text and bottom-of-page notes or references. "
            "Do not merge footnotes, note blocks, or bibliography tails into the main body if they are visually separated. "
            "Keep the original wording, ordering, and numbering. "
            "Do not summarize, correct facts, or add commentary. "
            "Return only the page text."
        )
        try:
            response = client.models.generate_content(
                model=self.config.gemini_model,
                contents=[
                    prompt,
                    genai_types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                ],
            )
            return self._normalize_text(self._extract_response_text(response))
        except Exception:
            return ""

    def _run_tesseract(
        self,
        page: fitz.Page,
        render_scale: float | None = None,
        image_path: str | Path | None = None,
    ) -> str:
        with tempfile.TemporaryDirectory(prefix="ocr-agent-") as tmp_dir:
            if image_path is not None:
                image_path = Path(image_path)
            else:
                image_path = Path(tmp_dir) / "page.png"
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(render_scale or self.config.render_scale, render_scale or self.config.render_scale),
                    alpha=False,
                )
                pixmap.save(str(image_path))
            command = [
                "tesseract",
                str(image_path),
                "stdout",
                "-l",
                self.config.language,
                "--psm",
                str(self.config.tesseract_psm),
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()
                raise RuntimeError(f"Tesseract failed: {stderr or 'unknown error'}")
            return self._normalize_text(result.stdout)

    def _load_png_bytes(
        self,
        page: fitz.Page,
        *,
        render_scale: float | None = None,
        image_path: str | Path | None = None,
    ) -> bytes:
        if image_path is not None:
            return Path(image_path).read_bytes()
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(render_scale or self.config.render_scale, render_scale or self.config.render_scale),
            alpha=False,
        )
        return pixmap.tobytes("png")

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _segment_page_layout(
        self,
        page: fitz.Page,
        selected_text: str,
        precomputed_blocks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        skip_reason = self._preclassify_skip_reason(selected_text)
        if skip_reason == "toc_index_page":
            return {
                "body_text": "",
                "notes_text": "",
                "reference_text": "",
                "page_type": "toc_or_index",
                "layout_status": "page_classifier",
                "blocks": [],
            }
        if skip_reason == "reference_only_page":
            return {
                "body_text": "",
                "notes_text": "",
                "reference_text": selected_text,
                "page_type": "reference_only",
                "layout_status": "page_classifier",
                "blocks": [],
            }
        if skip_reason == "glossary_page":
            return {
                "body_text": "",
                "notes_text": "",
                "reference_text": selected_text,
                "page_type": "glossary_page",
                "layout_status": "page_classifier",
                "blocks": [],
            }
        if skip_reason == "publisher_meta_page":
            return {
                "body_text": "",
                "notes_text": "",
                "reference_text": "",
                "page_type": "publisher_meta",
                "layout_status": "page_classifier",
                "blocks": [],
            }
        blocks = precomputed_blocks if precomputed_blocks is not None else self._extract_pdf_text_blocks(page)
        if not blocks:
            return {
                "body_text": selected_text,
                "notes_text": "",
                "reference_text": "",
                "page_type": "body_only" if selected_text else "empty",
                "layout_status": "text_fallback",
                "blocks": [],
            }

        classified = self._classify_blocks(blocks, page.rect.height)
        body_parts: list[str] = []
        notes_parts: list[str] = []
        reference_parts: list[str] = []

        for block in classified:
            text = (block.get("text") or "").strip()
            if not text:
                continue
            role = block["role"]
            if role in {"body", "heading"}:
                body_parts.append(text)
            elif role == "footnote_body":
                notes_parts.append(text)
            elif role == "reference_block":
                reference_parts.append(text)

        body_text = self._normalize_text("\n\n".join(body_parts))
        notes_text = self._normalize_text("\n\n".join(notes_parts))
        reference_text = self._normalize_text("\n\n".join(reference_parts))

        if not body_text and (notes_text or reference_text):
            page_type = "notes_only"
        elif body_text and (notes_text or reference_text):
            page_type = "body_with_notes"
        elif body_text:
            page_type = "body_only"
        else:
            page_type = "empty"

        return {
            "body_text": body_text or selected_text,
            "notes_text": notes_text,
            "reference_text": reference_text,
            "page_type": page_type,
            "layout_status": "pdf_blocks",
            "blocks": classified,
        }

    def _segment_text_only_layout(self, selected_text: str) -> dict[str, Any]:
        skip_reason = self._preclassify_skip_reason(selected_text)
        if skip_reason == "toc_index_page":
            return {
                "body_text": "",
                "notes_text": "",
                "reference_text": "",
                "page_type": "toc_or_index",
                "layout_status": "text_classifier",
                "blocks": [],
            }
        if skip_reason == "reference_only_page":
            return {
                "body_text": "",
                "notes_text": "",
                "reference_text": selected_text,
                "page_type": "reference_only",
                "layout_status": "text_classifier",
                "blocks": [],
            }
        if skip_reason == "glossary_page":
            return {
                "body_text": "",
                "notes_text": "",
                "reference_text": selected_text,
                "page_type": "glossary_page",
                "layout_status": "text_classifier",
                "blocks": [],
            }
        if skip_reason == "publisher_meta_page":
            return {
                "body_text": "",
                "notes_text": "",
                "reference_text": "",
                "page_type": "publisher_meta",
                "layout_status": "text_classifier",
                "blocks": [],
            }
        return {
            "body_text": selected_text,
            "notes_text": "",
            "reference_text": "",
            "page_type": "body_only" if selected_text else "empty",
            "layout_status": "text_fallback",
            "blocks": [],
        }

    def _extract_pdf_text_blocks(self, page: fitz.Page) -> list[dict[str, Any]]:
        page_dict = page.get_text("dict")
        results: list[dict[str, Any]] = []
        for block_index, block in enumerate(page_dict.get("blocks", []), start=1):
            if block.get("type") != 0:
                continue
            lines: list[str] = []
            font_sizes: list[float] = []
            for line in block.get("lines", []):
                parts: list[str] = []
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    if span_text:
                        parts.append(span_text)
                    size = span.get("size")
                    if isinstance(size, (int, float)):
                        font_sizes.append(float(size))
                line_text = "".join(parts).strip()
                if line_text:
                    lines.append(line_text)
            if not lines:
                continue
            text = self._normalize_text("\n".join(lines))
            bbox = [float(v) for v in block.get("bbox", [0, 0, 0, 0])]
            results.append(
                {
                    "id": f"b{block_index}",
                    "bbox": bbox,
                    "text": text,
                    "line_count": len(lines),
                    "char_count": self._char_count(text),
                    "avg_font_size": (sum(font_sizes) / len(font_sizes)) if font_sizes else 0.0,
                }
            )
        return results

    def _classify_blocks(self, blocks: list[dict[str, Any]], page_height: float) -> list[dict[str, Any]]:
        font_candidates = [block["avg_font_size"] for block in blocks if block["avg_font_size"] and block["char_count"] >= 40]
        body_font = statistics.median(font_candidates) if font_candidates else 0.0
        content_top = min(block["bbox"][1] for block in blocks)
        content_bottom = max(block["bbox"][3] for block in blocks)
        classified: list[dict[str, Any]] = []
        previous_bottom = 0.0

        for index, block in enumerate(blocks):
            role = self._classify_block(block, page_height, body_font, previous_bottom, content_top, content_bottom)
            previous_bottom = block["bbox"][3]
            enriched = dict(block)
            enriched["role"] = role
            classified.append(enriched)

        return classified

    def _classify_block(
        self,
        block: dict[str, Any],
        page_height: float,
        body_font: float,
        previous_bottom: float,
        content_top: float,
        content_bottom: float,
    ) -> str:
        text = (block.get("text") or "").strip()
        compact = re.sub(r"\s+", "", text)
        bbox = block["bbox"]
        y0, y1 = bbox[1], bbox[3]
        top_ratio_abs = y0 / page_height if page_height else 0.0
        bottom_ratio_abs = y1 / page_height if page_height else 0.0
        content_height = max(1.0, content_bottom - content_top)
        top_ratio = (y0 - content_top) / content_height
        bottom_ratio = (y1 - content_top) / content_height
        gap_above = max(0.0, y0 - previous_bottom)
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        if PAGE_NUMBER_RE.fullmatch(compact) and bottom_ratio >= 0.96:
            return "page_number"
        if self._looks_like_heading_block(text, block, body_font):
            return "heading"
        if (
            top_ratio_abs <= 0.10
            and len(text) <= 140
            and len(lines) <= 2
            and not self._looks_like_body_continuation(text)
        ):
            return "header"
        if (
            bottom_ratio_abs >= 0.90
            and len(text) <= 140
            and len(lines) <= 2
            and not self._looks_like_reference_block(text)
            and not self._looks_like_body_continuation(text)
        ):
            return "footer"

        note_score = 0
        if top_ratio >= 0.70:
            note_score += 2
        elif top_ratio >= 0.58:
            note_score += 1
        if bottom_ratio >= 0.88:
            note_score += 2
        elif bottom_ratio >= 0.78:
            note_score += 1

        avg_font_size = block.get("avg_font_size") or 0.0
        if body_font and avg_font_size and avg_font_size < body_font * 0.92:
            note_score += 1
        if gap_above >= max(10.0, body_font * 0.9 if body_font else 12.0) and top_ratio >= 0.55:
            note_score += 1
        if len(lines) >= 3 and top_ratio >= 0.58:
            note_score += 1
        if text.startswith("(") and len(lines) >= 2 and top_ratio >= 0.55:
            note_score += 2
        if block.get("char_count", 0) >= 180 and gap_above >= 18.0 and top_ratio >= 0.55:
            note_score += 1
        if any(FOOTNOTE_START_RE.match(line) for line in lines):
            note_score += 2
        if self._looks_like_reference_block(text):
            note_score += 2

        if note_score >= 4:
            return "reference_block" if self._looks_like_reference_block(text) else "footnote_body"

        return "body"

    def _looks_like_heading_block(self, text: str, block: dict[str, Any], body_font: float) -> bool:
        compact = re.sub(r"\s+", " ", text.strip())
        if not compact:
            return False
        if HEADING_START_RE.match(compact):
            return True
        letters = re.sub(r"[^A-Za-zА-Яа-яЁё]", "", compact)
        uppercase_letters = re.sub(r"[^A-ZА-ЯЁ]", "", compact)
        if letters and len(letters) >= 10 and len(uppercase_letters) >= int(len(letters) * 0.85) and len(compact) <= 180:
            return True
        avg_font_size = block.get("avg_font_size") or 0.0
        if body_font and avg_font_size >= body_font * 1.15 and block.get("line_count", 0) <= 4 and len(compact) <= 220:
            return True
        return False

    def _looks_like_reference_block(self, text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False
        numbered_lines = sum(1 for line in lines if NUMBERED_REFERENCE_LINE_RE.match(line))
        cue_lines = sum(1 for line in lines if REFERENCE_LINE_CUE_RE.search(line))
        year_lines = sum(1 for line in lines if re.search(r"(?:19|20)\d{2}", line))
        if numbered_lines >= 2:
            return True
        if cue_lines >= 2:
            return True
        if numbered_lines >= 1 and (cue_lines >= 1 or year_lines >= 2):
            return True
        return False

    def _looks_like_toc_page(self, text: str) -> bool:
        if not text:
            return False
        if TOC_HEADING_RE.search(text):
            return True
        entries = len(TOC_ENTRY_RE.findall(text))
        return entries >= 5

    def _looks_like_publisher_meta_page(self, text: str) -> bool:
        if not text:
            return False
        return bool(PUBLISHER_META_RE.search(text))

    def _looks_like_glossary_page(self, text: str) -> bool:
        if not text:
            return False
        if GLOSSARY_RE.search(text):
            return True
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        dash_entries = sum(1 for line in lines if re.match(r"^\s*[\w/+ -]{2,40}\s*[—-]\s+.+", line))
        return dash_entries >= 8

    def _looks_like_reference_only_page(self, text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 4:
            return False
        numbered_lines = sum(1 for line in lines if NUMBERED_REFERENCE_LINE_RE.match(line))
        cue_lines = sum(1 for line in lines if REFERENCE_LINE_CUE_RE.search(line))
        avg_line_len = sum(len(line) for line in lines) / max(1, len(lines))
        if numbered_lines >= max(3, len(lines) // 3):
            return True
        if cue_lines >= max(3, len(lines) // 3) and avg_line_len >= 55:
            return True
        return False

    def _looks_like_body_continuation(self, text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False
        first = lines[0]
        compact = re.sub(r"\s+", " ", first)
        if len(compact) < 12:
            return False
        if LOWER_START_RE.match(compact):
            return True
        if BODYISH_LINE_RE.match(compact) and any(char in compact for char in ",;:"):
            return True
        alpha_chars = re.sub(r"[^A-Za-zА-Яа-яЁё]", "", compact)
        if alpha_chars:
            upper_chars = re.sub(r"[^A-ZА-ЯЁ]", "", compact)
            if len(upper_chars) <= int(len(alpha_chars) * 0.45) and compact[-1:] not in {".", "!", "?"}:
                return True
        return False

    def _find_epub_package(self, zf: zipfile.ZipFile) -> str | None:
        try:
            container = zf.read("META-INF/container.xml")
        except KeyError:
            return None
        root = ElementTree.fromstring(container)
        ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
        for rootfile in root.findall(".//c:rootfile", ns):
            full_path = rootfile.attrib.get("full-path")
            if full_path:
                return full_path
        return None

    def _extract_epub_spine_items(self, epub_path: Path) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        with zipfile.ZipFile(epub_path) as zf:
            package_path = self._find_epub_package(zf)
            if not package_path:
                raise RuntimeError("EPUB missing META-INF/container.xml or OPF package.")

            package_dir = Path(package_path).parent
            package = ElementTree.fromstring(zf.read(package_path))
            ns = {"opf": "http://www.idpf.org/2007/opf"}
            manifest_items = {
                item.attrib.get("id"): item.attrib.get("href", "")
                for item in package.findall(".//opf:manifest/opf:item", ns)
            }
            spine_ids = [
                item.attrib.get("idref", "")
                for item in package.findall(".//opf:spine/opf:itemref", ns)
            ]

            for item_id in spine_ids:
                href = manifest_items.get(item_id)
                if not href:
                    continue
                item_path = str((package_dir / href).as_posix())
                try:
                    raw = zf.read(item_path).decode("utf-8", errors="ignore")
                except KeyError:
                    continue
                text = self._html_to_text(raw)
                text = self._normalize_text(text)
                if text:
                    items.append({"id": item_id, "href": item_path, "text": text})
        return items

    def _html_to_text(self, raw_html: str) -> str:
        text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", raw_html)
        text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
        text = re.sub(r"(?i)<br\\s*/?>", "\n", text)
        text = re.sub(r"(?i)</p\\s*>", "\n\n", text)
        text = re.sub(r"(?i)</div\\s*>", "\n", text)
        text = re.sub(r"(?i)</h[1-6]\\s*>", "\n\n", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = unescape(text)
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _char_count(self, text: str) -> int:
        return len(WHITESPACE_RE.sub("", text))

    def _looks_mojibake(self, text: str) -> bool:
        if not text:
            return False
        compact = WHITESPACE_RE.sub("", text)
        if len(compact) < 80:
            return False
        suspicious = len(SUSPICIOUS_MOJIBAKE_RE.findall(compact))
        cyrillic = len(CYRILLIC_RE.findall(compact))
        if suspicious >= 12 and suspicious > cyrillic * 0.35:
            return True
        return self._has_localized_mojibake_line(text)

    def _looks_extract_low_quality(self, text: str) -> bool:
        if not text:
            return False
        compact = WHITESPACE_RE.sub("", text)
        if len(compact) < 120:
            return False

        cyrillic = len(CYRILLIC_RE.findall(compact))
        if cyrillic < 60:
            return False

        tokens = re.findall(r"\b\S+\b", text)
        token_count = max(1, len(tokens))
        mixed_hits = len(MIXED_SCRIPT_TOKEN_RE.findall(text))
        punct_hits = len(CYRILLIC_INTRAWORD_PUNCT_RE.findall(text))
        latin_fragment_hits = len(CYRILLIC_WITH_LATIN_FRAGMENT_RE.findall(text))
        artifact_hits = len(EXTRACT_ARTIFACT_BANNER_RE.findall(text))
        noisy_hits = mixed_hits + punct_hits + latin_fragment_hits

        if mixed_hits >= 4:
            return True
        if latin_fragment_hits >= 3 and punct_hits >= 1:
            return True
        if artifact_hits >= 2 and noisy_hits >= 2:
            return True
        if noisy_hits >= 8 and (noisy_hits / token_count) >= 0.035:
            return True
        return False

    def _has_localized_mojibake_line(self, text: str) -> bool:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if len(WHITESPACE_RE.sub("", line)) < 8:
                continue
            suspicious = len(SUSPICIOUS_MOJIBAKE_RE.findall(line))
            if suspicious < 6:
                continue
            cyrillic = len(CYRILLIC_RE.findall(line))
            latin = len(re.findall(r"[A-Za-z]", line))
            if suspicious >= 8 and latin <= 2:
                return True
            if cyrillic >= 4 and suspicious >= max(6, int(cyrillic * 0.45)):
                return True
        return False

    def _get_gemini_client(self):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Gemini backend selected but GOOGLE_API_KEY or GEMINI_API_KEY is not set.")
        if genai is None or genai_types is None:
            raise RuntimeError("google-genai package is not installed.")
        return genai.Client(api_key=api_key)

    def _extract_response_text(self, response: Any) -> str:
        text = getattr(response, "text", "") or ""
        text = text.strip()
        if text.startswith("```") and text.endswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return text.strip()

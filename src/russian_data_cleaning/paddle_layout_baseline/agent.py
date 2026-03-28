from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .document_io import iter_document_pages
from .export import build_final_text
from .layout import PaddleLayoutDetector, PaddleLayoutDetectorConfig
from .routing import PaddleRegionOCR, PaddleRegionOCRConfig, crop_region, map_and_route_blocks
from .sanitizer import build_sanitized_page
from .types import DocumentLayoutResult, LayoutRegion, PageLayoutResult


@dataclass
class PaddleLayoutOCRConfig:
    render_scale: float = 2.0
    ocr_lang: str = "ru"
    show_log: bool = False
    use_gpu: bool = False
    layout_score_threshold: float = 0.0
    perform_region_ocr: bool = False
    mask_fill: int = 255
    split_landscape_spreads: bool = True
    split_aspect_ratio_threshold: float = 1.35


class PaddleLayoutOCRAgent:
    def __init__(self, config: PaddleLayoutOCRConfig | None = None) -> None:
        self.config = config or PaddleLayoutOCRConfig()
        self.layout_detector = PaddleLayoutDetector(
            PaddleLayoutDetectorConfig(
                show_log=self.config.show_log,
                use_gpu=self.config.use_gpu,
                layout_score_threshold=self.config.layout_score_threshold,
            )
        )
        self.region_ocr = None
        if self.config.perform_region_ocr:
            self.region_ocr = PaddleRegionOCR(
                PaddleRegionOCRConfig(
                    lang=self.config.ocr_lang,
                    show_log=self.config.show_log,
                    use_angle_cls=True,
                    use_gpu=self.config.use_gpu,
                )
            )

    def process_document(
        self,
        document_path: str | Path,
        *,
        progress_callback: Callable[[int, int, int], None] | None = None,
    ) -> tuple[DocumentLayoutResult, dict[str, np.ndarray]]:
        pages = iter_document_pages(document_path, render_scale=self.config.render_scale)
        page_results: list[PageLayoutResult] = []
        sanitized_pages: dict[str, np.ndarray] = {}
        total_pages = len(pages)
        for index, page in enumerate(pages, start=1):
            page_result, sanitized_page = self._process_page(page)
            sanitized_pages[page.page_id] = sanitized_page
            page_results.append(page_result)
            if progress_callback is not None:
                progress_callback(index, total_pages, page.page_number)

        document = DocumentLayoutResult(
            source_path=Path(document_path).as_posix(),
            source_type=page_results[0].source_type if page_results else Path(document_path).suffix.lower().lstrip("."),
            pages=page_results,
            sanitized_pages_dir=None,
        )
        if self.region_ocr is not None:
            document.final_text = build_final_text(document)
        return document, sanitized_pages

    def _process_page(self, page) -> tuple[PageLayoutResult, np.ndarray]:
        regions: list[LayoutRegion] = []
        sanitized_page = page.image.copy()
        reading_order = 0
        for x_offset, segment_image in self._iter_layout_segments(page):
            blocks = self.layout_detector.detect(segment_image)
            routed_blocks = map_and_route_blocks(blocks)
            sanitized_segment = build_sanitized_page(
                segment_image,
                routed_blocks,
                mask_fill=self.config.mask_fill,
            )
            segment_width = segment_image.shape[1]
            sanitized_page[:, x_offset : x_offset + segment_width] = sanitized_segment
            for block in routed_blocks:
                bbox = self._offset_bbox(block["bbox"], x_offset=x_offset)
                action = str(block["action"])
                text = ""
                ocr_confidence: float | None = None
                if action == "keep" and self.region_ocr is not None:
                    cropped = crop_region(segment_image, block["bbox"])
                    text, ocr_confidence = self.region_ocr.recognize(cropped)
                layout_confidence = float(block.get("layout_confidence") or 0.0)
                confidence = ocr_confidence if ocr_confidence is not None else layout_confidence
                regions.append(
                    LayoutRegion(
                        page_id=page.page_id,
                        bbox=bbox,
                        raw_label=str(block.get("raw_label") or ""),
                        mapped_label=str(block["mapped_label"]),
                        action=action,
                        confidence=float(confidence or 0.0),
                        ocr_text=text,
                        reading_order=reading_order,
                        layout_confidence=layout_confidence,
                        ocr_confidence=ocr_confidence,
                    )
                )
                reading_order += 1
        return (
            PageLayoutResult(
                page_id=page.page_id,
                page_number=page.page_number,
                source_path=page.source_path,
                source_type=page.source_type,
                width=page.width,
                height=page.height,
                sanitized_image_path=None,
                regions=regions,
            ),
            sanitized_page,
        )

    def _iter_layout_segments(self, page) -> list[tuple[int, np.ndarray]]:
        if not self._should_split_wide_page(page):
            return [(0, page.image)]
        midpoint = page.width // 2
        left = page.image[:, :midpoint].copy()
        right = page.image[:, midpoint:].copy()
        if left.shape[1] < 32 or right.shape[1] < 32:
            return [(0, page.image)]
        return [(0, left), (midpoint, right)]

    def _should_split_wide_page(self, page) -> bool:
        if not self.config.split_landscape_spreads:
            return False
        if page.height <= 0:
            return False
        return (page.width / page.height) >= self.config.split_aspect_ratio_threshold

    @staticmethod
    def _offset_bbox(bbox: list[int], *, x_offset: int = 0, y_offset: int = 0) -> list[int]:
        x1, y1, x2, y2 = bbox
        return [x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset]

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass
class SplitSummary:
    input_path: str
    output_path: str
    input_pages: int
    output_pages: int
    split_pages: int
    copied_pages: int
    split_order: str
    aspect_ratio_threshold: float


def should_split_page(page: fitz.Page, *, aspect_ratio_threshold: float = 1.35) -> bool:
    rect = page.rect
    if rect.height <= 0:
        return False
    return (rect.width / rect.height) >= aspect_ratio_threshold


def split_landscape_pdf(
    input_path: str | Path,
    output_path: str | Path,
    *,
    aspect_ratio_threshold: float = 1.35,
    split_order: str = "left-right",
) -> SplitSummary:
    src_path = Path(input_path)
    dst_path = Path(output_path)
    if split_order not in {"left-right", "right-left"}:
        raise ValueError("split_order must be 'left-right' or 'right-left'")
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    split_pages = 0
    copied_pages = 0
    with fitz.open(src_path) as src_doc, fitz.open() as out_doc:
        for page in src_doc:
            if not should_split_page(page, aspect_ratio_threshold=aspect_ratio_threshold):
                copied_pages += 1
                out_doc.insert_pdf(src_doc, from_page=page.number, to_page=page.number)
                continue

            split_pages += 1
            rect = page.rect
            midpoint = rect.x0 + (rect.width / 2.0)
            left_clip = fitz.Rect(rect.x0, rect.y0, midpoint, rect.y1)
            right_clip = fitz.Rect(midpoint, rect.y0, rect.x1, rect.y1)
            clips = [left_clip, right_clip] if split_order == "left-right" else [right_clip, left_clip]

            for clip in clips:
                new_page = out_doc.new_page(width=clip.width, height=clip.height)
                new_page.show_pdf_page(new_page.rect, src_doc, page.number, clip=clip)

        out_doc.save(dst_path)

        return SplitSummary(
            input_path=src_path.as_posix(),
            output_path=dst_path.as_posix(),
            input_pages=src_doc.page_count,
            output_pages=out_doc.page_count,
            split_pages=split_pages,
            copied_pages=copied_pages,
            split_order=split_order,
            aspect_ratio_threshold=aspect_ratio_threshold,
        )

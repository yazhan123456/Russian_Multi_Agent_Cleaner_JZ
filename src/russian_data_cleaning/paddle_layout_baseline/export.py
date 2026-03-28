from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .types import DocumentLayoutResult


def build_final_text(document: DocumentLayoutResult) -> str:
    page_chunks: list[str] = []
    for page in document.pages:
        region_chunks: list[str] = []
        for region in sorted(page.regions, key=lambda item: item.reading_order):
            if region.mapped_label not in {"title", "body"}:
                continue
            if not region.ocr_text.strip():
                continue
            region_chunks.append(region.ocr_text.strip())
        if region_chunks:
            page_chunks.append("\n".join(region_chunks))
    return "\n\n".join(page_chunks).strip()


def export_document_result(
    document: DocumentLayoutResult,
    out_dir: str | Path,
    *,
    sanitized_pages: dict[str, np.ndarray] | None = None,
) -> tuple[Path, Path | None]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / f"{document.stem}.layout_ocr.json"
    txt_path = out_root / f"{document.stem}.layout_ocr.txt"
    pages_dir = out_root / f"{document.stem}.sanitized_pages"

    if sanitized_pages:
        pages_dir.mkdir(parents=True, exist_ok=True)
        document.sanitized_pages_dir = pages_dir.as_posix()
        for page in document.pages:
            image = sanitized_pages.get(page.page_id)
            if image is None:
                continue
            page_path = pages_dir / f"page_{page.page_number:04d}.png"
            Image.fromarray(image).save(page_path)
            page.sanitized_image_path = page_path.as_posix()

    if document.final_text:
        document.final_text = build_final_text(document)
    json_path.write_text(json.dumps(document.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if document.final_text:
        txt_path.write_text(document.final_text + "\n", encoding="utf-8")
        return json_path, txt_path
    if txt_path.exists():
        txt_path.unlink()
    return json_path, None

from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

from .types import PageImage


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _pixmap_to_ndarray(pixmap: fitz.Pixmap) -> np.ndarray:
    channels = 4 if pixmap.alpha else 3
    array = np.frombuffer(pixmap.samples, dtype=np.uint8)
    array = array.reshape(pixmap.height, pixmap.width, channels)
    if channels == 4:
        array = array[:, :, :3]
    return array.copy()


def iter_document_pages(document_path: str | Path, *, render_scale: float = 2.0) -> list[PageImage]:
    path = Path(document_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _iter_pdf_pages(path, render_scale=render_scale)
    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        return [_load_image_page(path)]
    raise ValueError(f"Unsupported document type: {path.suffix}")


def _iter_pdf_pages(path: Path, *, render_scale: float) -> list[PageImage]:
    pages: list[PageImage] = []
    matrix = fitz.Matrix(render_scale, render_scale)
    with fitz.open(path) as document:
        for index, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = _pixmap_to_ndarray(pixmap)
            pages.append(
                PageImage(
                    page_id=f"{path.stem}:page_{index}",
                    page_number=index,
                    source_path=path.as_posix(),
                    source_type="pdf",
                    width=image.shape[1],
                    height=image.shape[0],
                    image=image,
                )
            )
    return pages


def _load_image_page(path: Path) -> PageImage:
    if Image is None:  # pragma: no cover
        raise RuntimeError("Pillow is required for image input support. Install with: python3 -m pip install pillow")
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.array(rgb)
    return PageImage(
        page_id=f"{path.stem}:page_1",
        page_number=1,
        source_path=path.as_posix(),
        source_type="image",
        width=array.shape[1],
        height=array.shape[0],
        image=array,
    )

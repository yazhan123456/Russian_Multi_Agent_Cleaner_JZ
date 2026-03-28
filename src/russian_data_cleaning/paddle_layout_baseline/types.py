from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PageImage:
    page_id: str
    page_number: int
    source_path: str
    source_type: str
    width: int
    height: int
    image: np.ndarray


@dataclass
class LayoutRegion:
    page_id: str
    bbox: list[int]
    raw_label: str
    mapped_label: str
    action: str
    confidence: float
    ocr_text: str
    reading_order: int
    layout_confidence: float | None = None
    ocr_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageLayoutResult:
    page_id: str
    page_number: int
    source_path: str
    source_type: str
    width: int
    height: int
    sanitized_image_path: str | None = None
    regions: list[LayoutRegion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "page_number": self.page_number,
            "source_path": self.source_path,
            "source_type": self.source_type,
            "width": self.width,
            "height": self.height,
            "sanitized_image_path": self.sanitized_image_path,
            "regions": [region.to_dict() for region in self.regions],
        }


@dataclass
class DocumentLayoutResult:
    source_path: str
    source_type: str
    pages: list[PageLayoutResult]
    sanitized_pages_dir: str | None = None
    final_text: str = ""

    @property
    def stem(self) -> str:
        return Path(self.source_path).stem

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "source_type": self.source_type,
            "page_count": len(self.pages),
            "sanitized_pages_dir": self.sanitized_pages_dir,
            "pages": [page.to_dict() for page in self.pages],
            "final_text": self.final_text,
        }

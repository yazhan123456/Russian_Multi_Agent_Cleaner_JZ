from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from statistics import mean
from typing import Any

import numpy as np

from .mapping import action_for_label, map_layout_label


def _load_paddleocr():
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PaddleOCR is not installed. Install with: python3 -m pip install paddleocr paddlepaddle pillow numpy"
        ) from exc
    return PaddleOCR


@dataclass
class PaddleRegionOCRConfig:
    lang: str = "ru"
    show_log: bool = False
    use_angle_cls: bool = True
    use_gpu: bool = False


class PaddleRegionOCR:
    def __init__(self, config: PaddleRegionOCRConfig | None = None) -> None:
        self.config = config or PaddleRegionOCRConfig()
        PaddleOCR = _load_paddleocr()
        signature = inspect.signature(PaddleOCR)
        kwargs: dict[str, Any] = {
            "lang": _normalize_ocr_lang(self.config.lang),
        }
        if "show_log" in signature.parameters:
            kwargs["show_log"] = self.config.show_log
        if "use_textline_orientation" in signature.parameters:
            kwargs["use_textline_orientation"] = self.config.use_angle_cls
        elif "use_angle_cls" in signature.parameters:
            kwargs["use_angle_cls"] = self.config.use_angle_cls
        if "use_gpu" in signature.parameters:
            kwargs["use_gpu"] = self.config.use_gpu
        elif "device" in signature.parameters:
            kwargs["device"] = "gpu" if self.config.use_gpu else "cpu"
        self._engine = PaddleOCR(**kwargs)

    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        if hasattr(self._engine, "predict"):
            result = self._engine.predict(image)
        elif hasattr(self._engine, "ocr"):
            result = self._engine.ocr(image, cls=self.config.use_angle_cls)
        else:
            return "", 0.0
        if not result:
            return "", 0.0

        lines: list[str] = []
        scores: list[float] = []
        for block in result:
            if not block:
                continue
            if hasattr(block, "to_dict"):
                block = block.to_dict()
            if isinstance(block, dict):
                rec_texts = block.get("rec_texts") or []
                rec_scores = block.get("rec_scores") or []
                for idx, text in enumerate(rec_texts):
                    text = str(text).strip()
                    score = float(rec_scores[idx] if idx < len(rec_scores) else 0.0)
                    if text:
                        lines.append(text)
                        scores.append(score)
                continue
            for item in block:
                if not item or len(item) < 2:
                    continue
                payload = item[1]
                if not isinstance(payload, (list, tuple)) or len(payload) < 2:
                    continue
                text = str(payload[0]).strip()
                score = float(payload[1] or 0.0)
                if text:
                    lines.append(text)
                    scores.append(score)
        return "\n".join(lines).strip(), (mean(scores) if scores else 0.0)


def map_and_route_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    routed: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        raw_label = str(block.get("raw_label") or "")
        mapped_label = map_layout_label(raw_label)
        routed.append(
            {
                **block,
                "mapped_label": mapped_label,
                "action": action_for_label(mapped_label),
                "order": int(block.get("order", index)),
            }
        )
    return sort_blocks_for_reading_order(routed)


def sort_blocks_for_reading_order(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(block: dict[str, Any]) -> tuple[int, int, int, int]:
        bbox = block.get("bbox") or [0, 0, 0, 0]
        x1, y1, _, _ = bbox
        row_bucket = int(round(y1 / 32.0))
        return (row_bucket, y1, x1, int(block.get("order", 0)))

    return sorted(blocks, key=key)


def crop_region(image: np.ndarray, bbox: list[int]) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return image[y1:y2, x1:x2]


def _normalize_ocr_lang(lang: str | None) -> str:
    normalized = str(lang or "").strip().lower()
    aliases = {
        "cyrillic": "ru",
        "russian": "ru",
        "rus": "ru",
    }
    return aliases.get(normalized, normalized or "ru")

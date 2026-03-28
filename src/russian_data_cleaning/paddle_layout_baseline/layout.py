from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np


def _load_layout_model():
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    try:
        from paddlex import create_model
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PaddleX layout detection is not installed. Install with: "
            'python3 -m pip install "paddlex[ocr]==3.4.2" paddlepaddle pillow numpy pymupdf'
        ) from exc
    return create_model


@dataclass
class PaddleLayoutDetectorConfig:
    show_log: bool = False
    use_gpu: bool = False
    layout_score_threshold: float = 0.0
    model_name: str = "PP-DocLayout_plus-L"


class PaddleLayoutDetector:
    def __init__(self, config: PaddleLayoutDetectorConfig | None = None) -> None:
        self.config = config or PaddleLayoutDetectorConfig()
        create_model = _load_layout_model()
        device = "gpu" if self.config.use_gpu else "cpu"
        self._engine = create_model(self.config.model_name, device=device)

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        results = list(self._engine.predict(image))
        normalized: list[dict[str, Any]] = []
        for item in results:
            payload = item.to_dict() if hasattr(item, "to_dict") else item
            if not isinstance(payload, dict):
                continue
            boxes = payload.get("boxes") or []
            for box in boxes:
                if not isinstance(box, dict):
                    continue
                score = float(box.get("score") or box.get("confidence") or 0.0)
                if score < self.config.layout_score_threshold:
                    continue
                bbox = box.get("coordinate") or box.get("bbox")
                if not bbox:
                    continue
                normalized.append(
                    {
                        "raw_label": str(box.get("label") or box.get("type") or "text"),
                        "bbox": _normalize_bbox(bbox),
                        "layout_confidence": score,
                        "order": len(normalized),
                    }
                )
        return normalized


def _normalize_bbox(bbox: Any) -> list[int]:
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and not isinstance(bbox[0], (list, tuple)):
        x1, y1, x2, y2 = bbox
        return [int(round(float(x1))), int(round(float(y1))), int(round(float(x2))), int(round(float(y2)))]

    points = list(bbox)
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [
        int(round(min(xs))),
        int(round(min(ys))),
        int(round(max(xs))),
        int(round(max(ys))),
    ]

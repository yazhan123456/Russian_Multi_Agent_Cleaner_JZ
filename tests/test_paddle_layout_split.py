from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from russian_data_cleaning.paddle_layout_baseline.agent import PaddleLayoutOCRAgent, PaddleLayoutOCRConfig
from russian_data_cleaning.paddle_layout_baseline.types import PageImage


class _FakeDetector:
    def __init__(self, outputs: list[list[dict]]) -> None:
        self.outputs = outputs
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[dict]:
        result = self.outputs[self.calls]
        self.calls += 1
        return result


class PaddleLayoutSplitTests(unittest.TestCase):
    def test_wide_page_is_split_and_recombined(self) -> None:
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        page = PageImage(
            page_id="demo:page_1",
            page_number=1,
            source_path="demo.pdf",
            source_type="pdf",
            width=80,
            height=40,
            image=image,
        )
        left_blocks = [
            {"raw_label": "text", "bbox": [0, 0, 40, 40], "layout_confidence": 0.9, "order": 0},
        ]
        right_blocks = [
            {"raw_label": "footnote", "bbox": [0, 0, 40, 40], "layout_confidence": 0.8, "order": 0},
        ]

        agent = object.__new__(PaddleLayoutOCRAgent)
        agent.config = PaddleLayoutOCRConfig(
            split_landscape_spreads=True,
            split_aspect_ratio_threshold=1.2,
            mask_fill=255,
        )
        agent.layout_detector = _FakeDetector([left_blocks, right_blocks])
        agent.region_ocr = None

        page_result, sanitized = agent._process_page(page)

        self.assertEqual(agent.layout_detector.calls, 2)
        self.assertEqual(len(page_result.regions), 2)
        self.assertEqual(page_result.regions[0].bbox, [0, 0, 40, 40])
        self.assertEqual(page_result.regions[0].mapped_label, "body")
        self.assertEqual(page_result.regions[1].bbox, [40, 0, 80, 40])
        self.assertEqual(page_result.regions[1].mapped_label, "note")
        self.assertEqual(int(sanitized[1, 1, 0]), 0)
        self.assertEqual(int(sanitized[1, 60, 0]), 255)

    def test_portrait_page_is_not_split(self) -> None:
        image = np.zeros((8, 4, 3), dtype=np.uint8)
        page = PageImage(
            page_id="demo:page_1",
            page_number=1,
            source_path="demo.pdf",
            source_type="pdf",
            width=4,
            height=8,
            image=image,
        )

        agent = object.__new__(PaddleLayoutOCRAgent)
        agent.config = PaddleLayoutOCRConfig(
            split_landscape_spreads=True,
            split_aspect_ratio_threshold=1.2,
        )

        segments = agent._iter_layout_segments(page)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0][0], 0)
        self.assertEqual(segments[0][1].shape, image.shape)


if __name__ == "__main__":
    unittest.main()

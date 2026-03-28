from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from russian_data_cleaning.paddle_layout_baseline.sanitizer import build_sanitized_page


class PaddleLayoutSanitizerTests(unittest.TestCase):
    def test_masks_only_mask_regions(self) -> None:
        image = np.zeros((6, 6, 3), dtype=np.uint8)
        blocks = [
            {"bbox": [0, 0, 3, 3], "action": "keep"},
            {"bbox": [3, 0, 6, 3], "action": "mask"},
        ]
        sanitized = build_sanitized_page(image, blocks, mask_fill=255)
        self.assertEqual(int(sanitized[1, 1, 0]), 0)
        self.assertEqual(int(sanitized[1, 4, 0]), 255)

    def test_mask_region_preserves_overlapping_keep_area(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        blocks = [
            {"bbox": [1, 1, 7, 7], "action": "mask"},
            {"bbox": [3, 3, 5, 5], "action": "keep"},
        ]
        sanitized = build_sanitized_page(image, blocks, mask_fill=255)
        self.assertEqual(int(sanitized[2, 2, 0]), 255)
        self.assertEqual(int(sanitized[3, 3, 0]), 0)
        self.assertEqual(int(sanitized[4, 4, 0]), 0)
        self.assertEqual(int(sanitized[6, 6, 0]), 255)


if __name__ == "__main__":
    unittest.main()

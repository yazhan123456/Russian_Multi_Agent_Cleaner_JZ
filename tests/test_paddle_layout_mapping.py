from __future__ import annotations

import unittest

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from russian_data_cleaning.paddle_layout_baseline.mapping import action_for_label, map_layout_label


class PaddleLayoutMappingTests(unittest.TestCase):
    def test_maps_expected_labels(self) -> None:
        self.assertEqual(map_layout_label("text"), "body")
        self.assertEqual(map_layout_label("title"), "title")
        self.assertEqual(map_layout_label("table"), "table")
        self.assertEqual(map_layout_label("picture"), "picture")
        self.assertEqual(map_layout_label("figure"), "picture")
        self.assertEqual(map_layout_label("footnote"), "note")
        self.assertEqual(map_layout_label("reference"), "note")
        self.assertEqual(map_layout_label("caption"), "note")

    def test_unknown_defaults_to_body_for_recall(self) -> None:
        self.assertEqual(map_layout_label("mysterious_block"), "body")

    def test_actions_follow_fixed_routing(self) -> None:
        self.assertEqual(action_for_label("title"), "keep")
        self.assertEqual(action_for_label("body"), "keep")
        self.assertEqual(action_for_label("note"), "mask")
        self.assertEqual(action_for_label("picture"), "mask")
        self.assertEqual(action_for_label("table"), "mask")


if __name__ == "__main__":
    unittest.main()

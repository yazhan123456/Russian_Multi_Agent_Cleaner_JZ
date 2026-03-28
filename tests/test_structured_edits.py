from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from russian_data_cleaning.structured_edits import apply_edit_plan, execute_edit_plan


class StructuredEditPlanTests(unittest.TestCase):
    def test_apply_edit_plan_merges_broken_hyphen_line(self) -> None:
        text = "пред-\nставитель заявителя"
        cleaned_text, edits, notes, drop_page = apply_edit_plan(
            text,
            {
                "drop_page": False,
                "operations": [
                    {"op": "merge_with_next", "line": 1, "reason": "merge broken hyphenation"},
                ],
            },
            allow_drop_page=False,
        )

        self.assertEqual(cleaned_text, "представитель заявителя")
        self.assertFalse(drop_page)
        self.assertTrue(edits)
        self.assertFalse(notes)

    def test_execute_edit_plan_matches_legacy_wrapper(self) -> None:
        text = "foo [12]\nbar"
        payload = {
            "drop_page": False,
            "operations": [
                {"op": "remove_inline_pattern", "pattern": "bracket_note_markers", "reason": "strip note markers"},
                {"op": "normalize_spacing", "reason": "cleanup"},
            ],
        }

        result = execute_edit_plan(text, payload, allow_drop_page=False)
        legacy = apply_edit_plan(text, payload, allow_drop_page=False)

        self.assertEqual(result.text, legacy[0])
        self.assertEqual(result.applied_edits, legacy[1])
        self.assertEqual(result.notes, legacy[2])
        self.assertEqual(result.drop_page, legacy[3])


if __name__ == "__main__":
    unittest.main()

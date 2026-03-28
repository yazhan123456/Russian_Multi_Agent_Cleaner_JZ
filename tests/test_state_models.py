from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from russian_data_cleaning.state_models import PageProcessingState, PageState


class PageStateModelTests(unittest.TestCase):
    def test_page_state_round_trip_preserves_core_fields(self) -> None:
        page_state = PageState.create(
            doc_id="book_a",
            page_num=3,
            source_path="/tmp/book.pdf",
            source_type="pdf",
        )
        page_state.route_decision = "easy_page"
        page_state.ocr_mode = "extract"
        page_state.raw_text = "Первый абзац."
        page_state.rule_cleaned_text = "Первый абзац."
        page_state.current_state = PageProcessingState.RULE_CLEANED
        page_state.last_success_state = PageProcessingState.RULE_CLEANED
        page_state.record_provenance(
            agent="test",
            input_fields=["source_path"],
            output_fields=["raw_text"],
            note="seed",
        )

        restored = PageState.from_dict(page_state.to_dict())
        self.assertEqual(restored.doc_id, "book_a")
        self.assertEqual(restored.page_num, 3)
        self.assertEqual(restored.current_state, PageProcessingState.RULE_CLEANED)
        self.assertEqual(restored.last_success_state, PageProcessingState.RULE_CLEANED)
        self.assertEqual(restored.raw_text, "Первый абзац.")
        self.assertEqual(restored.provenance[0].agent, "test")


if __name__ == "__main__":
    unittest.main()

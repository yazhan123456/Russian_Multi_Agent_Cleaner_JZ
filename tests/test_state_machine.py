from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from russian_data_cleaning.state_models import PageProcessingState, PageState
from russian_data_cleaning.state_machine import effective_state, mark_failed, transition


class PageStateMachineTests(unittest.TestCase):
    def test_state_transitions_follow_expected_lifecycle(self) -> None:
        page_state = PageState.create(
            doc_id="book_a",
            page_num=1,
            source_path="/tmp/book.pdf",
            source_type="pdf",
        )
        transition(page_state, PageProcessingState.EXTRACTED, agent="OCRAgent", note="extract")
        transition(page_state, PageProcessingState.RULE_CLEANED, agent="CleaningAgent")
        transition(page_state, PageProcessingState.PRIMARY_CLEANED, agent="PrimaryCleaningStage")

        self.assertEqual(page_state.current_state, PageProcessingState.PRIMARY_CLEANED)
        self.assertEqual(page_state.last_success_state, PageProcessingState.PRIMARY_CLEANED)
        self.assertEqual(page_state.processing_history[-1].to_state, PageProcessingState.PRIMARY_CLEANED.value)

    def test_failed_page_keeps_last_successful_state_for_resume(self) -> None:
        page_state = PageState.create(
            doc_id="book_a",
            page_num=2,
            source_path="/tmp/book.pdf",
            source_type="pdf",
        )
        transition(page_state, PageProcessingState.OCR_DONE, agent="OCRAgent", note="ocr")
        mark_failed(page_state, agent="CleaningAgent", error="boom")

        self.assertEqual(page_state.current_state, PageProcessingState.FAILED)
        self.assertEqual(page_state.last_success_state, PageProcessingState.OCR_DONE)
        self.assertEqual(effective_state(page_state), PageProcessingState.OCR_DONE)

    def test_failed_page_can_resume_transition_from_last_success_state(self) -> None:
        page_state = PageState.create(
            doc_id="book_a",
            page_num=3,
            source_path="/tmp/book.pdf",
            source_type="pdf",
        )
        transition(page_state, PageProcessingState.OCR_DONE, agent="OCRAgent", note="ocr")
        mark_failed(page_state, agent="CleaningAgent", error="transient error")

        transition(page_state, PageProcessingState.RULE_CLEANED, agent="CleaningAgent", note="resume after failure")

        self.assertEqual(page_state.current_state, PageProcessingState.RULE_CLEANED)
        self.assertEqual(page_state.last_success_state, PageProcessingState.RULE_CLEANED)
        self.assertEqual(page_state.processing_history[-1].from_state, PageProcessingState.OCR_DONE.value)
        self.assertEqual(page_state.processing_history[-1].to_state, PageProcessingState.RULE_CLEANED.value)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from russian_data_cleaning.state_models import PageProcessingState, PageState


PROCESS_BOOKS_PATH = ROOT / "scripts" / "process_books.py"
PROCESS_BOOKS_SPEC = importlib.util.spec_from_file_location("process_books_optimizations_under_test", PROCESS_BOOKS_PATH)
assert PROCESS_BOOKS_SPEC is not None and PROCESS_BOOKS_SPEC.loader is not None
process_books = importlib.util.module_from_spec(PROCESS_BOOKS_SPEC)
PROCESS_BOOKS_SPEC.loader.exec_module(process_books)


class ProcessOptimizationsTests(unittest.TestCase):
    def test_should_early_exit_after_review_only_on_safe_extract_body_page(self) -> None:
        page_state = PageState.create(
            doc_id="book_safe",
            page_num=1,
            source_path="/tmp/book.pdf",
            source_type="pdf",
        )
        page_state.current_state = PageProcessingState.REVIEWED
        page_state.last_success_state = PageProcessingState.REVIEWED
        page_state.route_decision = "easy_page"
        page_state.risk_level = "low"

        should_exit, reason = process_books.should_early_exit_after_review(
            page_state=page_state,
            ocr_page={
                "source": "extract",
                "page_type": "body_only",
                "layout_status": "pdf_blocks",
            },
            cleaned_page={
                "cleaned_text": "Это безопасная страница с достаточным количеством русского текста. " * 6,
                "status": "deepseek_structured",
                "flags": [],
                "homoglyph_audit": {"warned": 0},
            },
            review_page={"page_verdict": "approve", "issue_tags": []},
            gemini_review_enabled=False,
        )

        self.assertTrue(should_exit)
        self.assertEqual(reason, "easy_extract_body_only_review_approve")

    def test_apply_early_exit_after_review_promotes_to_structure_restored(self) -> None:
        page_state = PageState.create(
            doc_id="book_safe",
            page_num=2,
            source_path="/tmp/book.pdf",
            source_type="pdf",
        )
        page_state.current_state = PageProcessingState.REVIEWED
        page_state.last_success_state = PageProcessingState.REVIEWED
        page_state.review_tags = []

        process_books.apply_early_exit_after_review(
            page_state,
            cleaned_page={"cleaned_text": "Чистый текст страницы."},
            reason="easy_extract_body_only_review_approve",
            backend="deepseek",
            model="deepseek-chat",
            include_primary_payload=True,
        )

        self.assertEqual(page_state.current_state, PageProcessingState.STRUCTURE_RESTORED)
        self.assertEqual(page_state.repaired_text, "Чистый текст страницы.")
        self.assertEqual(page_state.final_text, "Чистый текст страницы.")
        self.assertIn("repaired", page_state.stage_payloads)
        self.assertIn("repaired_primary", page_state.stage_payloads)
        self.assertEqual(page_state.structure_plan["status"], "early_exit_passthrough")


if __name__ == "__main__":
    unittest.main()

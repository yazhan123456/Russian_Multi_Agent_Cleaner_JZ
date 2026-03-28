from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from russian_data_cleaning.checkpoints import PageCheckpointStore
from russian_data_cleaning.state_models import PageProcessingState, PageState


class CheckpointResumeTests(unittest.TestCase):
    def test_page_checkpoint_store_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            book_dir = Path(tmp_dir) / "book"
            store = PageCheckpointStore(book_dir)
            page_state = PageState.create(
                doc_id="book_a",
                page_num=8,
                source_path="/tmp/book.pdf",
                source_type="pdf",
            )
            page_state.current_state = PageProcessingState.RULE_CLEANED
            page_state.last_success_state = PageProcessingState.RULE_CLEANED
            page_state.rule_cleaned_text = "Готовый текст."
            page_state.stage_payloads["rule_cleaned"] = {"page_number": 8, "cleaned_text": "Готовый текст."}

            store.save_page(page_state)
            restored = store.load_page(8)

            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(restored.page_num, 8)
            self.assertEqual(restored.current_state, PageProcessingState.RULE_CLEANED)
            self.assertEqual(restored.rule_cleaned_text, "Готовый текст.")

    def test_page_checkpoint_store_preserves_repaired_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            book_dir = Path(tmp_dir) / "book"
            store = PageCheckpointStore(book_dir)
            page_state = PageState.create(
                doc_id="book_a",
                page_num=11,
                source_path="/tmp/book.pdf",
                source_type="pdf",
            )
            page_state.current_state = PageProcessingState.REPAIRED
            page_state.last_success_state = PageProcessingState.REPAIRED
            page_state.review_tags = ["heading_structure_risky"]
            page_state.repaired_text = "Исправленный текст"
            page_state.stage_payloads["repaired"] = {"page_number": 11, "cleaned_text": "Исправленный текст"}

            store.save_page(page_state)
            restored = store.load_page(11)

            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(restored.current_state, PageProcessingState.REPAIRED)
            self.assertEqual(restored.repaired_text, "Исправленный текст")
            self.assertEqual(restored.review_tags, ["heading_structure_risky"])

    def test_page_checkpoint_store_preserves_structure_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            book_dir = Path(tmp_dir) / "book"
            store = PageCheckpointStore(book_dir)
            page_state = PageState.create(
                doc_id="book_a",
                page_num=15,
                source_path="/tmp/book.pdf",
                source_type="pdf",
            )
            page_state.current_state = PageProcessingState.STRUCTURE_RESTORED
            page_state.last_success_state = PageProcessingState.STRUCTURE_RESTORED
            page_state.repaired_text = "VII. Заголовок Текст."
            page_state.final_text = "VII. Заголовок\nТекст."
            page_state.structure_plan = {
                "backend": "gemini",
                "model": "gemini-2.5-flash",
                "status": "gemini",
                "notes": [],
                "skipped_reason": None,
                "final_text_source": "structure_restore_generated",
            }

            store.save_page(page_state)
            restored = store.load_page(15)

            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(restored.current_state, PageProcessingState.STRUCTURE_RESTORED)
            self.assertEqual(restored.final_text, "VII. Заголовок\nТекст.")
            self.assertEqual(restored.structure_plan["final_text_source"], "structure_restore_generated")


if __name__ == "__main__":
    unittest.main()

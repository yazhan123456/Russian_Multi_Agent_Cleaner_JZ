from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from russian_data_cleaning.gemini_structure_agent import GeminiStructureAgent, GeminiStructureConfig
from russian_data_cleaning.deepseek_structure_agent import DeepSeekStructureAgent, DeepSeekStructureConfig
from russian_data_cleaning.state_models import PageProcessingState, PageState


PROCESS_BOOKS_PATH = ROOT / "scripts" / "process_books.py"
PROCESS_BOOKS_SPEC = importlib.util.spec_from_file_location("process_books_under_test", PROCESS_BOOKS_PATH)
assert PROCESS_BOOKS_SPEC is not None and PROCESS_BOOKS_SPEC.loader is not None
process_books = importlib.util.module_from_spec(PROCESS_BOOKS_SPEC)
PROCESS_BOOKS_SPEC.loader.exec_module(process_books)


class StructureExportFlowTests(unittest.TestCase):
    def test_gemini_structure_run_writes_plan_and_final_text(self) -> None:
        agent = GeminiStructureAgent.__new__(GeminiStructureAgent)
        agent.config = GeminiStructureConfig(model="gemini-2.5-flash")
        agent.restore_page = lambda ocr_page, repaired_page: {
            "page_number": repaired_page["page_number"],
            "restored_text": "VII. Заголовок\nТекст абзаца.",
            "status": "gemini",
            "notes": [],
            "skipped_reason": None,
            "final_text_source": "structure_restore_generated",
        }

        page_state = PageState.create(
            doc_id="book_a",
            page_num=12,
            source_path="/tmp/book.pdf",
            source_type="pdf",
        )
        page_state.current_state = PageProcessingState.REPAIRED
        page_state.last_success_state = PageProcessingState.REPAIRED
        page_state.repaired_text = "VII. Заголовок Текст абзаца."
        page_state.stage_payloads["ocr"] = {
            "page_number": 12,
            "body_text": "VII. Заголовок\nТекст абзаца.",
            "selected_text": "VII. Заголовок\nТекст абзаца.",
        }

        agent.run(page_state)

        self.assertEqual(page_state.current_state, PageProcessingState.STRUCTURE_RESTORED)
        self.assertEqual(page_state.final_text, "VII. Заголовок\nТекст абзаца.")
        self.assertEqual(page_state.repaired_text, "VII. Заголовок Текст абзаца.")
        self.assertEqual(page_state.structure_plan["status"], "gemini")
        self.assertEqual(page_state.structure_plan["final_text_source"], "structure_restore_generated")
        self.assertIn("source=structure_restore_generated", page_state.provenance[-1].note)
        self.assertIn("model=gemini-2.5-flash", page_state.processing_history[-1].note)

    def test_deepseek_structure_run_writes_plan_and_final_text(self) -> None:
        agent = DeepSeekStructureAgent.__new__(DeepSeekStructureAgent)
        agent.config = DeepSeekStructureConfig(model="deepseek-chat")
        agent.restore_page = lambda ocr_page, repaired_page: {
            "page_number": repaired_page["page_number"],
            "restored_text": "VIII. Заголовок\nТекст абзаца.",
            "status": "deepseek",
            "notes": [],
            "skipped_reason": None,
            "final_text_source": "structure_restore_generated",
        }

        page_state = PageState.create(
            doc_id="book_ds",
            page_num=7,
            source_path="/tmp/book.pdf",
            source_type="pdf",
        )
        page_state.current_state = PageProcessingState.REPAIRED
        page_state.last_success_state = PageProcessingState.REPAIRED
        page_state.repaired_text = "VIII. Заголовок Текст абзаца."
        page_state.stage_payloads["ocr"] = {
            "page_number": 7,
            "body_text": "VIII. Заголовок\nТекст абзаца.",
            "selected_text": "VIII. Заголовок\nТекст абзаца.",
        }

        agent.run(page_state)

        self.assertEqual(page_state.current_state, PageProcessingState.STRUCTURE_RESTORED)
        self.assertEqual(page_state.final_text, "VIII. Заголовок\nТекст абзаца.")
        self.assertEqual(page_state.structure_plan["status"], "deepseek")
        self.assertEqual(page_state.structure_plan["backend"], "deepseek")
        self.assertEqual(page_state.structure_plan["final_text_source"], "structure_restore_generated")
        self.assertIn("backend=deepseek", page_state.provenance[-1].note)
        self.assertIn("model=deepseek-chat", page_state.processing_history[-1].note)

    def test_apply_structure_passthrough_tracks_passthrough_source(self) -> None:
        page_state = PageState.create(
            doc_id="book_b",
            page_num=3,
            source_path="/tmp/book.pdf",
            source_type="pdf",
        )
        page_state.current_state = PageProcessingState.REPAIRED
        page_state.last_success_state = PageProcessingState.REPAIRED
        page_state.repaired_text = "Текст после repair."

        process_books.apply_structure_passthrough(
            page_state,
            backend="gemini",
            model="gemini-2.5-flash",
            status="skipped_safe_page",
            skipped_reason="safe_page",
            notes=["skipped_safe_page"],
        )

        self.assertEqual(page_state.current_state, PageProcessingState.STRUCTURE_RESTORED)
        self.assertEqual(page_state.final_text, "Текст после repair.")
        self.assertEqual(page_state.structure_plan["final_text_source"], "repaired_passthrough")
        self.assertEqual(page_state.structure_plan["skipped_reason"], "safe_page")
        self.assertIn("skipped_reason=safe_page", page_state.provenance[-1].note)
        self.assertIn("source=repaired_passthrough", page_state.processing_history[-1].note)

    def test_build_export_document_prefers_final_then_repaired_then_legacy(self) -> None:
        page1 = PageState.create(doc_id="book_c", page_num=1, source_path="/tmp/book.pdf", source_type="pdf")
        page1.current_state = PageProcessingState.STRUCTURE_RESTORED
        page1.last_success_state = PageProcessingState.STRUCTURE_RESTORED
        page1.final_text = "final text"
        page1.structure_plan = {"final_text_source": "structure_restore_generated", "status": "gemini"}

        page2 = PageState.create(doc_id="book_c", page_num=2, source_path="/tmp/book.pdf", source_type="pdf")
        page2.current_state = PageProcessingState.REPAIRED
        page2.last_success_state = PageProcessingState.REPAIRED
        page2.repaired_text = "repaired text"

        page3 = PageState.create(doc_id="book_c", page_num=3, source_path="/tmp/book.pdf", source_type="pdf")
        page3.current_state = PageProcessingState.REVIEWED
        page3.last_success_state = PageProcessingState.REVIEWED

        export_doc, export_sources = process_books.build_export_document(
            relative_path="book.pdf",
            page_states={1: page1, 2: page2, 3: page3},
            page_numbers=[1, 2, 3],
            restored_doc=None,
            repaired_doc=None,
            cleaned_doc={
                "relative_path": "book.pdf",
                "pages": [
                    {"page_number": 1, "cleaned_text": "legacy 1"},
                    {"page_number": 2, "cleaned_text": "legacy 2"},
                    {"page_number": 3, "cleaned_text": "legacy 3"},
                ],
            },
        )

        self.assertEqual(export_doc["pages"][0]["cleaned_text"], "final text")
        self.assertEqual(export_doc["pages"][1]["cleaned_text"], "repaired text")
        self.assertEqual(export_doc["pages"][2]["cleaned_text"], "legacy 3")
        self.assertEqual(export_sources, {1: "final_text", 2: "repaired_text", 3: "legacy_cleaned"})

    def test_build_export_document_filters_frontmatter_backmatter_and_note_heavy_pages(self) -> None:
        page1 = PageState.create(doc_id="book_f", page_num=1, source_path="/tmp/book.pdf", source_type="pdf")
        page1.current_state = PageProcessingState.STRUCTURE_RESTORED
        page1.last_success_state = PageProcessingState.STRUCTURE_RESTORED
        page1.page_type = "body_only"
        page1.route_decision = "risky_extract_page"
        page1.final_text = "Научный совет по истории мировой культуры РАН\nМатериалы ХI международной научной конференции"
        page1.structure_plan = {"final_text_source": "structure_restore_generated", "status": "gemini"}

        page2 = PageState.create(doc_id="book_f", page_num=12, source_path="/tmp/book.pdf", source_type="pdf")
        page2.current_state = PageProcessingState.REPAIRED
        page2.last_success_state = PageProcessingState.REPAIRED
        page2.page_type = "body_with_notes"
        page2.route_decision = "risky_extract_page"
        page2.review_tags = ["footnote_marker_left", "endnote_block_left"]
        page2.repaired_text = "1) Там же.\n2) Архитектурное наследство. — №20.\n3) Там же."

        page3 = PageState.create(doc_id="book_f", page_num=10, source_path="/tmp/book.pdf", source_type="pdf")
        page3.current_state = PageProcessingState.REPAIRED
        page3.last_success_state = PageProcessingState.REPAIRED
        page3.page_type = "body_only"
        page3.route_decision = "easy_page"
        page3.repaired_text = "Основной текст страницы."

        page4 = PageState.create(doc_id="book_f", page_num=20, source_path="/tmp/book.pdf", source_type="pdf")
        page4.current_state = PageProcessingState.STRUCTURE_RESTORED
        page4.last_success_state = PageProcessingState.STRUCTURE_RESTORED
        page4.page_type = "publisher_meta"
        page4.route_decision = "skip_nonbody_page"
        page4.final_text = "КОРОТКО ОБ АВТОРАХ\nISBN 978-5-0000-0000-0"
        page4.structure_plan = {"final_text_source": "structure_restore_generated", "status": "gemini"}

        export_doc, export_sources = process_books.build_export_document(
            relative_path="book.pdf",
            page_states={1: page1, 10: page3, 12: page2, 20: page4},
            page_numbers=[1, 10, 12, 20],
            restored_doc=None,
            repaired_doc=None,
            cleaned_doc={"relative_path": "book.pdf", "pages": []},
        )

        self.assertEqual([page["page_number"] for page in export_doc["pages"]], [10])
        self.assertEqual(export_doc["pages"][0]["cleaned_text"], "Основной текст страницы.")
        self.assertEqual(export_sources[1], "filtered_frontmatter_page")
        self.assertEqual(export_sources[12], "filtered_note_heavy_page")
        self.assertEqual(export_sources[10], "repaired_text")
        self.assertEqual(export_sources[20], "filtered_nonbody_page")

    def test_backfill_structure_states_restores_final_text_from_legacy_structure_json(self) -> None:
        page_state = PageState.create(
            doc_id="book_d",
            page_num=4,
            source_path="/tmp/book.pdf",
            source_type="pdf",
        )
        page_state.current_state = PageProcessingState.REPAIRED
        page_state.last_success_state = PageProcessingState.REPAIRED
        page_state.repaired_text = "Текст."

        process_books.backfill_structure_states(
            page_states={4: page_state},
            page_numbers=[4],
            restored_doc={
                "relative_path": "book.pdf",
                "model": "gemini-2.5-flash",
                "pages": [
                    {
                        "page_number": 4,
                        "restored_text": "IV. Заголовок\nТекст.",
                        "status": "gemini",
                        "notes": [],
                    }
                ],
            },
        )

        self.assertEqual(page_state.current_state, PageProcessingState.STRUCTURE_RESTORED)
        self.assertEqual(page_state.final_text, "IV. Заголовок\nТекст.")
        self.assertEqual(page_state.structure_plan["model"], "gemini-2.5-flash")
        self.assertEqual(page_state.structure_plan["final_text_source"], "structure_restore_generated")

    def test_ensure_repaired_state_for_structure_recovers_from_failed_page(self) -> None:
        page_state = PageState.create(
            doc_id="book_e",
            page_num=6,
            source_path="/tmp/book.pdf",
            source_type="pdf",
        )
        page_state.current_state = PageProcessingState.FAILED
        page_state.last_success_state = PageProcessingState.REVIEWED
        page_state.primary_clean_text = "Текст после primary."

        process_books.ensure_repaired_state_for_structure(page_state)

        self.assertEqual(page_state.current_state, PageProcessingState.REPAIRED)
        self.assertEqual(page_state.repaired_text, "Текст после primary.")
        self.assertEqual(page_state.processing_history[-1].to_state, PageProcessingState.REPAIRED.value)


if __name__ == "__main__":
    unittest.main()

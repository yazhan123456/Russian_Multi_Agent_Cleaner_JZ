from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from russian_data_cleaning.deepseek_repair_agent import DeepSeekRepairAgent, DeepSeekRepairConfig
from russian_data_cleaning.gemini_review import GeminiReviewAgent, GeminiReviewConfig
from russian_data_cleaning.review_agent import ReviewAgent
from russian_data_cleaning.state_models import PageProcessingState, PageState


class ReviewRepairFlowTests(unittest.TestCase):
    def test_review_agent_run_writes_tags_and_state_without_mutating_primary_text(self) -> None:
        page_state = PageState.create(
            doc_id="book_a",
            page_num=5,
            source_path="/tmp/book.pdf",
            source_type="pdf",
        )
        page_state.current_state = PageProcessingState.PRIMARY_CLEANED
        page_state.last_success_state = PageProcessingState.PRIMARY_CLEANED
        page_state.primary_clean_text = "Текст [12]\n123 https://example.com"
        page_state.stage_payloads["ocr"] = {
            "page_number": 5,
            "source": "extract",
            "body_text": "Текст [12]\n123 https://example.com",
            "selected_text": "Текст [12]\n123 https://example.com",
        }
        page_state.stage_payloads["cleaned"] = {
            "page_number": 5,
            "source": "extract",
            "cleaned_text": "Текст [12]\n123 https://example.com",
            "edits": [],
            "flags": [],
            "protected_hits": [],
        }

        original_text = page_state.primary_clean_text
        ReviewAgent().run(page_state)

        self.assertEqual(page_state.current_state, PageProcessingState.REVIEWED)
        self.assertEqual(page_state.primary_clean_text, original_text)
        self.assertIn("footnote_marker_left", page_state.review_tags)
        self.assertEqual(page_state.risk_level, "medium")
        self.assertIn("review", page_state.stage_payloads)

    def test_gemini_review_run_only_enriches_and_upgrades_risk(self) -> None:
        agent = GeminiReviewAgent.__new__(GeminiReviewAgent)
        agent.config = GeminiReviewConfig(model="gemini-2.5-flash", risky_only=True)
        agent.review_page = lambda cleaned_page, heuristic_page: {
            "page_number": cleaned_page["page_number"],
            "llm_verdict": "reject",
            "summary": "Heading probably lost.",
            "concerns": ["Possible heading loss"],
        }

        page_state = PageState.create(
            doc_id="book_a",
            page_num=7,
            source_path="/tmp/book.pdf",
            source_type="pdf",
        )
        page_state.current_state = PageProcessingState.REVIEWED
        page_state.last_success_state = PageProcessingState.REVIEWED
        page_state.review_tags = ["heading_structure_risky"]
        page_state.risk_level = "medium"
        page_state.primary_clean_text = "VII. Текст"
        page_state.stage_payloads["cleaned"] = {
            "page_number": 7,
            "cleaned_text": "VII. Текст",
            "flags": [],
            "protected_hits": [],
            "edits": [],
        }
        page_state.stage_payloads["review"] = {
            "page_number": 7,
            "page_verdict": "escalate",
            "issue_tags": ["heading_structure_risky"],
            "review_records": [],
        }

        agent.run(page_state)

        self.assertEqual(page_state.current_state, PageProcessingState.REVIEWED)
        self.assertEqual(page_state.review_tags, ["heading_structure_risky"])
        self.assertEqual(page_state.risk_level, "high")
        self.assertEqual(page_state.primary_clean_text, "VII. Текст")
        self.assertIn("gemini_review", page_state.stage_payloads)

    def test_repair_run_writes_repaired_text_without_new_taxonomy(self) -> None:
        agent = DeepSeekRepairAgent.__new__(DeepSeekRepairAgent)
        agent.config = DeepSeekRepairConfig(model="deepseek-chat", risky_only=True)
        agent.repair_page = lambda ocr_page, cleaned_page, heuristic_page, gemini_review_page=None, review_tags=None, risk_level=None: {
            **cleaned_page,
            "cleaned_text": "Исправленный текст",
            "repaired_text": "Исправленный текст",
            "llm_repair_plan": {"drop_page": False, "operations": []},
            "repair_status": "deepseek_structured",
            "repair_notes": [],
            "repair_issue_tags": list(review_tags or []),
        }

        page_state = PageState.create(
            doc_id="book_a",
            page_num=9,
            source_path="/tmp/book.pdf",
            source_type="pdf",
        )
        page_state.current_state = PageProcessingState.REVIEWED
        page_state.last_success_state = PageProcessingState.REVIEWED
        page_state.review_tags = ["heading_structure_risky"]
        page_state.risk_level = "medium"
        page_state.edit_plan = {"drop_page": False, "operations": []}
        page_state.stage_payloads["ocr"] = {"page_number": 9, "source": "extract", "body_text": "VII. текст"}
        page_state.stage_payloads["cleaned"] = {
            "page_number": 9,
            "source": "extract",
            "cleaned_text": "VII. текст",
            "flags": [],
            "protected_hits": [],
            "edits": [],
        }
        page_state.stage_payloads["review"] = {
            "page_number": 9,
            "page_verdict": "escalate",
            "issue_tags": ["heading_structure_risky"],
            "review_records": [],
        }

        agent.run(page_state)

        self.assertEqual(page_state.current_state, PageProcessingState.REPAIRED)
        self.assertEqual(page_state.repaired_text, "Исправленный текст")
        self.assertEqual(page_state.repair_plan, {"drop_page": False, "operations": []})
        self.assertEqual(page_state.review_tags, ["heading_structure_risky"])
        self.assertIn("repaired", page_state.stage_payloads)
        self.assertEqual(page_state.stage_payloads["repaired"]["repair_issue_tags"], ["heading_structure_risky"])


if __name__ == "__main__":
    unittest.main()

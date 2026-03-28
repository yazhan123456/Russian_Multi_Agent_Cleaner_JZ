from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from russian_data_cleaning.deepseek_repair_agent import DeepSeekRepairAgent
from russian_data_cleaning.gemini_repair_agent import GeminiRepairAgent
from russian_data_cleaning.ocr_agent import OCRAgent, OCRAgentConfig


class NarrowQualityFixTests(unittest.TestCase):
    def test_localized_mojibake_line_is_treated_as_mojibake(self) -> None:
        agent = OCRAgent(OCRAgentConfig(backend="auto"))
        text = (
            "Нормальный русский абзац.\n"
            "8.4. Ƀɬɪɩɦɨɠɨɣɠ ɨɛɥɛɢɛɨɣɺ\n"
            "ɝ ɝɣɟɠ ɣɬɪɫɛɝɣɭɠɦɷɨɶɰ ɫɛɜɩɭ\n"
            "Исправительные работы применяются только в качестве основного вида наказания.\n"
        )
        self.assertTrue(agent._looks_mojibake(text))
        self.assertEqual(
            agent._select_source("pdf_extract_then_clean", text, extracted_char_count=len(text)),
            "ocr",
        )

    def test_deepseek_repair_restores_inline_list_break(self) -> None:
        agent = DeepSeekRepairAgent.__new__(DeepSeekRepairAgent)
        updated, notes = agent._restore_inline_list_breaks(
            "- в период отбывания исправительных работ; - осужденный не вправе отказаться"
        )
        self.assertIn(";\n- осужденный", updated)
        self.assertEqual(notes, ["restored_inline_list_breaks"])

    def test_gemini_repair_restores_inline_list_break(self) -> None:
        agent = GeminiRepairAgent.__new__(GeminiRepairAgent)
        updated, notes = agent._restore_inline_list_breaks(
            "- в период отбывания исправительных работ; - осужденный не вправе отказаться"
        )
        self.assertIn(";\n- осужденный", updated)
        self.assertEqual(notes, ["restored_inline_list_breaks"])


if __name__ == "__main__":
    unittest.main()

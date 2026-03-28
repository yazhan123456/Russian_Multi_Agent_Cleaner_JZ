from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import unittest

import fitz
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from russian_data_cleaning.ocr_agent import OCRAgent, OCRAgentConfig
from russian_data_cleaning.page_commander import OCRPlan


PROCESS_BOOKS_PATH = ROOT / "scripts" / "process_books.py"
PROCESS_BOOKS_SPEC = importlib.util.spec_from_file_location("process_books_under_test", PROCESS_BOOKS_PATH)
assert PROCESS_BOOKS_SPEC and PROCESS_BOOKS_SPEC.loader
process_books = importlib.util.module_from_spec(PROCESS_BOOKS_SPEC)
PROCESS_BOOKS_SPEC.loader.exec_module(process_books)


class LayoutSanitizeIntegrationTests(unittest.TestCase):
    def test_load_layout_sanitize_page_map_uses_existing_sanitized_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "page_0001.png"
            Image.new("RGB", (16, 16), color=(255, 255, 255)).save(image_path)
            json_path = tmp_path / "demo.layout_ocr.json"
            payload = {
                "pages": [
                    {"page_number": 1, "sanitized_image_path": image_path.as_posix()},
                    {"page_number": 2, "sanitized_image_path": (tmp_path / "missing.png").as_posix()},
                    {"page_number": "bad", "sanitized_image_path": image_path.as_posix()},
                ]
            }
            json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            page_map = process_books.load_layout_sanitize_page_map(json_path)

            self.assertEqual(page_map, {1: image_path.as_posix()})

    def test_load_layout_sanitize_layout_map_uses_page_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            json_path = tmp_path / "demo.layout_ocr.json"
            payload = {
                "pages": [
                    {"page_number": 1, "regions": [{"action": "mask"}]},
                    {"page_number": 2, "regions": [{"action": "keep"}]},
                    {"page_number": "bad", "regions": []},
                ]
            }
            json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            page_map = process_books.load_layout_sanitize_layout_map(json_path)

            self.assertEqual(sorted(page_map.keys()), [1, 2])
            self.assertEqual(page_map[1]["regions"][0]["action"], "mask")

    def test_ocr_agent_forces_ocr_on_sanitized_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pdf_path = tmp_path / "sample.pdf"
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), "Пример текста с примечанием 1)")
            doc.save(pdf_path)
            doc.close()

            sanitized_path = tmp_path / "page_0001.png"
            Image.new("RGB", (128, 128), color=(255, 255, 255)).save(sanitized_path)

            agent = OCRAgent(OCRAgentConfig(backend="qwen"))
            called: dict[str, str | None] = {}

            def fake_run_ocr(page, render_scale=None, image_path=None):
                called["image_path"] = str(image_path) if image_path is not None else None
                return "Очищенный основной текст"

            agent._run_ocr = fake_run_ocr  # type: ignore[method-assign]

            with fitz.open(pdf_path) as reopened:
                result = agent._process_pdf_page(
                    reopened[0],
                    page_number=1,
                    route_hint="pdf_mixed_extract_plus_ocr",
                    sanitized_image_path=sanitized_path,
                    layout_sanitize_backend="paddle",
                )

            self.assertEqual(result.source, "ocr")
            self.assertEqual(result.selected_text, "Очищенный основной текст")
            self.assertEqual(result.layout_sanitize_backend, "paddle")
            self.assertEqual(result.sanitized_image_path, sanitized_path.as_posix())
            self.assertEqual(called["image_path"], sanitized_path.as_posix())
            self.assertIn("layout_sanitized=paddle", result.notes)

    def test_ocr_agent_bypasses_ocr_for_sanitized_easy_extract_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pdf_path = tmp_path / "sample.pdf"
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text(
                (72, 72),
                (
                    "Это страница с хорошим текстовым слоем и небольшим примечанием 1). "
                    "Основной текст достаточно длинный, чтобы командир уверенно оставил страницу на extract. "
                    "Мы специально добавляем еще несколько предложений про историю, археологию и источники, "
                    "чтобы объем текста был выше порога и не вызывал переход на OCR."
                ),
            )
            doc.save(pdf_path)
            doc.close()

            sanitized_path = tmp_path / "page_0001.png"
            Image.new("RGB", (128, 128), color=(255, 255, 255)).save(sanitized_path)

            agent = OCRAgent(OCRAgentConfig(backend="qwen"))
            agent.commander.plan_ocr_page = lambda **kwargs: OCRPlan(  # type: ignore[method-assign]
                source="extract",
                render_scale=2.2,
                difficulty="easy_extract",
                reason="test_easy_extract",
            )

            def fake_run_ocr(page, render_scale=None, image_path=None):
                raise AssertionError("sanitized easy-extract page should not call OCR")

            agent._run_ocr = fake_run_ocr  # type: ignore[method-assign]

            with fitz.open(pdf_path) as reopened:
                result = agent._process_pdf_page(
                    reopened[0],
                    page_number=1,
                    route_hint="pdf_mixed_extract_plus_ocr",
                    sanitized_image_path=sanitized_path,
                    layout_sanitize_backend="paddle",
                )

            self.assertEqual(result.source, "extract")
            self.assertEqual(result.layout_sanitize_backend, "paddle")
            self.assertEqual(result.sanitized_image_path, sanitized_path.as_posix())
            self.assertIn("sanitized_extract_bypass_used", result.notes)
            self.assertNotIn("sanitized_ocr_empty_used_extract_fallback", result.notes)
            self.assertGreater(result.extracted_char_count, 0)

    def test_ocr_agent_filters_masked_extract_blocks_on_sanitized_easy_extract_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pdf_path = tmp_path / "sample.pdf"
            doc = fitz.open()
            page = doc.new_page(width=200, height=200)
            page.insert_text((20, 20), "HEADER TITLE")
            page.insert_text((20, 42), "952")
            page.insert_text((20, 80), "Body text should stay after layout filtering.")
            doc.save(pdf_path)
            doc.close()

            sanitized_path = tmp_path / "page_0001.png"
            Image.new("RGB", (200, 200), color=(255, 255, 255)).save(sanitized_path)

            layout_payload = {
                "page_number": 1,
                "width": 200,
                "height": 200,
                "regions": [
                    {"bbox": [15, 8, 190, 28], "action": "mask", "mapped_label": "note"},
                    {"bbox": [15, 30, 50, 48], "action": "mask", "mapped_label": "note"},
                    {"bbox": [15, 65, 195, 100], "action": "keep", "mapped_label": "body"},
                ],
            }

            agent = OCRAgent(OCRAgentConfig(backend="qwen"))
            agent.commander.plan_ocr_page = lambda **kwargs: OCRPlan(  # type: ignore[method-assign]
                source="extract",
                render_scale=2.2,
                difficulty="easy_extract",
                reason="test_easy_extract",
            )

            def fake_run_ocr(page, render_scale=None, image_path=None):
                raise AssertionError("filtered easy-extract path should not call OCR")

            agent._run_ocr = fake_run_ocr  # type: ignore[method-assign]

            with fitz.open(pdf_path) as reopened:
                result = agent._process_pdf_page(
                    reopened[0],
                    page_number=1,
                    route_hint="pdf_mixed_extract_plus_ocr",
                    sanitized_image_path=sanitized_path,
                    layout_sanitize_backend="paddle",
                    sanitized_layout_payload=layout_payload,
                )

            self.assertEqual(result.source, "extract")
            self.assertIn("sanitized_extract_bypass_used", result.notes)
            self.assertIn("layout_extract_filter_applied:kept=1,removed=2", result.notes)
            self.assertNotIn("HEADER TITLE", result.selected_text)
            self.assertNotIn("952", result.selected_text)
            self.assertIn("Body text should stay", result.selected_text)


if __name__ == "__main__":
    unittest.main()

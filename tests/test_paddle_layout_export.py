from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from russian_data_cleaning.paddle_layout_baseline.export import build_final_text, export_document_result
from russian_data_cleaning.paddle_layout_baseline.types import DocumentLayoutResult, LayoutRegion, PageLayoutResult


class PaddleLayoutExportTests(unittest.TestCase):
    def test_build_final_text_keeps_only_title_and_body(self) -> None:
        document = DocumentLayoutResult(
            source_path="/tmp/sample.pdf",
            source_type="pdf",
            pages=[
                PageLayoutResult(
                    page_id="sample:page_1",
                    page_number=1,
                    source_path="/tmp/sample.pdf",
                    source_type="pdf",
                    width=100,
                    height=100,
                    regions=[
                        LayoutRegion("sample:page_1", [0, 0, 10, 10], "title", "title", "ocr", 0.9, "TITLE", 0),
                        LayoutRegion("sample:page_1", [0, 10, 10, 20], "text", "body", "keep", 0.8, "BODY", 1),
                        LayoutRegion("sample:page_1", [0, 20, 10, 30], "footnote", "note", "mask", 0.7, "NOTE", 2),
                    ],
                )
            ],
        )
        self.assertEqual(build_final_text(document), "TITLE\nBODY")

    def test_export_writes_json_and_txt(self) -> None:
        document = DocumentLayoutResult(
            source_path="/tmp/sample.pdf",
            source_type="pdf",
            pages=[
                PageLayoutResult(
                    page_id="sample:page_1",
                    page_number=1,
                    source_path="/tmp/sample.pdf",
                    source_type="pdf",
                    width=100,
                    height=100,
                    regions=[
                        LayoutRegion("sample:page_1", [0, 0, 10, 10], "title", "title", "ocr", 0.9, "TITLE", 0),
                        LayoutRegion("sample:page_1", [0, 10, 10, 20], "text", "body", "keep", 0.8, "BODY", 1),
                    ],
                )
            ],
            final_text="TITLE\nBODY",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, txt_path = export_document_result(document, temp_dir)
            self.assertTrue(json_path.exists())
            self.assertTrue(txt_path.exists())
            self.assertEqual(txt_path.read_text(encoding="utf-8").strip(), "TITLE\nBODY")
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["page_count"], 1)
            self.assertEqual(payload["final_text"], "TITLE\nBODY")

    def test_export_writes_sanitized_pages_without_txt_when_no_ocr(self) -> None:
        document = DocumentLayoutResult(
            source_path="/tmp/sample.pdf",
            source_type="pdf",
            pages=[
                PageLayoutResult(
                    page_id="sample:page_1",
                    page_number=1,
                    source_path="/tmp/sample.pdf",
                    source_type="pdf",
                    width=4,
                    height=4,
                    regions=[
                        LayoutRegion("sample:page_1", [0, 0, 2, 2], "text", "body", "keep", 0.8, "", 0),
                        LayoutRegion("sample:page_1", [2, 0, 4, 2], "table", "table", "mask", 0.7, "", 1),
                    ],
                )
            ],
        )
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[:, 2:, :] = 99
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, txt_path = export_document_result(
                document,
                temp_dir,
                sanitized_pages={"sample:page_1": image},
            )
            self.assertTrue(json_path.exists())
            self.assertIsNone(txt_path)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertIn("sanitized_pages_dir", payload)
            self.assertTrue(payload["pages"][0]["sanitized_image_path"])


if __name__ == "__main__":
    unittest.main()

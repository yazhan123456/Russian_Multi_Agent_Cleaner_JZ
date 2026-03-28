from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import fitz


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from russian_data_cleaning.pdf_splitter import should_split_page, split_landscape_pdf


class SplitLandscapePdfTests(unittest.TestCase):
    def test_should_split_page_uses_aspect_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "demo.pdf"
            doc = fitz.open()
            doc.new_page(width=300, height=600)
            doc.new_page(width=600, height=300)
            doc.save(pdf_path)
            doc.close()

            with fitz.open(pdf_path) as reopened:
                self.assertFalse(should_split_page(reopened[0], aspect_ratio_threshold=1.35))
                self.assertTrue(should_split_page(reopened[1], aspect_ratio_threshold=1.35))

    def test_split_landscape_pdf_preserves_order_and_splits_wide_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pdf_path = tmp_path / "source.pdf"
            out_path = tmp_path / "split.pdf"

            doc = fitz.open()
            portrait = doc.new_page(width=300, height=600)
            portrait.insert_text((72, 72), "PORTRAIT")
            landscape = doc.new_page(width=600, height=300)
            landscape.insert_text((50, 72), "LEFT")
            landscape.insert_text((350, 72), "RIGHT")
            doc.save(pdf_path)
            doc.close()

            summary = split_landscape_pdf(pdf_path, out_path)

            self.assertEqual(summary.input_pages, 2)
            self.assertEqual(summary.output_pages, 3)
            self.assertEqual(summary.split_pages, 1)
            self.assertEqual(summary.copied_pages, 1)

            with fitz.open(out_path) as result:
                self.assertEqual(result.page_count, 3)
                self.assertAlmostEqual(result[0].rect.width, 300, delta=1)
                self.assertAlmostEqual(result[0].rect.height, 600, delta=1)
                self.assertAlmostEqual(result[1].rect.width, 300, delta=1)
                self.assertAlmostEqual(result[1].rect.height, 300, delta=1)
                self.assertAlmostEqual(result[2].rect.width, 300, delta=1)
                self.assertAlmostEqual(result[2].rect.height, 300, delta=1)
                self.assertIn("PORTRAIT", result[0].get_text())
                self.assertIn("LEFT", result[1].get_text())
                self.assertIn("RIGHT", result[2].get_text())


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from russian_data_cleaning.paddle_layout_baseline import (  # noqa: E402
    PaddleLayoutOCRAgent,
    PaddleLayoutOCRConfig,
    export_document_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Paddle PP-Structure layout routing OCR on a PDF or image.")
    parser.add_argument("--input", required=True, help="Path to a PDF, PNG, JPG, or JPEG file.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "paddle_layout_ocr",
        help="Directory for JSON and TXT outputs.",
    )
    parser.add_argument(
        "--render-scale",
        type=float,
        default=2.0,
        help="PDF render scale before layout detection/OCR.",
    )
    parser.add_argument(
        "--ocr-lang",
        default="ru",
        help="PaddleOCR recognition language. For Russian use: ru. Old alias cyrillic is also accepted.",
    )
    parser.add_argument(
        "--with-paddle-ocr",
        action="store_true",
        help="Also run Paddle OCR on kept title/body regions. Default is off; sanitized pages are meant for the existing OCR agent.",
    )
    parser.add_argument(
        "--mask-fill",
        type=int,
        default=255,
        help="Fill value for masked note/picture/table regions in sanitized pages. Default is white (255).",
    )
    parser.add_argument(
        "--layout-score-threshold",
        type=float,
        default=0.0,
        help="Minimum layout confidence to keep a detected region.",
    )
    parser.add_argument(
        "--show-log",
        action="store_true",
        help="Show Paddle logs.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable Paddle GPU if available. Default is CPU for Apple Silicon compatibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = PaddleLayoutOCRAgent(
        PaddleLayoutOCRConfig(
            render_scale=args.render_scale,
            ocr_lang=args.ocr_lang,
            show_log=args.show_log,
            use_gpu=args.use_gpu,
            layout_score_threshold=args.layout_score_threshold,
            perform_region_ocr=args.with_paddle_ocr,
            mask_fill=args.mask_fill,
        )
    )
    def progress(index: int, total: int, page_number: int) -> None:
        print(f"Layout sanitize {index}/{total} (page {page_number})", flush=True)

    document, sanitized_pages = agent.process_document(args.input, progress_callback=progress)
    json_path, txt_path = export_document_result(document, args.out_dir, sanitized_pages=sanitized_pages)
    print(f"JSON: {json_path}", flush=True)
    if document.sanitized_pages_dir:
        print(f"SANITIZED_PAGES: {document.sanitized_pages_dir}", flush=True)
    if txt_path:
        print(f"TXT:  {txt_path}", flush=True)
    print(f"Pages: {len(document.pages)}", flush=True)
    print(f"Chars: {len(document.final_text)}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from russian_data_cleaning.pdf_splitter import split_landscape_pdf  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split wide landscape PDF pages into two portrait-style pages before OCR/layout processing."
    )
    parser.add_argument("--input", required=True, help="Input PDF path.")
    parser.add_argument("--output", help="Output PDF path. Defaults to <input_stem>_split.pdf next to the input.")
    parser.add_argument(
        "--aspect-ratio-threshold",
        type=float,
        default=1.35,
        help="Split pages whose width/height ratio is at least this value. Default: 1.35",
    )
    parser.add_argument(
        "--split-order",
        choices=["left-right", "right-left"],
        default="left-right",
        help="Reading order for split pages. Default: left-right",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_split.pdf")
    summary = split_landscape_pdf(
        input_path,
        output_path,
        aspect_ratio_threshold=args.aspect_ratio_threshold,
        split_order=args.split_order,
    )
    print(f"INPUT:  {summary.input_path}")
    print(f"OUTPUT: {summary.output_path}")
    print(f"INPUT_PAGES:  {summary.input_pages}")
    print(f"OUTPUT_PAGES: {summary.output_pages}")
    print(f"SPLIT_PAGES:  {summary.split_pages}")
    print(f"COPIED_PAGES: {summary.copied_pages}")
    print(f"SPLIT_ORDER:  {summary.split_order}")
    print(f"THRESHOLD:    {summary.aspect_ratio_threshold}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from russian_data_cleaning import CleaningAgent, GeminiReviewAgent, OCRAgent, ReviewAgent  # noqa: E402
from russian_data_cleaning.ocr_agent import OCRAgentConfig  # noqa: E402
from russian_data_cleaning.gemini_review import GeminiReviewConfig  # noqa: E402


TRIAL_BOOKS = [
    {
        "category": "extract_then_clean",
        "relative_path": "4/The Political System of the USA and the Russian Federation (Политическая система США и Российской Федерации) электронное.pdf",
        "route_hint": "pdf_extract_then_clean",
        "reason": "digital bilingual PDF with TOC-like front matter",
    },
    {
        "category": "extract_then_clean",
        "relative_path": "14/Сталин и народ. Почему не было восстания (Виктор Земсков).pdf",
        "route_hint": "pdf_extract_then_clean",
        "reason": "digital Russian PDF with suspicious encoding artifacts in extracted text",
    },
    {
        "category": "mixed_extract_plus_ocr",
        "relative_path": "1/Early Russian History. Key Issues Учебно-методическое пособие по английскому языку.pdf",
        "route_hint": "pdf_mixed_extract_plus_ocr",
        "reason": "mixed bilingual educational PDF with some extractable pages",
    },
    {
        "category": "mixed_extract_plus_ocr",
        "relative_path": "10/Мои первые дни в Белом доме.pdf",
        "route_hint": "pdf_mixed_extract_plus_ocr",
        "reason": "mixed PDF where front matter is image-only and body pages have text layer",
    },
    {
        "category": "ocr_then_clean",
        "relative_path": "5/Государственные займы Российской империи 1798-1917 годов.pdf",
        "route_hint": "pdf_ocr_then_clean",
        "reason": "image-only Russian scan with tabular and numeric content",
    },
    {
        "category": "ocr_then_clean",
        "relative_path": "4/Trekhetazhnyi amerikanets Stalina. Tank M3 General Li (in Russian).pdf",
        "route_hint": "pdf_ocr_then_clean",
        "reason": "image-only scan for second OCR-only control sample",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the terminal-first OCR/clean/review pilot.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs" / "book_manifest.csv",
        help="Manifest CSV with route hints.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "trial_runs",
        help="Directory for trial run outputs.",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "tesseract", "extract_only", "gemini", "google_documentai"],
        help="OCR backend. `auto` uses extraction when appropriate and Tesseract otherwise.",
    )
    parser.add_argument(
        "--review-backend",
        default="heuristic",
        choices=["heuristic", "gemini"],
        help="Review backend. `gemini` adds LLM review for risky pages.",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-2.5-flash",
        help="Gemini model for OCR or review when those backends are selected.",
    )
    parser.add_argument(
        "--max-pages-per-book",
        type=int,
        default=6,
        help="How many representative pages to process per book when not using --all-pages.",
    )
    parser.add_argument(
        "--all-pages",
        action="store_true",
        help="Process the entire book instead of representative pages.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return {row["relative_path"]: row for row in csv.DictReader(fh)}


def choose_pages(page_count: int, max_pages: int) -> list[int]:
    candidates = [1, 2, 3, 5, 10, max(1, page_count // 2), page_count]
    pages = []
    for page in candidates:
        if 1 <= page <= page_count and page not in pages:
            pages.append(page)
    return pages[:max_pages]


def build_book_summary(
    book: dict[str, str],
    ocr_doc: dict[str, Any],
    cleaned_doc: dict[str, Any],
    reviewed_doc: dict[str, Any],
    selected_pages: list[int],
    gemini_review_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_counts = Counter(page["source"] for page in ocr_doc["pages"])
    verdict_counts = Counter(page["page_verdict"] for page in reviewed_doc["pages"])
    flag_count = sum(len(page["flags"]) for page in cleaned_doc["pages"])
    edit_count = sum(len(page["edits"]) for page in cleaned_doc["pages"])
    protected_count = sum(len(page["protected_hits"]) for page in cleaned_doc["pages"])
    gemini_verdict_counts = None
    if gemini_review_doc is not None:
        gemini_verdict_counts = dict(Counter(page["llm_verdict"] for page in gemini_review_doc["pages"]))

    summary = {
        "category": book["category"],
        "relative_path": book["relative_path"],
        "route_hint": book["route_hint"],
        "reason": book["reason"],
        "selected_pages": selected_pages,
        "source_counts": dict(source_counts),
        "verdict_counts": dict(verdict_counts),
        "edit_count": edit_count,
        "flag_count": flag_count,
        "protected_hit_count": protected_count,
    }
    if gemini_verdict_counts is not None:
        summary["gemini_verdict_counts"] = gemini_verdict_counts
    return summary


def safe_stem_from_relative_path(relative_path: str) -> str:
    stem = Path(relative_path).stem[:120]
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem).strip("_") or "book"


def write_unified_text_export(
    export_dir: Path,
    book: dict[str, str],
    ocr_doc: dict[str, Any],
    cleaned_doc: dict[str, Any],
    reviewed_doc: dict[str, Any],
    gemini_review_doc: dict[str, Any] | None = None,
) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = safe_stem_from_relative_path(book["relative_path"])
    out_path = export_dir / f"{safe_stem}.txt"

    cleaned_map = {page["page_number"]: page for page in cleaned_doc["pages"]}
    review_map = {page["page_number"]: page for page in reviewed_doc["pages"]}
    gemini_map = {}
    if gemini_review_doc is not None:
        gemini_map = {page["page_number"]: page for page in gemini_review_doc["pages"]}

    lines = [
        f"Book: {book['relative_path']}",
        f"Category: {book['category']}",
        f"Route hint: {book['route_hint']}",
        f"Reason: {book['reason']}",
        "",
    ]

    for ocr_page in ocr_doc["pages"]:
        page_number = ocr_page["page_number"]
        cleaned_page = cleaned_map.get(page_number, {})
        review_page = review_map.get(page_number, {})
        gemini_page = gemini_map.get(page_number, {})

        lines.extend(
            [
                f"===== Page {page_number} =====",
                f"Source: {ocr_page.get('source', 'unknown')}",
                "",
                "[RAW]",
                ocr_page.get("selected_text", "") or "",
                "",
                "[CLEANED]",
                cleaned_page.get("cleaned_text", "") or "",
                "",
                "[HEURISTIC REVIEW]",
                f"Verdict: {review_page.get('page_verdict', 'n/a')}",
                f"Deletion ratio: {review_page.get('deletion_ratio', 'n/a')}",
            ]
        )

        review_records = review_page.get("review_records", [])
        if review_records:
            lines.append("Top review records:")
            for record in review_records[:8]:
                lines.append(
                    f"- {record.get('kind')} | {record.get('target_rule_id')} | {record.get('verdict')} | {record.get('detail')}"
                )

        flags = cleaned_page.get("flags", [])
        if flags:
            lines.append("Flags:")
            for flag in flags[:8]:
                lines.append(f"- {flag.get('rule_id')}: {flag.get('evidence')}")

        protected_hits = cleaned_page.get("protected_hits", [])
        if protected_hits:
            lines.append("Protected hits:")
            for hit in protected_hits[:8]:
                lines.append(f"- {hit.get('rule_id')}: {hit.get('evidence')}")

        if gemini_page:
            lines.extend(
                [
                    "",
                    "[GEMINI REVIEW]",
                    f"Verdict: {gemini_page.get('llm_verdict', 'n/a')}",
                    f"Summary: {gemini_page.get('summary', '')}",
                ]
            )
            concerns = gemini_page.get("concerns", [])
            if concerns:
                lines.append("Concerns:")
                for concern in concerns[:8]:
                    lines.append(f"- {concern}")

        lines.extend(["", ""])

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out_path


def write_text_export_index(export_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"Generated at: {summary['generated_at']}",
        f"OCR backend: {summary['backend']}",
        f"Review backend: {summary['review_backend']}",
        "",
        "Books:",
    ]
    for book in summary["books"]:
        lines.append(f"- {book['relative_path']}")
    (export_dir / "_index.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cleaned_text_export(
    export_dir: Path,
    book: dict[str, str],
    cleaned_doc: dict[str, Any],
) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = safe_stem_from_relative_path(book["relative_path"])
    out_path = export_dir / f"{safe_stem}.txt"

    lines = []
    for page in cleaned_doc["pages"]:
        text = (page.get("cleaned_text") or "").strip()
        if not text:
            continue
        lines.append(f"===== Page {page['page_number']} =====")
        lines.append(text)
        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out_path


def write_cleaned_text_index(export_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"Generated at: {summary['generated_at']}",
        f"OCR backend: {summary['backend']}",
        f"Review backend: {summary['review_backend']}",
        "",
        "Cleaned text files:",
    ]
    for book in summary["books"]:
        lines.append(f"- {book['relative_path']}")
    (export_dir / "_index.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_summary(path: Path, run_summary: dict[str, Any]) -> None:
    lines = [
        "# Trial Run Summary",
        "",
        f"- Generated at: `{run_summary['generated_at']}`",
        f"- OCR backend: `{run_summary['backend']}`",
        f"- Review backend: `{run_summary['review_backend']}`",
        f"- Books processed: `{len(run_summary['books'])}`",
        "",
    ]

    for book in run_summary["books"]:
        lines.append(f"## {book['relative_path']}")
        lines.append(f"- Category: `{book['category']}`")
        lines.append(f"- Route: `{book['route_hint']}`")
        lines.append(f"- Pages: `{book['selected_pages']}`")
        lines.append(f"- Sources: `{book['source_counts']}`")
        lines.append(f"- Verdicts: `{book['verdict_counts']}`")
        if "gemini_verdict_counts" in book:
            lines.append(f"- Gemini verdicts: `{book['gemini_verdict_counts']}`")
        lines.append(f"- Edits: `{book['edit_count']}`")
        lines.append(f"- Flags: `{book['flag_count']}`")
        lines.append(f"- Protected hits: `{book['protected_hit_count']}`")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / f"terminal_pilot_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    text_export_dir = ROOT / "outputs" / "text_exports"
    text_export_dir.mkdir(parents=True, exist_ok=True)
    cleaned_text_dir = ROOT / "outputs" / "cleaned_txt"
    cleaned_text_dir.mkdir(parents=True, exist_ok=True)

    ocr_agent = OCRAgent(OCRAgentConfig(backend=args.backend))
    cleaning_agent = CleaningAgent()
    review_agent = ReviewAgent()
    gemini_review_agent = None
    if args.backend == "gemini":
        ocr_agent = OCRAgent(OCRAgentConfig(backend=args.backend, gemini_model=args.gemini_model))
    if args.review_backend == "gemini":
        gemini_review_agent = GeminiReviewAgent(GeminiReviewConfig(model=args.gemini_model, risky_only=True))

    book_summaries = []

    for book in TRIAL_BOOKS:
        relative_path = book["relative_path"]
        pdf_path = ROOT / relative_path
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing trial book: {pdf_path}")

        manifest_row = manifest.get(relative_path, {})
        route_hint = manifest_row.get("route", book["route_hint"]) or book["route_hint"]

        import fitz

        doc = fitz.open(pdf_path)
        page_count = len(doc)
        selected_pages = list(range(1, page_count + 1)) if args.all_pages else choose_pages(page_count, args.max_pages_per_book)

        ocr_doc = ocr_agent.process_pdf(pdf_path, pages=selected_pages, route_hint=route_hint)
        cleaned_doc = cleaning_agent.process_document(ocr_doc)
        reviewed_doc = review_agent.review_document(ocr_doc, cleaned_doc)
        gemini_review_doc = None
        if gemini_review_agent is not None:
            gemini_review_doc = gemini_review_agent.review_document(cleaned_doc, reviewed_doc)

        safe_stem = safe_stem_from_relative_path(relative_path)
        book_dir = run_dir / safe_stem
        book_dir.mkdir(parents=True, exist_ok=True)
        (book_dir / "ocr.json").write_text(json.dumps(ocr_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (book_dir / "cleaned.json").write_text(json.dumps(cleaned_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (book_dir / "review.json").write_text(json.dumps(reviewed_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if gemini_review_doc is not None:
            (book_dir / "gemini_review.json").write_text(
                json.dumps(gemini_review_doc, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        write_unified_text_export(
            text_export_dir,
            book,
            ocr_doc,
            cleaned_doc,
            reviewed_doc,
            gemini_review_doc=gemini_review_doc,
        )
        write_cleaned_text_export(cleaned_text_dir, book, cleaned_doc)

        book_summaries.append(
            build_book_summary(
                book,
                ocr_doc,
                cleaned_doc,
                reviewed_doc,
                selected_pages,
                gemini_review_doc=gemini_review_doc,
            )
        )

    run_summary = {
        "generated_at": timestamp,
        "backend": args.backend,
        "review_backend": args.review_backend,
        "books": book_summaries,
    }
    (run_dir / "summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown_summary(run_dir / "summary.md", run_summary)
    write_text_export_index(text_export_dir, run_summary)
    write_cleaned_text_index(cleaned_text_dir, run_summary)

    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to {run_dir}")
    print(f"Unified text exports written to {text_export_dir}")
    print(f"Cleaned text exports written to {cleaned_text_dir}")


if __name__ == "__main__":
    main()

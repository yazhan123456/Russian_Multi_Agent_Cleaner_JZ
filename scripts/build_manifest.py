#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from pypdf import PdfReader


CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
FOOTNOTE_RE = re.compile(r"(?:\[\d{1,3}\]|\(\d{1,3}\)|\b\d{1,3}\)(?=\s+\S))")
DOT_LEADER_RE = re.compile(r"\.{4,}")
SHORT_LINE_LIMIT = 48
STRONG_TEXT_CHARS = 400
NONEMPTY_TEXT_CHARS = 40


@dataclass
class BookRecord:
    relative_path: str
    extension: str
    top_level_group: str
    file_size_mb: float
    page_count: int | None
    sample_pages: str
    sample_text_chars: int | None
    sample_nonempty_pages: int | None
    avg_chars_per_sampled_page: float | None
    text_layer_status: str
    route: str
    needs_google_ocr: bool
    needs_layout_review: bool
    layout_risk: str
    review_priority: str
    name_has_cyrillic: bool
    name_has_latin: bool
    bilingual_name_hint: bool
    possible_multi_column: bool
    possible_footnotes: bool
    possible_toc_or_index: bool
    notes: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a processing manifest for Russian OCR cleaning inputs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Workspace root containing PDFs and EPUBs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for CSV, JSONL, and summary outputs.",
    )
    parser.add_argument(
        "--sample-page-cap",
        type=int,
        default=7,
        help="Maximum number of PDF pages to sample for text-layer heuristics.",
    )
    return parser.parse_args()


def has_cyrillic(text: str) -> bool:
    return bool(CYRILLIC_RE.search(text))


def has_latin(text: str) -> bool:
    return bool(LATIN_RE.search(text))


def clean_sample_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sample_page_indices(page_count: int, cap: int) -> list[int]:
    if page_count <= 0:
        return []

    candidates = {
        0,
        1,
        2,
        page_count // 4,
        page_count // 2,
        (3 * page_count) // 4,
        page_count - 3,
        page_count - 2,
        page_count - 1,
    }
    indices = [index for index in sorted(candidates) if 0 <= index < page_count]
    if len(indices) <= cap:
        return indices

    step = max(1, len(indices) // cap)
    trimmed = indices[::step][:cap]
    if indices[-1] not in trimmed:
        trimmed[-1] = indices[-1]
    return sorted(set(trimmed))


def extract_page_text(reader: PdfReader, page_index: int) -> str:
    page = reader.pages[page_index]
    text = page.extract_text() or ""
    return clean_sample_text(text)


def analyze_pdf(path: Path, sample_page_cap: int) -> dict[str, Any]:
    analysis: dict[str, Any] = {
        "page_count": None,
        "sample_pages": [],
        "sample_text_chars": None,
        "sample_nonempty_pages": None,
        "avg_chars_per_sampled_page": None,
        "text_layer_status": "unknown",
        "route": "manual_check",
        "needs_google_ocr": False,
        "needs_layout_review": True,
        "layout_risk": "high",
        "review_priority": "high",
        "possible_multi_column": False,
        "possible_footnotes": False,
        "possible_toc_or_index": False,
        "notes": [],
        "error": "",
    }

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                analysis["error"] = "encrypted_pdf"
                analysis["notes"].append("encrypted_pdf")
                return analysis

        page_count = len(reader.pages)
        analysis["page_count"] = page_count

        sample_indices = sample_page_indices(page_count, sample_page_cap)
        analysis["sample_pages"] = [index + 1 for index in sample_indices]

        extracted = []
        line_lengths: list[int] = []
        footnote_hits = 0
        toc_hits = 0

        for index in sample_indices:
            try:
                text = extract_page_text(reader, index)
            except Exception:
                text = ""

            extracted.append(text)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            line_lengths.extend(len(line) for line in lines)
            footnote_hits += len(FOOTNOTE_RE.findall(text))
            toc_hits += len(DOT_LEADER_RE.findall(text))

        char_counts = [len(text.replace(" ", "")) for text in extracted]
        nonempty_pages = sum(count >= NONEMPTY_TEXT_CHARS for count in char_counts)
        strong_pages = sum(count >= STRONG_TEXT_CHARS for count in char_counts)
        total_chars = sum(char_counts)

        analysis["sample_text_chars"] = total_chars
        analysis["sample_nonempty_pages"] = nonempty_pages
        if sample_indices:
            analysis["avg_chars_per_sampled_page"] = round(
                total_chars / len(sample_indices), 1
            )

        if not sample_indices:
            analysis["text_layer_status"] = "unknown"
        elif nonempty_pages == 0:
            analysis["text_layer_status"] = "image_only"
        elif nonempty_pages == len(sample_indices):
            analysis["text_layer_status"] = (
                "text_rich" if strong_pages >= max(1, len(sample_indices) // 2) else "text_sparse"
            )
        else:
            analysis["text_layer_status"] = "mixed"

        short_line_ratio = 0.0
        if line_lengths:
            short_line_ratio = sum(length <= SHORT_LINE_LIMIT for length in line_lengths) / len(
                line_lengths
            )

        analysis["possible_multi_column"] = bool(
            line_lengths
            and statistics.median(line_lengths) <= SHORT_LINE_LIMIT
            and short_line_ratio >= 0.58
        )
        analysis["possible_footnotes"] = footnote_hits >= max(2, len(sample_indices))
        analysis["possible_toc_or_index"] = toc_hits >= 2

        if analysis["text_layer_status"] in {"text_rich", "text_sparse"}:
            analysis["route"] = "pdf_extract_then_clean"
            analysis["needs_google_ocr"] = False
        elif analysis["text_layer_status"] == "mixed":
            analysis["route"] = "pdf_mixed_extract_plus_ocr"
            analysis["needs_google_ocr"] = True
            analysis["notes"].append("partial_text_layer")
        else:
            analysis["route"] = "pdf_ocr_then_clean"
            analysis["needs_google_ocr"] = True

        layout_score = 0
        if analysis["text_layer_status"] == "mixed":
            layout_score += 2
        if analysis["possible_multi_column"]:
            layout_score += 2
            analysis["notes"].append("possible_multi_column")
        if analysis["possible_footnotes"]:
            layout_score += 1
            analysis["notes"].append("possible_footnotes")
        if analysis["possible_toc_or_index"]:
            layout_score += 1
            analysis["notes"].append("possible_toc_or_index")
        if page_count >= 500:
            layout_score += 1
            analysis["notes"].append("long_book")

        if layout_score <= 1:
            analysis["layout_risk"] = "low"
        elif layout_score <= 3:
            analysis["layout_risk"] = "medium"
        else:
            analysis["layout_risk"] = "high"

        analysis["needs_layout_review"] = analysis["layout_risk"] != "low"

        if analysis["route"] == "pdf_ocr_then_clean" and page_count >= 300:
            analysis["review_priority"] = "high"
        elif analysis["layout_risk"] == "high":
            analysis["review_priority"] = "high"
        elif analysis["route"] == "pdf_mixed_extract_plus_ocr":
            analysis["review_priority"] = "medium"
        elif analysis["layout_risk"] == "medium":
            analysis["review_priority"] = "medium"
        else:
            analysis["review_priority"] = "low"

    except Exception as exc:
        analysis["error"] = f"pdf_read_error:{type(exc).__name__}"
        analysis["notes"].append("pdf_read_error")

    return analysis


def find_epub_package(zf: zipfile.ZipFile) -> str | None:
    try:
        container = zf.read("META-INF/container.xml")
    except KeyError:
        return None

    root = ElementTree.fromstring(container)
    ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
    for rootfile in root.findall(".//c:rootfile", ns):
        full_path = rootfile.attrib.get("full-path")
        if full_path:
            return full_path
    return None


def analyze_epub(path: Path) -> dict[str, Any]:
    analysis: dict[str, Any] = {
        "page_count": None,
        "sample_pages": [],
        "sample_text_chars": None,
        "sample_nonempty_pages": None,
        "avg_chars_per_sampled_page": None,
        "text_layer_status": "epub_text",
        "route": "epub_extract_then_clean",
        "needs_google_ocr": False,
        "needs_layout_review": False,
        "layout_risk": "low",
        "review_priority": "low",
        "possible_multi_column": False,
        "possible_footnotes": False,
        "possible_toc_or_index": False,
        "notes": [],
        "error": "",
    }

    try:
        with zipfile.ZipFile(path) as zf:
            package_path = find_epub_package(zf)
            if not package_path:
                analysis["notes"].append("missing_container_or_opf")
                return analysis

            package_dir = Path(package_path).parent
            package = ElementTree.fromstring(zf.read(package_path))
            ns = {"opf": "http://www.idpf.org/2007/opf"}

            manifest_items = {
                item.attrib.get("id"): item.attrib.get("href", "")
                for item in package.findall(".//opf:manifest/opf:item", ns)
            }
            spine_ids = [
                item.attrib.get("idref", "")
                for item in package.findall(".//opf:spine/opf:itemref", ns)
            ]
            doc_count = sum(1 for item_id in spine_ids if manifest_items.get(item_id))
            analysis["notes"].append(f"epub_spine_docs:{doc_count}")

            sample_texts = []
            for item_id in spine_ids[:3]:
                href = manifest_items.get(item_id)
                if not href:
                    continue
                item_path = str((package_dir / href).as_posix())
                try:
                    raw = zf.read(item_path).decode("utf-8", errors="ignore")
                except KeyError:
                    continue
                sample_texts.append(clean_sample_text(re.sub(r"<[^>]+>", " ", raw)))

            char_counts = [len(text.replace(" ", "")) for text in sample_texts]
            if char_counts:
                analysis["sample_text_chars"] = sum(char_counts)
                analysis["sample_nonempty_pages"] = sum(count >= NONEMPTY_TEXT_CHARS for count in char_counts)
                analysis["avg_chars_per_sampled_page"] = round(
                    sum(char_counts) / len(char_counts), 1
                )

    except Exception as exc:
        analysis["error"] = f"epub_read_error:{type(exc).__name__}"
        analysis["notes"].append("epub_read_error")
        analysis["review_priority"] = "medium"

    return analysis


def build_record(root: Path, path: Path, sample_page_cap: int) -> BookRecord:
    rel_path = path.relative_to(root).as_posix()
    name = path.name
    extension = path.suffix.lower()
    group = rel_path.split("/", 1)[0] if "/" in rel_path else "."
    size_mb = round(path.stat().st_size / (1024 * 1024), 2)

    if extension == ".pdf":
        analysis = analyze_pdf(path, sample_page_cap)
    elif extension == ".epub":
        analysis = analyze_epub(path)
    else:
        analysis = {
            "page_count": None,
            "sample_pages": [],
            "sample_text_chars": None,
            "sample_nonempty_pages": None,
            "avg_chars_per_sampled_page": None,
            "text_layer_status": "unsupported",
            "route": "manual_check",
            "needs_google_ocr": False,
            "needs_layout_review": True,
            "layout_risk": "high",
            "review_priority": "high",
            "possible_multi_column": False,
            "possible_footnotes": False,
            "possible_toc_or_index": False,
            "notes": ["unsupported_extension"],
            "error": "unsupported_extension",
        }

    name_has_cyrillic = has_cyrillic(name)
    name_has_latin = has_latin(name)
    bilingual_name_hint = name_has_cyrillic and name_has_latin
    notes = list(analysis["notes"])

    if bilingual_name_hint:
        notes.append("bilingual_name_hint")
        if analysis["layout_risk"] == "low":
            analysis["layout_risk"] = "medium"
            analysis["needs_layout_review"] = True
        if analysis["review_priority"] == "low":
            analysis["review_priority"] = "medium"

    return BookRecord(
        relative_path=rel_path,
        extension=extension,
        top_level_group=group,
        file_size_mb=size_mb,
        page_count=analysis["page_count"],
        sample_pages=",".join(str(page) for page in analysis["sample_pages"]),
        sample_text_chars=analysis["sample_text_chars"],
        sample_nonempty_pages=analysis["sample_nonempty_pages"],
        avg_chars_per_sampled_page=analysis["avg_chars_per_sampled_page"],
        text_layer_status=analysis["text_layer_status"],
        route=analysis["route"],
        needs_google_ocr=analysis["needs_google_ocr"],
        needs_layout_review=analysis["needs_layout_review"],
        layout_risk=analysis["layout_risk"],
        review_priority=analysis["review_priority"],
        name_has_cyrillic=name_has_cyrillic,
        name_has_latin=name_has_latin,
        bilingual_name_hint=bilingual_name_hint,
        possible_multi_column=analysis["possible_multi_column"],
        possible_footnotes=analysis["possible_footnotes"],
        possible_toc_or_index=analysis["possible_toc_or_index"],
        notes=";".join(sorted(set(note for note in notes if note))),
        error=analysis["error"],
    )


def write_csv(path: Path, records: list[BookRecord]) -> None:
    fieldnames = list(asdict(records[0]).keys()) if records else list(BookRecord.__annotations__.keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_jsonl(path: Path, records: list[BookRecord]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def build_summary(root: Path, records: list[BookRecord]) -> dict[str, Any]:
    route_counts = Counter(record.route for record in records)
    priority_counts = Counter(record.review_priority for record in records)
    text_status_counts = Counter(record.text_layer_status for record in records)

    pdf_records = [record for record in records if record.extension == ".pdf"]
    epub_records = [record for record in records if record.extension == ".epub"]
    pdf_pages_total = sum(record.page_count or 0 for record in pdf_records)
    google_ocr_pages = sum(
        record.page_count or 0 for record in pdf_records if record.needs_google_ocr
    )

    top_ocr_candidates = sorted(
        [record for record in pdf_records if record.needs_google_ocr],
        key=lambda item: (item.page_count or 0, item.file_size_mb),
        reverse=True,
    )[:10]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root.resolve()),
        "book_count": len(records),
        "pdf_count": len(pdf_records),
        "epub_count": len(epub_records),
        "pdf_pages_total": pdf_pages_total,
        "estimated_google_ocr_pages": google_ocr_pages,
        "routes": dict(route_counts),
        "review_priority": dict(priority_counts),
        "text_layer_status": dict(text_status_counts),
        "top_google_ocr_candidates": [
            {
                "relative_path": record.relative_path,
                "page_count": record.page_count,
                "route": record.route,
                "layout_risk": record.layout_risk,
            }
            for record in top_ocr_candidates
        ],
    }


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Russian Data Cleaning Manifest Summary",
        "",
        f"- Generated at (UTC): `{summary['generated_at_utc']}`",
        f"- Root: `{summary['root']}`",
        f"- Books: `{summary['book_count']}`",
        f"- PDFs: `{summary['pdf_count']}`",
        f"- EPUBs: `{summary['epub_count']}`",
        f"- Total PDF pages: `{summary['pdf_pages_total']}`",
        f"- Estimated Google OCR pages: `{summary['estimated_google_ocr_pages']}`",
        "",
        "## Routes",
    ]
    for route, count in sorted(summary["routes"].items()):
        lines.append(f"- `{route}`: `{count}`")

    lines.extend(["", "## Review Priority"])
    for priority, count in sorted(summary["review_priority"].items()):
        lines.append(f"- `{priority}`: `{count}`")

    lines.extend(["", "## Text Layer Status"])
    for status, count in sorted(summary["text_layer_status"].items()):
        lines.append(f"- `{status}`: `{count}`")

    lines.extend(["", "## Largest OCR Candidates"])
    for item in summary["top_google_ocr_candidates"]:
        lines.append(
            f"- `{item['relative_path']}` | pages=`{item['page_count']}` | route=`{item['route']}` | layout_risk=`{item['layout_risk']}`"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    book_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".epub"}
    )

    records = [build_record(root, path, args.sample_page_cap) for path in book_paths]

    write_csv(out_dir / "book_manifest.csv", records)
    write_jsonl(out_dir / "book_manifest.jsonl", records)

    summary = build_summary(root, records)
    (out_dir / "book_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary_markdown(out_dir / "book_manifest_summary.md", summary)

    print(f"Wrote {len(records)} records to {out_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

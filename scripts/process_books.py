#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from russian_data_cleaning import (  # noqa: E402
    CleaningAgent,
    DeepSeekCleaningAgent,
    DeepSeekRepairAgent,
    DeepSeekStructureAgent,
    DocumentState,
    PageCheckpointStore,
    PageProcessingState,
    PageState,
    GeminiCleaningAgent,
    GeminiRepairAgent,
    GeminiReviewAgent,
    GeminiStructureAgent,
    OCRAgent,
    ReviewAgent,
    effective_state,
    mark_failed,
    state_at_least,
    transition,
)
from russian_data_cleaning.deepseek_cleaning_agent import DeepSeekCleaningConfig  # noqa: E402
from russian_data_cleaning.deepseek_repair_agent import DeepSeekRepairConfig  # noqa: E402
from russian_data_cleaning.deepseek_structure_agent import DeepSeekStructureConfig  # noqa: E402
from russian_data_cleaning.gemini_cleaning_agent import GeminiCleaningConfig  # noqa: E402
from russian_data_cleaning.gemini_repair_agent import GeminiRepairConfig  # noqa: E402
from russian_data_cleaning.gemini_review import GeminiReviewConfig  # noqa: E402
from russian_data_cleaning.gemini_structure_agent import GeminiStructureConfig  # noqa: E402
from russian_data_cleaning.ocr_agent import OCRAgentConfig  # noqa: E402
from russian_data_cleaning.russian_homoglyph_audit import audit_russian_homoglyphs  # noqa: E402
from russian_data_cleaning.page_commander import CommanderConfig, PageCommander  # noqa: E402


SENTENCE_END_RE = re.compile(r'[.!?…:;"»)\]]$')
LOWER_START_RE = re.compile(r"^[a-zа-яё]")
HEADING_LINE_RE = re.compile(r"^(?:[IVXLCDM]+\.\s+)?[A-ZА-ЯЁ0-9\s«»\"()\-]{8,}$")
WORD_TAIL_HYPHEN_RE = re.compile(r"([A-Za-zА-Яа-яЁё]+)[-\u2010\u2011]$")
WORD_HEAD_RE = re.compile(r"^([A-Za-zА-Яа-яЁё]+)(.*)$")
OCR_BULLET_LINE_RE = re.compile(r"^\s*[xхXХ]\s+(?=[A-Za-zА-Яа-яЁё])")
REVIEW_RISK_BY_VERDICT = {"approve": "low", "escalate": "medium", "reject": "high"}
RISK_ORDER = {"low": 0, "medium": 1, "high": 2}
POST_CLEAN_SCRIPT = ROOT / "scripts" / "post_clean_final_txt.py"
NONBODY_EXPORT_PAGE_TYPES = {"toc_or_index", "reference_only", "glossary_page", "publisher_meta", "empty"}
FRONTMATTER_EXPORT_RE = re.compile(
    r"(?i)\b(?:научный\s+совет|российская\s+национальная\s+библиотека|организация\s+российских\s+библиофилов|"
    r"материалы\s+[IVXLCDMА-ЯЁ0-9]+\s+(?:международной\s+)?научной\s+конференции|санкт[\s-]*петербург|"
    r"дом\s+ученых|редколлегия|книжные\s+памятники\s+и\s+библиофильство)\b"
)
BACKMATTER_EXPORT_RE = re.compile(
    r"(?i)\b(?:коротко\s+об\s+авторах|сведения\s+об\s+авторах|about\s+the\s+authors|isbn\b|удк\b|ббк\b|"
    r"подписано\s+в\s+печать|тираж\b|отпечатано\b)\b"
)
EXPORT_NOTE_LINE_RE = re.compile(r"^\s*(?:[0-9]{1,3}[.)]|\(?\d{1,3}\)|\[\d{1,3}\]|[°*†‡])\s*")
EXPORT_REFERENCE_CUE_RE = re.compile(
    r"(?i)(?:там\s+же|указ\.\s*соч\.|архитектурное\s+наследство|сборник|№\s*\d+|—\s*с\.\s*\d|"
    r"\bс\.\s*\d|//\s*http|https?://|isbn\b|вып\.\s*\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process full books and export plain cleaned TXT.")
    parser.add_argument(
        "--profile",
        default="custom",
        choices=["custom", "balanced_cost", "max_quality"],
        help="Pipeline preset. balanced_cost uses Qwen/DeepSeek/Gemini routing to control cost.",
    )
    parser.add_argument(
        "--book",
        action="append",
        dest="books",
        required=True,
        help="Relative path to a PDF or EPUB in the workspace. Repeat for multiple books.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs" / "book_manifest.csv",
        help="Manifest CSV with route hints.",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "tesseract", "extract_only", "gemini", "qwen", "google_documentai"],
        help="OCR backend.",
    )
    parser.add_argument(
        "--layout-sanitize-backend",
        default="none",
        choices=["none", "paddle"],
        help="Optional layout sanitizer run before OCR. Paddle keeps title/body and masks note/picture/table.",
    )
    parser.add_argument(
        "--layout-sanitize-python",
        type=Path,
        default=ROOT / ".venv-paddle310" / "bin" / "python",
        help="Python interpreter used to run the Paddle layout sanitizer subprocess.",
    )
    parser.add_argument(
        "--layout-sanitize-render-scale",
        type=float,
        default=2.0,
        help="PDF render scale used by the Paddle layout sanitizer.",
    )
    parser.add_argument(
        "--layout-sanitize-mask-fill",
        type=int,
        default=255,
        help="Fill value used to mask note/picture/table regions in sanitized pages.",
    )
    parser.add_argument(
        "--cleaning-backend",
        default="rules",
        choices=["rules", "gemini", "deepseek"],
        help="Cleaning backend.",
    )
    parser.add_argument(
        "--cleaning-escalation-backend",
        default="none",
        choices=["none", "gemini", "deepseek"],
        help="Optional second-pass cleaning backend for risky pages.",
    )
    parser.add_argument(
        "--review-backend",
        default="heuristic",
        choices=["heuristic", "gemini"],
        help="Review backend.",
    )
    parser.add_argument(
        "--repair-backend",
        default="none",
        choices=["none", "gemini", "deepseek"],
        help="Repair backend applied after review on risky pages.",
    )
    parser.add_argument(
        "--repair-escalation-backend",
        default="none",
        choices=["none", "gemini", "deepseek"],
        help="Optional second-pass repair backend for still-risky pages.",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-2.5-flash",
        help="Legacy fallback Gemini model for stages without a stage-specific model override.",
    )
    parser.add_argument(
        "--ocr-model",
        help="Stage-specific OCR model. For example: qwen-vl-ocr-latest or gemini-2.5-flash.",
    )
    parser.add_argument(
        "--ocr-render-scale",
        type=float,
        default=2.4,
        help="PDF render scale for image OCR. Higher values improve OCR fidelity but slow processing.",
    )
    parser.add_argument(
        "--force-ocr-body-pages",
        action="store_true",
        help="Force all non-skipped PDF body pages through OCR instead of trusting extracted text.",
    )
    parser.add_argument(
        "--cleaning-model",
        help="Stage-specific primary cleaning model.",
    )
    parser.add_argument(
        "--cleaning-escalation-model",
        help="Stage-specific second-pass cleaning model.",
    )
    parser.add_argument(
        "--review-model",
        help="Stage-specific review model.",
    )
    parser.add_argument(
        "--repair-model",
        help="Stage-specific primary repair model.",
    )
    parser.add_argument(
        "--repair-escalation-model",
        help="Stage-specific second-pass repair model.",
    )
    parser.add_argument(
        "--final-structure-model",
        help="Stage-specific final structure restoration model.",
    )
    parser.add_argument(
        "--final-structure-backend",
        default="rules",
        choices=["rules", "gemini", "deepseek"],
        help="Final full-text restoration backend.",
    )
    parser.add_argument(
        "--final-structure-risky-only",
        action="store_true",
        help="Run final structure restoration only on pages with heading/structure risk instead of the full book.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "outputs" / "full_book_runs",
        help="Directory for JSON run artifacts.",
    )
    parser.add_argument(
        "--final-txt-dir",
        type=Path,
        default=ROOT / "outputs" / "final_txt",
        help="Directory for final cleaned TXT files.",
    )
    parser.add_argument(
        "--notes-policy",
        default="delete",
        choices=["delete", "keep"],
        help="Whether to delete footnotes/endnotes during cleaning.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest run directory under --run-root, reusing per-page checkpoints when available.",
    )
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        help="Resume from a specific run directory instead of creating a new one.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=20,
        help="Emit a still-running progress heartbeat when a single page step takes this many seconds.",
    )
    parser.add_argument(
        "--prevent-sleep",
        action="store_true",
        help="On macOS, use caffeinate to prevent idle system sleep while the run is active.",
    )
    parser.add_argument(
        "--deepseek-max-concurrency",
        type=int,
        default=4,
        help="Maximum number of pages to send concurrently to DeepSeek-backed stages.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return {row["relative_path"]: row for row in csv.DictReader(fh)}


def safe_stem_from_relative_path(relative_path: str) -> str:
    stem = Path(relative_path).stem[:140]
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem).strip("_") or "book"


def stitch_cleaned_pages(cleaned_doc: dict[str, Any], text_key: str = "cleaned_text") -> str:
    relative_path = str(cleaned_doc.get("relative_path") or "")
    if Path(relative_path).suffix.lower() == ".epub":
        return stitch_epub_pages(cleaned_doc, text_key=text_key)

    page_blocks: list[str] = []
    for page in cleaned_doc["pages"]:
        text = (page.get(text_key) or "").strip()
        if not text:
            continue
        paragraphs = page_text_to_paragraphs(text)
        merged_paragraphs = merge_paragraphs_within_page(paragraphs)
        if not merged_paragraphs:
            continue
        page_blocks.append("\n".join(merged_paragraphs))

    merged = "\n".join(block for block in page_blocks if block.strip()).strip()
    return compact_no_blank_lines(merged)


def stitch_epub_pages(cleaned_doc: dict[str, Any], text_key: str = "cleaned_text") -> str:
    # EPUB extraction already contains meaningful line boundaries.
    # Preserve line breaks and avoid aggressive paragraph merging.
    page_blocks: list[str] = []
    for page in cleaned_doc["pages"]:
        text = (page.get(text_key) or "").strip()
        if not text:
            continue
        normalized_lines: list[str] = []
        for raw_line in text.splitlines():
            if not raw_line.strip():
                continue
            line = normalize_line(raw_line)
            if not line:
                continue
            if normalized_lines:
                hyphen_merged = merge_line_end_hyphenation_boundary(normalized_lines[-1], line)
                if hyphen_merged is not None:
                    normalized_lines[-1] = hyphen_merged
                    continue
            normalized_lines.append(line)

        page_text = "\n".join(normalized_lines).strip()
        if page_text:
            page_blocks.append(page_text)

    merged = "\n".join(page_blocks).strip()
    return compact_no_blank_lines(merged)


def merge_paragraphs_within_page(paragraphs: list[str]) -> list[str]:
    stitched: list[str] = []
    for paragraph in paragraphs:
        if not stitched:
            stitched.append(paragraph)
            continue

        prev = stitched[-1].rstrip()
        curr = paragraph.lstrip()

        hyphen_merged = merge_line_end_hyphenation_boundary(prev, curr)
        if hyphen_merged is not None:
            stitched[-1] = hyphen_merged
        elif len(prev) >= 300 or len(curr) >= 300:
            stitched.append(curr)
        elif prev.endswith(":") and curr.startswith("-"):
            stitched.append(curr)
        elif prev.endswith(":") and curr and LOWER_START_RE.match(curr):
            stitched[-1] = prev + " " + curr
        elif not SENTENCE_END_RE.search(prev) and curr and LOWER_START_RE.match(curr):
            stitched[-1] = prev + " " + curr
        else:
            stitched.append(curr)
    return [paragraph for paragraph in stitched if paragraph.strip()]


def compact_no_blank_lines(text: str) -> str:
    if not text:
        return ""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    merged = "\n".join(lines).strip()
    return merged + ("\n" if merged else "")


def page_text_to_paragraphs(text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n", text)
    paragraphs: list[str] = []
    for block in blocks:
        lines = [normalize_line(line) for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        current: list[str] = []
        for line in lines:
            if should_start_new_paragraph(current, line):
                if current:
                    paragraphs.append(join_paragraph_lines(current))
                current = [line]
            else:
                current.append(line)
        if current:
            paragraphs.append(join_paragraph_lines(current))

    return [paragraph for paragraph in paragraphs if paragraph]


def normalize_line(line: str) -> str:
    line = line.replace("\u00AD", "")
    line = OCR_BULLET_LINE_RE.sub("- ", line, count=1)
    line = re.sub(r"\s+", " ", line.strip())
    line = re.sub(r"\s+([,.;:!?])", r"\1", line)
    line = re.sub(r"([(\[«„])\s+", r"\1", line)
    line = re.sub(r"\s+([)\]»“])", r"\1", line)
    return line.strip()


def join_paragraph_lines(lines: list[str]) -> str:
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0]
    if looks_like_heading(lines[0]):
        heading_parts = [lines[0]]
        tail_start = 1
        for i, line in enumerate(lines[1:], start=1):
            if looks_like_heading_fragment(line):
                heading_parts.append(line)
                tail_start = i + 1
            else:
                break
        head = " ".join(heading_parts).strip()
        tail = join_lines_with_hyphen_fix(lines[tail_start:])
        return f"{head}\n{tail}" if tail else head
    return join_lines_with_hyphen_fix(lines)


def join_lines_with_hyphen_fix(lines: list[str]) -> str:
    if not lines:
        return ""
    merged = lines[0].strip()
    for raw_line in lines[1:]:
        line = raw_line.strip()
        if not line:
            continue
        hyphen_merged = merge_line_end_hyphenation_boundary(merged, line)
        if hyphen_merged is not None:
            merged = hyphen_merged
        else:
            merged = f"{merged.rstrip()} {line.lstrip()}"
    return merged.strip()


def merge_line_end_hyphenation_boundary(left_text: str, right_text: str) -> str | None:
    left = left_text.rstrip()
    right = right_text.lstrip()
    if not left or not right:
        return None
    if not LOWER_START_RE.match(right):
        return None
    left_match = WORD_TAIL_HYPHEN_RE.search(left)
    right_match = WORD_HEAD_RE.match(right)
    if not left_match or not right_match:
        return None

    left_part = left_match.group(1)
    right_part = right_match.group(1)
    glue = "" if len(left_part) >= 3 and len(right_part) >= 3 else "-"
    return f"{left[:-1]}{glue}{right}"


def should_start_new_paragraph(current: list[str], line: str) -> bool:
    if not current:
        return True
    prev = current[-1]
    if looks_like_heading(line):
        return True
    if line.startswith("-"):
        return True
    if prev.endswith(":") and line.startswith("-"):
        return True
    if len(line) <= 25 and line.endswith(":"):
        return True
    return False


def looks_like_heading(line: str) -> bool:
    compact = re.sub(r"\s+", " ", line.strip())
    return bool(HEADING_LINE_RE.fullmatch(compact))


def looks_like_heading_fragment(line: str) -> bool:
    compact = re.sub(r"\s+", " ", line.strip())
    if not compact or len(compact) > 40:
        return False
    letters = re.sub(r"[^A-ZА-ЯЁ]", "", compact)
    return len(letters) >= 4 and letters == letters.upper()


def build_summary(
    relative_path: str,
    ocr_doc: dict[str, Any],
    cleaned_doc: dict[str, Any],
    reviewed_doc: dict[str, Any],
    gemini_review_doc: dict[str, Any] | None,
    repaired_doc: dict[str, Any] | None,
    restored_doc: dict[str, Any] | None,
) -> dict[str, Any]:
    summary = {
        "relative_path": relative_path,
        "page_count": len(ocr_doc["pages"]),
        "source_counts": dict(Counter(page["source"] for page in ocr_doc["pages"])),
        "heuristic_verdict_counts": dict(Counter(page["page_verdict"] for page in reviewed_doc["pages"])),
        "edit_count": sum(len(page["edits"]) for page in cleaned_doc["pages"]),
        "flag_count": sum(len(page["flags"]) for page in cleaned_doc["pages"]),
    }
    if gemini_review_doc is not None:
        summary["gemini_verdict_counts"] = dict(Counter(page["llm_verdict"] for page in gemini_review_doc["pages"]))
    if repaired_doc is not None:
        summary["repair_status_counts"] = dict(Counter(page.get("repair_status", "unknown") for page in repaired_doc["pages"]))
        repair_issue_tags = Counter()
        for page in repaired_doc["pages"]:
            repair_issue_tags.update(page.get("repair_issue_tags", []))
        if repair_issue_tags:
            summary["repair_issue_tag_counts"] = dict(repair_issue_tags)
    if restored_doc is not None:
        summary["structure_status_counts"] = dict(Counter(page["status"] for page in restored_doc["pages"]))
    cleaning_statuses = [page.get("status") for page in cleaned_doc["pages"] if page.get("status")]
    if cleaning_statuses:
        summary["cleaning_status_counts"] = dict(Counter(cleaning_statuses))
    return summary


def log_progress(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


class ProgressTracker:
    def __init__(self, status_paths: list[Path], heartbeat_seconds: int = 20) -> None:
        self.status_paths = status_paths
        self.heartbeat_seconds = max(5, heartbeat_seconds)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._current_task: dict[str, Any] | None = None
        self._last_completed: dict[str, Any] | None = None
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, name="progress-heartbeat", daemon=True)
        self._write_status({"state": "idle"})
        self._heartbeat_thread.start()

    def close(self) -> None:
        self._stop_event.set()
        self._heartbeat_thread.join(timeout=1.0)
        with self._lock:
            self._current_task = None
        self._write_status({"state": "idle", "last_completed": self._last_completed_snapshot()})

    def note_book_start(self, relative_path: str, page_count: int) -> None:
        self._write_status(
            {
                "state": "book_starting",
                "relative_path": relative_path,
                "page_count": page_count,
                "last_completed": self._last_completed_snapshot(),
            }
        )

    def note_book_finish(self, relative_path: str) -> None:
        self._write_status(
            {
                "state": "book_finished",
                "relative_path": relative_path,
                "last_completed": self._last_completed_snapshot(),
            }
        )

    def start_page(self, relative_path: str, stage: str, index: int, total: int, page_number: int) -> None:
        now = time.monotonic()
        task = {
            "relative_path": relative_path,
            "stage": stage,
            "index": index,
            "total": total,
            "page_number": page_number,
            "started_monotonic": now,
            "last_heartbeat_elapsed": 0,
        }
        with self._lock:
            self._current_task = task
        self._write_status(self._status_payload(task, state="running"))
        log_progress(f"[{relative_path}] {stage} {index}/{total} (page {page_number})")

    def finish_page(self) -> None:
        with self._lock:
            task = self._current_task
            self._current_task = None
        if task is None:
            return
        completed = self._status_payload(task, state="waiting")
        self._last_completed = self._compact_status_payload(completed)
        self._write_status(
            {
                "state": "waiting",
                "relative_path": task["relative_path"],
                "last_completed": self._last_completed_snapshot(),
            }
        )

    def note_page_completion(self, relative_path: str, stage: str, index: int, total: int, page_number: int) -> None:
        payload = {
            "state": "running",
            "relative_path": relative_path,
            "stage": stage,
            "index": index,
            "total": total,
            "page_number": page_number,
            "elapsed_seconds": 0,
        }
        self._last_completed = self._compact_status_payload(payload)
        self._write_status(
            {
                "state": "running",
                "relative_path": relative_path,
                "stage": stage,
                "index": index,
                "total": total,
                "page_number": page_number,
                "last_completed": self._last_completed_snapshot(),
            }
        )

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(1.0):
            snapshot: dict[str, Any] | None = None
            elapsed_seconds = 0
            with self._lock:
                if self._current_task is None:
                    continue
                elapsed_seconds = int(time.monotonic() - float(self._current_task["started_monotonic"]))
                if elapsed_seconds < self.heartbeat_seconds:
                    continue
                if elapsed_seconds - int(self._current_task["last_heartbeat_elapsed"]) < self.heartbeat_seconds:
                    continue
                self._current_task["last_heartbeat_elapsed"] = elapsed_seconds
                snapshot = dict(self._current_task)
            if snapshot is None:
                continue
            payload = self._status_payload(snapshot, state="running", elapsed_seconds=elapsed_seconds)
            self._write_status(payload)
            log_progress(
                f"[{snapshot['relative_path']}] {snapshot['stage']} "
                f"{snapshot['index']}/{snapshot['total']} (page {snapshot['page_number']}) still running {elapsed_seconds}s"
            )

    def _status_payload(
        self,
        task: dict[str, Any],
        *,
        state: str,
        elapsed_seconds: int | None = None,
    ) -> dict[str, Any]:
        if elapsed_seconds is None:
            elapsed_seconds = int(time.monotonic() - float(task["started_monotonic"]))
        return {
            "state": state,
            "relative_path": task["relative_path"],
            "stage": task["stage"],
            "index": task["index"],
            "total": task["total"],
            "page_number": task["page_number"],
            "elapsed_seconds": elapsed_seconds,
            "last_completed": self._last_completed_snapshot(),
        }

    def _compact_status_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        compact = dict(payload)
        compact.pop("last_completed", None)
        return compact

    def _last_completed_snapshot(self) -> dict[str, Any] | None:
        if self._last_completed is None:
            return None
        return dict(self._last_completed)

    def _write_status(self, payload: dict[str, Any]) -> None:
        enriched = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            **payload,
        }
        for path in self.status_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = path.with_name(
                f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
            )
            try:
                temp_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                temp_path.replace(path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()


def build_caffeinate_command(pid: int) -> list[str]:
    return ["caffeinate", "-i", "-m", "-w", str(pid)]


class SleepPreventer:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        if not self.enabled or sys.platform != "darwin":
            return
        caffeinate = shutil.which("caffeinate")
        if not caffeinate:
            log_progress("[system] caffeinate not found; continuing without sleep prevention")
            return
        command = build_caffeinate_command(os.getpid())
        command[0] = caffeinate
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        log_progress("[system] Sleep prevention enabled via caffeinate (-i -m)")

    def close(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=1.0)
        self._process = None


def default_model_for_backend(backend: str) -> str:
    mapping = {
        "gemini": "gemini-2.5-flash",
        "deepseek": "deepseek-chat",
        "qwen": "qwen-vl-ocr-latest",
    }
    return mapping.get(backend, "")


PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "balanced_cost": {
        "backend": "qwen",
        "ocr_model": "qwen-vl-ocr-latest",
        "layout_sanitize_backend": "paddle",
        "cleaning_backend": "deepseek",
        "cleaning_model": "deepseek-chat",
        "cleaning_escalation_backend": "deepseek",
        "cleaning_escalation_model": "deepseek-chat",
        "review_backend": "heuristic",
        "repair_backend": "deepseek",
        "repair_model": "deepseek-chat",
        "repair_escalation_backend": "deepseek",
        "repair_escalation_model": "deepseek-chat",
        "final_structure_backend": "deepseek",
        "final_structure_model": "deepseek-chat",
        "final_structure_risky_only": True,
        "notes_policy": "delete",
    },
    "max_quality": {
        "backend": "qwen",
        "ocr_model": "qwen-vl-ocr-latest",
        "layout_sanitize_backend": "paddle",
        "cleaning_backend": "gemini",
        "cleaning_model": "gemini-2.5-flash",
        "cleaning_escalation_backend": "gemini",
        "cleaning_escalation_model": "gemini-2.5-pro",
        "review_backend": "heuristic",
        "repair_backend": "gemini",
        "repair_model": "gemini-2.5-flash",
        "repair_escalation_backend": "gemini",
        "repair_escalation_model": "gemini-2.5-pro",
        "final_structure_backend": "gemini",
        "final_structure_model": "gemini-2.5-pro",
        "final_structure_risky_only": False,
        "notes_policy": "delete",
    },
}


def apply_profile_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.profile == "custom":
        return args

    provided_flags = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
    option_map = {
        "backend": "--backend",
        "ocr_model": "--ocr-model",
        "layout_sanitize_backend": "--layout-sanitize-backend",
        "cleaning_backend": "--cleaning-backend",
        "cleaning_model": "--cleaning-model",
        "cleaning_escalation_backend": "--cleaning-escalation-backend",
        "cleaning_escalation_model": "--cleaning-escalation-model",
        "review_backend": "--review-backend",
        "review_model": "--review-model",
        "repair_backend": "--repair-backend",
        "repair_model": "--repair-model",
        "repair_escalation_backend": "--repair-escalation-backend",
        "repair_escalation_model": "--repair-escalation-model",
        "final_structure_backend": "--final-structure-backend",
        "final_structure_model": "--final-structure-model",
        "final_structure_risky_only": "--final-structure-risky-only",
        "notes_policy": "--notes-policy",
    }
    for attr, value in PROFILE_DEFAULTS[args.profile].items():
        if option_map[attr] not in provided_flags:
            setattr(args, attr, value)
    return args


def resolve_model(stage_model: str | None, backend: str, legacy_gemini_model: str) -> str:
    if stage_model:
        return stage_model
    if backend == "gemini":
        return legacy_gemini_model
    return default_model_for_backend(backend)


def build_cleaning_stage_agent(backend: str, model: str, notes_policy: str):
    if backend == "gemini":
        return GeminiCleaningAgent(GeminiCleaningConfig(model=model, notes_policy=notes_policy))
    if backend == "deepseek":
        return DeepSeekCleaningAgent(DeepSeekCleaningConfig(model=model, notes_policy=notes_policy))
    return None


def build_repair_stage_agent(backend: str, model: str, notes_policy: str):
    if backend == "gemini":
        return GeminiRepairAgent(GeminiRepairConfig(model=model, notes_policy=notes_policy, risky_only=True))
    if backend == "deepseek":
        return DeepSeekRepairAgent(DeepSeekRepairConfig(model=model, notes_policy=notes_policy, risky_only=True))
    return None


CLEANING_ESCALATION_TAGS = {
    "footnote_marker_left",
    "endnote_block_left",
    "citation_url_left",
    "reference_suffix_left",
    "heading_structure_risky",
    "excessive_blank_lines",
    "garbled_page",
    "publisher_meta_left",
    "empty_page_after_cleaning",
}
REPAIR_ESCALATION_TAGS = {
    "footnote_marker_left",
    "endnote_block_left",
    "citation_url_left",
    "reference_suffix_left",
    "heading_structure_risky",
    "garbled_page",
    "publisher_meta_left",
    "empty_page_after_cleaning",
}
STRUCTURE_RISK_TAGS = {"heading_structure_risky", "excessive_blank_lines"}
EARLY_EXIT_BLOCKING_FLAGS = {
    "footnote_markers",
    "citations_and_bibliographic_refs",
    "toc_index_material",
    "figure_table_formula_labels",
    "line_end_hyphenation_ambiguous",
    "reference_only_page",
    "garbled_page",
}


def should_escalate_cleaning(
    raw_page: dict[str, Any],
    cleaned_page: dict[str, Any],
    review_page: dict[str, Any],
) -> bool:
    issue_tags = set(review_page.get("issue_tags", []))
    if issue_tags & CLEANING_ESCALATION_TAGS:
        return True
    if cleaned_page.get("status") == "fallback":
        return True
    if raw_page.get("source") in {"ocr", "extract_fallback"} and issue_tags:
        return True
    risky_flag_ids = {
        "footnote_markers",
        "citations_and_bibliographic_refs",
        "toc_index_material",
        "figure_table_formula_labels",
        "line_end_hyphenation_ambiguous",
    }
    flag_ids = {flag["rule_id"] for flag in cleaned_page.get("flags", [])}
    if len(flag_ids & risky_flag_ids) >= 2:
        return True
    return False


def should_escalate_repair(
    raw_page: dict[str, Any],
    repaired_page: dict[str, Any],
    review_page: dict[str, Any],
) -> bool:
    issue_tags = set(review_page.get("issue_tags", []))
    if issue_tags & REPAIR_ESCALATION_TAGS:
        return True
    if repaired_page.get("repair_status") == "fallback" and issue_tags:
        return True
    if raw_page.get("source") in {"ocr", "extract_fallback"} and review_page.get("deletion_ratio", 0) > 0.25 and issue_tags:
        return True
    return False


def should_restore_structure(
    raw_page: dict[str, Any],
    page_after_repair: dict[str, Any],
    review_page: dict[str, Any],
    risky_only: bool,
) -> bool:
    if not risky_only:
        return True
    issue_tags = set(review_page.get("issue_tags", []))
    if issue_tags & STRUCTURE_RISK_TAGS:
        return True
    if page_after_repair.get("repair_status") == "fallback" and "heading_structure_risky" in page_after_repair.get(
        "repair_issue_tags", []
    ):
        return True
    raw_text = raw_page.get("body_text") or raw_page.get("selected_text") or ""
    cleaned_text = page_after_repair.get("cleaned_text") or ""
    raw_has_heading = bool(re.search(r"(?m)^(?:Глава\s+\d+|[IVXLCDM]+\.\s+.+)$", raw_text))
    cleaned_has_heading = bool(re.search(r"(?m)^(?:Глава\s+\d+|[IVXLCDM]+\.\s+.+)$", cleaned_text))
    return raw_has_heading and not cleaned_has_heading


def should_early_exit_after_review(
    *,
    page_state: PageState,
    ocr_page: dict[str, Any],
    cleaned_page: dict[str, Any],
    review_page: dict[str, Any],
    gemini_review_enabled: bool,
) -> tuple[bool, str]:
    if gemini_review_enabled:
        return False, "gemini_review_enabled"
    if page_is_trivially_empty_or_dropped(cleaned_page):
        return False, "empty_or_dropped_page"
    if page_state.route_decision != "easy_page":
        return False, f"route_decision={page_state.route_decision or 'unknown'}"
    if str(ocr_page.get("source") or "") not in {"extract", "epub_extract"}:
        return False, f"ocr_source={ocr_page.get('source') or 'unknown'}"
    if str(ocr_page.get("page_type") or page_state.page_type or "") != "body_only":
        return False, f"page_type={ocr_page.get('page_type') or page_state.page_type or 'unknown'}"
    if str(ocr_page.get("layout_status") or "") == "text_fallback":
        return False, "layout_status=text_fallback"
    if str(review_page.get("page_verdict") or "") != "approve":
        return False, f"review_verdict={review_page.get('page_verdict') or 'unknown'}"
    if page_state.risk_level not in {None, "low"}:
        return False, f"risk_level={page_state.risk_level or 'unknown'}"
    if review_page.get("issue_tags"):
        return False, "issue_tags_present"
    if cleaned_page.get("status") == "fallback":
        return False, "primary_cleaning_fallback"
    risky_flags = {flag["rule_id"] for flag in cleaned_page.get("flags", [])}
    if risky_flags & EARLY_EXIT_BLOCKING_FLAGS:
        return False, "risky_flags_present"
    audit_payload = cleaned_page.get("homoglyph_audit", {})
    if int(audit_payload.get("warned", 0) or 0) > 0:
        return False, "homoglyph_warnings_present"
    text = str(cleaned_page.get("cleaned_text") or "")
    if len(text.strip()) < 200:
        return False, "cleaned_text_too_short"
    return True, "easy_extract_body_only_review_approve"


def apply_early_exit_after_review(
    page_state: PageState,
    *,
    cleaned_page: dict[str, Any],
    reason: str,
    backend: str,
    model: str,
    include_primary_payload: bool,
) -> PageState:
    repaired_text = str(cleaned_page.get("cleaned_text") or "")
    repaired_page = dict(cleaned_page)
    repaired_page["cleaned_text"] = repaired_text
    repaired_page["repaired_text"] = repaired_text
    repaired_page["repair_status"] = "early_exit_passthrough"
    repaired_page["repair_stage"] = "early_exit"
    repaired_page["repair_notes"] = ["early_exit_after_review", reason]
    repaired_page["repair_issue_tags"] = list(page_state.review_tags)
    page_state.repaired_text = repaired_text
    page_state.repair_plan = None
    page_state.stage_payloads["repaired"] = repaired_page
    if include_primary_payload:
        page_state.stage_payloads["repaired_primary"] = dict(repaired_page)
    page_state.record_provenance(
        agent="EarlyExitAfterReview",
        input_fields=["stage_payloads.review", "stage_payloads.cleaned"],
        output_fields=["repaired_text", "stage_payloads.repaired"],
        note=f"repair_passthrough:{reason}",
    )
    transition(
        page_state,
        PageProcessingState.REPAIRED,
        agent="EarlyExitAfterReview",
        note=f"repair_passthrough:{reason}",
    )
    apply_structure_passthrough(
        page_state,
        backend=backend,
        model=model,
        status="early_exit_passthrough",
        skipped_reason="early_exit_safe_page",
        notes=["early_exit_after_review", reason],
        agent="EarlyExitAfterReview",
    )
    return page_state


def make_page_progress_logger(stage: str, relative_path: str):
    def _logger(index: int, total: int, page_number: int) -> None:
        log_progress(f"[{relative_path}] {stage} {index}/{total} (page {page_number})")

    return _logger


def run_page_step(
    tracker: ProgressTracker,
    relative_path: str,
    stage: str,
    index: int,
    total: int,
    page_number: int,
    callback,
):
    tracker.start_page(relative_path, stage, index, total, page_number)
    try:
        return callback()
    finally:
        tracker.finish_page()


class AsyncStageFailure(Exception):
    def __init__(self, page_number: int, cause: Exception) -> None:
        super().__init__(f"page {page_number}: {cause}")
        self.page_number = page_number
        self.cause = cause


async def run_async_stage_pages(
    *,
    tracker: ProgressTracker,
    relative_path: str,
    stage: str,
    page_numbers: list[int],
    max_concurrency: int,
    worker,
    on_success,
    on_error,
) -> None:
    if not page_numbers:
        return

    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    completed = 0

    async def _wrapped(page_number: int) -> int:
        async with semaphore:
            try:
                await worker(page_number)
                return page_number
            except Exception as exc:  # pragma: no cover - thin orchestration wrapper
                raise AsyncStageFailure(page_number, exc) from exc

    failures: list[AsyncStageFailure] = []
    tasks = [asyncio.create_task(_wrapped(page_number)) for page_number in page_numbers]
    try:
        for future in asyncio.as_completed(tasks):
            try:
                page_number = await future
                completed += 1
                tracker.note_page_completion(relative_path, stage, completed, len(page_numbers), page_number)
                log_progress(f"[{relative_path}] {stage} {completed}/{len(page_numbers)} (page {page_number})")
                on_success(page_number)
            except AsyncStageFailure as exc:
                completed += 1
                tracker.note_page_completion(relative_path, stage, completed, len(page_numbers), exc.page_number)
                log_progress(f"[{relative_path}] {stage} {completed}/{len(page_numbers)} (page {exc.page_number}) failed")
                on_error(exc.page_number, exc.cause)
                failures.append(exc)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    if failures:
        raise failures[0].cause


def latest_run_dir(run_root: Path) -> Path | None:
    candidates = sorted((path for path in run_root.glob("run_*") if path.is_dir()), key=lambda path: path.name)
    return candidates[-1] if candidates else None


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.resume_run_dir is not None:
        return args.resume_run_dir
    if args.resume:
        existing = latest_run_dir(args.run_root)
        if existing is not None:
            return existing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return args.run_root / f"run_{timestamp}"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def load_layout_sanitize_page_map(layout_json_path: Path) -> dict[int, str]:
    payload = read_json(layout_json_path)
    if not payload:
        return {}
    page_map: dict[int, str] = {}
    for page in payload.get("pages", []):
        try:
            page_number = int(page.get("page_number"))
        except (TypeError, ValueError):
            continue
        image_path = page.get("sanitized_image_path")
        if not image_path:
            continue
        if Path(image_path).exists():
            page_map[page_number] = str(image_path)
    return page_map


def load_layout_sanitize_layout_map(layout_json_path: Path) -> dict[int, dict[str, Any]]:
    payload = read_json(layout_json_path)
    if not payload:
        return {}
    layout_map: dict[int, dict[str, Any]] = {}
    for page in payload.get("pages", []):
        try:
            page_number = int(page.get("page_number"))
        except (TypeError, ValueError):
            continue
        if not isinstance(page, dict):
            continue
        layout_map[page_number] = page
    return layout_map


def ensure_layout_sanitized_pages(
    *,
    document_path: Path,
    book_dir: Path,
    page_numbers: list[int],
    args: argparse.Namespace,
    relative_path: str,
) -> tuple[dict[int, str], Path | None]:
    if args.layout_sanitize_backend == "none":
        return {}, None
    if document_path.suffix.lower() != ".pdf":
        return {}, None

    layout_dir = book_dir / "layout_sanitize"
    layout_json_path = layout_dir / f"{document_path.stem}.layout_ocr.json"
    cached_page_map = load_layout_sanitize_page_map(layout_json_path)
    if len(cached_page_map) >= len(page_numbers):
        return cached_page_map, layout_json_path

    python_executable = Path(args.layout_sanitize_python)
    if not python_executable.exists():
        raise FileNotFoundError(
            f"Layout sanitizer python not found: {python_executable}. "
            "Create .venv-paddle310 or pass --layout-sanitize-python."
        )

    command = [
        str(python_executable),
        str(ROOT / "scripts" / "run_paddle_layout_ocr.py"),
        "--input",
        str(document_path),
        "--out-dir",
        str(layout_dir),
        "--render-scale",
        str(args.layout_sanitize_render_scale),
        "--mask-fill",
        str(args.layout_sanitize_mask_fill),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines: list[str] = []
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue
        output_lines.append(line)
        if line.startswith("Layout sanitize "):
            log_progress(f"[{relative_path}] {line}")
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(
            f"Layout sanitizer failed for {relative_path}: "
            f"{output_lines[-1] if output_lines else 'unknown error'}"
        )

    page_map = load_layout_sanitize_page_map(layout_json_path)
    if not page_map:
        raise RuntimeError(f"Layout sanitizer produced no sanitized pages for {relative_path}")
    return page_map, layout_json_path


def page_map(document: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(page["page_number"]): page for page in document.get("pages", [])}


def ordered_pages(pages_by_number: dict[int, dict[str, Any]], page_numbers: list[int]) -> list[dict[str, Any]]:
    return [pages_by_number[page_number] for page_number in page_numbers if page_number in pages_by_number]


def missing_pages(document: dict[str, Any], page_numbers: list[int]) -> list[int]:
    existing = page_map(document)
    return [page_number for page_number in page_numbers if page_number not in existing]


def get_page_numbers(document_path: Path, ocr_agent: OCRAgent) -> list[int]:
    return ocr_agent.get_page_numbers(document_path)


def cached_page_count(document: dict[str, Any]) -> int:
    return len(document.get("pages", []))


def page_is_trivially_empty_or_dropped(page: dict[str, Any]) -> bool:
    if page.get("drop_page"):
        return True
    if page.get("allow_empty_output") and not (page.get("cleaned_text") or "").strip():
        return True
    return False


def source_type_from_path(document_path: Path) -> str:
    return document_path.suffix.lower().lstrip(".") or "unknown"


def stage_state_from_ocr_source(source: str) -> PageProcessingState:
    if source in {"ocr", "extract_fallback"}:
        return PageProcessingState.OCR_DONE
    return PageProcessingState.EXTRACTED


def build_document_state(
    *,
    doc_id: str,
    document_path: Path,
    route_hint: str,
    page_numbers: list[int],
) -> DocumentState:
    return DocumentState(
        doc_id=doc_id,
        source_path=document_path.as_posix(),
        source_type=source_type_from_path(document_path),
        route_hint=route_hint,
        page_numbers=page_numbers,
    )


def base_page_state(
    *,
    doc_state: DocumentState,
    page_number: int,
) -> PageState:
    return PageState.create(
        doc_id=doc_state.doc_id,
        page_num=page_number,
        source_path=doc_state.source_path,
        source_type=doc_state.source_type,
    )


def page_state_to_ocr_page(page_state: PageState) -> dict[str, Any]:
    payload = dict(page_state.stage_payloads.get("ocr", {}))
    if payload:
        return payload
    text = page_state.raw_text
    return {
        "page_number": page_state.page_num,
        "page_index": page_state.page_num - 1,
        "route_hint": "",
        "source": page_state.ocr_mode or "",
        "selected_text": text,
        "body_text": text,
        "notes_text": "",
        "reference_text": "",
        "page_type": page_state.page_type or "empty",
        "layout_status": "",
        "extracted_text": text,
        "ocr_text": "",
        "extracted_char_count": len(text),
        "ocr_char_count": 0,
        "width": 0.0,
        "height": 0.0,
        "blocks": list(page_state.layout_blocks),
        "notes": [],
    }


def page_state_to_rule_page(page_state: PageState) -> dict[str, Any]:
    payload = dict(page_state.stage_payloads.get("rule_cleaned", {}))
    if payload:
        return payload
    return {
        "page_number": page_state.page_num,
        "source": page_state.ocr_mode or "",
        "raw_text": page_state.raw_text,
        "cleaned_text": page_state.rule_cleaned_text,
        "edits": [],
        "flags": [],
        "protected_hits": [],
        "allow_empty_output": False,
        "drop_page": False,
        "drop_reason": "",
    }


def page_state_to_primary_page(page_state: PageState) -> dict[str, Any]:
    payload = dict(page_state.stage_payloads.get("primary_cleaned", {}))
    if payload:
        return payload
    fallback_text = page_state.primary_clean_text or page_state.rule_cleaned_text
    return {
        "page_number": page_state.page_num,
        "source": page_state.ocr_mode or "",
        "raw_text": page_state.raw_text,
        "cleaned_text": fallback_text,
        "edits": [],
        "flags": [],
        "protected_hits": [],
        "status": "state_fallback",
        "notes": [],
        "allow_empty_output": False,
        "drop_page": False,
        "drop_reason": "",
    }


def page_state_to_review_page(page_state: PageState) -> dict[str, Any]:
    payload = dict(page_state.stage_payloads.get("review", {}))
    if payload:
        return payload
    page_verdict = "approve" if not page_state.review_tags else "escalate"
    if page_state.risk_level == "high":
        page_verdict = "reject"
    return {
        "page_number": page_state.page_num,
        "source": page_state.ocr_mode or "",
        "page_verdict": page_verdict,
        "raw_length": len(page_state.raw_text),
        "cleaned_length": len(page_state.primary_clean_text),
        "deletion_ratio": 0.0,
        "issue_tags": list(page_state.review_tags),
        "review_records": [],
    }


def page_state_to_gemini_review_page(page_state: PageState) -> dict[str, Any] | None:
    payload = page_state.stage_payloads.get("gemini_review")
    return dict(payload) if payload else None


def page_state_to_repaired_page(page_state: PageState, *, payload_key: str = "repaired") -> dict[str, Any]:
    payload = dict(page_state.stage_payloads.get(payload_key, {}))
    if payload:
        return payload
    repaired_text = page_state.repaired_text or page_state.primary_clean_text or page_state.rule_cleaned_text
    return {
        "page_number": page_state.page_num,
        "source": page_state.ocr_mode or "",
        "raw_text": page_state.raw_text,
        "cleaned_text": repaired_text,
        "repaired_text": repaired_text,
        "llm_repair_edits": [],
        "llm_repair_plan": page_state.repair_plan,
        "repair_status": "state_fallback",
        "repair_notes": [],
        "repair_issue_tags": list(page_state.review_tags),
        "allow_empty_output": False,
        "drop_page": False,
        "drop_reason": "",
    }


def page_state_to_structure_page(page_state: PageState) -> dict[str, Any]:
    structure_plan = dict(page_state.structure_plan or {})
    return {
        "page_number": page_state.page_num,
        "restored_text": page_state.final_text,
        "status": structure_plan.get("status", "state_fallback"),
        "notes": list(structure_plan.get("notes", [])),
        "skipped_reason": structure_plan.get("skipped_reason"),
        "backend": structure_plan.get("backend"),
        "model": structure_plan.get("model"),
        "final_text_source": structure_plan.get("final_text_source", "repaired_passthrough"),
    }


def ordered_stage_pages(page_states: dict[int, PageState], page_numbers: list[int], stage: str) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for page_number in page_numbers:
        page_state = page_states.get(page_number)
        if page_state is None:
            continue
        if stage == "ocr" and state_at_least(effective_state(page_state), PageProcessingState.EXTRACTED):
            pages.append(page_state_to_ocr_page(page_state))
        elif stage == "rule_cleaned" and state_at_least(effective_state(page_state), PageProcessingState.RULE_CLEANED):
            pages.append(page_state_to_rule_page(page_state))
        elif stage == "primary_cleaned" and state_at_least(effective_state(page_state), PageProcessingState.PRIMARY_CLEANED):
            pages.append(page_state_to_primary_page(page_state))
        elif stage == "review" and state_at_least(effective_state(page_state), PageProcessingState.REVIEWED):
            pages.append(page_state_to_review_page(page_state))
        elif stage == "gemini_review":
            payload = page_state_to_gemini_review_page(page_state)
            if payload is not None:
                pages.append(payload)
        elif stage == "repaired_primary":
            if page_state.stage_payloads.get("repaired_primary"):
                pages.append(page_state_to_repaired_page(page_state, payload_key="repaired_primary"))
        elif stage == "repaired" and state_at_least(effective_state(page_state), PageProcessingState.REPAIRED):
            pages.append(page_state_to_repaired_page(page_state))
        elif stage == "restored" and state_at_least(effective_state(page_state), PageProcessingState.STRUCTURE_RESTORED):
            pages.append(page_state_to_structure_page(page_state))
    return pages


def review_risk_level(review_page: dict[str, Any]) -> str:
    return REVIEW_RISK_BY_VERDICT.get(str(review_page.get("page_verdict") or ""), "medium")


def gemini_review_risk_level(gemini_review_page: dict[str, Any]) -> str:
    return REVIEW_RISK_BY_VERDICT.get(str(gemini_review_page.get("llm_verdict") or ""), "low")


def max_risk_level(current: str | None, candidate: str | None) -> str | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return current if RISK_ORDER.get(current, 0) >= RISK_ORDER.get(candidate, 0) else candidate


def structure_note(
    *,
    final_text_source: str,
    backend: str,
    model: str,
    status: str,
    skipped_reason: str | None = None,
) -> str:
    note = f"source={final_text_source};backend={backend};model={model};status={status}"
    if skipped_reason:
        note += f";skipped_reason={skipped_reason}"
    return note


def apply_structure_passthrough(
    page_state: PageState,
    *,
    backend: str,
    model: str,
    status: str,
    skipped_reason: str,
    notes: list[str] | None = None,
    text: str | None = None,
    agent: str = "StructureStage",
) -> PageState:
    final_text = text
    if final_text is None:
        final_text = page_state.repaired_text or page_state.primary_clean_text or page_state.rule_cleaned_text
    plan = {
        "backend": backend,
        "model": model,
        "status": status,
        "notes": list(notes or []),
        "skipped_reason": skipped_reason,
        "final_text_source": "repaired_passthrough",
    }
    page_state.structure_plan = plan
    page_state.final_text = final_text
    note = structure_note(
        final_text_source="repaired_passthrough",
        backend=backend,
        model=model,
        status=status,
        skipped_reason=skipped_reason,
    )
    page_state.record_provenance(
        agent=agent,
        input_fields=["repaired_text"],
        output_fields=["structure_plan", "final_text"],
        note=note,
    )
    transition(
        page_state,
        PageProcessingState.STRUCTURE_RESTORED,
        agent=agent,
        note=note,
    )
    return page_state


def ensure_repaired_state_for_structure(page_state: PageState) -> PageState:
    if page_state.current_state == PageProcessingState.FAILED:
        page_state.current_state = page_state.last_success_state
    if state_at_least(effective_state(page_state), PageProcessingState.REPAIRED):
        if not page_state.repaired_text:
            page_state.repaired_text = page_state.primary_clean_text or page_state.rule_cleaned_text
        return page_state

    page_state.repaired_text = page_state.primary_clean_text or page_state.rule_cleaned_text
    page_state.record_provenance(
        agent="StructureBridge",
        input_fields=["primary_clean_text"],
        output_fields=["repaired_text"],
        note="repair_passthrough_for_structure",
    )
    transition(
        page_state,
        PageProcessingState.REPAIRED,
        agent="StructureBridge",
        note="repair_passthrough_for_structure",
    )
    return page_state


def sync_cleaned_stage_payloads(
    page_states: dict[int, PageState],
    cleaned_doc: dict[str, Any],
) -> None:
    for cleaned_page in cleaned_doc.get("pages", []):
        page_number = int(cleaned_page["page_number"])
        page_state = page_states.get(page_number)
        if page_state is None:
            continue
        page_state.stage_payloads["cleaned"] = dict(cleaned_page)


def apply_rule_cleaned_homoglyph_audit(page_state: PageState) -> bool:
    rule_page = page_state.stage_payloads.get("rule_cleaned")
    if rule_page is None:
        return False
    if "homoglyph_audit" in rule_page:
        return False

    original_text = str(rule_page.get("cleaned_text") or "")
    audit_result = audit_russian_homoglyphs(original_text)
    audit_payload = {
        "detected": audit_result["detected"],
        "auto_fixed": audit_result["auto_fixed"],
        "warned": audit_result["warned"],
        "samples": audit_result["samples"],
    }
    rule_page["homoglyph_audit"] = audit_payload
    rule_page["cleaned_text"] = audit_result["text"]
    page_state.rule_cleaned_text = audit_result["text"]
    if audit_result["detected"] or audit_result["auto_fixed"] or audit_result["warned"]:
        page_state.record_provenance(
            agent="RussianHomoglyphAudit",
            input_fields=["rule_cleaned_text"],
            output_fields=["rule_cleaned_text", "stage_payloads.rule_cleaned.homoglyph_audit"],
            note=(
                f"detected={audit_result['detected']};"
                f"auto_fixed={audit_result['auto_fixed']};warned={audit_result['warned']}"
            ),
        )
    return True


def backfill_review_and_repair_states(
    *,
    page_states: dict[int, PageState],
    page_numbers: list[int],
    reviewed_doc: dict[str, Any] | None,
    gemini_review_doc: dict[str, Any] | None,
    primary_repaired_doc: dict[str, Any] | None,
    repaired_doc: dict[str, Any] | None,
) -> None:
    reviewed_pages = page_map(reviewed_doc or {"pages": []})
    gemini_review_pages = page_map(gemini_review_doc or {"pages": []})
    primary_repaired_pages = page_map(primary_repaired_doc or {"pages": []})
    repaired_pages = page_map(repaired_doc or {"pages": []})

    for page_number in page_numbers:
        page_state = page_states[page_number]
        review_page = reviewed_pages.get(page_number)
        gemini_review_page = gemini_review_pages.get(page_number)
        primary_repaired_page = primary_repaired_pages.get(page_number)
        repaired_page = repaired_pages.get(page_number)

        if review_page is not None:
            page_state.review_tags = list(review_page.get("issue_tags", []))
            page_state.risk_level = max_risk_level(page_state.risk_level, review_risk_level(review_page))
            page_state.stage_payloads.setdefault("review", dict(review_page))
            if not state_at_least(effective_state(page_state), PageProcessingState.REVIEWED):
                page_state.current_state = PageProcessingState.REVIEWED
                page_state.last_success_state = PageProcessingState.REVIEWED
                page_state.record_provenance(
                    agent="legacy_resume",
                    input_fields=["review.json"],
                    output_fields=["review_tags", "risk_level", "stage_payloads.review"],
                    note="restored_from_review_checkpoint",
                )

        if gemini_review_page is not None:
            had_gemini_review = "gemini_review" in page_state.stage_payloads
            page_state.stage_payloads.setdefault("gemini_review", dict(gemini_review_page))
            page_state.risk_level = max_risk_level(page_state.risk_level, gemini_review_risk_level(gemini_review_page))
            if not had_gemini_review:
                page_state.record_provenance(
                    agent="legacy_resume",
                    input_fields=["gemini_review.json"],
                    output_fields=["risk_level", "stage_payloads.gemini_review"],
                    note="restored_from_gemini_review_checkpoint",
                )

        if primary_repaired_page is not None:
            page_state.stage_payloads.setdefault("repaired_primary", dict(primary_repaired_page))

        final_repaired_page = repaired_page or primary_repaired_page
        if final_repaired_page is not None:
            page_state.repaired_text = str(
                final_repaired_page.get("repaired_text")
                or final_repaired_page.get("cleaned_text")
                or page_state.repaired_text
            )
            page_state.repair_plan = final_repaired_page.get("llm_repair_plan") or page_state.repair_plan
            page_state.stage_payloads.setdefault("repaired", dict(final_repaired_page))
            if not state_at_least(effective_state(page_state), PageProcessingState.REPAIRED):
                page_state.current_state = PageProcessingState.REPAIRED
                page_state.last_success_state = PageProcessingState.REPAIRED
                page_state.record_provenance(
                    agent="legacy_resume",
                    input_fields=["repaired.json" if repaired_page is not None else "repaired_primary.json"],
                    output_fields=["repaired_text", "repair_plan", "stage_payloads.repaired"],
                    note="restored_from_repair_checkpoint",
                )


def backfill_structure_states(
    *,
    page_states: dict[int, PageState],
    page_numbers: list[int],
    restored_doc: dict[str, Any] | None,
) -> None:
    if restored_doc is None:
        return

    restored_pages = page_map(restored_doc)
    model = str(restored_doc.get("model") or "")
    for page_number in page_numbers:
        restored_page = restored_pages.get(page_number)
        if restored_page is None:
            continue
        page_state = page_states[page_number]
        if state_at_least(effective_state(page_state), PageProcessingState.STRUCTURE_RESTORED):
            continue

        status = str(restored_page.get("status") or "legacy_restored")
        final_text_source = str(
            restored_page.get("final_text_source")
            or ("structure_restore_generated" if status == "gemini" else "repaired_passthrough")
        )
        skipped_reason = restored_page.get("skipped_reason")
        structure_plan = {
            "backend": str(restored_page.get("backend") or "gemini"),
            "model": str(restored_page.get("model") or model),
            "status": status,
            "notes": list(restored_page.get("notes", [])),
            "skipped_reason": skipped_reason,
            "final_text_source": final_text_source,
        }
        page_state.structure_plan = structure_plan
        page_state.final_text = str(restored_page.get("restored_text") or "")
        page_state.current_state = PageProcessingState.STRUCTURE_RESTORED
        page_state.last_success_state = PageProcessingState.STRUCTURE_RESTORED
        page_state.record_provenance(
            agent="legacy_resume",
            input_fields=["gemini_structure.json"],
            output_fields=["structure_plan", "final_text"],
            note=structure_note(
                final_text_source=final_text_source,
                backend=structure_plan["backend"],
                model=structure_plan["model"],
                status=status,
                skipped_reason=str(skipped_reason) if skipped_reason else None,
            ),
        )


def resolve_export_text(
    page_state: PageState,
    *,
    legacy_restored_page: dict[str, Any] | None = None,
    legacy_repaired_page: dict[str, Any] | None = None,
    legacy_cleaned_page: dict[str, Any] | None = None,
) -> tuple[str, str]:
    if page_state.structure_plan is not None or state_at_least(effective_state(page_state), PageProcessingState.STRUCTURE_RESTORED):
        return page_state.final_text, "final_text"
    if page_state.repaired_text or page_state.repair_plan is not None or state_at_least(
        effective_state(page_state), PageProcessingState.REPAIRED
    ):
        return page_state.repaired_text, "repaired_text"
    if legacy_restored_page is not None:
        return str(legacy_restored_page.get("restored_text") or ""), "legacy_restored"
    if legacy_repaired_page is not None:
        return str(legacy_repaired_page.get("repaired_text") or legacy_repaired_page.get("cleaned_text") or ""), "legacy_repaired"
    if legacy_cleaned_page is not None:
        return str(legacy_cleaned_page.get("cleaned_text") or ""), "legacy_cleaned"
    return page_state.primary_clean_text or page_state.rule_cleaned_text or page_state.raw_text, "page_state_fallback"


def _looks_like_substantial_body_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    cyrillic_words = re.findall(r"\b[А-Яа-яЁё]{4,}\b", stripped)
    sentence_like = re.findall(r"[.!?…]", stripped)
    return len(cyrillic_words) >= 25 or len(sentence_like) >= 3


def _looks_like_note_heavy_export_page(page_state: PageState, text: str) -> bool:
    if not text.strip():
        return False
    if page_state.page_type == "reference_only":
        return True

    note_tags = {"footnote_marker_left", "endnote_block_left", "reference_suffix_left", "citation_url_left"}
    if not (set(page_state.review_tags) & note_tags):
        return False

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    note_like = sum(
        1
        for line in lines
        if EXPORT_NOTE_LINE_RE.match(line) or EXPORT_REFERENCE_CUE_RE.search(line)
    )
    return note_like >= 3 and note_like >= max(3, len(lines) // 2)


def classify_export_filter_reason(
    *,
    page_state: PageState,
    page_number: int,
    total_pages: int,
    text: str,
) -> str | None:
    stripped = text.strip()
    if page_state.route_decision == "skip_nonbody_page" or (page_state.page_type or "") in NONBODY_EXPORT_PAGE_TYPES:
        return "filtered_nonbody_page"
    if not stripped:
        return "filtered_empty_page"
    if _looks_like_note_heavy_export_page(page_state, stripped):
        return "filtered_note_heavy_page"
    if total_pages >= 20 and page_number <= min(4, total_pages):
        if FRONTMATTER_EXPORT_RE.search(stripped) or not _looks_like_substantial_body_text(stripped):
            return "filtered_frontmatter_page"
    if total_pages >= 20 and page_number >= max(1, total_pages - 3):
        if BACKMATTER_EXPORT_RE.search(stripped):
            return "filtered_backmatter_page"
    return None


def build_export_document(
    *,
    relative_path: str,
    page_states: dict[int, PageState],
    page_numbers: list[int],
    restored_doc: dict[str, Any] | None,
    repaired_doc: dict[str, Any] | None,
    cleaned_doc: dict[str, Any],
) -> tuple[dict[str, Any], dict[int, str]]:
    restored_pages = page_map(restored_doc or {"pages": []})
    repaired_pages = page_map(repaired_doc or {"pages": []})
    cleaned_pages = page_map(cleaned_doc)
    export_pages: list[dict[str, Any]] = []
    export_sources: dict[int, str] = {}
    total_pages = max(page_numbers) if page_numbers else 0
    for page_number in page_numbers:
        page_state = page_states[page_number]
        text, source = resolve_export_text(
            page_state,
            legacy_restored_page=restored_pages.get(page_number),
            legacy_repaired_page=repaired_pages.get(page_number),
            legacy_cleaned_page=cleaned_pages.get(page_number),
        )
        filter_reason = classify_export_filter_reason(
            page_state=page_state,
            page_number=page_number,
            total_pages=total_pages,
            text=text,
        )
        if filter_reason is not None:
            export_sources[page_number] = filter_reason
            continue
        export_pages.append(
            {
                "page_number": page_number,
                "cleaned_text": text,
                "export_source": source,
            }
        )
        export_sources[page_number] = source
    return {"relative_path": relative_path, "pages": export_pages}, export_sources


def run_post_clean_final_txt(final_txt_path: Path, *, backup_root: Path | None = None) -> None:
    backup_dir = backup_root or (ROOT / "outputs" / "final_txt_backups")
    command = [
        sys.executable,
        str(POST_CLEAN_SCRIPT),
        "--txt-dir",
        str(final_txt_path.parent),
        "--glob",
        final_txt_path.name,
        "--backup-root",
        str(backup_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return
    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    detail = stderr or stdout or f"exit_code={completed.returncode}"
    raise RuntimeError(f"post_clean_final_txt failed for {final_txt_path.name}: {detail}")


def hydrate_page_state_from_legacy(
    *,
    doc_state: DocumentState,
    page_number: int,
    ocr_page: dict[str, Any] | None,
    rule_page: dict[str, Any] | None,
    primary_page: dict[str, Any] | None,
) -> PageState:
    page_state = base_page_state(doc_state=doc_state, page_number=page_number)
    if ocr_page is not None:
        source = str(ocr_page.get("source") or "")
        page_state.page_type = ocr_page.get("page_type")
        page_state.ocr_mode = source
        page_state.raw_text = (ocr_page.get("body_text") or ocr_page.get("selected_text") or "").strip()
        page_state.layout_blocks = list(ocr_page.get("blocks", []))
        page_state.stage_payloads["ocr"] = dict(ocr_page)
        page_state.current_state = stage_state_from_ocr_source(source)
        page_state.last_success_state = page_state.current_state
        page_state.record_provenance(
            agent="legacy_resume",
            input_fields=["ocr.json"],
            output_fields=["raw_text", "layout_blocks", "page_type", "ocr_mode"],
            note="restored_from_ocr_checkpoint",
        )
    if rule_page is not None:
        page_state.rule_cleaned_text = str(rule_page.get("cleaned_text") or "")
        page_state.stage_payloads["rule_cleaned"] = dict(rule_page)
        page_state.current_state = PageProcessingState.RULE_CLEANED
        page_state.last_success_state = PageProcessingState.RULE_CLEANED
        page_state.record_provenance(
            agent="legacy_resume",
            input_fields=["rule_cleaned.json"],
            output_fields=["rule_cleaned_text"],
            note="restored_from_rule_checkpoint",
        )
    if primary_page is not None:
        page_state.primary_clean_text = str(primary_page.get("cleaned_text") or "")
        page_state.edit_plan = primary_page.get("llm_edit_plan")
        page_state.stage_payloads["primary_cleaned"] = dict(primary_page)
        page_state.current_state = PageProcessingState.PRIMARY_CLEANED
        page_state.last_success_state = PageProcessingState.PRIMARY_CLEANED
        page_state.record_provenance(
            agent="legacy_resume",
            input_fields=["cleaned_primary.json"],
            output_fields=["primary_clean_text", "edit_plan"],
            note="restored_from_primary_checkpoint",
        )
    return page_state


def load_page_states(
    *,
    checkpoint_store: PageCheckpointStore,
    doc_state: DocumentState,
    page_numbers: list[int],
    ocr_doc: dict[str, Any] | None,
    rule_cleaned_doc: dict[str, Any] | None,
    primary_cleaned_doc: dict[str, Any] | None,
) -> dict[int, PageState]:
    loaded = checkpoint_store.load_pages(page_numbers)
    ocr_pages = page_map(ocr_doc or {"pages": []})
    rule_pages = page_map(rule_cleaned_doc or {"pages": []})
    primary_pages = page_map(primary_cleaned_doc or {"pages": []})

    page_states: dict[int, PageState] = {}
    for page_number in page_numbers:
        page_state = loaded.get(page_number)
        if page_state is None:
            page_state = hydrate_page_state_from_legacy(
                doc_state=doc_state,
                page_number=page_number,
                ocr_page=ocr_pages.get(page_number),
                rule_page=rule_pages.get(page_number),
                primary_page=primary_pages.get(page_number),
            )
        page_states[page_number] = page_state
    return page_states


def main() -> None:
    args = parse_args()
    args = apply_profile_defaults(args)
    if args.layout_sanitize_backend != "none" and args.backend == "extract_only":
        raise ValueError("Layout sanitizer requires an OCR backend. It cannot be combined with --backend extract_only.")
    manifest = load_manifest(args.manifest)
    run_dir = resolve_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    args.final_txt_dir.mkdir(parents=True, exist_ok=True)
    progress_tracker = ProgressTracker(
        [run_dir / "progress.json", ROOT / "outputs" / "latest_progress.json"],
        heartbeat_seconds=args.heartbeat_seconds,
    )
    sleep_preventer = SleepPreventer(args.prevent_sleep)
    sleep_preventer.start()

    try:
        ocr_model = resolve_model(args.ocr_model, args.backend, args.gemini_model)
        cleaning_model = resolve_model(args.cleaning_model, args.cleaning_backend, args.gemini_model)
        cleaning_escalation_model = resolve_model(
            args.cleaning_escalation_model,
            args.cleaning_escalation_backend,
            args.gemini_model,
        )
        review_model = resolve_model(args.review_model, args.review_backend, args.gemini_model)
        repair_model = resolve_model(args.repair_model, args.repair_backend, args.gemini_model)
        repair_escalation_model = resolve_model(
            args.repair_escalation_model,
            args.repair_escalation_backend,
            args.gemini_model,
        )
        final_structure_model = resolve_model(
            args.final_structure_model,
            args.final_structure_backend,
            args.gemini_model,
        )

        ocr_agent = OCRAgent(
            OCRAgentConfig(
                backend=args.backend,
                gemini_model=ocr_model or args.gemini_model,
                qwen_model=ocr_model or default_model_for_backend("qwen"),
                render_scale=args.ocr_render_scale,
                force_ocr_body_pages=args.force_ocr_body_pages,
            )
        )
        page_commander = PageCommander(
            CommanderConfig(
                ocr_base_render_scale=args.ocr_render_scale,
                ocr_high_render_scale=max(args.ocr_render_scale, 2.6),
            )
        )
        cleaning_agent = CleaningAgent()
        primary_cleaning_agent = build_cleaning_stage_agent(args.cleaning_backend, cleaning_model, args.notes_policy)
        cleaning_escalation_agent = build_cleaning_stage_agent(
            args.cleaning_escalation_backend,
            cleaning_escalation_model,
            args.notes_policy,
        )
        review_agent = ReviewAgent()
        gemini_review_agent = None
        primary_repair_agent = build_repair_stage_agent(args.repair_backend, repair_model, args.notes_policy)
        repair_escalation_agent = build_repair_stage_agent(
            args.repair_escalation_backend,
            repair_escalation_model,
            args.notes_policy,
        )
        structure_agent = None
        structure_backend_label = args.final_structure_backend
        if args.review_backend == "gemini":
            gemini_review_agent = GeminiReviewAgent(GeminiReviewConfig(model=review_model, risky_only=True))
        if args.final_structure_backend == "gemini":
            structure_agent = GeminiStructureAgent(GeminiStructureConfig(model=final_structure_model))
            structure_backend_label = "gemini"
        elif args.final_structure_backend == "deepseek":
            structure_agent = DeepSeekStructureAgent(DeepSeekStructureConfig(model=final_structure_model))
            structure_backend_label = "deepseek"

        summaries = []
        for relative_path in args.books:
            log_progress(f"[{relative_path}] Starting")
            document_path = ROOT / relative_path
            if not document_path.exists():
                raise FileNotFoundError(f"Missing book: {document_path}")

            manifest_row = manifest.get(relative_path, {})
            route_hint = manifest_row.get("route", "auto") or "auto"
            if document_path.suffix.lower() == ".epub" and route_hint == "auto":
                route_hint = "epub_extract_then_clean"
            safe_stem = safe_stem_from_relative_path(relative_path)
            book_dir = run_dir / safe_stem
            book_dir.mkdir(parents=True, exist_ok=True)
            page_numbers = get_page_numbers(document_path, ocr_agent)
            progress_tracker.note_book_start(relative_path, len(page_numbers))
            doc_state = build_document_state(
                doc_id=safe_stem,
                document_path=document_path,
                route_hint=route_hint,
                page_numbers=page_numbers,
            )
            checkpoint_store = PageCheckpointStore(book_dir)
            ocr_path = book_dir / "ocr.json"
            rule_cleaned_path = book_dir / "rule_cleaned.json"
            primary_cleaned_path = book_dir / ("cleaned_primary.json" if cleaning_escalation_agent is not None else "cleaned.json")

            legacy_ocr_doc = read_json(ocr_path)
            legacy_rule_cleaned_doc = read_json(rule_cleaned_path)
            legacy_primary_cleaned_doc = read_json(primary_cleaned_path)
            page_states = load_page_states(
                checkpoint_store=checkpoint_store,
                doc_state=doc_state,
                page_numbers=page_numbers,
                ocr_doc=legacy_ocr_doc,
                rule_cleaned_doc=legacy_rule_cleaned_doc,
                primary_cleaned_doc=legacy_primary_cleaned_doc,
            )

            ocr_doc = legacy_ocr_doc or {
                "relative_path": document_path.as_posix(),
                "page_count": len(page_numbers),
                "route_hint": route_hint,
                "backend": args.backend,
                "pages": [],
            }
            ocr_doc["relative_path"] = document_path.as_posix()
            ocr_doc["page_count"] = len(page_numbers)
            ocr_doc["route_hint"] = route_hint
            ocr_doc["backend"] = args.backend
            ocr_doc["layout_sanitize_backend"] = args.layout_sanitize_backend
            ocr_doc["pages"] = ordered_stage_pages(page_states, page_numbers, "ocr")
            if ocr_doc["pages"]:
                log_progress(f"[{relative_path}] Reusing OCR checkpoint {len(ocr_doc['pages'])}/{len(page_numbers)} pages")
            ocr_missing = [
                page_number
                for page_number in page_numbers
                if not state_at_least(effective_state(page_states[page_number]), PageProcessingState.EXTRACTED)
            ]
            sanitized_page_map: dict[int, str] = {}
            sanitized_layout_map: dict[int, dict[str, Any]] = {}
            if ocr_missing:
                if args.layout_sanitize_backend != "none":
                    log_progress(f"[{relative_path}] Layout sanitize backend={args.layout_sanitize_backend}")
                    sanitized_page_map, layout_json_path = ensure_layout_sanitized_pages(
                        document_path=document_path,
                        book_dir=book_dir,
                        page_numbers=page_numbers,
                        args=args,
                        relative_path=relative_path,
                    )
                    if sanitized_page_map:
                        ocr_doc["layout_sanitize_json"] = layout_json_path.as_posix() if layout_json_path else ""
                        sanitized_layout_map = (
                            load_layout_sanitize_layout_map(layout_json_path) if layout_json_path is not None else {}
                        )
                        log_progress(
                            f"[{relative_path}] Layout sanitize ready {len(sanitized_page_map)}/{len(page_numbers)} pages"
                        )
                log_progress(f"[{relative_path}] OCR backend={args.backend} route={route_hint}")
                ocr_iter = ocr_agent.iterate_document_pages(
                    document_path,
                    pages=ocr_missing,
                    route_hint=route_hint,
                    sanitized_page_map=sanitized_page_map,
                    sanitized_layout_map=sanitized_layout_map,
                )
                for index, expected_page_number in enumerate(ocr_missing, start=1):
                    page_state = page_states[expected_page_number]
                    try:
                        page_result = run_page_step(
                            progress_tracker,
                            relative_path,
                            "OCR",
                            index,
                            len(ocr_missing),
                            expected_page_number,
                            lambda ocr_iter=ocr_iter: next(ocr_iter),
                        )
                        ocr_agent.run(page_state, page_result=page_result, route_hint=route_hint)
                        page_commander.run(page_state, stage="ocr_route", ocr_payload=page_state.stage_payloads["ocr"])
                        checkpoint_store.save_page(page_state)
                    except Exception as exc:
                        page_state.record_provenance(
                            agent="orchestrator",
                            input_fields=[],
                            output_fields=["errors"],
                            note=f"ocr_failed:{type(exc).__name__}",
                        )
                        mark_failed(page_state, agent="OCRStage", error=f"{type(exc).__name__}: {exc}")
                        checkpoint_store.save_page(page_state)
                        raise
                    ocr_doc["pages"] = ordered_stage_pages(page_states, page_numbers, "ocr")
                    write_json(ocr_path, ocr_doc)
            else:
                log_progress(f"[{relative_path}] OCR already complete")

            repeated_headers, repeated_footers = cleaning_agent.detect_repeated_edges(ocr_doc["pages"])
            rule_cleaned_doc = legacy_rule_cleaned_doc or {
                "relative_path": ocr_doc["relative_path"],
                "route_hint": route_hint,
                "repeated_headers": sorted(repeated_headers),
                "repeated_footers": sorted(repeated_footers),
                "pages": [],
            }
            rule_cleaned_doc["relative_path"] = ocr_doc["relative_path"]
            rule_cleaned_doc["route_hint"] = route_hint
            rule_cleaned_doc["repeated_headers"] = sorted(repeated_headers)
            rule_cleaned_doc["repeated_footers"] = sorted(repeated_footers)
            rule_cleaned_doc["pages"] = ordered_stage_pages(page_states, page_numbers, "rule_cleaned")
            if rule_cleaned_doc["pages"]:
                log_progress(
                    f"[{relative_path}] Reusing rule-cleaned checkpoint {len(rule_cleaned_doc['pages'])}/{len(page_numbers)} pages"
                )
            rule_missing = [
                page_number
                for page_number in page_numbers
                if not state_at_least(effective_state(page_states[page_number]), PageProcessingState.RULE_CLEANED)
            ]
            if rule_missing:
                log_progress(f"[{relative_path}] Rule cleaning")
                for index, page_number in enumerate(rule_missing, start=1):
                    page_state = page_states[page_number]
                    try:
                        run_page_step(
                            progress_tracker,
                            relative_path,
                            "Rule cleaning",
                            index,
                            len(rule_missing),
                            page_number,
                            lambda page_state=page_state: cleaning_agent.run(
                                page_state,
                                repeated_headers=repeated_headers,
                                repeated_footers=repeated_footers,
                            ),
                        )
                        checkpoint_store.save_page(page_state)
                    except Exception as exc:
                        page_state.record_provenance(
                            agent="orchestrator",
                            input_fields=["stage_payloads.ocr"],
                            output_fields=["errors"],
                            note=f"rule_cleaning_failed:{type(exc).__name__}",
                        )
                        mark_failed(page_state, agent="RuleCleaningStage", error=f"{type(exc).__name__}: {exc}")
                        checkpoint_store.save_page(page_state)
                        raise
                    rule_cleaned_doc["pages"] = ordered_stage_pages(page_states, page_numbers, "rule_cleaned")
                    write_json(rule_cleaned_path, rule_cleaned_doc)
            else:
                log_progress(f"[{relative_path}] Rule cleaning already complete")

            audit_updates = 0
            audit_fixed = 0
            audit_warned = 0
            for page_number in page_numbers:
                page_state = page_states[page_number]
                if not state_at_least(effective_state(page_state), PageProcessingState.RULE_CLEANED):
                    continue
                if apply_rule_cleaned_homoglyph_audit(page_state):
                    checkpoint_store.save_page(page_state)
                    audit_updates += 1
                    audit_payload = page_state.stage_payloads.get("rule_cleaned", {}).get("homoglyph_audit", {})
                    audit_fixed += int(audit_payload.get("auto_fixed", 0) or 0)
                    audit_warned += int(audit_payload.get("warned", 0) or 0)
            if audit_updates:
                rule_cleaned_doc["pages"] = ordered_stage_pages(page_states, page_numbers, "rule_cleaned")
                write_json(rule_cleaned_path, rule_cleaned_doc)
                log_progress(
                    f"[{relative_path}] Homoglyph audit updated {audit_updates} pages "
                    f"(auto_fixed={audit_fixed}, warned={audit_warned})"
                )

            primary_cleaned_doc = legacy_primary_cleaned_doc or {
                "relative_path": ocr_doc["relative_path"],
                "route_hint": route_hint,
                "backend": cleaning_model if primary_cleaning_agent is not None else "rule_only",
                "pages": [],
            }
            primary_cleaned_doc["relative_path"] = ocr_doc["relative_path"]
            primary_cleaned_doc["route_hint"] = route_hint
            primary_cleaned_doc["backend"] = cleaning_model if primary_cleaning_agent is not None else "rule_only"
            primary_cleaned_doc["pages"] = ordered_stage_pages(page_states, page_numbers, "primary_cleaned")
            if primary_cleaned_doc["pages"]:
                log_progress(
                    f"[{relative_path}] Reusing primary-cleaned checkpoint {len(primary_cleaned_doc['pages'])}/{len(page_numbers)} pages"
                )

            primary_cleaned_missing = [
                page_number
                for page_number in page_numbers
                if not state_at_least(effective_state(page_states[page_number]), PageProcessingState.PRIMARY_CLEANED)
            ]
            if primary_cleaned_missing:
                ocr_pages = page_map(ocr_doc)
                rule_pages = page_map(rule_cleaned_doc)
                if primary_cleaning_agent is not None:
                    log_progress(f"[{relative_path}] Primary cleaning start backend={args.cleaning_backend}")
                else:
                    log_progress(f"[{relative_path}] Primary cleaning bypassed; promoting rule-cleaned pages")
                if isinstance(primary_cleaning_agent, DeepSeekCleaningAgent) and args.deepseek_max_concurrency > 1:
                    async def run_primary_cleaning_async(page_number: int) -> None:
                        page_state = page_states[page_number]
                        rule_page = rule_pages.get(page_number)
                        page_commander.run(page_state, stage="primary_cleaning", rule_page=rule_page)
                        decision = page_state.stage_payloads.get("primary_cleaning_plan", {})
                        if decision.get("action") == "skip_primary_cleaning" and rule_page is not None:
                            final_page = dict(rule_page)
                            final_page["status"] = "rule_only"
                            notes = list(final_page.get("notes", []))
                            notes.append(f"commander_skipped_primary_cleaning:{decision.get('reason', '')}")
                            final_page["notes"] = notes
                            final_page["llm_edits"] = []
                            final_page["llm_edit_plan"] = None
                        else:
                            final_page = await primary_cleaning_agent.clean_page_async(
                                ocr_pages[page_number],
                                rule_page,
                            )

                        page_state.primary_clean_text = str(final_page.get("cleaned_text") or "")
                        page_state.edit_plan = final_page.get("llm_edit_plan")
                        page_state.stage_payloads["primary_cleaned"] = final_page
                        page_state.record_provenance(
                            agent="PrimaryCleaningStage",
                            input_fields=["stage_payloads.ocr", "stage_payloads.rule_cleaned"],
                            output_fields=["primary_clean_text", "edit_plan", "stage_payloads.primary_cleaned"],
                            note=str(final_page.get("status") or "primary_cleaned"),
                        )
                        transition(
                            page_state,
                            PageProcessingState.PRIMARY_CLEANED,
                            agent="PrimaryCleaningStage",
                            note=str(final_page.get("status") or ""),
                        )

                    def primary_cleaning_success(page_number: int) -> None:
                        checkpoint_store.save_page(page_states[page_number])
                        primary_cleaned_doc["pages"] = ordered_stage_pages(page_states, page_numbers, "primary_cleaned")
                        write_json(primary_cleaned_path, primary_cleaned_doc)

                    def primary_cleaning_error(page_number: int, exc: Exception) -> None:
                        page_state = page_states[page_number]
                        page_state.record_provenance(
                            agent="orchestrator",
                            input_fields=["stage_payloads.rule_cleaned"],
                            output_fields=["errors"],
                            note=f"primary_cleaning_failed:{type(exc).__name__}",
                        )
                        mark_failed(page_state, agent="PrimaryCleaningStage", error=f"{type(exc).__name__}: {exc}")
                        checkpoint_store.save_page(page_state)

                    asyncio.run(
                        run_async_stage_pages(
                            tracker=progress_tracker,
                            relative_path=relative_path,
                            stage="Primary cleaning",
                            page_numbers=primary_cleaned_missing,
                            max_concurrency=args.deepseek_max_concurrency,
                            worker=run_primary_cleaning_async,
                            on_success=primary_cleaning_success,
                            on_error=primary_cleaning_error,
                        )
                    )
                else:
                    for index, page_number in enumerate(primary_cleaned_missing, start=1):
                        page_state = page_states[page_number]

                        def run_primary_cleaning(page_state: PageState = page_state, page_number: int = page_number):
                            rule_page = rule_pages.get(page_number)
                            page_commander.run(page_state, stage="primary_cleaning", rule_page=rule_page)
                            decision = page_state.stage_payloads.get("primary_cleaning_plan", {})
                            if primary_cleaning_agent is None:
                                final_page = dict(rule_page or page_state_to_rule_page(page_state))
                                final_page["status"] = "rule_only"
                                final_page["llm_edits"] = []
                                final_page["llm_edit_plan"] = None
                            elif decision.get("action") == "skip_primary_cleaning" and rule_page is not None:
                                final_page = dict(rule_page)
                                final_page["status"] = "rule_only"
                                notes = list(final_page.get("notes", []))
                                notes.append(f"commander_skipped_primary_cleaning:{decision.get('reason', '')}")
                                final_page["notes"] = notes
                                final_page["llm_edits"] = []
                                final_page["llm_edit_plan"] = None
                            else:
                                final_page = primary_cleaning_agent.clean_page(
                                    ocr_pages[page_number],
                                    rule_page,
                                )

                            page_state.primary_clean_text = str(final_page.get("cleaned_text") or "")
                            page_state.edit_plan = final_page.get("llm_edit_plan")
                            page_state.stage_payloads["primary_cleaned"] = final_page
                            page_state.record_provenance(
                                agent="PrimaryCleaningStage",
                                input_fields=["stage_payloads.ocr", "stage_payloads.rule_cleaned"],
                                output_fields=["primary_clean_text", "edit_plan", "stage_payloads.primary_cleaned"],
                                note=str(final_page.get("status") or "primary_cleaned"),
                            )
                            transition(
                                page_state,
                                PageProcessingState.PRIMARY_CLEANED,
                                agent="PrimaryCleaningStage",
                                note=str(final_page.get("status") or ""),
                            )
                            return page_state

                        try:
                            run_page_step(
                                progress_tracker,
                                relative_path,
                                "Primary cleaning",
                                index,
                                len(primary_cleaned_missing),
                                page_number,
                                run_primary_cleaning,
                            )
                            checkpoint_store.save_page(page_state)
                        except Exception as exc:
                            page_state.record_provenance(
                                agent="orchestrator",
                                input_fields=["stage_payloads.rule_cleaned"],
                                output_fields=["errors"],
                                note=f"primary_cleaning_failed:{type(exc).__name__}",
                            )
                            mark_failed(page_state, agent="PrimaryCleaningStage", error=f"{type(exc).__name__}: {exc}")
                            checkpoint_store.save_page(page_state)
                            raise
                        primary_cleaned_doc["pages"] = ordered_stage_pages(page_states, page_numbers, "primary_cleaned")
                        write_json(primary_cleaned_path, primary_cleaned_doc)
                log_progress(f"[{relative_path}] Primary cleaning done")
            else:
                log_progress(f"[{relative_path}] Primary cleaning already complete")

            if primary_cleaning_agent is None:
                write_json(primary_cleaned_path, primary_cleaned_doc)

            cleaned_path = book_dir / "cleaned.json"
            if cleaning_escalation_agent is not None:
                cleaned_doc = read_json(cleaned_path) or {
                    "relative_path": primary_cleaned_doc["relative_path"],
                    "route_hint": route_hint,
                    "backend": cleaning_escalation_model,
                    "pages": [],
                }
                cleaned_doc["relative_path"] = primary_cleaned_doc["relative_path"]
                cleaned_doc["route_hint"] = route_hint
                cleaned_doc["backend"] = cleaning_escalation_model
                cleaned_pages = page_map(cleaned_doc)
                if cleaned_pages:
                    log_progress(
                        f"[{relative_path}] Reusing cleaning-escalation checkpoint {len(cleaned_pages)}/{len(page_numbers)} pages"
                    )
                cleaned_missing = [page_number for page_number in page_numbers if page_number not in cleaned_pages]
                if cleaned_missing:
                    log_progress(f"[{relative_path}] Cleaning escalation start backend={args.cleaning_escalation_backend}")
                    ocr_pages = page_map(ocr_doc)
                    primary_pages = page_map(primary_cleaned_doc)
                    if isinstance(cleaning_escalation_agent, DeepSeekCleaningAgent) and args.deepseek_max_concurrency > 1:
                        async def run_cleaning_escalation_async(page_number: int) -> None:
                            primary_page = primary_pages[page_number]
                            if page_is_trivially_empty_or_dropped(primary_page):
                                final_page = dict(primary_page)
                                final_page["cleaning_stage"] = "skipped_safe_page"
                            else:
                                pre_review = review_agent.review_page(ocr_pages[page_number], primary_page)
                                if should_escalate_cleaning(ocr_pages[page_number], primary_page, pre_review):
                                    final_page = await cleaning_escalation_agent.clean_page_async(
                                        ocr_pages[page_number],
                                        primary_page,
                                    )
                                    final_page["cleaning_stage"] = "escalated"
                                else:
                                    final_page = dict(primary_page)
                                    final_page["cleaning_stage"] = "primary"
                            cleaned_pages[page_number] = final_page

                        def cleaning_escalation_success(page_number: int) -> None:
                            cleaned_doc["pages"] = ordered_pages(cleaned_pages, page_numbers)
                            write_json(cleaned_path, cleaned_doc)

                        def cleaning_escalation_error(page_number: int, exc: Exception) -> None:
                            page_state = page_states[page_number]
                            page_state.record_provenance(
                                agent="orchestrator",
                                input_fields=["stage_payloads.primary_cleaned"],
                                output_fields=["errors"],
                                note=f"cleaning_escalation_failed:{type(exc).__name__}",
                            )
                            mark_failed(page_state, agent="CleaningEscalationStage", error=f"{type(exc).__name__}: {exc}")
                            checkpoint_store.save_page(page_state)

                        asyncio.run(
                            run_async_stage_pages(
                                tracker=progress_tracker,
                                relative_path=relative_path,
                                stage="Cleaning escalation",
                                page_numbers=cleaned_missing,
                                max_concurrency=args.deepseek_max_concurrency,
                                worker=run_cleaning_escalation_async,
                                on_success=cleaning_escalation_success,
                                on_error=cleaning_escalation_error,
                            )
                        )
                    else:
                        for index, page_number in enumerate(cleaned_missing, start=1):
                            def run_cleaning_escalation(page_number: int = page_number):
                                primary_page = primary_pages[page_number]
                                if page_is_trivially_empty_or_dropped(primary_page):
                                    final_page = dict(primary_page)
                                    final_page["cleaning_stage"] = "skipped_safe_page"
                                    return final_page
                                pre_review = review_agent.review_page(ocr_pages[page_number], primary_page)
                                if should_escalate_cleaning(ocr_pages[page_number], primary_page, pre_review):
                                    final_page = cleaning_escalation_agent.clean_page(ocr_pages[page_number], primary_page)
                                    final_page["cleaning_stage"] = "escalated"
                                    return final_page
                                final_page = dict(primary_page)
                                final_page["cleaning_stage"] = "primary"
                                return final_page

                            cleaned_pages[page_number] = run_page_step(
                                progress_tracker,
                                relative_path,
                                "Cleaning escalation",
                                index,
                                len(cleaned_missing),
                                page_number,
                                run_cleaning_escalation,
                            )
                            cleaned_doc["pages"] = ordered_pages(cleaned_pages, page_numbers)
                            write_json(cleaned_path, cleaned_doc)
                    log_progress(f"[{relative_path}] Cleaning escalation done")
                else:
                    log_progress(f"[{relative_path}] Cleaning escalation already complete")
            else:
                cleaned_doc = primary_cleaned_doc
                write_json(cleaned_path, cleaned_doc)

            sync_cleaned_stage_payloads(page_states, cleaned_doc)
            review_path = book_dir / "review.json"
            legacy_reviewed_doc = read_json(review_path)
            gemini_review_path = book_dir / "gemini_review.json"
            legacy_gemini_review_doc = read_json(gemini_review_path)
            primary_repaired_path = book_dir / ("repaired_primary.json" if repair_escalation_agent is not None else "repaired.json")
            legacy_primary_repaired_doc = read_json(primary_repaired_path)
            repaired_path = book_dir / "repaired.json"
            legacy_repaired_doc = read_json(repaired_path)
            gemini_structure_path = book_dir / "gemini_structure.json"
            legacy_restored_doc = read_json(gemini_structure_path)
            backfill_review_and_repair_states(
                page_states=page_states,
                page_numbers=page_numbers,
                reviewed_doc=legacy_reviewed_doc,
                gemini_review_doc=legacy_gemini_review_doc,
                primary_repaired_doc=legacy_primary_repaired_doc,
                repaired_doc=legacy_repaired_doc,
            )
            backfill_structure_states(
                page_states=page_states,
                page_numbers=page_numbers,
                restored_doc=legacy_restored_doc,
            )

            reviewed_doc = legacy_reviewed_doc or {
                "relative_path": cleaned_doc["relative_path"],
                "route_hint": route_hint,
                "pages": [],
            }
            reviewed_doc["relative_path"] = cleaned_doc["relative_path"]
            reviewed_doc["route_hint"] = route_hint
            reviewed_doc["pages"] = ordered_stage_pages(page_states, page_numbers, "review")
            if reviewed_doc["pages"]:
                log_progress(
                    f"[{relative_path}] Reusing heuristic-review checkpoint {len(reviewed_doc['pages'])}/{len(page_numbers)} pages"
                )
            review_missing = [
                page_number
                for page_number in page_numbers
                if not state_at_least(effective_state(page_states[page_number]), PageProcessingState.REVIEWED)
            ]
            if review_missing:
                log_progress(f"[{relative_path}] Heuristic review")
                ocr_pages = page_map(ocr_doc)
                cleaned_pages = page_map(cleaned_doc)
                for index, page_number in enumerate(review_missing, start=1):
                    page_state = page_states[page_number]
                    try:
                        run_page_step(
                            progress_tracker,
                            relative_path,
                            "Heuristic review",
                            index,
                            len(review_missing),
                            page_number,
                            lambda page_state=page_state, page_number=page_number: review_agent.run(
                                page_state,
                                raw_page=ocr_pages[page_number],
                                cleaned_page=cleaned_pages[page_number],
                            ),
                        )
                        checkpoint_store.save_page(page_state)
                    except Exception as exc:
                        page_state.record_provenance(
                            agent="orchestrator",
                            input_fields=["stage_payloads.cleaned"],
                            output_fields=["errors"],
                            note=f"review_failed:{type(exc).__name__}",
                        )
                        mark_failed(page_state, agent="ReviewStage", error=f"{type(exc).__name__}: {exc}")
                        checkpoint_store.save_page(page_state)
                        raise
                    reviewed_doc["pages"] = ordered_stage_pages(page_states, page_numbers, "review")
                    write_json(review_path, reviewed_doc)
            else:
                log_progress(f"[{relative_path}] Heuristic review already complete")

            gemini_review_doc = None
            if gemini_review_agent is not None:
                gemini_review_doc = legacy_gemini_review_doc or {
                    "relative_path": cleaned_doc["relative_path"],
                    "pages": [],
                }
                gemini_review_doc["relative_path"] = cleaned_doc["relative_path"]
                gemini_review_doc["pages"] = ordered_stage_pages(page_states, page_numbers, "gemini_review")
                if gemini_review_doc["pages"]:
                    log_progress(
                        f"[{relative_path}] Reusing Gemini-review checkpoint {len(gemini_review_doc['pages'])}/{len(page_numbers)} pages"
                    )
                gemini_review_missing = [
                    page_number
                    for page_number in page_numbers
                    if state_at_least(effective_state(page_states[page_number]), PageProcessingState.REVIEWED)
                    and not state_at_least(effective_state(page_states[page_number]), PageProcessingState.REPAIRED)
                    and "gemini_review" not in page_states[page_number].stage_payloads
                ]
                if gemini_review_missing:
                    log_progress(f"[{relative_path}] Gemini review start")
                    cleaned_pages = page_map(cleaned_doc)
                    reviewed_pages = page_map(reviewed_doc)
                    for index, page_number in enumerate(gemini_review_missing, start=1):
                        page_state = page_states[page_number]
                        try:
                            run_page_step(
                                progress_tracker,
                                relative_path,
                                "Gemini review",
                                index,
                                len(gemini_review_missing),
                                page_number,
                                lambda page_state=page_state, page_number=page_number: gemini_review_agent.run(
                                    page_state,
                                    cleaned_page=cleaned_pages[page_number],
                                    heuristic_page=reviewed_pages[page_number],
                                ),
                            )
                            checkpoint_store.save_page(page_state)
                        except Exception as exc:
                            page_state.record_provenance(
                                agent="orchestrator",
                                input_fields=["stage_payloads.review"],
                                output_fields=["errors"],
                                note=f"gemini_review_failed:{type(exc).__name__}",
                            )
                            mark_failed(page_state, agent="GeminiReviewStage", error=f"{type(exc).__name__}: {exc}")
                            checkpoint_store.save_page(page_state)
                            raise
                        gemini_review_doc["pages"] = ordered_stage_pages(page_states, page_numbers, "gemini_review")
                        write_json(gemini_review_path, gemini_review_doc)
                    log_progress(f"[{relative_path}] Gemini review done")
                else:
                    log_progress(f"[{relative_path}] Gemini review already complete")

            early_exit_count = 0
            cleaned_pages = page_map(cleaned_doc)
            reviewed_pages = page_map(reviewed_doc)
            for page_number in page_numbers:
                page_state = page_states[page_number]
                if not state_at_least(effective_state(page_state), PageProcessingState.REVIEWED):
                    continue
                if state_at_least(effective_state(page_state), PageProcessingState.STRUCTURE_RESTORED):
                    continue
                if "repaired" in page_state.stage_payloads:
                    continue
                ocr_page = page_state.stage_payloads.get("ocr")
                cleaned_page = cleaned_pages.get(page_number)
                review_page = reviewed_pages.get(page_number)
                if ocr_page is None or cleaned_page is None or review_page is None:
                    continue
                should_exit, reason = should_early_exit_after_review(
                    page_state=page_state,
                    ocr_page=ocr_page,
                    cleaned_page=cleaned_page,
                    review_page=review_page,
                    gemini_review_enabled=gemini_review_agent is not None,
                )
                if not should_exit:
                    continue
                apply_early_exit_after_review(
                    page_state,
                    cleaned_page=cleaned_page,
                    reason=reason,
                    backend=structure_backend_label,
                    model=final_structure_model or args.final_structure_backend,
                    include_primary_payload=repair_escalation_agent is not None,
                )
                checkpoint_store.save_page(page_state)
                early_exit_count += 1
            if early_exit_count:
                log_progress(f"[{relative_path}] Early exit promoted {early_exit_count} safe pages")

            repaired_doc = None
            primary_repaired_doc = None
            if primary_repair_agent is not None:
                primary_repaired_doc = legacy_primary_repaired_doc or {
                    "relative_path": cleaned_doc["relative_path"],
                    "route_hint": route_hint,
                    "backend": repair_model,
                    "pages": [],
                }
                primary_repaired_doc["relative_path"] = cleaned_doc["relative_path"]
                primary_repaired_doc["route_hint"] = route_hint
                primary_repaired_doc["backend"] = repair_model
                primary_repaired_doc["pages"] = ordered_stage_pages(
                    page_states,
                    page_numbers,
                    "repaired_primary" if repair_escalation_agent is not None else "repaired",
                )
                if primary_repaired_doc["pages"]:
                    log_progress(
                        f"[{relative_path}] Reusing primary-repair checkpoint {len(primary_repaired_doc['pages'])}/{len(page_numbers)} pages"
                    )
                primary_repaired_missing = [
                    page_number
                    for page_number in page_numbers
                    if "repaired_primary" not in page_states[page_number].stage_payloads
                    and not state_at_least(effective_state(page_states[page_number]), PageProcessingState.REPAIRED)
                ]
                if primary_repaired_missing:
                    log_progress(f"[{relative_path}] Primary repair start backend={args.repair_backend}")
                    ocr_pages = page_map(ocr_doc)
                    cleaned_pages = page_map(cleaned_doc)
                    reviewed_pages = page_map(reviewed_doc)
                    gemini_review_pages = page_map(gemini_review_doc or {"pages": []})
                    if isinstance(primary_repair_agent, DeepSeekRepairAgent) and args.deepseek_max_concurrency > 1:
                        async def run_primary_repair_async(page_number: int) -> None:
                            page_state = page_states[page_number]
                            cleaned_page = cleaned_pages[page_number]
                            if page_is_trivially_empty_or_dropped(cleaned_page):
                                repaired_page = dict(cleaned_page)
                                repaired_page["repair_status"] = "skipped_safe_page"
                                repaired_page["repair_notes"] = ["skipped_empty_or_dropped_page"]
                                repaired_page["repair_issue_tags"] = list(page_state.review_tags)
                                repaired_page["repaired_text"] = repaired_page.get("cleaned_text", "")
                                page_state.repaired_text = str(repaired_page.get("cleaned_text") or "")
                                page_state.repair_plan = repaired_page.get("llm_repair_plan")
                                page_state.stage_payloads["repaired"] = repaired_page
                                page_state.record_provenance(
                                    agent="PrimaryRepairStage",
                                    input_fields=["stage_payloads.cleaned", "review_tags", "risk_level"],
                                    output_fields=["repaired_text", "stage_payloads.repaired"],
                                    note="skipped_safe_page",
                                )
                                transition(
                                    page_state,
                                    PageProcessingState.REPAIRED,
                                    agent="PrimaryRepairStage",
                                    note="skipped_safe_page",
                                )
                                return
                            await primary_repair_agent.run_async(
                                page_state,
                                ocr_page=ocr_pages[page_number],
                                cleaned_page=cleaned_page,
                                heuristic_page=reviewed_pages[page_number],
                                gemini_review_page=gemini_review_pages.get(page_number),
                            )

                        def primary_repair_success(page_number: int) -> None:
                            page_state = page_states[page_number]
                            if repair_escalation_agent is not None and "repaired" in page_state.stage_payloads:
                                page_state.stage_payloads["repaired_primary"] = dict(page_state.stage_payloads["repaired"])
                                page_state.stage_payloads.pop("repaired", None)
                            checkpoint_store.save_page(page_state)
                            primary_repaired_doc["pages"] = ordered_stage_pages(
                                page_states,
                                page_numbers,
                                "repaired_primary" if repair_escalation_agent is not None else "repaired",
                            )
                            write_json(primary_repaired_path, primary_repaired_doc)

                        def primary_repair_error(page_number: int, exc: Exception) -> None:
                            page_state = page_states[page_number]
                            page_state.record_provenance(
                                agent="orchestrator",
                                input_fields=["stage_payloads.review"],
                                output_fields=["errors"],
                                note=f"primary_repair_failed:{type(exc).__name__}",
                            )
                            mark_failed(page_state, agent="PrimaryRepairStage", error=f"{type(exc).__name__}: {exc}")
                            checkpoint_store.save_page(page_state)

                        asyncio.run(
                            run_async_stage_pages(
                                tracker=progress_tracker,
                                relative_path=relative_path,
                                stage="Primary repair",
                                page_numbers=primary_repaired_missing,
                                max_concurrency=args.deepseek_max_concurrency,
                                worker=run_primary_repair_async,
                                on_success=primary_repair_success,
                                on_error=primary_repair_error,
                            )
                        )
                    else:
                        for index, page_number in enumerate(primary_repaired_missing, start=1):
                            page_state = page_states[page_number]

                            def run_primary_repair(page_state: PageState = page_state, page_number: int = page_number):
                                cleaned_page = cleaned_pages[page_number]
                                if page_is_trivially_empty_or_dropped(cleaned_page):
                                    repaired_page = dict(cleaned_page)
                                    repaired_page["repair_status"] = "skipped_safe_page"
                                    repaired_page["repair_notes"] = ["skipped_empty_or_dropped_page"]
                                    repaired_page["repair_issue_tags"] = list(page_state.review_tags)
                                    repaired_page["repaired_text"] = repaired_page.get("cleaned_text", "")
                                    page_state.repaired_text = str(repaired_page.get("cleaned_text") or "")
                                    page_state.repair_plan = repaired_page.get("llm_repair_plan")
                                    page_state.stage_payloads["repaired"] = repaired_page
                                    page_state.record_provenance(
                                        agent="PrimaryRepairStage",
                                        input_fields=["stage_payloads.cleaned", "review_tags", "risk_level"],
                                        output_fields=["repaired_text", "stage_payloads.repaired"],
                                        note="skipped_safe_page",
                                    )
                                    transition(
                                        page_state,
                                        PageProcessingState.REPAIRED,
                                        agent="PrimaryRepairStage",
                                        note="skipped_safe_page",
                                    )
                                    return page_state
                                primary_repair_agent.run(
                                    page_state,
                                    ocr_page=ocr_pages[page_number],
                                    cleaned_page=cleaned_page,
                                    heuristic_page=reviewed_pages[page_number],
                                    gemini_review_page=gemini_review_pages.get(page_number),
                                )
                                return page_state

                            try:
                                run_page_step(
                                    progress_tracker,
                                    relative_path,
                                    "Primary repair",
                                    index,
                                    len(primary_repaired_missing),
                                    page_number,
                                    run_primary_repair,
                                )
                                if repair_escalation_agent is not None and "repaired" in page_state.stage_payloads:
                                    page_state.stage_payloads["repaired_primary"] = dict(page_state.stage_payloads["repaired"])
                                    page_state.stage_payloads.pop("repaired", None)
                                checkpoint_store.save_page(page_state)
                            except Exception as exc:
                                page_state.record_provenance(
                                    agent="orchestrator",
                                    input_fields=["stage_payloads.review"],
                                    output_fields=["errors"],
                                    note=f"primary_repair_failed:{type(exc).__name__}",
                                )
                                mark_failed(page_state, agent="PrimaryRepairStage", error=f"{type(exc).__name__}: {exc}")
                                checkpoint_store.save_page(page_state)
                                raise
                            primary_repaired_doc["pages"] = ordered_stage_pages(
                                page_states,
                                page_numbers,
                                "repaired_primary" if repair_escalation_agent is not None else "repaired",
                            )
                            write_json(primary_repaired_path, primary_repaired_doc)
                    log_progress(f"[{relative_path}] Primary repair done")
                else:
                    log_progress(f"[{relative_path}] Primary repair already complete")

            if repair_escalation_agent is not None:
                base_doc = primary_repaired_doc or cleaned_doc
                repaired_doc = legacy_repaired_doc or {
                    "relative_path": base_doc["relative_path"],
                    "route_hint": route_hint,
                    "backend": repair_escalation_model,
                    "pages": [],
                }
                repaired_doc["relative_path"] = base_doc["relative_path"]
                repaired_doc["route_hint"] = route_hint
                repaired_doc["backend"] = repair_escalation_model
                repaired_doc["pages"] = ordered_stage_pages(page_states, page_numbers, "repaired")
                if repaired_doc["pages"]:
                    log_progress(
                        f"[{relative_path}] Reusing repair-escalation checkpoint {len(repaired_doc['pages'])}/{len(page_numbers)} pages"
                    )
                repaired_missing = [
                    page_number
                    for page_number in page_numbers
                    if "repaired" not in page_states[page_number].stage_payloads
                ]
                if repaired_missing:
                    log_progress(f"[{relative_path}] Repair escalation start backend={args.repair_escalation_backend}")
                    ocr_pages = page_map(ocr_doc)
                    base_pages = page_map(base_doc)
                    reviewed_pages = page_map(reviewed_doc)
                    gemini_review_pages = page_map(gemini_review_doc or {"pages": []})
                    if isinstance(repair_escalation_agent, DeepSeekRepairAgent) and args.deepseek_max_concurrency > 1:
                        async def run_repair_escalation_async(page_number: int) -> None:
                            page_state = page_states[page_number]
                            base_page = base_pages[page_number]
                            if page_is_trivially_empty_or_dropped(base_page):
                                final_page = dict(base_page)
                                final_page["repair_stage"] = "skipped_safe_page"
                                final_page["repair_issue_tags"] = list(page_state.review_tags)
                            elif should_escalate_repair(ocr_pages[page_number], base_page, reviewed_pages[page_number]):
                                final_page = await repair_escalation_agent.repair_page_async(
                                    ocr_pages[page_number],
                                    base_page,
                                    reviewed_pages[page_number],
                                    gemini_review_pages.get(page_number),
                                    review_tags=page_state.review_tags,
                                    risk_level=page_state.risk_level,
                                )
                                final_page["repair_stage"] = "escalated"
                            else:
                                final_page = dict(base_page)
                                final_page["repair_stage"] = "primary" if primary_repaired_doc is not None else "none"
                                final_page["repair_issue_tags"] = list(page_state.review_tags)
                            page_state.repaired_text = str(
                                final_page.get("repaired_text") or final_page.get("cleaned_text") or page_state.repaired_text
                            )
                            page_state.repair_plan = final_page.get("llm_repair_plan") or page_state.repair_plan
                            page_state.stage_payloads["repaired"] = final_page
                            page_state.record_provenance(
                                agent="RepairEscalationStage",
                                input_fields=["stage_payloads.review", "stage_payloads.repaired_primary"],
                                output_fields=["repaired_text", "repair_plan", "stage_payloads.repaired"],
                                note=str(final_page.get("repair_stage") or "unknown"),
                            )
                            if not state_at_least(effective_state(page_state), PageProcessingState.REPAIRED):
                                transition(
                                    page_state,
                                    PageProcessingState.REPAIRED,
                                    agent="RepairEscalationStage",
                                    note=str(final_page.get("repair_stage") or ""),
                                )

                        def repair_escalation_success(page_number: int) -> None:
                            checkpoint_store.save_page(page_states[page_number])
                            repaired_doc["pages"] = ordered_stage_pages(page_states, page_numbers, "repaired")
                            write_json(repaired_path, repaired_doc)

                        def repair_escalation_error(page_number: int, exc: Exception) -> None:
                            page_state = page_states[page_number]
                            page_state.record_provenance(
                                agent="orchestrator",
                                input_fields=["stage_payloads.review", "stage_payloads.repaired_primary"],
                                output_fields=["errors"],
                                note=f"repair_escalation_failed:{type(exc).__name__}",
                            )
                            mark_failed(page_state, agent="RepairEscalationStage", error=f"{type(exc).__name__}: {exc}")
                            checkpoint_store.save_page(page_state)

                        asyncio.run(
                            run_async_stage_pages(
                                tracker=progress_tracker,
                                relative_path=relative_path,
                                stage="Repair escalation",
                                page_numbers=repaired_missing,
                                max_concurrency=args.deepseek_max_concurrency,
                                worker=run_repair_escalation_async,
                                on_success=repair_escalation_success,
                                on_error=repair_escalation_error,
                            )
                        )
                    else:
                        for index, page_number in enumerate(repaired_missing, start=1):
                            page_state = page_states[page_number]

                            def run_repair_escalation(page_state: PageState = page_state, page_number: int = page_number):
                                base_page = base_pages[page_number]
                                if page_is_trivially_empty_or_dropped(base_page):
                                    final_page = dict(base_page)
                                    final_page["repair_stage"] = "skipped_safe_page"
                                    final_page["repair_issue_tags"] = list(page_state.review_tags)
                                elif should_escalate_repair(ocr_pages[page_number], base_page, reviewed_pages[page_number]):
                                    final_page = repair_escalation_agent.repair_page(
                                        ocr_pages[page_number],
                                        base_page,
                                        reviewed_pages[page_number],
                                        gemini_review_pages.get(page_number),
                                        review_tags=page_state.review_tags,
                                        risk_level=page_state.risk_level,
                                    )
                                    final_page["repair_stage"] = "escalated"
                                else:
                                    final_page = dict(base_page)
                                    final_page["repair_stage"] = "primary" if primary_repaired_doc is not None else "none"
                                    final_page["repair_issue_tags"] = list(page_state.review_tags)
                                page_state.repaired_text = str(
                                    final_page.get("repaired_text") or final_page.get("cleaned_text") or page_state.repaired_text
                                )
                                page_state.repair_plan = final_page.get("llm_repair_plan") or page_state.repair_plan
                                page_state.stage_payloads["repaired"] = final_page
                                page_state.record_provenance(
                                    agent="RepairEscalationStage",
                                    input_fields=["stage_payloads.review", "stage_payloads.repaired_primary"],
                                    output_fields=["repaired_text", "repair_plan", "stage_payloads.repaired"],
                                    note=str(final_page.get("repair_stage") or "unknown"),
                                )
                                if not state_at_least(effective_state(page_state), PageProcessingState.REPAIRED):
                                    transition(
                                        page_state,
                                        PageProcessingState.REPAIRED,
                                        agent="RepairEscalationStage",
                                        note=str(final_page.get("repair_stage") or ""),
                                    )
                                return page_state

                            try:
                                run_page_step(
                                    progress_tracker,
                                    relative_path,
                                    "Repair escalation",
                                    index,
                                    len(repaired_missing),
                                    page_number,
                                    run_repair_escalation,
                                )
                                checkpoint_store.save_page(page_state)
                            except Exception as exc:
                                page_state.record_provenance(
                                    agent="orchestrator",
                                    input_fields=["stage_payloads.review", "stage_payloads.repaired_primary"],
                                    output_fields=["errors"],
                                    note=f"repair_escalation_failed:{type(exc).__name__}",
                                )
                                mark_failed(page_state, agent="RepairEscalationStage", error=f"{type(exc).__name__}: {exc}")
                                checkpoint_store.save_page(page_state)
                                raise
                            repaired_doc["pages"] = ordered_stage_pages(page_states, page_numbers, "repaired")
                            write_json(repaired_path, repaired_doc)
                    log_progress(f"[{relative_path}] Repair escalation done")
                else:
                    log_progress(f"[{relative_path}] Repair escalation already complete")
            elif primary_repaired_doc is not None:
                repaired_doc = primary_repaired_doc
                write_json(book_dir / "repaired.json", repaired_doc)

            restored_doc = None
            structure_input_doc = repaired_doc or cleaned_doc
            restored_doc = legacy_restored_doc if structure_agent is not None else {
                "relative_path": structure_input_doc["relative_path"],
                "model": final_structure_model or args.final_structure_backend,
                "risky_only": args.final_structure_risky_only,
                "pages": [],
            }
            restored_doc = restored_doc or {
                "relative_path": structure_input_doc["relative_path"],
                "model": final_structure_model or args.final_structure_backend,
                "risky_only": args.final_structure_risky_only,
                "pages": [],
            }
            restored_doc["relative_path"] = structure_input_doc["relative_path"]
            restored_doc["model"] = final_structure_model or args.final_structure_backend
            restored_doc["risky_only"] = args.final_structure_risky_only
            restored_doc["pages"] = ordered_stage_pages(page_states, page_numbers, "restored")
            restored_count = len(restored_doc["pages"])
            if args.final_structure_backend == "gemini":
                structure_stage_label = "Gemini structure"
            elif args.final_structure_backend == "deepseek":
                structure_stage_label = "DeepSeek structure"
            else:
                structure_stage_label = "Structure passthrough"
            if restored_count:
                log_progress(f"[{relative_path}] Reusing structure checkpoint {restored_count}/{len(page_numbers)} pages")
            restored_missing = [
                page_number
                for page_number in page_numbers
                if not state_at_least(effective_state(page_states[page_number]), PageProcessingState.STRUCTURE_RESTORED)
            ]
            if restored_missing:
                log_progress(f"[{relative_path}] {structure_stage_label} start")
                ocr_pages = page_map(ocr_doc)
                reviewed_pages = page_map(reviewed_doc)
                if isinstance(structure_agent, DeepSeekStructureAgent) and args.deepseek_max_concurrency > 1:
                    async def run_structure_stage_async(page_number: int) -> None:
                        page_state = page_states[page_number]
                        ensure_repaired_state_for_structure(page_state)
                        repaired_page = page_state_to_repaired_page(page_state)
                        if page_is_trivially_empty_or_dropped(repaired_page):
                            apply_structure_passthrough(
                                page_state,
                                backend=structure_backend_label,
                                model=final_structure_model or args.final_structure_backend,
                                status="skipped_empty_or_dropped_page",
                                skipped_reason="empty_or_dropped_page",
                                notes=["skipped_empty_or_dropped_page"],
                                text="",
                            )
                            return
                        if should_restore_structure(
                            ocr_pages[page_number],
                            repaired_page,
                            reviewed_pages[page_number],
                            args.final_structure_risky_only,
                        ):
                            await structure_agent.run_async(
                                page_state,
                                ocr_page=ocr_pages[page_number],
                                repaired_page=repaired_page,
                            )
                            return
                        passthrough_reason = "safe_page"
                        passthrough_status = "skipped_safe_page"
                        apply_structure_passthrough(
                            page_state,
                            backend=structure_backend_label,
                            model=final_structure_model or args.final_structure_backend,
                            status=passthrough_status,
                            skipped_reason=passthrough_reason,
                            notes=[passthrough_status],
                        )

                    def structure_success(page_number: int) -> None:
                        checkpoint_store.save_page(page_states[page_number])
                        restored_doc["pages"] = ordered_stage_pages(page_states, page_numbers, "restored")
                        if structure_agent is not None:
                            write_json(gemini_structure_path, restored_doc)

                    def structure_error(page_number: int, exc: Exception) -> None:
                        page_state = page_states[page_number]
                        page_state.record_provenance(
                            agent="orchestrator",
                            input_fields=["repaired_text"],
                            output_fields=["errors"],
                            note=f"structure_failed:{type(exc).__name__}",
                        )
                        mark_failed(page_state, agent="StructureStage", error=f"{type(exc).__name__}: {exc}")
                        checkpoint_store.save_page(page_state)

                    asyncio.run(
                        run_async_stage_pages(
                            tracker=progress_tracker,
                            relative_path=relative_path,
                            stage=structure_stage_label,
                            page_numbers=restored_missing,
                            max_concurrency=args.deepseek_max_concurrency,
                            worker=run_structure_stage_async,
                            on_success=structure_success,
                            on_error=structure_error,
                        )
                    )
                else:
                    for index, page_number in enumerate(restored_missing, start=1):
                        page_state = page_states[page_number]

                        def run_structure_stage(page_state: PageState = page_state, page_number: int = page_number):
                            ensure_repaired_state_for_structure(page_state)
                            repaired_page = page_state_to_repaired_page(page_state)
                            if page_is_trivially_empty_or_dropped(repaired_page):
                                return apply_structure_passthrough(
                                    page_state,
                                    backend=structure_backend_label,
                                    model=final_structure_model or args.final_structure_backend,
                                    status="skipped_empty_or_dropped_page",
                                    skipped_reason="empty_or_dropped_page",
                                    notes=["skipped_empty_or_dropped_page"],
                                    text="",
                                )
                            if structure_agent is not None and should_restore_structure(
                                ocr_pages[page_number],
                                repaired_page,
                                reviewed_pages[page_number],
                                args.final_structure_risky_only,
                            ):
                                return structure_agent.run(
                                    page_state,
                                    ocr_page=ocr_pages[page_number],
                                    repaired_page=repaired_page,
                                )
                            passthrough_reason = "safe_page" if structure_agent is not None else "rules_backend_passthrough"
                            passthrough_status = "skipped_safe_page" if structure_agent is not None else "rules_passthrough"
                            return apply_structure_passthrough(
                                page_state,
                                backend=structure_backend_label,
                                model=final_structure_model or args.final_structure_backend,
                                status=passthrough_status,
                                skipped_reason=passthrough_reason,
                                notes=[passthrough_status],
                            )

                        try:
                            run_page_step(
                                progress_tracker,
                                relative_path,
                                structure_stage_label,
                                index,
                                len(restored_missing),
                                page_number,
                                run_structure_stage,
                            )
                            checkpoint_store.save_page(page_state)
                        except Exception as exc:
                            page_state.record_provenance(
                                agent="orchestrator",
                                input_fields=["repaired_text"],
                                output_fields=["errors"],
                                note=f"structure_failed:{type(exc).__name__}",
                            )
                            mark_failed(page_state, agent="StructureStage", error=f"{type(exc).__name__}: {exc}")
                            checkpoint_store.save_page(page_state)
                            raise
                        restored_doc["pages"] = ordered_stage_pages(page_states, page_numbers, "restored")
                        if structure_agent is not None:
                            write_json(gemini_structure_path, restored_doc)
                log_progress(f"[{relative_path}] {structure_stage_label} done")
            else:
                log_progress(f"[{relative_path}] {structure_stage_label} already complete")

            final_txt_path = args.final_txt_dir / f"{safe_stem}.txt"
            export_complete = final_txt_path.exists() and all(
                state_at_least(effective_state(page_states[page_number]), PageProcessingState.EXPORTED)
                for page_number in page_numbers
            )
            if export_complete:
                log_progress(f"[{relative_path}] Export already complete")
            else:
                export_doc, export_sources = build_export_document(
                    relative_path=relative_path,
                    page_states=page_states,
                    page_numbers=page_numbers,
                    restored_doc=restored_doc,
                    repaired_doc=repaired_doc,
                    cleaned_doc=cleaned_doc,
                )
                final_text = stitch_cleaned_pages(export_doc, text_key="cleaned_text")
                final_txt_path.write_text(final_text, encoding="utf-8")
                run_post_clean_final_txt(final_txt_path)
                for page_number in page_numbers:
                    page_state = page_states[page_number]
                    if page_state.current_state == PageProcessingState.FAILED:
                        page_state.current_state = page_state.last_success_state
                    export_source = export_sources.get(page_number, "page_state_fallback")
                    if not state_at_least(effective_state(page_state), PageProcessingState.EXPORTED):
                        page_state.record_provenance(
                            agent="ExportStage",
                            input_fields=[export_source],
                            output_fields=[],
                            note=f"target={final_txt_path.name};source={export_source}",
                        )
                        transition(
                            page_state,
                            PageProcessingState.EXPORTED,
                            agent="ExportStage",
                            note=f"target={final_txt_path.name};source={export_source}",
                        )
                    checkpoint_store.save_page(page_state)
                log_progress(f"[{relative_path}] Export done")
            summaries.append(
                build_summary(relative_path, ocr_doc, cleaned_doc, reviewed_doc, gemini_review_doc, repaired_doc, restored_doc)
            )
            progress_tracker.note_book_finish(relative_path)
            log_progress(f"[{relative_path}] Finished")

        run_summary = {
            "generated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "profile": args.profile,
            "backend": args.backend,
            "ocr_model": ocr_model,
            "ocr_render_scale": args.ocr_render_scale,
            "layout_sanitize_backend": args.layout_sanitize_backend,
            "layout_sanitize_python": str(args.layout_sanitize_python),
            "layout_sanitize_render_scale": args.layout_sanitize_render_scale,
            "force_ocr_body_pages": args.force_ocr_body_pages,
            "deepseek_max_concurrency": args.deepseek_max_concurrency,
            "cleaning_backend": args.cleaning_backend,
            "cleaning_model": cleaning_model,
            "cleaning_escalation_backend": args.cleaning_escalation_backend,
            "cleaning_escalation_model": cleaning_escalation_model,
            "notes_policy": args.notes_policy,
            "review_backend": args.review_backend,
            "review_model": review_model,
            "repair_backend": args.repair_backend,
            "repair_model": repair_model,
            "repair_escalation_backend": args.repair_escalation_backend,
            "repair_escalation_model": repair_escalation_model,
            "final_structure_backend": args.final_structure_backend,
            "final_structure_model": final_structure_model,
            "final_structure_risky_only": args.final_structure_risky_only,
            "resume": args.resume or args.resume_run_dir is not None,
            "run_dir": run_dir.as_posix(),
            "books": summaries,
        }
        write_json(run_dir / "summary.json", run_summary)
        print(json.dumps(run_summary, ensure_ascii=False, indent=2))
        print(f"Run artifacts written to {run_dir}")
        print(f"Final TXT written to {args.final_txt_dir}")
    finally:
        sleep_preventer.close()
        progress_tracker.close()


if __name__ == "__main__":
    main()

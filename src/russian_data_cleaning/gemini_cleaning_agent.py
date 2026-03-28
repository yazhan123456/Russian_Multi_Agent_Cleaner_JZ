from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable
from typing import Any

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None

from .structured_edits import (
    ALLOWED_INLINE_PATTERN_NAMES,
    ALLOWED_OPERATIONS,
    apply_edit_plan,
    parse_json_object,
    render_numbered_text,
)


WHITESPACE_RE = re.compile(r"\s+")
HEADING_MARKER_RE = re.compile(r"(?m)^(?:[IVXLCDM]+\.)\s+", re.IGNORECASE)


@dataclass
class GeminiCleaningConfig:
    model: str = "gemini-2.5-pro"
    max_output_tokens: int = 4096
    notes_policy: str = "delete"


class GeminiCleaningAgent:
    def __init__(self, config: GeminiCleaningConfig | None = None) -> None:
        self.config = config or GeminiCleaningConfig()
        self.client = self._build_client()

    def clean_document(
        self,
        ocr_document: dict[str, Any],
        hint_document: dict[str, Any] | None = None,
        progress_hook: Callable[[int, int, int], None] | None = None,
    ) -> dict[str, Any]:
        hint_map = {}
        if hint_document is not None:
            hint_map = {page["page_number"]: page for page in hint_document["pages"]}

        cleaned_pages = []
        total = len(ocr_document["pages"])
        for index, ocr_page in enumerate(ocr_document["pages"], start=1):
            if progress_hook is not None:
                progress_hook(index, total, ocr_page["page_number"])
            hint_page = hint_map.get(ocr_page["page_number"])
            cleaned_pages.append(self.clean_page(ocr_page, hint_page))

        return {
            "relative_path": ocr_document["relative_path"],
            "route_hint": ocr_document["route_hint"],
            "backend": self.config.model,
            "pages": cleaned_pages,
        }

    def clean_page(
        self,
        ocr_page: dict[str, Any],
        hint_page: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return self._clean_page(ocr_page, hint_page)

    def _clean_page(
        self,
        ocr_page: dict[str, Any],
        hint_page: dict[str, Any] | None,
    ) -> dict[str, Any]:
        raw_text = (ocr_page.get("body_text") or ocr_page.get("selected_text") or "").strip()
        hint_text = ""
        flags: list[dict[str, Any]] = []
        protected_hits: list[dict[str, Any]] = []
        if hint_page is not None:
            hint_text = (hint_page.get("cleaned_text") or "").strip()
            flags = hint_page.get("flags", [])
            protected_hits = hint_page.get("protected_hits", [])
            if hint_page.get("drop_page"):
                return {
                    "page_number": ocr_page["page_number"],
                    "source": ocr_page["source"],
                    "raw_text": raw_text,
                    "cleaned_text": "",
                    "edits": [],
                    "flags": flags,
                    "protected_hits": protected_hits,
                    "status": "dropped_by_rules",
                    "notes": [hint_page.get("drop_reason", "dropped_by_rules")],
                    "allow_empty_output": True,
                    "drop_page": True,
                    "drop_reason": hint_page.get("drop_reason", "dropped_by_rules"),
                }

        fallback_text = hint_text or raw_text
        note_page = self._looks_like_note_page(raw_text)
        if not raw_text and not fallback_text:
            return {
                "page_number": ocr_page["page_number"],
                "source": ocr_page["source"],
                "raw_text": raw_text,
                "cleaned_text": "",
                "edits": [],
                "flags": flags,
                "protected_hits": protected_hits,
                "status": "empty",
                "notes": [],
                "allow_empty_output": True,
            }

        prompt = self._build_prompt(raw_text=raw_text, hint_text=hint_text, flags=flags, protected_hits=protected_hits)
        notes: list[str] = []
        llm_edits: list[dict[str, Any]] = []
        structured_plan: dict[str, Any] | None = None
        used_structured_plan = False
        model_drop_page = False
        try:
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                    maxOutputTokens=self.config.max_output_tokens,
                ),
            )
            structured_plan = parse_json_object(self._extract_response_text(response))
            cleaned_text, llm_edits, plan_notes, model_drop_page = apply_edit_plan(
                fallback_text,
                structured_plan,
                allow_drop_page=self._allow_model_drop(flags, note_page),
            )
            used_structured_plan = True
            notes.extend(plan_notes)
            cleaned_text = self._normalize_text(cleaned_text)
        except Exception as exc:
            cleaned_text = fallback_text
            notes.append(f"gemini_request_failed_used_fallback:{type(exc).__name__}")

        if not cleaned_text:
            if model_drop_page:
                notes.append("model_marked_non_body_page")
            elif note_page and self.config.notes_policy == "delete":
                notes.append("note_page_removed")
            else:
                notes.append("empty_model_output_used_fallback")
                cleaned_text = fallback_text

        raw_len = self._compact_len(raw_text)
        cleaned_len = self._compact_len(cleaned_text)
        if (
            raw_len >= 200
            and cleaned_len < int(raw_len * 0.72)
            and not (note_page and self.config.notes_policy == "delete")
            and not model_drop_page
        ):
            notes.append("model_output_too_short_used_fallback")
            cleaned_text = fallback_text

        missing_markers = self._missing_heading_markers(raw_text, cleaned_text)
        if missing_markers and not (note_page and self.config.notes_policy == "delete") and not model_drop_page:
            notes.append(f"missing_heading_markers_used_fallback:{','.join(missing_markers[:8])}")
            cleaned_text = fallback_text

        return {
            "page_number": ocr_page["page_number"],
            "source": ocr_page["source"],
            "raw_text": raw_text,
            "cleaned_text": cleaned_text.strip(),
            "edits": [],
            "llm_edits": llm_edits,
            "llm_edit_plan": structured_plan,
            "flags": flags,
            "protected_hits": protected_hits,
            "status": "gemini_structured" if used_structured_plan and not any("used_fallback" in note for note in notes) else "fallback",
            "notes": notes,
            "allow_empty_output": bool((model_drop_page or (note_page and self.config.notes_policy == "delete")) and not cleaned_text),
            "drop_page": bool(model_drop_page and not cleaned_text),
            "drop_reason": "llm_non_body_page" if model_drop_page and not cleaned_text else "",
        }

    def _build_prompt(
        self,
        raw_text: str,
        hint_text: str,
        flags: list[dict[str, Any]],
        protected_hits: list[dict[str, Any]],
    ) -> str:
        flags_preview = "\n".join(f"- {flag.get('rule_id')}: {flag.get('evidence', '')}" for flag in flags[:12]) or "(none)"
        protected_preview = (
            "\n".join(f"- {hit.get('rule_id')}: {hit.get('evidence', '')}" for hit in protected_hits[:12]) or "(none)"
        )
        working_text = hint_text or raw_text
        operation_list = ", ".join(ALLOWED_OPERATIONS)
        inline_pattern_list = ", ".join(ALLOWED_INLINE_PATTERN_NAMES)
        raw_section = f"RAW OCR PAGE:\n{raw_text}\n\n" if raw_text and raw_text != working_text else ""
        hint_section = f"RULE HINT PAGE:\n{hint_text}\n\n" if hint_text and hint_text != working_text else ""
        return (
            "You are the cleaning agent for Russian PDF OCR text.\n"
            "Clean and normalize the page conservatively using a structured edit plan.\n"
            "Preserve all original content and ordering.\n"
            "Do not summarize, rewrite, translate, or shorten.\n"
            "Return JSON only.\n\n"
            "The deterministic pipeline has already removed obvious page drops, simple inline note markers, and some trailing reference blocks.\n"
            "The OCR stage may already have separated body text from footnote/reference blocks; trust the provided body-text hint.\n"
            "Focus on the remaining body-text cleanup and heading/paragraph structure only.\n\n"
            "Target text:\n"
            "- apply edits against CURRENT WORKING PAGE, not by rewriting from scratch\n"
            "- CURRENT WORKING PAGE is the page below with line numbers\n\n"
            "Must clean:\n"
            "- merge artificial line-wrap hyphenation\n"
            "- merge fake line breaks inside normal paragraphs\n"
            "- normalize spaces and punctuation spacing\n"
            "- remove obvious OCR garbage lines and repeated extraction artifacts if they are clearly not content\n"
            "- keep chapter headings and subtitles readable\n\n"
            "Conditional clean:\n"
            "- citations, table-of-contents fragments, figure labels, numeric formatting\n"
            "- keep them unless they are clearly extraction noise\n\n"
            "Notes policy:\n"
            f"- {self._notes_policy_text()}\n\n"
            "Hard constraints:\n"
            "- if the page text is obviously garbled, mojibake, watermark-like, or unrecoverable, do not invent words or silently rewrite it into fluent prose\n"
            "- if the page is publisher metadata, sales/contact information, or back-matter logistics rather than body text, you may return an empty string\n"
            "- if the page is a glossary, abbreviation list, contents page, or references-only back-matter page rather than body text, you may return an empty string\n"
            "- when uncertain, prefer preserving the hint text over hallucinating a repaired version\n\n"
            "Do not delete:\n"
            "- chapter numbers, Roman numerals, headings, subtitles\n"
            "- lexical hyphens, abbreviations, initials, legal numbering, dates, values, URLs, emails\n"
            "- lines that carry content even if they are short or uppercase\n\n"
            "Allowed operations:\n"
            f"- {operation_list}\n"
            f"- remove_inline_pattern supports only: {inline_pattern_list}\n\n"
            "JSON schema:\n"
            "{\n"
            '  "drop_page": false,\n'
            '  "operations": [\n'
            '    {"op": "merge_with_next", "line": 12, "reason": "merge broken paragraph"},\n'
            '    {"op": "split_before_text", "line": 5, "text": "VII.", "reason": "restore heading"},\n'
            '    {"op": "delete_line_range", "start_line": 28, "end_line": 30, "reason": "remove obvious OCR garbage"},\n'
            '    {"op": "remove_inline_pattern", "pattern": "inline_numeric_note_markers", "reason": "strip note markers"},\n'
            '    {"op": "strip_trailing_reference_block", "reason": "drop trailing notes"},\n'
            '    {"op": "normalize_spacing", "reason": "cleanup spacing"}\n'
            "  ]\n"
            "}\n"
            "Use the smallest edit set that fixes the page. If no change is needed, return {\"drop_page\": false, \"operations\": []}.\n\n"
            f"Risk signals from rule engine:\n{flags_preview}\n\n"
            f"Protected spans detected:\n{protected_preview}\n\n"
            f"{raw_section}"
            f"{hint_section}"
            f"CURRENT WORKING PAGE WITH LINE NUMBERS:\n{render_numbered_text(working_text)}\n"
        )

    def _build_client(self):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Gemini cleaning selected but GOOGLE_API_KEY or GEMINI_API_KEY is not set.")
        if genai is None or genai_types is None:
            raise RuntimeError("google-genai package is not installed.")
        return genai.Client(api_key=api_key)

    def _extract_response_text(self, response: Any) -> str:
        text = getattr(response, "text", "") or ""
        text = text.strip()
        if text.startswith("```") and text.endswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return text.strip()

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _compact_len(self, text: str) -> int:
        return len(WHITESPACE_RE.sub("", text))

    def _missing_heading_markers(self, raw_text: str, cleaned_text: str) -> list[str]:
        raw_markers = sorted(set(marker.upper() for marker in HEADING_MARKER_RE.findall(raw_text)))
        cleaned_upper = cleaned_text.upper()
        return [marker for marker in raw_markers if marker not in cleaned_upper]

    def _looks_like_note_page(self, text: str) -> bool:
        if not text:
            return False
        note_lines = 0
        for line in text.splitlines():
            stripped = line.strip()
            if re.match(r"^\[\d{1,3}\]", stripped):
                note_lines += 1
        markers = len(re.findall(r"\[\d{1,3}\]", text))
        return note_lines >= 5 or markers >= 8

    def _notes_policy_text(self) -> str:
        if self.config.notes_policy == "delete":
            return (
                "Delete footnotes, endnotes, note markers, editor notes, and note-list pages. "
                "If a page is mostly notes/endnotes, return an empty string."
            )
        return "Keep footnotes and endnotes."

    def _allow_model_drop(self, flags: list[dict[str, Any]], note_page: bool) -> bool:
        if note_page and self.config.notes_policy == "delete":
            return True
        flag_ids = {flag.get("rule_id") for flag in flags}
        return bool(flag_ids & {"publisher_meta_page", "glossary_page", "toc_index_material", "reference_only_page"})

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from .structured_edits import (
    ALLOWED_INLINE_PATTERN_NAMES,
    ALLOWED_OPERATIONS,
    apply_edit_plan,
    parse_json_object,
    render_numbered_text,
)
from .vendor_clients import deepseek_chat_completion, deepseek_chat_completion_async, extract_openai_message_text


WHITESPACE_RE = re.compile(r"\s+")
HEADING_MARKER_RE = re.compile(r"(?m)^(?:[IVXLCDM]+\.)\s+", re.IGNORECASE)


@dataclass
class DeepSeekCleaningConfig:
    model: str = "deepseek-chat"
    max_output_tokens: int = 4096
    notes_policy: str = "delete"
    request_timeout: int = 180
    request_retries: int = 3
    retry_delay: float = 1.5


class DeepSeekCleaningAgent:
    def __init__(self, config: DeepSeekCleaningConfig | None = None) -> None:
        self.config = config or DeepSeekCleaningConfig()
        self.api_key = self._get_api_key()

    def clean_page(
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
            response = deepseek_chat_completion(
                api_key=self.api_key,
                payload={
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": "You are a conservative Russian OCR text cleaning agent."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": self.config.max_output_tokens,
                },
            )
            structured_plan = parse_json_object(extract_openai_message_text(response))
            cleaned_text, llm_edits, plan_notes, model_drop_page = apply_edit_plan(
                fallback_text,
                structured_plan,
                allow_drop_page=self._allow_model_drop(flags, note_page),
            )
            used_structured_plan = True
            notes.extend(plan_notes)
            cleaned_text = self._normalize_text(cleaned_text)
        except Exception as exc:  # pragma: no cover - networked path
            notes.append(f"deepseek_request_failed_used_fallback:{type(exc).__name__}")
            cleaned_text = fallback_text

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
            "status": "deepseek_structured" if used_structured_plan and not any("used_fallback" in note for note in notes) else "fallback",
            "notes": notes,
            "allow_empty_output": bool((model_drop_page or (note_page and self.config.notes_policy == "delete")) and not cleaned_text),
            "drop_page": bool(model_drop_page and not cleaned_text),
            "drop_reason": "llm_non_body_page" if model_drop_page and not cleaned_text else "",
        }

    async def clean_page_async(
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
            response, request_meta = await deepseek_chat_completion_async(
                api_key=self.api_key,
                payload={
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": "You are a conservative Russian OCR text cleaning agent."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": self.config.max_output_tokens,
                },
                timeout=self.config.request_timeout,
                retries=self.config.request_retries,
                retry_delay=self.config.retry_delay,
            )
            if request_meta.retries_used:
                notes.append(f"deepseek_retries={request_meta.retries_used}")
            structured_plan = parse_json_object(extract_openai_message_text(response))
            cleaned_text, llm_edits, plan_notes, model_drop_page = apply_edit_plan(
                fallback_text,
                structured_plan,
                allow_drop_page=self._allow_model_drop(flags, note_page),
            )
            used_structured_plan = True
            notes.extend(plan_notes)
            cleaned_text = self._normalize_text(cleaned_text)
        except Exception as exc:  # pragma: no cover - networked path
            notes.append(f"deepseek_request_failed_used_fallback:{type(exc).__name__}")
            cleaned_text = fallback_text

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
            "status": "deepseek_structured" if used_structured_plan and not any("used_fallback" in note for note in notes) else "fallback",
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
            "Clean this Russian PDF OCR page conservatively with a structured edit plan.\n"
            "Return JSON only. Do not summarize, translate, paraphrase, or shorten content.\n\n"
            "The deterministic pipeline has already handled obvious page drops and simple note stripping.\n"
            "The OCR stage may already have separated body text from footnote/reference blocks; trust the provided body-text hint.\n"
            "Your job is to fix only the remaining paragraph and heading structure issues.\n\n"
            "Target text:\n"
            "- edit CURRENT WORKING PAGE with line numbers\n"
            "- do not rewrite the whole page from scratch\n\n"
            "Must clean:\n"
            "- merge artificial line-wrap hyphenation\n"
            "- merge fake line breaks inside normal paragraphs\n"
            "- normalize spaces and punctuation spacing\n"
            "- remove obvious OCR garbage lines and repeated extraction artifacts if clearly not content\n"
            "- keep chapter headings and subtitles readable\n\n"
            "Notes policy:\n"
            f"- {self._notes_policy_text()}\n\n"
            "Hard constraints:\n"
            "- if the page text is obviously garbled, mojibake, watermark-like, or unrecoverable, do not invent words or silently rewrite it into fluent prose\n"
            "- if the page is publisher metadata, sales/contact information, or back-matter logistics rather than body text, you may return an empty string\n"
            "- if the page is a glossary, abbreviation list, contents page, or references-only back-matter page rather than body text, you may return an empty string\n"
            "- when uncertain, prefer preserving the hint text over hallucinating a repaired version\n\n"
            "Never delete real headings, Roman numerals, section numbers, dates, values, URLs that are part of body content, abbreviations, initials, or lexical hyphens.\n\n"
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
            f"Risk signals:\n{flags_preview}\n\n"
            f"Protected spans:\n{protected_preview}\n\n"
            f"{raw_section}"
            f"{hint_section}"
            f"CURRENT WORKING PAGE WITH LINE NUMBERS:\n{render_numbered_text(working_text)}\n"
        )

    def _get_api_key(self) -> str:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DeepSeek cleaning selected but DEEPSEEK_API_KEY is not set.")
        return api_key

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

from __future__ import annotations

import os
import re
from dataclasses import dataclass
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
    execute_edit_plan,
    parse_json_object,
    render_numbered_text,
)
from .state_models import PageProcessingState, PageState
from .state_machine import transition


WHITESPACE_RE = re.compile(r"\s+")
HEADING_MARKER_RE = re.compile(r"(?m)^(?:[IVXLCDM]+\.)\s+", re.IGNORECASE)
BRACKET_NOTE_RE = re.compile(r"\[\d{1,3}\]")
URL_RE = re.compile(r"https?://\S+")
INLINE_FOOTNOTE_RE = re.compile(r"(?<=[\w»”\")\]])\d{1,3}(?=\s|[.,;:!?])")
INLINE_BRACKET_FOOTNOTE_RE = re.compile(r"(?<=[^\W\d_»”\")\]])\[\d{1,3}\]")
REFERENCE_CUE_RE = re.compile(
    r"(?i)\b(?:см\.|цит\. по:|ibid\.|op\. cit\.|ргали\.|ф\.\s*\d+|оп\.\s*\d+|ед\.\s*хр\.|л\.\s*\d+|спб\.|м\.:|л\.;\s*м\.:|//)\b"
)
INLINE_LIST_ITEM_FUSION_RE = re.compile(r"([;:])\s+-\s+(?=[A-Za-zА-Яа-яЁё])")


@dataclass
class GeminiRepairConfig:
    model: str = "gemini-2.5-pro"
    max_output_tokens: int = 4096
    notes_policy: str = "delete"
    risky_only: bool = True


class GeminiRepairAgent:
    def __init__(self, config: GeminiRepairConfig | None = None) -> None:
        self.config = config or GeminiRepairConfig()
        self.client = self._build_client()

    def run(
        self,
        page_state: PageState,
        *,
        ocr_page: dict[str, Any] | None = None,
        cleaned_page: dict[str, Any] | None = None,
        heuristic_page: dict[str, Any] | None = None,
        gemini_review_page: dict[str, Any] | None = None,
    ) -> PageState:
        ocr_payload = ocr_page or page_state.stage_payloads.get("ocr")
        clean_payload = cleaned_page or page_state.stage_payloads.get("cleaned") or page_state.stage_payloads.get("primary_cleaned")
        review_payload = heuristic_page or page_state.stage_payloads.get("review")
        gemini_payload = gemini_review_page or page_state.stage_payloads.get("gemini_review")
        if ocr_payload is None or clean_payload is None or review_payload is None:
            raise ValueError("GeminiRepairAgent.run requires OCR, cleaned, and review payloads.")

        repaired_page = self.repair_page(
            ocr_payload,
            clean_payload,
            review_payload,
            gemini_payload,
            review_tags=page_state.review_tags,
            risk_level=page_state.risk_level,
        )
        page_state.repaired_text = str(repaired_page.get("cleaned_text") or "")
        page_state.repair_plan = repaired_page.get("llm_repair_plan")
        page_state.stage_payloads["repaired"] = repaired_page
        page_state.record_provenance(
            agent="GeminiRepairAgent",
            input_fields=["stage_payloads.cleaned", "review_tags", "risk_level", "edit_plan"],
            output_fields=["repaired_text", "repair_plan", "stage_payloads.repaired"],
            note=str(repaired_page.get("repair_status") or "unknown"),
        )
        transition(
            page_state,
            PageProcessingState.REPAIRED,
            agent="GeminiRepairAgent",
            note=str(repaired_page.get("repair_status") or ""),
        )
        return page_state

    def repair_page(
        self,
        ocr_page: dict[str, Any],
        cleaned_page: dict[str, Any],
        heuristic_page: dict[str, Any],
        gemini_review_page: dict[str, Any] | None = None,
        *,
        review_tags: list[str] | None = None,
        risk_level: str | None = None,
    ) -> dict[str, Any]:
        raw_text = (ocr_page.get("body_text") or ocr_page.get("selected_text") or "").strip()
        cleaned_text = (cleaned_page.get("cleaned_text") or "").strip()
        resolved_review_tags = self._resolve_review_tags(review_tags, heuristic_page)
        resolved_risk_level = self._resolve_risk_level(risk_level, heuristic_page, gemini_review_page)
        if cleaned_page.get("allow_empty_output") and not cleaned_text:
            repaired_page = dict(cleaned_page)
            repaired_page["repair_status"] = "cleaned_empty"
            repaired_page["repair_notes"] = ["empty_cleaned_page_preserved"]
            repaired_page["repair_issue_tags"] = list(resolved_review_tags)
            return repaired_page

        if self.config.risky_only and not self._should_attempt_repair(resolved_review_tags, resolved_risk_level, gemini_review_page):
            repaired_page = dict(cleaned_page)
            repaired_page["repair_status"] = "skipped"
            repaired_page["repair_notes"] = ["no_repair_needed"]
            repaired_page["repair_issue_tags"] = list(resolved_review_tags)
            return repaired_page

        fallback_text = cleaned_text or raw_text
        note_heavy = self._looks_like_note_page(raw_text) or self._looks_like_note_page(cleaned_text)
        prompt = self._build_prompt(
            raw_text,
            cleaned_text,
            cleaned_page,
            heuristic_page,
            gemini_review_page,
            resolved_review_tags,
        )
        repair_notes: list[str] = []
        structured_plan: dict[str, Any] | None = None
        used_structured_plan = False
        model_drop_page = False
        llm_edits: list[dict[str, Any]] = []
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
            execution = execute_edit_plan(
                fallback_text,
                structured_plan,
                allow_drop_page=self._allow_model_drop(resolved_review_tags, note_heavy),
            )
            used_structured_plan = True
            llm_edits = execution.applied_edits
            repair_notes.extend(execution.notes)
            model_drop_page = execution.drop_page
            repaired_text = self._normalize_text(execution.text)
        except Exception as exc:
            repaired_text = fallback_text
            repair_notes.append(f"gemini_request_failed_used_fallback:{type(exc).__name__}")

        if not repaired_text:
            if model_drop_page:
                repair_notes.append("model_marked_non_body_page")
            elif note_heavy and self.config.notes_policy == "delete":
                repair_notes.append("note_page_removed")
            else:
                repair_notes.append("empty_model_output_used_fallback")
                repaired_text = fallback_text

        repaired_text, cleanup_notes = self._apply_aggressive_note_cleanup(repaired_text, resolved_review_tags)
        repair_notes.extend(cleanup_notes)
        repaired_text, list_notes = self._restore_inline_list_breaks(repaired_text)
        repair_notes.extend(list_notes)

        base_len = self._compact_len(fallback_text)
        repaired_len = self._compact_len(repaired_text)
        min_ratio = self._minimum_allowed_ratio(resolved_review_tags, note_heavy)
        if base_len >= 200 and repaired_len < int(base_len * min_ratio) and not model_drop_page:
            repair_notes.append("model_output_too_short_used_fallback")
            repaired_text = fallback_text
            repaired_text, cleanup_notes = self._apply_aggressive_note_cleanup(repaired_text, resolved_review_tags)
            repair_notes.extend(cleanup_notes)
            repaired_text, list_notes = self._restore_inline_list_breaks(repaired_text)
            repair_notes.extend(list_notes)

        missing_markers = self._missing_heading_markers(raw_text, repaired_text)
        if missing_markers and not self._allows_aggressive_deletion(resolved_review_tags, note_heavy) and not model_drop_page:
            repair_notes.append(f"missing_heading_markers_used_fallback:{','.join(missing_markers[:8])}")
            repaired_text = fallback_text

        repaired_page = dict(cleaned_page)
        repaired_page["cleaned_text"] = repaired_text.strip()
        repaired_page["repaired_text"] = repaired_text.strip()
        repaired_page["llm_repair_edits"] = llm_edits
        repaired_page["llm_repair_plan"] = structured_plan
        repaired_page["repair_status"] = (
            "repaired_structured" if used_structured_plan and not any("used_fallback" in note for note in repair_notes) else "fallback"
        )
        repaired_page["repair_notes"] = repair_notes
        repaired_page["repair_issue_tags"] = list(resolved_review_tags)
        repaired_page["allow_empty_output"] = bool(
            (note_heavy or model_drop_page) and self.config.notes_policy == "delete" and not repaired_page["cleaned_text"]
        )
        repaired_page["drop_page"] = bool(model_drop_page and not repaired_page["cleaned_text"])
        repaired_page["drop_reason"] = "llm_non_body_page" if model_drop_page and not repaired_page["cleaned_text"] else cleaned_page.get("drop_reason", "")
        return repaired_page

    def _build_prompt(
        self,
        raw_text: str,
        cleaned_text: str,
        cleaned_page: dict[str, Any],
        heuristic_page: dict[str, Any],
        gemini_review_page: dict[str, Any] | None,
        review_tags: list[str],
    ) -> str:
        flags_preview = ", ".join(flag["rule_id"] for flag in cleaned_page.get("flags", [])[:12]) or "(none)"
        gemini_summary = ""
        if gemini_review_page is not None:
            gemini_summary = (
                f"Gemini review verdict: {gemini_review_page.get('llm_verdict', 'unknown')}\n"
                f"Gemini concerns: {', '.join(gemini_review_page.get('concerns', [])[:8]) or '(none)'}\n"
            )
        operation_list = ", ".join(ALLOWED_OPERATIONS)
        inline_pattern_list = ", ".join(ALLOWED_INLINE_PATTERN_NAMES)
        return (
            "You are the repair agent for Russian PDF OCR cleanup.\n"
            "Repair only the issues listed below. Preserve wording, ordering, and factual content.\n"
            "Do not summarize, rewrite, translate, paraphrase, or add any text.\n"
            "Return JSON only.\n\n"
            "A deterministic pipeline already removed obvious note markers and some reference tails.\n"
            "The OCR stage may already have separated body text from note/reference blocks; do not reintroduce them into body text.\n"
            "Only fix the remaining residual issues and structural mistakes.\n\n"
            "Target text:\n"
            "- edit CURRENT CLEANED PAGE with line numbers\n"
            "- do not rewrite the whole page from scratch\n\n"
            "Primary goals:\n"
            "- remove remaining footnote markers, endnotes, note-list blocks, and citation-only trailing lines\n"
            "- keep real headings, chapter numbers, Roman numerals, section numbers, dates, values, and abbreviations\n"
            "- restore headings or subtitles to their own line when they were merged into paragraphs\n"
            "- merge fake line breaks inside normal paragraphs\n"
            "- keep bullet lists and dialogue lines distinct\n"
            "- keep URLs only when they are substantive body content; remove them when they are only note/citation residue\n\n"
            "Priority override for notes:\n"
            "- if the page ends with numbered references, bibliography lines, note lines, or citation URLs, delete the entire trailing note/reference block from the first note line onward\n"
            "- when in doubt between preserving a trailing numbered reference block and deleting it, prefer deleting the trailing reference block\n"
            "- remove inline footnote numbers such as word43 or [12] when they are note markers, even if the surrounding sentence remains\n"
            "- do not keep page-bottom references just because they look grammatical\n"
            "- if the page is only glossary/abbreviation items or publisher back-matter, return an empty string\n\n"
            f"Notes policy: {'Delete notes/endnotes completely.' if self.config.notes_policy == 'delete' else 'Keep notes.'}\n"
            f"Review tags: {', '.join(review_tags) or '(none)'}\n"
            f"Rule flags: {flags_preview}\n"
            f"Heuristic review verdict: {heuristic_page.get('page_verdict', 'unknown')}\n"
            f"Heuristic issue tags: {', '.join(heuristic_page.get('issue_tags', [])) or '(none)'}\n"
            f"Allowed operations: {operation_list}\n"
            f"remove_inline_pattern supports only: {inline_pattern_list}\n"
            "JSON schema:\n"
            "{\n"
            '  "drop_page": false,\n'
            '  "operations": [\n'
            '    {"op": "strip_trailing_reference_block", "reason": "remove page-bottom notes"},\n'
            '    {"op": "remove_inline_pattern", "pattern": "bracket_note_markers", "reason": "strip note markers"},\n'
            '    {"op": "merge_with_next", "line": 14, "reason": "merge paragraph wrap"},\n'
            '    {"op": "split_before_text", "line": 7, "text": "VII.", "reason": "restore heading"},\n'
            '    {"op": "normalize_spacing", "reason": "cleanup spacing"}\n'
            "  ]\n"
            "}\n"
            "Use the smallest edit set that fixes the listed problems. If no repair is needed, return {\"drop_page\": false, \"operations\": []}.\n"
            "Prefer deleting trailing note/reference residue over preserving it when the block is clearly non-body.\n\n"
            f"{gemini_summary}\n"
            f"RAW OCR PAGE:\n{raw_text}\n\n"
            f"CURRENT CLEANED PAGE WITH LINE NUMBERS:\n{render_numbered_text(cleaned_text or raw_text)}\n"
        )

    def _resolve_review_tags(self, review_tags: list[str] | None, heuristic_page: dict[str, Any]) -> list[str]:
        tags = list(review_tags if review_tags is not None else heuristic_page.get("issue_tags", []))
        seen: set[str] = set()
        return [tag for tag in tags if not (tag in seen or seen.add(tag))]

    def _resolve_risk_level(
        self,
        risk_level: str | None,
        heuristic_page: dict[str, Any],
        gemini_review_page: dict[str, Any] | None,
    ) -> str:
        if risk_level:
            return str(risk_level)
        if gemini_review_page is not None and gemini_review_page.get("llm_verdict") == "reject":
            return "high"
        if heuristic_page.get("page_verdict") == "reject":
            return "high"
        if (
            heuristic_page.get("page_verdict") == "escalate"
            or (gemini_review_page is not None and gemini_review_page.get("llm_verdict") == "escalate")
        ):
            return "medium"
        return "low"

    def _should_attempt_repair(
        self,
        review_tags: list[str],
        risk_level: str,
        gemini_review_page: dict[str, Any] | None,
    ) -> bool:
        if review_tags:
            return True
        if risk_level in {"medium", "high"}:
            return True
        if gemini_review_page is not None and gemini_review_page.get("llm_verdict") in {"escalate", "reject"}:
            return True
        return False

    def _suspicious_inline_footnotes(self, text: str) -> bool:
        matches = INLINE_FOOTNOTE_RE.findall(text)
        return len(matches) >= 2

    def _heading_structure_risky(self, raw_text: str, cleaned_text: str) -> bool:
        raw_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        raw_headings = [
            line for line in raw_lines
            if re.match(r"^(?:Глава\s+\d+|[IVXLCDM]+\.\s+.+|[A-ZА-ЯЁ][A-ZА-ЯЁ0-9\s«»\"()\-]{10,})$", line)
        ]
        if not raw_headings:
            return False
        cleaned_lines = {line.strip() for line in cleaned_text.splitlines() if line.strip()}
        return any(heading not in cleaned_lines for heading in raw_headings[:5])

    def _allows_aggressive_deletion(self, issue_tags: list[str], note_heavy: bool) -> bool:
        if note_heavy and self.config.notes_policy == "delete":
            return True
        return any(
            tag in {
                "footnote_marker_left",
                "endnote_block_left",
                "citation_url_left",
                "reference_suffix_left",
                "publisher_meta_left",
                "garbled_page",
                "empty_page_after_cleaning",
            }
            for tag in issue_tags
        )

    def _minimum_allowed_ratio(self, issue_tags: list[str], note_heavy: bool) -> float:
        if note_heavy and self.config.notes_policy == "delete":
            return 0.05
        if any(tag in {"publisher_meta_left", "garbled_page", "empty_page_after_cleaning"} for tag in issue_tags):
            return 0.01
        if any(tag in {"endnote_block_left", "citation_url_left", "reference_suffix_left"} for tag in issue_tags):
            return 0.12
        if any(tag in {"footnote_marker_left", "inline_note_marker_left"} for tag in issue_tags):
            return 0.30
        return 0.72

    def _apply_aggressive_note_cleanup(self, text: str, issue_tags: list[str]) -> tuple[str, list[str]]:
        if self.config.notes_policy != "delete" or not text:
            return text, []
        notes: list[str] = []
        updated = text
        if any(tag in {"publisher_meta_left", "garbled_page", "empty_page_after_cleaning"} for tag in issue_tags):
            return "", ["post_dropped_non_body_page"]
        if any(tag in {"endnote_block_left", "citation_url_left", "reference_suffix_left"} for tag in issue_tags):
            trimmed = self._strip_trailing_note_block(updated)
            if trimmed != updated:
                updated = trimmed
                notes.append("post_trimmed_trailing_note_block")
        if any(tag in {"footnote_marker_left", "inline_note_marker_left"} for tag in issue_tags):
            stripped = self._strip_inline_note_markers(updated)
            if stripped != updated:
                updated = stripped
                notes.append("post_removed_inline_note_markers")
        return updated.strip(), notes

    def _strip_inline_note_markers(self, text: str) -> str:
        text = INLINE_BRACKET_FOOTNOTE_RE.sub("", text)
        text = re.sub(r"(?<=[^\W\d_»”\")\]])\[\d{1,3}\]", "", text)
        text = re.sub(r"(?<=[^\W\d_»”\")\]])\d{1,3}(?=(?:\s|[.,;:!?]))", "", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r" +\n", "\n", text)
        return text

    def _strip_trailing_note_block(self, text: str) -> str:
        lines = text.splitlines()
        if len(lines) < 2:
            return text
        cut_index: int | None = None
        for idx in range(max(0, len(lines) - 20), len(lines)):
            suffix = [line for line in lines[idx:] if line.strip()]
            if len(suffix) < 2:
                continue
            note_like = sum(1 for line in suffix if self._is_note_like_line(line))
            if note_like >= 2 and self._is_note_like_line(suffix[0]):
                cut_index = idx
                break
        if cut_index is None:
            return text
        kept = "\n".join(lines[:cut_index]).rstrip()
        return kept

    def _is_note_like_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if re.match(r"^\d{1,3}\s+", stripped):
            return True
        if re.match(r"^\[\d{1,3}\]\s+", stripped):
            return True
        if "http://" in stripped or "https://" in stripped or " // " in stripped or stripped.endswith("//"):
            return True
        if re.match(r"^(См\.|Ibid\.|Цит\. соч\.)", stripped):
            return True
        if REFERENCE_CUE_RE.search(stripped):
            return True
        return False

    def _looks_like_note_page(self, text: str) -> bool:
        if not text:
            return False
        markers = len(BRACKET_NOTE_RE.findall(text))
        note_lines = self._note_block_line_count(text)
        return markers >= 5 or note_lines >= 5

    def _note_block_line_count(self, text: str) -> int:
        count = 0
        for line in text.splitlines():
            stripped = line.strip()
            if not re.match(r"^\d{1,3}\s+", stripped):
                continue
            if "http://" in stripped or "https://" in stripped or "//" in stripped:
                count += 1
        return count

    def _reference_suffix_line_count(self, text: str) -> int:
        count = 0
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if REFERENCE_CUE_RE.search(stripped):
                count += 1
        return count

    def _build_client(self):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Gemini repair selected but GOOGLE_API_KEY or GEMINI_API_KEY is not set.")
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

    def _restore_inline_list_breaks(self, text: str) -> tuple[str, list[str]]:
        updated = INLINE_LIST_ITEM_FUSION_RE.sub(r"\1\n- ", text)
        if updated == text:
            return text, []
        return updated, ["restored_inline_list_breaks"]

    def _compact_len(self, text: str) -> int:
        return len(WHITESPACE_RE.sub("", text))

    def _missing_heading_markers(self, raw_text: str, repaired_text: str) -> list[str]:
        raw_markers = sorted(set(marker.upper() for marker in HEADING_MARKER_RE.findall(raw_text)))
        repaired_upper = repaired_text.upper()
        return [marker for marker in raw_markers if marker not in repaired_upper]

    def _allow_model_drop(self, issue_tags: list[str], note_heavy: bool) -> bool:
        if note_heavy and self.config.notes_policy == "delete":
            return True
        return any(
            tag in {"publisher_meta_left", "garbled_page", "empty_page_after_cleaning", "reference_suffix_left"}
            for tag in issue_tags
        )

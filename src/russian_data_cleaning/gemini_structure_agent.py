from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any
from typing import Callable

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None

from .state_models import PageProcessingState, PageState
from .state_machine import transition


WHITESPACE_RE = re.compile(r"\s+")
HEADING_MARKER_RE = re.compile(r"(?m)^(?:[IVXLCDM]+\.)\s+", re.IGNORECASE)


@dataclass
class GeminiStructureConfig:
    model: str = "gemini-2.5-pro"
    max_output_tokens: int = 4096


class GeminiStructureAgent:
    def __init__(self, config: GeminiStructureConfig | None = None) -> None:
        self.config = config or GeminiStructureConfig()
        self.client = self._build_client()

    def run(
        self,
        page_state: PageState,
        *,
        ocr_page: dict[str, Any] | None = None,
        repaired_page: dict[str, Any] | None = None,
    ) -> PageState:
        ocr_payload = ocr_page or page_state.stage_payloads.get("ocr")
        if ocr_payload is None:
            raise ValueError("GeminiStructureAgent.run requires OCR payload.")

        repaired_payload = repaired_page or page_state.stage_payloads.get("repaired") or page_state.stage_payloads.get("repaired_primary")
        if repaired_payload is None:
            fallback_text = page_state.repaired_text or page_state.primary_clean_text or page_state.rule_cleaned_text
            repaired_payload = {
                "page_number": page_state.page_num,
                "repaired_text": fallback_text,
                "cleaned_text": fallback_text,
                "allow_empty_output": False,
                "drop_page": False,
            }

        restored_page = self.restore_page(ocr_payload, repaired_payload)
        final_text_source = str(restored_page.get("final_text_source") or "repaired_passthrough")
        structure_plan = {
            "backend": "gemini",
            "model": self.config.model,
            "status": str(restored_page.get("status") or "unknown"),
            "notes": list(restored_page.get("notes", [])),
            "skipped_reason": restored_page.get("skipped_reason"),
            "final_text_source": final_text_source,
        }

        page_state.structure_plan = structure_plan
        page_state.final_text = str(restored_page.get("restored_text") or "")
        note = self._structure_note(structure_plan)
        page_state.record_provenance(
            agent="GeminiStructureAgent",
            input_fields=["raw_text", "repaired_text"],
            output_fields=["structure_plan", "final_text"],
            note=note,
        )
        transition(
            page_state,
            PageProcessingState.STRUCTURE_RESTORED,
            agent="GeminiStructureAgent",
            note=note,
        )
        return page_state

    def restore_document(
        self,
        ocr_document: dict[str, Any],
        cleaned_document: dict[str, Any],
        progress_hook: Callable[[int, int, int], None] | None = None,
    ) -> dict[str, Any]:
        ocr_map = {page["page_number"]: page for page in ocr_document["pages"]}
        restored_pages = []
        total = len(cleaned_document["pages"])
        for index, cleaned_page in enumerate(cleaned_document["pages"], start=1):
            if progress_hook is not None:
                progress_hook(index, total, cleaned_page["page_number"])
            ocr_page = ocr_map[cleaned_page["page_number"]]
            restored_pages.append(self.restore_page(ocr_page, cleaned_page))
        return {
            "relative_path": cleaned_document["relative_path"],
            "model": self.config.model,
            "pages": restored_pages,
        }

    def restore_page(
        self,
        ocr_page: dict[str, Any],
        repaired_page: dict[str, Any],
    ) -> dict[str, Any]:
        return self._restore_page(ocr_page, repaired_page)

    def _restore_page(
        self,
        ocr_page: dict[str, Any],
        repaired_page: dict[str, Any],
    ) -> dict[str, Any]:
        raw_text = (ocr_page.get("body_text") or ocr_page.get("selected_text") or "").strip()
        repaired_text = (repaired_page.get("repaired_text") or repaired_page.get("cleaned_text") or "").strip()
        if repaired_page.get("allow_empty_output") and not repaired_text:
            return {
                "page_number": repaired_page["page_number"],
                "restored_text": "",
                "status": "repaired_empty",
                "notes": ["empty_repaired_page_preserved"],
                "skipped_reason": "empty_repaired_page",
                "final_text_source": "repaired_passthrough",
            }
        fallback_text = repaired_text or raw_text

        if not raw_text and not repaired_text:
            return {
                "page_number": repaired_page["page_number"],
                "restored_text": "",
                "status": "empty",
                "notes": [],
                "skipped_reason": "empty_input_page",
                "final_text_source": "repaired_passthrough",
            }

        prompt = self._build_prompt(raw_text=raw_text, repaired_text=repaired_text)
        notes: list[str] = []
        used_generated_output = False
        try:
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                    maxOutputTokens=self.config.max_output_tokens,
                ),
            )
            restored_text = self._normalize_text(self._extract_response_text(response))
            used_generated_output = True
        except Exception as exc:
            restored_text = fallback_text
            notes.append(f"gemini_request_failed_used_fallback:{type(exc).__name__}")

        if not restored_text:
            notes.append("empty_model_output_used_fallback")
            restored_text = fallback_text

        raw_len = self._compact_len(raw_text)
        restored_len = self._compact_len(restored_text)
        if raw_len >= 200 and restored_len < int(raw_len * 0.72):
            notes.append("model_output_too_short_used_fallback")
            restored_text = fallback_text

        missing_markers = self._missing_heading_markers(raw_text, restored_text)
        if missing_markers:
            notes.append(f"missing_heading_markers_used_fallback:{','.join(missing_markers[:8])}")
            restored_text = fallback_text

        return {
            "page_number": repaired_page["page_number"],
            "restored_text": restored_text.strip(),
            "status": "gemini" if not notes else "fallback",
            "notes": notes,
            "skipped_reason": None,
            "final_text_source": "structure_restore_generated" if used_generated_output and not notes else "repaired_passthrough",
        }

    def _build_prompt(self, raw_text: str, repaired_text: str) -> str:
        return (
            "You are restoring the page structure of a Russian book OCR export.\n"
            "Your task is block-level structural cleanup only.\n"
            "Preserve the original wording and order exactly.\n"
            "Do not summarize, translate, paraphrase, polish, censor, shorten, or add commentary.\n"
            "Do not change wording inside sentences.\n"
            "Do not replace names, dates, numeric values, legal numbering, identifiers, or abbreviations.\n"
            "Do not add or remove sentence content.\n"
            "Do not remove chapter numbers, Roman numerals, section headings, or title lines.\n"
            "Use the repaired text as the source of truth for sentence content.\n"
            "Use the raw OCR page only as a hint for paragraph breaks, headings, and block boundaries.\n"
            "Do not pull separated footnote or reference blocks back into the main body.\n\n"
            "Rules:\n"
            "1. Merge artificial line wraps inside normal paragraphs.\n"
            "2. Keep chapter and section headings on their own line.\n"
            "3. If a heading is split across multiple uppercase lines, join it into one heading line.\n"
            "4. Keep short heading summaries or subtitles on a separate line after the heading when present.\n"
            "5. Keep dialogue lines beginning with '-' as separate paragraphs.\n"
            "6. Keep list items and block boundaries separate when they are clearly separate blocks.\n"
            "7. Remove artificial blank lines caused by PDF extraction, but do not drop content.\n"
            "8. Return plain text only.\n\n"
            f"RAW OCR PAGE:\n{raw_text}\n\n"
            f"REPAIRED PAGE:\n{repaired_text}\n"
        )

    def _build_client(self):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Gemini structure cleanup selected but GOOGLE_API_KEY or GEMINI_API_KEY is not set.")
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

    def _missing_heading_markers(self, raw_text: str, restored_text: str) -> list[str]:
        raw_markers = sorted(set(marker.upper() for marker in HEADING_MARKER_RE.findall(raw_text)))
        restored_upper = restored_text.upper()
        return [marker for marker in raw_markers if marker not in restored_upper]

    def _structure_note(self, structure_plan: dict[str, Any]) -> str:
        source = str(structure_plan.get("final_text_source") or "repaired_passthrough")
        backend = str(structure_plan.get("backend") or "gemini")
        model = str(structure_plan.get("model") or self.config.model)
        status = str(structure_plan.get("status") or "unknown")
        skipped_reason = structure_plan.get("skipped_reason")
        note = f"source={source};backend={backend};model={model};status={status}"
        if skipped_reason:
            note += f";skipped_reason={skipped_reason}"
        return note

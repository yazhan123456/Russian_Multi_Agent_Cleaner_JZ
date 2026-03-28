from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from .state_models import PageState


LOWER_START_RE = re.compile(r"^[a-zа-яё]")
BODYISH_LINE_RE = re.compile(r"^(?:[-*•]\s+|\(?\d+[.)]?\s+|[a-zа-яё])")


@dataclass
class CommanderConfig:
    ocr_base_render_scale: float = 2.2
    ocr_high_render_scale: float = 2.6
    skip_model_cleaning_extract_chars: int = 240


@dataclass
class OCRPlan:
    source: str
    render_scale: float
    difficulty: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CleaningPlan:
    action: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PageCommander:
    RISKY_RULE_FLAGS = {
        "footnote_markers",
        "citations_and_bibliographic_refs",
        "line_end_hyphenation_ambiguous",
        "reference_suffix_material",
        "figure_table_formula_labels",
        "garbled_page",
    }

    def __init__(self, config: CommanderConfig | None = None) -> None:
        self.config = config or CommanderConfig()

    def run(
        self,
        page_state: PageState,
        *,
        stage: str,
        ocr_payload: dict[str, Any] | None = None,
        rule_page: dict[str, Any] | None = None,
    ) -> PageState:
        if stage == "ocr_route":
            if ocr_payload is None:
                raise ValueError("ocr_payload is required for PageCommander.run(stage='ocr_route').")
            route_decision, risk_level, note = self._route_from_ocr_payload(ocr_payload)
            page_state.route_decision = route_decision
            page_state.ocr_mode = str(ocr_payload.get("source") or "")
            page_state.risk_level = risk_level
            page_state.record_provenance(
                agent="PageCommander",
                input_fields=["stage_payloads.ocr"],
                output_fields=["route_decision", "ocr_mode", "risk_level"],
                note=note,
            )
            return page_state
        if stage == "primary_cleaning":
            ocr_page = page_state.stage_payloads.get("ocr")
            if not ocr_page:
                raise ValueError("OCR payload is required before PageCommander.run(stage='primary_cleaning').")
            decision = self.plan_primary_cleaning(ocr_page=ocr_page, rule_page=rule_page)
            page_state.stage_payloads["primary_cleaning_plan"] = decision.to_dict()
            page_state.record_provenance(
                agent="PageCommander",
                input_fields=["stage_payloads.ocr", "stage_payloads.rule_cleaned"],
                output_fields=["stage_payloads.primary_cleaning_plan"],
                note=f"{decision.action}:{decision.reason}",
            )
            return page_state
        raise ValueError(f"Unsupported commander stage: {stage}")

    def plan_ocr_page(
        self,
        *,
        route_hint: str,
        extracted_text: str,
        extracted_char_count: int,
        pre_ocr_skip_reason: str | None,
        looks_mojibake: bool,
        looks_low_quality_extract: bool,
        extracted_blocks: list[dict[str, Any]],
        backend: str,
    ) -> OCRPlan:
        if pre_ocr_skip_reason and extracted_text:
            return OCRPlan(
                source="ocr_skipped_nonbody",
                render_scale=self.config.ocr_base_render_scale,
                difficulty="skip_nonbody",
                reason=pre_ocr_skip_reason,
            )

        if backend == "extract_only":
            return OCRPlan(
                source="extract",
                render_scale=self.config.ocr_base_render_scale,
                difficulty="extract_only",
                reason="backend_extract_only",
            )
        if backend == "tesseract":
            return OCRPlan(
                source="ocr",
                render_scale=self.config.ocr_high_render_scale,
                difficulty="forced_ocr",
                reason="backend_tesseract",
            )
        if backend in {"gemini", "google_documentai"}:
            return OCRPlan(
                source="ocr",
                render_scale=self.config.ocr_high_render_scale,
                difficulty="forced_ocr",
                reason=f"backend_{backend}",
            )

        block_count = len(extracted_blocks)
        nonempty_blocks = [block for block in extracted_blocks if (block.get("text") or "").strip()]
        short_blocks = sum(1 for block in nonempty_blocks if len((block.get("text") or "").strip()) <= 70)
        short_block_ratio = (short_blocks / len(nonempty_blocks)) if nonempty_blocks else 0.0
        top_bodyish_block = False
        if nonempty_blocks:
            top_block = min(nonempty_blocks, key=lambda block: float(block.get("bbox", [0, 0, 0, 0])[1]))
            y0 = float(top_block.get("bbox", [0, 0, 0, 0])[1])
            page_height = max(1.0, float(max((block.get("bbox", [0, 0, 0, 0])[3] for block in nonempty_blocks), default=1.0)))
            top_ratio = y0 / page_height
            top_bodyish_block = top_ratio <= 0.12 and self._looks_like_body_continuation((top_block.get("text") or "").strip())

        if route_hint == "pdf_ocr_then_clean":
            return OCRPlan(
                source="ocr",
                render_scale=self.config.ocr_high_render_scale,
                difficulty="hard_scan",
                reason="route_pdf_ocr_then_clean",
            )

        if looks_mojibake or looks_low_quality_extract:
            return OCRPlan(
                source="ocr",
                render_scale=self.config.ocr_high_render_scale,
                difficulty="dirty_extract",
                reason="extract_quality_risky",
            )

        if extracted_char_count < 80:
            return OCRPlan(
                source="ocr",
                render_scale=self.config.ocr_high_render_scale,
                difficulty="sparse_text",
                reason="extract_sparse",
            )

        if top_bodyish_block and extracted_char_count < 260:
            return OCRPlan(
                source="ocr",
                render_scale=self.config.ocr_high_render_scale,
                difficulty="top_body_risk",
                reason="top_block_looks_like_body_continuation",
            )

        if block_count >= 14 and short_block_ratio >= 0.70:
            return OCRPlan(
                source="ocr",
                render_scale=self.config.ocr_high_render_scale,
                difficulty="layout_complex",
                reason="many_short_blocks",
            )

        if route_hint == "pdf_mixed_extract_plus_ocr":
            if extracted_char_count >= 260 and short_block_ratio < 0.65:
                return OCRPlan(
                    source="extract",
                    render_scale=self.config.ocr_base_render_scale,
                    difficulty="easy_extract",
                    reason="mixed_route_extract_good",
                )
            return OCRPlan(
                source="ocr",
                render_scale=self.config.ocr_base_render_scale,
                difficulty="mixed_risky",
                reason="mixed_route_needs_ocr",
            )

        if extracted_char_count >= 260 and short_block_ratio < 0.60:
            return OCRPlan(
                source="extract",
                render_scale=self.config.ocr_base_render_scale,
                difficulty="easy_extract",
                reason="extract_quality_good",
            )

        if route_hint == "pdf_extract_then_clean":
            return OCRPlan(
                source="extract",
                render_scale=self.config.ocr_base_render_scale,
                difficulty="extract_default",
                reason="route_pdf_extract_then_clean",
            )

        return OCRPlan(
            source="ocr" if extracted_char_count < 160 else "extract",
            render_scale=self.config.ocr_base_render_scale,
            difficulty="fallback",
            reason="default_route",
        )

    def plan_primary_cleaning(
        self,
        *,
        ocr_page: dict[str, Any],
        rule_page: dict[str, Any] | None,
    ) -> CleaningPlan:
        if rule_page is None:
            return CleaningPlan(action="run_primary_cleaning", reason="missing_rule_page")
        if rule_page.get("drop_page"):
            return CleaningPlan(action="skip_primary_cleaning", reason="dropped_by_rules")
        cleaned_text = (rule_page.get("cleaned_text") or "").strip()
        if not cleaned_text:
            return CleaningPlan(action="skip_primary_cleaning", reason="empty_after_rules")

        source = str(ocr_page.get("source") or "")
        page_type = str(ocr_page.get("page_type") or "")
        layout_status = str(ocr_page.get("layout_status") or "")
        extracted_char_count = int(ocr_page.get("extracted_char_count") or 0)
        flag_ids = {flag.get("rule_id") for flag in rule_page.get("flags", [])}

        if source in {"ocr", "extract_fallback"}:
            return CleaningPlan(action="run_primary_cleaning", reason=f"ocr_source:{source}")
        if page_type in {"body_with_notes", "notes_only"}:
            return CleaningPlan(action="run_primary_cleaning", reason=f"page_type:{page_type}")
        if layout_status == "text_fallback" and source != "extract":
            return CleaningPlan(action="run_primary_cleaning", reason="layout_text_fallback")
        if flag_ids & self.RISKY_RULE_FLAGS:
            risky = sorted(flag for flag in flag_ids if flag in self.RISKY_RULE_FLAGS)
            return CleaningPlan(action="run_primary_cleaning", reason=f"risky_flags:{','.join(risky[:4])}")
        if extracted_char_count < self.config.skip_model_cleaning_extract_chars:
            return CleaningPlan(action="run_primary_cleaning", reason="short_extract_page")
        return CleaningPlan(action="skip_primary_cleaning", reason="rules_look_sufficient")

    def _looks_like_body_continuation(self, text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False
        first = lines[0]
        compact = re.sub(r"\s+", " ", first)
        if len(compact) < 12:
            return False
        if LOWER_START_RE.match(compact):
            return True
        if BODYISH_LINE_RE.match(compact) and any(char in compact for char in ",;:"):
            return True
        alpha_chars = re.sub(r"[^A-Za-zА-Яа-яЁё]", "", compact)
        if alpha_chars:
            upper_chars = re.sub(r"[^A-ZА-ЯЁ]", "", compact)
            if len(upper_chars) <= int(len(alpha_chars) * 0.45) and compact[-1:] not in {".", "!", "?"}:
                return True
        return False

    def _route_from_ocr_payload(self, ocr_payload: dict[str, Any]) -> tuple[str, str, str]:
        source = str(ocr_payload.get("source") or "")
        page_type = str(ocr_payload.get("page_type") or "")
        layout_status = str(ocr_payload.get("layout_status") or "")
        char_count = int(ocr_payload.get("extracted_char_count") or 0)

        if page_type in {"toc_or_index", "reference_only", "glossary_page", "publisher_meta"}:
            return ("skip_nonbody_page", "low", f"page_type={page_type}")
        if source in {"ocr", "extract_fallback"}:
            return ("hard_ocr_page", "high", f"ocr_source={source}")
        if source in {"ocr_skipped_nonbody", "epub_skip_nonbody"}:
            return ("skip_nonbody_page", "low", f"source={source}")
        if source in {"extract", "epub_extract"}:
            if page_type == "body_with_notes" or layout_status == "text_fallback" or char_count < 260:
                return ("risky_extract_page", "medium", "extract_with_layout_risk")
            return ("easy_page", "low", "extract_looks_safe")
        return ("risky_extract_page", "medium", "default_route")

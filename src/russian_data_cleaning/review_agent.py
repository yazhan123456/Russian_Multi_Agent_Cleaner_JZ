from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from .state_models import PageProcessingState, PageState
from .state_machine import transition


SAFE_RULES = {
    "soft_hyphen",
    "zero_width_chars",
    "spacing_cleanup",
    "punctuation_variant_unification",
    "inline_note_marker_strip",
}
CONDITIONAL_RULES = {
    "repeated_headers_footers",
    "line_end_hyphenation",
    "fake_paragraph_breaks",
    "isolated_ocr_noise",
    "trailing_reference_block_strip",
}
RISKY_FLAGS = {
    "footnote_markers",
    "citations_and_bibliographic_refs",
    "toc_index_material",
    "figure_table_formula_labels",
    "number_and_date_formatting",
    "line_end_hyphenation_ambiguous",
    "reference_suffix_material",
    "reference_only_page",
    "glossary_page",
    "publisher_meta_page",
    "garbled_page",
}
PROTECTED_RULES = {
    "lexical_hyphen_words",
    "abbreviations_with_periods",
    "initials_and_names",
    "legal_structural_numbering",
    "latin_spans_and_identifiers",
}

BRACKET_NOTE_RE = re.compile(r"\[\d{1,3}\]")
URL_RE = re.compile(r"https?://\S+")
EXCESSIVE_BLANKS_RE = re.compile(r"\n\s*\n\s*\n")
RAW_HEADING_RE = re.compile(r"(?m)^(?:Глава\s+\d+|[IVXLCDM]+\.\s+.+|[A-ZА-ЯЁ][A-ZА-ЯЁ0-9\s«»\"()\-]{10,})$")
REFERENCE_CUE_RE = re.compile(
    r"(?i)\b(?:см\.:|цит\. по:|ргали\.|ф\.\s*\d+|оп\.\s*\d+|ед\.\s*хр\.|л\.\s*\d+|//|спб\.|м\.:)\b"
)
ISBN_RE = re.compile(r"(?i)\b(?:isbn|i5bn|15вм|исвn)\b")
EMAIL_RE = re.compile(r"\b\S+@\S+\b")
PHONE_RE = re.compile(r"\(\d{3,4}\)\s*\d{2,3}[-–]\d{2}[-–]\d{2}")
PUBLISHER_CUE_RE = re.compile(
    r"(?i)\b(?:издательство|книг[аи]\s*[—-]\s*почтой|книжный салон|можно купить|российская национальная библиотека|bookshop|copyright)\b"
)
GLOSSARY_HEADING_RE = re.compile(r"(?mi)^\s*(?:список терминов(?: и сокращений)?|глоссарий|список сокращений)\s*$")
CYRILLIC_LETTER_RE = re.compile(r"[А-Яа-яЁё]")
LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
EXTENDED_LATIN_RE = re.compile(r"[\u0180-\u024F\u0250-\u02AF]")
ARCHIVE_CUE_RE = re.compile(r"(?i)\b(?:archive\.org|internet archive|kahle|foundation)\b")


@dataclass
class ReviewRecord:
    kind: str
    target_rule_id: str
    verdict: str
    detail: str
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReviewAgent:
    RISK_BY_VERDICT = {
        "approve": "low",
        "escalate": "medium",
        "reject": "high",
    }
    CONFIDENCE_BY_VERDICT = {
        "approve": 0.9,
        "escalate": 0.65,
        "reject": 0.35,
    }

    def review_document(
        self,
        ocr_document: dict[str, Any],
        cleaned_document: dict[str, Any],
    ) -> dict[str, Any]:
        page_map = {page["page_number"]: page for page in ocr_document["pages"]}
        reviewed_pages = []

        for page in cleaned_document["pages"]:
            raw_page = page_map[page["page_number"]]
            reviewed_pages.append(self._review_page(raw_page, page))

        return {
            "relative_path": cleaned_document["relative_path"],
            "route_hint": cleaned_document["route_hint"],
            "pages": reviewed_pages,
        }

    def review_page(
        self,
        raw_page: dict[str, Any],
        cleaned_page: dict[str, Any],
    ) -> dict[str, Any]:
        return self._review_page(raw_page, cleaned_page)

    def run(
        self,
        page_state: PageState,
        *,
        raw_page: dict[str, Any] | None = None,
        cleaned_page: dict[str, Any] | None = None,
    ) -> PageState:
        ocr_payload = raw_page or page_state.stage_payloads.get("ocr")
        if ocr_payload is None:
            raise ValueError("ReviewAgent.run requires OCR payload.")

        cleaned_payload = cleaned_page or page_state.stage_payloads.get("cleaned") or page_state.stage_payloads.get("primary_cleaned")
        if cleaned_payload is None:
            raise ValueError("ReviewAgent.run requires cleaned page payload.")

        review_payload = self.review_page(ocr_payload, cleaned_payload)
        page_state.review_tags = list(review_payload.get("issue_tags", []))
        page_state.risk_level = self._risk_level_for_verdict(review_payload.get("page_verdict"))
        page_state.confidence = self._confidence_for_verdict(review_payload.get("page_verdict"))
        page_state.stage_payloads["review"] = review_payload
        page_state.record_provenance(
            agent="ReviewAgent",
            input_fields=["stage_payloads.ocr", "stage_payloads.cleaned"],
            output_fields=["review_tags", "risk_level", "confidence", "stage_payloads.review"],
            note=str(review_payload.get("page_verdict") or "unknown"),
        )
        transition(
            page_state,
            PageProcessingState.REVIEWED,
            agent="ReviewAgent",
            note=str(review_payload.get("page_verdict") or ""),
        )
        return page_state

    def _review_page(
        self,
        raw_page: dict[str, Any],
        cleaned_page: dict[str, Any],
    ) -> dict[str, Any]:
        review_records: list[ReviewRecord] = []
        raw_text = (raw_page.get("body_text") or raw_page.get("selected_text") or "")
        cleaned_text = cleaned_page["cleaned_text"] or ""
        if cleaned_page.get("allow_empty_output") and cleaned_page.get("drop_page"):
            return {
                "page_number": cleaned_page["page_number"],
                "source": cleaned_page["source"],
                "page_verdict": "approve",
                "raw_length": len(raw_text),
                "cleaned_length": 0,
                "deletion_ratio": 1.0 if raw_text else 0.0,
                "issue_tags": [],
                "review_records": [
                    ReviewRecord(
                        kind="drop",
                        target_rule_id=cleaned_page.get("drop_reason", "drop_page"),
                        verdict="approve",
                        detail="Page intentionally dropped by deterministic routing rule.",
                        evidence=(raw_text[:160] if raw_text else ""),
                    ).to_dict()
                ],
            }
        raw_len = max(1, len(raw_text))
        cleaned_len = len(cleaned_text)
        deletion_ratio = max(0.0, (raw_len - cleaned_len) / raw_len)
        protected_hits = cleaned_page["protected_hits"]
        issue_tags = self._detect_issue_tags(raw_text, cleaned_text, cleaned_page)

        for edit in cleaned_page["edits"]:
            rule_id = edit["rule_id"]
            if rule_id in SAFE_RULES:
                review_records.append(
                    ReviewRecord(
                        kind="edit",
                        target_rule_id=rule_id,
                        verdict="approve",
                        detail="Low-risk normalization.",
                        evidence=edit["before"][:160],
                    )
                )
                continue

            if protected_hits and rule_id in CONDITIONAL_RULES:
                review_records.append(
                    ReviewRecord(
                        kind="edit",
                        target_rule_id=rule_id,
                        verdict="escalate",
                        detail="Conditional edit touches a page with protected spans.",
                        evidence=edit["before"][:160],
                    )
                )
                continue

            if rule_id == "repeated_headers_footers":
                review_records.append(
                    ReviewRecord(
                        kind="edit",
                        target_rule_id=rule_id,
                        verdict="approve",
                        detail="Repeated edge line removal is acceptable.",
                        evidence=edit["before"][:160],
                    )
                )
                continue

            if rule_id == "line_end_hyphenation":
                if "\\n" in edit["before"] and len(edit["after"]) >= 6:
                    verdict = "approve"
                    detail = "Likely OCR line-wrap hyphen fixed."
                else:
                    verdict = "escalate"
                    detail = "Hyphen merge is ambiguous."
                review_records.append(
                    ReviewRecord(
                        kind="edit",
                        target_rule_id=rule_id,
                        verdict=verdict,
                        detail=detail,
                        evidence=edit["before"][:160],
                    )
                )
                continue

            if rule_id == "fake_paragraph_breaks":
                verdict = "approve" if deletion_ratio < 0.25 else "escalate"
                review_records.append(
                    ReviewRecord(
                        kind="edit",
                        target_rule_id=rule_id,
                        verdict=verdict,
                        detail="Paragraph line merge reviewed.",
                        evidence=edit["before"][:160],
                    )
                )
                continue

            if rule_id == "isolated_ocr_noise":
                verdict = "approve" if len(edit["before"].strip()) <= 12 else "escalate"
                review_records.append(
                    ReviewRecord(
                        kind="edit",
                        target_rule_id=rule_id,
                        verdict=verdict,
                        detail="Noise-line deletion reviewed.",
                        evidence=edit["before"][:160],
                    )
                )
                continue

            if rule_id == "trailing_reference_block_strip":
                verdict = "approve" if deletion_ratio < 0.55 else "escalate"
                review_records.append(
                    ReviewRecord(
                        kind="edit",
                        target_rule_id=rule_id,
                        verdict=verdict,
                        detail="Trailing reference block removal reviewed.",
                        evidence=edit["before"][:160],
                    )
                )
                continue

            review_records.append(
                ReviewRecord(
                    kind="edit",
                    target_rule_id=rule_id,
                    verdict="escalate",
                    detail="Unclassified edit requires review.",
                    evidence=edit["before"][:160],
                )
            )

        for flag in cleaned_page["flags"]:
            rule_id = flag["rule_id"]
            verdict = "escalate" if rule_id in RISKY_FLAGS else "approve"
            review_records.append(
                ReviewRecord(
                    kind="flag",
                    target_rule_id=rule_id,
                    verdict=verdict,
                    detail="Conditional content detected." if verdict == "escalate" else "Low-risk flag.",
                    evidence=flag["evidence"][:160],
                )
            )

        for protected in protected_hits:
            rule_id = protected["rule_id"]
            verdict = "approve" if rule_id in PROTECTED_RULES else "escalate"
            review_records.append(
                ReviewRecord(
                    kind="protected_hit",
                    target_rule_id=rule_id,
                    verdict=verdict,
                    detail="Protected span preserved." if verdict == "approve" else "Unexpected protected match.",
                    evidence=protected["evidence"][:160],
                )
            )

        verdicts = [record.verdict for record in review_records]
        if "reject" in verdicts:
            page_verdict = "reject"
        elif "escalate" in verdicts or deletion_ratio >= 0.35 or issue_tags:
            page_verdict = "escalate"
        else:
            page_verdict = "approve"

        return {
            "page_number": cleaned_page["page_number"],
            "source": cleaned_page["source"],
            "page_verdict": page_verdict,
            "raw_length": raw_len,
            "cleaned_length": cleaned_len,
            "deletion_ratio": round(deletion_ratio, 4),
            "issue_tags": issue_tags,
            "review_records": [record.to_dict() for record in review_records],
        }

    def _detect_issue_tags(
        self,
        raw_text: str,
        cleaned_text: str,
        cleaned_page: dict[str, Any],
    ) -> list[str]:
        issue_tags: list[str] = []
        flags = {flag["rule_id"] for flag in cleaned_page.get("flags", [])}
        note_block_count = self._note_block_line_count(cleaned_text)

        if BRACKET_NOTE_RE.search(cleaned_text):
            issue_tags.append("footnote_marker_left")
        if note_block_count >= 2:
            issue_tags.append("endnote_block_left")
        if self._reference_suffix_line_count(cleaned_text) >= 2:
            issue_tags.append("reference_suffix_left")
        if URL_RE.search(cleaned_text) and (
            note_block_count >= 1 or {
            "citations_and_bibliographic_refs",
            "footnote_markers",
            "toc_index_material",
        } & flags):
            issue_tags.append("citation_url_left")
        if EXCESSIVE_BLANKS_RE.search(cleaned_text):
            issue_tags.append("excessive_blank_lines")
        if self._heading_structure_risky(raw_text, cleaned_text):
            issue_tags.append("heading_structure_risky")
        if self._looks_like_garbled_text(cleaned_text or raw_text):
            issue_tags.append("garbled_page")
        if GLOSSARY_HEADING_RE.search(cleaned_text):
            issue_tags.append("glossary_left")
        if self._looks_like_publisher_meta_text(cleaned_text):
            issue_tags.append("publisher_meta_left")
        if not cleaned_text.strip() and raw_text.strip():
            issue_tags.append("empty_page_after_cleaning")

        seen: set[str] = set()
        return [tag for tag in issue_tags if not (tag in seen or seen.add(tag))]

    def _note_block_line_count(self, text: str) -> int:
        count = 0
        for line in text.splitlines():
            stripped = line.strip()
            if not re.match(r"^\d{1,3}\s+", stripped):
                continue
            if "http://" in stripped or "https://" in stripped or "//" in stripped:
                count += 1
        return count

    def _heading_structure_risky(self, raw_text: str, cleaned_text: str) -> bool:
        raw_headings = [line.strip() for line in raw_text.splitlines() if RAW_HEADING_RE.match(line.strip())]
        if not raw_headings:
            return False
        cleaned_lines = {line.strip() for line in cleaned_text.splitlines() if line.strip()}
        missing = 0
        for heading in raw_headings[:5]:
            if heading not in cleaned_lines:
                missing += 1
        return missing >= 1

    def _reference_suffix_line_count(self, text: str) -> int:
        count = 0
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and REFERENCE_CUE_RE.search(stripped):
                count += 1
        return count

    def _looks_like_publisher_meta_text(self, text: str) -> bool:
        if not text.strip():
            return False
        nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
        cues = 0
        if ISBN_RE.search(text):
            cues += 1
        if EMAIL_RE.search(text):
            cues += 1
        if PHONE_RE.search(text):
            cues += 1
        if PUBLISHER_CUE_RE.search(text):
            cues += 1
        if cues >= 2:
            return True
        if ISBN_RE.search(text) and len(nonempty_lines) <= 10:
            return True
        return False

    def _looks_like_garbled_text(self, text: str) -> bool:
        stripped = text.strip()
        if len(stripped) < 40:
            return False
        weird = len(EXTENDED_LATIN_RE.findall(stripped))
        cyr = len(CYRILLIC_LETTER_RE.findall(stripped))
        lat = len(LATIN_LETTER_RE.findall(stripped))
        mixed_tokens = 0
        for token in re.findall(r"\S+", stripped):
            token = token.strip(".,;:!?()[]{}«»\"'")
            if len(token) < 4:
                continue
            if EXTENDED_LATIN_RE.search(token):
                mixed_tokens += 1
                continue
            if LATIN_LETTER_RE.search(token) and CYRILLIC_LETTER_RE.search(token):
                mixed_tokens += 1
        if ARCHIVE_CUE_RE.search(stripped) and mixed_tokens >= 2:
            return True
        if "://" in stripped and mixed_tokens >= 3:
            return True
        return weird >= 2 and cyr <= 12 and lat >= 12 and mixed_tokens >= 3

    def _risk_level_for_verdict(self, verdict: str | None) -> str:
        return self.RISK_BY_VERDICT.get(str(verdict or ""), "medium")

    def _confidence_for_verdict(self, verdict: str | None) -> float:
        return self.CONFIDENCE_BY_VERDICT.get(str(verdict or ""), 0.5)

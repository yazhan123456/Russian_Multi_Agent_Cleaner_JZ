from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .state_models import PageState
from .state_machine import PageProcessingState, transition


SPACE_RUN_RE = re.compile(r"[ \t\u00A0]{2,}")
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")
SOFT_HYPHEN_RE = re.compile("\u00AD")
NOISE_LINE_RE = re.compile(r"^(?:[~`*_]{3,}|[|/\\]{3,})$")
CYRILLIC_HYPHEN_RE = re.compile(r"([А-Яа-яЁё]{3,})[-\u2010\u2011]\s*\n\s*([А-Яа-яЁё]{3,})")
FOOTNOTE_RE = re.compile(r"(?:\[\d{1,3}\]|\(\d{1,3}\)|(?<=[А-Яа-яЁё])[¹²³⁴⁵⁶⁷⁸⁹])")
CITATION_RE = re.compile(r"\[[0-9,\- ]+\]|\([А-ЯA-ZЁ][^)]{0,80},\s*(?:19|20)\d{2}\)")
DOT_LEADER_RE = re.compile(r"\.{4,}")
TOC_HEADING_RE = re.compile(r"(?mi)^(?:содержание|оглавление|contents|table of contents|index)\s*$")
TOC_ENTRY_RE = re.compile(
    r"(?m)^(?:[A-ZА-ЯЁ0-9][^\n]{3,120}?)(?:\.{2,}|\s{2,}|\s)\d{1,4}\s*$"
)
CYRILLIC_LETTER_RE = re.compile(r"[А-Яа-яЁё]")
LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
EXTENDED_LATIN_RE = re.compile(r"[\u0180-\u024F\u0250-\u02AF]")
ISBN_RE = re.compile(r"(?i)\b(?:isbn|i5bn|15вм|исвn)\b")
EMAIL_RE = re.compile(r"\b\S+@\S+\b")
PHONE_RE = re.compile(r"\(\d{3,4}\)\s*\d{2,3}[-–]\d{2}[-–]\d{2}")
PUBLISHER_CUE_RE = re.compile(
    r"(?i)\b(?:издательство|книг[аи]\s*[—-]\s*почтой|книжный салон|можно купить|российская национальная библиотека|bookshop|copyright)\b"
)
PUBLISHER_BACKMATTER_RE = re.compile(
    r"(?i)\b(?:научное издание|подписано в печать|макет и техническое редактирование|художественное оформление|корректор|гарнитура|тираж\b|заказ №|формат \d{2}х\d{2})\b"
)
FRONTMATTER_CUE_RE = re.compile(
    r"(?i)\b(?:текст предоставлен правообладателем|все права защищены|переводчик\b|главный редактор\b|руководитель проекта\b|дизайн обложки|компьютерная верстка|произведение предназначено исключительно|copyright\b|text provided by)\b"
)
BODY_START_HEADING_RE = re.compile(
    r"(?mi)^\s*(?:предисловие|введение|глава\s+\d+|chapter\s+\d+|часть\s+\d+|part\s+\d+)\s*$"
)
BODY_START_INLINE_RE = re.compile(
    r"(?i)\b(?:предисловие|введение|глава\s+\d+|chapter\s+\d+|часть\s+\d+|part\s+\d+)\b"
)
ARCHIVE_CUE_RE = re.compile(r"(?i)\b(?:archive\.org|internet archive|kahle|foundation)\b")
FIGURE_RE = re.compile(r"^(?:Рис\.|Табл\.|Figure|Table)\s*\d+", re.IGNORECASE)
VALUE_RE = re.compile(r"(?:№\s*\d+|\d+\s*%|\d+[,.]\d+|\d+\s?(?:кг|г|м|см|мм|л|°C|°С))")
SENTENCE_END_RE = re.compile(r'[.!?…:;"»)\]]$')
LOWER_START_RE = re.compile(r"^[a-zа-яё]")
ROMAN_NUMERAL_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
HEADING_LINE_RE = re.compile(r"^(?:[IVXLCDM]+\.\s+)?[A-ZА-ЯЁ\s«»\"()\-]{12,}$")
OCR_BULLET_LINE_RE = re.compile(r"^\s*[xхXХ]\s+(?=[A-Za-zА-Яа-яЁё])")
INLINE_NOTE_SUFFIX_RE = re.compile(r"(?<=[A-Za-zА-Яа-яЁё])\d{1,3}(?=(?:\s|[.,;:!?…])|$)")
INLINE_BRACKET_NOTE_RE = re.compile(r"(?<=[A-Za-zА-Яа-яЁё])\[\d{1,3}\]")
INLINE_SUPERSCRIPT_NOTE_RE = re.compile(r"(?<=[A-Za-zА-Яа-яЁё])[¹²³⁴⁵⁶⁷⁸⁹]")
REFERENCE_LINE_CUE_RE = re.compile(
    r"(?i)(?:https?://|www\.|//|ibid\.|op\. cit\.|цит\. соч\.|цит\. по:|там же|см\.:|ргали\.|ф\.\s*\d+|оп\.\s*\d+|ед\.\s*хр\.|л\.\s*\d+)"
)
NUMBERED_REFERENCE_LINE_RE = re.compile(r"^\s*\d{1,3}[.)]?\s+")
REFERENCE_BACKMATTER_CUE_RE = re.compile(
    r"(?i)(?:пма\.\s*сообщение|указ\.\s*соч\.|смомпк|ргвиа|цга\b|автореферат|дисс\.|там же\.?|ibid\.?)"
)
ABBREVIATION_LIST_HEADING_RE = re.compile(
    r"(?mi)^\s*(?:список сокращений|список аббревиатур|условные сокращения)\s*$"
)
GLOSSARY_HEADING_RE = re.compile(
    r"(?mi)^\s*(?:список терминов(?: и сокращений)?|глоссарий|термины и сокращения)\s*$"
)
GLOSSARY_ENTRY_RE = re.compile(
    r"(?m)^\s*(?:[A-Z]{2,}[A-Z0-9&/+ -]{0,40}|\w[\w/+ -]{1,40}\([^)]+\))\s*[—-]\s+"
)
PROTECTED_PATTERNS = {
    "lexical_hyphen_words": re.compile(r"\b[А-Яа-яЁё]+-[А-Яа-яЁё]+\b"),
    "abbreviations_with_periods": re.compile(r"\b(?:т\.д\.|т\.п\.|г\.|стр\.|им\.)"),
    "initials_and_names": re.compile(r"\b[А-ЯЁ]\.\s?[А-ЯЁ](?:\.\s?)?[А-ЯЁа-яё-]+\b"),
    "legal_structural_numbering": re.compile(r"\bст\.\s*\d+\b|§\s*\d+|\b\d+(?:\.\d+)+\b"),
    "latin_spans_and_identifiers": re.compile(r"https?://\S+|\b\S+@\S+\b|\b[A-Z]{2,}[A-Z0-9-]*\b"),
}


@dataclass
class EditRecord:
    rule_id: str
    action: str
    before: str
    after: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FlagRecord:
    rule_id: str
    detail: str
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CleaningAgent:
    def __init__(self, rules_path: str | Path | None = None) -> None:
        if rules_path is None:
            rules_path = Path(__file__).resolve().parents[2] / "config" / "russian_ocr_cleaning_rules.json"
        self.rules_path = Path(rules_path)
        self.rules = json.loads(self.rules_path.read_text(encoding="utf-8"))

    def process_document(self, ocr_document: dict[str, Any]) -> dict[str, Any]:
        pages = ocr_document["pages"]
        repeated_headers, repeated_footers = self.detect_repeated_edges(pages)
        cleaned_pages = []

        for page in pages:
            cleaned_pages.append(
                self.clean_page(
                    page,
                    repeated_headers=repeated_headers,
                    repeated_footers=repeated_footers,
                )
            )

        return {
            "relative_path": ocr_document["relative_path"],
            "route_hint": ocr_document["route_hint"],
            "repeated_headers": sorted(repeated_headers),
            "repeated_footers": sorted(repeated_footers),
            "pages": cleaned_pages,
        }

    def detect_repeated_edges(
        self,
        pages: list[dict[str, Any]],
    ) -> tuple[set[str], set[str]]:
        return self._detect_repeated_edge_lines(pages)

    def clean_page(
        self,
        page: dict[str, Any],
        repeated_headers: set[str],
        repeated_footers: set[str],
    ) -> dict[str, Any]:
        return self._clean_page(
            page,
            repeated_headers=repeated_headers,
            repeated_footers=repeated_footers,
        )

    def run(
        self,
        page_state: PageState,
        *,
        repeated_headers: set[str],
        repeated_footers: set[str],
    ) -> PageState:
        ocr_page = page_state.stage_payloads.get("ocr")
        if not ocr_page:
            raise ValueError("CleaningAgent.run requires OCR payload in page_state.stage_payloads['ocr'].")
        cleaned_page = self.clean_page(
            ocr_page,
            repeated_headers=repeated_headers,
            repeated_footers=repeated_footers,
        )
        page_state.rule_cleaned_text = str(cleaned_page.get("cleaned_text") or "")
        page_state.stage_payloads["rule_cleaned"] = cleaned_page
        page_state.record_provenance(
            agent="CleaningAgent",
            input_fields=["stage_payloads.ocr"],
            output_fields=["rule_cleaned_text", "stage_payloads.rule_cleaned"],
            note=f"drop_page={bool(cleaned_page.get('drop_page'))}",
        )
        transition(page_state, PageProcessingState.RULE_CLEANED, agent="CleaningAgent")
        return page_state

    def _detect_repeated_edge_lines(
        self,
        pages: list[dict[str, Any]],
    ) -> tuple[set[str], set[str]]:
        header_counts: dict[str, int] = {}
        footer_counts: dict[str, int] = {}

        for page in pages:
            lines = self._split_lines(page.get("body_text") or page["selected_text"])
            if not lines:
                continue
            header = self._normalize_line_key(lines[0])
            footer = self._normalize_line_key(lines[-1])
            if 0 < len(header) <= 120:
                header_counts[header] = header_counts.get(header, 0) + 1
            if 0 < len(footer) <= 120:
                footer_counts[footer] = footer_counts.get(footer, 0) + 1

        threshold = max(2, len(pages) // 2)
        repeated_headers = {line for line, count in header_counts.items() if count >= threshold}
        repeated_footers = {line for line, count in footer_counts.items() if count >= threshold}
        return repeated_headers, repeated_footers

    def _clean_page(
        self,
        page: dict[str, Any],
        repeated_headers: set[str],
        repeated_footers: set[str],
    ) -> dict[str, Any]:
        raw_text = page.get("body_text") or page["selected_text"] or ""
        text = raw_text
        edits: list[EditRecord] = []
        flags: list[FlagRecord] = []

        text, header_footer_edits = self._remove_repeated_edges(text, repeated_headers, repeated_footers)
        edits.extend(header_footer_edits)

        text, changed = self._replace_regex(text, SOFT_HYPHEN_RE, "", "soft_hyphen", "normalize", "Removed soft hyphen.")
        edits.extend(changed)

        text, changed = self._replace_regex(text, ZERO_WIDTH_RE, "", "zero_width_chars", "normalize", "Removed zero-width character.")
        edits.extend(changed)

        text, bullet_edits = self._normalize_ocr_bullet_markers(text)
        edits.extend(bullet_edits)

        text, hyphen_edits, hyphen_flags = self._merge_line_end_hyphenation(text)
        edits.extend(hyphen_edits)
        flags.extend(hyphen_flags)

        text, merge_edits = self._merge_paragraph_lines(text)
        edits.extend(merge_edits)

        text, spacing_edits = self._normalize_spacing(text)
        edits.extend(spacing_edits)

        text, noise_edits = self._remove_noise_lines(text)
        edits.extend(noise_edits)

        text, block_edits = self._normalize_short_blocks(text)
        edits.extend(block_edits)

        text, note_marker_edits = self._strip_inline_note_markers(text)
        edits.extend(note_marker_edits)

        if self._should_trim_trailing_reference_block(page, text):
            text, trailing_reference_edits = self._trim_trailing_reference_block(text)
        else:
            trailing_reference_edits = []
        edits.extend(trailing_reference_edits)

        text, frontmatter_edits = self._trim_leading_front_matter_block(page["page_number"], text)
        edits.extend(frontmatter_edits)

        protected_hits = self._find_protected_hits(text)
        flags.extend(self._find_conditional_flags(text))

        if self._looks_like_toc_page(page["selected_text"], text):
            flags.append(
                FlagRecord(
                    rule_id="toc_index_material",
                    detail="Page classified as table of contents/index and dropped.",
                    evidence=(text or page["selected_text"])[:160],
                )
            )
            return {
                "page_number": page["page_number"],
                "source": page["source"],
                "raw_text": raw_text,
                "cleaned_text": "",
                "edits": [edit.to_dict() for edit in edits],
                "flags": [flag.to_dict() for flag in flags],
                "protected_hits": protected_hits,
                "allow_empty_output": True,
                "drop_page": True,
                "drop_reason": "toc_index_page",
            }

        if self._looks_like_front_matter_title_page(page["page_number"], page["selected_text"], text):
            flags.append(
                FlagRecord(
                    rule_id="front_matter_title_page",
                    detail="Page classified as title/front-matter page; kept for full-text completeness.",
                    evidence=(text or page["selected_text"])[:160],
                )
            )

        if self._looks_like_publisher_meta_page(page["selected_text"], text):
            flags.append(
                FlagRecord(
                    rule_id="publisher_meta_page",
                    detail="Page classified as publisher/contact/back-matter metadata; kept for full-text completeness.",
                    evidence=(text or page["selected_text"])[:160],
                )
            )

        if self._looks_like_garbled_page(page["selected_text"], text, page.get("source", "")):
            flags.append(
                FlagRecord(
                    rule_id="garbled_page",
                    detail="Page classified as garbled/unrecoverable extraction and dropped.",
                    evidence=(text or page["selected_text"])[:160],
                )
            )
            return {
                "page_number": page["page_number"],
                "source": page["source"],
                "raw_text": raw_text,
                "cleaned_text": "",
                "edits": [edit.to_dict() for edit in edits],
                "flags": [flag.to_dict() for flag in flags],
                "protected_hits": protected_hits,
                "allow_empty_output": True,
                "drop_page": True,
                "drop_reason": "garbled_page",
            }

        if self._looks_like_reference_only_page(page["selected_text"], text) and not self._should_keep_reference_heavy_body_page(page, text):
            flags.append(
                FlagRecord(
                    rule_id="reference_only_page",
                    detail="Page classified as notes/references only and dropped.",
                    evidence=(text or page["selected_text"])[:160],
                )
            )
            return {
                "page_number": page["page_number"],
                "source": page["source"],
                "raw_text": raw_text,
                "cleaned_text": "",
                "edits": [edit.to_dict() for edit in edits],
                "flags": [flag.to_dict() for flag in flags],
                "protected_hits": protected_hits,
                "allow_empty_output": True,
                "drop_page": True,
                "drop_reason": "reference_only_page",
            }

        if self._looks_like_glossary_page(page["selected_text"], text):
            flags.append(
                FlagRecord(
                    rule_id="glossary_page",
                    detail="Page classified as glossary/abbreviation list and dropped.",
                    evidence=(text or page["selected_text"])[:160],
                )
            )
            return {
                "page_number": page["page_number"],
                "source": page["source"],
                "raw_text": raw_text,
                "cleaned_text": "",
                "edits": [edit.to_dict() for edit in edits],
                "flags": [flag.to_dict() for flag in flags],
                "protected_hits": protected_hits,
                "allow_empty_output": True,
                "drop_page": True,
                "drop_reason": "glossary_page",
            }

        return {
            "page_number": page["page_number"],
            "source": page["source"],
            "raw_text": raw_text,
            "cleaned_text": text.strip(),
            "edits": [edit.to_dict() for edit in edits],
            "flags": [flag.to_dict() for flag in flags],
            "protected_hits": protected_hits,
        }

    def _remove_repeated_edges(
        self,
        text: str,
        repeated_headers: set[str],
        repeated_footers: set[str],
    ) -> tuple[str, list[EditRecord]]:
        lines = self._split_lines(text)
        edits: list[EditRecord] = []
        if not lines:
            return text, edits

        if self._normalize_line_key(lines[0]) in repeated_headers:
            edits.append(
                EditRecord(
                    rule_id="repeated_headers_footers",
                    action="delete",
                    before=lines[0],
                    after="",
                    detail="Removed repeated header line.",
                )
            )
            lines = lines[1:]
        if lines and self._normalize_line_key(lines[-1]) in repeated_footers:
            edits.append(
                EditRecord(
                    rule_id="repeated_headers_footers",
                    action="delete",
                    before=lines[-1],
                    after="",
                    detail="Removed repeated footer line.",
                )
            )
            lines = lines[:-1]
        return "\n".join(lines), edits

    def _merge_line_end_hyphenation(self, text: str) -> tuple[str, list[EditRecord], list[FlagRecord]]:
        edits: list[EditRecord] = []
        flags: list[FlagRecord] = []

        def replacer(match: re.Match[str]) -> str:
            left = match.group(1)
            right = match.group(2)
            if len(left) <= 2 or len(right) <= 3:
                flags.append(
                    FlagRecord(
                        rule_id="line_end_hyphenation_ambiguous",
                        detail="Ambiguous line-end hyphenation kept for review.",
                        evidence=match.group(0).replace("\n", "\\n"),
                    )
                )
                return match.group(0)
            merged = f"{left}{right}"
            edits.append(
                EditRecord(
                    rule_id="line_end_hyphenation",
                    action="normalize",
                    before=match.group(0).replace("\n", "\\n"),
                    after=merged,
                    detail="Merged likely artificial line-wrap hyphenation.",
                )
            )
            return merged

        text = CYRILLIC_HYPHEN_RE.sub(replacer, text)
        return text, edits, flags

    def _merge_paragraph_lines(self, text: str) -> tuple[str, list[EditRecord]]:
        lines = self._split_lines(text, keep_empty=True)
        if not lines:
            return text, []

        merged: list[str] = []
        edits: list[EditRecord] = []
        i = 0
        while i < len(lines):
            current = lines[i]
            if not current.strip():
                merged.append("")
                i += 1
                continue

            buffer = current.rstrip()
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if not nxt.strip():
                    break
                if not self._should_merge_lines(buffer, nxt):
                    break
                before = f"{buffer}\\n{nxt}"
                buffer = f"{buffer.rstrip()} {nxt.lstrip()}"
                edits.append(
                    EditRecord(
                        rule_id="fake_paragraph_breaks",
                        action="normalize",
                        before=before,
                        after=buffer,
                        detail="Merged likely false paragraph line break.",
                    )
                )
                j += 1
            merged.append(buffer)
            i = j

        return "\n".join(merged), edits

    def _normalize_spacing(self, text: str) -> tuple[str, list[EditRecord]]:
        edits: list[EditRecord] = []
        original = text
        text = text.replace("\u00A0", " ")
        text = SPACE_RUN_RE.sub(" ", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([(\[«„])\s+", r"\1", text)
        text = re.sub(r"\s+([)\]»“])", r"\1", text)
        if text != original:
            edits.append(
                EditRecord(
                    rule_id="spacing_cleanup",
                    action="normalize",
                    before=original[:160],
                    after=text[:160],
                    detail="Normalized repeated spaces and punctuation spacing.",
                )
            )
        return text, edits

    def _normalize_ocr_bullet_markers(self, text: str) -> tuple[str, list[EditRecord]]:
        edits: list[EditRecord] = []
        normalized_lines: list[str] = []
        for line in self._split_lines(text, keep_empty=True):
            if OCR_BULLET_LINE_RE.match(line):
                updated = OCR_BULLET_LINE_RE.sub("- ", line, count=1)
                edits.append(
                    EditRecord(
                        rule_id="ocr_bullet_marker_normalization",
                        action="normalize",
                        before=line,
                        after=updated,
                        detail="Normalized OCR bullet marker 'x/х' to a list dash.",
                    )
                )
                normalized_lines.append(updated)
                continue
            normalized_lines.append(line)
        return "\n".join(normalized_lines), edits

    def _remove_noise_lines(self, text: str) -> tuple[str, list[EditRecord]]:
        edits: list[EditRecord] = []
        kept_lines: list[str] = []
        for line in self._split_lines(text, keep_empty=True):
            if line and self._is_noise_line(line.strip()):
                edits.append(
                    EditRecord(
                        rule_id="isolated_ocr_noise",
                        action="delete",
                        before=line,
                        after="",
                        detail="Removed isolated OCR noise line.",
                    )
                )
                continue
            kept_lines.append(line)
        return "\n".join(kept_lines), edits

    def _normalize_short_blocks(self, text: str) -> tuple[str, list[EditRecord]]:
        edits: list[EditRecord] = []
        blocks = re.split(r"\n\s*\n", text)
        normalized_blocks: list[str] = []

        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue

            if self._should_flatten_block(lines):
                merged = self._flatten_block(lines)
                if merged != "\n".join(lines):
                    edits.append(
                        EditRecord(
                            rule_id="short_block_reflow",
                            action="normalize",
                            before="\\n".join(lines[:12]),
                            after=merged[:240],
                            detail="Flattened a short wrapped heading/summary block.",
                        )
                    )
                normalized_blocks.append(merged)
                continue

            repaired_lines: list[str] = []
            i = 0
            while i < len(lines):
                buffer = lines[i]
                while i + 1 < len(lines) and self._should_merge_short_line(buffer, lines[i + 1]):
                    before = f"{buffer}\\n{lines[i + 1]}"
                    buffer = self._join_lines([buffer, lines[i + 1]])
                    edits.append(
                        EditRecord(
                            rule_id="short_line_reflow",
                            action="normalize",
                            before=before,
                            after=buffer,
                            detail="Merged short wrapped line fragment.",
                        )
                    )
                    i += 1
                repaired_lines.append(buffer)
                i += 1

            normalized_blocks.append("\n".join(repaired_lines))

        return "\n\n".join(normalized_blocks), edits

    def _replace_regex(
        self,
        text: str,
        pattern: re.Pattern[str],
        replacement: str,
        rule_id: str,
        action: str,
        detail: str,
    ) -> tuple[str, list[EditRecord]]:
        edits: list[EditRecord] = []

        def replacer(match: re.Match[str]) -> str:
            edits.append(
                EditRecord(
                    rule_id=rule_id,
                    action=action,
                    before=match.group(0),
                    after=replacement,
                    detail=detail,
                )
            )
            return replacement

        return pattern.sub(replacer, text), edits

    def _strip_inline_note_markers(self, text: str) -> tuple[str, list[EditRecord]]:
        edits: list[EditRecord] = []

        def replacer(match: re.Match[str]) -> str:
            edits.append(
                EditRecord(
                    rule_id="inline_note_marker_strip",
                    action="delete",
                    before=match.group(0),
                    after="",
                    detail="Removed inline footnote marker attached to nearby text.",
                )
            )
            return ""

        text = INLINE_BRACKET_NOTE_RE.sub(replacer, text)
        text = INLINE_SUPERSCRIPT_NOTE_RE.sub(replacer, text)
        text = INLINE_NOTE_SUFFIX_RE.sub(replacer, text)
        return text, edits

    def _trim_trailing_reference_block(self, text: str) -> tuple[str, list[EditRecord]]:
        lines = text.splitlines()
        if len(lines) < 4:
            return text, []

        tail_start = max(0, len(lines) - 20)
        cut_index: int | None = None
        for idx in range(tail_start, len(lines) - 1):
            if not self._is_reference_line(lines[idx]):
                continue
            suffix_lines = lines[idx:]
            nonempty_suffix = [line for line in suffix_lines if line.strip()]
            if len(nonempty_suffix) < 2:
                continue
            reference_count = sum(1 for line in nonempty_suffix if self._is_reference_line(line))
            if reference_count >= 2 and reference_count >= max(2, int(len(nonempty_suffix) * 0.6)):
                cut_index = idx
                break

        if cut_index is None:
            return text, []

        before = "\n".join(lines[cut_index:])
        kept_lines = lines[:cut_index]
        while kept_lines and not kept_lines[-1].strip():
            kept_lines.pop()
        edits = [
            EditRecord(
                rule_id="trailing_reference_block_strip",
                action="delete",
                before=before[:400],
                after="",
                detail="Removed trailing note/reference block from page tail.",
            )
        ]
        return "\n".join(kept_lines), edits

    def _should_trim_trailing_reference_block(self, page: dict[str, Any], text: str) -> bool:
        page_type = str(page.get("page_type") or "")
        body_words = len((page.get("body_text") or "").split())
        notes_words = len((page.get("notes_text") or "").split())
        reference_words = len((page.get("reference_text") or "").split())

        # OCR has already separated note/reference material on many pages.
        # On substantive body pages, prefer recall over aggressive tail stripping.
        if page_type in {"body_with_notes", "body_only"}:
            if body_words >= 120:
                return False
            if page_type == "body_with_notes" and body_words >= 80 and (notes_words >= 40 or reference_words >= 40):
                return False
            if body_words >= 90 and self._looks_like_substantive_body_page(text):
                return False
        return True

    def _looks_like_substantive_body_page(self, text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 3:
            return False
        prose_like = 0
        for line in lines[:18]:
            words = line.split()
            if len(words) >= 8 and (SENTENCE_END_RE.search(line) or line[-1].isalnum()):
                prose_like += 1
        return prose_like >= 3

    def _find_protected_hits(self, text: str) -> list[dict[str, str]]:
        hits: list[dict[str, str]] = []
        for rule_id, pattern in PROTECTED_PATTERNS.items():
            for match in pattern.finditer(text):
                hits.append({"rule_id": rule_id, "evidence": match.group(0)})
        return hits

    def _find_conditional_flags(self, text: str) -> list[FlagRecord]:
        flags: list[FlagRecord] = []
        for match in FOOTNOTE_RE.finditer(text):
            flags.append(
                FlagRecord(
                    rule_id="footnote_markers",
                    detail="Footnote marker detected.",
                    evidence=match.group(0),
                )
            )
        for match in CITATION_RE.finditer(text):
            flags.append(
                FlagRecord(
                    rule_id="citations_and_bibliographic_refs",
                    detail="Citation-like span detected.",
                    evidence=match.group(0),
                )
            )
        reference_lines = [line.strip() for line in self._split_lines(text) if self._is_reference_line(line)]
        if len(reference_lines) >= 2:
            flags.append(
                FlagRecord(
                    rule_id="reference_suffix_material",
                    detail="Trailing bibliography/reference style lines detected.",
                    evidence=" | ".join(reference_lines[:3])[:160],
                )
            )
        if DOT_LEADER_RE.search(text):
            flags.append(
                FlagRecord(
                    rule_id="toc_index_material",
                    detail="Table-of-contents style dot leaders detected.",
                    evidence="dot_leaders",
                )
            )
        for line in self._split_lines(text):
            if FIGURE_RE.match(line):
                flags.append(
                    FlagRecord(
                        rule_id="figure_table_formula_labels",
                        detail="Figure or table label detected.",
                        evidence=line[:120],
                    )
                )
        for match in VALUE_RE.finditer(text):
            flags.append(
                FlagRecord(
                    rule_id="number_and_date_formatting",
                    detail="Potentially sensitive numeric value detected.",
                    evidence=match.group(0),
                )
            )
        return flags

    def _looks_like_toc_page(self, raw_text: str, cleaned_text: str) -> bool:
        raw = raw_text or ""
        text = cleaned_text or raw
        if not text.strip():
            return False
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if len(lines) < 4:
            return False
        heading_hit = bool(TOC_HEADING_RE.search(raw)) or bool(TOC_HEADING_RE.search(text))
        dot_leaders = len(DOT_LEADER_RE.findall(raw)) + len(DOT_LEADER_RE.findall(text))
        toc_entries = len(TOC_ENTRY_RE.findall(raw)) + len(TOC_ENTRY_RE.findall(text))
        page_number_lines = sum(
            1
            for line in lines
            if re.search(r"(?:\.{2,}|\s)\d{1,4}\s*$", line)
        )
        short_catalog_lines = sum(1 for line in lines[:40] if 8 <= len(line) <= 140)
        if heading_hit and (dot_leaders >= 2 or toc_entries >= 4 or page_number_lines >= 4):
            return True
        if heading_hit and short_catalog_lines >= 8 and page_number_lines >= 3:
            return True
        return False

    def _looks_like_front_matter_title_page(self, page_number: int, raw_text: str, cleaned_text: str) -> bool:
        if page_number > 6:
            return False
        raw = raw_text or ""
        text = cleaned_text or raw
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if not lines or len(lines) > 20:
            return False
        compact_len = len(" ".join(lines))
        if compact_len >= 320:
            return False
        if compact_len > 1200:
            return False
        heading_like = sum(
            1 for line in lines
            if self._looks_like_heading_line(line) or self._looks_like_heading_fragment(line) or line.isupper()
        )
        sentence_like = sum(
            1 for line in lines
            if len(line.split()) >= 6 and SENTENCE_END_RE.search(line)
        )
        if sentence_like >= 2:
            return False
        if page_number <= 2 and FRONTMATTER_CUE_RE.search(text) and compact_len <= 500:
            return True
        if heading_like >= 2 and sentence_like <= 1 and page_number <= 4:
            return True
        if FRONTMATTER_CUE_RE.search(text) and sentence_like == 0 and len(lines) <= 12:
            return True
        return False

    def _looks_like_publisher_meta_page(self, raw_text: str, cleaned_text: str) -> bool:
        text = cleaned_text or raw_text
        if not text.strip():
            return False
        nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
        compact_len = len(re.sub(r"\s+", "", text))
        cues = 0
        if ISBN_RE.search(text):
            cues += 1
        if EMAIL_RE.search(text):
            cues += 1
        if PHONE_RE.search(text):
            cues += 1
        if PUBLISHER_CUE_RE.search(text):
            cues += 1
        if PUBLISHER_BACKMATTER_RE.search(text):
            cues += 1
        if FRONTMATTER_CUE_RE.search(text):
            cues += 1
        meta_lines = 0
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if EMAIL_RE.search(stripped) or PHONE_RE.search(stripped):
                meta_lines += 1
            elif ISBN_RE.search(stripped):
                meta_lines += 1
            elif PUBLISHER_CUE_RE.search(stripped):
                meta_lines += 1
            elif PUBLISHER_BACKMATTER_RE.search(stripped):
                meta_lines += 1
            elif FRONTMATTER_CUE_RE.search(stripped):
                meta_lines += 1
        if cues >= 2 and meta_lines >= 3 and meta_lines >= max(3, int(len(nonempty_lines) * 0.6)):
            return True
        if ISBN_RE.search(text) and len(nonempty_lines) <= 3 and compact_len <= 220:
            return True
        if PUBLISHER_BACKMATTER_RE.search(text) and meta_lines >= 2 and compact_len <= 800:
            return True
        if FRONTMATTER_CUE_RE.search(text) and meta_lines >= 2 and len(nonempty_lines) <= 20 and compact_len <= 900:
            return True
        if ("можно купить" in text.lower() or "книжный салон" in text.lower()) and compact_len <= 900:
            return True
        return False

    def _looks_like_garbled_page(self, raw_text: str, cleaned_text: str, source: str) -> bool:
        text = cleaned_text or raw_text
        stripped = text.strip()
        if len(stripped) < 40:
            return False
        weird_chars = len(EXTENDED_LATIN_RE.findall(stripped))
        cyr = len(CYRILLIC_LETTER_RE.findall(stripped))
        lat = len(LATIN_LETTER_RE.findall(stripped))
        garbled_tokens = sum(1 for token in re.findall(r"\S+", stripped) if self._is_garbled_token(token))
        if ARCHIVE_CUE_RE.search(stripped) and garbled_tokens >= 2:
            return True
        if source == "extract_fallback" and "://" in stripped and garbled_tokens >= 3:
            return True
        if source in {"extract_fallback", "ocr"} and weird_chars >= 2 and cyr <= 12 and garbled_tokens >= 3:
            return True
        if weird_chars >= 4 and cyr <= 20 and lat >= 20 and garbled_tokens >= 4:
            return True
        if source == "extract_fallback" and garbled_tokens >= 4 and len(stripped.splitlines()) <= 8:
            return True
        return False

    def _looks_like_reference_only_page(self, raw_text: str, cleaned_text: str) -> bool:
        text = cleaned_text or raw_text
        if ABBREVIATION_LIST_HEADING_RE.search(text):
            return True
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 4:
            return False
        reference_lines = sum(1 for line in lines if self._is_reference_line(line))
        numbered_lines = sum(1 for line in lines if NUMBERED_REFERENCE_LINE_RE.match(line))
        backmatter_cue_lines = sum(1 for line in lines if REFERENCE_BACKMATTER_CUE_RE.search(line))
        long_paragraph_lines = sum(
            1 for line in lines if len(line) >= 140 and not NUMBERED_REFERENCE_LINE_RE.match(line)
        )
        if reference_lines >= max(4, int(len(lines) * 0.6)):
            return True
        if reference_lines >= 3 and sum(1 for line in lines if "http://" in line or "https://" in line or "www." in line) >= 2:
            return True
        if numbered_lines >= 6 and backmatter_cue_lines >= 4 and long_paragraph_lines <= 2:
            return True
        if numbered_lines >= 10 and long_paragraph_lines <= 1:
            return True
        return False

    def _should_keep_reference_heavy_body_page(self, page: dict[str, Any], text: str) -> bool:
        page_type = str(page.get("page_type") or "")
        body_words = len((page.get("body_text") or "").split())
        notes_words = len((page.get("notes_text") or "").split())
        reference_words = len((page.get("reference_text") or "").split())

        if page_type not in {"body_with_notes", "body_only"}:
            return False
        if body_words < 80:
            return False
        if self._looks_like_substantive_body_page(text):
            return True
        if body_words >= 120 and (notes_words >= 40 or reference_words >= 40):
            return True
        return False

    def _looks_like_glossary_page(self, raw_text: str, cleaned_text: str) -> bool:
        text = cleaned_text or raw_text
        if not text.strip():
            return False
        if GLOSSARY_HEADING_RE.search(text):
            return True
        entry_count = len(GLOSSARY_ENTRY_RE.findall(text))
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if entry_count >= 6 and len(lines) <= 40:
            return True
        return False

    def _is_reference_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if REFERENCE_BACKMATTER_CUE_RE.search(stripped) and NUMBERED_REFERENCE_LINE_RE.match(stripped):
            return True
        if NUMBERED_REFERENCE_LINE_RE.match(stripped) and REFERENCE_LINE_CUE_RE.search(stripped):
            return True
        if NUMBERED_REFERENCE_LINE_RE.match(stripped) and re.search(r"(?:19|20)\d{2}", stripped):
            return True
        if NUMBERED_REFERENCE_LINE_RE.match(stripped) and len(stripped) >= 60:
            return True
        if REFERENCE_LINE_CUE_RE.search(stripped):
            return True
        return False

    def _trim_leading_front_matter_block(self, page_number: int, text: str) -> tuple[str, list[EditRecord]]:
        if page_number > 8:
            return text, []
        if not FRONTMATTER_CUE_RE.search(text):
            return text, []
        lines = text.splitlines()
        start_idx = None
        for idx, line in enumerate(lines):
            if BODY_START_HEADING_RE.match(line.strip()):
                start_idx = idx
                break
        if start_idx is None:
            match = BODY_START_INLINE_RE.search(text)
            if match and match.start() >= 80:
                trimmed = text[match.start():].strip()
                if trimmed:
                    return trimmed, [
                        EditRecord(
                            rule_id="leading_front_matter_trim",
                            action="delete",
                            before=text[:400],
                            after=trimmed[:400],
                            detail="Trimmed leading rights/credits/front-matter block before first body heading.",
                        )
                    ]
        if start_idx is None or start_idx == 0:
            return text, []
        trimmed = "\n".join(lines[start_idx:]).strip()
        if not trimmed:
            return text, []
        return trimmed, [
            EditRecord(
                rule_id="leading_front_matter_trim",
                action="delete",
                before=text[:400],
                after=trimmed[:400],
                detail="Trimmed leading rights/credits/front-matter block before first body heading.",
            )
        ]

    def _is_garbled_token(self, token: str) -> bool:
        token = token.strip(".,;:!?()[]{}«»\"'")
        if len(token) < 4:
            return False
        if EXTENDED_LATIN_RE.search(token):
            return True
        has_lat = bool(LATIN_LETTER_RE.search(token))
        has_cyr = bool(CYRILLIC_LETTER_RE.search(token))
        if has_lat and has_cyr:
            return True
        if re.search(r"[{}|\\$]", token):
            return True
        if re.search(r"(?:0ОО|О00|00О)", token):
            return True
        return False

    def _should_merge_lines(self, current: str, nxt: str) -> bool:
        current_s = current.strip()
        next_s = nxt.strip()
        if not current_s or not next_s:
            return False
        if re.match(r"^(?:[-*•]|\d+[.)]|[A-Za-zА-Яа-яЁё]\)|[xхXХ]\s+)", next_s):
            return False
        if FIGURE_RE.match(next_s):
            return False
        if current_s.endswith((".", "!", "?", ":", ";")):
            return False
        if len(current_s) <= 2 or len(next_s) <= 2:
            return False
        if self._looks_like_heading_line(current_s):
            return False
        if next_s[0].isdigit():
            return False
        if re.match(r"^[A-ZА-ЯЁ]{4,}$", current_s):
            return False
        if re.match(r"^[A-ZА-ЯЁ]{3,}", next_s):
            return False
        return True

    def _should_merge_short_line(self, current: str, nxt: str) -> bool:
        current_s = current.strip()
        next_s = nxt.strip()
        if not current_s or not next_s:
            return False
        if self._is_noise_line(current_s) or self._is_noise_line(next_s):
            return False
        if self._looks_like_heading_fragment(current_s) and not self._looks_like_heading_fragment(next_s):
            return False
        if re.match(r"^(?:[-*•]|\d+[.)]|[A-Za-zА-Яа-яЁё]\)|[xхXХ]\s+)", next_s):
            return False
        if FIGURE_RE.match(next_s):
            return False
        if current_s.endswith(":") and next_s.startswith("-"):
            return True
        if len(current_s) <= 3 or len(next_s) <= 3:
            return True
        if len(current_s) <= 18 and not SENTENCE_END_RE.search(current_s):
            return True
        if not SENTENCE_END_RE.search(current_s) and LOWER_START_RE.match(next_s):
            return True
        return False

    def _should_flatten_block(self, lines: list[str]) -> bool:
        if len(lines) < 3:
            return False
        lengths = [len(line) for line in lines]
        short_lines = sum(length <= 24 for length in lengths)
        tiny_lines = sum(length <= 3 for length in lengths)
        terminal_lines = sum(1 for line in lines if SENTENCE_END_RE.search(line))
        if tiny_lines >= 2:
            return True
        if short_lines >= max(3, len(lines) - 1) and terminal_lines <= 1:
            return True
        if any(len(line.split()) == 1 for line in lines[1:-1]) and max(lengths) <= 80:
            return True
        return False

    def _join_lines(self, lines: list[str]) -> str:
        text = " ".join(line.strip() for line in lines if line.strip())
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([(\[«„])\s+", r"\1", text)
        text = re.sub(r"\s+([)\]»“])", r"\1", text)
        return text.strip()

    def _flatten_block(self, lines: list[str]) -> str:
        if len(lines) >= 2 and self._looks_like_heading_line(lines[0]):
            head = lines[0].strip()
            tail = self._join_lines(lines[1:])
            if tail:
                return f"{head}\n{tail}"
            return head
        return self._join_lines(lines)

    def _looks_like_heading_line(self, line: str) -> bool:
        compact = re.sub(r"\s+", " ", line.strip())
        if not compact:
            return False
        return bool(HEADING_LINE_RE.fullmatch(compact))

    def _looks_like_heading_fragment(self, line: str) -> bool:
        compact = re.sub(r"\s+", " ", line.strip())
        if not compact:
            return False
        if len(compact) > 40:
            return False
        letters = re.sub(r"[^A-ZА-ЯЁ]", "", compact)
        return len(letters) >= 4 and letters == letters.upper()

    def _is_noise_line(self, line: str) -> bool:
        if not line:
            return False
        if NOISE_LINE_RE.fullmatch(line):
            return True
        if line in {"®", "©", "{", "}", "=", "_", "`"}:
            return True
        if line.isdigit() and len(line) <= 3:
            return True
        if ROMAN_NUMERAL_RE.fullmatch(line) and len(line) <= 6:
            return True
        if len(line) == 1 and not line.isalnum():
            return True
        return False

    def _normalize_line_key(self, line: str) -> str:
        line = re.sub(r"\s+", " ", line.strip())
        return line

    def _split_lines(self, text: str, keep_empty: bool = False) -> list[str]:
        lines = [line.rstrip() for line in text.splitlines()]
        if keep_empty:
            return lines
        return [line for line in lines if line.strip()]

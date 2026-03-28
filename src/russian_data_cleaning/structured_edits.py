from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any


SOFT_HYPHEN_RE = re.compile("\u00AD")
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")
SPACE_RUN_RE = re.compile(r"[ \t\u00A0]{2,}")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
INLINE_BRACKET_NOTE_RE = re.compile(r"(?<=[A-Za-zА-Яа-яЁё])\[\d{1,3}\]")
INLINE_NUMERIC_NOTE_RE = re.compile(r"(?<=[A-Za-zА-Яа-яЁё»”\")\]])\d{1,3}(?=(?:\s|[.,;:!?…])|$)")
INLINE_SUPERSCRIPT_NOTE_RE = re.compile(r"(?<=[A-Za-zА-Яа-яЁё])[¹²³⁴⁵⁶⁷⁸⁹]")
ANGLE_PLACEHOLDER_RE = re.compile(r"<(?:…|\.{3})>")
RETURN_TO_INDEX_RE = re.compile(r"(?im)^\s*ВЕРНУТЬСЯ\s+К\s+ИНДЕКСУ\s*$")
REFERENCE_CUE_RE = re.compile(
    r"(?i)\b(?:см\.|цит\. по:|ibid\.|op\. cit\.|ргали\.|ф\.\s*\d+|оп\.\s*\d+|ед\.\s*хр\.|л\.\s*\d+|спб\.|м\.:|л\.;\s*м\.:|//)\b"
)

ALLOWED_INLINE_PATTERN_NAMES = (
    "angle_placeholders",
    "bracket_note_markers",
    "inline_numeric_note_markers",
    "return_to_index",
    "superscript_note_markers",
)

ALLOWED_OPERATIONS = (
    "delete_line_range",
    "merge_with_next",
    "normalize_spacing",
    "remove_inline_pattern",
    "split_after_text",
    "split_before_text",
    "strip_trailing_reference_block",
)


@dataclass
class AppliedEditRecord:
    rule_id: str
    action: str
    before: str
    after: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EditExecutionResult:
    text: str
    applied_edits: list[dict[str, Any]]
    notes: list[str]
    drop_page: bool


def render_numbered_text(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines:
        lines = [""]
    return "\n".join(f"{index:03d}| {line}" for index, line in enumerate(lines, start=1))


def parse_json_object(text: str) -> dict[str, Any]:
    normalized = strip_code_fences(text).strip()
    if not normalized:
        raise ValueError("empty model response")
    try:
        payload = json.loads(normalized)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(normalized):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(normalized[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("no JSON object found in model response")


def strip_code_fences(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


def execute_edit_plan(
    text: str,
    payload: dict[str, Any],
    *,
    allow_drop_page: bool = False,
    max_operations: int = 24,
) -> EditExecutionResult:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    nodes: list[dict[str, Any]] = [{"id": index, "text": line} for index, line in enumerate(normalized.split("\n"), start=1)]
    applied: list[dict[str, Any]] = []
    notes: list[str] = []
    synthetic_id = len(nodes) + 1

    drop_page = bool(payload.get("drop_page")) and allow_drop_page
    if bool(payload.get("drop_page")) and not allow_drop_page:
        notes.append("model_drop_page_ignored")

    operations = payload.get("operations", [])
    if not isinstance(operations, list):
        notes.append("invalid_operations_list_ignored")
        operations = []
    if len(operations) > max_operations:
        notes.append(f"operation_count_capped:{len(operations)}->{max_operations}")
        operations = operations[:max_operations]

    if drop_page:
        applied.append(
            AppliedEditRecord(
                rule_id="llm_non_body_page_drop",
                action="drop_page",
                before=normalized.strip(),
                after="",
                detail="Model marked page as obvious non-body content.",
            ).to_dict()
        )
        return EditExecutionResult(text="", applied_edits=applied, notes=notes, drop_page=True)

    for operation in operations:
        if not isinstance(operation, dict):
            notes.append("non_object_operation_ignored")
            continue
        op_name = str(operation.get("op", "")).strip().lower()
        if op_name not in ALLOWED_OPERATIONS:
            notes.append(f"unsupported_operation_ignored:{op_name or 'missing'}")
            continue

        if op_name == "delete_line_range":
            start_line = _coerce_positive_int(operation.get("start_line"))
            end_line = _coerce_positive_int(operation.get("end_line")) or start_line
            if start_line is None:
                notes.append("delete_line_range_missing_start_ignored")
                continue
            matched = [node for node in nodes if start_line <= int(node["id"]) <= int(end_line)]
            if not matched:
                notes.append(f"delete_line_range_missing_target:{start_line}-{end_line}")
                continue
            before = "\n".join(node["text"] for node in matched)
            nodes = [node for node in nodes if not (start_line <= int(node["id"]) <= int(end_line))]
            applied.append(
                AppliedEditRecord(
                    rule_id=_delete_rule_id(operation),
                    action="delete",
                    before=before,
                    after="",
                    detail=_operation_detail(operation, f"Deleted lines {start_line}-{end_line}."),
                ).to_dict()
            )
            continue

        if op_name == "merge_with_next":
            line_id = _coerce_positive_int(operation.get("line"))
            if line_id is None:
                notes.append("merge_with_next_missing_line_ignored")
                continue
            index = _find_node_index(nodes, line_id)
            if index is None or index + 1 >= len(nodes):
                notes.append(f"merge_with_next_missing_target:{line_id}")
                continue
            before = f"{nodes[index]['text']}\n{nodes[index + 1]['text']}"
            merged = _merge_lines(nodes[index]["text"], nodes[index + 1]["text"])
            nodes[index]["text"] = merged
            del nodes[index + 1]
            applied.append(
                AppliedEditRecord(
                    rule_id=_merge_rule_id(before, merged),
                    action="merge",
                    before=before,
                    after=merged,
                    detail=_operation_detail(operation, f"Merged line {line_id} with the following line."),
                ).to_dict()
            )
            continue

        if op_name == "split_before_text":
            line_id = _coerce_positive_int(operation.get("line"))
            marker = str(operation.get("text", "") or "")
            if line_id is None or not marker:
                notes.append("split_before_text_missing_args_ignored")
                continue
            index = _find_node_index(nodes, line_id)
            if index is None:
                notes.append(f"split_before_text_missing_target:{line_id}")
                continue
            current = nodes[index]["text"]
            position = current.find(marker)
            if position <= 0:
                notes.append(f"split_before_text_marker_not_found:{line_id}")
                continue
            left = current[:position].rstrip()
            right = current[position:].lstrip()
            nodes[index]["text"] = left
            nodes.insert(index + 1, {"id": synthetic_id, "text": right})
            synthetic_id += 1
            applied.append(
                AppliedEditRecord(
                    rule_id="heading_structure_repair",
                    action="split",
                    before=current,
                    after=f"{left}\n{right}".strip(),
                    detail=_operation_detail(operation, f"Split line {line_id} before marker text."),
                ).to_dict()
            )
            continue

        if op_name == "split_after_text":
            line_id = _coerce_positive_int(operation.get("line"))
            marker = str(operation.get("text", "") or "")
            if line_id is None or not marker:
                notes.append("split_after_text_missing_args_ignored")
                continue
            index = _find_node_index(nodes, line_id)
            if index is None:
                notes.append(f"split_after_text_missing_target:{line_id}")
                continue
            current = nodes[index]["text"]
            position = current.find(marker)
            if position < 0:
                notes.append(f"split_after_text_marker_not_found:{line_id}")
                continue
            position += len(marker)
            if position >= len(current):
                notes.append(f"split_after_text_marker_at_end:{line_id}")
                continue
            left = current[:position].rstrip()
            right = current[position:].lstrip()
            nodes[index]["text"] = left
            nodes.insert(index + 1, {"id": synthetic_id, "text": right})
            synthetic_id += 1
            applied.append(
                AppliedEditRecord(
                    rule_id="heading_structure_repair",
                    action="split",
                    before=current,
                    after=f"{left}\n{right}".strip(),
                    detail=_operation_detail(operation, f"Split line {line_id} after marker text."),
                ).to_dict()
            )
            continue

        if op_name == "remove_inline_pattern":
            pattern_name = str(operation.get("pattern", "") or "").strip()
            if pattern_name not in ALLOWED_INLINE_PATTERN_NAMES:
                notes.append(f"unsupported_inline_pattern_ignored:{pattern_name or 'missing'}")
                continue
            line_id = _coerce_positive_int(operation.get("line"))
            indexes = range(len(nodes)) if line_id is None else [idx for idx in [_find_node_index(nodes, line_id)] if idx is not None]
            changed_before: list[str] = []
            changed_after: list[str] = []
            for index in indexes:
                before = nodes[index]["text"]
                after = _apply_inline_pattern(before, pattern_name)
                if after == before:
                    continue
                nodes[index]["text"] = after
                changed_before.append(before)
                changed_after.append(after)
            if not changed_before:
                notes.append(f"inline_pattern_noop:{pattern_name}")
                continue
            applied.append(
                AppliedEditRecord(
                    rule_id=_inline_pattern_rule_id(pattern_name),
                    action="replace",
                    before="\n".join(changed_before),
                    after="\n".join(changed_after),
                    detail=_operation_detail(operation, f"Removed inline pattern: {pattern_name}."),
                ).to_dict()
            )
            continue

        if op_name == "strip_trailing_reference_block":
            start_line = _coerce_positive_int(operation.get("start_line"))
            cut_index = _find_trailing_reference_cut(nodes, start_line=start_line)
            if cut_index is None:
                notes.append("strip_trailing_reference_block_noop")
                continue
            before = "\n".join(node["text"] for node in nodes[cut_index:]).strip()
            nodes = nodes[:cut_index]
            applied.append(
                AppliedEditRecord(
                    rule_id="trailing_reference_block_strip",
                    action="delete",
                    before=before,
                    after="",
                    detail=_operation_detail(operation, "Removed trailing reference or note block."),
                ).to_dict()
            )
            continue

        if op_name == "normalize_spacing":
            before = "\n".join(node["text"] for node in nodes)
            for node in nodes:
                node["text"] = _normalize_line_spacing(node["text"])
            nodes = _collapse_extra_blank_lines(nodes)
            after = "\n".join(node["text"] for node in nodes)
            if after == before:
                notes.append("normalize_spacing_noop")
                continue
            applied.append(
                AppliedEditRecord(
                    rule_id="spacing_cleanup",
                    action="normalize",
                    before=before,
                    after=after,
                    detail=_operation_detail(operation, "Normalized spacing and blank-line runs."),
                ).to_dict()
            )
            continue

    updated = "\n".join(node["text"] for node in nodes)
    updated = updated.replace("\r\n", "\n").replace("\r", "\n")
    updated = re.sub(r"[ \t]+\n", "\n", updated)
    updated = re.sub(r"\n{3,}", "\n\n", updated)
    return EditExecutionResult(text=updated.strip(), applied_edits=applied, notes=notes, drop_page=False)


def apply_edit_plan(
    text: str,
    payload: dict[str, Any],
    *,
    allow_drop_page: bool = False,
    max_operations: int = 24,
) -> tuple[str, list[dict[str, Any]], list[str], bool]:
    result = execute_edit_plan(
        text,
        payload,
        allow_drop_page=allow_drop_page,
        max_operations=max_operations,
    )
    return result.text, result.applied_edits, result.notes, result.drop_page


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _find_node_index(nodes: list[dict[str, Any]], line_id: int) -> int | None:
    for index, node in enumerate(nodes):
        if int(node["id"]) == line_id:
            return index
    return None


def _merge_lines(left: str, right: str) -> str:
    left_clean = left.rstrip()
    right_clean = right.lstrip()
    if not left_clean:
        return right_clean
    if not right_clean:
        return left_clean
    if re.search(r"[A-Za-zА-Яа-яЁё]-$", left_clean) and re.match(r"^[A-Za-zА-Яа-яЁё]", right_clean):
        return left_clean[:-1] + right_clean
    if left_clean.endswith(("(", "«", "\"", "“")):
        return left_clean + right_clean
    if right_clean.startswith((",", ".", ";", ":", "!", "?", ")", "]", "»")):
        return left_clean + right_clean
    return f"{left_clean} {right_clean}"


def _normalize_line_spacing(text: str) -> str:
    updated = SOFT_HYPHEN_RE.sub("", text)
    updated = ZERO_WIDTH_RE.sub("", updated)
    updated = SPACE_RUN_RE.sub(" ", updated)
    updated = SPACE_BEFORE_PUNCT_RE.sub(r"\1", updated)
    return updated.strip()


def _collapse_extra_blank_lines(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collapsed: list[dict[str, Any]] = []
    blank_streak = 0
    for node in nodes:
        if node["text"].strip():
            blank_streak = 0
            collapsed.append(node)
            continue
        blank_streak += 1
        if blank_streak <= 1:
            collapsed.append({"id": node["id"], "text": ""})
    return collapsed


def _apply_inline_pattern(text: str, pattern_name: str) -> str:
    if pattern_name == "bracket_note_markers":
        return INLINE_BRACKET_NOTE_RE.sub("", text)
    if pattern_name == "inline_numeric_note_markers":
        return INLINE_NUMERIC_NOTE_RE.sub("", text)
    if pattern_name == "superscript_note_markers":
        return INLINE_SUPERSCRIPT_NOTE_RE.sub("", text)
    if pattern_name == "return_to_index":
        return RETURN_TO_INDEX_RE.sub("", text).strip()
    if pattern_name == "angle_placeholders":
        return ANGLE_PLACEHOLDER_RE.sub("", text)
    return text


def _inline_pattern_rule_id(pattern_name: str) -> str:
    if pattern_name in {"bracket_note_markers", "inline_numeric_note_markers", "superscript_note_markers"}:
        return "inline_note_marker_strip"
    if pattern_name == "return_to_index":
        return "isolated_ocr_noise"
    return "spacing_cleanup"


def _merge_rule_id(before: str, after: str) -> str:
    if "-\n" in before and len(after) >= 6:
        return "line_end_hyphenation"
    return "fake_paragraph_breaks"


def _delete_rule_id(operation: dict[str, Any]) -> str:
    reason = str(operation.get("reason", "") or "").lower()
    if any(token in reason for token in ("reference", "note", "citation", "footnote", "bibliograph")):
        return "trailing_reference_block_strip"
    return "isolated_ocr_noise"


def _operation_detail(operation: dict[str, Any], fallback: str) -> str:
    reason = str(operation.get("reason", "") or "").strip()
    return reason or fallback


def _find_trailing_reference_cut(nodes: list[dict[str, Any]], *, start_line: int | None = None) -> int | None:
    if len(nodes) < 2:
        return None
    if start_line is not None:
        explicit_index = _find_node_index(nodes, start_line)
        if explicit_index is not None:
            suffix = [node for node in nodes[explicit_index:] if node["text"].strip()]
            if len(suffix) >= 2 and _count_note_like_lines(suffix) >= 2:
                return explicit_index
    start_index = max(0, len(nodes) - 20)
    for index in range(start_index, len(nodes)):
        suffix = [node for node in nodes[index:] if node["text"].strip()]
        if len(suffix) < 2:
            continue
        if _count_note_like_lines(suffix) >= 2 and _is_note_like_line(suffix[0]["text"]):
            return index
    return None


def _count_note_like_lines(nodes: list[dict[str, Any]]) -> int:
    return sum(1 for node in nodes if _is_note_like_line(node["text"]))


def _is_note_like_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r"^\d{1,3}[.)]?\s+", stripped):
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

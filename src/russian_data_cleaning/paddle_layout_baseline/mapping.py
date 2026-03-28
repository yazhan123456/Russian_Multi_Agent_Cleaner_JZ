from __future__ import annotations

import re


RAW_LABEL_MAP = {
    "text": "body",
    "paragraph": "body",
    "plain text": "body",
    "plain_text": "body",
    "article": "body",
    "title": "title",
    "header": "note",
    "footer": "note",
    "footnote": "note",
    "reference": "note",
    "references": "note",
    "caption": "note",
    "figure caption": "note",
    "figure_caption": "note",
    "table caption": "note",
    "table_caption": "note",
    "number": "note",
    "list": "note",
    "index": "note",
    "toc": "note",
    "table of contents": "note",
    "contents": "note",
    "formula": "note",
    "equation": "note",
    "table": "table",
    "picture": "picture",
    "image": "picture",
    "figure": "picture",
    "chart": "picture",
}

TEXT_LIKE_RE = re.compile(r"(?:text|paragraph|article)")
TITLE_RE = re.compile(r"(?:title|heading)")
TABLE_RE = re.compile(r"(?:table|spreadsheet)")
PICTURE_RE = re.compile(r"(?:figure|picture|image|photo|chart|illustration)")
NOTE_RE = re.compile(
    r"(?:header|footer|footnote|reference|caption|list|index|contents|toc|equation|formula|bibliography|number)"
)

ACTION_BY_MAPPED_LABEL = {
    "title": "keep",
    "body": "keep",
    "note": "mask",
    "picture": "mask",
    "table": "mask",
}


def normalize_raw_label(raw_label: str | None) -> str:
    if not raw_label:
        return ""
    return re.sub(r"[\s_-]+", " ", str(raw_label).strip().lower())


def map_layout_label(raw_label: str | None) -> str:
    normalized = normalize_raw_label(raw_label)
    if normalized in RAW_LABEL_MAP:
        return RAW_LABEL_MAP[normalized]
    if TITLE_RE.search(normalized):
        return "title"
    if TABLE_RE.search(normalized):
        return "table"
    if PICTURE_RE.search(normalized):
        return "picture"
    if NOTE_RE.search(normalized):
        return "note"
    if TEXT_LIKE_RE.search(normalized):
        return "body"
    # Prefer recall over precision for unknown text-like outputs.
    return "body"


def action_for_label(mapped_label: str) -> str:
    return ACTION_BY_MAPPED_LABEL.get(mapped_label, "mask")

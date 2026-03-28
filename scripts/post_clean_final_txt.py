#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path


SOFT_HYPHEN_RE = re.compile("\u00AD")
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")
SPACE_RUN_RE = re.compile(r"[ \t\u00A0]{2,}")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
INDEX_MARKER_RE = re.compile(r"ВЕРНУТЬСЯ\s+К\s+ИНДЕКСУ", re.IGNORECASE)
PLACEHOLDER_RE = re.compile(r"<\s*(?:\.{2,}|\u2026+)\s*>")
ANGLE_WRAPPED_TEXT_RE = re.compile(r"<\s*([A-Za-zА-Яа-яЁё0-9][^<>]{0,38}?)\s*>")
ARROW_ARTIFACT_RE = re.compile(r"—>")
EMBEDDED_ALLCAPS_HEADER_RE = re.compile(
    r"(\b[А-Яа-яЁё]{2,}-)\s+(?:[A-ZА-ЯЁ]{4,}(?:\s+[A-ZА-ЯЁ]{2,}){0,4})\s+([а-яё]{2,}\b)"
)
LINE_WRAP_HYPHEN_RE = re.compile(r"([A-Za-zА-Яа-яЁё]+)[-\u2010\u2011]\s*\n\s*([a-zа-яё][A-Za-zА-Яа-яЁё]*)")
SPACED_HYPHEN_RE = re.compile(r"\b([A-Za-zА-Яа-яЁё]+)[-\u2010\u2011]\s+([a-zа-яё][A-Za-zА-Яа-яЁё]*)")
COMPACT_SHORT_HYPHEN_RE = re.compile(r"\b([A-Za-zА-Яа-яЁё]{1,3})[-\u2010\u2011]([a-zа-яё]{4,})\b")
LONG_STEM_SHORT_SUFFIX_RE = re.compile(r"\b([A-Za-zА-Яа-яЁё]{4,})[-\u2010\u2011]([a-zа-яё]{1,3})\b")
COMPACT_MIDWORD_HYPHEN_RE = re.compile(r"\b([А-Яа-яЁё]{2,5})[-\u2010\u2011]([а-яё]{2,4})\b")
PAGE_HEADING_FRAGMENT = (
    r"(?:Глава|Часть|ЧАСТЬ)\s+[IVXLCDM0-9]+(?:[.:]\s*|$)|РОСПУСК\b|ПРОЛОГ\b|ЭПИЛОГ\b"
)
PAGE_NUMBER_HEADING_START_RE = re.compile(rf"(?m)^\s*\d{{1,4}}\s+(?={PAGE_HEADING_FRAGMENT})")
PAGE_NUMBER_HEADING_INLINE_RE = re.compile(rf"(?<!\n)[ \t]+\d{{1,4}}\s+(?={PAGE_HEADING_FRAGMENT})")
INLINE_HEADING_PREFIX_RE = re.compile(
    r'^(?P<header>(?:Глава|Часть|ЧАСТЬ)\s+[IVXLCDM0-9]+(?:[.:]\s*|\s+)'
    r'(?:[«»"A-ZА-ЯЁ0-9 .,:;—–\-()]{3,160}\s+)?)'
    r'(?P<rest>[а-яё].+)$'
)
LEADING_SECTION_PREFIX_RE = re.compile(
    r'^(?P<label>(?:Глава|Часть|ЧАСТЬ)\s+[IVXLCDM0-9]+(?:[.:]?\s+))(?P<body>.+)$'
)
SPACE_SPLIT_LONG_STEM_RE = re.compile(r"\b([А-Яа-яЁё]{6,})\s+([а-яё]{2,6})\b")
SPACE_SPLIT_PREFIX_RE = re.compile(r"\b([А-Яа-яЁё]{2,4})\s+([а-яё]{6,})\b")
SPACE_SPLIT_MIDWORD_RE = re.compile(r"\b([А-Яа-яЁё]{4,6})\s+([а-яё]{3,8})\b")
CYRILLIC_WORD_RE = re.compile(r"^[А-Яа-яЁё]+$")
MOJIBAKE_CHAR_RE = re.compile(r"[\u0180-\u02AF]")
MOJIBAKE_LEADING_RE = re.compile(
    r"^(?!\d+\.\d+\.\s+)(?P<garble>(?=[^\n]*[\u0180-\u02AF])[\u0180-\u04FFA-Za-z0-9\s.\-–—()«»\"'/]{4,}?)\s+(?=[А-ЯЁ][а-яё])"
)
MOJIBAKE_NUMBERED_RE = re.compile(
    r"(\b\d+\.\d+\.\s+)(?P<garble>(?=[^\n]*[\u0180-\u02AF])[\u0180-\u04FFA-Za-z0-9\s.\-–—()«»\"'/]{4,}?)\s+(?=[А-ЯЁ][а-яё])"
)
MOJIBAKE_NUMBERED_EOL_RE = re.compile(
    r"(\b\d+\.\d+\.\s+)(?P<garble>(?=[^\n]*[\u0180-\u02AF])[\u0180-\u04FFA-Za-z0-9\s.\-–—()«»\"'/]{4,})$"
)
PROMO_LINE_RE = re.compile(
    r"^(?:Опубликовано:|AnonDir:|«Исследование теорий заговора|Conspiracy Theories Explored)",
    re.IGNORECASE,
)
RUNNING_HEADER_LINE_RE = re.compile(r"^(?:БЕЛОЕ ДВИЖЕНИЕ|К\. С\. Веселовский)$")
NOISE_ONLY_LINE_RE = re.compile(r"^[~`^_=+*#@&¥<>/\\|:;.,'\"!?()\[\]{}-]{4,}$")
HEADER_NOISE_FRAGMENT_RE = re.compile(
    r"(?:r&\?-\.\.\.|tQ-\.\.\.|fQ=;|ENG о А ¥: #2|[~_=\-<>/\\|#@&¥]{4,}[A-Za-zА-Яа-яЁё0-9~_=\-<>/\\|#@&¥]*)"
)
PROMPT_LEAK_START_RE = re.compile(
    r"^(?:Here's a step-bystep breakdown of the structural cleanup|1\.\s+\*\*Identify and remove artificial blank lines|This results in the REPAIRED PAGE output)\b",
    re.IGNORECASE,
)
TRAILING_BACKMATTER_START_RE = re.compile(
    r"^(?:Именной указатель|СОДЕРЖАНИЕ|Научное издание|Научно-просветительное издание|"
    r"Список иллюстраций\b|КОММЕНТАРИИ\b|Список\s+сокращен(?:ий|ИЯ)\b|"
    r"КОРОТКО\s+ОБ\s+АВТОРАХ\b|СВЕДЕНИЯ\s+ОБ\s+АВТОРАХ\b|ABOUT\s+THE\s+AUTHORS\b|"
    r"Вклейка\s+[A-ZА-ЯЁ])",
    re.IGNORECASE,
)
HEADING_RE = re.compile(
    r"^(?:ГЛАВА|Глава|CHAPTER|Chapter|ЧАСТЬ|Часть|PART|Part)\s+[IVXLCDM0-9]+(?:[.:]\s*|\s+|$)"
)
OCR_BULLET_LINE_RE = re.compile(r"^\s*[xхXХ]\s+(?=[A-Za-zА-Яа-яЁё])")
OCR_BULLET_INLINE_RE = re.compile(r"(?:(?<=^)|(?<=[;:]))\s*[xхXХ]\s+(?=[A-Za-zА-Яа-яЁё])")
NOTE_LINE_RE = re.compile(r"^\s*(?:\(?\d{1,3}\)|\[\d{1,3}\]|\d{1,3}[.)])\s+")
NUMBERED_SECTION_RE = re.compile(r"^\d+(?:\.\d+){1,4}\.?(?:\s+|$)")
DIALOGUE_LINE_RE = re.compile(r"^[—–-]\s*\S")
LIST_LINE_RE = re.compile(r"^(?:[-•*]\s+|\d{1,3}[.)]\s+)")
ALL_CAPS_TITLE_RE = re.compile(r"^(?=.*[А-ЯЁA-Z])(?:[А-ЯЁA-Z0-9IVXLCDM][А-ЯЁA-Z0-9IVXLCDM\s«»\"'()\-]{2,})$")
LOWERCASE_CONTINUATION_RE = re.compile(r'^[«"“„(]*[а-яё]')
PARAGRAPH_TERMINAL_RE = re.compile(r"[.!?…][\"'»”)]*$")
FIGURE_LABEL_RE = re.compile(r"(?:Рис|Илл|Табл)\.", re.IGNORECASE)
STANDALONE_FIGURE_CAPTION_RE = re.compile(
    r"^(?:Рис|Илл|Табл)\.\s*\d+[а-яё]?(?:\s*[–—-]\s*\d+[а-яё]?)?(?:\s*,\s*\d+[а-яё]?)?\.\s+.{3,260}$",
    re.IGNORECASE,
)
LEADING_FIGURE_CAPTION_PREFIX_RE = re.compile(
    r"^(?P<caption>(?:Рис|Илл|Табл)\.\s*\d+[а-яё]?(?:\s*[–—-]\s*\d+[а-яё]?)?"
    r"(?:\s*,\s*\d+[а-яё]?)?\.\s+.{3,260}?\.)\s+(?P<rest>[а-яё].+)$",
    re.IGNORECASE,
)
TRAILING_FIGURE_CAPTION_SUFFIX_RE = re.compile(
    r"^(?P<body>.+?[.!?…»”\)])\s+(?P<caption>(?:Рис|Илл|Табл)\.\s*\d+[а-яё]?"
    r"(?:\s*[–—-]\s*\d+[а-яё]?)?(?:\s*,\s*\d+[а-яё]?)?\.\s+.{3,260})$",
    re.IGNORECASE,
)
FIGURE_CAPTION_SPLIT_RE = re.compile(r"\.\s+(?=[а-яё])")
SENTENCE_SPLIT_RE = re.compile(r"\.\s+(?=[А-ЯЁа-яё])")
INLINE_BIBLIOGRAPHY_HEADING_RE = re.compile(r"(?i)\bСписок литературы\b[:.]?\s*")
INLINE_FIGURE_CONTINUATION_RE = re.compile(r"(?<=\.)\s+(?=[а-яё][а-яё-]{2,})")
BODY_LIKE_CONTINUATION_START_RE = re.compile(r'^[«"“„(]*[А-ЯЁа-яё][А-Яа-яёЁ\-]{2,}')
BODY_LIKE_CONTINUATION_CUE_RE = re.compile(
    r"(?i)\b(?:это|этот|эта|эти|однако|поэтому|при этом|в результате|в данном|"
    r"исследовател|археолог|можно|следует|позвол|показыва|служил|служила|служили|"
    r"являл|являет|были|было|стал|стала|стали|оказал|оказыва|имеет|имели|"
    r"обнаруж|применен|применя|доказыва|характериз|подтвержда|свидетельств|"
    r"например|прежде|кроме|здесь|далее|первый|второй|вопрос|пример)\b"
)
SECTION_HEADING_FIGURE_PREFIX_RE = re.compile(
    r"^(?P<header>(?:Глава|Часть|ЧАСТЬ)\s+[IVXLCDM0-9]+(?:\.\s*|\s+).{3,120}?)\s+(?=(?:Рис|Илл|Табл)\.)",
    re.IGNORECASE,
)
AUTHOR_TITLE_PREFIX_RE = re.compile(r"^[А-ЯЁ]\.[А-ЯЁ]\.\s+[А-ЯЁ][а-яё]+(?:\.\s+|\s+).{10,160}$")
AUTHOR_TITLE_FIGURE_PREFIX_RE = re.compile(
    r"^(?P<header>[А-ЯЁ]\.[А-ЯЁ]\.\s+[А-ЯЁ][а-яё]+(?:\.\s+|\s+).{10,180}?)\s+(?=(?:Рис|Илл|Табл)\.)",
    re.IGNORECASE,
)
CAPTION_SOURCE_CUE_RE = re.compile(r"(?i)\b(?:РГИА|РГАВМФ|ОР\s+БАН|АСП-\d+|Публикуется впервые|музей|архив)\b")
REFERENCE_CUE_RE = re.compile(
    r"(?i)(?:цит\.\s*соч\.|там же|references?|bibliography|список литературы|doi\b|isbn\b|источник:|//\s*http|http://|https://)"
)
REFERENCE_BLOCK_START_RE = re.compile(
    r"^(?:список литературы|библиограф(?:ия|ический список)|references?)\b",
    re.IGNORECASE,
)
SLASH_LINK_RE = re.compile(r"\s+//\s+https?://\S+", re.IGNORECASE)
TRAILING_SOURCE_RE = re.compile(r"(?i)\bИсточник:\b.*$")
FOOTNOTE_SUFFIX_RE = re.compile(r'(?<=[\w»”"\)])(?:\?{0,2}\d+(?:[!?])?|\?{2})\)')
PRESERVED_SPACED_HYPHEN_PAIRS = {
    ("из", "за"),
    ("из", "под"),
    ("по", "видимому"),
    ("по", "прежнему"),
    ("во", "первых"),
    ("во", "вторых"),
    ("в", "третьих"),
}
PRESERVED_SHORT_SUFFIX_HYPHEN_RIGHTS = {"то"}
PRESERVED_COMPACT_MIDWORD_PAIRS = {
    ("из", "за"),
    ("из", "под"),
    ("кто", "то"),
    ("что", "то"),
    ("где", "то"),
    ("как", "то"),
    ("когда", "то"),
    ("какой", "то"),
    ("чей", "то"),
    ("кое", "как"),
}
PRESERVED_COMPACT_MIDWORD_LEFTS = {"кое", "экс", "вице", "лейб", "обер", "унтер"}
PRESERVED_COMPACT_MIDWORD_RIGHTS = {"то", "ли", "де"}
COMMON_COMPOUND_NORMALIZATIONS = {
    "поразному": "по-разному",
    "всетаки": "все-таки",
    "когдалибо": "когда-либо",
    "когданибудь": "когда-нибудь",
    "ктолибо": "кто-либо",
    "ктонибудь": "кто-нибудь",
    "чтолибо": "что-либо",
    "чтонибудь": "что-нибудь",
    "гделибо": "где-либо",
    "гденибудь": "где-нибудь",
    "какойлибо": "какой-либо",
    "какойнибудь": "какой-нибудь",
    "ктото": "кто-то",
    "чтото": "что-то",
    "какойто": "какой-то",
}
COMMON_SPACE_SPLIT_NORMALIZATIONS = {
    "опре деления": "определения",
    "сожа ление": "сожаление",
    "сожа лением": "сожалением",
    "британско го": "британского",
    "про буждение": "пробуждение",
    "исто рии": "истории",
}
COMMON_JOINED_WORD_NORMALIZATIONS = {
    "траурногозала": "траурного зала",
    "конюшенногомоста": "конюшенного моста",
    "гвардейскихполка": "гвардейских полка",
    "крестнымходом": "крестным ходом",
    "монастырскогодвора": "монастырского двора",
    "головленковатюрьма": "Головленкова тюрьма",
    "трапезныхпалатах": "трапезных палатах",
    "подъемныхдлиною": "подъемных длиною",
    "центральногомонастырскогодвора": "центрального монастырского двора",
    "крепостистоит": "крепости стоит",
    "всекомнаты": "все комнаты",
    "внутреннююситуацию": "внутреннюю ситуацию",
    "межеваниятакже": "межевания также",
    "выровненнымкантом": "выровненным кантом",
    "ивсе": "и все",
    "устроенныеосями": "устроенные осями",
    "километрахниже": "километрах ниже",
    "покрываемойпланом": "покрываемой планом",
    "частойпричиной": "частой причиной",
    "незавершенностиработ": "незавершенности работ",
    "населенногопункта": "населенного пункта",
    "поворотногопункта": "поворотного пункта",
    "межевогоплана": "межевого плана",
    "уездныхпланах": "уездных планах",
    "землемернойкниги": "землемерной книги",
    "деталипроцесса": "детали процесса",
    "экспозициимузея": "экспозиции музея",
    "объединенногомузея": "объединенного музея",
    "обобщающихработ": "обобщающих работ",
    "землянымвалам": "земляным валам",
    "временипочти": "времени почти",
    "планамповезло": "планам повезло",
    "земликоторой": "земли которой",
    "острожныхстен": "острожных стен",
    "оштукатуренныхстенах": "оштукатуренных стенах",
    "монастырясто": "монастыря сто",
    "Семеновскогополка": "Семеновского полка",
    "Преображенскогополка": "Преображенского полка",
    "Измайловскогополка": "Измайловского полка",
    "Ямбургскогополка": "Ямбургского полка",
    "гарнизонногополка": "гарнизонного полка",
    "Конногвардейскогополка": "Конногвардейского полка",
    "бывшегополка": "бывшего полка",
    "Троицкогособора": "Троицкого собора",
    "Софийскогособора": "Софийского собора",
    "Успенскогособора": "Успенского собора",
    "Георгиевскогособора": "Георгиевского собора",
    "Богоявленскогособора": "Богоявленского собора",
    "Благовещенскогособора": "Благовещенского собора",
    "Петропавловскогособора": "Петропавловского собора",
    "предшествующегособора": "предшествующего собора",
    "апсидысобора": "апсиды собора",
    "Всешутейшегособора": "Всешутейшего собора",
    "Музеязаповедника": "Музея-заповедника",
    "Москвойрекой": "Москвой-рекой",
}
COMMON_SHORT_STANDALONE_WORDS = {
    "бы",
    "был",
    "была",
    "были",
    "было",
    "вас",
    "век",
    "вне",
    "все",
    "всех",
    "где",
    "года",
    "даже",
    "два",
    "для",
    "друг",
    "его",
    "ее",
    "если",
    "еще",
    "или",
    "им",
    "их",
    "как",
    "кто",
    "лет",
    "ли",
    "лишь",
    "мир",
    "мы",
    "над",
    "нам",
    "нас",
    "него",
    "ней",
    "не",
    "них",
    "но",
    "об",
    "одна",
    "одни",
    "одну",
    "она",
    "они",
    "оно",
    "от",
    "под",
    "по",
    "при",
    "про",
    "раз",
    "роль",
    "себе",
    "себя",
    "сил",
    "сила",
    "силы",
    "так",
    "тех",
    "то",
    "уже",
    "это",
    "эту",
    "этого",
    "этом",
}
SPACE_SPLIT_SUFFIX_FALLBACKS = {
    "ого",
    "его",
    "ому",
    "ими",
    "ыми",
    "ому",
    "ыми",
    "ыми",
    "ого",
    "ского",
    "скому",
    "скими",
    "ских",
    "ской",
    "скую",
    "ское",
    "ские",
    "ства",
    "стве",
    "ству",
    "ством",
    "ствии",
    "ствия",
    "ствием",
    "ся",
    "сь",
}
SPACE_SPLIT_PREFIX_ROOTS = {
    "ис": ("след", "следов", "польз", "ключ", "чез", "черп", "прав", "кус", "ход"),
    "про": ("извод",),
}


@dataclass
class FileReport:
    file_name: str
    before_chars: int
    after_chars: int
    before_lines: int
    after_lines: int
    removed_sentences: int
    removed_promos: int
    removed_note_like: int
    removed_reference_like: int
    url_count_before: int
    url_count_after: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "file_name": self.file_name,
            "before_chars": self.before_chars,
            "after_chars": self.after_chars,
            "before_lines": self.before_lines,
            "after_lines": self.after_lines,
            "removed_sentences": self.removed_sentences,
            "removed_promos": self.removed_promos,
            "removed_note_like": self.removed_note_like,
            "removed_reference_like": self.removed_reference_like,
            "url_count_before": self.url_count_before,
            "url_count_after": self.url_count_after,
        }


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = SOFT_HYPHEN_RE.sub("", text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = INDEX_MARKER_RE.sub("", text)
    text = PLACEHOLDER_RE.sub("", text)
    text = ANGLE_WRAPPED_TEXT_RE.sub(r"\1", text)
    text = ARROW_ARTIFACT_RE.sub("—", text)
    text = EMBEDDED_ALLCAPS_HEADER_RE.sub(r"\1 \2", text)
    text = fix_line_wrap_hyphenation(text)
    text = normalize_common_compounds(text)
    text = fix_intraword_space_splits(text)
    text = normalize_common_joined_words(text)
    text = INLINE_BIBLIOGRAPHY_HEADING_RE.sub("", text)
    text = strip_mojibake_heading_fragments(text)
    text = normalize_ocr_bullet_markers(text)
    text = text.replace("\t", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_page_number_heading_artifacts(text: str) -> str:
    text = PAGE_NUMBER_HEADING_START_RE.sub("", text)
    text = PAGE_NUMBER_HEADING_INLINE_RE.sub("\n", text)
    return text


def _hyphen_replacer(match: re.Match[str]) -> str:
    left = match.group(1)
    right = match.group(2)
    # Long-long fragments are usually artificial OCR line-wrap hyphenation.
    # Short prefixes like "по-" keep the lexical hyphen.
    if len(left) >= 3 and len(right) >= 3:
        return f"{left}{right}"
    return f"{left}-{right}"


def _spaced_hyphen_replacer(match: re.Match[str]) -> str:
    left = match.group(1)
    right = match.group(2)
    lowered = (left.lower(), right.lower())
    # Keep a narrow set of real lexical compounds normalized as "left-right".
    if lowered in PRESERVED_SPACED_HYPHEN_PAIRS:
        return f"{left}-{right}"
    # Long modifier stems like "социально-экономический" should retain the hyphen.
    if len(left) >= 8 and len(right) >= 6 and left.lower().endswith(("о", "е")):
        return f"{left}-{right}"
    return f"{left}{right}"


def _compact_short_hyphen_replacer(match: re.Match[str]) -> str:
    left = match.group(1)
    right = match.group(2)
    lowered = (left.lower(), right.lower())
    if lowered in PRESERVED_SPACED_HYPHEN_PAIRS:
        return f"{left}-{right}"
    return f"{left}{right}"


def _long_stem_short_suffix_replacer(match: re.Match[str]) -> str:
    left = match.group(1)
    right = match.group(2)
    if right.lower() in PRESERVED_SHORT_SUFFIX_HYPHEN_RIGHTS:
        return f"{left}-{right}"
    return f"{left}{right}"


def _compact_midword_hyphen_replacer(match: re.Match[str]) -> str:
    left = match.group(1)
    right = match.group(2)
    lowered = (left.lower(), right.lower())
    if lowered in PRESERVED_SPACED_HYPHEN_PAIRS or lowered in PRESERVED_COMPACT_MIDWORD_PAIRS:
        return f"{left}-{right}"
    if left.lower() in PRESERVED_COMPACT_MIDWORD_LEFTS or right.lower() in PRESERVED_COMPACT_MIDWORD_RIGHTS:
        return f"{left}-{right}"
    if len(left) + len(right) < 5:
        return f"{left}-{right}"
    return f"{left}{right}"


def fix_line_wrap_hyphenation(text: str) -> str:
    previous = None
    current = text
    for _ in range(4):
        if current == previous:
            break
        previous = current
        current = LINE_WRAP_HYPHEN_RE.sub(_hyphen_replacer, current)
        current = SPACED_HYPHEN_RE.sub(_spaced_hyphen_replacer, current)
        current = COMPACT_SHORT_HYPHEN_RE.sub(_compact_short_hyphen_replacer, current)
        current = LONG_STEM_SHORT_SUFFIX_RE.sub(_long_stem_short_suffix_replacer, current)
        current = COMPACT_MIDWORD_HYPHEN_RE.sub(_compact_midword_hyphen_replacer, current)
    return current


def _preserve_case_replacement(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def normalize_common_compounds(text: str) -> str:
    for source, target in COMMON_COMPOUND_NORMALIZATIONS.items():
        pattern = re.compile(rf"\b{re.escape(source)}\b", re.IGNORECASE)
        text = pattern.sub(lambda m, repl=target: _preserve_case_replacement(m.group(0), repl), text)
    return text


def normalize_common_space_splits(text: str) -> str:
    for source, target in COMMON_SPACE_SPLIT_NORMALIZATIONS.items():
        pattern = re.compile(rf"\b{re.escape(source)}\b", re.IGNORECASE)
        text = pattern.sub(lambda m, repl=target: _preserve_case_replacement(m.group(0), repl), text)
    return text


def normalize_common_joined_words(text: str) -> str:
    for source, target in COMMON_JOINED_WORD_NORMALIZATIONS.items():
        pattern = re.compile(rf"\b{re.escape(source)}\b", re.IGNORECASE)
        text = pattern.sub(lambda m, repl=target: _preserve_case_replacement(m.group(0), repl), text)
    return text


def _ensure_pymorphy2_compat() -> None:
    if hasattr(inspect, "getargspec"):
        return
    from collections import namedtuple

    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def getargspec(func):  # type: ignore[override]
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = getargspec  # type: ignore[attr-defined]


@lru_cache(maxsize=1)
def _get_morph_analyzer():
    try:
        _ensure_pymorphy2_compat()
        import pymorphy2

        return pymorphy2.MorphAnalyzer()
    except Exception:
        return None


@lru_cache(maxsize=50000)
def _parse_score(word: str) -> float:
    if not CYRILLIC_WORD_RE.fullmatch(word):
        return 0.0
    morph = _get_morph_analyzer()
    if morph is None:
        return 0.0
    try:
        return float(morph.parse(word)[0].score or 0.0)
    except Exception:
        return 0.0


def _should_merge_long_stem_space_split(left: str, right: str) -> bool:
    lowered_right = right.lower()
    if lowered_right in COMMON_SHORT_STANDALONE_WORDS:
        return False
    if len(lowered_right) == 2 and lowered_right not in {"ся", "сь"}:
        return False

    combined = f"{left}{right}"
    combined_score = _parse_score(combined)
    if combined_score >= 0.65:
        if lowered_right in {"ся", "сь"}:
            return True
        left_score = _parse_score(left)
        right_score = _parse_score(right)
        return left_score <= 0.4 or right_score <= 0.4

    return lowered_right in SPACE_SPLIT_SUFFIX_FALLBACKS


def _should_merge_prefix_space_split(left: str, right: str) -> bool:
    lowered_left = left.lower()
    lowered_right = right.lower()
    roots = SPACE_SPLIT_PREFIX_ROOTS.get(lowered_left)
    if not roots:
        return False
    if not any(lowered_right.startswith(root) for root in roots):
        return False

    combined = f"{left}{right}"
    combined_score = _parse_score(combined)
    if combined_score < 0.55:
        return False
    return True


def _should_merge_midword_space_split(left: str, right: str) -> bool:
    lowered_left = left.lower()
    lowered_right = right.lower()
    if lowered_left in COMMON_SHORT_STANDALONE_WORDS or lowered_right in COMMON_SHORT_STANDALONE_WORDS:
        return False

    combined = f"{left}{right}"
    combined_score = _parse_score(combined)
    if combined_score < 0.7:
        return False

    left_score = _parse_score(left)
    right_score = _parse_score(right)
    return left_score <= 0.4 or right_score <= 0.4


def _long_stem_space_split_replacer(match: re.Match[str]) -> str:
    left = match.group(1)
    right = match.group(2)
    if _should_merge_long_stem_space_split(left, right):
        return f"{left}{right}"
    return match.group(0)


def _prefix_space_split_replacer(match: re.Match[str]) -> str:
    left = match.group(1)
    right = match.group(2)
    if _should_merge_prefix_space_split(left, right):
        return f"{left}{right}"
    return match.group(0)


def _midword_space_split_replacer(match: re.Match[str]) -> str:
    left = match.group(1)
    right = match.group(2)
    if _should_merge_midword_space_split(left, right):
        return f"{left}{right}"
    return match.group(0)


def fix_intraword_space_splits(text: str) -> str:
    text = normalize_common_space_splits(text)
    text = SPACE_SPLIT_LONG_STEM_RE.sub(_long_stem_space_split_replacer, text)
    text = SPACE_SPLIT_PREFIX_RE.sub(_prefix_space_split_replacer, text)
    text = SPACE_SPLIT_MIDWORD_RE.sub(_midword_space_split_replacer, text)
    return text


def strip_mojibake_heading_fragments(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or not MOJIBAKE_CHAR_RE.search(line):
            cleaned_lines.append(raw_line)
            continue
        line = MOJIBAKE_LEADING_RE.sub("", line)
        line = MOJIBAKE_NUMBERED_RE.sub(r"\1", line)
        line = MOJIBAKE_NUMBERED_EOL_RE.sub(r"\1", line)
        line = SPACE_RUN_RE.sub(" ", line).strip()
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def strip_running_headers_and_noise(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue
        line = HEADER_NOISE_FRAGMENT_RE.sub(" ", line)
        line = SPACE_RUN_RE.sub(" ", line).strip()
        if not line:
            continue
        if is_running_header_line(line):
            continue
        if NOISE_ONLY_LINE_RE.fullmatch(line):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def normalize_header_candidate(line: str) -> str:
    normalized = line.upper()
    normalized = re.sub(r"[`'\"“”„<>|=+_~*.,:;!?-]+", " ", normalized)
    normalized = SPACE_RUN_RE.sub(" ", normalized).strip()
    normalized = re.sub(r"\bCUA\b", "СИЛ", normalized)
    normalized = re.sub(r"\bPOCCUH\b", "РОССИИ", normalized)
    normalized = re.sub(r"\bРОССИ\b", "РОССИИ", normalized)
    return normalized


def is_running_header_line(line: str) -> bool:
    if RUNNING_HEADER_LINE_RE.fullmatch(line):
        return True
    normalized = normalize_header_candidate(line)
    if normalized == "БЕЛОЕ ДВИЖЕНИЕ":
        return True
    return bool(
        re.fullmatch(
            r"(?:ПОСЛЕДНИЕ|ЛЕДНИЕ|НИЕ)\s+БОИ\s+ВООРУЖЕННЫХ\s+СИЛ\s+ЮГА\s+РОССИИ",
            normalized,
        )
    )


def trim_trailing_backmatter(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    min_index = max(1, len(lines) // 3) if len(lines) <= 40 else max(1, int(len(lines) * 0.6))
    cutoff: int | None = None
    for index, raw_line in enumerate(lines):
        if index < min_index:
            continue
        line = raw_line.strip()
        if not line:
            continue
        if PROMPT_LEAK_START_RE.search(line):
            cutoff = index
            break
        if TRAILING_BACKMATTER_START_RE.match(line) and is_probable_trailing_backmatter(lines, index):
            cutoff = index
            break
    if cutoff is None:
        return text
    return "\n".join(lines[:cutoff]).rstrip()


def is_probable_trailing_backmatter(lines: list[str], start_index: int) -> bool:
    total_lines = len(lines)
    if total_lines <= 1:
        return True

    if total_lines <= 40:
        return start_index >= max(1, total_lines // 3)

    # Very near the end, allow a direct cut.
    if start_index >= int(total_lines * 0.9):
        return True

    tail_lines = [line.strip() for line in lines[start_index:] if line.strip()]
    if not tail_lines:
        return False

    sample = tail_lines[: min(40, len(tail_lines))]
    supportive = 0
    for line in sample:
        if TRAILING_BACKMATTER_START_RE.match(line):
            supportive += 1
            continue
        if REFERENCE_BLOCK_START_RE.match(line):
            supportive += 1
            continue
        if looks_reference_like(line):
            supportive += 1
            continue
        if len(line) <= 100 and not PARAGRAPH_TERMINAL_RE.search(line):
            supportive += 1
            continue

    ratio = supportive / len(sample)
    return start_index >= int(total_lines * 0.75) and ratio >= 0.7


def trim_trailing_reference_block(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    min_index = max(1, int(len(lines) * 0.75))
    cutoff: int | None = None
    for index, raw_line in enumerate(lines):
        if index < min_index:
            continue
        line = raw_line.strip()
        if not line:
            continue
        if not REFERENCE_BLOCK_START_RE.match(line):
            continue
        if is_probable_trailing_backmatter(lines, index):
            cutoff = index
            break

    if cutoff is None:
        return text
    return "\n".join(lines[:cutoff]).rstrip()


def normalize_ocr_bullet_markers(text: str) -> str:
    text = OCR_BULLET_INLINE_RE.sub(lambda m: ("\n- " if m.start() > 0 else "- "), text)
    normalized_lines: list[str] = []
    for raw_line in text.splitlines():
        normalized_lines.append(OCR_BULLET_LINE_RE.sub("- ", raw_line, count=1))
    return "\n".join(normalized_lines)


def is_heading_like_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if HEADING_RE.search(stripped):
        return True
    if NUMBERED_SECTION_RE.match(stripped) and len(stripped) <= 140:
        return True
    if len(stripped) <= 90 and ALL_CAPS_TITLE_RE.fullmatch(stripped):
        upper_hits = len(re.findall(r"[А-ЯЁA-Z]", stripped))
        return upper_hits >= 4
    if len(stripped) <= 220 and not re.search(r"[А-ЯЁA-Z][а-яёa-z]", stripped):
        heading_tokens = [token for token in stripped.split() if is_heading_token(token)]
        if len(heading_tokens) >= 3:
            return True
    return False


def is_block_boundary_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return bool(
        is_heading_like_line(stripped)
        or DIALOGUE_LINE_RE.match(stripped)
        or LIST_LINE_RE.match(stripped)
    )


def should_start_new_paragraph(previous: str, current: str) -> bool:
    previous = previous.strip()
    current = current.strip()
    if not previous or not current:
        return True
    if is_block_boundary_line(previous) or is_block_boundary_line(current):
        return True
    if current[:1].islower():
        return False
    if previous.endswith((",", ";", ":", "—", "-", "(")):
        return False
    if PARAGRAPH_TERMINAL_RE.search(previous):
        return True
    return False


def is_sentence_continuation_start(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return bool(LOWERCASE_CONTINUATION_RE.match(stripped))


def is_heading_token(token: str) -> bool:
    stripped = token.strip("«»\"'“”„()[]{}:;,.!?—–-")
    if not stripped:
        return False
    if re.fullmatch(r"[IVXLCDM0-9]+", stripped):
        return True
    letters_only = re.sub(r"[^A-Za-zА-Яа-яЁё]", "", stripped)
    if not letters_only:
        return False
    return letters_only.upper() == letters_only


def canonicalize_section_heading(heading: str) -> str:
    heading = heading.strip()
    heading = re.sub(r"^(Глава|Часть|ЧАСТЬ)\s+([IVXLCDM0-9]+)\.?\s*", r"\1 \2 ", heading)
    return SPACE_RUN_RE.sub(" ", heading).strip()


def split_leading_section_heading(line: str) -> tuple[str, str, str] | None:
    stripped = line.strip()
    match = LEADING_SECTION_PREFIX_RE.match(stripped)
    if not match:
        return None

    label = match.group("label").strip()
    body = match.group("body").strip()
    if not body:
        return None

    tokens = body.split()
    heading_tokens: list[str] = []
    for token in tokens:
        if not is_heading_token(token):
            break
        heading_tokens.append(token)

    if not heading_tokens:
        return None

    rest_tokens = tokens[len(heading_tokens) :]
    heading = f"{label} {' '.join(heading_tokens)}".strip()
    rest = " ".join(rest_tokens).strip()
    return canonicalize_section_heading(heading), heading, rest


def strip_intrusive_standalone_headings(lines: list[str]) -> list[str]:
    if len(lines) < 3:
        return lines
    cleaned: list[str] = []
    total = len(lines)
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line or index == 0 or index == total - 1:
            cleaned.append(raw_line)
            continue

        previous = lines[index - 1].strip()
        next_line = lines[index + 1].strip()
        if not previous or not next_line:
            cleaned.append(raw_line)
            continue

        if not is_heading_like_line(line):
            cleaned.append(raw_line)
            continue

        if is_block_boundary_line(previous) or PARAGRAPH_TERMINAL_RE.search(previous):
            cleaned.append(raw_line)
            continue

        if not is_sentence_continuation_start(next_line):
            cleaned.append(raw_line)
            continue

        # This heading-like line is sandwiched between two fragments of the same sentence,
        # which is characteristic of a repeated running header leaking into the text body.
        continue
    return cleaned


def strip_intrusive_inline_heading_prefixes(lines: list[str]) -> list[str]:
    if len(lines) < 2:
        return lines
    cleaned: list[str] = []
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if index == 0 or not line:
            cleaned.append(raw_line)
            continue

        previous = lines[index - 1].strip()
        if not previous or is_block_boundary_line(previous) or PARAGRAPH_TERMINAL_RE.search(previous):
            cleaned.append(raw_line)
            continue

        match = INLINE_HEADING_PREFIX_RE.match(line)
        if not match:
            cleaned.append(raw_line)
            continue

        cleaned.append(match.group("rest"))
    return cleaned


def strip_repeated_heading_prefixes(lines: list[str]) -> list[str]:
    if not lines:
        return lines

    seen_prefixes: list[str] = []
    cleaned: list[str] = []
    for raw_line in lines:
        line = SPACE_RUN_RE.sub(" ", raw_line.strip()).strip()
        if not line:
            cleaned.append(raw_line)
            continue

        stripped = line
        if FIGURE_LABEL_RE.search(stripped):
            direct_match = SECTION_HEADING_FIGURE_PREFIX_RE.match(stripped)
            if direct_match:
                stripped = stripped[direct_match.end() :].strip()
            direct_author_match = AUTHOR_TITLE_FIGURE_PREFIX_RE.match(stripped)
            if direct_author_match:
                stripped = stripped[direct_author_match.end() :].strip()

        for prefix in sorted(set(seen_prefixes), key=len, reverse=True):
            if stripped.startswith(prefix + " "):
                remainder = stripped[len(prefix) :].strip()
                if remainder and not is_heading_like_line(remainder):
                    stripped = remainder
                    break

        if FIGURE_LABEL_RE.search(stripped):
            for prefix in sorted(set(seen_prefixes), key=len, reverse=True):
                if stripped.startswith(prefix + " "):
                    stripped = stripped[len(prefix) :].strip()
                    break
            cleaned.append(stripped)
            continue

        if len(stripped) <= 180 and (
            is_heading_like_line(stripped)
            or re.match(r"^[А-ЯЁA-Z][^.!?…]{15,140}$", stripped)
            or AUTHOR_TITLE_PREFIX_RE.match(stripped)
        ):
            seen_prefixes.append(stripped)
            cleaned.append(raw_line)
            continue

        cleaned.append(stripped)

    return cleaned


def normalize_inline_section_headings(lines: list[str]) -> list[str]:
    if not lines:
        return lines

    cleaned: list[str] = []
    seen_headings: dict[str, int] = {}
    seen_heading_texts: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            cleaned.append(raw_line)
            continue

        repeated_prefix_handled = False
        for seen_heading in sorted(seen_heading_texts, key=len, reverse=True):
            if line == seen_heading:
                repeated_prefix_handled = True
                break
            if line.startswith(f"{seen_heading} "):
                remainder = line[len(seen_heading) :].strip()
                if remainder:
                    cleaned.append(remainder)
                repeated_prefix_handled = True
                break
        if repeated_prefix_handled:
            continue

        parsed = split_leading_section_heading(line)
        if not parsed:
            cleaned.append(raw_line)
            continue

        heading_key, heading, rest = parsed
        seen_count = seen_headings.get(heading_key, 0)
        seen_headings[heading_key] = seen_count + 1

        if seen_count == 0:
            seen_heading_texts.append(heading)
            cleaned.append(heading)
            if rest:
                cleaned.append(rest)
            continue

        if rest:
            cleaned.append(rest)

    return cleaned


def split_leading_allcaps_headings(lines: list[str]) -> list[str]:
    if not lines:
        return lines

    cleaned: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            cleaned.append(raw_line)
            continue

        if line.startswith(("Глава ", "ГЛАВА ", "Часть ", "ЧАСТЬ ", "Chapter ", "CHAPTER ", "Part ", "PART ")):
            cleaned.append(raw_line)
            continue

        tokens = line.split()
        heading_tokens: list[str] = []
        for token in tokens:
            if is_heading_token(token):
                heading_tokens.append(token)
            else:
                break

        rest_tokens = tokens[len(heading_tokens) :]
        if len(heading_tokens) < 3 or not rest_tokens:
            cleaned.append(raw_line)
            continue

        if not re.match(r"^[А-ЯЁA-Z][а-яёa-z]", rest_tokens[0]):
            cleaned.append(raw_line)
            continue

        heading = SPACE_RUN_RE.sub(" ", " ".join(heading_tokens)).strip()
        if len(heading) < 18:
            cleaned.append(raw_line)
            continue

        cleaned.append(heading)
        cleaned.append(" ".join(rest_tokens).strip())

    return cleaned


def _strip_leading_figure_caption_prefix(line: str) -> str | None:
    if not FIGURE_LABEL_RE.match(line):
        return None

    for split_match in SENTENCE_SPLIT_RE.finditer(line):
        prefix = line[: split_match.start() + 1].strip()
        rest = line[split_match.end() :].strip()
        if not rest:
            continue
        if not re.match(r"^(?:Рис|Илл|Табл)\.\s*\d+", prefix, re.IGNORECASE):
            continue
        if len(prefix.split()) < 4:
            continue
        if _looks_like_body_continuation(rest, prefix):
            return rest

    return None


def _looks_like_body_continuation(text: str, prefix: str = "") -> bool:
    candidate = SPACE_RUN_RE.sub(" ", text).strip()
    if len(candidate) < 25:
        return False
    if len(candidate.split()) < 4:
        return False
    if not BODY_LIKE_CONTINUATION_START_RE.match(candidate):
        return False
    if LOWERCASE_CONTINUATION_RE.match(candidate):
        return True
    if BODY_LIKE_CONTINUATION_CUE_RE.search(candidate):
        return True
    if prefix and CAPTION_SOURCE_CUE_RE.search(prefix) and len(candidate.split()) >= 6:
        return True
    if len(candidate) >= 90 and len(candidate.split()) >= 12:
        return True
    if PARAGRAPH_TERMINAL_RE.search(candidate):
        return True
    return candidate.count(". ") >= 1


def _strip_inline_figure_caption(line: str, intrusive_match: re.Match[str]) -> str | None:
    prefix = line[: intrusive_match.start()].strip()
    tail = line[intrusive_match.start() :].strip()
    if not prefix:
        return None

    for split_match in SENTENCE_SPLIT_RE.finditer(tail):
        caption_prefix = tail[: split_match.start() + 1].strip()
        rest = tail[split_match.end() :].strip()
        if not rest:
            continue
        if not re.match(r"^(?:Рис|Илл|Табл)\.\s*\d+", caption_prefix, re.IGNORECASE):
            continue
        if len(caption_prefix.split()) < 4:
            continue
        if _looks_like_body_continuation(rest, caption_prefix):
            return SPACE_RUN_RE.sub(" ", f"{prefix} {rest}").strip()

    continuations = list(INLINE_FIGURE_CONTINUATION_RE.finditer(tail))
    if continuations:
        suffix = tail[continuations[-1].end() :].strip()
        if len(suffix) >= 12:
            return SPACE_RUN_RE.sub(" ", f"{prefix} {suffix}").strip()
        return prefix

    return prefix


def strip_figure_caption_artifacts(lines: list[str]) -> list[str]:
    if not lines:
        return lines

    cleaned: list[str] = []
    for raw_line in lines:
        line = SPACE_RUN_RE.sub(" ", raw_line.strip()).strip()
        if not line:
            cleaned.append(raw_line)
            continue

        line = INLINE_BIBLIOGRAPHY_HEADING_RE.sub("", line).strip()
        if not line:
            continue

        intrusive_match = None
        for candidate in FIGURE_LABEL_RE.finditer(line):
            if candidate.start() > 0 and line[candidate.start() - 1] in {"(", "["}:
                continue
            intrusive_match = candidate
            break

        if intrusive_match is None:
            cleaned.append(line)
            continue

        while True:
            stripped_prefix = _strip_leading_figure_caption_prefix(line)
            if not stripped_prefix or stripped_prefix == line:
                break
            line = stripped_prefix.strip()
        if not line:
            continue

        intrusive_match = None
        for candidate in FIGURE_LABEL_RE.finditer(line):
            if candidate.start() > 0 and line[candidate.start() - 1] in {"(", "["}:
                continue
            intrusive_match = candidate
            break

        if intrusive_match and not line.startswith(("Рис.", "Илл.", "Табл.")):
            stripped_inline = _strip_inline_figure_caption(line, intrusive_match)
            if stripped_inline:
                line = stripped_inline

        trailing_match = TRAILING_FIGURE_CAPTION_SUFFIX_RE.match(line)
        if trailing_match:
            cleaned.append(trailing_match.group("body").strip())
            continue

        if line.startswith(("Рис.", "Илл.", "Табл.")):
            stripped_prefix = _strip_leading_figure_caption_prefix(line)
            if stripped_prefix:
                cleaned.append(stripped_prefix)
                continue

        if line.startswith(("Рис.", "Илл.", "Табл.")) and line.count("Рис.") + line.count("Илл.") + line.count("Табл.") >= 2:
            continue

        if STANDALONE_FIGURE_CAPTION_RE.fullmatch(line):
            continue

        cleaned.append(line)

    return cleaned


def restore_paragraph_lines(lines: list[str]) -> list[str]:
    restored: list[str] = []
    current_parts: list[str] = []

    def flush() -> None:
        if not current_parts:
            return
        paragraph = SPACE_RUN_RE.sub(" ", " ".join(part.strip() for part in current_parts if part.strip())).strip()
        if paragraph:
            restored.append(paragraph)
        current_parts.clear()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if is_block_boundary_line(line):
            flush()
            restored.append(line)
            continue
        if current_parts and should_start_new_paragraph(current_parts[-1], line):
            flush()
        current_parts.append(line)

    flush()
    return restored


def split_long_line(line: str) -> list[str]:
    line = SPACE_RUN_RE.sub(" ", line).strip()
    if not line:
        return []
    # Keep original line boundaries by default; do not split by punctuation.
    return [line]


def looks_reference_like(sentence: str) -> bool:
    if NOTE_LINE_RE.search(sentence) and REFERENCE_CUE_RE.search(sentence):
        return True
    if REFERENCE_CUE_RE.search(sentence):
        year_hits = len(re.findall(r"\b(?:19|20)\d{2}\b", sentence))
        if year_hits >= 2:
            return True
        if len(sentence) < 260:
            return True
    if sentence.count(" // ") >= 2:
        return True
    return False


def clean_sentence(sentence: str) -> str:
    sentence = sentence.strip()
    if not sentence:
        return ""
    has_bullet_prefix = sentence.startswith("- ")
    sentence = URL_RE.sub("", sentence)
    sentence = PLACEHOLDER_RE.sub("", sentence)
    sentence = ANGLE_WRAPPED_TEXT_RE.sub(r"\1", sentence)
    sentence = ARROW_ARTIFACT_RE.sub("—", sentence)
    sentence = FOOTNOTE_SUFFIX_RE.sub("", sentence)
    sentence = SLASH_LINK_RE.sub("", sentence)
    sentence = re.sub(r"\s+\(\s*\)", "", sentence)
    sentence = re.sub(r"\s+[.,;:!?…]", lambda m: m.group(0).strip(), sentence)
    sentence = SPACE_RUN_RE.sub(" ", sentence).strip()
    sentence = sentence.strip(";:,")
    if has_bullet_prefix:
        sentence = sentence.lstrip("- ").strip()
        return f"- {sentence}".strip()
    return sentence.strip()


def clean_document(text: str) -> tuple[str, dict[str, int]]:
    text = normalize_text(text)
    text = strip_page_number_heading_artifacts(text)
    text = strip_running_headers_and_noise(text)
    text = trim_trailing_backmatter(text)
    text = trim_trailing_reference_block(text)

    removed_sentences = 0
    removed_promos = 0
    removed_note_like = 0
    removed_reference_like = 0

    output_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if output_lines and output_lines[-1] != "":
                output_lines.append("")
            continue

        # Trim obvious table/source tail while preserving table body text.
        if "Источник:" in line:
            line = TRAILING_SOURCE_RE.sub("", line).strip()
            if not line:
                removed_reference_like += 1
                removed_sentences += 1
                continue

        segments = split_long_line(line)
        for seg in segments:
            if not seg:
                continue
            if PROMO_LINE_RE.search(seg):
                removed_promos += 1
                removed_sentences += 1
                continue
            if looks_reference_like(seg):
                if NOTE_LINE_RE.search(seg):
                    removed_note_like += 1
                removed_reference_like += 1
                removed_sentences += 1
                continue

            seg = clean_sentence(seg)
            if not seg:
                removed_sentences += 1
                continue
            if len(seg) <= 2:
                removed_sentences += 1
                continue

            if HEADING_RE.search(seg):
                output_lines.append(seg)
                continue

            output_lines.append(seg)

    output_lines = strip_intrusive_standalone_headings(output_lines)
    output_lines = strip_intrusive_inline_heading_prefixes(output_lines)
    output_lines = strip_repeated_heading_prefixes(output_lines)
    output_lines = normalize_inline_section_headings(output_lines)
    output_lines = split_leading_allcaps_headings(output_lines)
    output_lines = strip_figure_caption_artifacts(output_lines)
    restored_lines = restore_paragraph_lines(output_lines)
    joined = "\n".join(restored_lines).strip()
    joined = fix_line_wrap_hyphenation(joined)
    joined = normalize_common_compounds(joined)
    joined = fix_intraword_space_splits(joined)
    joined = normalize_common_joined_words(joined)
    joined = re.sub(r"\n{2,}", "\n", joined)
    return joined, {
        "removed_sentences": removed_sentences,
        "removed_promos": removed_promos,
        "removed_note_like": removed_note_like,
        "removed_reference_like": removed_reference_like,
    }


def process_file(path: Path, backup_dir: Path) -> FileReport:
    original = path.read_text(encoding="utf-8", errors="replace")
    backup_path = backup_dir / path.name
    shutil.copy2(path, backup_path)

    cleaned, stats = clean_document(original)
    path.write_text(cleaned + "\n", encoding="utf-8")

    return FileReport(
        file_name=path.name,
        before_chars=len(original),
        after_chars=len(cleaned),
        before_lines=original.count("\n") + (1 if original else 0),
        after_lines=cleaned.count("\n") + (1 if cleaned else 0),
        removed_sentences=stats["removed_sentences"],
        removed_promos=stats["removed_promos"],
        removed_note_like=stats["removed_note_like"],
        removed_reference_like=stats["removed_reference_like"],
        url_count_before=len(URL_RE.findall(original)),
        url_count_after=len(URL_RE.findall(cleaned)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-clean final TXT outputs.")
    parser.add_argument(
        "--txt-dir",
        type=Path,
        default=Path("outputs/final_txt"),
        help="Directory containing final TXT files.",
    )
    parser.add_argument(
        "--glob",
        default="*.txt",
        help="Glob pattern under --txt-dir.",
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=Path("outputs/final_txt_backups"),
        help="Root folder to store backups before overwrite.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    txt_dir = args.txt_dir
    if not txt_dir.exists():
        raise SystemExit(f"TXT directory not found: {txt_dir}")

    files = sorted(txt_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched {args.glob} in {txt_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = args.backup_root / f"postclean_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    reports: list[FileReport] = []
    for file_path in files:
        reports.append(process_file(file_path, backup_dir))

    report_data = {
        "generated_at": timestamp,
        "txt_dir": str(txt_dir.resolve()),
        "backup_dir": str(backup_dir.resolve()),
        "files": [item.to_dict() for item in reports],
    }
    report_path = txt_dir / f"post_clean_report_{timestamp}.json"
    report_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

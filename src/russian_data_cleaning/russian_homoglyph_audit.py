from __future__ import annotations

import re
from typing import Any


MIXED_TOKEN_RE = re.compile(r"\b[\w-]{4,}\b", re.UNICODE)
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
LATIN_RE = re.compile(r"[A-Za-z]")
ASCII_ONLY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

LATIN_TO_CYRILLIC = str.maketrans(
    {
        "A": "А",
        "a": "а",
        "B": "В",
        "C": "С",
        "c": "с",
        "E": "Е",
        "e": "е",
        "H": "Н",
        "K": "К",
        "k": "к",
        "M": "М",
        "m": "м",
        "O": "О",
        "o": "о",
        "P": "Р",
        "p": "р",
        "T": "Т",
        "X": "Х",
        "x": "х",
        "Y": "У",
        "y": "у",
    }
)

SAFE_AUTO_LATIN = set("AaBCcEeHKkMmOoPpTXxYy")
SUSPICIOUS_LATIN = set("bnht")


def _is_safe_mixed_token(token: str) -> bool:
    latin_chars = {char for char in token if "A" <= char <= "Z" or "a" <= char <= "z"}
    if not latin_chars:
        return False
    if not latin_chars <= SAFE_AUTO_LATIN:
        return False
    if len(token) <= 4 and token.isupper():
        return False
    if token.startswith(("http", "www")) or "@" in token:
        return False
    return True


def audit_russian_homoglyphs(text: str) -> dict[str, Any]:
    if not text:
        return {"text": text, "detected": 0, "auto_fixed": 0, "warned": 0, "samples": []}

    samples: list[dict[str, Any]] = []
    detected = 0
    auto_fixed = 0
    warned = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal detected, auto_fixed, warned
        token = match.group(0)
        if not CYRILLIC_RE.search(token):
            return token
        if not LATIN_RE.search(token):
            return token
        if ASCII_ONLY_RE.fullmatch(token) and token.isupper() and len(token) <= 6:
            return token

        detected += 1
        latin_chars = {char for char in token if char.isascii() and char.isalpha()}
        if _is_safe_mixed_token(token):
            fixed = token.translate(LATIN_TO_CYRILLIC)
            if fixed != token:
                auto_fixed += 1
                samples.append({"kind": "auto_fix", "before": token, "after": fixed})
                return fixed

        warned += 1
        warning_kind = "warn"
        if latin_chars & SUSPICIOUS_LATIN:
            warning_kind = "warn_suspicious"
        samples.append({"kind": warning_kind, "before": token, "after": token})
        return token

    fixed_text = MIXED_TOKEN_RE.sub(replace, text)
    return {
        "text": fixed_text,
        "detected": detected,
        "auto_fixed": auto_fixed,
        "warned": warned,
        "samples": samples[:12],
    }

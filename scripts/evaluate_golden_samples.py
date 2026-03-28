#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES_DIR = ROOT / "golden_samples" / "samples"

MIXED_SCRIPT_TOKEN_RE = re.compile(r"\b(?=\w*[A-Za-z])(?=\w*[А-Яа-яЁё])[\w-]{4,}\b")
SPACED_HYPHEN_RE = re.compile(r"\b[А-Яа-яЁёA-Za-z]+-\s+[А-Яа-яЁёA-Za-z]+\b")
COMPACT_HYPHEN_SHORT_RE = re.compile(r"\b[А-Яа-яЁёA-Za-z]{2,}-[А-Яа-яЁёA-Za-z]{1,4}\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a minimal golden sample set for document cleaning.")
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument("--out", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        curr = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def normalized_edit_distance(expected: str, actual: str) -> float | None:
    if not expected and not actual:
        return 0.0
    if not expected:
        return None
    return round(levenshtein_distance(expected, actual) / max(len(expected), 1), 4)


def keyword_retention_rate(keywords: list[str], actual: str) -> float | None:
    if not keywords:
        return None
    kept = sum(1 for keyword in keywords if keyword and keyword.lower() in actual.lower())
    return round(kept / max(len(keywords), 1), 4)


def structure_overreach_rate(protected_spans: list[str], actual: str) -> float | None:
    if not protected_spans:
        return None
    kept = sum(1 for span in protected_spans if span and span in actual)
    return round(1.0 - (kept / max(len(protected_spans), 1)), 4)


def russian_anomaly_counts(text: str) -> dict[str, int]:
    return {
        "mixed_script_tokens": len(MIXED_SCRIPT_TOKEN_RE.findall(text)),
        "spaced_hyphen": len(SPACED_HYPHEN_RE.findall(text)),
        "compact_hyphen_short": len(COMPACT_HYPHEN_SHORT_RE.findall(text)),
    }


def page_type_misclassification(metadata: dict[str, Any]) -> float | None:
    expected = metadata.get("expected_page_type")
    predicted = metadata.get("predicted_page_type")
    if not expected or predicted is None:
        return None
    return 0.0 if expected == predicted else 1.0


def load_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def evaluate_sample(sample_dir: Path) -> dict[str, Any]:
    metadata_path = sample_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    actual = load_text(sample_dir / "actual.txt")
    expected = load_text(sample_dir / "expected.txt")
    raw = load_text(sample_dir / "raw.txt")
    return {
        "sample_id": sample_dir.name,
        "page_type": metadata.get("expected_page_type"),
        "normalized_edit_distance": normalized_edit_distance(expected, actual),
        "keyword_retention_rate": keyword_retention_rate(list(metadata.get("keywords", [])), actual),
        "structure_overreach_rate": structure_overreach_rate(list(metadata.get("protected_spans", [])), actual),
        "russian_anomalies": russian_anomaly_counts(actual),
        "page_type_misclassification": page_type_misclassification(metadata),
        "raw_length": len(raw),
        "expected_length": len(expected),
        "actual_length": len(actual),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sample_count": len(results),
        "samples": results,
    }
    numeric_fields = [
        "normalized_edit_distance",
        "keyword_retention_rate",
        "structure_overreach_rate",
        "page_type_misclassification",
    ]
    for field in numeric_fields:
        values = [result[field] for result in results if result.get(field) is not None]
        if values:
            summary[f"avg_{field}"] = round(sum(values) / len(values), 4)
    anomaly_totals = {"mixed_script_tokens": 0, "spaced_hyphen": 0, "compact_hyphen_short": 0}
    for result in results:
        for key, value in result.get("russian_anomalies", {}).items():
            anomaly_totals[key] = anomaly_totals.get(key, 0) + int(value)
    summary["russian_anomaly_totals"] = anomaly_totals
    return summary


def main() -> None:
    args = parse_args()
    sample_dirs = sorted(path for path in args.samples_dir.iterdir() if path.is_dir()) if args.samples_dir.exists() else []
    results = [evaluate_sample(sample_dir) for sample_dir in sample_dirs]
    payload = summarize(results)
    output = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

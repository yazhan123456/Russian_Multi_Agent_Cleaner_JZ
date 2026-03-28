#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize minimal monitoring counters from page_state checkpoints.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing one or more book subdirectories with page_states/*.json",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser.parse_args()


def load_page_states(run_dir: Path) -> list[dict[str, Any]]:
    page_states: list[dict[str, Any]] = []
    for page_path in sorted(run_dir.glob("*/page_states/*.json")):
        page_states.append(json.loads(page_path.read_text(encoding="utf-8")))
    return page_states


def build_monitoring_summary(page_states: list[dict[str, Any]]) -> dict[str, Any]:
    state_counts: Counter[str] = Counter()
    failed_by_agent: Counter[str] = Counter()
    review_risk_counts: Counter[str] = Counter()
    repair_status_counts: Counter[str] = Counter()
    structure_status_counts: Counter[str] = Counter()
    final_text_source_counts: Counter[str] = Counter()

    for page_state in page_states:
        state_counts.update([str(page_state.get("current_state") or "UNKNOWN")])

        risk_level = page_state.get("risk_level")
        if risk_level:
            review_risk_counts.update([str(risk_level)])

        repaired_payload = page_state.get("stage_payloads", {}).get("repaired")
        if isinstance(repaired_payload, dict):
            repair_status = repaired_payload.get("repair_status")
            if repair_status:
                repair_status_counts.update([str(repair_status)])

        structure_plan = page_state.get("structure_plan")
        if isinstance(structure_plan, dict):
            structure_status = structure_plan.get("status")
            final_text_source = structure_plan.get("final_text_source")
            if structure_status:
                structure_status_counts.update([str(structure_status)])
            if final_text_source:
                final_text_source_counts.update([str(final_text_source)])

        for event in page_state.get("processing_history", []):
            if str(event.get("to_state") or "") == "FAILED":
                failed_by_agent.update([str(event.get("agent") or "unknown")])

    return {
        "state_counts": dict(state_counts),
        "failed_by_agent": dict(failed_by_agent),
        "review_risk_counts": dict(review_risk_counts),
        "repair_status_counts": dict(repair_status_counts),
        "structure_status_counts": dict(structure_status_counts),
        "final_text_source_counts": dict(final_text_source_counts),
    }


def main() -> None:
    args = parse_args()
    page_states = load_page_states(args.run_dir)
    summary = build_monitoring_summary(page_states)
    indent = 2 if args.pretty else None
    print(json.dumps(summary, ensure_ascii=False, indent=indent))


if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "monitor_run.py"
SPEC = importlib.util.spec_from_file_location("monitor_run_under_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
monitor_run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor_run)


class MonitorRunTests(unittest.TestCase):
    def test_build_monitoring_summary_counts_expected_fields(self) -> None:
        page_states = [
            {
                "current_state": "EXPORTED",
                "risk_level": "medium",
                "stage_payloads": {"repaired": {"repair_status": "deepseek_structured"}},
                "structure_plan": {
                    "status": "gemini",
                    "final_text_source": "structure_restore_generated",
                },
                "processing_history": [
                    {"agent": "RepairStage", "to_state": "FAILED"},
                    {"agent": "RepairStage", "to_state": "REPAIRED"},
                ],
            },
            {
                "current_state": "FAILED",
                "risk_level": "high",
                "stage_payloads": {"repaired": {"repair_status": "fallback"}},
                "structure_plan": {
                    "status": "skipped_safe_page",
                    "final_text_source": "repaired_passthrough",
                },
                "processing_history": [
                    {"agent": "StructureStage", "to_state": "FAILED"},
                ],
            },
        ]

        summary = monitor_run.build_monitoring_summary(page_states)

        self.assertEqual(summary["state_counts"], {"EXPORTED": 1, "FAILED": 1})
        self.assertEqual(summary["failed_by_agent"], {"RepairStage": 1, "StructureStage": 1})
        self.assertEqual(summary["review_risk_counts"], {"medium": 1, "high": 1})
        self.assertEqual(summary["repair_status_counts"], {"deepseek_structured": 1, "fallback": 1})
        self.assertEqual(summary["structure_status_counts"], {"gemini": 1, "skipped_safe_page": 1})
        self.assertEqual(
            summary["final_text_source_counts"],
            {"structure_restore_generated": 1, "repaired_passthrough": 1},
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "evaluate_golden_samples.py"
SPEC = importlib.util.spec_from_file_location("evaluate_golden_samples_under_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
golden_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(golden_eval)


class GoldenEvaluatorTests(unittest.TestCase):
    def test_keyword_retention_and_edit_distance(self) -> None:
        self.assertEqual(golden_eval.normalized_edit_distance("текст", "текст"), 0.0)
        self.assertEqual(golden_eval.keyword_retention_rate(["право", "участник"], "Право участника общества"), 1.0)

    def test_russian_anomaly_counts(self) -> None:
        anomalies = golden_eval.russian_anomaly_counts("Poccия построе- ния кто-то")

        self.assertEqual(anomalies["mixed_script_tokens"], 1)
        self.assertEqual(anomalies["spaced_hyphen"], 1)


if __name__ == "__main__":
    unittest.main()

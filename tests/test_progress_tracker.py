from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.process_books import ProgressTracker


class ProgressTrackerTests(unittest.TestCase):
    def test_last_completed_status_is_flat_and_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status_path = Path(tmpdir) / "progress.json"
            tracker = ProgressTracker([status_path], heartbeat_seconds=999)
            try:
                tracker.start_page("book.pdf", "Heuristic review", 1, 10, 1)
                tracker.finish_page()
                tracker.start_page("book.pdf", "Heuristic review", 2, 10, 2)
                tracker.finish_page()
                tracker.close()
            finally:
                tracker._stop_event.set()

            payload = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["state"], "idle")
            self.assertIn("last_completed", payload)
            self.assertIsInstance(payload["last_completed"], dict)
            self.assertNotIn("last_completed", payload["last_completed"])
            self.assertEqual(payload["last_completed"]["index"], 2)
            self.assertEqual(payload["last_completed"]["page_number"], 2)


if __name__ == "__main__":
    unittest.main()

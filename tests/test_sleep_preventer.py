from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.process_books import build_caffeinate_command


class SleepPreventerTests(unittest.TestCase):
    def test_build_caffeinate_command_uses_idle_sleep_prevention(self) -> None:
        self.assertEqual(build_caffeinate_command(12345), ["caffeinate", "-i", "-m", "-w", "12345"])


if __name__ == "__main__":
    unittest.main()

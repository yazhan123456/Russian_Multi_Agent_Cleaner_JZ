from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from russian_data_cleaning.russian_homoglyph_audit import audit_russian_homoglyphs


class RussianHomoglyphAuditTests(unittest.TestCase):
    def test_auto_fixes_safe_mixed_script_russian_token(self) -> None:
        result = audit_russian_homoglyphs("Poccия и Mосква")

        self.assertEqual(result["text"], "Россия и Москва")
        self.assertEqual(result["auto_fixed"], 2)
        self.assertEqual(result["warned"], 0)

    def test_warns_on_suspicious_mixed_token_without_forcing_fix(self) -> None:
        result = audit_russian_homoglyphs("праbо и закоh")

        self.assertEqual(result["text"], "праbо и закоh")
        self.assertEqual(result["detected"], 2)
        self.assertEqual(result["auto_fixed"], 0)
        self.assertEqual(result["warned"], 2)


if __name__ == "__main__":
    unittest.main()

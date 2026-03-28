from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from russian_data_cleaning.cleaning_agent import CleaningAgent


class CleaningAgentTests(unittest.TestCase):
    def test_body_with_notes_page_keeps_substantive_body_when_reference_tail_exists(self) -> None:
        agent = CleaningAgent()
        page = {
            "page_number": 512,
            "source": "extract",
            "selected_text": (
                "Глава VI. Создание и прекращение корпораций\n"
                "Учет особенностей участия государства в деятельности организации можно продемонстрировать на примере этой статьи.\n"
                "На примере этой же статьи можно продемонстрировать учет такого фактора, как сложность внутренней дифференциации организаций.\n"
                "Специфика участия может быть продемонстрирована положениями устава и практикой корпоративного управления.\n"
                "1. Федеральный закон ... // СЗ РФ. 2010. № 31. Ст. 4238.\n"
                "2. Постановление Правительства ... // Российская газета. 2012. 28 августа.\n"
            ),
            "body_text": (
                "Глава VI. Создание и прекращение корпораций\n"
                "Учет особенностей участия государства в деятельности организации можно продемонстрировать на примере этой статьи.\n"
                "На примере этой же статьи можно продемонстрировать учет такого фактора, как сложность внутренней дифференциации организаций.\n"
                "Специфика участия может быть продемонстрирована положениями устава и практикой корпоративного управления.\n"
                "1. Федеральный закон ... // СЗ РФ. 2010. № 31. Ст. 4238.\n"
                "2. Постановление Правительства ... // Российская газета. 2012. 28 августа.\n"
            ),
            "notes_text": (
                "1. Федеральный закон ... // СЗ РФ. 2010. № 31. Ст. 4238.\n"
                "2. Постановление Правительства ... // Российская газета. 2012. 28 августа.\n"
            ),
            "reference_text": "",
            "page_type": "body_with_notes",
        }

        cleaned = agent.clean_page(page, repeated_headers=set(), repeated_footers=set())

        cleaned_text = cleaned["cleaned_text"]
        self.assertIn("Учет особенностей участия государства", cleaned_text)
        self.assertIn("сложность внутренней дифференциации организаций", cleaned_text)
        self.assertNotEqual(cleaned_text.strip(), "Глава VI. Создание и прекращение корпораций")

    def test_reference_heavy_body_page_is_not_dropped_when_body_is_substantive(self) -> None:
        agent = CleaningAgent()
        page = {
            "page_number": 862,
            "source": "extract",
            "selected_text": (
                "Глава IX. Права и обязанности участников (членов) корпорации\n"
                "Континентальной Европы данные права традиционно разграничиваются.\n"
                "В своей основе такое разделение имеет различную трактовку понятий управления и контроля.\n"
                "Изначально в правовой системе контроль ассоциировался с проверкой деятельности субъектов.\n"
                "1. Berle A., Means G. The Modern Corporation and Private Property. N.Y., 1934.\n"
                "2. Winter R. Government and Corporation. Washington, 1978.\n"
            ),
            "body_text": (
                "Глава IX. Права и обязанности участников (членов) корпорации\n"
                "Континентальной Европы данные права традиционно разграничиваются.\n"
                "В своей основе такое разделение имеет различную трактовку понятий управления и контроля.\n"
                "Изначально в правовой системе контроль ассоциировался с проверкой деятельности субъектов.\n"
                "Правом воздействовать на указанных лиц наделялись контролирующие органы.\n"
            ),
            "notes_text": "",
            "reference_text": (
                "1. Berle A., Means G. The Modern Corporation and Private Property. N.Y., 1934.\n"
                "2. Winter R. Government and Corporation. Washington, 1978.\n"
            ),
            "page_type": "body_with_notes",
        }

        cleaned = agent.clean_page(page, repeated_headers=set(), repeated_footers=set())

        self.assertFalse(cleaned.get("drop_page"))
        self.assertIn("Континентальной Европы данные права", cleaned["cleaned_text"])
        self.assertIn("Правом воздействовать на указанных лиц", cleaned["cleaned_text"])


if __name__ == "__main__":
    unittest.main()

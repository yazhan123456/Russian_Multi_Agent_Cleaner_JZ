from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "process_books.py"
SPEC = importlib.util.spec_from_file_location("process_books_under_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
process_books = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(process_books)


class ExportPostCleanTests(unittest.TestCase):
    def test_run_post_clean_final_txt_fixes_spaced_hyphenation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_dir = Path(temp_dir) / "txt"
            backup_dir = Path(temp_dir) / "backups"
            txt_dir.mkdir()
            file_path = txt_dir / "sample.txt"
            file_path.write_text(
                "Это толь- ко тест.\n"
                "построе- ния текста.\n"
                "ин-фекционной болезни.\n"
                "из- за ошибки.\n"
                "большинст-ва случаев.\n"
                "начальни-ка учреждения.\n"
                "ребен-ка осужденной.\n"
                "какой-то пример.\n"
                "1.4. Ƀɬɭɩɲɨɣɥɣ ɣ ɨɛɮɥɛ ɮɞɩɦɩɝɨɩ-ɣɬɪɩɦɨɣɭɠɦɷɨɩɞɩ ɪɫɛɝɛ Уголовно-исполнительное законодательство.\n"
                "Ɋɫɠɟɣɬɦɩɝɣɠ Российская система исполнения наказаний.\n\n"
                "8.5. Ƀɬɪɩɦɨɠɨɣɠ ɨɛɥɛɢɛɨɣɺ ɝ ɝɣɟɠ ɩɞɫɛɨɣɲɠɨɣɺ ɬɝɩɜɩɟɶ\n"
                "Нормальный текст.\n\n"
                "БЕЛОЕ ДВИЖЕНИЕ\n"
                "ПОСЛЕДНИЕ БОИ ВООРУЖЕННЫХ СИЛ ЮГА РОССИИ\n"
                "< ПОСЛЕДНИЕ БОИ ВООРУЖЕННЫХ CUA ЮГА POCCUH\n"
                "К. С. Веселовский\n"
                "r&?-... шумная строка заголовка\n"
                "ENG о А ¥: #2 шумная строка\n"
                "Текст в лицее»3!).\n"
                "Документы вплоть до 1750 года?3).\n"
                "ВЕРНУТЬСЯ К ИНДЕКСУ\n",
                encoding="utf-8",
            )

            process_books.run_post_clean_final_txt(file_path, backup_root=backup_dir)

            cleaned = file_path.read_text(encoding="utf-8")
            self.assertIn("только", cleaned)
            self.assertIn("построения", cleaned)
            self.assertIn("инфекционной", cleaned)
            self.assertIn("большинства", cleaned)
            self.assertIn("начальника", cleaned)
            self.assertIn("ребенка", cleaned)
            self.assertIn("из-за", cleaned)
            self.assertIn("какой-то", cleaned)
            self.assertIn("1.4. Уголовно-исполнительное законодательство.", cleaned)
            self.assertIn("Российская система исполнения наказаний.", cleaned)
            self.assertIn("8.5.", cleaned)
            self.assertIn("Нормальный текст.", cleaned)
            self.assertIn("Текст в лицее».", cleaned)
            self.assertIn("Документы вплоть до 1750 года.", cleaned)
            self.assertNotIn("толь- ко", cleaned)
            self.assertNotIn("построе- ния", cleaned)
            self.assertNotIn("ин-фекционной", cleaned)
            self.assertNotIn("большинст-ва", cleaned)
            self.assertNotIn("начальни-ка", cleaned)
            self.assertNotIn("ребен-ка", cleaned)
            self.assertNotIn("из- за", cleaned)
            self.assertNotIn("Ƀɬɭɩɲɨɣɥɣ", cleaned)
            self.assertNotIn("Ɋɫɠɟɣɬɦɩɝɣɠ", cleaned)
            self.assertNotIn("Ƀɬɪɩɦɨɠɨɣɠ", cleaned)
            self.assertNotIn("ВЕРНУТЬСЯ К ИНДЕКСУ", cleaned)
            self.assertNotIn("БЕЛОЕ ДВИЖЕНИЕ", cleaned)
            self.assertNotIn("ПОСЛЕДНИЕ БОИ ВООРУЖЕННЫХ СИЛ ЮГА РОССИИ", cleaned)
            self.assertNotIn("ПОСЛЕДНИЕ БОИ ВООРУЖЕННЫХ CUA ЮГА POCCUH", cleaned)
            self.assertNotIn("К. С. Веселовский\n", cleaned)
            self.assertNotIn("r&?-...", cleaned)
            self.assertNotIn("ENG о А ¥: #2", cleaned)
            self.assertNotIn("лицее»3!)", cleaned)
            self.assertNotIn("1750 года?3)", cleaned)
            self.assertTrue(any(backup_dir.iterdir()))
            self.assertEqual(len(list(txt_dir.glob("post_clean_report_*.json"))), 1)

    def test_run_post_clean_final_txt_trims_trailing_backmatter_and_prompt_leak(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_dir = Path(temp_dir) / "txt"
            backup_dir = Path(temp_dir) / "backups"
            txt_dir.mkdir()
            file_path = txt_dir / "sample.txt"
            file_path.write_text(
                "Основной текст книги.\n"
                "Еще один абзац.\n"
                "Именной указатель\n"
                "Шумная служебная строка\n"
                "Here's a step-bystep breakdown of the structural cleanup\n"
                "Remaining content\n",
                encoding="utf-8",
            )

            process_books.run_post_clean_final_txt(file_path, backup_root=backup_dir)

            cleaned = file_path.read_text(encoding="utf-8")
            self.assertIn("Основной текст книги.", cleaned)
            self.assertIn("Еще один абзац.", cleaned)
            self.assertNotIn("Именной указатель", cleaned)
            self.assertNotIn("Here's a step-bystep breakdown", cleaned)

    def test_run_post_clean_final_txt_keeps_midbook_bibliography_headings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_dir = Path(temp_dir) / "txt"
            backup_dir = Path(temp_dir) / "backups"
            txt_dir.mkdir()
            file_path = txt_dir / "sample.txt"
            file_path.write_text(
                "Первая статья начинается здесь и продолжается достаточно долго, чтобы пройти эвристику очистки.\r"
                "Список литературы Ключевые слова: археология, история, монастыри.\r"
                "Следующая статья начинается сразу после этого заголовка и не должна быть отрезана.\r"
                "Еще один абзац второй статьи с нормальным содержанием.\r"
                "КОРОТКО ОБ АВТОРАХ\r"
                "Служебный хвост, который действительно нужно отбросить.\r",
                encoding="utf-8",
            )

            process_books.run_post_clean_final_txt(file_path, backup_root=backup_dir)

            cleaned = file_path.read_text(encoding="utf-8")
            self.assertIn("Первая статья начинается здесь", cleaned)
            self.assertIn("Следующая статья начинается сразу после этого заголовка", cleaned)
            self.assertIn("Еще один абзац второй статьи", cleaned)
            self.assertNotIn("Список литературы", cleaned)
            self.assertNotIn("КОРОТКО ОБ АВТОРАХ", cleaned)
            self.assertNotIn("Служебный хвост", cleaned)

    def test_run_post_clean_final_txt_strips_inline_bibliography_and_inline_figure_caption(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_dir = Path(temp_dir) / "txt"
            backup_dir = Path(temp_dir) / "backups"
            txt_dir.mkdir()
            file_path = txt_dir / "sample.txt"
            file_path.write_text(
                "Полезный абзац. Список литературы Следующий раздел начинается здесь и должен сохраниться.\n"
                "Таким образом, Рис. 4. План парада 5 сентября 1752 г. ОР БАН. планы парадов показывают отсутствующую в других источниках расстановку войск.\n"
                "В изразцовой Рис. 3. Геральдические изразцы: 1 – карнизный; 2 – фризовый. серии Нового Иерусалима выполнена оригинальная версия.\n",
                encoding="utf-8",
            )

            process_books.run_post_clean_final_txt(file_path, backup_root=backup_dir)

            cleaned = file_path.read_text(encoding="utf-8")
            self.assertIn("Полезный абзац.", cleaned)
            self.assertIn("Следующий раздел начинается здесь и должен сохраниться.", cleaned)
            self.assertIn("Таким образом, планы парадов показывают отсутствующую в других источниках расстановку войск.", cleaned)
            self.assertIn("В изразцовой серии Нового Иерусалима выполнена оригинальная версия.", cleaned)
            self.assertNotIn("Список литературы", cleaned)
            self.assertNotIn("Рис. 4. План парада 5 сентября 1752 г. ОР БАН.", cleaned)
            self.assertNotIn("Рис. 3. Геральдические изразцы", cleaned)

    def test_run_post_clean_final_txt_restores_paragraph_lines_and_compact_ocr_hyphenation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_dir = Path(temp_dir) / "txt"
            backup_dir = Path(temp_dir) / "backups"
            txt_dir.mkdir()
            file_path = txt_dir / "sample.txt"
            file_path.write_text(
                "Первый абзац начинается\n"
                "на второй строке и\n"
                "заканчивается здесь.\n\n"
                "Следующий абзац зи-мой\n"
                "видел тру-пы и моз-гу\n"
                "это не мешало.\n\n"
                "1.2. Заголовок раздела\n"
                "Третий абзац идет\n"
                "дальше без разрыва.\n"
                "— Отдельная реплика.\n"
                "из-за этого кто-то спорил.\n"
                "Мы поразному и всетаки это понимали, а когдалибо и ктонибудь спорили.\n",
                encoding="utf-8",
            )

            process_books.run_post_clean_final_txt(file_path, backup_root=backup_dir)

            cleaned = file_path.read_text(encoding="utf-8")
            lines = [line for line in cleaned.splitlines() if line.strip()]

            self.assertIn("Первый абзац начинается на второй строке и заканчивается здесь.", cleaned)
            self.assertIn("Следующий абзац зимой видел трупы и мозгу это не мешало.", cleaned)
            self.assertIn("1.2. Заголовок раздела", cleaned)
            self.assertIn("Третий абзац идет дальше без разрыва.", cleaned)
            self.assertIn("— Отдельная реплика.", cleaned)
            self.assertIn("из-за этого кто-то спорил.", cleaned)
            self.assertIn("Мы по-разному и все-таки это понимали, а когда-либо и кто-нибудь спорили.", cleaned)
            self.assertNotIn("зи-мой", cleaned)
            self.assertNotIn("тру-пы", cleaned)
            self.assertNotIn("моз-гу", cleaned)
            self.assertNotIn("поразному", cleaned)
            self.assertNotIn("всетаки", cleaned)
            self.assertNotIn("когдалибо", cleaned)
            self.assertNotIn("ктонибудь", cleaned)
            self.assertNotIn("\n\n", cleaned)
            self.assertEqual(lines[0], "Первый абзац начинается на второй строке и заканчивается здесь.")
            self.assertEqual(lines[1], "Следующий абзац зимой видел трупы и мозгу это не мешало.")
            self.assertEqual(lines[2], "1.2. Заголовок раздела")

    def test_run_post_clean_final_txt_fixes_intraword_space_splits_conservatively(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_dir = Path(temp_dir) / "txt"
            backup_dir = Path(temp_dir) / "backups"
            txt_dir.mkdir()
            file_path = txt_dir / "sample.txt"
            file_path.write_text(
                "Это модель историче ского времени.\n"
                "Как конструировал ся XIX век.\n"
                "Рост про изводительных сил.\n"
                "Современные ис следователи спорят.\n"
                "Но которые были рядом, не только наблюдали век прогресса.\n"
                "Мы говорили про историю и про изводство фабрик.\n",
                encoding="utf-8",
            )

            process_books.run_post_clean_final_txt(file_path, backup_root=backup_dir)

            cleaned = file_path.read_text(encoding="utf-8")
            self.assertIn("исторического", cleaned)
            self.assertIn("конструировался", cleaned)
            self.assertIn("производительных", cleaned)
            self.assertIn("исследователи", cleaned)
            self.assertIn("которые были", cleaned)
            self.assertIn("не только", cleaned)
            self.assertIn("век прогресса", cleaned)
            self.assertIn("про историю", cleaned)
            self.assertIn("производство фабрик", cleaned)
            self.assertNotIn("историче ского", cleaned)
            self.assertNotIn("конструировал ся", cleaned)
            self.assertNotIn("про изводительных", cleaned)
            self.assertNotIn("ис следователи", cleaned)

    def test_run_post_clean_final_txt_fixes_high_confidence_joined_words(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_dir = Path(temp_dir) / "txt"
            backup_dir = Path(temp_dir) / "backups"
            txt_dir.mkdir()
            file_path = txt_dir / "sample.txt"
            file_path.write_text(
                "Текст с покрываемойпланом территорией и частойпричиной ошибок.\n"
                "Также встречаются населенногопункта, поворотногопункта и незавершенностиработ.\n"
                "Упоминаются Семеновскогополка и Троицкогособора.\n",
                encoding="utf-8",
            )

            process_books.run_post_clean_final_txt(file_path, backup_root=backup_dir)

            cleaned = file_path.read_text(encoding="utf-8")
            self.assertIn("покрываемой планом", cleaned)
            self.assertIn("частой причиной", cleaned)
            self.assertIn("населенного пункта", cleaned)
            self.assertIn("поворотного пункта", cleaned)
            self.assertIn("незавершенности работ", cleaned)
            self.assertIn("Семеновского полка", cleaned)
            self.assertIn("Троицкого собора", cleaned)
            self.assertNotIn("покрываемойпланом", cleaned)
            self.assertNotIn("частойпричиной", cleaned)
            self.assertNotIn("населенногопункта", cleaned)
            self.assertNotIn("поворотногопункта", cleaned)
            self.assertNotIn("незавершенностиработ", cleaned)

    def test_run_post_clean_final_txt_removes_page_numbers_around_headings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_dir = Path(temp_dir) / "txt"
            backup_dir = Path(temp_dir) / "backups"
            txt_dir.mkdir()
            file_path = txt_dir / "sample.txt"
            file_path.write_text(
                "Введение в книгу.\n"
                "117 Глава 5. ПОВЕРЖЕННЫЙ ДРАКОН\n"
                "Текст главы начинается здесь.\n"
                "Фраза перед разрывом 502 Часть IV. УГЛЕВОДОРОДНЫЙ ВЕК продолжается дальше.\n"
                "Конец по- ДОБЫЧА ложили революцию.\n",
                encoding="utf-8",
            )

            process_books.run_post_clean_final_txt(file_path, backup_root=backup_dir)

            cleaned = file_path.read_text(encoding="utf-8")
            self.assertIn("Глава 5. ПОВЕРЖЕННЫЙ ДРАКОН", cleaned)
            self.assertIn("положили революцию.", cleaned)
            self.assertNotIn("117 Глава 5", cleaned)
            self.assertNotIn("502 Часть IV", cleaned)
            self.assertNotIn("Часть IV. УГЛЕВОДОРОДНЫЙ ВЕК", cleaned)
            self.assertNotIn("по- ДОБЫЧА ложили", cleaned)
            self.assertIn("Фраза перед разрывом продолжается дальше.", cleaned)

    def test_run_post_clean_final_txt_strips_intrusive_standalone_heading_between_sentence_parts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_dir = Path(temp_dir) / "txt"
            backup_dir = Path(temp_dir) / "backups"
            txt_dir.mkdir()
            file_path = txt_dir / "sample.txt"
            file_path.write_text(
                "На Вагита Алекперова большое впечатление произвела модель «интегрированной нефтяной\n"
                "ПРЕДИСЛОВИЕ КО ВТОРОМУ РУССКОМУ ИЗДАНИЮ\n"
                "компании», с которой он познакомился на Западе.\n",
                encoding="utf-8",
            )

            process_books.run_post_clean_final_txt(file_path, backup_root=backup_dir)

            cleaned = file_path.read_text(encoding="utf-8")
            self.assertIn("модель «интегрированной нефтяной компании», с которой он познакомился на Западе.", cleaned)
            self.assertNotIn("ПРЕДИСЛОВИЕ КО ВТОРОМУ РУССКОМУ ИЗДАНИЮ", cleaned)

    def test_run_post_clean_final_txt_strips_intrusive_inline_heading_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_dir = Path(temp_dir) / "txt"
            backup_dir = Path(temp_dir) / "backups"
            txt_dir.mkdir()
            file_path = txt_dir / "sample.txt"
            file_path.write_text(
                "И если бы не государственное\n"
                "Глава 30. «НАША ЖИЗНЬ ВЫСТАВЛЕНА НА ТОРГИ» регулирование, цены на туалетную бумагу повысились бы.\n",
                encoding="utf-8",
            )

            process_books.run_post_clean_final_txt(file_path, backup_root=backup_dir)

            cleaned = file_path.read_text(encoding="utf-8")
            self.assertIn("И если бы не государственное регулирование, цены на туалетную бумагу повысились бы.", cleaned)
            self.assertNotIn("Глава 30. «НАША ЖИЗНЬ ВЫСТАВЛЕНА НА ТОРГИ»", cleaned)

    def test_run_post_clean_final_txt_splits_first_section_heading_and_strips_repeated_running_header(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_dir = Path(temp_dir) / "txt"
            backup_dir = Path(temp_dir) / "backups"
            txt_dir.mkdir()
            file_path = txt_dir / "sample.txt"
            file_path.write_text(
                "Часть I\n"
                "ОТЦЫ-ОСНОВАТЕЛИ\n"
                "Глава 1. «НА УМЕ ТОЛЬКО НЕФТЬ»: НАЧАЛО Вся проблема заключалась в невыплаченных деньгах.\n"
                "Глава 30. «НАША ЖИЗНЬ ВЫСТАВЛЕНА НА ТОРГИ» король Фейсал не был склонен.\n"
                "И если бы не государственное\n"
                "Глава 30. «НАША ЖИЗНЬ ВЫСТАВЛЕНА НА ТОРГИ» регулирование, цены на туалетную бумагу повысились бы.\n"
                "Глава 30. «НАША ЖИЗНЬ ВЫСТАВЛЕНА НА ТОРГИ» «НЕФТЯНОЕ ОРУЖИЕ» ЗАЧЕХЛЕНО\n"
                "Текст перед повтором главы\n"
                "Глава 1. «НА УМЕ ТОЛЬКО НЕФТЬ»: НАЧАЛО ным судам приходилось отправляться дальше.\n",
                encoding="utf-8",
            )

            process_books.run_post_clean_final_txt(file_path, backup_root=backup_dir)

            cleaned = file_path.read_text(encoding="utf-8")
            self.assertIn("Часть I", cleaned)
            self.assertIn("ОТЦЫ-ОСНОВАТЕЛИ", cleaned)
            self.assertIn("Глава 1. «НА УМЕ ТОЛЬКО НЕФТЬ»: НАЧАЛО", cleaned)
            self.assertIn("Вся проблема заключалась в невыплаченных деньгах.", cleaned)
            self.assertIn("Глава 30. «НАША ЖИЗНЬ ВЫСТАВЛЕНА НА ТОРГИ»", cleaned)
            self.assertIn("король Фейсал не был склонен.", cleaned)
            self.assertIn("И если бы не государственное регулирование, цены на туалетную бумагу повысились бы.", cleaned)
            self.assertIn("«НЕФТЯНОЕ ОРУЖИЕ» ЗАЧЕХЛЕНО", cleaned)
            self.assertIn("Текст перед повтором главы ным судам приходилось отправляться дальше.", cleaned)
            self.assertEqual(cleaned.count("Глава 1. «НА УМЕ ТОЛЬКО НЕФТЬ»: НАЧАЛО"), 1)
            self.assertEqual(cleaned.count("Глава 30. «НАША ЖИЗНЬ ВЫСТАВЛЕНА НА ТОРГИ»"), 1)

    def test_run_post_clean_final_txt_splits_leading_allcaps_heading_from_paragraph(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_dir = Path(temp_dir) / "txt"
            backup_dir = Path(temp_dir) / "backups"
            txt_dir.mkdir()
            file_path = txt_dir / "sample.txt"
            file_path.write_text(
                "XIX ВЕК В МИРОВОЙ ИСТОРИИ: ПРОБЛЕМЫ, ПОДХОДЫ, МОДЕЛИ ВРЕМЕНИ Говоря о XIX веке, мы сразу задумываемся о содержании этого понятия.\n"
                "PAX BRITANNICA: ИНДИЯ Колониальный период в истории Индии распадается на две части.\n",
                encoding="utf-8",
            )

            process_books.run_post_clean_final_txt(file_path, backup_root=backup_dir)

            cleaned = file_path.read_text(encoding="utf-8")
            self.assertIn("XIX ВЕК В МИРОВОЙ ИСТОРИИ: ПРОБЛЕМЫ, ПОДХОДЫ, МОДЕЛИ ВРЕМЕНИ", cleaned)
            self.assertIn("Говоря о XIX веке, мы сразу задумываемся о содержании этого понятия.", cleaned)
            self.assertIn("PAX BRITANNICA: ИНДИЯ", cleaned)
            self.assertIn("Колониальный период в истории Индии распадается на две части.", cleaned)
            self.assertNotIn("ВРЕМЕНИ Говоря", cleaned)
            self.assertNotIn("ИНДИЯ Колониальный", cleaned)

    def test_run_post_clean_final_txt_strips_standalone_figure_captions_but_keeps_inline_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_dir = Path(temp_dir) / "txt"
            backup_dir = Path(temp_dir) / "backups"
            txt_dir.mkdir()
            file_path = txt_dir / "sample.txt"
            file_path.write_text(
                "Основной текст с отсылкой (рис. 3 и 4) продолжается.\n"
                "Рис. 4. План парада 5 сентября 1752 г. РГИА.\n"
                "Илл. 2. Зарисовка печи из собрания музея.\n"
                "Табл. 1. Сводные показатели раскопов.\n"
                "Далее обычный абзац.\n",
                encoding="utf-8",
            )

            process_books.run_post_clean_final_txt(file_path, backup_root=backup_dir)

            cleaned = file_path.read_text(encoding="utf-8")
            self.assertIn("Основной текст с отсылкой (рис. 3 и 4) продолжается.", cleaned)
            self.assertIn("Далее обычный абзац.", cleaned)
            self.assertNotIn("Рис. 4. План парада 5 сентября 1752 г. РГИА.", cleaned)
            self.assertNotIn("Илл. 2. Зарисовка печи из собрания музея.", cleaned)
            self.assertNotIn("Табл. 1. Сводные показатели раскопов.", cleaned)

    def test_run_post_clean_final_txt_strips_leading_and_trailing_figure_caption_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_dir = Path(temp_dir) / "txt"
            backup_dir = Path(temp_dir) / "backups"
            txt_dir.mkdir()
            file_path = txt_dir / "sample.txt"
            file_path.write_text(
                "Рис. 12. План парада 6 января 1760 г. РГИА. по Неве по всем полкам стояли войска.\n"
                "XVIII век оказался временем перемен. Рис. 11. Изразцовая печь в Трапезных палатах монастыря.\n"
                "Законная ссылка на рисунок (Рис. 5) должна остаться.\n",
                encoding="utf-8",
            )

            process_books.run_post_clean_final_txt(file_path, backup_root=backup_dir)

            cleaned = file_path.read_text(encoding="utf-8")
            self.assertIn("по Неве по всем полкам", cleaned)
            self.assertIn("стояли войска.", cleaned)
            self.assertIn("XVIII век оказался временем перемен.", cleaned)
            self.assertIn("Законная ссылка на рисунок (Рис. 5) должна остаться.", cleaned)
            self.assertNotIn("Рис. 12. План парада 6 января 1760 г. РГИА.", cleaned)
            self.assertNotIn("Рис. 11. Изразцовая печь в Трапезных палатах монастыря.", cleaned)

    def test_run_post_clean_final_txt_strips_leading_figure_caption_before_uppercase_body(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_dir = Path(temp_dir) / "txt"
            backup_dir = Path(temp_dir) / "backups"
            txt_dir.mkdir()
            file_path = txt_dir / "sample.txt"
            file_path.write_text(
                "Рис. 18. План трона Всеобщего орденского праздника 8 ноября 1798 г. РГИА. "
                "Первый опыт оценки археологического потенциала царских резиденций России был поставлен более четверти века назад. "
                "Исторические свидетельства о дворце царя Алексея Михайловича позволили поставить вопрос о степени информативности материалов.\n",
                encoding="utf-8",
            )

            process_books.run_post_clean_final_txt(file_path, backup_root=backup_dir)

            cleaned = file_path.read_text(encoding="utf-8")
            self.assertIn("Первый опыт оценки археологического потенциала царских резиденций России", cleaned)
            self.assertNotIn("Рис. 18. План трона Всеобщего орденского праздника 8 ноября 1798 г. РГИА.", cleaned)

    def test_run_post_clean_final_txt_strips_seen_heading_prefix_before_figure_caption(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_dir = Path(temp_dir) / "txt"
            backup_dir = Path(temp_dir) / "backups"
            txt_dir.mkdir()
            file_path = txt_dir / "sample.txt"
            file_path.write_text(
                "Часть I. Города, дворцы и крепости\n"
                "Часть I. Города, дворцы и крепости Рис. 5. План парада 5 сентября 1752 г. РГИА.\n"
                "Часть I. Города, дворцы и крепости Рис. 13. План парада 6 января 1765 г. РГИА. источниками и подчас содержат ценную информацию.\n",
                encoding="utf-8",
            )

            process_books.run_post_clean_final_txt(file_path, backup_root=backup_dir)

            cleaned = file_path.read_text(encoding="utf-8")
            self.assertIn("Часть I. Города, дворцы и крепости", cleaned)
            self.assertIn("источниками и подчас содержат ценную информацию.", cleaned)
            self.assertNotIn("Часть I. Города, дворцы и крепости Рис. 5.", cleaned)
            self.assertNotIn("Часть I. Города, дворцы и крепости Рис. 13.", cleaned)

    def test_run_post_clean_final_txt_fixes_common_joined_words(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_dir = Path(temp_dir) / "txt"
            backup_dir = Path(temp_dir) / "backups"
            txt_dir.mkdir()
            file_path = txt_dir / "sample.txt"
            file_path.write_text(
                "Офорты траурногозала сохранились.\n"
                "От Конюшенногомоста шли дальше.\n"
                "Крестнымходом прошли к храму.\n"
                "Головленковатюрьма стояла у стены.\n"
                "Ивсе вещи были внутри, устроенныеосями вдоль стен и в километрахниже реки.\n",
                encoding="utf-8",
            )

            process_books.run_post_clean_final_txt(file_path, backup_root=backup_dir)

            cleaned = file_path.read_text(encoding="utf-8")
            self.assertIn("траурного зала", cleaned)
            self.assertIn("Конюшенного моста", cleaned)
            self.assertIn("Крестным ходом", cleaned)
            self.assertIn("Головленкова тюрьма", cleaned)
            self.assertIn("И все вещи были внутри, устроенные осями вдоль стен и в километрах ниже реки.", cleaned)
            self.assertNotIn("траурногозала", cleaned)
            self.assertNotIn("Конюшенногомоста", cleaned)
            self.assertNotIn("крестнымходом", cleaned)
            self.assertNotIn("Головленковатюрьма", cleaned)
            self.assertNotIn("ивсе", cleaned.lower())
            self.assertNotIn("устроенныеосями", cleaned)
            self.assertNotIn("километрахниже", cleaned)

    def test_run_post_clean_final_txt_strips_author_title_prefix_before_figure_caption(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_dir = Path(temp_dir) / "txt"
            backup_dir = Path(temp_dir) / "backups"
            txt_dir.mkdir()
            file_path = txt_dir / "sample.txt"
            file_path.write_text(
                "В.А. Буров. ХVIII век на Соловках в зеркале археологических источников Рис. 4. Деревянная платформа под пушку 1790 г.\n"
                "А.А. Голубинский. Точность в изображении города XVIII в. на планах Генерального межевания Рис. 4. Различие в границах планов.\n",
                encoding="utf-8",
            )

            process_books.run_post_clean_final_txt(file_path, backup_root=backup_dir)

            cleaned = file_path.read_text(encoding="utf-8")
            self.assertNotIn("В.А. Буров. ХVIII век на Соловках в зеркале археологических источников Рис. 4.", cleaned)
            self.assertNotIn("А.А. Голубинский. Точность в изображении города XVIII в. на планах Генерального межевания Рис. 4.", cleaned)


if __name__ == "__main__":
    unittest.main()

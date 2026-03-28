from __future__ import annotations

import json
from pathlib import Path

from .state_models import PageState


class PageCheckpointStore:
    def __init__(self, book_dir: Path) -> None:
        self.root = book_dir / "page_states"

    def page_path(self, page_num: int) -> Path:
        return self.root / f"{page_num:04d}.json"

    def save_page(self, page_state: PageState) -> Path:
        path = self.page_path(page_state.page_num)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(page_state.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(path)
        return path

    def load_page(self, page_num: int) -> PageState | None:
        path = self.page_path(page_num)
        if not path.exists():
            return None
        return PageState.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def load_pages(self, page_numbers: list[int]) -> dict[int, PageState]:
        loaded: dict[int, PageState] = {}
        for page_num in page_numbers:
            page_state = self.load_page(page_num)
            if page_state is not None:
                loaded[page_num] = page_state
        return loaded

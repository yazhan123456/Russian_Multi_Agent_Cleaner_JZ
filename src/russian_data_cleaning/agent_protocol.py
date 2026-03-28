from __future__ import annotations

from typing import Any, Protocol

from .state_models import PageState


class PageStateAgent(Protocol):
    def run(self, page_state: PageState, **kwargs: Any) -> PageState:
        ...

from __future__ import annotations

from datetime import datetime
from typing import Iterable

from .state_models import PageProcessingState, PageState, ProcessingEvent


STATE_ORDER = [
    PageProcessingState.NEW,
    PageProcessingState.EXTRACTED,
    PageProcessingState.OCR_DONE,
    PageProcessingState.RULE_CLEANED,
    PageProcessingState.PRIMARY_CLEANED,
    PageProcessingState.REVIEWED,
    PageProcessingState.REPAIRED,
    PageProcessingState.STRUCTURE_RESTORED,
    PageProcessingState.EXPORTED,
]
STATE_RANK = {state: index for index, state in enumerate(STATE_ORDER)}
ALLOWED_TRANSITIONS: dict[PageProcessingState, set[PageProcessingState]] = {
    PageProcessingState.NEW: {PageProcessingState.EXTRACTED, PageProcessingState.OCR_DONE, PageProcessingState.FAILED},
    PageProcessingState.EXTRACTED: {PageProcessingState.RULE_CLEANED, PageProcessingState.FAILED},
    PageProcessingState.OCR_DONE: {PageProcessingState.RULE_CLEANED, PageProcessingState.FAILED},
    PageProcessingState.RULE_CLEANED: {PageProcessingState.PRIMARY_CLEANED, PageProcessingState.FAILED},
    PageProcessingState.PRIMARY_CLEANED: {
        PageProcessingState.REVIEWED,
        PageProcessingState.EXPORTED,
        PageProcessingState.FAILED,
    },
    PageProcessingState.REVIEWED: {
        PageProcessingState.REPAIRED,
        PageProcessingState.EXPORTED,
        PageProcessingState.FAILED,
    },
    PageProcessingState.REPAIRED: {
        PageProcessingState.STRUCTURE_RESTORED,
        PageProcessingState.EXPORTED,
        PageProcessingState.FAILED,
    },
    PageProcessingState.STRUCTURE_RESTORED: {PageProcessingState.EXPORTED, PageProcessingState.FAILED},
    PageProcessingState.EXPORTED: {PageProcessingState.FAILED},
    PageProcessingState.FAILED: {PageProcessingState.FAILED},
}


def can_transition(from_state: PageProcessingState, to_state: PageProcessingState) -> bool:
    if from_state == to_state:
        return True
    return to_state in ALLOWED_TRANSITIONS.get(from_state, set())


def require_transition(from_state: PageProcessingState, to_state: PageProcessingState) -> None:
    if not can_transition(from_state, to_state):
        raise ValueError(f"Invalid page-state transition: {from_state.value} -> {to_state.value}")


def state_at_least(state: PageProcessingState, target: PageProcessingState) -> bool:
    if state == PageProcessingState.FAILED:
        return False
    return STATE_RANK.get(state, -1) >= STATE_RANK.get(target, -1)


def effective_state(page_state: PageState) -> PageProcessingState:
    return page_state.effective_state()


def transition(page_state: PageState, to_state: PageProcessingState, *, agent: str, note: str = "") -> PageState:
    recorded_from_state = page_state.current_state
    from_state = effective_state(page_state) if recorded_from_state == PageProcessingState.FAILED else recorded_from_state
    require_transition(from_state, to_state)
    if recorded_from_state != PageProcessingState.FAILED and from_state == to_state:
        return page_state
    page_state.current_state = to_state
    if to_state != PageProcessingState.FAILED:
        page_state.last_success_state = to_state
    page_state.processing_history.append(
        ProcessingEvent(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            agent=agent,
            from_state=from_state.value,
            to_state=to_state.value,
            note=note,
        )
    )
    return page_state


def mark_failed(page_state: PageState, *, agent: str, error: str) -> PageState:
    from_state = page_state.current_state
    require_transition(from_state, PageProcessingState.FAILED)
    page_state.current_state = PageProcessingState.FAILED
    page_state.add_error(error)
    page_state.processing_history.append(
        ProcessingEvent(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            agent=agent,
            from_state=from_state.value,
            to_state=PageProcessingState.FAILED.value,
            note=error,
        )
    )
    return page_state


def latest_reached_state(states: Iterable[PageState], target: PageProcessingState) -> int:
    return sum(1 for state in states if state_at_least(effective_state(state), target))

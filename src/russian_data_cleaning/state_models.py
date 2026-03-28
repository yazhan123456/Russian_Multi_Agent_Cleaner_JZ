from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


class PageProcessingState(str, Enum):
    NEW = "NEW"
    EXTRACTED = "EXTRACTED"
    OCR_DONE = "OCR_DONE"
    RULE_CLEANED = "RULE_CLEANED"
    PRIMARY_CLEANED = "PRIMARY_CLEANED"
    REVIEWED = "REVIEWED"
    REPAIRED = "REPAIRED"
    STRUCTURE_RESTORED = "STRUCTURE_RESTORED"
    EXPORTED = "EXPORTED"
    FAILED = "FAILED"


@dataclass
class ProvenanceRecord:
    timestamp: str
    agent: str
    input_fields: list[str] = field(default_factory=list)
    output_fields: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProvenanceRecord":
        return cls(
            timestamp=str(payload.get("timestamp") or _timestamp()),
            agent=str(payload.get("agent") or ""),
            input_fields=[str(item) for item in payload.get("input_fields", [])],
            output_fields=[str(item) for item in payload.get("output_fields", [])],
            note=str(payload.get("note") or ""),
        )


@dataclass
class ProcessingEvent:
    timestamp: str
    agent: str
    from_state: str
    to_state: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProcessingEvent":
        return cls(
            timestamp=str(payload.get("timestamp") or _timestamp()),
            agent=str(payload.get("agent") or ""),
            from_state=str(payload.get("from_state") or PageProcessingState.NEW.value),
            to_state=str(payload.get("to_state") or PageProcessingState.NEW.value),
            note=str(payload.get("note") or ""),
        )


@dataclass
class PageState:
    doc_id: str
    page_num: int
    source_path: str
    source_type: str
    page_type: str | None = None
    route_decision: str | None = None
    ocr_mode: str | None = None
    raw_text: str = ""
    layout_blocks: list[dict[str, Any]] = field(default_factory=list)
    rule_cleaned_text: str = ""
    primary_clean_text: str = ""
    review_tags: list[str] = field(default_factory=list)
    risk_level: str | None = None
    edit_plan: dict[str, Any] | None = None
    repair_plan: dict[str, Any] | None = None
    structure_plan: dict[str, Any] | None = None
    repaired_text: str = ""
    final_text: str = ""
    confidence: float | None = None
    provenance: list[ProvenanceRecord] = field(default_factory=list)
    processing_history: list[ProcessingEvent] = field(default_factory=list)
    current_state: PageProcessingState = PageProcessingState.NEW
    last_success_state: PageProcessingState = PageProcessingState.NEW
    errors: list[str] = field(default_factory=list)
    stage_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        doc_id: str,
        page_num: int,
        source_path: str,
        source_type: str,
    ) -> "PageState":
        return cls(
            doc_id=doc_id,
            page_num=page_num,
            source_path=source_path,
            source_type=source_type,
        )

    def record_provenance(
        self,
        *,
        agent: str,
        input_fields: list[str] | None = None,
        output_fields: list[str] | None = None,
        note: str = "",
    ) -> None:
        self.provenance.append(
            ProvenanceRecord(
                timestamp=_timestamp(),
                agent=agent,
                input_fields=input_fields or [],
                output_fields=output_fields or [],
                note=note,
            )
        )

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def effective_state(self) -> PageProcessingState:
        if self.current_state == PageProcessingState.FAILED:
            return self.last_success_state
        return self.current_state

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "page_num": self.page_num,
            "source_path": self.source_path,
            "source_type": self.source_type,
            "page_type": self.page_type,
            "route_decision": self.route_decision,
            "ocr_mode": self.ocr_mode,
            "raw_text": self.raw_text,
            "layout_blocks": self.layout_blocks,
            "rule_cleaned_text": self.rule_cleaned_text,
            "primary_clean_text": self.primary_clean_text,
            "review_tags": self.review_tags,
            "risk_level": self.risk_level,
            "edit_plan": self.edit_plan,
            "repair_plan": self.repair_plan,
            "structure_plan": self.structure_plan,
            "repaired_text": self.repaired_text,
            "final_text": self.final_text,
            "confidence": self.confidence,
            "provenance": [record.to_dict() for record in self.provenance],
            "processing_history": [event.to_dict() for event in self.processing_history],
            "current_state": self.current_state.value,
            "last_success_state": self.last_success_state.value,
            "errors": list(self.errors),
            "stage_payloads": self.stage_payloads,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PageState":
        current_state = PageProcessingState(str(payload.get("current_state") or PageProcessingState.NEW.value))
        last_success_state = PageProcessingState(
            str(payload.get("last_success_state") or payload.get("current_state") or PageProcessingState.NEW.value)
        )
        return cls(
            doc_id=str(payload.get("doc_id") or ""),
            page_num=int(payload.get("page_num") or 0),
            source_path=str(payload.get("source_path") or ""),
            source_type=str(payload.get("source_type") or ""),
            page_type=payload.get("page_type"),
            route_decision=payload.get("route_decision"),
            ocr_mode=payload.get("ocr_mode"),
            raw_text=str(payload.get("raw_text") or ""),
            layout_blocks=list(payload.get("layout_blocks", [])),
            rule_cleaned_text=str(payload.get("rule_cleaned_text") or ""),
            primary_clean_text=str(payload.get("primary_clean_text") or ""),
            review_tags=[str(tag) for tag in payload.get("review_tags", [])],
            risk_level=payload.get("risk_level"),
            edit_plan=payload.get("edit_plan"),
            repair_plan=payload.get("repair_plan"),
            structure_plan=payload.get("structure_plan"),
            repaired_text=str(payload.get("repaired_text") or ""),
            final_text=str(payload.get("final_text") or ""),
            confidence=float(payload["confidence"]) if payload.get("confidence") is not None else None,
            provenance=[ProvenanceRecord.from_dict(item) for item in payload.get("provenance", [])],
            processing_history=[ProcessingEvent.from_dict(item) for item in payload.get("processing_history", [])],
            current_state=current_state,
            last_success_state=last_success_state,
            errors=[str(item) for item in payload.get("errors", [])],
            stage_payloads=dict(payload.get("stage_payloads", {})),
        )


@dataclass
class DocumentState:
    doc_id: str
    source_path: str
    source_type: str
    route_hint: str
    page_numbers: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "source_type": self.source_type,
            "route_hint": self.route_hint,
            "page_numbers": list(self.page_numbers),
        }

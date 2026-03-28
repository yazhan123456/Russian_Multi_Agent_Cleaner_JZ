from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable
from typing import Any

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None

from .state_models import PageState


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "page_verdict": {
            "type": "string",
            "enum": ["approve", "reject", "escalate"],
        },
        "summary": {"type": "string"},
        "concerns": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["page_verdict", "summary", "concerns"],
}


@dataclass
class GeminiReviewConfig:
    model: str = "gemini-2.5-flash"
    risky_only: bool = True


class GeminiReviewAgent:
    RISK_ORDER = {"low": 0, "medium": 1, "high": 2}
    RISK_BY_VERDICT = {
        "approve": "low",
        "escalate": "medium",
        "reject": "high",
    }

    def __init__(self, config: GeminiReviewConfig | None = None) -> None:
        self.config = config or GeminiReviewConfig()
        self.client = self._build_client()

    def review_document(
        self,
        cleaned_document: dict[str, Any],
        heuristic_review: dict[str, Any],
        progress_hook: Callable[[int, int, int], None] | None = None,
    ) -> dict[str, Any]:
        heuristic_map = {page["page_number"]: page for page in heuristic_review["pages"]}
        reviewed_pages = []
        total = len(cleaned_document["pages"])
        for index, page in enumerate(cleaned_document["pages"], start=1):
            if progress_hook is not None:
                progress_hook(index, total, page["page_number"])
            heuristic_page = heuristic_map[page["page_number"]]
            reviewed_pages.append(self.review_page(page, heuristic_page))

        return {
            "relative_path": cleaned_document["relative_path"],
            "pages": reviewed_pages,
        }

    def review_page(
        self,
        cleaned_page: dict[str, Any],
        heuristic_page: dict[str, Any],
    ) -> dict[str, Any]:
        if self.config.risky_only and heuristic_page["page_verdict"] == "approve" and not cleaned_page["flags"]:
            return {
                "page_number": cleaned_page["page_number"],
                "llm_verdict": "skipped",
                "summary": "Skipped because heuristic review found no risky signals.",
                "concerns": [],
            }
        return self._review_page(cleaned_page, heuristic_page)

    def _review_page(
        self,
        cleaned_page: dict[str, Any],
        heuristic_page: dict[str, Any],
    ) -> dict[str, Any]:
        raw_text = (cleaned_page.get("raw_text") or "")[:3000]
        cleaned_text = (cleaned_page.get("cleaned_text") or "")[:3000]
        edits = json.dumps(cleaned_page["edits"][:20], ensure_ascii=False)
        flags = json.dumps(cleaned_page["flags"][:20], ensure_ascii=False)
        protected_hits = json.dumps(cleaned_page["protected_hits"][:20], ensure_ascii=False)
        prompt = (
            "You are reviewing OCR cleanup on a Russian or mixed-language PDF page.\n"
            "Decide if the cleaned text should be approved, rejected, or escalated.\n"
            "Be conservative about removing lexical hyphens, abbreviations, numbering, citations, and values.\n"
            "Return JSON only.\n\n"
            f"Raw text:\n{raw_text}\n\n"
            f"Cleaned text:\n{cleaned_text}\n\n"
            f"Edits:\n{edits}\n\n"
            f"Flags:\n{flags}\n\n"
            f"Protected hits:\n{protected_hits}\n\n"
            f"Heuristic review:\n{json.dumps(heuristic_page, ensure_ascii=False)}"
        )
        response = None
        try:
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    responseMimeType="application/json",
                    responseSchema=RESPONSE_SCHEMA,
                    temperature=0.0,
                    maxOutputTokens=256,
                ),
            )
            payload = self._parse_payload(response)
        except Exception as exc:
            excerpt = ""
            if response is not None:
                excerpt = self._extract_response_text(response)[:240]
            detail = f"{type(exc).__name__}: {exc}"
            if excerpt:
                detail = f"{detail} | raw={excerpt!r}"
            return {
                "page_number": cleaned_page["page_number"],
                "llm_verdict": "escalate",
                "summary": "Gemini review failed; falling back to escalation.",
                "concerns": [detail],
            }
        return {
            "page_number": cleaned_page["page_number"],
            "llm_verdict": payload.get("page_verdict", "escalate"),
            "summary": payload.get("summary", ""),
            "concerns": payload.get("concerns", []),
        }

    def run(
        self,
        page_state: PageState,
        *,
        cleaned_page: dict[str, Any] | None = None,
        heuristic_page: dict[str, Any] | None = None,
    ) -> PageState:
        cleaned_payload = cleaned_page or page_state.stage_payloads.get("cleaned") or page_state.stage_payloads.get("primary_cleaned")
        if cleaned_payload is None:
            raise ValueError("GeminiReviewAgent.run requires cleaned page payload.")
        heuristic_payload = heuristic_page or page_state.stage_payloads.get("review")
        if heuristic_payload is None:
            raise ValueError("GeminiReviewAgent.run requires heuristic review payload.")

        gemini_payload = self.review_page(cleaned_payload, heuristic_payload)
        page_state.stage_payloads["gemini_review"] = gemini_payload
        page_state.risk_level = self._max_risk_level(
            page_state.risk_level,
            self.RISK_BY_VERDICT.get(str(gemini_payload.get("llm_verdict") or ""), "low"),
        )
        page_state.record_provenance(
            agent="GeminiReviewAgent",
            input_fields=["stage_payloads.cleaned", "stage_payloads.review"],
            output_fields=["risk_level", "stage_payloads.gemini_review"],
            note=str(gemini_payload.get("llm_verdict") or "unknown"),
        )
        return page_state

    def _build_client(self):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Gemini review selected but GOOGLE_API_KEY or GEMINI_API_KEY is not set.")
        if genai is None or genai_types is None:
            raise RuntimeError("google-genai package is not installed.")
        return genai.Client(api_key=api_key)

    def _parse_payload(self, response: Any) -> dict[str, Any]:
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            if hasattr(parsed, "model_dump"):
                return self._validate_payload(parsed.model_dump())
            if isinstance(parsed, dict):
                return self._validate_payload(parsed)

        text = self._extract_response_text(response)
        candidates = self._candidate_json_strings(text)
        for candidate in candidates:
            try:
                return self._validate_payload(json.loads(candidate))
            except json.JSONDecodeError:
                continue

        raise ValueError(f"Unable to parse Gemini JSON response: {text[:240]!r}")

    def _extract_response_text(self, response: Any) -> str:
        text = getattr(response, "text", "") or ""
        text = text.strip()
        if text.startswith("```") and text.endswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        return text.strip()

    def _candidate_json_strings(self, text: str) -> list[str]:
        candidates: list[str] = []
        if not text:
            return candidates

        stripped = text.strip()
        candidates.append(stripped)

        first_brace = stripped.find("{")
        last_brace = stripped.rfind("}")
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            candidates.append(stripped[first_brace : last_brace + 1])

        compact = re.sub(r"[\x00-\x1F]+", " ", stripped)
        if compact != stripped:
            candidates.append(compact)

        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                unique_candidates.append(candidate)
        return unique_candidates

    def _validate_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        verdict = payload.get("page_verdict")
        if verdict not in {"approve", "reject", "escalate"}:
            payload["page_verdict"] = "escalate"
        summary = payload.get("summary")
        if not isinstance(summary, str):
            payload["summary"] = ""
        concerns = payload.get("concerns")
        if not isinstance(concerns, list):
            payload["concerns"] = []
        else:
            payload["concerns"] = [str(item) for item in concerns]
        return payload

    def _max_risk_level(self, current: str | None, candidate: str | None) -> str:
        current_key = str(current or "low")
        candidate_key = str(candidate or "low")
        return current_key if self.RISK_ORDER.get(current_key, 0) >= self.RISK_ORDER.get(candidate_key, 0) else candidate_key

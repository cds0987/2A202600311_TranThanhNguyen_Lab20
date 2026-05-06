"""LLM client abstraction backed by OpenAI Responses API."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency during tests
    OpenAI = None  # type: ignore[assignment,misc]


GPT_4O_MINI_INPUT_PRICE_PER_1M = 0.15
GPT_4O_MINI_OUTPUT_PRICE_PER_1M = 0.60


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client with OpenAI implementation."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = self.settings.openai_model or "gpt-4o-mini"
        self._client = self._build_client()

    def _build_client(self) -> Any | None:
        if not self.settings.openai_api_key or OpenAI is None:
            return None
        return OpenAI(api_key=self.settings.openai_api_key, timeout=self.settings.timeout_seconds)

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion."""

        if self._client is None:
            return self._mock_complete(system_prompt=system_prompt, user_prompt=user_prompt)

        try:
            response = self._client.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=user_prompt,
            )
        except Exception as exc:
            logger.warning(
                "OpenAI request failed; falling back to deterministic local "
                "output: %s",
                exc,
            )
            return self._mock_complete(system_prompt=system_prompt, user_prompt=user_prompt)
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        content = getattr(response, "output_text", "") or ""
        cost = self._estimate_cost(input_tokens=input_tokens, output_tokens=output_tokens)
        return LLMResponse(
            content=content.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

    def _mock_complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Deterministic fallback used when the OpenAI SDK or API key is unavailable."""

        logger.info("OpenAI client unavailable; using deterministic local fallback.")
        user_lines = [line.strip() for line in user_prompt.splitlines() if line.strip()]
        preview = " ".join(user_lines[:6])[:600]
        system_text = system_prompt.lower()
        citations = [line[2:] for line in user_lines if line.startswith("- [")]

        if "single-agent research assistant" in system_text:
            content = (
                "This offline fallback summary explains the topic at a high "
                "level, but it does not claim live retrieval or fresh "
                "benchmarking data. It should be treated as a local demo "
                "answer."
            )
        elif "research agent" in system_text:
            content = (
                "Key findings:\n"
                "- The query benefits from grounded retrieval before synthesis.\n"
                "- Multiple sources help reduce unsupported claims.\n"
                "- Remaining uncertainty should be called out explicitly.\n"
                f"Query snapshot: {preview}"
            )
        elif "analyst agent" in system_text:
            content = (
                "Key claims:\n"
                "- Specialized agent roles improve observability.\n"
                "Supporting evidence:\n"
                "- Retrieval, analysis, and writing are separated.\n"
                "Gaps:\n"
                "- Stronger live sources would improve technical depth."
            )
        elif "writing agent" in system_text:
            source_lines = citations or ["[1] Local fallback source"]
            content = (
                "GraphRAG-style research workflows benefit from separating "
                "retrieval, analysis, and writing so each step can be "
                "inspected and improved [1]. In this offline fallback run, "
                "the system uses seed sources rather than live web results, "
                "which keeps the workflow testable but limits claim freshness "
                "[1]. A production-ready version should combine broader "
                "retrieval with stricter claim-to-source "
                "mapping and richer benchmarking [2].\n\n"
                "### Sources\n"
                + "\n".join(f"- {line}" for line in source_lines[:3])
            )
        elif "critic agent" in system_text:
            content = (
                "Review notes:\n"
                "- The answer is grounded, but live retrieval was unavailable.\n"
                "- Add more inline citations if additional sources are available.\n"
                "- Validate technical comparisons against broader evidence before publication."
            )
        else:
            content = f"[mock:{self.model}] {preview}".strip()
        input_tokens = max(1, (len(system_prompt) + len(user_prompt)) // 4)
        output_tokens = max(1, len(content) // 4)
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self._estimate_cost(input_tokens=input_tokens, output_tokens=output_tokens),
        )

    def _estimate_cost(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        if input_tokens is None or output_tokens is None:
            return None
        input_cost = (input_tokens / 1_000_000) * GPT_4O_MINI_INPUT_PRICE_PER_1M
        output_cost = (output_tokens / 1_000_000) * GPT_4O_MINI_OUTPUT_PRICE_PER_1M
        return input_cost + output_cost

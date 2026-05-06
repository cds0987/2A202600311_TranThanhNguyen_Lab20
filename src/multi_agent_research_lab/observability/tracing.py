"""Tracing hooks and export helpers.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or simple JSON traces.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.state import ResearchState


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context used by the lab."""

    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "duration_seconds": None,
        "status": "ok",
    }
    try:
        yield span
    except Exception:
        span["status"] = "error"
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started


def build_trace_artifact(state: ResearchState) -> dict[str, Any]:
    """Build an exportable trace payload for reports or screenshots."""

    return {
        "query": state.request.query,
        "audience": state.request.audience,
        "route_history": state.route_history,
        "iteration": state.iteration,
        "events": state.trace,
        "errors": state.errors,
        "usage": {
            "input_tokens": state.total_input_tokens,
            "output_tokens": state.total_output_tokens,
            "cost_usd": state.total_cost_usd,
        },
        "artifacts": {
            "source_count": len(state.sources),
            "has_research_notes": bool(state.research_notes),
            "has_analysis_notes": bool(state.analysis_notes),
            "has_final_answer": bool(state.final_answer),
            "has_critic_notes": bool(state.critic_notes),
        },
    }

"""Benchmark skeleton for single-agent vs multi-agent."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and estimate quality, cost, citation coverage, and failure rate."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    citation_coverage = _compute_citation_coverage(state)
    quality_breakdown = _score_quality_proxy(state, citation_coverage)
    quality = min(10.0, sum(quality_breakdown.values()))
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=state.total_cost_usd or None,
        quality_score=max(0.0, quality),
        quality_method="rubric-proxy",
        quality_breakdown=quality_breakdown,
        citation_coverage=citation_coverage,
        failure_rate=1.0 if state.errors or not state.final_answer else 0.0,
        notes=(
            f"iterations={state.iteration}, sources={len(state.sources)}, "
            f"errors={len(state.errors)}"
        ),
    )
    return state, metrics


def _compute_citation_coverage(state: ResearchState) -> float | None:
    if not state.sources or not state.final_answer:
        return None

    body = state.final_answer.split("### Sources", maxsplit=1)[0]
    cited_source_indexes: set[int] = set()
    for match in re.findall(r"\[(\d+)\]", body):
        cited_index = int(match)
        if 1 <= cited_index <= len(state.sources):
            cited_source_indexes.add(cited_index)

    for index, source in enumerate(state.sources, start=1):
        title_present = source.title.lower() in body.lower()
        url_present = bool(source.url and source.url.lower() in body.lower())
        if title_present or url_present:
            cited_source_indexes.add(index)

    return len(cited_source_indexes) / len(state.sources)


def _score_quality_proxy(state: ResearchState, citation_coverage: float | None) -> dict[str, float]:
    final_answer = state.final_answer or ""
    source_count = len(state.sources)
    breakdown = {
        "grounding": 0.0,
        "analysis": 0.0,
        "citations": 0.0,
        "completeness": 0.0,
        "traceability": 0.0,
    }

    if source_count >= 3:
        breakdown["grounding"] = 2.0
    elif source_count >= 1:
        breakdown["grounding"] = 1.0

    if state.analysis_notes and state.critic_notes:
        breakdown["analysis"] = 2.0
    elif state.analysis_notes or state.critic_notes:
        breakdown["analysis"] = 1.0

    if citation_coverage is not None:
        breakdown["citations"] = round(min(2.0, citation_coverage * 2), 2)

    if state.research_notes and state.analysis_notes and state.final_answer:
        breakdown["completeness"] = 2.0
    elif state.final_answer:
        breakdown["completeness"] = 1.0

    if state.trace and state.route_history:
        breakdown["traceability"] = 2.0
    elif state.route_history:
        breakdown["traceability"] = 1.0

    if len(final_answer.split()) >= 250 and breakdown["completeness"] > 0:
        breakdown["completeness"] = min(2.0, breakdown["completeness"] + 0.5)

    if state.errors:
        breakdown["traceability"] = max(0.0, breakdown["traceability"] - 1.0)

    return breakdown

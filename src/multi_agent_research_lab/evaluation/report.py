"""Benchmark report rendering."""

from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown."""

    lines = [
        "| Run | Latency (s) | Cost (USD) | Quality | Citation Coverage | Failure Rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation_coverage = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure_rate = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | "
            f"{citation_coverage} | {failure_rate} | {item.notes} |"
        )
        if item.quality_breakdown:
            breakdown = ", ".join(f"{name}={score:.2f}" for name, score in item.quality_breakdown.items())
            lines.append(f"| `{item.run_name}` rubric proxy |  |  |  |  |  | {breakdown} |")
    return "\n".join(lines) + "\n"


def render_benchmark_summary(
    query: str,
    baseline_metrics: BenchmarkMetrics,
    multi_agent_metrics: BenchmarkMetrics,
    multi_agent_state: ResearchState,
) -> str:
    """Render the richer benchmark artifact expected by the lab."""

    route = " -> ".join(multi_agent_state.route_history) if multi_agent_state.route_history else "none"
    retrieval_modes = sorted(
        {
            source.metadata.get("retrieval", "unknown")
            for source in multi_agent_state.sources
        }
    )
    source_lines = [
        f"- [{index}] {source.title} ({source.metadata.get('retrieval', 'unknown')})"
        for index, source in enumerate(multi_agent_state.sources, start=1)
    ] or ["- No sources retrieved."]
    return "\n".join(
        [
            "# Benchmark Report",
            "",
            "## Query",
            "",
            f"`{query}`",
            "",
            "## Benchmark results",
            "",
            render_markdown_report([baseline_metrics, multi_agent_metrics]).strip(),
            "",
            "## Workflow used",
            "",
            f"`{route}`",
            "",
            "## Retrieval modes",
            "",
            ", ".join(retrieval_modes) if retrieval_modes else "none",
            "",
            "## Sources retrieved",
            "",
            *source_lines,
            "",
            "## Failure mode and fix",
            "",
            _render_failure_mode(multi_agent_metrics, multi_agent_state),
            "",
            "## Exit ticket",
            "",
            "1. Use multi-agent when the task benefits from separate retrieval, analysis, writing, and review steps with observable handoffs.",
            "2. Avoid multi-agent for short or latency-sensitive tasks where orchestration overhead outweighs quality gains.",
            "",
        ]
    )


def render_requirements_report(
    baseline_metrics: BenchmarkMetrics,
    multi_agent_metrics: BenchmarkMetrics,
    multi_agent_state: ResearchState,
) -> str:
    """Render a requirement checklist based on the shipped implementation."""

    completed = [
        "Implemented OpenAI-backed LLM calls with retry and deterministic local fallback",
        "Implemented single-agent baseline execution",
        "Implemented multi-agent workflow with Supervisor, Researcher, Analyst, Writer, and Critic",
        "Implemented shared state handoff across agents",
        "Implemented trace events and exportable trace artifacts",
        "Implemented benchmark metrics plus generated benchmark artifacts",
        "Implemented query expansion and fallback retrieval behavior",
        "Added tests for workflow, reporting, and evaluation behavior",
    ]
    remaining = []
    if not multi_agent_state.trace:
        remaining.append("Trace export is empty for the current run.")
    if (multi_agent_metrics.citation_coverage or 0.0) == 0.0:
        remaining.append("The current benchmark query did not achieve positive citation coverage.")
    if len(multi_agent_state.sources) < 2:
        remaining.append("The current benchmark run still retrieved fewer than two sources.")

    lines = [
        "# Requirements Report",
        "",
        "## Completed requirements",
        "",
        *[f"- {item}" for item in completed],
        "",
        "## Current run notes",
        "",
        f"- Baseline quality score: {baseline_metrics.quality_score or 0:.2f}",
        f"- Multi-agent quality score: {multi_agent_metrics.quality_score or 0:.2f}",
        f"- Multi-agent citation coverage: {(multi_agent_metrics.citation_coverage or 0.0):.0%}",
        f"- Multi-agent trace events: {len(multi_agent_state.trace)}",
        "",
        "## Remaining risks",
        "",
    ]
    if remaining:
        lines.extend(f"- {item}" for item in remaining)
    else:
        lines.append("- No material requirement gaps detected in the current run.")

    return "\n".join(lines) + "\n"


def _render_failure_mode(metrics: BenchmarkMetrics, state: ResearchState) -> str:
    if len(state.sources) < 2:
        return (
            "The main failure mode is still narrow retrieval breadth. "
            "The next fix is to add a research-oriented retrieval backend or domain-specific source adapters."
        )
    if (metrics.citation_coverage or 0.0) < 0.5:
        return (
            "The main failure mode is weak inline citation use. "
            "The next fix is to tighten writer prompting so each major claim maps to one numbered source."
        )
    if state.errors:
        return (
            "The main failure mode is execution reliability. "
            "The next fix is to inspect trace events and harden the failing agent path."
        )
    return (
        "No critical failure mode was detected in this run. "
        "The next improvement is broader retrieval coverage to strengthen technical depth."
    )

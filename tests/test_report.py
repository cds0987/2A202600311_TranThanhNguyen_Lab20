from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.evaluation.report import render_markdown_report, render_requirements_report


def test_report_renders_markdown() -> None:
    report = render_markdown_report(
        [
            BenchmarkMetrics(
                run_name="baseline",
                latency_seconds=1.23,
                quality_breakdown={"grounding": 1.0},
            )
        ]
    )
    assert "| Run | Latency (s) |" in report
    assert "baseline" in report


def test_requirements_report_mentions_trace_events() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.add_trace_event("route", {"next": "researcher"})
    report = render_requirements_report(
        BenchmarkMetrics(run_name="baseline", latency_seconds=1.0),
        BenchmarkMetrics(run_name="multi-agent", latency_seconds=2.0, citation_coverage=1.0),
        state,
    )
    assert "trace events" in report

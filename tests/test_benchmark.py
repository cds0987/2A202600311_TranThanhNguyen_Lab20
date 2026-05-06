from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark


def test_benchmark_scores_citation_coverage_and_breakdown() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        sources=[
            SourceDocument(title="Source A", url="https://example.com/a", snippet="A"),
            SourceDocument(title="Source B", url="https://example.com/b", snippet="B"),
        ],
        research_notes="notes",
        analysis_notes="analysis",
        final_answer=(
            "A grounded answer with evidence [1] and [2].\n\n"
            "### Sources\n- [1] Source A\n- [2] Source B"
        ),
        critic_notes="critic",
    )
    state.route_history = ["researcher", "analyst", "writer", "critic", "done"]
    state.trace = [{"name": "route", "payload": {"next": "researcher"}}]

    _, metrics = run_benchmark("multi-agent", state.request.query, lambda query: state)

    assert metrics.citation_coverage == 1.0
    assert metrics.quality_score is not None and metrics.quality_score >= 8.0
    assert metrics.quality_breakdown["traceability"] >= 1.0

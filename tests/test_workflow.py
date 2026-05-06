from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import build_trace_artifact


class _DummyResponse:
    def __init__(self, content: str) -> None:
        self.content = content
        self.input_tokens = 10
        self.output_tokens = 20
        self.cost_usd = 0.01


class _DummyLLM:
    model = "dummy"

    def __init__(self, content: str) -> None:
        self._content = content

    def complete(self, system_prompt: str, user_prompt: str) -> _DummyResponse:
        return _DummyResponse(self._content)


def test_writer_adds_citations_and_sources_section() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain GraphRAG for technical learners"),
        sources=[
            SourceDocument(
                title="Microsoft GraphRAG repository",
                url="https://github.com/microsoft/graphrag",
                snippet="Reference implementation",
            )
        ],
        research_notes="GraphRAG uses graph structure for retrieval.",
        analysis_notes="The main benefit is better relationship-aware retrieval.",
    )
    result = WriterAgent(llm_client=_DummyLLM("GraphRAG improves retrieval quality.")).run(state)
    assert "[1]" in result.final_answer or ""
    assert "### Sources" in result.final_answer or ""


def test_build_trace_artifact_includes_usage_and_events() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.record_route("researcher")
    state.add_trace_event("route", {"next": "researcher"})
    state.record_usage(input_tokens=5, output_tokens=8, cost_usd=0.002)

    artifact = build_trace_artifact(state)

    assert artifact["route_history"] == ["researcher"]
    assert artifact["usage"]["input_tokens"] == 5
    assert artifact["events"][0]["name"] == "route"

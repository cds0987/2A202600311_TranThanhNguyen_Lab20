"""Researcher agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        search_client: SearchClient | None = None,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.search_client = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        state.sources = self.search_client.search(
            query=state.request.query,
            max_results=state.request.max_sources,
        )
        source_digest = "\n".join(
            (
                f"- {source.title}: {source.snippet} "
                f"({source.url or 'no-url'})"
            )
            for source in state.sources
        )
        response = self.llm_client.complete(
            system_prompt=(
                "You are a research agent. Summarize the most relevant "
                "findings, list consensus points, "
                "and note any uncertainty. Keep it concise and factual."
            ),
            user_prompt=f"Query: {state.request.query}\n\nSources:\n{source_digest}",
        )
        state.research_notes = response.content
        state.record_usage(response.input_tokens, response.output_tokens, response.cost_usd)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={"source_count": len(state.sources)},
            )
        )
        state.add_trace_event(
            "researcher.completed",
            {"source_count": len(state.sources), "used_model": self.llm_client.model},
        )
        return state

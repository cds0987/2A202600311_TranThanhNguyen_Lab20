"""Analyst agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        source_titles = ", ".join(source.title for source in state.sources) or "no sources"
        response = self.llm_client.complete(
            system_prompt=(
                "You are an analyst agent. Convert research notes into "
                "structured insights with sections "
                "for key claims, supporting evidence, disagreements, and gaps."
            ),
            user_prompt=(
                f"User query: {state.request.query}\n"
                f"Audience: {state.request.audience}\n"
                f"Sources: {source_titles}\n\n"
                f"Research notes:\n{state.research_notes or ''}"
            ),
        )
        state.analysis_notes = response.content
        state.record_usage(response.input_tokens, response.output_tokens, response.cost_usd)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={"source_count": len(state.sources)},
            )
        )
        state.add_trace_event("analyst.completed", {"used_model": self.llm_client.model})
        return state

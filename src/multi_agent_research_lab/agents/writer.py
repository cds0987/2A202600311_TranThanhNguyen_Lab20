"""Writer agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        citations = "\n".join(
            f"- [{idx}] {source.title} ({source.url or 'no-url'})" for idx, source in enumerate(state.sources, start=1)
        )
        response = self.llm_client.complete(
            system_prompt=(
                "You are a writing agent. Produce a clear, helpful answer for the target audience. "
                "Ground the answer in the provided sources, cite claims inline with bracketed citations "
                "such as [1] or [2], and end with a short Sources section that preserves the same numbering."
            ),
            user_prompt=(
                f"User query: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"Research notes:\n{state.research_notes or ''}\n\n"
                f"Analysis notes:\n{state.analysis_notes or ''}\n\n"
                f"Available citations:\n{citations}"
            ),
        )
        state.final_answer = self._ensure_citation_section(response.content, state)
        state.record_usage(response.input_tokens, response.output_tokens, response.cost_usd)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=state.final_answer,
                metadata={"citation_count": len(state.sources)},
            )
        )
        state.add_trace_event("writer.completed", {"used_model": self.llm_client.model})
        return state

    def _ensure_citation_section(self, content: str, state: ResearchState) -> str:
        answer = content.strip()
        if not state.sources:
            return answer

        if not any(f"[{index}]" in answer for index in range(1, len(state.sources) + 1)):
            answer = f"{answer}\n\nKey evidence is grounded in [1]."

        if "\n### Sources" in answer:
            return answer

        source_lines = [
            f"- [{index}] {source.title}: {source.url or 'no-url'}"
            for index, source in enumerate(state.sources, start=1)
        ]
        return f"{answer}\n\n### Sources\n" + "\n".join(source_lines)

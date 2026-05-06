"""Critic agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""

        source_titles = "\n".join(f"- {source.title}" for source in state.sources)
        response = self.llm_client.complete(
            system_prompt=(
                "You are a critic agent. Review the answer for unsupported "
                "claims, missing caveats, "
                "and citation gaps. Return concise review notes."
            ),
            user_prompt=(
                f"Query: {state.request.query}\n\n"
                f"Final answer:\n{state.final_answer or ''}\n\n"
                f"Available sources:\n{source_titles}"
            ),
        )
        state.critic_notes = response.content
        state.record_usage(response.input_tokens, response.output_tokens, response.cost_usd)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=response.content,
                metadata={"reviewed": bool(state.final_answer)},
            )
        )
        state.add_trace_event("critic.completed", {"used_model": self.llm_client.model})
        return state

"""Supervisor / router implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""

        settings = get_settings()
        if state.final_answer and state.critic_notes:
            next_route = "done"
        elif state.iteration >= settings.max_iterations:
            next_route = "writer" if not state.final_answer else "critic"
        elif not state.sources or not state.research_notes:
            next_route = AgentName.RESEARCHER.value
        elif not state.analysis_notes:
            next_route = AgentName.ANALYST.value
        elif not state.final_answer:
            next_route = AgentName.WRITER.value
        elif not state.critic_notes:
            next_route = AgentName.CRITIC.value
        else:
            next_route = "done"

        state.record_route(next_route)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.SUPERVISOR,
                content=f"Supervisor selected next route: {next_route}",
                metadata={"iteration": state.iteration},
            )
        )
        state.add_trace_event("supervisor.route", {"next": next_route, "iteration": state.iteration})
        return state

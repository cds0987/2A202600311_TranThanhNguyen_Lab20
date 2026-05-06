"""Workflow orchestration for the multi-agent lab."""

from __future__ import annotations

from collections.abc import Callable

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

AgentRunner = Callable[[ResearchState], ResearchState]


class MultiAgentWorkflow:
    """Builds and runs the multi-agent workflow."""

    def build(self) -> dict[str, AgentRunner]:
        """Create a lightweight executable graph."""

        return {
            "supervisor": SupervisorAgent().run,
            "researcher": ResearcherAgent().run,
            "analyst": AnalystAgent().run,
            "writer": WriterAgent().run,
            "critic": CriticAgent().run,
        }

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""

        settings = get_settings()
        graph = self.build()

        while state.iteration < settings.max_iterations + 4:
            with trace_span("supervisor", {"iteration_before": state.iteration}) as span:
                state = graph["supervisor"](state)
            state.add_trace_event(
                "span.completed",
                {
                    "name": span["name"],
                    "duration_seconds": span["duration_seconds"],
                    "status": span["status"],
                    "attributes": span["attributes"],
                },
            )
            route = state.route_history[-1]
            if route == "done":
                break
            runner = graph.get(route)
            if runner is None:
                raise AgentExecutionError(f"Unknown route selected by supervisor: {route}")
            try:
                with trace_span(route, {"iteration": state.iteration}) as span:
                    state = runner(state)
                state.add_trace_event(
                    "span.completed",
                    {
                        "name": span["name"],
                        "duration_seconds": span["duration_seconds"],
                        "status": span["status"],
                        "attributes": span["attributes"],
                    },
                )
            except Exception as exc:  # pragma: no cover - defensive orchestration path
                state.errors.append(f"{route}: {exc}")
                state.add_trace_event("workflow.error", {"route": route, "error": str(exc)})
                if route == "writer" and state.final_answer:
                    break
                if len(state.errors) >= 3:
                    raise AgentExecutionError(
                        "Workflow failed after repeated agent errors: "
                        f"{state.errors}"
                    ) from exc

        if not state.final_answer:
            raise AgentExecutionError("Workflow finished without producing a final answer.")
        return state

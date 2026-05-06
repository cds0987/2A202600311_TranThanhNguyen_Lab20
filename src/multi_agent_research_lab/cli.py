"""Command-line entrypoint for the lab starter."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import (
    render_benchmark_summary,
    render_requirements_report,
)
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import build_trace_artifact
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline."""

    _init()
    state = run_baseline_query(query)
    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    result = run_multi_agent_query(query)
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    query: Annotated[
        str,
        typer.Option(
            "--query",
            "-q",
            help="Research query used for baseline and multi-agent benchmarking",
        ),
    ] = "Research GraphRAG state-of-the-art and write a 500-word summary for technical learners.",
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for generated artifacts"),
    ] = Path("reports"),
) -> None:
    """Run the benchmark and write the required report artifacts."""

    _init()
    store = LocalArtifactStore(output_dir)

    baseline_state, baseline_metrics = run_benchmark("baseline", query, run_baseline_query)
    multi_state, multi_metrics = run_benchmark("multi-agent", query, run_multi_agent_query)

    benchmark_payload = {
        "query": query,
        "baseline_metrics": baseline_metrics.model_dump(),
        "multi_agent_metrics": multi_metrics.model_dump(),
        "baseline_final_answer": baseline_state.final_answer,
        "multi_agent_final_answer": multi_state.final_answer,
        "multi_agent_routes": multi_state.route_history,
        "multi_agent_sources": [source.model_dump() for source in multi_state.sources],
        "critic_notes": multi_state.critic_notes,
        "baseline_tokens": {
            "input": baseline_state.total_input_tokens,
            "output": baseline_state.total_output_tokens,
            "cost_usd": baseline_state.total_cost_usd,
        },
        "multi_agent_tokens": {
            "input": multi_state.total_input_tokens,
            "output": multi_state.total_output_tokens,
            "cost_usd": multi_state.total_cost_usd,
        },
        "retrieval_modes": sorted(
            {
                source.metadata.get("retrieval", "unknown")
                for source in multi_state.sources
            }
        ),
    }
    store.write_json("benchmark_results.json", benchmark_payload)
    store.write_text(
        "benchmark_report.md",
        render_benchmark_summary(query, baseline_metrics, multi_metrics, multi_state),
    )
    store.write_text(
        "requirements_report.md",
        render_requirements_report(baseline_metrics, multi_metrics, multi_state),
    )
    store.write_json("trace_baseline.json", build_trace_artifact(baseline_state))
    store.write_json("trace_multi_agent.json", build_trace_artifact(multi_state))
    console.print(Panel.fit(f"Artifacts written to {output_dir}", title="Benchmark Complete"))


def run_baseline_query(query: str) -> ResearchState:
    """Execute the single-agent baseline and return shared state."""

    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    llm_client = LLMClient()
    response = llm_client.complete(
        system_prompt=(
            "You are a single-agent research assistant. Answer the user's "
            "query directly and clearly for technical learners. Use numbered "
            "inline citations like [1] when source material is provided."
        ),
        user_prompt=f"Query: {query}\nAudience: {request.audience}",
    )
    state.final_answer = response.content
    state.record_usage(response.input_tokens, response.output_tokens, response.cost_usd)
    return state


def run_multi_agent_query(query: str) -> ResearchState:
    """Execute the multi-agent workflow and return shared state."""

    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


if __name__ == "__main__":
    app()

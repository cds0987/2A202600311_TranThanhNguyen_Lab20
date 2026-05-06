# Design Template

## Problem

Build a research assistant that takes an open-ended technical question, gathers relevant sources,
distills the evidence, and produces a final answer with references and review notes.

## Why multi-agent?

Single-agent prompting is fast, but it blends retrieval, analysis, synthesis, and review into one
step. Splitting responsibilities across agents makes the workflow easier to debug, lets us inspect
handoffs in shared state, and gives us a clearer place to measure quality, cost, and failure modes.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Decide the next step and stop condition | Request, state completeness, iteration count | Next route | Loops too long or routes to the wrong worker |
| Researcher | Gather sources and summarize findings | User query, max source count | `sources`, `research_notes` | Weak retrieval or low-quality notes |
| Analyst | Turn notes into structured insights | `research_notes`, `sources` | `analysis_notes` | Missed contradictions or shallow analysis |
| Writer | Produce the user-facing answer | Query, research notes, analysis notes, citations | `final_answer` | Overconfident synthesis or weak citation use |
| Critic | Review answer quality and support | Final answer, sources | `critic_notes` | Flags issues without enough specificity |

## Shared state

- `request`: source-of-truth for the user query and audience.
- `iteration` and `route_history`: make routing observable and stop infinite loops.
- `sources`: preserve retrieved evidence for later citations.
- `research_notes`: handoff from researcher to analyst and writer.
- `analysis_notes`: handoff from analyst to writer.
- `final_answer`: user-facing deliverable.
- `critic_notes`: post-write quality review.
- `agent_results`: compact per-agent outputs for auditing.
- `trace`: step-level observability payloads.
- `errors`: collect failures without losing the whole run immediately.
- `total_input_tokens`, `total_output_tokens`, `total_cost_usd`: benchmark support.

## Routing policy

`supervisor -> researcher -> analyst -> writer -> critic -> done`

Conditional policy:
- If sources or research notes are missing, run `researcher`.
- Else if analysis notes are missing, run `analyst`.
- Else if final answer is missing, run `writer`.
- Else if critic notes are missing, run `critic`.
- Else stop with `done`.
- If the max iteration budget is nearly exhausted, prefer finishing with `writer` or `critic`.

## Guardrails

- Max iterations: `MAX_ITERATIONS`, default `6`
- Timeout: `TIMEOUT_SECONDS`, default `60`
- Retry: LLM requests retry up to 3 times with exponential backoff
- Fallback: deterministic local mock LLM output when OpenAI credentials or SDK are unavailable
- Validation: Pydantic schemas for request, sources, agent results, and metrics

## Benchmark plan

- Query: "Explain multi-agent systems and when they outperform a single agent"
  Metric: latency, cost, quality, citation coverage, failure rate
  Expected outcome: multi-agent answer is slower but better structured and better grounded
- Query: "Summarize GraphRAG state of the art for technical learners"
  Metric: quality and citation coverage
  Expected outcome: researcher plus analyst produces stronger synthesis than baseline
- Query: "Compare orchestration patterns for agent workflows"
  Metric: completeness and review quality
  Expected outcome: critic notes identify unsupported claims or missing caveats

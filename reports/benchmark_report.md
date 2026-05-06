# Benchmark Report

## Query

`Research GraphRAG state-of-the-art and write a 500-word summary for technical learners.`

## Benchmark results

| Run | Latency (s) | Cost (USD) | Quality | Citation Coverage | Failure Rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| baseline | 8.81 | 0.0000 | 1.0 |  | 0% | iterations=0, sources=0, errors=0 |
| `baseline` rubric proxy |  |  |  |  |  | grounding=0.00, analysis=0.00, citations=0.00, completeness=1.00, traceability=0.00 |
| multi-agent | 40.96 | 0.0005 | 8.8 | 40% | 0% | iterations=5, sources=5, errors=0 |
| `multi-agent` rubric proxy |  |  |  |  |  | grounding=2.00, analysis=2.00, citations=0.80, completeness=2.00, traceability=2.00 |

## Workflow used

`researcher -> analyst -> writer -> critic -> done`

## Retrieval modes

local-seed

## Sources retrieved

- [1] Microsoft GraphRAG repository (local-seed)
- [2] OpenAI docs: Orchestration and handoffs (local-seed)
- [3] OpenAI API docs: Responses API (local-seed)
- [4] LangSmith observability quickstart (local-seed)
- [5] LangGraph concepts overview (local-seed)

## Failure mode and fix

The main failure mode is weak inline citation use. The next fix is to tighten writer prompting so each major claim maps to one numbered source.

## Exit ticket

1. Use multi-agent when the task benefits from separate retrieval, analysis, writing, and review steps with observable handoffs.
2. Avoid multi-agent for short or latency-sensitive tasks where orchestration overhead outweighs quality gains.

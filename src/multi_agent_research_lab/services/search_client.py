"""Search client abstraction for ResearcherAgent."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.services.fetch_ai_articles import fetch_feed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SeedSource:
    title: str
    url: str
    snippet: str
    tags: tuple[str, ...]


_SEED_SOURCES: tuple[_SeedSource, ...] = (
    _SeedSource(
        title="Microsoft GraphRAG repository",
        url="https://github.com/microsoft/graphrag",
        snippet=(
            "Reference implementation and documentation for graph-based "
            "retrieval-augmented generation workflows."
        ),
        tags=("graphrag", "graph", "retrieval", "rag", "microsoft"),
    ),
    _SeedSource(
        title="OpenAI API docs: Responses API",
        url="https://platform.openai.com/docs/api-reference/responses",
        snippet="The Responses API provides a unified interface for text generation and tool use.",
        tags=("openai", "responses", "api", "agent", "tools"),
    ),
    _SeedSource(
        title="OpenAI docs: Orchestration and handoffs",
        url="https://platform.openai.com/docs/guides/agents/orchestration",
        snippet=(
            "Agentic systems benefit from clear role boundaries, shared "
            "state, and controlled handoffs."
        ),
        tags=("openai", "orchestration", "handoff", "multi-agent", "supervisor"),
    ),
    _SeedSource(
        title="LangGraph concepts overview",
        url="https://langchain-ai.github.io/langgraph/concepts/",
        snippet=(
            "Graph-based orchestration helps model multi-step workflows "
            "with conditional routing."
        ),
        tags=("langgraph", "graph", "workflow", "routing", "state"),
    ),
    _SeedSource(
        title="Anthropic: Building effective agents",
        url="https://www.anthropic.com/engineering/building-effective-agents",
        snippet=(
            "Use multiple agents only when role specialization creates "
            "measurable quality or reliability gains."
        ),
        tags=("agents", "research", "analysis", "quality", "specialization"),
    ),
    _SeedSource(
        title="LangSmith observability quickstart",
        url="https://docs.smith.langchain.com/",
        snippet=(
            "Trace agent workflows, inspect handoffs, and compare runs "
            "with consistent observability metadata."
        ),
        tags=("benchmark", "trace", "observability", "evaluation", "langsmith"),
    ),
)


class SearchClient:
    """Google News RSS search with a local fallback set."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""

        combined: list[SourceDocument] = []
        seen_keys: set[str] = set()
        for candidate_query in self._expand_queries(query):
            for document in self._search_google_news(
                query=candidate_query,
                max_results=max_results,
            ):
                key = self._document_key(document)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                combined.append(document)
                if len(combined) >= max_results:
                    return combined

        for document in self._search_seed_sources(query=query, max_results=max_results):
            key = self._document_key(document)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            combined.append(document)
            if len(combined) >= max_results:
                break

        return combined

    def _search_google_news(self, query: str, max_results: int) -> list[SourceDocument]:
        try:
            articles = fetch_feed(query)
        except Exception as exc:  # pragma: no cover - network-dependent fallback path
            logger.warning("Google News RSS search failed for query %r: %s", query, exc)
            return []

        results: list[SourceDocument] = []
        for article in articles[:max_results]:
            snippet = article.get("summary") or article.get("source") or "No summary available."
            results.append(
                SourceDocument(
                    title=article.get("title", "Untitled article"),
                    url=article.get("url"),
                    snippet=snippet,
                    metadata={
                        "source": article.get("source"),
                        "published_at": article.get("published_at"),
                        "query": article.get("query", query),
                        "retrieval": "google-news-rss",
                    },
                )
            )
        return results

    def _search_seed_sources(self, query: str, max_results: int) -> list[SourceDocument]:
        query_terms = {term.lower() for term in query.replace("-", " ").split() if term.strip()}
        ranked: list[tuple[int, _SeedSource]] = []
        for source in _SEED_SOURCES:
            haystack = " ".join((source.title, source.snippet, " ".join(source.tags))).lower()
            overlap = sum(1 for term in query_terms if term in haystack)
            ranked.append((overlap, source))

        ranked.sort(key=lambda item: (item[0], item[1].title), reverse=True)
        selected = [source for score, source in ranked if score > 0][:max_results]
        if len(selected) < max_results:
            already_selected = {source.title for source in selected}
            for _, source in ranked:
                if source.title in already_selected:
                    continue
                selected.append(source)
                already_selected.add(source.title)
                if len(selected) >= max_results:
                    break

        return [
            SourceDocument(
                title=source.title,
                url=source.url,
                snippet=source.snippet,
                metadata={"tags": list(source.tags), "retrieval": "local-seed"},
            )
            for source in selected
        ]

    def _expand_queries(self, query: str) -> list[str]:
        query = query.strip()
        expansions = [
            query,
            f"{query} research paper",
            f"{query} benchmark",
            f"{query} technical overview",
        ]
        seen: set[str] = set()
        unique_expansions: list[str] = []
        for candidate in expansions:
            normalized = candidate.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_expansions.append(candidate)
        return unique_expansions

    def _document_key(self, document: SourceDocument) -> str:
        return (document.url or document.title).strip().lower()

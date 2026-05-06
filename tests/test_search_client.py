from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.services.search_client import SearchClient


def test_search_client_merges_live_results_with_seed_fallback(monkeypatch) -> None:
    client = SearchClient()

    def fake_live_search(query: str, max_results: int) -> list[SourceDocument]:
        if query.endswith("research paper"):
            return [
                SourceDocument(
                    title="Live GraphRAG source",
                    url="https://example.com/live",
                    snippet="Live result",
                    metadata={"retrieval": "google-news-rss"},
                )
            ]
        return []

    monkeypatch.setattr(client, "_search_google_news", fake_live_search)

    results = client.search("GraphRAG", max_results=3)

    assert any(result.title == "Live GraphRAG source" for result in results)
    assert len(results) == 3
    assert any(result.metadata.get("retrieval") == "local-seed" for result in results)

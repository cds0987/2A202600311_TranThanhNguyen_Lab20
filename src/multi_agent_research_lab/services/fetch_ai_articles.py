import csv
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path


GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
REQUEST_HEADERS = {
    "User-Agent": "Lab19-GraphRAG-Fetcher/1.0 (educational use)"
}


def clean_text(value: str) -> str:
    return " ".join((value or "").split())


def fetch_feed(query: str) -> list[dict]:
    params = {
        "q": query,
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    }
    url = f"{GOOGLE_NEWS_RSS}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers=REQUEST_HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
        xml_data = response.read()

    root = ET.fromstring(xml_data)
    records: list[dict] = []
    for item in root.findall("./channel/item"):
        title = clean_text(item.findtext("title", default=""))
        link = clean_text(item.findtext("link", default=""))
        pub_date = clean_text(item.findtext("pubDate", default=""))
        description = clean_text(item.findtext("description", default=""))

        source_element = item.find("source")
        source = clean_text(source_element.text if source_element is not None else "")

        try:
            published_at = parsedate_to_datetime(pub_date).isoformat()
        except Exception:
            published_at = pub_date

        records.append(
            {
                "title": title,
                "url": link,
                "published_at": published_at,
                "source": source,
                "query": query,
                "summary": description,
            }
        )

    return records


def fetch_articles(limit: int = 100) -> list[dict]:
    queries = [
        "artificial intelligence",
        "machine learning",
        "\"generative AI\"",
        "\"large language model\"",
        "OpenAI OR Anthropic OR Google DeepMind OR Meta AI",
        "AI startup OR AI research OR AI model",
    ]

    results: list[dict] = []
    seen_urls: set[str] = set()

    while len(results) < limit:
        added_this_round = 0
        for query in queries:
            for article in fetch_feed(query):
                url = article["url"]
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                results.append(article)
                added_this_round += 1
                if len(results) >= limit:
                    return results[:limit]
            time.sleep(0.5)

        if added_this_round == 0:
            break

    return results[:limit]


def write_outputs(articles: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "ai_articles.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(articles, handle, ensure_ascii=False, indent=2)

    jsonl_path = output_dir / "ai_articles.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for article in articles:
            handle.write(json.dumps(article, ensure_ascii=False) + "\n")

    csv_path = output_dir / "ai_articles.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["title", "source", "published_at", "query", "url", "summary"],
        )
        writer.writeheader()
        writer.writerows(articles)


def main() -> int:
    limit = 100
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])

    articles = fetch_articles(limit=limit)
    output_dir = Path("data")
    write_outputs(articles, output_dir)

    print(f"Fetched {len(articles)} AI articles into {output_dir}.")
    if articles:
        print(f"First article: {articles[0]['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

.PHONY: install test lint format typecheck run-baseline run-multi clean

install:
	python -m pip install -e ".[dev,llm]"

test:
	python -m pytest

lint:
	python -m ruff check src tests

format:
	python -m ruff format src tests

typecheck:
	python -m mypy --cache-dir "$${MYPY_CACHE_DIR:-.mypy_cache}" src

run-baseline:
	python -m multi_agent_research_lab.cli baseline --query "Research GraphRAG state-of-the-art"

run-multi:
	python -m multi_agent_research_lab.cli multi-agent --query "Research GraphRAG state-of-the-art"

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build *.egg-info

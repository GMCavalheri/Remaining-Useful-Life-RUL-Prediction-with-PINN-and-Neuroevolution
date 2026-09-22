.PHONY: sync lint test

sync:
	uv sync --extra dev --extra evolution

lint:
	uv run ruff check src scripts tests

test:
	uv run pytest -q

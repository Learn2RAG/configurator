test:
	yamllint .
	uv sync --locked --all-extras --dev
	uv run mypy --no-incremental --warn-unused-configs
	uv run ruff check --config ./pyproject.toml learn2rag
	uv run pytest

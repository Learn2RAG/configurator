test:
	uv sync --locked --all-extras --dev
	uv run mypy --no-incremental --warn-unused-configs
	uv run pytest

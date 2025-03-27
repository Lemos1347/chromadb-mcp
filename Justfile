_default:
  @just --list

run:
  @uv run --no-dev main.py

dev:
  @npx @modelcontextprotocol/inspector uv run --no-dev main.py

setup:
  @uv run --only-dev ./scripts/files_ingestion/main.py
  @uv sync --no-dev
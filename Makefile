.PHONY: sync test test-cov cov-report cov lint-typing lint-style lint-fmt lint-all build publish help

help:
	@echo "Available make targets:"
	@echo "  make sync          - Sync project and development dependencies"
	@echo "  make test          - Run tests"
	@echo "  make test-cov      - Run tests with coverage collection"
	@echo "  make cov-report    - Generate the XML coverage report"
	@echo "  make cov           - Run coverage collection and reporting"
	@echo "  make lint-typing   - Type check with mypy"
	@echo "  make lint-style    - Check Ruff and Black formatting"
	@echo "  make lint-fmt      - Format with Black and apply Ruff fixes"
	@echo "  make lint-all      - Format, lint, and type check"
	@echo "  make build         - Build wheel and source distributions"
	@echo "  make publish       - Publish distributions with uv"

sync:
	uv sync --all-groups

test:
	uv run pytest tests

test-cov:
	uv run coverage run -m pytest tests

cov-report:
	uv run coverage combine
	uv run coverage xml

cov: test-cov cov-report

lint-typing:
	uv run mypy --install-types --non-interactive src/llm_blender tests

lint-style:
	uv run ruff check .
	uv run black --check --diff .

lint-fmt:
	uv run black .
	uv run ruff check --fix --unsafe-fixes .
	$(MAKE) lint-style

lint-all: lint-fmt lint-typing

build:
	uv build

publish:
	uv publish

.PHONY: help install dev-install test demo lint typecheck clean docker-build docker-up docker-down

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install OffScan in editable mode
	$(PIP) install -e .

dev-install: ## Install with development dependencies
	$(PIP) install -e ".[dev]"

test: ## Run the test suite with coverage
	$(PYTHON) -m pytest

demo: ## Run the OffScan demo on a bundled sample image
	$(PYTHON) -m offscan.cli demo

lint: ## Run ruff linter and formatter check
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

typecheck: ## Run mypy type checking
	$(PYTHON) -m mypy src/offscan

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info src/*.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov .coverage coverage.xml

docker-build: ## Build the Docker image
	docker compose build

docker-up: ## Start the app via docker-compose
	docker compose up --build

docker-down: ## Stop the docker-compose stack
	docker compose down

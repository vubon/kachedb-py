# Strict Python environment bound to experiments/.venv
VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy

.PHONY: help venv-check test test-unit coverage lint format typecheck build clean

help:
	@echo "KacheDB Python SDK Makefile"
	@echo "  make test        - Run all tests (requires running kachedb-server)"
	@echo "  make test-unit   - Run unit tests without live daemon"
	@echo "  make coverage    - Run tests with code coverage report"
	@echo "  make lint        - Run ruff check and format check"
	@echo "  make format      - Autoformat code with ruff"
	@echo "  make typecheck   - Run mypy strict type checking"
	@echo "  make build       - Build wheel and sdist packages"

venv-check:
	@if [ ! -f $(PYTHON) ]; then \
		echo "Error: $(PYTHON) not found. Ensure experiments/.venv exists."; \
		exit 1; \
	fi

test: venv-check
	$(PYTEST) tests/ -v

test-unit: venv-check
	$(PYTEST) tests/ -m "not integration" -v

coverage: venv-check
	$(PYTEST) --cov=src/kachedb --cov-report=term-missing tests/

lint: venv-check
	$(RUFF) check src/ tests/
	$(RUFF) format --check src/ tests/

format: venv-check
	$(RUFF) format src/ tests/

typecheck: venv-check
	$(MYPY) src/kachedb/

build: venv-check
	$(PYTHON) -m build

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage

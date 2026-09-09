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
	@echo "  make test        - Run tests (auto-detects live server for integration tests)"
	@echo "  make test-unit   - Run unit tests only (no daemon needed)"
	@echo "  make test-all    - Force run all tests including integration tests"
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
	@if nc -z 127.0.0.1 6379 2>/dev/null; then \
		echo "⚡ Live KacheDB daemon detected on 127.0.0.1:6379. Running full test suite (unit + integration)..."; \
		$(PYTEST) tests/ -v; \
	else \
		echo "ℹ️  No live KacheDB daemon on 127.0.0.1:6379. Running unit tests only (start server to include integration tests)..."; \
		$(PYTEST) tests/ -m "not integration" -v; \
	fi

test-unit: venv-check
	$(PYTEST) tests/ -m "not integration" -v

test-all: venv-check
	@SERVER_PID=""; \
	if ! nc -z 127.0.0.1 6379 2>/dev/null; then \
		if [ -f ../kachedb/target/release/kachedb-server ]; then \
			echo "⚡ Auto-launching local kachedb-server on 127.0.0.1:6379..."; \
			../kachedb/target/release/kachedb-server >/dev/null 2>&1 & \
			SERVER_PID=$$!; \
			sleep 1; \
		else \
			echo "❌ Error: kachedb-server binary not found and port 6379 is not open."; \
			exit 1; \
		fi; \
	fi; \
	$(PYTEST) tests/ -v; \
	STATUS=$$?; \
	if [ -n "$$SERVER_PID" ]; then \
		echo "🛑 Stopping temporary kachedb-server (PID $$SERVER_PID)..."; \
		kill $$SERVER_PID 2>/dev/null || true; \
	fi; \
	exit $$STATUS

coverage: venv-check
	@if nc -z 127.0.0.1 6379 2>/dev/null; then \
		echo "⚡ Live KacheDB daemon detected on 127.0.0.1:6379. Running full coverage report..."; \
		$(PYTEST) --cov=src/kachedb --cov-report=term-missing tests/; \
	else \
		echo "ℹ️  No live KacheDB daemon on 127.0.0.1:6379. Running coverage report on unit tests..."; \
		$(PYTEST) --cov=src/kachedb --cov-report=term-missing tests/ -m "not integration"; \
	fi

coverage-all: venv-check
	@SERVER_PID=""; \
	if ! nc -z 127.0.0.1 6379 2>/dev/null; then \
		if [ -f ../kachedb/target/release/kachedb-server ]; then \
			echo "⚡ Auto-launching local kachedb-server on 127.0.0.1:6379..."; \
			../kachedb/target/release/kachedb-server >/dev/null 2>&1 & \
			SERVER_PID=$$!; \
			sleep 1; \
		else \
			echo "❌ Error: kachedb-server binary not found and port 6379 is not open."; \
			exit 1; \
		fi; \
	fi; \
	$(PYTEST) --cov=src/kachedb --cov-report=term-missing tests/; \
	STATUS=$$?; \
	if [ -n "$$SERVER_PID" ]; then \
		echo "🛑 Stopping temporary kachedb-server (PID $$SERVER_PID)..."; \
		kill $$SERVER_PID 2>/dev/null || true; \
	fi; \
	exit $$STATUS

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

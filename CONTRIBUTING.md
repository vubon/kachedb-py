# Contributing to KacheDB Python SDK

Thank you for your interest in contributing to `kachedb-py`! We welcome contributions from the community.

---

## 🛠️ Development Setup

### 1. Clone & Environment
```bash
git clone https://github.com/vubon/kachedb-py.git
cd kachedb-py

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev,torch,vllm]"
```

---

## 🧪 Testing & Code Quality

Before submitting a Pull Request, make sure all tests, linter checks, and type checks pass:

### 1. Run Unit Tests
```bash
pytest tests/ -v --ignore=tests/test_integration.py
```

### 2. Run Integration Tests (Requires running KacheDB server on 127.0.0.1:6379)
```bash
pytest tests/test_integration.py -v
```

### 3. Linting & Formatting
```bash
ruff check src/ tests/
ruff format --check src/ tests/
```

### 4. Type Checking
```bash
mypy src/kachedb/
```

---

## 📦 Pull Request Guidelines

1. **Create a branch:** Use descriptive branch names (e.g. `feat/paged-attention-stream`, `fix/async-pool-timeout`).
2. **Write tests:** Every new feature or bug fix must include corresponding tests in `tests/`.
3. **Update CHANGELOG.md:** Add your change under the `[Unreleased]` or current alpha release section.
4. **Follow Semantic Versioning:** Breaking changes require explicit RFC discussion in GitHub Issues.

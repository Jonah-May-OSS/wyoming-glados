# Tests

This directory contains automated tests for the wyoming-glados project.

## Running Tests

### Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```

### Run All Tests

```bash
pytest
```

### Run Tests with Coverage

```bash
pytest --cov=server --cov-report=html
```

View the coverage report by opening `htmlcov/index.html` in your browser.

### Run Specific Test Files

```bash
pytest tests/test_sentence_boundary.py
pytest tests/test_process.py
pytest tests/test_handler.py
```

### Run Tests in Verbose Mode

```bash
pytest -v
```

## Linting and Formatting

### Check Code with Ruff

```bash
ruff check .
```

### Auto-fix Linting Issues

```bash
ruff check --fix .
```

### Format Code

```bash
ruff format .
```

### Check Formatting

```bash
ruff format --check .
```

## Test Structure

- `test_sentence_boundary.py`: Tests for sentence boundary detection and asterisk removal
- `test_process.py`: Tests for GLaDOS process management (caching, async operations)
- `test_handler.py`: Tests for Wyoming event handler logic (streaming, audio events)
- `conftest.py`: Pytest configuration and shared fixtures

## Note on Dependencies

Some tests require the `regex` module which is a production dependency. The test suite is designed to gracefully skip tests that require unavailable dependencies.

The tests mock heavy dependencies like `wyoming`, `gladostts`, and CUDA-related libraries to allow testing without a full GPU setup.

## Continuous Integration

Tests and linting checks run automatically on pull requests via GitHub Actions. See `.github/workflows/pytest-ruff.yml` for the CI configuration.

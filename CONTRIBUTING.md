# Contributing to OffScan

Thank you for your interest in improving OffScan! This guide will help you get set up
and submit your first contribution.

## Quick Start

```bash
# 1. Fork & clone
git clone https://github.com/mrsmallflame-ai/OffScan.git
cd OffScan

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3. Install with dev dependencies
make dev-install

# 4. Run the test suite
make test
```

## System Dependencies

OffScan requires **Tesseract OCR** installed on your system:

| OS | Command |
|---|---|
| Ubuntu / Debian | `sudo apt-get install tesseract-ocr` |
| macOS (Homebrew) | `brew install tesseract` |
| Windows | Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) |

For a fully automated setup, run `scripts/install.sh`.

## Development Workflow

### Branch naming

Use descriptive branch names prefixed by type:

- `feat/add-batch-mode`
- `fix/ocr-garbage-chars`
- `docs/update-architecture`
- `chore/upgrade-deps`

### Commit messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add multi-page PDF support
fix: correct orientation detection for 90° rotation
docs: update pipeline architecture diagram
refactor: extract preprocessing into separate module
test: add parametrized OCR accuracy tests
```

### Code style

- **Formatter / linter:** Ruff (`make lint`)
- **Line length:** 100 characters
- **Type checker:** mypy (`make typecheck`)
- Run `ruff format` before committing to auto-fix style issues

### Pull Requests

1. Open an issue describing your change (for non-trivial work)
2. Create a branch from `main`
3. Write tests for new functionality
4. Ensure `make test` and `make lint` pass
5. Open a PR with a clear description and link to the issue
6. Request review from a maintainer

### Testing

- Place unit tests in `tests/` mirroring the `src/offscan/` structure
- Use `pytest` fixtures for shared setup
- Aim for >80% coverage on new code
- Integration tests that require Tesseract should be marked `@pytest.mark.integration`

```bash
make test                          # full suite
pytest tests/test_preprocess.py    # single file
pytest -k "ocr"                    # by keyword
```

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for a full description of the
pipeline and how modules fit together.

## Privacy

OffScan is designed to work **100% offline**. Never introduce code that makes
network requests, telemetry, or sends data to external services. See
[`docs/PRIVACY.md`](docs/PRIVACY.md) for the full privacy policy.

## Reporting Issues

- **Bugs:** Open an issue with OS, Python version, and steps to reproduce
- **Feature requests:** Describe the use case and expected behavior
- **Security issues:** Email contact@mrsmallflame.ai (do NOT open a public issue)

## Code of Conduct

Be respectful and constructive. Harassment or discrimination of any kind will
not be tolerated.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License.

# Contributing to Sandshrew

Thank you for your interest in contributing to Sandshrew! We welcome contributions from the community. This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a new branch for your feature or fix: `git checkout -b feature/your-feature-name`

## Development Setup

Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Code Quality

Before submitting a pull request, ensure your code passes quality checks:

```bash
# Run linting
ruff check .

# Format code
ruff format .

# Run tests
pytest tests/
```

## Making Changes

- Keep changes focused and atomic
- Write clear commit messages
- Add tests for new functionality
- Update documentation as needed
- Follow the existing code style and patterns

## Testing

All new features should include tests. Run the test suite with:
```bash
pytest tests/
```

## Submitting Changes

1. Push your branch to your fork
2. Open a pull request against the main repository
3. Provide a clear description of your changes
4. Reference any related issues

## Code Style

- Follow PEP 8 conventions
- Use type hints where applicable
- Write docstrings for functions and classes
- Keep functions focused and maintainable

## Questions?

Feel free to open an issue for questions or discussions about contributing.

Thank you for helping make Sandshrew better!

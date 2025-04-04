# GenerAI Tests

This directory contains unit tests for the GenerAI project. The tests are organized by module and use pytest as the testing framework.

## Test Structure

- `test_openai_client.py`: Tests for the OpenAI API client
- `test_cache_manager.py`: Tests for the caching functionality
- `test_medium_publisher.py`: Tests for Medium publishing
- `test_web_search.py`: Tests for web search functionality
- `test_feedback_manager.py`: Tests for the feedback loop mechanism
- `test_article_pipeline.py`: Tests for the modular pipeline approach
- `test_utils.py`: Tests for utility functions
- `test_generai.py`: Tests for the main entry point and CLI

## Running Tests

### Prerequisites

Install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

### Running All Tests

```bash
pytest
```

### Running Specific Tests

```bash
# Run tests for a specific module
pytest tests/test_openai_client.py

# Run a specific test class
pytest tests/test_openai_client.py::TestOpenAIClient

# Run a specific test method
pytest tests/test_openai_client.py::TestOpenAIClient::test_initialization
```

### Running with Coverage

```bash
pytest --cov=src tests/
```

## Test Design

The tests use mocking to avoid making actual API calls to external services like OpenAI, Medium, and Tavily. This ensures that tests can run quickly and reliably without depending on external services or API keys.

Common test fixtures are defined in `conftest.py` and can be used across multiple test files.

## Adding New Tests

When adding new functionality to the project, please also add corresponding tests. Follow these guidelines:

1. Create a new test file if testing a new module
2. Use the existing test structure as a template
3. Use mocking for external dependencies
4. Ensure tests are independent and can run in any order
5. Add appropriate assertions to verify functionality
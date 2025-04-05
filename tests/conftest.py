#!/usr/bin/env python3
"""
Test fixtures for GenerAI tests.

This module contains pytest fixtures that can be shared across multiple test files.
"""

import pytest
from unittest.mock import MagicMock, patch
import os
from pathlib import Path

from src.openai_client import OpenAIClient
from src.cache_manager import CacheManager
from src.medium_publisher import MediumPublisher
from src.web_search import WebSearchManager
from src.article_pipeline import ArticlePipeline


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock_client = MagicMock(spec=OpenAIClient)
    return mock_client


@pytest.fixture
def mock_cache_manager():
    """Create a mock cache manager."""
    mock_cache = MagicMock(spec=CacheManager)
    return mock_cache


@pytest.fixture
def mock_medium_publisher():
    """Create a mock Medium publisher."""
    mock_publisher = MagicMock(spec=MediumPublisher)
    return mock_publisher


@pytest.fixture
def mock_web_search():
    """Create a mock web search manager."""
    mock_search = MagicMock(spec=WebSearchManager)
    # Set default behavior
    mock_search.is_available.return_value = True
    return mock_search


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary data directory for tests."""
    # Create subdirectories
    ideas_dir = tmp_path / "ideas"
    article_queue_dir = tmp_path / "article_queue"
    projects_dir = tmp_path / "projects"
    feedback_dir = tmp_path / "feedback"
    
    # Create directories
    ideas_dir.mkdir()
    article_queue_dir.mkdir()
    projects_dir.mkdir()
    feedback_dir.mkdir()
    
    return tmp_path


@pytest.fixture
def sample_article_data():
    """Sample article data for tests."""
    return {
        "title": "Test Article",
        "content": "This is a test article content.\n\nIt has multiple paragraphs.\n\nAnd some formatting **bold** and *italic*.",
        "metadata": {
            "topic": "Test Topic",
            "tone": "informative",
            "length": "medium",
            "audience": "general",
            "keywords": ["test", "article", "sample"]
        }
    }


@pytest.fixture
def sample_idea_data():
    """Sample article idea data for tests."""
    return [
        {
            "id": "idea1",
            "title": "Test Idea 1",
            "summary": "Summary of test idea 1",
            "audience": "Audience 1",
            "keywords": ["keyword1", "keyword2"],
            "created_at": "2023-01-01T12:00:00"
        },
        {
            "id": "idea2",
            "title": "Test Idea 2",
            "summary": "Summary of test idea 2",
            "audience": "Audience 2",
            "keywords": ["keyword3", "keyword4"],
            "created_at": "2023-01-02T12:00:00"
        }
    ]


@pytest.fixture
def mock_environment():
    """Set up mock environment variables for tests."""
    original_environ = os.environ.copy()
    
    # Set test environment variables
    os.environ["OPENAI_API_KEY"] = "test_openai_key"
    os.environ["OPENAI_MODEL"] = "gpt-4o"
    os.environ["MEDIUM_INTEGRATION_TOKEN"] = "test_medium_token"
    os.environ["MEDIUM_AUTHOR_ID"] = "test_author_id"
    os.environ["TAVILY_API_KEY"] = "test_tavily_key"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_environ)
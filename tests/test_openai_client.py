#!/usr/bin/env python3
"""
Tests for the OpenAI client module.

These tests verify the functionality of the OpenAIClient class,
including article generation and API interactions.
"""

import unittest
from unittest.mock import patch, MagicMock
import json
from pathlib import Path

from src.openai_client import OpenAIClient
from src.cache_manager import CacheManager


class TestOpenAIClient(unittest.TestCase):
    """Test cases for the OpenAIClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock API key and model
        self.api_key = "test_api_key"
        self.model = "gpt-4"
        
        # Create a mock cache manager
        self.mock_cache_manager = MagicMock(spec=CacheManager)
        
        # Patch the OpenAI client initialization
        self.openai_patcher = patch('openai.OpenAI')
        self.mock_openai = self.openai_patcher.start()
        self.mock_client = MagicMock()
        self.mock_openai.return_value = self.mock_client
        
        # Create the OpenAIClient instance with caching disabled for most tests
        self.client = OpenAIClient(api_key=self.api_key, model=self.model, use_cache=False)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.openai_patcher.stop()
    
    def test_initialization(self):
        """Test that the client initializes correctly."""
        self.assertEqual(self.client.api_key, self.api_key)
        self.assertEqual(self.client.model, self.model)
        self.assertEqual(self.client.use_cache, False)
        self.assertIsNone(self.client.cache_manager)
        
        # Test initialization with caching enabled
        with patch('src.openai_client.CacheManager') as mock_cache_cls:
            mock_cache_cls.return_value = self.mock_cache_manager
            client_with_cache = OpenAIClient(api_key=self.api_key, model=self.model, use_cache=True)
            
            self.assertEqual(client_with_cache.use_cache, True)
            self.assertIsNotNone(client_with_cache.cache_manager)
            mock_cache_cls.assert_called_once()
    
    def test_generate_article(self):
        """Test article generation functionality."""
        # Mock the OpenAI API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "TITLE: Test Article\n\nThis is a test article content."
        self.mock_client.chat.completions.create.return_value = mock_response
        
        # Call the method
        topic = "Test Topic"
        tone = "informative"
        length = "medium"
        outline = ["Section 1", "Section 2"]
        
        result = self.client.generate_article(topic=topic, tone=tone, length=length, outline=outline)
        
        # Verify the result
        self.assertIn("title", result)
        self.assertIn("content", result)
        self.assertEqual(result["title"], "Test Article")
        self.assertEqual(result["content"], "This is a test article content.")
        
        # Verify that the API was called with the correct parameters
        self.mock_client.chat.completions.create.assert_called_once()
        call_args = self.mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_args["model"], self.model)
        self.assertEqual(len(call_args["messages"]), 2)  # System prompt and user prompt
        
        # Verify that the user prompt contains the topic, tone, and length
        user_prompt = call_args["messages"][1]["content"]
        self.assertIn(topic, user_prompt)
        self.assertIn(tone, user_prompt)
        self.assertIn("1500-2000 words", user_prompt)  # Medium length
        
        # Verify that the outline is included in the prompt
        for section in outline:
            self.assertIn(section, user_prompt)
    
    def test_generate_article_with_cache(self):
        """Test article generation with caching."""
        # Create a client with caching enabled
        with patch('src.openai_client.CacheManager') as mock_cache_cls:
            mock_cache_cls.return_value = self.mock_cache_manager
            client_with_cache = OpenAIClient(api_key=self.api_key, model=self.model, use_cache=True)
        
        # Set up the cache miss scenario first
        self.mock_cache_manager.get.return_value = None
        
        # Mock the OpenAI API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "TITLE: Cached Article\n\nThis is a cached article content."
        self.mock_client.chat.completions.create.return_value = mock_response
        
        # Call the method
        topic = "Cache Test"
        result = client_with_cache.generate_article(topic=topic)
        
        # Verify cache was checked
        self.mock_cache_manager.get.assert_called_once()
        
        # Verify API was called (cache miss)
        self.mock_client.chat.completions.create.assert_called_once()
        
        # Verify result was cached
        self.mock_cache_manager.set.assert_called_once()
        
        # Reset mocks for cache hit scenario
        self.mock_cache_manager.get.reset_mock()
        self.mock_client.chat.completions.create.reset_mock()
        self.mock_cache_manager.set.reset_mock()
        
        # Set up cache hit
        cached_result = {"title": "Cached Article", "content": "This is a cached article content."}
        self.mock_cache_manager.get.return_value = cached_result
        
        # Call the method again
        result2 = client_with_cache.generate_article(topic=topic)
        
        # Verify cache was checked
        self.mock_cache_manager.get.assert_called_once()
        
        # Verify API was NOT called (cache hit)
        self.mock_client.chat.completions.create.assert_not_called()
        
        # Verify result was NOT cached again
        self.mock_cache_manager.set.assert_not_called()
        
        # Verify the result matches the cached result
        self.assertEqual(result2, cached_result)


if __name__ == "__main__":
    unittest.main()
#!/usr/bin/env python3
"""
Tests for the web search module.

These tests verify the functionality of the WebSearchManager class,
including searching the web using the Brave Search API (default) and Tavily API.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
from datetime import datetime

from src.web_search import WebSearchManager, BraveSearchManager, TavilySearchManager


class TestWebSearchManager(unittest.TestCase):
    """Test cases for the WebSearchManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock API key
        self.api_key = "test_api_key"
        
        # Patch the TavilyClient
        self.tavily_patcher = patch('src.web_search.TavilyClient')
        self.mock_tavily_cls = self.tavily_patcher.start()
        self.mock_tavily = MagicMock()
        self.mock_tavily_cls.return_value = self.mock_tavily
        
        # Patch requests.get for Brave Search API
        self.requests_patcher = patch('src.web_search.requests.get')
        self.mock_requests_get = self.requests_patcher.start()
        self.mock_response = MagicMock()
        self.mock_response.status_code = 200
        self.mock_requests_get.return_value = self.mock_response
        
        # Patch os.environ for environment variable tests
        self.env_patcher = patch.dict('os.environ', {
            "TAVILY_API_KEY": "env_api_key",
            "BRAVE_API_KEY": "env_api_key"
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.tavily_patcher.stop()
        self.requests_patcher.stop()
        self.env_patcher.stop()
    
    def test_initialization_with_api_key(self):
        """Test initialization with API key provided."""
        manager = WebSearchManager(api_key=self.api_key)
        
        # Check that the provider was created with the API key
        self.assertEqual(manager.provider.api_key, self.api_key)
        self.assertIsNotNone(manager.provider.client)
        # Default provider should be BraveSearchManager
        self.assertIsInstance(manager.provider, BraveSearchManager)
    
    def test_initialization_with_env_api_key(self):
        """Test initialization with API key from environment."""
        manager = WebSearchManager()
        
        # Check that the provider was created with the environment API key
        self.assertEqual(manager.provider.api_key, "env_api_key")
        self.assertIsNotNone(manager.provider.client)
    
    def test_initialization_without_api_key(self):
        """Test initialization without API key."""
        # Remove API key from environment
        with patch.dict('os.environ', {}, clear=True):
            manager = WebSearchManager()
            
            # Check that the provider was created without an API key
            self.assertIsNone(manager.provider.api_key)
            self.assertIsNone(manager.provider.client)
    
    def test_initialization_with_tavily_error(self):
        """Test handling of errors during TavilyClient initialization."""
        # Make TavilyClient raise an exception
        self.mock_tavily_cls.side_effect = Exception("Tavily Error")
        
        # Create manager with Tavily provider
        manager = WebSearchManager(api_key=self.api_key, provider="tavily")
        
        # Check that the provider was created with the API key but client is None
        self.assertEqual(manager.provider.api_key, self.api_key)
        self.assertIsNone(manager.provider.client)
    
    def test_is_available(self):
        """Test availability check."""
        # Test when client is available
        manager = WebSearchManager(api_key=self.api_key)
        self.assertTrue(manager.is_available())
        
        # Test when client is not available
        # Make the request return a non-200 status code for Brave
        self.mock_response.status_code = 401
        with patch.dict('os.environ', {}, clear=True):
            manager = WebSearchManager()
            self.assertFalse(manager.is_available())
    
    def test_search_success_brave(self):
        """Test successful web search with Brave provider."""
        # Set up mock response for Brave
        mock_data = {
            "web": {
                "results": [
                    {
                        "title": "Result 1",
                        "url": "https://example.com/1",
                        "description": "Content 1",
                        "relevance_score": 0.9
                    },
                    {
                        "title": "Result 2",
                        "url": "https://example.com/2",
                        "description": "Content 2",
                        "relevance_score": 0.8
                    }
                ]
            }
        }
        self.mock_response.json.return_value = mock_data
        
        manager = WebSearchManager(api_key=self.api_key)
        
        # Call the method
        query = "test query"
        search_depth = "basic"
        max_results = 5
        
        result = manager.search(query=query, search_depth=search_depth, max_results=max_results)
        
        # Verify the result
        self.assertEqual(result["query"], query)
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["search_depth"], search_depth)
        self.assertEqual(result["result_count"], 2)
        self.assertIn("timestamp", result)
        
    def test_search_success_tavily(self):
        """Test successful web search with Tavily provider."""
        # Set up mock response for Tavily
        mock_results = [
            {"title": "Result 1", "url": "https://example.com/1", "content": "Content 1"},
            {"title": "Result 2", "url": "https://example.com/2", "content": "Content 2"}
        ]
        self.mock_tavily.search.return_value = {"results": mock_results}
        
        manager = WebSearchManager(api_key=self.api_key, provider="tavily")
        
        # Call the method
        query = "test query"
        search_depth = "basic"
        max_results = 5
        
        result = manager.search(query=query, search_depth=search_depth, max_results=max_results)
        
        # Verify the result
        self.assertEqual(result["query"], query)
        self.assertEqual(result["results"], mock_results)
        self.assertEqual(result["search_depth"], search_depth)
        self.assertEqual(result["result_count"], 2)
        self.assertIn("timestamp", result)
        
        # Verify that the API was called with the correct parameters
        self.mock_tavily.search.assert_called_once_with(
            query=query,
            search_depth="basic",
            max_results=max_results
        )
    
    def test_search_comprehensive(self):
        """Test comprehensive search depth with Tavily provider."""
        # Set up mock response
        self.mock_tavily.search.return_value = {"results": []}
        
        manager = WebSearchManager(api_key=self.api_key, provider="tavily")
        
        # Call the method with comprehensive depth
        manager.search(query="test", search_depth="comprehensive")
        
        # Verify that the API was called with comprehensive depth
        self.mock_tavily.search.assert_called_once_with(
            query="test",
            search_depth="comprehensive",
            max_results=5
        )
    
    def test_search_unavailable(self):
        """Test search when web search is unavailable."""
        # Create manager without client
        with patch.dict('os.environ', {}, clear=True):
            manager = WebSearchManager()
            
            # Call the method
            result = manager.search(query="test")
            
            # Verify the result indicates unavailability
            self.assertEqual(result["results"], [])
            self.assertIn("error", result)
    
    def test_search_error_tavily(self):
        """Test handling of API errors during search with Tavily provider."""
        # Make search raise an exception
        self.mock_tavily.search.side_effect = Exception("Search Error")
        
        manager = WebSearchManager(api_key=self.api_key, provider="tavily")
        
        # Call the method
        result = manager.search(query="test")
        
        # Verify the result indicates error
        self.assertEqual(result["results"], [])
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Search Error")
        
    def test_provider_selection(self):
        """Test that the correct provider is selected based on the provider parameter."""
        # Test Brave provider (default)
        brave_manager = WebSearchManager(api_key=self.api_key)
        self.assertIsInstance(brave_manager.provider, BraveSearchManager)
        
        # Test Tavily provider
        tavily_manager = WebSearchManager(api_key=self.api_key, provider="tavily")
        self.assertIsInstance(tavily_manager.provider, TavilySearchManager)


if __name__ == "__main__":
    unittest.main()
#!/usr/bin/env python3
"""
Tests for the Brave Search functionality in the web search module.

These tests verify the functionality of the BraveSearchManager class,
including searching the web using the Brave Search API.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
from datetime import datetime

from src.web_search import BraveSearchManager, WebSearchManager


class TestBraveSearchManager(unittest.TestCase):
    """Test cases for the BraveSearchManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock API key
        self.api_key = "test_api_key"
        
        # Patch the requests.get method
        self.requests_patcher = patch('src.web_search.requests.get')
        self.mock_requests_get = self.requests_patcher.start()
        
        # Create a mock response
        self.mock_response = MagicMock()
        self.mock_response.status_code = 200
        self.mock_requests_get.return_value = self.mock_response
        
        # Patch os.environ for environment variable tests
        self.env_patcher = patch.dict('os.environ', {"BRAVE_API_KEY": "env_api_key"})
        self.env_patcher.start()
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.requests_patcher.stop()
        self.env_patcher.stop()
    
    def test_initialization_with_api_key(self):
        """Test initialization with API key provided."""
        manager = BraveSearchManager(api_key=self.api_key)
        
        self.assertEqual(manager.api_key, self.api_key)
        self.assertIsNotNone(manager.client)
        self.mock_requests_get.assert_called_once()
    
    def test_initialization_with_env_api_key(self):
        """Test initialization with API key from environment."""
        manager = BraveSearchManager()
        
        self.assertEqual(manager.api_key, "env_api_key")
        self.assertIsNotNone(manager.client)
        self.mock_requests_get.assert_called_once()
    
    def test_initialization_without_api_key(self):
        """Test initialization without API key."""
        # Remove API key from environment
        with patch.dict('os.environ', {}, clear=True):
            manager = BraveSearchManager()
            
            self.assertIsNone(manager.api_key)
            self.assertIsNone(manager.client)
            self.mock_requests_get.assert_not_called()
    
    def test_initialization_with_api_error(self):
        """Test handling of errors during API initialization."""
        # Make the request return a non-200 status code
        self.mock_response.status_code = 401
        
        manager = BraveSearchManager(api_key=self.api_key)
        
        self.assertEqual(manager.api_key, self.api_key)
        self.assertIsNone(manager.client)
    
    def test_is_available(self):
        """Test availability check."""
        # Test when client is available
        manager = BraveSearchManager(api_key=self.api_key)
        self.assertTrue(manager.is_available())
        
        # Test when client is not available
        with patch.dict('os.environ', {}, clear=True):
            manager = BraveSearchManager()
            self.assertFalse(manager.is_available())
    
    def test_search_success(self):
        """Test successful web search."""
        # Set up mock response
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
        
        manager = BraveSearchManager(api_key=self.api_key)
        
        # Call the method
        query = "test query"
        search_depth = "basic"
        max_results = 5
        result = manager.search(query, search_depth, max_results)
        
        # Verify the result
        self.assertEqual(result["query"], query)
        self.assertEqual(result["search_depth"], search_depth)
        self.assertEqual(result["result_count"], 2)
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["title"], "Result 1")
        self.assertEqual(result["results"][0]["url"], "https://example.com/1")
        self.assertEqual(result["results"][0]["content"], "Content 1")
        self.assertEqual(result["results"][0]["score"], 0.9)
        
        # Verify the API call
        self.mock_requests_get.assert_called_with(
            manager.base_url,
            headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
            params={"q": query, "count": max_results, "text_detail": "snippet"}
        )
    
    def test_search_comprehensive(self):
        """Test comprehensive search."""
        # Set up mock response
        mock_data = {"web": {"results": []}}
        self.mock_response.json.return_value = mock_data
        
        manager = BraveSearchManager(api_key=self.api_key)
        
        # Call the method with comprehensive depth
        result = manager.search("test query", "comprehensive", 5)
        
        # Verify the API call used paragraph detail
        self.mock_requests_get.assert_called_with(
            manager.base_url,
            headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
            params={"q": "test query", "count": 5, "text_detail": "paragraph"}
        )
    
    def test_search_unavailable(self):
        """Test search when client is not available."""
        # Create a manager with no client
        with patch.dict('os.environ', {}, clear=True):
            manager = BraveSearchManager()
            
            # Call the method
            result = manager.search("test query")
            
            # Verify the result
            self.assertEqual(result["results"], [])
            self.assertIn("error", result)
            self.mock_requests_get.assert_not_called()
    
    def test_search_error(self):
        """Test handling of errors during search."""
        # Make the request raise an exception
        self.mock_requests_get.side_effect = Exception("Search Error")
        
        manager = BraveSearchManager(api_key=self.api_key)
        
        # Call the method
        result = manager.search("test query")
        
        # Verify the result
        self.assertEqual(result["results"], [])
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Search Error")


class TestWebSearchManagerFactory(unittest.TestCase):
    """Test cases for the WebSearchManager factory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Patch the BraveSearchManager and TavilySearchManager classes
        self.brave_patcher = patch('src.web_search.BraveSearchManager')
        self.tavily_patcher = patch('src.web_search.TavilySearchManager')
        
        self.mock_brave_cls = self.brave_patcher.start()
        self.mock_tavily_cls = self.tavily_patcher.start()
        
        self.mock_brave = MagicMock()
        self.mock_tavily = MagicMock()
        
        self.mock_brave_cls.return_value = self.mock_brave
        self.mock_tavily_cls.return_value = self.mock_tavily
        
        # Set up availability
        self.mock_brave.is_available.return_value = True
        self.mock_tavily.is_available.return_value = True
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.brave_patcher.stop()
        self.tavily_patcher.stop()
    
    def test_default_provider(self):
        """Test that Brave is the default provider."""
        manager = WebSearchManager(api_key="test_key")
        
        self.assertEqual(manager.provider_name, "brave")
        self.assertEqual(manager.provider, self.mock_brave)
        self.mock_brave_cls.assert_called_once_with(api_key="test_key")
        self.mock_tavily_cls.assert_not_called()
    
    def test_brave_provider(self):
        """Test explicitly selecting Brave provider."""
        manager = WebSearchManager(api_key="test_key", provider="brave")
        
        self.assertEqual(manager.provider_name, "brave")
        self.assertEqual(manager.provider, self.mock_brave)
        self.mock_brave_cls.assert_called_once_with(api_key="test_key")
        self.mock_tavily_cls.assert_not_called()
    
    def test_tavily_provider(self):
        """Test selecting Tavily provider."""
        manager = WebSearchManager(api_key="test_key", provider="tavily")
        
        self.assertEqual(manager.provider_name, "tavily")
        self.assertEqual(manager.provider, self.mock_tavily)
        self.mock_tavily_cls.assert_called_once_with(api_key="test_key")
        self.mock_brave_cls.assert_not_called()
    
    def test_method_delegation(self):
        """Test that methods are delegated to the provider."""
        manager = WebSearchManager(api_key="test_key")
        
        # Test search method delegation
        manager.search("test query", "basic", 5)
        self.mock_brave.search.assert_called_once_with("test query", "basic", 5)
        
        # Test search_news method delegation
        manager.search_news("test topic", 3)
        self.mock_brave.search_news.assert_called_once_with("test topic", 3)
        
        # Test get_topic_insights method delegation
        manager.get_topic_insights("test topic")
        self.mock_brave.get_topic_insights.assert_called_once_with("test topic")
        
        # Test get_competitor_content method delegation
        manager.get_competitor_content("test topic", 4)
        self.mock_brave.get_competitor_content.assert_called_once_with("test topic", 4)


if __name__ == "__main__":
    unittest.main()
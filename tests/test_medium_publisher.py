#!/usr/bin/env python3
"""
Tests for the Medium publisher module.

These tests verify the functionality of the MediumPublisher class,
including publishing articles to Medium.
"""

import unittest
from unittest.mock import patch, MagicMock
import json

from src.medium_publisher import MediumPublisher


class TestMediumPublisher(unittest.TestCase):
    """Test cases for the MediumPublisher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock API token and author ID
        self.integration_token = "test_token"
        self.author_id = "test_author_id"
        
        # Patch the requests module
        self.requests_patcher = patch('src.medium_publisher.requests')
        self.mock_requests = self.requests_patcher.start()
        
        # Set up mock response for API calls
        self.mock_response = MagicMock()
        self.mock_requests.get.return_value = self.mock_response
        self.mock_requests.post.return_value = self.mock_response
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.requests_patcher.stop()
    
    def test_initialization_with_author_id(self):
        """Test initialization with author ID provided."""
        publisher = MediumPublisher(integration_token=self.integration_token, author_id=self.author_id)
        
        self.assertEqual(publisher.integration_token, self.integration_token)
        self.assertEqual(publisher.author_id, self.author_id)
        self.assertEqual(publisher.api_url, "https://api.medium.com/v1")
        
        # Verify that _get_user_info was not called
        self.mock_requests.get.assert_not_called()
    
    def test_initialization_without_author_id(self):
        """Test initialization without author ID (should fetch from API)."""
        # Set up mock response for user info
        self.mock_response.json.return_value = {
            "data": {
                "id": "fetched_author_id",
                "name": "Test Author",
                "username": "testauthor"
            }
        }
        
        publisher = MediumPublisher(integration_token=self.integration_token)
        
        self.assertEqual(publisher.integration_token, self.integration_token)
        self.assertEqual(publisher.author_id, "fetched_author_id")
        
        # Verify that _get_user_info was called
        self.mock_requests.get.assert_called_once_with(
            "https://api.medium.com/v1/me",
            headers=publisher.headers
        )
    
    def test_get_user_info_error(self):
        """Test handling of errors when getting user info."""
        # Set up mock response to raise an exception
        self.mock_response.raise_for_status.side_effect = Exception("API Error")
        
        publisher = MediumPublisher(integration_token=self.integration_token)
        
        # Verify that author_id is None after error
        self.assertIsNone(publisher.author_id)
        
        # Verify that the API was called
        self.mock_requests.get.assert_called_once()
    
    def test_publish_article_success(self):
        """Test successful article publication."""
        # Set up mock response for publication
        self.mock_response.json.return_value = {
            "data": {
                "id": "post_id",
                "title": "Test Article",
                "url": "https://medium.com/p/post_id"
            }
        }
        
        publisher = MediumPublisher(
            integration_token=self.integration_token,
            author_id=self.author_id
        )
        
        # Call the method
        title = "Test Article"
        content = "This is a test article."
        tags = ["test", "article"]
        publish_status = "draft"
        
        result = publisher.publish_article(
            title=title,
            content=content,
            tags=tags,
            publish_status=publish_status
        )
        
        # Verify the result
        self.assertTrue(result["success"])
        self.assertEqual(result["post_id"], "post_id")
        self.assertEqual(result["url"], "https://medium.com/p/post_id")
        
        # Verify that the API was called with the correct parameters
        self.mock_requests.post.assert_called_once()
        call_args = self.mock_requests.post.call_args
        
        # Check URL
        self.assertEqual(
            call_args[0][0],
            f"https://api.medium.com/v1/users/{self.author_id}/posts"
        )
        
        # Check payload
        payload = call_args[1]["json"]
        self.assertEqual(payload["title"], title)
        self.assertEqual(payload["content"], content)
        self.assertEqual(payload["tags"], tags)
        self.assertEqual(payload["publishStatus"], publish_status)
    
    def test_publish_article_no_author_id(self):
        """Test article publication without author ID."""
        publisher = MediumPublisher(integration_token=self.integration_token)
        publisher.author_id = None  # Ensure author_id is None
        
        # Call the method
        result = publisher.publish_article(
            title="Test Article",
            content="This is a test article."
        )
        
        # Verify the result indicates failure
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        
        # Verify that the API was not called
        self.mock_requests.post.assert_not_called()
    
    def test_publish_article_api_error(self):
        """Test handling of API errors during publication."""
        # Set up mock response to raise an exception
        self.mock_response.raise_for_status.side_effect = Exception("API Error")
        
        publisher = MediumPublisher(
            integration_token=self.integration_token,
            author_id=self.author_id
        )
        
        # Call the method
        result = publisher.publish_article(
            title="Test Article",
            content="This is a test article."
        )
        
        # Verify the result indicates failure
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        
        # Verify that the API was called
        self.mock_requests.post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
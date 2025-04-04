#!/usr/bin/env python3
"""
Tests for the cache manager module.

These tests verify the functionality of the CacheManager class,
including storing and retrieving cached responses.
"""

import unittest
from unittest.mock import patch, mock_open, MagicMock
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

from src.cache_manager import CacheManager


class TestCacheManager(unittest.TestCase):
    """Test cases for the CacheManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use a temporary directory for testing
        self.cache_dir = "test_cache"
        self.ttl_days = 7
        
        # Create a patcher for Path.mkdir
        self.mkdir_patcher = patch('pathlib.Path.mkdir')
        self.mock_mkdir = self.mkdir_patcher.start()
        
        # Create the CacheManager instance
        self.cache_manager = CacheManager(cache_dir=self.cache_dir, ttl_days=self.ttl_days)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.mkdir_patcher.stop()
    
    def test_initialization(self):
        """Test that the cache manager initializes correctly."""
        self.assertEqual(self.cache_manager.cache_dir, Path(self.cache_dir))
        self.assertEqual(self.cache_manager.ttl_days, self.ttl_days)
        self.mock_mkdir.assert_called_once_with(exist_ok=True)
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        # Test with a simple dictionary
        params = {"type": "test", "value": 123}
        expected_key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
        actual_key = self.cache_manager._generate_cache_key(params)
        self.assertEqual(actual_key, expected_key)
        
        # Test with a more complex dictionary
        params = {"type": "test", "nested": {"a": 1, "b": 2}, "list": [3, 4, 5]}
        expected_key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
        actual_key = self.cache_manager._generate_cache_key(params)
        self.assertEqual(actual_key, expected_key)
    
    def test_get_cache_miss(self):
        """Test cache retrieval when the cache file doesn't exist."""
        with patch('pathlib.Path.exists', return_value=False):
            params = {"type": "test"}
            result = self.cache_manager.get(params)
            self.assertIsNone(result)
    
    def test_get_cache_expired(self):
        """Test cache retrieval when the cache has expired."""
        params = {"type": "test"}
        cache_key = self.cache_manager._generate_cache_key(params)
        
        # Create a cache file with an expired timestamp
        expired_time = datetime.now() - timedelta(days=self.ttl_days + 1)
        cache_data = {
            "cached_at": expired_time.isoformat(),
            "params": params,
            "response": {"result": "expired"}
        }
        
        # Mock file operations
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(cache_data))), \
             patch('datetime.datetime') as mock_datetime:
            
            # Set the current time
            mock_datetime.now.return_value = datetime.now()
            mock_datetime.fromisoformat.return_value = expired_time
            
            # Get the cache
            result = self.cache_manager.get(params)
            
            # Verify the result is None (cache expired)
            self.assertIsNone(result)
    
    def test_get_cache_hit(self):
        """Test cache retrieval when there's a valid cache hit."""
        params = {"type": "test"}
        cache_key = self.cache_manager._generate_cache_key(params)
        
        # Create a cache file with a valid timestamp
        valid_time = datetime.now() - timedelta(days=self.ttl_days - 1)
        expected_response = {"result": "valid"}
        cache_data = {
            "cached_at": valid_time.isoformat(),
            "params": params,
            "response": expected_response
        }
        
        # Mock file operations
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(cache_data))), \
             patch('datetime.datetime') as mock_datetime:
            
            # Set the current time
            mock_datetime.now.return_value = datetime.now()
            mock_datetime.fromisoformat.return_value = valid_time
            
            # Get the cache
            result = self.cache_manager.get(params)
            
            # Verify the result matches the expected response
            self.assertEqual(result, expected_response)
    
    def test_set_cache(self):
        """Test setting a cache entry."""
        params = {"type": "test"}
        response = {"result": "test_result"}
        cache_key = self.cache_manager._generate_cache_key(params)
        
        # Mock file operations
        mock_file = MagicMock()
        with patch('builtins.open', mock_open()) as mock_file_open, \
             patch('json.dump') as mock_json_dump, \
             patch('datetime.datetime') as mock_datetime:
            
            # Set the current time
            current_time = datetime.now()
            mock_datetime.now.return_value = current_time
            
            # Set the cache
            self.cache_manager.set(params, response)
            
            # Verify the file was opened with the correct path
            expected_path = Path(self.cache_dir) / f"{cache_key}.json"
            mock_file_open.assert_called_once_with(expected_path, "w")
            
            # Verify json.dump was called with the correct data
            expected_data = {
                "cached_at": current_time.isoformat(),
                "params": params,
                "response": response
            }
            mock_json_dump.assert_called_once()
            actual_data = mock_json_dump.call_args[0][0]
            self.assertEqual(actual_data["params"], expected_data["params"])
            self.assertEqual(actual_data["response"], expected_data["response"])
            # We can't directly compare the cached_at timestamps, but we can check it exists
            self.assertIn("cached_at", actual_data)


if __name__ == "__main__":
    unittest.main()
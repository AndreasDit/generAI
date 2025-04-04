#!/usr/bin/env python3
"""
Tests for the utils module.

These tests verify the functionality of utility functions in the utils module.
"""

import unittest
from unittest.mock import patch, mock_open, MagicMock
import os

from src.utils import parse_outline, setup_logging


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_parse_outline_empty(self):
        """Test parsing an empty outline string."""
        result = parse_outline("")
        self.assertIsNone(result)
        
        result = parse_outline("")
        self.assertIsNone(result)
    
    def test_parse_outline_single_section(self):
        """Test parsing an outline with a single section."""
        outline_str = "Introduction"
        expected = ["Introduction"]
        result = parse_outline(outline_str)
        self.assertEqual(result, expected)
    
    def test_parse_outline_multiple_sections(self):
        """Test parsing an outline with multiple sections."""
        outline_str = "Introduction, Main Points, Conclusion"
        expected = ["Introduction", "Main Points", "Conclusion"]
        result = parse_outline(outline_str)
        self.assertEqual(result, expected)
    
    def test_parse_outline_with_whitespace(self):
        """Test parsing an outline with extra whitespace."""
        outline_str = " Introduction ,  Main Points , Conclusion "
        expected = ["Introduction", "Main Points", "Conclusion"]
        result = parse_outline(outline_str)
        self.assertEqual(result, expected)
    
    def test_setup_logging(self):
        """Test the setup_logging function."""
        # Mock os.makedirs and logger.add
        with patch('os.makedirs') as mock_makedirs, \
             patch('loguru.logger.add') as mock_logger_add:
            
            # Call the function
            setup_logging()
            
            # Verify os.makedirs was called with the correct directory
            mock_makedirs.assert_called_once_with("logs", exist_ok=True)
            
            # Verify logger.add was called
            mock_logger_add.assert_called_once()
            
            # Check the arguments to logger.add
            args, kwargs = mock_logger_add.call_args
            self.assertTrue(args[0].startswith("logs/article_generator_"))
            self.assertEqual(kwargs["rotation"], "10 MB")
            self.assertEqual(kwargs["retention"], "1 week")
            self.assertEqual(kwargs["level"], "INFO")
            self.assertTrue("format" in kwargs)
            self.assertEqual(kwargs["enqueue"], True)


if __name__ == "__main__":
    unittest.main()
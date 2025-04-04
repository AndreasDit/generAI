#!/usr/bin/env python3
"""
Tests for the feedback manager module.

These tests verify the functionality of the FeedbackManager class,
including recording article metrics and providing feedback for improvement.
"""

import unittest
from unittest.mock import patch, mock_open, MagicMock
import json
from pathlib import Path
from datetime import datetime

from src.feedback_manager import FeedbackManager


class TestFeedbackManager(unittest.TestCase):
    """Test cases for the FeedbackManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use a test directory for data
        self.data_dir = "test_data"
        self.feedback_dir = Path(self.data_dir) / "feedback"
        self.analytics_file = self.feedback_dir / "analytics.json"
        
        # Create patchers
        self.mkdir_patcher = patch('pathlib.Path.mkdir')
        self.exists_patcher = patch('pathlib.Path.exists')
        
        # Start patchers
        self.mock_mkdir = self.mkdir_patcher.start()
        self.mock_exists = self.exists_patcher.start()
        
        # Set up default behavior
        self.mock_exists.return_value = False  # Analytics file doesn't exist by default
        
        # Mock the _initialize_analytics method
        self.init_analytics_patcher = patch.object(FeedbackManager, '_initialize_analytics')
        self.mock_init_analytics = self.init_analytics_patcher.start()
        
        # Create the FeedbackManager instance
        self.feedback_manager = FeedbackManager(data_dir=self.data_dir)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.mkdir_patcher.stop()
        self.exists_patcher.stop()
        self.init_analytics_patcher.stop()
    
    def test_initialization(self):
        """Test that the feedback manager initializes correctly."""
        self.assertEqual(self.feedback_manager.data_dir, Path(self.data_dir))
        self.assertEqual(self.feedback_manager.feedback_dir, self.feedback_dir)
        self.assertEqual(self.feedback_manager.analytics_file, self.analytics_file)
        
        # Verify directories were created
        self.mock_mkdir.assert_called_with(exist_ok=True)
        
        # Verify analytics file was checked
        self.mock_exists.assert_called_with()
        
        # Verify analytics was initialized (since file doesn't exist)
        self.mock_init_analytics.assert_called_once()
    
    def test_initialize_analytics(self):
        """Test analytics file initialization."""
        # Restore the real method
        self.init_analytics_patcher.stop()
        
        # Mock the open function
        mock_file = MagicMock()
        with patch('builtins.open', mock_open()) as mock_file_open, \
             patch('json.dump') as mock_json_dump, \
             patch('datetime.datetime') as mock_datetime:
            
            # Set the current time
            current_time = datetime.now()
            mock_datetime.now.return_value = current_time
            
            # Call the method
            self.feedback_manager._initialize_analytics()
            
            # Verify the file was opened with the correct path
            mock_file_open.assert_called_once_with(self.analytics_file, "w")
            
            # Verify json.dump was called with the correct data structure
            mock_json_dump.assert_called_once()
            actual_data = mock_json_dump.call_args[0][0]
            
            # Check the structure of the analytics data
            self.assertIn("articles", actual_data)
            self.assertIn("topic_performance", actual_data)
            self.assertIn("audience_performance", actual_data)
            self.assertIn("style_performance", actual_data)
            self.assertIn("last_updated", actual_data)
            
            # Restart the patcher for other tests
            self.init_analytics_patcher = patch.object(FeedbackManager, '_initialize_analytics')
            self.mock_init_analytics = self.init_analytics_patcher.start()
    
    def test_record_article_metrics(self):
        """Test recording article performance metrics."""
        # Test data
        project_id = "test_project"
        metrics = {"views": 100, "reads": 75, "claps": 25}
        
        # Mock project directory and metadata file
        project_dir = Path(self.data_dir) / "projects" / project_id
        metadata_file = project_dir / "metadata.json"
        
        # Mock metadata content
        metadata = {
            "title": "Test Article",
            "topic": "Test Topic",
            "audience": "Test Audience",
            "style": "Informative"
        }
        
        # Set up patchers
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(metadata))), \
             patch('json.load', return_value=metadata), \
             patch('json.dump') as mock_json_dump, \
             patch.object(FeedbackManager, '_update_analytics') as mock_update_analytics, \
             patch('datetime.datetime') as mock_datetime:
            
            # Set the current time
            current_time = datetime.now()
            mock_datetime.now.return_value = current_time
            
            # Call the method
            self.feedback_manager.record_article_metrics(project_id, metrics)
            
            # Verify _update_analytics was called with the correct data
            mock_update_analytics.assert_called_once()
            performance_record = mock_update_analytics.call_args[0][0]
            
            # Check the performance record
            self.assertEqual(performance_record["project_id"], project_id)
            self.assertEqual(performance_record["title"], metadata["title"])
            self.assertEqual(performance_record["topic"], metadata["topic"])
            self.assertEqual(performance_record["audience"], metadata["audience"])
            self.assertEqual(performance_record["style"], metadata["style"])
            self.assertEqual(performance_record["metrics"], metrics)
            self.assertIn("recorded_at", performance_record)
    
    def test_get_topic_feedback(self):
        """Test getting feedback for a specific topic."""
        # Test data
        topic = "Test Topic"
        
        # Mock analytics data
        analytics_data = {
            "topic_performance": {
                topic: {
                    "articles": 5,
                    "avg_views": 150,
                    "avg_reads": 100,
                    "avg_claps": 30,
                    "best_performing": "test_project_1",
                    "worst_performing": "test_project_2"
                }
            }
        }
        
        # Set up patchers
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(analytics_data))), \
             patch('json.load', return_value=analytics_data):
            
            # Call the method
            feedback = self.feedback_manager.get_topic_feedback(topic)
            
            # Verify the feedback contains the expected data
            self.assertIsNotNone(feedback)
            self.assertIn("performance", feedback)
            self.assertEqual(feedback["performance"]["articles"], 5)
            self.assertEqual(feedback["performance"]["avg_views"], 150)
            self.assertEqual(feedback["performance"]["avg_reads"], 100)
            self.assertEqual(feedback["performance"]["avg_claps"], 30)


if __name__ == "__main__":
    unittest.main()
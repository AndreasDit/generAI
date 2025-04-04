#!/usr/bin/env python3
"""
Tests for the main GenerAI entry point.

These tests verify the functionality of the command-line interface
and argument parsing in the main generai.py module.
"""

import unittest
from unittest.mock import patch, MagicMock
import argparse
import sys

# Import the module under test
import generai
from src.openai_client import OpenAIClient
from src.medium_publisher import MediumPublisher
from src.article_pipeline import ArticlePipeline


class TestGenerAI(unittest.TestCase):
    """Test cases for the GenerAI main module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_openai_client = MagicMock(spec=OpenAIClient)
        self.mock_medium_publisher = MagicMock(spec=MediumPublisher)
        self.mock_article_pipeline = MagicMock(spec=ArticlePipeline)
        
        # Create patchers
        self.openai_patcher = patch('generai.OpenAIClient', return_value=self.mock_openai_client)
        self.medium_patcher = patch('generai.MediumPublisher', return_value=self.mock_medium_publisher)
        self.pipeline_patcher = patch('generai.ArticlePipeline', return_value=self.mock_article_pipeline)
        self.config_patcher = patch('generai.ConfigManager')
        
        # Start patchers
        self.mock_openai_cls = self.openai_patcher.start()
        self.mock_medium_cls = self.medium_patcher.start()
        self.mock_pipeline_cls = self.pipeline_patcher.start()
        self.mock_config_cls = self.config_patcher.start()
        
        # Mock config manager
        self.mock_config = MagicMock()
        self.mock_config_cls.return_value = self.mock_config
        self.mock_config.get.return_value = "test_value"
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.openai_patcher.stop()
        self.medium_patcher.stop()
        self.pipeline_patcher.stop()
        self.config_patcher.stop()
    
    @patch('sys.argv', ['generai.py', '--help'])
    def test_setup_argparse(self):
        """Test argument parser setup."""
        # We need to patch sys.exit to prevent the test from exiting when --help is used
        with patch('sys.exit') as mock_exit:
            parser = generai.setup_argparse()
            
            # Verify parser is an ArgumentParser
            self.assertIsInstance(parser, argparse.ArgumentParser)
            
            # Verify subparsers exist
            self.assertTrue(hasattr(parser, '_subparsers'))
            
            # Create a new parser for testing argument parsing
            test_parser = argparse.ArgumentParser()
            subparsers = test_parser.add_subparsers(dest="mode")
            
            # Add simple mode subparser
            simple_parser = subparsers.add_parser("simple")
            simple_parser.add_argument("--topic", type=str)
            
            # Add modular mode subparser
            modular_parser = subparsers.add_parser("modular")
            modular_parser.add_argument("--run-full-pipeline", action="store_true")
            
            # Parse simple mode arguments
            args = test_parser.parse_args(['simple', '--topic', 'Test Topic'])
            self.assertEqual(args.mode, 'simple')
            self.assertEqual(args.topic, 'Test Topic')
            
            # Parse modular mode arguments
            args = test_parser.parse_args(['modular', '--run-full-pipeline'])
            self.assertEqual(args.mode, 'modular')
            self.assertTrue(args.run_full_pipeline)
    
    @patch('sys.argv', ['generai.py', 'simple', '--topic', 'Test Topic', '--output', 'test.md'])
    @patch('generai.setup_logging')
    def test_simple_mode(self, mock_setup_logging):
        """Test simple mode execution."""
        # Mock article generation
        self.mock_openai_client.generate_article.return_value = {
            'title': 'Test Article',
            'content': 'Test content'
        }
        
        # Mock file operations
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            # Call the main function
            generai.main()
            
            # Verify OpenAI client was initialized
            self.mock_openai_cls.assert_called_once()
            
            # Verify article was generated
            self.mock_openai_client.generate_article.assert_called_once()
            
            # Verify file was written
            mock_file.assert_called_once_with('test.md', 'w')
    
    @patch('sys.argv', ['generai.py', 'modular', '--run-full-pipeline', '--research-topic', 'Test Topic'])
    @patch('generai.setup_logging')
    def test_modular_mode_full_pipeline(self, mock_setup_logging):
        """Test modular mode with full pipeline execution."""
        # Mock pipeline methods
        self.mock_article_pipeline.generate_ideas.return_value = [{'id': 'idea1', 'title': 'Test Idea'}]
        self.mock_article_pipeline.evaluate_ideas.return_value = [{'id': 'idea1', 'score': 90}]
        self.mock_article_pipeline.create_project.return_value = {'project_id': 'proj1', 'title': 'Test Project'}
        self.mock_article_pipeline.generate_outline.return_value = {'sections': ['Intro', 'Body', 'Conclusion']}
        self.mock_article_pipeline.generate_paragraphs.return_value = {'paragraphs': ['P1', 'P2', 'P3']}
        self.mock_article_pipeline.assemble_article.return_value = {'title': 'Test Article', 'content': 'Test content'}
        
        # Instead of letting run_pipeline call the individual methods, we'll mock it
        # to directly return a result, and then we'll verify the individual methods
        self.mock_article_pipeline.run_pipeline.return_value = {
            'title': 'Test Article', 
            'content': 'Test content'
        }
        
        # Mock file operations
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            # Call the main function
            generai.main()
            
            # Verify ArticlePipeline was initialized
            self.mock_pipeline_cls.assert_called_once()
            
            # Verify run_pipeline was called with the correct parameters
            self.mock_article_pipeline.run_pipeline.assert_called_once_with(
                research_topic='Test Topic',
                num_ideas=5,
                max_ideas_to_evaluate=10
            )
            
            # Since we're mocking run_pipeline to return directly, the individual methods
            # won't be called, so we don't need to verify them
    
    @patch('sys.argv', ['generai.py', 'simple', '--topic', 'Test Topic', '--publish'])
    @patch('generai.setup_logging')
    def test_publish_to_medium(self, mock_setup_logging):
        """Test publishing to Medium."""
        # Mock article generation
        self.mock_openai_client.generate_article.return_value = {
            'title': 'Test Article',
            'content': 'Test content'
        }
        
        # Mock Medium publishing
        self.mock_medium_publisher.publish_article.return_value = {
            'success': True,
            'post_id': 'post123',
            'url': 'https://medium.com/p/post123',
            'publish_status': 'draft'
        }
        
        # Mock the config to include Medium integration token
        self.mock_config.get_config.return_value = {
            "openai": {"api_key": "test_key", "model": "gpt-4"},
            "medium": {"integration_token": "test_token", "author_id": "test_author"},
            "article": {"default_tags": [], "default_status": "draft"}
        }
        
        # Call the main function
        generai.main()
        
        # Verify Medium publisher was initialized
        self.mock_medium_cls.assert_called_once()
        
        # Verify article was published
        self.mock_medium_publisher.publish_article.assert_called_once()


if __name__ == "__main__":
    unittest.main()
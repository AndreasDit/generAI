#!/usr/bin/env python3
"""
Utility functions for GenerAI

This module contains utility functions for the GenerAI application.
"""

import os
import argparse
from typing import List
from loguru import logger


def setup_argparse() -> argparse.Namespace:
    """Set up command line argument parsing.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Generate articles with OpenAI and publish to Medium")
    
    # Article generation options
    parser.add_argument("--topic", type=str, help="Topic for the article")
    parser.add_argument("--tone", type=str, default="informative", 
                        choices=["informative", "casual", "professional", "technical", "conversational"],
                        help="Tone of the article")
    parser.add_argument("--length", type=str, default="medium", 
                        choices=["short", "medium", "long"],
                        help="Length of the article")
    parser.add_argument("--outline", type=str, help="Comma-separated list of sections to include")
    
    # Medium publishing options
    parser.add_argument("--publish", action="store_true", help="Publish to Medium after generation")
    parser.add_argument("--tags", type=str, help="Comma-separated list of tags for Medium")
    parser.add_argument("--status", type=str, default="draft", 
                        choices=["draft", "public", "unlisted"],
                        help="Publication status on Medium")
    parser.add_argument("--canonical-url", type=str, help="Original URL if this is a cross-post")
    
    # Output options
    parser.add_argument("--output", type=str, help="Save article to file")
    
    return parser.parse_args()


def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure logger
    logger.add(
        "logs/article_generator_{time}.log",
        rotation="10 MB",
        retention="1 week",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        enqueue=True,
    )


def parse_outline(outline_str: str) -> List[str]:
    """Parse a comma-separated outline string into a list of sections.
    
    Args:
        outline_str: Comma-separated string of section titles
        
    Returns:
        List of section titles
    """
    if not outline_str:
        return None
    return [section.strip() for section in outline_str.split(",")]
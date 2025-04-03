#!/usr/bin/env python3
"""
Configuration Manager for GenerAI

This module handles loading and managing configuration settings from environment variables.
Includes settings for caching system and feedback loop mechanism.
"""

import os
from typing import Dict, Any

from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


class ConfigManager:
    """Manages configuration settings from environment variables."""
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": os.getenv("OPENAI_MODEL", "gpt-4"),
                "use_cache": os.getenv("OPENAI_USE_CACHE", "true").lower() == "true",
            },
            "medium": {
                "integration_token": os.getenv("MEDIUM_INTEGRATION_TOKEN"),
                "author_id": os.getenv("MEDIUM_AUTHOR_ID"),
            },
            "article": {
                "default_tags": [tag.strip() for tag in os.getenv("DEFAULT_TAGS", "AI,Technology").split(",")],
                "default_status": os.getenv("DEFAULT_STATUS", "draft"),
            },
            "cache": {
                "enabled": os.getenv("CACHE_ENABLED", "true").lower() == "true",
                "ttl_days": int(os.getenv("CACHE_TTL_DAYS", "7")),
                "directory": os.getenv("CACHE_DIRECTORY", "cache"),
            },
            "feedback": {
                "enabled": os.getenv("FEEDBACK_ENABLED", "true").lower() == "true",
                "apply_to_idea_evaluation": os.getenv("FEEDBACK_APPLY_TO_IDEA_EVALUATION", "true").lower() == "true",
                "min_articles_for_insights": int(os.getenv("FEEDBACK_MIN_ARTICLES", "3")),
            }
        }
        return config
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self.config
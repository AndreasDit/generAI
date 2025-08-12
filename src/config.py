"""Configuration settings for the article generation pipeline."""

import os
from pathlib import Path
from typing import Dict, Any, Optional

from loguru import logger

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
PROJECTS_DIR = DATA_DIR / "projects"
IDEAS_DIR = DATA_DIR / "ideas"
ARTICLE_QUEUE_DIR = DATA_DIR / "article_queue"
FEEDBACK_DIR = DATA_DIR / "feedback"

# Create directories if they don't exist
for directory in [DATA_DIR, CACHE_DIR, PROJECTS_DIR, IDEAS_DIR, ARTICLE_QUEUE_DIR, FEEDBACK_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# LLM Configuration
LLM_CONFIG = {
    "default_provider": "openai",  # "claude" or "openai"
    "claude": {
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 4096,
        "temperature": 1
    },
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": os.getenv("OPENAI_MODEL"),
        "text_generation_model": os.getenv("OPENAI_MODEL_TEXT_GENERATION"),
        "max_tokens": 4096,
        "temperature": 1
    }
}

# Cache Configuration
CACHE_CONFIG = {
    "ttl_days": int(os.getenv("CACHE_TTL_DAYS", "7")),
    "max_size_mb": int(os.getenv("CACHE_MAX_SIZE_MB", "1000"))
}

# Web Search Configuration
WEB_SEARCH_CONFIG = {
    "default_provider": "brave",  # "brave" or "tavily"
    "brave": {
        "api_key": os.getenv("BRAVE_API_KEY"),
        "max_results": 10,
        "search_depth": "comprehensive"
    },
    "tavily": {
        "api_key": os.getenv("TAVILY_API_KEY"),
        "max_results": 10,
        "search_depth": "advanced"
    }
}

# Project Configuration
PROJECT_CONFIG = {
    "max_projects": int(os.getenv("MAX_PROJECTS", "100")),
    "auto_cleanup": os.getenv("AUTO_CLEANUP", "true").lower() == "true"
}

# Feedback Configuration
FEEDBACK_CONFIG = {
    "max_feedback_items": int(os.getenv("MAX_FEEDBACK_ITEMS", "1000")),
    "auto_cleanup": os.getenv("FEEDBACK_AUTO_CLEANUP", "true").lower() == "true"
}

# Medium publishing configuration
MEDIUM_CONFIG = {
    "api_key": os.getenv("MEDIUM_API_KEY"),
    "author_id": os.getenv("MEDIUM_AUTHOR_ID"),
    "publication_id": os.getenv("MEDIUM_PUBLICATION_ID")
}

# Directory configuration
DIR_CONFIG = {
    "data_dir": Path("data"),
    "logs_dir": Path("logs"),
    "projects_dir": Path("data/projects"),
    "ideas_dir": Path("data/ideas"),
    "feedback_dir": Path("data/feedback"),
    "searches_dir": Path("data/searches")
}

# Create required directories
for dir_path in DIR_CONFIG.values():
    dir_path.mkdir(parents=True, exist_ok=True)

def get_llm_config() -> Dict[str, Any]:
    """Get the LLM configuration for the current provider."""
    provider = LLM_CONFIG["default_provider"].lower()
    if provider not in ["openai", "claude"]:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    return LLM_CONFIG[provider]

def get_cache_config() -> Dict[str, Any]:
    """Get the cache configuration."""
    return CACHE_CONFIG

def get_web_search_config() -> Dict[str, Any]:
    """Get the web search configuration."""
    return WEB_SEARCH_CONFIG

def get_project_config() -> Dict[str, Any]:
    """Get the project configuration."""
    return PROJECT_CONFIG

def get_feedback_config() -> Dict[str, Any]:
    """Get the feedback configuration."""
    return FEEDBACK_CONFIG

def load_config() -> Dict[str, Any]:
    """Load all configuration settings.
    
    Returns:
        Dictionary containing all configuration settings
    """
    config = {
        "llm": LLM_CONFIG,
        "cache": CACHE_CONFIG,
        "web_search": WEB_SEARCH_CONFIG,
        "project": PROJECT_CONFIG,
        "feedback": FEEDBACK_CONFIG,
        "medium": MEDIUM_CONFIG,
        "dir": DIR_CONFIG
    }
    
    # Validate required API keys
    required_keys = {
        "OPENAI_API_KEY": "OpenAI API key",
        "ANTHROPIC_API_KEY": "Anthropic API key",
        "BRAVE_API_KEY": "Brave API key"
    }
    
    missing_keys = []
    for key, description in required_keys.items():
        if not os.getenv(key):
            missing_keys.append(f"{description} ({key})")
    
    if missing_keys:
        logger.error("Missing required API keys:")
        for key in missing_keys:
            logger.error(f"- {key}")
    
    return config 
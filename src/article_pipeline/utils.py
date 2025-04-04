"""Utility functions for the article pipeline."""

from pathlib import Path
from typing import Tuple
from loguru import logger

def setup_directory_structure(data_dir: Path) -> Tuple[Path, Path, Path]:
    """Set up the directory structure for the article pipeline.
    
    Args:
        data_dir: Base directory for storing article data
        
    Returns:
        Tuple of (ideas_dir, article_queue_dir, projects_dir)
    """
    # Create main data directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for each stage of the pipeline
    ideas_dir = data_dir / "ideas"
    ideas_dir.mkdir(exist_ok=True)
    
    article_queue_dir = data_dir / "article_queue"
    article_queue_dir.mkdir(exist_ok=True)
    
    projects_dir = data_dir / "projects"
    projects_dir.mkdir(exist_ok=True)
    
    logger.info(f"Directory structure set up in {data_dir}")
    return ideas_dir, article_queue_dir, projects_dir

def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing or replacing invalid characters.
    
    Args:
        filename: Original filename to sanitize
        
    Returns:
        Sanitized filename
    """
    # Replace invalid characters with underscores
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    # Ensure filename is not empty
    if not filename:
        filename = 'untitled'
    
    return filename 
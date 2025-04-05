"""Project management functionality for the article pipeline."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger

from src.llm_client import LLMClient
from .utils import sanitize_filename

class ProjectManager:
    """Handles project creation and management."""
    
    def __init__(self, openai_client: LLMClient, projects_dir: Path):
        """Initialize the project manager.
        
        Args:
            openai_client: LLM client for API interactions
            projects_dir: Directory to store project data
        """
        self.llm_client = openai_client
        self.projects_dir = projects_dir
        self.projects_dir.mkdir(parents=True, exist_ok=True)
    
    def create_project(self, idea: Dict[str, Any]) -> str:
        """Create a new project from an idea.
        
        Args:
            idea: Dictionary containing idea data
            
        Returns:
            Project ID
        """
        logger.info("Creating new project from idea")
        
        # Generate project ID
        project_id = f"project_{idea.get('id', '')}"
        
        # Create project directory
        project_dir = self.projects_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Save idea data
        idea_file = project_dir / "idea.json"
        with open(idea_file, "w") as f:
            json.dump(idea, f, indent=2)
        
        # Initialize project metadata
        metadata = {
            "id": project_id,
            "status": "created",
            "idea_id": idea.get("id", ""),
            "created_at": idea.get("created_at", ""),
            "updated_at": idea.get("created_at", "")
        }
        
        # Save metadata
        metadata_file = project_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created project: {project_id}")
        return project_id
    
    def get_project(self, project_id: str) -> Dict[str, Any]:
        """Get project data.
        
        Args:
            project_id: ID of the project to get
            
        Returns:
            Dictionary containing project data
        """
        logger.info(f"Getting project: {project_id}")
        
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            logger.error(f"Project not found: {project_id}")
            return {}
        
        # Load metadata
        metadata_file = project_dir / "metadata.json"
        if not metadata_file.exists():
            logger.error(f"Project metadata not found: {project_id}")
            return {}
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        # Load idea data
        idea_file = project_dir / "idea.json"
        if not idea_file.exists():
            logger.error(f"Project idea not found: {project_id}")
            return metadata
        
        with open(idea_file) as f:
            idea = json.load(f)
        
        # Combine data
        project_data = {
            **metadata,
            "idea": idea
        }
        
        return project_data
    
    def update_project(self, project_id: str, updates: Dict[str, Any]) -> bool:
        """Update project data.
        
        Args:
            project_id: ID of the project to update
            updates: Dictionary containing updates
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Updating project: {project_id}")
        
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            logger.error(f"Project not found: {project_id}")
            return False
        
        # Load current metadata
        metadata_file = project_dir / "metadata.json"
        if not metadata_file.exists():
            logger.error(f"Project metadata not found: {project_id}")
            return False
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        # Update metadata
        metadata.update(updates)
        metadata["updated_at"] = updates.get("updated_at", metadata["updated_at"])
        
        # Save updated metadata
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Updated project: {project_id}")
        return True
    
    def delete_project(self, project_id: str) -> bool:
        """Delete a project and all its data.
        
        Args:
            project_id: ID of the project to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            return False
        
        try:
            shutil.rmtree(project_dir)
            logger.info(f"Deleted project {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting project: {e}")
            return False
    
    def list_projects(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all projects, optionally filtered by status.
        
        Args:
            status: Optional status to filter by
            
        Returns:
            List of project data dictionaries
        """
        projects = []
        
        for project_dir in self.projects_dir.iterdir():
            if not project_dir.is_dir():
                continue
            
            metadata_file = project_dir / "metadata.json"
            if not metadata_file.exists():
                continue
            
            try:
                with open(metadata_file) as f:
                    project_data = json.load(f)
                
                if status is None or project_data.get("status") == status:
                    projects.append(project_data)
                    
            except Exception as e:
                logger.error(f"Error reading project {project_dir.name}: {e}")
        
        return sorted(projects, key=lambda x: x.get("created_at", ""), reverse=True) 
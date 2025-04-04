"""Project management functionality for the article pipeline."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger

from src.openai_client import OpenAIClient
from .utils import sanitize_filename

class ProjectManager:
    """Handles project creation and management."""
    
    def __init__(self, openai_client: OpenAIClient, projects_dir: Path):
        """Initialize the project manager.
        
        Args:
            openai_client: OpenAI client for API interactions
            projects_dir: Directory to store project data
        """
        self.openai_client = openai_client
        self.projects_dir = projects_dir
    
    def create_project(self, idea: Dict[str, Any]) -> Optional[str]:
        """Create a new project from a selected idea.
        
        Args:
            idea: Selected idea dictionary
            
        Returns:
            Project ID or None if creation failed
        """
        try:
            # Generate project ID
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            project_id = f"project_{timestamp}"
            
            # Create project directory
            project_dir = self.projects_dir / project_id
            project_dir.mkdir(exist_ok=True)
            
            # Create project metadata
            project_data = {
                "id": project_id,
                "idea": idea,
                "created_at": timestamp,
                "status": "created",
                "outline": None,
                "paragraphs": [],
                "final_article": None
            }
            
            # Save project metadata
            metadata_file = project_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(project_data, f, indent=2)
            
            # Create directories for project assets
            (project_dir / "outline").mkdir(exist_ok=True)
            (project_dir / "paragraphs").mkdir(exist_ok=True)
            (project_dir / "drafts").mkdir(exist_ok=True)
            
            logger.info(f"Created project {project_id} from idea {idea.get('id', 'unknown')}")
            return project_id
            
        except Exception as e:
            logger.error(f"Error creating project: {e}")
            return None
    
    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve project data by ID.
        
        Args:
            project_id: ID of the project to retrieve
            
        Returns:
            Project data dictionary or None if not found
        """
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            return None
        
        metadata_file = project_dir / "metadata.json"
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading project metadata: {e}")
            return None
    
    def update_project(self, project_id: str, updates: Dict[str, Any]) -> bool:
        """Update project metadata.
        
        Args:
            project_id: ID of the project to update
            updates: Dictionary of updates to apply
            
        Returns:
            True if update successful, False otherwise
        """
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            return False
        
        metadata_file = project_dir / "metadata.json"
        if not metadata_file.exists():
            return False
        
        try:
            # Read current metadata
            with open(metadata_file) as f:
                project_data = json.load(f)
            
            # Apply updates
            project_data.update(updates)
            
            # Save updated metadata
            with open(metadata_file, "w") as f:
                json.dump(project_data, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating project: {e}")
            return False
    
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
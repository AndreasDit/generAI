"""Content generation functionality for the article pipeline."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger

from src.openai_client import OpenAIClient

class ContentGenerator:
    """Handles content generation for articles."""
    
    def __init__(self, openai_client: OpenAIClient, projects_dir: Path):
        """Initialize the content generator.
        
        Args:
            openai_client: OpenAI client for API interactions
            projects_dir: Directory containing project data
        """
        self.openai_client = openai_client
        self.projects_dir = projects_dir
    
    def generate_outline(self, project_id: str) -> Optional[List[str]]:
        """Generate an outline for an article.
        
        Args:
            project_id: ID of the project to generate outline for
            
        Returns:
            List of outline sections or None if generation failed
        """
        try:
            # Get project data
            project_dir = self.projects_dir / project_id
            if not project_dir.exists():
                logger.error(f"Project directory not found: {project_id}")
                return None
            
            metadata_file = project_dir / "metadata.json"
            if not metadata_file.exists():
                logger.error(f"Project metadata not found: {project_id}")
                return None
            
            with open(metadata_file) as f:
                project_data = json.load(f)
            
            idea = project_data.get("idea", {})
            if not idea:
                logger.error(f"No idea data found for project: {project_id}")
                return None
            
            # Generate outline using OpenAI
            system_prompt = (
                "You are an expert content strategist who creates detailed article outlines. "
                "Your outlines should be well-structured, comprehensive, and engaging."
            )
            
            user_prompt = f"""Create a detailed outline for an article with the following details:
            
            Title: {idea.get('title', 'No title')}
            Description: {idea.get('description', 'No description')}
            Target Audience: {idea.get('audience', 'No audience specified')}
            Key Points: {idea.get('key_points', 'No key points')}
            
            The outline should:
            1. Start with an engaging introduction
            2. Include 4-6 main sections
            3. Each section should have 2-3 subsections
            4. End with a strong conclusion
            
            Format the outline as a list of sections, with subsections indented.
            """
            
            response = self.openai_client.client.chat.completions.create(
                model=self.openai_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
            )
            
            outline_text = response.choices[0].message.content
            
            # Parse outline into sections
            outline = []
            current_section = None
            
            for line in outline_text.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                
                # Check if this is a main section (no indentation)
                if not line.startswith(" "):
                    if current_section:
                        outline.append(current_section)
                    current_section = {
                        "title": line,
                        "subsections": []
                    }
                elif current_section:
                    # This is a subsection
                    current_section["subsections"].append(line)
            
            # Add the last section
            if current_section:
                outline.append(current_section)
            
            # Save outline
            outline_file = project_dir / "outline" / "outline.json"
            with open(outline_file, "w") as f:
                json.dump(outline, f, indent=2)
            
            # Update project metadata
            project_data["outline"] = outline
            with open(metadata_file, "w") as f:
                json.dump(project_data, f, indent=2)
            
            logger.info(f"Generated outline for project {project_id}")
            return outline
            
        except Exception as e:
            logger.error(f"Error generating outline: {e}")
            return None
    
    def generate_paragraphs(self, project_id: str) -> bool:
        """Generate paragraphs for each section of the outline.
        
        Args:
            project_id: ID of the project to generate paragraphs for
            
        Returns:
            True if generation successful, False otherwise
        """
        try:
            # Get project data
            project_dir = self.projects_dir / project_id
            if not project_dir.exists():
                logger.error(f"Project directory not found: {project_id}")
                return False
            
            metadata_file = project_dir / "metadata.json"
            if not metadata_file.exists():
                logger.error(f"Project metadata not found: {project_id}")
                return False
            
            with open(metadata_file) as f:
                project_data = json.load(f)
            
            outline = project_data.get("outline")
            if not outline:
                logger.error(f"No outline found for project: {project_id}")
                return False
            
            idea = project_data.get("idea", {})
            if not idea:
                logger.error(f"No idea data found for project: {project_id}")
                return False
            
            # Generate paragraphs for each section
            paragraphs = []
            for section_index, section in enumerate(outline):
                # Generate main section paragraph
                main_paragraph = self._generate_paragraph(
                    idea, outline, section["title"], section_index
                )
                if main_paragraph:
                    paragraphs.append({
                        "section": section["title"],
                        "content": main_paragraph,
                        "type": "main"
                    })
                
                # Generate subsections
                for subsection in section.get("subsections", []):
                    subsection_paragraph = self._generate_paragraph(
                        idea, outline, subsection, section_index
                    )
                    if subsection_paragraph:
                        paragraphs.append({
                            "section": subsection,
                            "content": subsection_paragraph,
                            "type": "subsection"
                        })
            
            # Save paragraphs
            paragraphs_file = project_dir / "paragraphs" / "paragraphs.json"
            with open(paragraphs_file, "w") as f:
                json.dump(paragraphs, f, indent=2)
            
            # Update project metadata
            project_data["paragraphs"] = paragraphs
            with open(metadata_file, "w") as f:
                json.dump(project_data, f, indent=2)
            
            logger.info(f"Generated paragraphs for project {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating paragraphs: {e}")
            return False
    
    def _generate_paragraph(self, idea: Dict[str, Any], outline: List[Dict[str, Any]], 
                          section: str, section_index: int) -> Optional[str]:
        """Generate a paragraph for a specific section.
        
        Args:
            idea: Project idea data
            outline: Article outline
            section: Section title
            section_index: Index of the section in the outline
            
        Returns:
            Generated paragraph text or None if generation failed
        """
        try:
            system_prompt = (
                "You are an expert content writer who creates engaging, informative paragraphs. "
                "Your writing should be clear, concise, and valuable to the target audience."
            )
            
            # Get context from previous sections
            context = ""
            for i in range(section_index):
                prev_section = outline[i]
                context += f"\nPrevious section '{prev_section['title']}':\n"
                for subsection in prev_section.get("subsections", []):
                    context += f"- {subsection}\n"
            
            user_prompt = f"""Write a paragraph for the following section of an article:
            
            Article Title: {idea.get('title', 'No title')}
            Target Audience: {idea.get('audience', 'No audience specified')}
            Key Points: {idea.get('key_points', 'No key points')}
            
            Current Section: {section}
            
            Context from previous sections:{context}
            
            The paragraph should:
            1. Be 3-5 sentences long
            2. Be engaging and informative
            3. Flow naturally from previous sections
            4. Include relevant details and examples
            5. Maintain a consistent tone and style
            """
            
            response = self.openai_client.client.chat.completions.create(
                model=self.openai_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500,
            )
            
            paragraph = response.choices[0].message.content.strip()
            return paragraph
            
        except Exception as e:
            logger.error(f"Error generating paragraph for section '{section}': {e}")
            return None 
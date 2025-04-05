"""Content generator for article generation."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from loguru import logger

from src.llm_client import LLMClient


class ContentGenerator:
    """Generates content for articles."""
    
    def __init__(self, openai_client: LLMClient, projects_dir: Path):
        """Initialize the content generator.
        
        Args:
            openai_client: LLM client for API interactions
            projects_dir: Directory to store project data
        """
        self.llm_client = openai_client
        self.projects_dir = projects_dir
    
    def generate_outline(self, project_id: str) -> Dict[str, Any]:
        """Generate an outline for a project.
        
        Args:
            project_id: ID of the project to generate outline for
            
        Returns:
            Dictionary containing the generated outline
        """
        logger.info(f"Generating outline for project: {project_id}")
        
        # Get project data
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            logger.error(f"Project not found: {project_id}")
            return {}
        
        idea_file = project_dir / "idea.json"
        if not idea_file.exists():
            logger.error(f"Project idea not found: {project_id}")
            return {}
        
        with open(idea_file) as f:
            idea = json.load(f)
        
        # Generate outline using LLM
        system_prompt = (
            "You are an expert content strategist who creates detailed article outlines. "
            "Your outlines should be well-structured, comprehensive, and engaging."
        )
        
        user_prompt = f"""Create a detailed outline for an article about '{idea.get('title', '')}'.

        Here is the idea data to inform your outline:
        {json.dumps(idea, indent=2)}
        
        The outline should include:
        1. A compelling introduction
        2. 3-5 main sections with 2-3 subsections each
        3. Key points to cover in each section
        4. A strong conclusion
        
        Format your outline as follows:
        INTRODUCTION: [Brief description of the introduction]
        MAIN_SECTIONS:
        - Section 1 Title:
          - Subsection 1.1: [Description]
          - Subsection 1.2: [Description]
        - Section 2 Title:
          - Subsection 2.1: [Description]
          - Subsection 2.2: [Description]
        CONCLUSION: [Brief description of the conclusion]
        """
        
        try:
            response = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse the outline
            outline = {}
            current_section = None
            current_subsections = []
            
            for line in response.split("\n"):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("INTRODUCTION:"):
                    current_section = "introduction"
                    outline[current_section] = line[14:].strip()
                elif line.startswith("CONCLUSION:"):
                    if current_section:
                        outline[current_section] = current_subsections
                    current_section = "conclusion"
                    outline[current_section] = line[11:].strip()
                elif line.startswith("- "):
                    if current_section and current_section not in ["introduction", "conclusion"]:
                        if ":" in line:
                            title = line[2:].split(":")[0].strip()
                            description = line[2:].split(":")[1].strip()
                            current_subsections.append({
                                "title": title,
                                "description": description
                            })
                        else:
                            current_subsections.append({
                                "title": line[2:].strip(),
                                "description": ""
                            })
                elif line.endswith(":"):
                    if current_section and current_section not in ["introduction", "conclusion"]:
                        outline[current_section] = current_subsections
                    current_section = line[:-1].lower().replace(" ", "_")
                    current_subsections = []
            
            # Add the last section
            if current_section and current_section not in ["introduction", "conclusion"]:
                outline[current_section] = current_subsections
            
            # Save the outline
            outline_file = project_dir / "outline.json"
            with open(outline_file, "w") as f:
                json.dump(outline, f, indent=2)
            
            # Update project metadata
            metadata_file = project_dir / "metadata.json"
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            metadata["status"] = "outline_generated"
            metadata["updated_at"] = idea.get("created_at", "")
            
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Generated outline for project: {project_id}")
            return outline
            
        except Exception as e:
            logger.error(f"Error generating outline: {e}")
            return {}
    
    def generate_paragraphs(self, project_id: str) -> List[Dict[str, Any]]:
        """Generate paragraphs for a project.
        
        Args:
            project_id: ID of the project to generate paragraphs for
            
        Returns:
            List of generated paragraphs
        """
        logger.info(f"Generating paragraphs for project: {project_id}")
        
        # Get project data
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            logger.error(f"Project not found: {project_id}")
            return []
        
        outline_file = project_dir / "outline.json"
        if not outline_file.exists():
            logger.error(f"Project outline not found: {project_id}")
            return []
        
        with open(outline_file) as f:
            outline = json.load(f)
        
        # Generate paragraphs using LLM
        system_prompt = (
            "You are an expert content writer who creates engaging, informative paragraphs. "
            "Your writing should be clear, concise, and well-structured."
        )
        
        paragraphs = []
        
        # Generate introduction
        if "introduction" in outline:
            user_prompt = f"""Write an engaging introduction for an article about '{outline.get('introduction', '')}'.

            The introduction should:
            1. Hook the reader's attention
            2. Provide context for the topic
            3. Outline what the article will cover
            4. Be 2-3 paragraphs long
            """
            
            try:
                response = self.llm_client.chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                
                paragraphs.append({
                    "type": "introduction",
                    "content": response.strip()
                })
                
            except Exception as e:
                logger.error(f"Error generating introduction: {e}")
        
        # Generate main sections
        for section_key, section_data in outline.items():
            if section_key in ["introduction", "conclusion"]:
                continue
            
            if isinstance(section_data, list):
                for subsection in section_data:
                    user_prompt = f"""Write a detailed paragraph for the section '{subsection.get('title', '')}' with the following description:
                    {subsection.get('description', '')}
                    
                    The paragraph should:
                    1. Be informative and engaging
                    2. Include relevant details and examples
                    3. Flow naturally from previous content
                    4. Be 2-3 paragraphs long
                    """
                    
                    try:
                        response = self.llm_client.chat_completion(
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=0.7,
                            max_tokens=500
                        )
                        
                        paragraphs.append({
                            "type": "section",
                            "title": subsection.get("title", ""),
                            "content": response.strip()
                        })
                        
                    except Exception as e:
                        logger.error(f"Error generating section paragraph: {e}")
        
        # Generate conclusion
        if "conclusion" in outline:
            user_prompt = f"""Write a strong conclusion for an article about '{outline.get('conclusion', '')}'.

            The conclusion should:
            1. Summarize key points
            2. Provide final insights
            3. Leave a lasting impression
            4. Be 2-3 paragraphs long
            """
            
            try:
                response = self.llm_client.chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                
                paragraphs.append({
                    "type": "conclusion",
                    "content": response.strip()
                })
                
            except Exception as e:
                logger.error(f"Error generating conclusion: {e}")
        
        # Save the paragraphs
        paragraphs_file = project_dir / "paragraphs.json"
        with open(paragraphs_file, "w") as f:
            json.dump(paragraphs, f, indent=2)
        
        # Update project metadata
        metadata_file = project_dir / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        metadata["status"] = "paragraphs_generated"
        metadata["updated_at"] = outline.get("created_at", "")
        
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Generated paragraphs for project: {project_id}")
        return paragraphs 
"""Article assembler for article generation."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from loguru import logger

from src.llm_client import LLMClient


class ArticleAssembler:
    """Assembles articles from generated content."""
    
    def __init__(self, openai_client: LLMClient, projects_dir: Path):
        """Initialize the article assembler.
        
        Args:
            openai_client: LLM client for API interactions
            projects_dir: Directory to store project data
        """
        self.llm_client = openai_client
        self.projects_dir = projects_dir
    
    def assemble_article(self, project_id: str) -> Dict[str, Any]:
        """Assemble an article from generated content.
        
        Args:
            project_id: ID of the project to assemble article for
            
        Returns:
            Dictionary containing the assembled article
        """
        logger.info(f"Assembling article for project: {project_id}")
        
        # Get project data
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            logger.error(f"Project not found: {project_id}")
            return {}
        
        paragraphs_file = project_dir / "paragraphs.json"
        if not paragraphs_file.exists():
            logger.error(f"Project paragraphs not found: {project_id}")
            return {}
        
        with open(paragraphs_file) as f:
            paragraphs = json.load(f)
        
        # Assemble article using LLM
        system_prompt = (
            "You are an expert editor who assembles and refines articles. "
            "Your task is to combine paragraphs into a cohesive, well-structured article."
        )
        
        # Prepare content for assembly
        content = ""
        for paragraph in paragraphs:
            if paragraph["type"] == "introduction":
                content += f"\nINTRODUCTION:\n{paragraph['content']}\n"
            elif paragraph["type"] == "section":
                content += f"\nSECTION: {paragraph['title']}\n{paragraph['content']}\n"
            elif paragraph["type"] == "conclusion":
                content += f"\nCONCLUSION:\n{paragraph['content']}\n"
        
        user_prompt = f"""Assemble the following content into a cohesive article:

        {content}
        
        The assembled article should:
        1. Flow naturally between sections
        2. Maintain consistent tone and style
        3. Include appropriate transitions
        4. Be well-structured and engaging
        
        Format the article with clear section headings and paragraphs.
        """
        
        try:
            response = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                use_text_generation_model=True
            )
            
            # Parse the assembled article
            article = {
                "content": response.strip(),
                "paragraphs": paragraphs
            }
            
            # Save the assembled article
            article_file = project_dir / "article.json"
            with open(article_file, "w") as f:
                json.dump(article, f, indent=2)
            
            # Update project metadata
            metadata_file = project_dir / "metadata.json"
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            metadata["status"] = "article_assembled"
            metadata["updated_at"] = datetime.now().isoformat()
            
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Assembled article for project: {project_id}")
            return article
            
        except Exception as e:
            logger.error(f"Error assembling article: {e}")
            return {}
    
    def refine_article(self, project_id: str) -> Dict[str, Any]:
        """Refine an assembled article.
        
        Args:
            project_id: ID of the project to refine article for
            
        Returns:
            Dictionary containing the refined article
        """
        logger.info(f"Refining article for project: {project_id}")
        
        # Get project data
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            logger.error(f"Project not found: {project_id}")
            return {}
        
        article_file = project_dir / "article.json"
        if not article_file.exists():
            logger.error(f"Project article not found: {project_id}")
            return {}
        
        with open(article_file) as f:
            article = json.load(f)
        
        # Ensure we have the expected content structure
        if "content" not in article and "introduction" in article:
            # This is likely a draft format with introduction, content, conclusion
            article_content = ""
            if "introduction" in article:
                article_content += article["introduction"] + "\n\n"
            if "content" in article:
                article_content += article["content"] + "\n\n"
            if "conclusion" in article:
                article_content += article["conclusion"]
                
            # Create a compatible article structure
            article = {
                "content": article_content.strip(),
                "paragraphs": article.get("paragraphs", [])
            }
        
        # Refine article using LLM
        system_prompt = (
            "You are an expert editor who refines and polishes articles. "
            "Your task is to improve the article's clarity, flow, and impact."
        )
        
        user_prompt = f"""Refine the following article:

        {article['content']}
        
        The refined article should:
        1. Have improved clarity and readability
        2. Flow more naturally between sections
        3. Use more engaging language
        4. Maintain consistent tone and style
        5. Have stronger transitions
        
        Format the article with clear section headings and paragraphs.
        """
        
        try:
            response = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                use_text_generation_model=True
            )
            
            # Update the article with refined content
            article["content"] = response.strip()
            
            # Save the refined article
            with open(article_file, "w") as f:
                json.dump(article, f, indent=2)
            
            # Update project metadata
            metadata_file = project_dir / "metadata.json"
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            metadata["status"] = "article_refined"
            metadata["updated_at"] = datetime.now().isoformat()
            
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Refined article for project: {project_id}")
            return article
            
        except Exception as e:
            logger.error(f"Error refining article: {e}")
            return {}
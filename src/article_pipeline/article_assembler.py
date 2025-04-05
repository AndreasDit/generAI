"""Article assembler for article generation."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

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
                max_tokens=2000
            )
            
            # Save the assembled article
            article = {
                "content": response.strip(),
                "paragraphs": paragraphs
            }
            
            article_file = project_dir / "article.json"
            with open(article_file, "w") as f:
                json.dump(article, f, indent=2)
            
            # Update project metadata
            metadata_file = project_dir / "metadata.json"
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            metadata["status"] = "article_assembled"
            metadata["updated_at"] = paragraphs[0].get("created_at", "") if paragraphs else ""
            
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
        
        # Check for article in main project directory
        article_file = project_dir / "article.json"
        
        # If not found, check in drafts directory
        if not article_file.exists():
            logger.info(f"Article not found in main directory, checking drafts directory")
            drafts_dir = project_dir / "drafts"
            
            if drafts_dir.exists():
                # Try to find initial_draft.json or refined_draft.json
                initial_draft_file = drafts_dir / "initial_draft.json"
                refined_draft_file = drafts_dir / "refined_draft.json"
                
                if refined_draft_file.exists():
                    article_file = refined_draft_file
                    logger.info(f"Using existing refined draft for project: {project_id}")
                elif initial_draft_file.exists():
                    article_file = initial_draft_file
                    logger.info(f"Using initial draft for project: {project_id}")
        
        if not article_file.exists():
            logger.error(f"No article or draft found for project: {project_id}")
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
                max_tokens=2000
            )
            
            # Save the refined article with title
            # Extract title from original article or metadata
            title = None
            
            # First try to get title from the article itself
            if "title" in article:
                title = article["title"]
            
            # If not found in article, try to get from metadata
            if not title:
                metadata_file = project_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                        if "title" in metadata:
                            title = metadata["title"]
            
            # If still not found, check if it's in the initial draft
            if not title:
                initial_draft_file = drafts_dir / "initial_draft.json"
                if initial_draft_file.exists():
                    try:
                        with open(initial_draft_file) as f:
                            initial_draft = json.load(f)
                            if "title" in initial_draft:
                                title = initial_draft["title"]
                    except Exception as e:
                        logger.error(f"Error reading initial draft: {e}")
            
            # Create refined article with title if found
            refined_article = {
                "content": response.strip(),
                "original_content": article["content"],
                "paragraphs": article.get("paragraphs", [])
            }
            
            # Add title to refined article if found
            if title:
                refined_article["title"] = title
            
            # Ensure drafts directory exists
            drafts_dir = project_dir / "drafts"
            drafts_dir.mkdir(exist_ok=True)
            
            # Save to drafts directory for consistency
            refined_file = drafts_dir / "refined_draft.json"
            with open(refined_file, "w") as f:
                json.dump(refined_article, f, indent=2)
                
            # Also save to project directory for SEO optimization step
            refined_article_file = project_dir / "refined_article.json"
            with open(refined_article_file, "w") as f:
                json.dump(refined_article, f, indent=2)
                
            logger.info(f"Saved refined article to both drafts directory and project directory for project: {project_id}")
            
            # Update project metadata
            metadata_file = project_dir / "metadata.json"
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            metadata["status"] = "article_refined"
            metadata["updated_at"] = article.get("created_at", "")
            
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Refined article for project: {project_id}")
            return refined_article
            
        except Exception as e:
            logger.error(f"Error refining article: {e}")
            return {}
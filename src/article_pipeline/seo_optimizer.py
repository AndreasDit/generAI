"""SEO optimizer for article generation."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from loguru import logger

from src.llm_client import LLMClient


class SEOOptimizer:
    """Optimizes articles for SEO."""
    
    def __init__(self, openai_client: LLMClient, projects_dir: Path):
        """Initialize the SEO optimizer.
        
        Args:
            openai_client: LLM client for API interactions
            projects_dir: Directory to store project data
        """
        self.llm_client = openai_client
        self.projects_dir = projects_dir
    
    def optimize_article(self, project_id: str) -> Dict[str, Any]:
        """Optimize an article for SEO.
        
        Args:
            project_id: ID of the project to optimize article for
            
        Returns:
            Dictionary containing the optimized article
        """
        logger.info(f"Optimizing article for project: {project_id}")
        
        # Get project data
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            logger.error(f"Project not found: {project_id} in {project_dir}")
            return {}
        
        article_file = project_dir / "refined_article.md"
        if not article_file.exists():
            logger.error(f"Project article not found: {project_id} under {article_file}")
            return {}

        idea_file = project_dir / "idea.json"
        if not idea_file.exists():
            logger.error(f"Project article not found: {project_id} under {idea_file}")
            return {}

        # Load files
        with open(article_file) as f:
            article_content = f.read()

        with open(idea_file) as f:
            idea = json.load(f)
        
        # Optimize article using LLM
        system_prompt = (
            "You are an expert SEO specialist who optimizes articles for search engines. "
            "Your task is to enhance the article's SEO while maintaining readability."
        )
        
        user_prompt = f"""
        Assume the role of an SEO specialist focused on optimizing Medium articles for search engines.
        Your task is to develop a comprehensive strategy for enhancing the SEO of posts about {idea.get('title', '') + ' ' + idea.get('description', '')}.
        Start by conducting keyword research to identify high-volume and long-tail keywords relevant to your topic.
        Incorporate these keywords naturally throughout the article, especially in key areas like the title, headings, subheadings, and the first paragraph.
        Ensure the content provides value and answers common questions associated with the keywords, which can improve the chances of appearing in featured snippets and voice search results.
        
        Optimize the following article for SEO:

        {article_content}
                
        Format the optimized article with clear section headings and paragraphs.
        Try to integrate the key words naturally without perform rewrite.
        Try to add the most relevant key words, but ensure the article remains mostly as it is, without making significant changes.
        Provide ONLY the optimized article content without any explanation or additional text.
        """
        
        try:
            response = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                use_text_generation_model=False
            )
            
            # Update the article with refined content
            logger.info(f"Optimizing article for project: {project_id}")
            seo_optimized_article = response.strip()
            
            # Save the refined article
            logger.info(f"Optimizing article for project: {project_id}")
            seo_optimized_article_file = project_dir / "seo_optimized_article.md"
            with open(seo_optimized_article_file, "w") as f:
                f.write(seo_optimized_article)
            logger.info(f"Optimized article for project: {project_id}")
            
            # Update project metadata
            logger.info(f"Optimizing article for project: {project_id}")
            metadata_file = project_dir / "metadata.json"
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            metadata["status"] = "article_optimized"
            metadata["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Optimized article for project: {project_id}")
            return seo_optimized_article
            
        except Exception as e:
            logger.error(f"Error optimizing article: {e}")
            return {}
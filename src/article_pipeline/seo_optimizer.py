"""SEO optimizer for article generation."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

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
        
        refined_file = project_dir / "refined_article.json"
        if not refined_file.exists():
            logger.error(f"Project refined article not found: {project_id} under {refined_file}")
            return {}
        
        with open(refined_file) as f:
            article = json.load(f)
        
        # Optimize article using LLM
        system_prompt = (
            "You are an expert SEO specialist who optimizes articles for search engines. "
            "Your task is to enhance the article's SEO while maintaining readability."
        )
        
        user_prompt = f"""Optimize the following article for SEO:

        {article['content']}
        
        The optimized article should:
        1. Include relevant keywords naturally
        2. Have optimized headings and subheadings
        3. Maintain readability and flow
        4. Include meta description and title suggestions
        5. Have proper internal linking structure
        
        Format the optimized article with clear section headings and paragraphs.
        Also provide SEO metadata suggestions.
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
            
            # Parse SEO metadata and content
            seo_data = {}
            content = []
            metadata = []
            current_section = "content"
            
            for line in response.split("\n"):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("SEO METADATA:"):
                    current_section = "metadata"
                    continue
                
                if current_section == "content":
                    content.append(line)
                else:
                    metadata.append(line)
            
            seo_data["content"] = "\n".join(content)
            seo_data["metadata"] = "\n".join(metadata)
            seo_data["original_content"] = article["content"]
            
            # Include the title from the refined article
            seo_data["title"] = article.get("title", "")
            
            # Save the optimized article
            optimized_file = project_dir / "optimized_article.json"
            with open(optimized_file, "w") as f:
                json.dump(seo_data, f, indent=2)
            
            # Update project metadata
            metadata_file = project_dir / "metadata.json"
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            metadata["status"] = "article_optimized"
            metadata["updated_at"] = article.get("created_at", "")
            
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Optimized article for project: {project_id}")
            return seo_data
            
        except Exception as e:
            logger.error(f"Error optimizing article: {e}")
            return {}
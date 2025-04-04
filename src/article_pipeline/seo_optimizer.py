"""SEO optimization functionality for the article pipeline."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger

from src.openai_client import OpenAIClient

class SEOOptimizer:
    """Handles SEO optimization for articles."""
    
    def __init__(self, openai_client: OpenAIClient, projects_dir: Path):
        """Initialize the SEO optimizer.
        
        Args:
            openai_client: OpenAI client for API interactions
            projects_dir: Directory containing project data
        """
        self.openai_client = openai_client
        self.projects_dir = projects_dir
    
    def optimize_seo(self, project_id: str) -> Optional[Dict[str, str]]:
        """Optimize an article for SEO.
        
        Args:
            project_id: ID of the project to optimize
            
        Returns:
            Dictionary containing the optimized article or None if optimization failed
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
            
            article = project_data.get("final_article")
            if not article:
                logger.error(f"No article found for project: {project_id}")
                return None
            
            idea = project_data.get("idea", {})
            if not idea:
                logger.error(f"No idea data found for project: {project_id}")
                return None
            
            # Generate SEO metadata
            seo_metadata = self._generate_seo_metadata(idea, article)
            if not seo_metadata:
                logger.error("Failed to generate SEO metadata")
                return None
            
            # Optimize content
            optimized_content = self._optimize_content(article, seo_metadata)
            if not optimized_content:
                logger.error("Failed to optimize content")
                return None
            
            # Create optimized article
            optimized_article = {
                "title": seo_metadata["title"],
                "meta_description": seo_metadata["meta_description"],
                "keywords": seo_metadata["keywords"],
                "introduction": optimized_content["introduction"],
                "content": optimized_content["content"],
                "conclusion": optimized_content["conclusion"],
                "seo_score": seo_metadata["seo_score"]
            }
            
            # Save optimized article
            optimized_file = project_dir / "drafts" / "optimized_draft.json"
            with open(optimized_file, "w") as f:
                json.dump(optimized_article, f, indent=2)
            
            # Update project metadata
            project_data["final_article"] = optimized_article
            with open(metadata_file, "w") as f:
                json.dump(project_data, f, indent=2)
            
            logger.info(f"Optimized article for project {project_id}")
            return optimized_article
            
        except Exception as e:
            logger.error(f"Error optimizing article: {e}")
            return None
    
    def _generate_seo_metadata(self, idea: Dict[str, Any], article: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Generate SEO metadata for an article.
        
        Args:
            idea: Project idea data
            article: Article content
            
        Returns:
            Dictionary containing SEO metadata or None if generation failed
        """
        try:
            system_prompt = (
                "You are an expert SEO strategist who optimizes content for search engines. "
                "Your optimizations should improve visibility while maintaining content quality."
            )
            
            user_prompt = f"""Generate SEO metadata for the following article:
            
            Title: {article['title']}
            
            Introduction:
            {article['introduction']}
            
            Main Content:
            {article['content']}
            
            Conclusion:
            {article['conclusion']}
            
            Target Audience: {idea.get('audience', 'No audience specified')}
            Key Points: {idea.get('key_points', 'No key points')}
            
            Please provide:
            1. An SEO-optimized title (50-60 characters)
            2. A compelling meta description (150-160 characters)
            3. 5-7 relevant keywords
            4. An SEO score (0-100) with explanation
            
            Format your response as follows:
            TITLE: [SEO-optimized title]
            META_DESCRIPTION: [Meta description]
            KEYWORDS: [Comma-separated keywords]
            SEO_SCORE: [Score]
            EXPLANATION: [Brief explanation of score]
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
            
            content = response.choices[0].message.content
            
            # Parse SEO metadata
            metadata = {}
            current_key = None
            current_value = []
            
            for line in content.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("TITLE:"):
                    if current_key:
                        metadata[current_key] = "\n".join(current_value)
                    current_key = "title"
                    current_value = [line[6:].strip()]
                elif line.startswith("META_DESCRIPTION:"):
                    if current_key:
                        metadata[current_key] = "\n".join(current_value)
                    current_key = "meta_description"
                    current_value = [line[17:].strip()]
                elif line.startswith("KEYWORDS:"):
                    if current_key:
                        metadata[current_key] = "\n".join(current_value)
                    current_key = "keywords"
                    current_value = [line[9:].strip()]
                elif line.startswith("SEO_SCORE:"):
                    if current_key:
                        metadata[current_key] = "\n".join(current_value)
                    current_key = "seo_score"
                    current_value = [line[10:].strip()]
                elif line.startswith("EXPLANATION:"):
                    if current_key:
                        metadata[current_key] = "\n".join(current_value)
                    current_key = "explanation"
                    current_value = [line[12:].strip()]
                elif current_key:
                    current_value.append(line)
            
            # Add the last key-value pair
            if current_key:
                metadata[current_key] = "\n".join(current_value)
            
            # Convert keywords to list
            if "keywords" in metadata:
                metadata["keywords"] = [k.strip() for k in metadata["keywords"].split(",")]
            
            # Convert SEO score to integer
            if "seo_score" in metadata:
                try:
                    metadata["seo_score"] = int(metadata["seo_score"])
                except ValueError:
                    metadata["seo_score"] = 0
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating SEO metadata: {e}")
            return None
    
    def _optimize_content(self, article: Dict[str, str], seo_metadata: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Optimize article content for SEO.
        
        Args:
            article: Original article content
            seo_metadata: Generated SEO metadata
            
        Returns:
            Dictionary containing optimized content or None if optimization failed
        """
        try:
            system_prompt = (
                "You are an expert content optimizer who enhances articles for SEO. "
                "Your optimizations should improve search visibility while maintaining readability."
            )
            
            user_prompt = f"""Optimize the following article for SEO:
            
            Original Title: {article['title']}
            SEO Title: {seo_metadata['title']}
            Keywords: {', '.join(seo_metadata['keywords'])}
            
            Introduction:
            {article['introduction']}
            
            Main Content:
            {article['content']}
            
            Conclusion:
            {article['conclusion']}
            
            Please optimize the content to:
            1. Naturally incorporate target keywords
            2. Improve heading structure
            3. Add relevant internal linking suggestions
            4. Enhance readability and scannability
            5. Maintain the original message and tone
            
            Format the optimized content with clear section breaks.
            """
            
            response = self.openai_client.client.chat.completions.create(
                model=self.openai_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
            )
            
            optimized_content = response.choices[0].message.content
            
            # Parse optimized content
            sections = optimized_content.strip().split("\n\n")
            return {
                "introduction": sections[0] if sections else "",
                "content": "\n\n".join(sections[1:-1]) if len(sections) > 2 else "",
                "conclusion": sections[-1] if len(sections) > 1 else ""
            }
            
        except Exception as e:
            logger.error(f"Error optimizing content: {e}")
            return None 
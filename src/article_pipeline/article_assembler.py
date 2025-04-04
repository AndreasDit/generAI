"""Article assembly and refinement functionality for the article pipeline."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger

from src.openai_client import OpenAIClient

class ArticleAssembler:
    """Handles article assembly and refinement."""
    
    def __init__(self, openai_client: OpenAIClient, projects_dir: Path):
        """Initialize the article assembler.
        
        Args:
            openai_client: OpenAI client for API interactions
            projects_dir: Directory containing project data
        """
        self.openai_client = openai_client
        self.projects_dir = projects_dir
    
    def assemble_article(self, project_id: str) -> Optional[Dict[str, str]]:
        """Assemble the final article from generated paragraphs.
        
        Args:
            project_id: ID of the project to assemble article for
            
        Returns:
            Dictionary containing the assembled article or None if assembly failed
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
            
            paragraphs = project_data.get("paragraphs", [])
            if not paragraphs:
                logger.error(f"No paragraphs found for project: {project_id}")
                return None
            
            idea = project_data.get("idea", {})
            if not idea:
                logger.error(f"No idea data found for project: {project_id}")
                return None
            
            # Generate introduction
            introduction = self._generate_introduction(idea, paragraphs)
            if not introduction:
                logger.error("Failed to generate introduction")
                return None
            
            # Generate conclusion
            conclusion = self._generate_conclusion(idea, paragraphs)
            if not conclusion:
                logger.error("Failed to generate conclusion")
                return None
            
            # Assemble article
            article = {
                "title": idea.get("title", "Untitled Article"),
                "introduction": introduction,
                "content": "\n\n".join(p["content"] for p in paragraphs),
                "conclusion": conclusion
            }
            
            # Save article
            article_file = project_dir / "drafts" / "initial_draft.json"
            with open(article_file, "w") as f:
                json.dump(article, f, indent=2)
            
            # Update project metadata
            project_data["final_article"] = article
            with open(metadata_file, "w") as f:
                json.dump(project_data, f, indent=2)
            
            logger.info(f"Assembled article for project {project_id}")
            return article
            
        except Exception as e:
            logger.error(f"Error assembling article: {e}")
            return None
    
    def refine_article(self, project_id: str) -> Optional[Dict[str, str]]:
        """Refine the assembled article.
        
        Args:
            project_id: ID of the project to refine article for
            
        Returns:
            Dictionary containing the refined article or None if refinement failed
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
            
            # Refine article using OpenAI
            system_prompt = (
                "You are an expert editor who refines articles for clarity, flow, and impact. "
                "Your refinements should enhance readability while maintaining the original message."
            )
            
            user_prompt = f"""Refine the following article:
            
            Title: {article['title']}
            
            Introduction:
            {article['introduction']}
            
            Main Content:
            {article['content']}
            
            Conclusion:
            {article['conclusion']}
            
            Target Audience: {idea.get('audience', 'No audience specified')}
            Key Points: {idea.get('key_points', 'No key points')}
            
            Please refine the article to:
            1. Improve flow and transitions between sections
            2. Enhance clarity and readability
            3. Strengthen the main arguments
            4. Add relevant examples or data points
            5. Ensure consistent tone and style
            6. Fix any grammatical or structural issues
            
            Format the refined article with clear section breaks.
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
            
            refined_content = response.choices[0].message.content
            
            # Parse refined content
            sections = refined_content.strip().split("\n\n")
            refined_article = {
                "title": article["title"],
                "introduction": sections[0] if sections else "",
                "content": "\n\n".join(sections[1:-1]) if len(sections) > 2 else "",
                "conclusion": sections[-1] if len(sections) > 1 else ""
            }
            
            # Save refined article
            refined_file = project_dir / "drafts" / "refined_draft.json"
            with open(refined_file, "w") as f:
                json.dump(refined_article, f, indent=2)
            
            # Update project metadata
            project_data["final_article"] = refined_article
            with open(metadata_file, "w") as f:
                json.dump(project_data, f, indent=2)
            
            logger.info(f"Refined article for project {project_id}")
            return refined_article
            
        except Exception as e:
            logger.error(f"Error refining article: {e}")
            return None
    
    def _generate_introduction(self, idea: Dict[str, Any], paragraphs: List[Dict[str, Any]]) -> Optional[str]:
        """Generate an introduction for the article.
        
        Args:
            idea: Project idea data
            paragraphs: List of generated paragraphs
            
        Returns:
            Generated introduction text or None if generation failed
        """
        try:
            system_prompt = (
                "You are an expert content writer who creates engaging introductions. "
                "Your introductions should hook readers and set up the article's main points."
            )
            
            # Get main points from paragraphs
            main_points = []
            for p in paragraphs:
                if p["type"] == "main":
                    main_points.append(p["section"])
            
            user_prompt = f"""Write an engaging introduction for an article with the following details:
            
            Title: {idea.get('title', 'No title')}
            Target Audience: {idea.get('audience', 'No audience specified')}
            Key Points: {idea.get('key_points', 'No key points')}
            
            Main Sections:
            {chr(10).join(f"- {point}" for point in main_points)}
            
            The introduction should:
            1. Hook the reader with an engaging opening
            2. Establish the article's relevance
            3. Preview the main points
            4. Set the tone for the article
            5. Be 2-3 paragraphs long
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
            
            introduction = response.choices[0].message.content.strip()
            return introduction
            
        except Exception as e:
            logger.error(f"Error generating introduction: {e}")
            return None
    
    def _generate_conclusion(self, idea: Dict[str, Any], paragraphs: List[Dict[str, Any]]) -> Optional[str]:
        """Generate a conclusion for the article.
        
        Args:
            idea: Project idea data
            paragraphs: List of generated paragraphs
            
        Returns:
            Generated conclusion text or None if generation failed
        """
        try:
            system_prompt = (
                "You are an expert content writer who creates impactful conclusions. "
                "Your conclusions should summarize key points and leave a lasting impression."
            )
            
            # Get main points from paragraphs
            main_points = []
            for p in paragraphs:
                if p["type"] == "main":
                    main_points.append(p["section"])
            
            user_prompt = f"""Write a strong conclusion for an article with the following details:
            
            Title: {idea.get('title', 'No title')}
            Target Audience: {idea.get('audience', 'No audience specified')}
            Key Points: {idea.get('key_points', 'No key points')}
            
            Main Sections:
            {chr(10).join(f"- {point}" for point in main_points)}
            
            The conclusion should:
            1. Summarize the main points
            2. Reinforce the key message
            3. Provide a call to action or next steps
            4. Leave a lasting impression
            5. Be 2-3 paragraphs long
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
            
            conclusion = response.choices[0].message.content.strip()
            return conclusion
            
        except Exception as e:
            logger.error(f"Error generating conclusion: {e}")
            return None 
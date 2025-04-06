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
            
        # Load idea and outline data to provide more context to the LLM
        idea_file = project_dir / "idea.json"
        outline_file = project_dir / "outline.json"
        
        idea_data = {}
        outline_data = []
        
        if idea_file.exists():
            try:
                with open(idea_file) as f:
                    idea_data = json.load(f)
                logger.info(f"Loaded idea data for project: {project_id}")
            except Exception as e:
                logger.error(f"Error loading idea data: {e}")
        else:
            logger.warning(f"Idea file not found for project: {project_id}")
            
        if outline_file.exists():
            try:
                with open(outline_file) as f:
                    outline_data = json.load(f)
                logger.info(f"Loaded outline data for project: {project_id}")
            except Exception as e:
                logger.error(f"Error loading outline data: {e}")
        else:
            logger.warning(f"Outline file not found for project: {project_id}")
        
        # Assemble article using LLM
        system_prompt = (
            "You are an expert editor who assembles and refines articles. "
            "Your task is to combine paragraphs into a cohesive, well-structured article. "
            "You must use the provided article idea and outline as a guide to ensure "
            "the final article aligns with the original concept and follows the intended structure."
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
        
        # Prepare idea and outline context for the prompt
        idea_context = ""
        if idea_data:
            idea_context = f"""ARTICLE IDEA:
        Title: {idea_data.get('title', 'No title')}
        Description: {idea_data.get('description', 'No description')}
        Target Audience: {idea_data.get('target_audience', 'General audience')}
        Key Points: {', '.join(idea_data.get('key_points', []))}
        Value Proposition: {idea_data.get('value_proposition', 'No value proposition')}
        """
        
        outline_context = ""
        if outline_data:
            outline_sections = []
            for i, section in enumerate(outline_data, 1):
                if isinstance(section, dict) and 'title' in section:
                    outline_sections.append(f"{i}. {section['title']}")
                elif isinstance(section, str):
                    outline_sections.append(f"{i}. {section}")
            
            outline_context = "ARTICLE OUTLINE:\n" + "\n".join(outline_sections)
        
        user_prompt = f"""Assemble the following content into a cohesive article.

        This is the article idea:
        {idea_context}
        
        This is the article outline:
        {outline_context}
        
        This is the content to assemble:
        {content}
        
        The assembled article should:
        1. Flow naturally between sections
        2. Maintain consistent tone and style
        3. Include appropriate transitions
        4. Be well-structured and engaging
        5. Align with the original article idea and follow the outline structure
        6. Address the target audience appropriately
        7. Deliver on the value proposition
        
        Format the article with clear section headings and paragraphs.
        Use all the provided inputs. Combine the paragraphs from content into a cohesive and well-structured article while using the idea and outline to guide the structure.
        Do NOT write a json file. Instead write a normal readable article. Use markdown formatting.
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
            
            # Parse the assembled article
            article = response.strip()
            
            # Save the assembled article
            article_file = project_dir / "article.md"
            with open(article_file, "w") as f:
                f.write(article)
            
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
        
        article_file = project_dir / "article.md"
        if not article_file.exists():
            logger.error(f"Project article not found: {project_id}")
            return {}
        
        with open(article_file) as f:
            article = f.read()
            
        # Load idea and outline data to provide more context to the LLM
        idea_file = project_dir / "idea.json"
        
        idea_data = {}
        
        if idea_file.exists():
            try:
                with open(idea_file) as f:
                    idea_data = json.load(f)
                logger.info(f"Loaded idea data for project: {project_id}")
            except Exception as e:
                logger.error(f"Error loading idea data: {e}")
        else:
            logger.warning(f"Idea file not found for project: {project_id}")
                    
        # Refine article using LLM
        system_prompt = (
            "You are an expert editor who refines and polishes articles. "
            "Your task is to improve the article's clarity, flow, and impact while ensuring "
            "it aligns with the original article idea and follows the intended structure."
        )
        
        # Prepare idea and outline context for the prompt
        idea_context = ""
        if idea_data:
            idea_context = f"""ARTICLE IDEA:
        Title: {idea_data.get('title', 'No title')}
        Description: {idea_data.get('description', 'No description')}
        Target Audience: {idea_data.get('target_audience', 'General audience')}
        Key Points: {', '.join(idea_data.get('key_points', []))}
        Value Proposition: {idea_data.get('value_proposition', 'No value proposition')}
        """
        
        user_prompt = f"""Refine the following article:

        This is the article idea:
        {idea_context}        
                
        This is the article content:
        {article['content']}
                
        The refined article should:
        1. Have improved clarity and readability
        2. Flow more naturally between sections
        3. Use more engaging language
        4. Maintain consistent tone and style
        5. Have stronger transitions
        6. Align with the original article idea
        7. Address the target audience appropriately
        8. Deliver on the value proposition
        
        Finally, follow following these guidelines:
        a) Use a conversational tone, concise language and avoid unnecessarily complex jargon. Example: "Hey friends, today I'll show you a really useful writing tip"
        b) Use short punchy sentences. Example: "And then… you enter the room. Your heart drops. The pressure is on."
        c) Use simple language. 7th grade readability or lower. Example: "Emails help businesses tell customers about their stuff."
        d) Use rhetorical fragments to improve readability. Example: “The good news? My 3-step process can be applied to any business"
        e) Use bullet points when relevant. Example: “Because anytime someone loves your product, chances are they’ll:
        * buy from you again
        * refer you to their friends"
        f) Use analogies or examples often. Example: "Creating an email course with AI is easier than stealing candies from a baby"
        g) Split up long sentences. Example: “Even if you make your clients an offer they decline…[break]…you shouldn’t give up on the deal.”
        h) Include personal anecdotes. Example: "I recently asked ChatGPT to write me…"
        i) Use bold and italic formatting to emphasize words.
        j) Do not use emojis or hashtags
        k) Avoid overly promotional words like "game-changing," "unlock," "master," "skyrocket," or "revolutionize."

        Remember, the goal is to make the text sound natural, engaging, and as if it were written by a human rather than an AI.
        
        Format the article with clear section headings and paragraphs.
        Use the content as the primary input to create a cohesive and well-structured article. Use the article idea as context to ensure the article aligns with the original idea.
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
            article = response.strip()
            
            # Save the refined article
            refined_article_file = project_dir / "refined_article.md"
            with open(refined_article_file, "w") as f:
                f.write(article)
            
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
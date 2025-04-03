#!/usr/bin/env python3
"""
OpenAI Client for GenerAI

This module handles interaction with OpenAI API for article generation.
"""

from typing import Dict, List, Optional

import openai
from loguru import logger


class OpenAIClient:
    """Client for interacting with OpenAI API."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
        """
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
        logger.info(f"OpenAI client initialized with model: {model}")
    
    def generate_article(self, topic: str, tone: str = "informative", 
                        length: str = "medium", outline: Optional[List[str]] = None) -> Dict[str, str]:
        """Generate an article using OpenAI.
        
        Args:
            topic: The main topic of the article
            tone: The tone of the article (informative, casual, professional, etc.)
            length: The length of the article (short, medium, long)
            outline: Optional outline of sections to include
        
        Returns:
            Dictionary containing title and content of the article
        """
        # Define length in words
        length_map = {
            "short": "800-1000 words",
            "medium": "1500-2000 words",
            "long": "2500-3000 words"
        }
        word_count = length_map.get(length, "1500-2000 words")
        
        # Construct the prompt
        system_prompt = (
            "You are an expert content writer who creates well-researched, engaging articles. "
            "Your articles should be informative, well-structured, and provide value to readers."
        )
        
        outline_text = ""
        if outline:
            outline_text = "\n\nPlease include these sections in your article:\n" + "\n".join([f"- {section}" for section in outline])
        
        user_prompt = f"""Write a {tone} article about '{topic}' that is approximately {word_count}.
        
        The article should have:
        1. An engaging title
        2. A compelling introduction
        3. Well-structured body with subheadings
        4. A conclusion that summarizes key points
        5. A call to action if appropriate
        {outline_text}
        
        Format the article in Markdown with appropriate headings, bullet points, and emphasis where needed.
        
        Return the article in this format:
        TITLE: [Your title here]
        
        [Full article content in Markdown]
        """
        
        try:
            logger.info(f"Generating article about '{topic}' with {word_count}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=4000,
            )
            
            content = response.choices[0].message.content
            
            # Extract title and content
            if "TITLE:" in content:
                title_parts = content.split("TITLE:", 1)
                title = title_parts[1].split("\n", 1)[0].strip()
                content = title_parts[1].split("\n", 1)[1].strip() if len(title_parts[1].split("\n", 1)) > 1 else ""
            else:
                # If no TITLE marker, try to extract the first heading
                lines = content.split("\n")
                title = ""
                for line in lines:
                    if line.startswith("# "):
                        title = line.replace("# ", "").strip()
                        break
                if not title and lines:
                    title = lines[0].strip()  # Use first line as title if no heading found
            
            logger.info(f"Generated article with title: '{title}'")
            return {"title": title, "content": content}
            
        except Exception as e:
            logger.error(f"Error generating article: {e}")
            return {"title": "", "content": ""}
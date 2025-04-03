#!/usr/bin/env python3
"""
OpenAI Client for GenerAI

This module handles interaction with OpenAI API for article generation.
Implements caching to avoid redundant API calls and improve performance.
"""

from typing import Dict, List, Optional, Any

import openai
from loguru import logger

from src.cache_manager import CacheManager


class OpenAIClient:
    """Client for interacting with OpenAI API with caching support."""
    
    def __init__(self, api_key: str, model: str = "gpt-4", use_cache: bool = True, cache_ttl_days: int = 7):
        """Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
            use_cache: Whether to use caching for API calls
            cache_ttl_days: Time-to-live for cache entries in days
        """
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
        self.use_cache = use_cache
        
        # Initialize cache manager if caching is enabled
        if self.use_cache:
            self.cache_manager = CacheManager(cache_dir="cache", cache_ttl_days=cache_ttl_days)
            logger.info(f"OpenAI client initialized with model: {model} (caching enabled)")
        else:
            self.cache_manager = None
            logger.info(f"OpenAI client initialized with model: {model} (caching disabled)")
    
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
        
        # Create request parameters for cache lookup
        request_params = {
            "type": "article_generation",
            "topic": topic,
            "tone": tone,
            "length": length,
            "outline": outline,
            "model": self.model
        }
        
        try:
            # Check cache if enabled
            if self.use_cache and self.cache_manager:
                cached_response = self.cache_manager.get(request_params)
                if cached_response:
                    logger.info(f"Using cached article about '{topic}' with title: '{cached_response['title']}'")
                    return cached_response
            
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
            
            # Create response object
            article_response = {"title": title, "content": content}
            
            # Cache the response if caching is enabled
            if self.use_cache and self.cache_manager:
                self.cache_manager.set(request_params, article_response)
            
            logger.info(f"Generated article with title: '{title}'")
            return article_response
            
        except Exception as e:
            logger.error(f"Error generating article: {e}")
            return {"title": "", "content": ""}
    
    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7, 
                       max_tokens: int = 1000) -> Optional[str]:
        """Make a chat completion API call with caching support.
        
        Args:
            messages: List of message dictionaries with role and content
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text or None if an error occurred
        """
        # Create request parameters for cache lookup
        request_params = {
            "type": "chat_completion",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "model": self.model
        }
        
        try:
            # Check cache if enabled
            if self.use_cache and self.cache_manager:
                cached_response = self.cache_manager.get(request_params)
                if cached_response:
                    logger.info("Using cached chat completion response")
                    return cached_response.get("content")
            
            logger.info("Making chat completion API call")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            content = response.choices[0].message.content
            
            # Cache the response if caching is enabled
            if self.use_cache and self.cache_manager:
                self.cache_manager.set(request_params, {"content": content})
            
            return content
            
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return None
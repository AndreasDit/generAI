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
            
    def generate_article_ideas(self, research_topic: str, trend_analysis: Dict[str, Any] = None, 
                             competitor_research: Dict[str, Any] = {}, num_ideas: int = 5) -> List[Dict[str, Any]]:
        """Generate article ideas based on research topic, trend analysis, and competitor research.
        
        Args:
            research_topic: General topic to research for ideas
            trend_analysis: Optional trend analysis data
            competitor_research: Optional competitor research data
            num_ideas: Number of ideas to generate
            
        Returns:
            List of generated ideas with metadata
        """
        logger.info(f"Generating {num_ideas} article ideas for topic: {research_topic}")
        
        # Construct the prompt for idea generation
        system_prompt = (
            "You are an expert content strategist who generates engaging article ideas. "
            "Your ideas should be specific, valuable to readers, and have potential for high engagement."
        )
        
        # Include insights from trend analysis and competitor research
        trend_insights = ""
        if trend_analysis:
            trend_insights = f"""\n\nTrend Analysis Insights:\n"""
            if "trending_subtopics" in trend_analysis:
                trend_insights += f"Trending Subtopics: {trend_analysis['trending_subtopics']}\n"
            if "key_questions" in trend_analysis:
                trend_insights += f"Key Questions: {trend_analysis['key_questions']}\n"
            if "recent_developments" in trend_analysis:
                trend_insights += f"Recent Developments: {trend_analysis['recent_developments']}\n"
        
        competitor_insights = ""
        if competitor_research:
            competitor_insights = f"""\n\nCompetitor Research Insights:\n"""
            if "common_themes" in competitor_research:
                competitor_insights += f"Common Themes: {competitor_research['common_themes']}\n"
            if "content_gaps" in competitor_research:
                competitor_insights += f"Content Gaps: {competitor_research['content_gaps']}\n"
            if "differentiation_opportunities" in competitor_research:
                competitor_insights += f"Differentiation Opportunities: {competitor_research['differentiation_opportunities']}\n"
        
        user_prompt = f"""Generate {num_ideas} article ideas related to '{research_topic}'.{trend_insights}{competitor_insights}
        
        For each idea, provide:
        1. A compelling title
        2. A brief summary (2-3 sentences)
        3. Target audience
        4. Key keywords (5-7 keywords)
        
        Format each idea as follows:
        TITLE: [Title]
        SUMMARY: [Summary]
        AUDIENCE: [Target audience]
        KEYWORDS: [keyword1, keyword2, keyword3, ...]
        
        Make each idea unique and specific. Focus on providing value to readers and addressing their needs.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=2000,
            )
            
            content = response.choices[0].message.content
            
            # Parse the ideas
            ideas = []
            current_idea = {}
            lines = content.strip().split("\n")
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith("TITLE:"):
                    # If we have a current idea, add it to the list
                    if current_idea and "title" in current_idea:
                        ideas.append(current_idea)
                    
                    # Start a new idea
                    current_idea = {"title": line[6:].strip()}
                elif line.startswith("SUMMARY:") and current_idea:
                    current_idea["summary"] = line[8:].strip()
                elif line.startswith("AUDIENCE:") and current_idea:
                    current_idea["audience"] = line[9:].strip()
                elif line.startswith("KEYWORDS:") and current_idea:
                    keywords_text = line[9:].strip()
                    keywords = [k.strip() for k in keywords_text.split(",")]
                    current_idea["keywords"] = keywords
            
            # Add the last idea if it exists
            if current_idea and "title" in current_idea:
                ideas.append(current_idea)
            
            # Limit to the requested number of ideas
            ideas = ideas[:num_ideas]
            
            logger.info(f"Generated {len(ideas)} article ideas")
            return ideas
            
        except Exception as e:
            logger.error(f"Error generating article ideas: {e}")
            return []
    
    def evaluate_article_ideas(self, ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate article ideas and score them based on potential engagement and value.
        
        Args:
            ideas: List of article ideas to evaluate
            
        Returns:
            List of evaluated ideas with scores and feedback
        """
        logger.info(f"Evaluating {len(ideas)} article ideas")
        
        if not ideas:
            logger.warning("No ideas provided for evaluation")
            return []
        
        # Construct the prompt for idea evaluation
        system_prompt = (
            "You are an expert content strategist who evaluates article ideas. "
            "Your evaluations should be based on potential engagement, value to readers, "
            "uniqueness, and feasibility."
        )
        
        # Format ideas for the prompt
        ideas_text = ""
        for i, idea in enumerate(ideas, 1):
            ideas_text += f"\nIDEA {i}:\n"
            ideas_text += f"ID: {idea.get('id', f'idea{i}')}\n"
            ideas_text += f"TITLE: {idea.get('title', 'No title')}\n"
            ideas_text += f"SUMMARY: {idea.get('summary', 'No summary')}\n"
            ideas_text += f"AUDIENCE: {idea.get('audience', 'No audience')}\n"
            keywords = idea.get('keywords', [])
            ideas_text += f"KEYWORDS: {', '.join(keywords) if keywords else 'None'}\n"
        
        user_prompt = f"""Evaluate the following article ideas:{ideas_text}
        
        For each idea, provide:
        1. A score from 0-100 based on potential engagement, value, uniqueness, and feasibility
        2. Brief feedback explaining the score and suggesting improvements
        
        Format your evaluation as follows for each idea:
        IDEA_ID: [ID from the input]
        SCORE: [0-100]
        FEEDBACK: [Your feedback and suggestions]
        
        Be objective and constructive in your evaluations.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=2000,
            )
            
            content = response.choices[0].message.content
            
            # Parse the evaluations
            evaluations = []
            current_eval = {}
            lines = content.strip().split("\n")
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith("IDEA_ID:"):
                    # If we have a current evaluation, add it to the list
                    if current_eval and "id" in current_eval:
                        evaluations.append(current_eval)
                    
                    # Start a new evaluation
                    current_eval = {"id": line[8:].strip()}
                elif line.startswith("SCORE:") and current_eval:
                    try:
                        score = int(line[6:].strip())
                        current_eval["score"] = score
                    except ValueError:
                        current_eval["score"] = 0
                elif line.startswith("FEEDBACK:") and current_eval:
                    current_eval["feedback"] = line[9:].strip()
            
            # Add the last evaluation if it exists
            if current_eval and "id" in current_eval:
                evaluations.append(current_eval)
            
            logger.info(f"Evaluated {len(evaluations)} article ideas")
            return evaluations
            
        except Exception as e:
            logger.error(f"Error evaluating article ideas: {e}")
            return []
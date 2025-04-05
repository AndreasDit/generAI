#!/usr/bin/env python3
"""
OpenAI Client for GenerAI

This module handles interaction with OpenAI API for article generation.
Implements caching to avoid redundant API calls and improve performance.
"""

from typing import Dict, List, Optional, Any
import re
from datetime import datetime

import openai
from loguru import logger

from src.cache_manager import CacheManager


class OpenAIClient:
    """Client for interacting with OpenAI API with caching support."""
    
    def __init__(self, api_key: str, model: str = "gpt-4", use_cache: bool = True, 
                 cache_ttl_days: int = 7, temperature: float = 0.7, max_tokens: int = 2000,
                 cache_dir: str = "cache"):
        """Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
            use_cache: Whether to use caching for API calls
            cache_ttl_days: Time-to-live for cache entries in days
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            cache_dir: Directory for caching API responses
        """
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
        self.use_cache = use_cache
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize cache manager if caching is enabled
        if self.use_cache:
            self.cache_manager = CacheManager(cache_dir=cache_dir, ttl_days=cache_ttl_days)
            logger.info(f"OpenAI client initialized with model: {model} (caching enabled)")
        else:
            self.cache_manager = None
            logger.info(f"OpenAI client initialized with model: {model} (caching disabled)")
    
    def generate_article(self, topic: str, tone: str = "informative", 
                        length: str = "medium", outline: Optional[List[str]] = None,
                        temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Dict[str, str]:
        """Generate an article using OpenAI.
        
        Args:
            topic: The main topic of the article
            tone: The tone of the article (informative, casual, professional, etc.)
            length: The length of the article (short, medium, long)
            outline: Optional outline of sections to include
            temperature: Sampling temperature (0.0 to 1.0), defaults to instance value
            max_tokens: Maximum number of tokens to generate, defaults to instance value
        
        Returns:
            Dictionary containing title and content of the article
        """
        # Use instance values if not specified
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
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
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens
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
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Log token usage
            if hasattr(response, 'usage'):
                logger.info(f"Token usage - Model: {self.model}, Prompt tokens: {response.usage.prompt_tokens}, Completion tokens: {response.usage.completion_tokens}, Total tokens: {response.usage.total_tokens}")
            
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
    
    def chat_completion(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, 
                       max_tokens: Optional[int] = None) -> Optional[str]:
        """Make a chat completion API call with caching support.
        
        Args:
            messages: List of message dictionaries with role and content
            temperature: Sampling temperature (0.0 to 1.0), defaults to instance value
            max_tokens: Maximum number of tokens to generate, defaults to instance value
            
        Returns:
            Generated text or None if an error occurred
        """
        # Use instance values if not specified
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
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
            
            # Log token usage
            if hasattr(response, 'usage'):
                logger.info(f"Token usage - Model: {self.model}, Prompt tokens: {response.usage.prompt_tokens}, Completion tokens: {response.usage.completion_tokens}, Total tokens: {response.usage.total_tokens}")
            
            content = response.choices[0].message.content
            
            # Cache the response if caching is enabled
            if self.use_cache and self.cache_manager:
                self.cache_manager.set(request_params, {"content": content})
            
            return content
            
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return None
            
    def generate_article_ideas(self, research_topic: str, trend_analysis: Dict[str, Any] = None, 
                             competitor_research: Dict[str, Any] = {}, num_ideas: int = 5,
                             temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate article ideas based on research topic, trend analysis, and competitor research.
        
        Args:
            research_topic: General topic to research for ideas
            trend_analysis: Optional trend analysis data
            competitor_research: Optional competitor research data
            num_ideas: Number of ideas to generate
            temperature: Sampling temperature (0.0 to 1.0), defaults to instance value
            max_tokens: Maximum number of tokens to generate, defaults to instance value
            
        Returns:
            List of generated ideas with metadata
        """
        # Use instance values if not specified
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
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
        IDEA 1:
        TITLE: [Article title]
        SUMMARY: [Brief summary]
        AUDIENCE: [Target audience]
        KEYWORDS: [Comma-separated keywords]
        ---
        IDEA 2:
        ...
        """
        
        # Create request parameters for cache lookup
        request_params = {
            "type": "idea_generation",
            "research_topic": research_topic,
            "trend_analysis": trend_analysis,
            "competitor_research": competitor_research,
            "num_ideas": num_ideas,
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            # Check cache if enabled
            if self.use_cache and self.cache_manager:
                cached_response = self.cache_manager.get(request_params)
                if cached_response:
                    logger.info(f"Using cached ideas for topic: '{research_topic}'")
                    return cached_response
            
            logger.info(f"Generating {num_ideas} ideas for topic: '{research_topic}'")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Log token usage
            if hasattr(response, 'usage'):
                logger.info(f"Token usage - Model: {self.model}, Prompt tokens: {response.usage.prompt_tokens}, Completion tokens: {response.usage.completion_tokens}, Total tokens: {response.usage.total_tokens}")
            
            content = response.choices[0].message.content
            
            # Parse the ideas from the response
            ideas = []
            idea_blocks = content.split("---")
            
            for block in idea_blocks:
                if not block.strip():
                    continue
                
                idea = {}
                
                # Extract title
                title_match = re.search(r"TITLE:\s*(.*?)(?:\n|$)", block)
                if title_match:
                    idea["title"] = title_match.group(1).strip()
                
                # Extract summary
                summary_match = re.search(r"SUMMARY:\s*(.*?)(?:\n|$)", block)
                if summary_match:
                    idea["summary"] = summary_match.group(1).strip()
                
                # Extract audience
                audience_match = re.search(r"AUDIENCE:\s*(.*?)(?:\n|$)", block)
                if audience_match:
                    idea["audience"] = audience_match.group(1).strip()
                
                # Extract keywords
                keywords_match = re.search(r"KEYWORDS:\s*(.*?)(?:\n|$)", block)
                if keywords_match:
                    keywords_str = keywords_match.group(1).strip()
                    idea["keywords"] = [k.strip() for k in keywords_str.split(",")]
                
                # Add metadata
                idea["research_topic"] = research_topic
                idea["created_at"] = datetime.now().isoformat()
                
                ideas.append(idea)
            
            # Cache the response if caching is enabled
            if self.use_cache and self.cache_manager:
                self.cache_manager.set(request_params, ideas)
            
            logger.info(f"Generated {len(ideas)} ideas for topic: '{research_topic}'")
            return ideas
            
        except Exception as e:
            logger.error(f"Error generating ideas: {e}")
            return []
    
    def evaluate_article_ideas(self, ideas: List[Dict[str, Any]], temperature: Optional[float] = None, 
                             max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """Evaluate a list of article ideas and rank them by potential.
        
        Args:
            ideas: List of article ideas to evaluate
            temperature: Sampling temperature (0.0 to 1.0), defaults to instance value
            max_tokens: Maximum number of tokens to generate, defaults to instance value
            
        Returns:
            List of evaluated ideas with scores and feedback
        """
        # Use instance values if not specified
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        if not ideas:
            logger.warning("No ideas provided for evaluation")
            return []
        
        logger.info(f"Evaluating {len(ideas)} article ideas")
        
        # Construct the prompt for idea evaluation
        system_prompt = (
            "You are an expert content strategist who evaluates article ideas. "
            "Your evaluation should consider factors like uniqueness, value to readers, "
            "searchability, and potential for engagement."
        )
        
        # Format the ideas for evaluation
        ideas_text = ""
        for i, idea in enumerate(ideas, 1):
            ideas_text += f"IDEA {i}:\n"
            ideas_text += f"TITLE: {idea.get('title', 'No title')}\n"
            ideas_text += f"SUMMARY: {idea.get('summary', 'No summary')}\n"
            ideas_text += f"AUDIENCE: {idea.get('audience', 'No audience specified')}\n"
            ideas_text += f"KEYWORDS: {', '.join(idea.get('keywords', []))}\n"
            ideas_text += "---\n"
        
        user_prompt = f"""Evaluate the following article ideas and rank them by potential success.
        
        {ideas_text}
        
        For each idea, provide:
        1. A score from 1-10 (10 being best)
        2. Strengths (2-3 points)
        3. Weaknesses (2-3 points)
        4. Suggestions for improvement
        
        Format your evaluation as follows:
        IDEA 1:
        SCORE: [1-10]
        STRENGTHS: [List of strengths]
        WEAKNESSES: [List of weaknesses]
        SUGGESTIONS: [Improvement suggestions]
        ---
        IDEA 2:
        ...
        """
        
        # Create request parameters for cache lookup
        request_params = {
            "type": "idea_evaluation",
            "ideas": ideas,
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            # Check cache if enabled
            if self.use_cache and self.cache_manager:
                cached_response = self.cache_manager.get(request_params)
                if cached_response:
                    logger.info("Using cached idea evaluation")
                    return cached_response
            
            logger.info("Evaluating article ideas")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Log token usage
            if hasattr(response, 'usage'):
                logger.info(f"Token usage - Model: {self.model}, Prompt tokens: {response.usage.prompt_tokens}, Completion tokens: {response.usage.completion_tokens}, Total tokens: {response.usage.total_tokens}")
            
            content = response.choices[0].message.content
            
            # Parse the evaluations
            evaluated_ideas = []
            idea_blocks = content.split("---")
            
            for i, block in enumerate(idea_blocks):
                if not block.strip() or i >= len(ideas):
                    continue
                
                # Get the original idea
                original_idea = ideas[i]
                evaluated_idea = original_idea.copy()
                
                # Extract score
                score_match = re.search(r"SCORE:\s*(\d+)", block)
                if score_match:
                    evaluated_idea["score"] = int(score_match.group(1))
                
                # Extract strengths
                strengths_match = re.search(r"STRENGTHS:\s*(.*?)(?:\nWEAKNESSES:|$)", block, re.DOTALL)
                if strengths_match:
                    strengths_text = strengths_match.group(1).strip()
                    evaluated_idea["strengths"] = [s.strip() for s in strengths_text.split("\n") if s.strip()]
                
                # Extract weaknesses
                weaknesses_match = re.search(r"WEAKNESSES:\s*(.*?)(?:\nSUGGESTIONS:|$)", block, re.DOTALL)
                if weaknesses_match:
                    weaknesses_text = weaknesses_match.group(1).strip()
                    evaluated_idea["weaknesses"] = [w.strip() for w in weaknesses_text.split("\n") if w.strip()]
                
                # Extract suggestions
                suggestions_match = re.search(r"SUGGESTIONS:\s*(.*?)(?:\n---|$)", block, re.DOTALL)
                if suggestions_match:
                    suggestions_text = suggestions_match.group(1).strip()
                    evaluated_idea["suggestions"] = [s.strip() for s in suggestions_text.split("\n") if s.strip()]
                
                # Add evaluation timestamp
                evaluated_idea["evaluated_at"] = datetime.now().isoformat()
                
                evaluated_ideas.append(evaluated_idea)
            
            # Sort by score (highest first)
            evaluated_ideas.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Cache the response if caching is enabled
            if self.use_cache and self.cache_manager:
                self.cache_manager.set(request_params, evaluated_ideas)
            
            logger.info(f"Evaluated {len(evaluated_ideas)} article ideas")
            return evaluated_ideas
            
        except Exception as e:
            logger.error(f"Error evaluating article ideas: {e}")
            return ideas  # Return original ideas if evaluation fails
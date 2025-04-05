"""Trend analyzer for article generation."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from loguru import logger

from src.llm_client import LLMClient
from src.web_search import WebSearchManager


class TrendAnalyzer:
    """Analyzes trends and competitor content for article generation."""
    
    def __init__(self, openai_client: LLMClient, web_search: WebSearchManager):
        """Initialize the trend analyzer.
        
        Args:
            openai_client: LLM client for API interactions
            web_search: Web search manager for competitor research
        """
        self.llm_client = openai_client
        self.web_search = web_search
    
    def analyze_trends(self, research_topic: str) -> Dict[str, Any]:
        """Analyze trends for a research topic.
        
        Args:
            research_topic: Topic to analyze trends for
            
        Returns:
            Dictionary containing trend analysis results
        """
        logger.info(f"Analyzing trends for topic: {research_topic}")
        
        # Search for trending content
        search_results = self.web_search.search(research_topic)
        
        # Extract relevant information
        trends = []
        for result in search_results:
            trends.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("snippet", ""),
                "source": result.get("source", ""),
                "date": result.get("date", "")
            })
        
        # Analyze trends using LLM
        system_prompt = (
            "You are an expert content strategist who analyzes trends in content. "
            "Your analysis should identify patterns, emerging topics, and opportunities."
        )
        
        user_prompt = f"""Analyze the following trending content related to '{research_topic}' and identify key patterns and opportunities:
        
        {json.dumps(trends, indent=2)}
        
        Provide your analysis in the following format:
        KEY_TRENDS: [List of 3-5 key trends identified]
        OPPORTUNITIES: [List of 3-5 content opportunities]
        RECOMMENDATIONS: [List of 3-5 recommendations for content creation]
        """
        
        try:
            response = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse the analysis
            analysis = {}
            current_section = None
            
            for line in response.split("\n"):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("KEY_TRENDS:"):
                    current_section = "key_trends"
                    analysis[current_section] = []
                elif line.startswith("OPPORTUNITIES:"):
                    current_section = "opportunities"
                    analysis[current_section] = []
                elif line.startswith("RECOMMENDATIONS:"):
                    current_section = "recommendations"
                    analysis[current_section] = []
                elif current_section and line.startswith("- "):
                    analysis[current_section].append(line[2:])
            
            # Add raw search results
            analysis["raw_trends"] = trends
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {
                "key_trends": [],
                "opportunities": [],
                "recommendations": [],
                "raw_trends": trends
            }
    
    def research_competitors(self, research_topic: str) -> Dict[str, Any]:
        """Research competitors for a topic.
        
        Args:
            research_topic: Topic to research competitors for
            
        Returns:
            Dictionary containing competitor research results
        """
        logger.info(f"Researching competitors for topic: {research_topic}")
        
        # Search for competitor content
        search_results = self.web_search.search(research_topic)
        
        # Extract relevant information
        competitors = []
        for result in search_results:
            competitors.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("snippet", ""),
                "source": result.get("source", ""),
                "date": result.get("date", "")
            })
        
        # Analyze competitors using LLM
        system_prompt = (
            "You are an expert content strategist who analyzes competitor content. "
            "Your analysis should identify strengths, weaknesses, and opportunities for differentiation."
        )
        
        user_prompt = f"""Analyze the following competitor content related to '{research_topic}' and identify key insights:
        
        {json.dumps(competitors, indent=2)}
        
        Provide your analysis in the following format:
        COMPETITOR_STRENGTHS: [List of 3-5 strengths of competitor content]
        COMPETITOR_WEAKNESSES: [List of 3-5 weaknesses of competitor content]
        DIFFERENTIATION_OPPORTUNITIES: [List of 3-5 opportunities to differentiate our content]
        """
        
        try:
            response = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse the analysis
            analysis = {}
            current_section = None
            
            for line in response.split("\n"):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("COMPETITOR_STRENGTHS:"):
                    current_section = "competitor_strengths"
                    analysis[current_section] = []
                elif line.startswith("COMPETITOR_WEAKNESSES:"):
                    current_section = "competitor_weaknesses"
                    analysis[current_section] = []
                elif line.startswith("DIFFERENTIATION_OPPORTUNITIES:"):
                    current_section = "differentiation_opportunities"
                    analysis[current_section] = []
                elif current_section and line.startswith("- "):
                    analysis[current_section].append(line[2:])
            
            # Add raw search results
            analysis["raw_competitors"] = competitors
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error researching competitors: {e}")
            return {
                "competitor_strengths": [],
                "competitor_weaknesses": [],
                "differentiation_opportunities": [],
                "raw_competitors": competitors
            } 
"""Trend analyzer for article generation."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from loguru import logger

from src.llm_client import LLMClient
# from src.web_search import BraveSearchManager
from src.web_search import TavilySearchManager

class TrendAnalyzer:
    """Analyzes trends and competitor content for article generation."""
    
    def __init__(self, openai_client: LLMClient, web_search: TavilySearchManager):
        """Initialize the trend analyzer.
        
        Args:
            openai_client: LLM client for API interactions
            web_search: Web search manager for competitor research (BraveSearchManager or TavilySearchManager)
        """
        self.llm_client = openai_client
        self.web_search = web_search
    
    def transform_search_term(self, research_topic: str) -> str:
        """Transform a research topic into an effective search term.
        
        Args:
            research_topic: The original research topic
            
        Returns:
            A transformed search term optimized for web search
        """
        logger.info(f"Transforming search term for topic: {research_topic}")
        
        system_prompt = (
            "You are an expert web researcher who creates effective search terms. "
            "Your task is to transform a general topic into a specific, targeted search term "
            "that will yield relevant and high-quality search results."
        )
        
        user_prompt = f"""Transform the following research topic into an effective search term:

        TOPIC: {research_topic}
        
        Guidelines:
        1. Make the search term more specific and targeted
        2. Include relevant keywords that will help find high-quality content
        3. Keep the search term concise (5-10 words maximum)
        4. Focus on the most important aspects of the topic
        5. Avoid overly broad or vague terms
        
        Provide ONLY the transformed search term without any explanation or additional text.
        """
        
        try:
            response = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )
            
            # Clean up the response to get just the search term
            search_term = response.strip()
            
            # Add the current year to focus on recent results
            current_year = datetime.now().year
            search_term = f"{search_term} Focus on results from the year {current_year}"
            
            logger.info(f"Transformed search term with year: {search_term}")
            return search_term
            
        except Exception as e:
            logger.error(f"Error transforming search term: {e}")
            # Return the original topic if transformation fails
            return research_topic
    
    def analyze_trends(self, research_topic: str) -> Dict[str, Any]:
        """Analyze trends for a research topic.
        
        Args:
            research_topic: Topic to analyze trends for
            
        Returns:
            Dictionary containing trend analysis results
        """
        logger.info(f"Analyzing trends for topic: {research_topic}")
        
        # Transform the research topic into an effective search term
        # search_term = self.transform_search_term(research_topic)
        # logger.info(f"Using transformed search term: {search_term}")
        
        # Search for trending content using the transformed search term
        logger.info(f"Searching for trending content for topic: {research_topic}") 
        search_results = self.web_search.search(research_topic)
        
        # Extract relevant information from search results
        trends = []
        extracted_contents = []
        
        logger.info(f"Parsing search results")
        if "results" in search_results:
            # Add basic information to trends list
            for result in search_results["results"]:
                trends.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("content", ""),
                    "source": result.get("source", ""),
                    "date": result.get("date", "")
                })
            
            # Extract full content from all URLs in a single batch
            urls = [result.get("url", "") for result in search_results["results"] if result.get("url", "")]
            logger.info(f"Extracting content from URLs: {urls}")
            if urls:
                extracted_contents_list = self.web_search.extract_content_from_url(urls)
                # logger.info(f"Extracted contents: {extracted_contents_list}")  # Add this line for debugging inf
                
                # Process extracted contents
                for i, result in enumerate(search_results["results"]):
                    condensed_content = self.web_search.condense_content(extracted_contents_list[i].get("content", ""), self.llm_client)
                    url = result.get("url", "")
                    if url and i < len(extracted_contents_list) and extracted_contents_list[i]["success"]:
                        extracted_contents.append({
                            "title": result.get("title", extracted_contents_list[i].get("title", "")),
                            "url": url,
                            "content": condensed_content,
                            "source": result.get("source", ""),
                            "date": result.get("date", "")
                        })
        
        # Analyze trends using LLM with extracted content
        logger.info(f"Analyzing trends using LLM with extracted content")
        system_prompt = (
            "You are an expert content strategist who analyzes trends in content. "
            "Your analysis should identify patterns, emerging topics, and opportunities."
        )
        
        # Use extracted content if available, otherwise fall back to basic trends
        content_for_analysis = extracted_contents if extracted_contents else trends
        
        user_prompt = f"""Analyze the following trending content related to '{research_topic}' and identify key patterns and opportunities:
        
        {json.dumps(content_for_analysis, indent=2)}
        
        Provide your analysis in the following format:
        KEY_TRENDS: [List of 3-5 key trends identified]
        OPPORTUNITIES: [List of 3-5 content opportunities]
        RECOMMENDATIONS: [List of 3-5 recommendations for content creation]
        """
        
        try:
            logger.info(f"Generating LLM response")
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
            
            logger.info(f"Parsing LLM response")
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
            
            # Add raw search results and original topic
            analysis["raw_trends"] = trends
            analysis["original_topic"] = research_topic
            analysis["extracted_contents"] = extracted_contents
            # analysis["search_term"] = search_term
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {
                "key_trends": [],
                "opportunities": [],
                "recommendations": [],
                "raw_trends": trends,
                "original_topic": research_topic,
                # "search_term": search_term
            }
    
    def research_competitors(self, research_topic: str) -> Dict[str, Any]:
        """Research competitors for a topic.
        
        Args:
            research_topic: Topic to research competitors for
            
        Returns:
            Dictionary containing competitor research results
        """
        logger.info(f"Researching competitors for topic: {research_topic}")
        
        # Transform the research topic into an effective search term
        # search_term = self.transform_search_term(research_topic)
        # logger.info(f"Using transformed search term for competitor research: {search_term}")
        
        # Search for competitor content using the transformed search term
        search_results = self.web_search.get_competitor_content(research_topic)
        
        # Extract relevant information from search results
        competitors = []
        extracted_contents = []
        
        if "results" in search_results:
            # Add basic information to competitors list
            for result in search_results["results"]:
                competitors.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("content", ""),
                    "source": result.get("source", ""),
                    "date": result.get("date", "")
                })
            
            # Extract full content from all URLs in a single batch
            urls = [result.get("url", "") for result in search_results["results"] if result.get("url", "")]
            if urls:
                extracted_contents_list = self.web_search.extract_content_from_url(urls)
                
                # Process extracted contents
                for i, result in enumerate(search_results["results"]):
                    url = result.get("url", "")
                    if url and i < len(extracted_contents_list) and extracted_contents_list[i]["success"]:
                        extracted_contents.append({
                            "title": result.get("title", extracted_contents_list[i].get("title", "")),
                            "url": url,
                            "content": extracted_contents_list[i].get("content", ""),
                            "source": result.get("source", ""),
                            "date": result.get("date", "")
                        })
        
        # Analyze competitors using LLM with extracted content
        system_prompt = (
            "You are an expert content strategist who analyzes competitor content. "
            "Your analysis should identify strengths, weaknesses, and opportunities for differentiation."
        )
        
        # Use extracted content if available, otherwise fall back to basic competitors
        content_for_analysis = extracted_contents if extracted_contents else competitors
        
        user_prompt = f"""Analyze the following competitor content related to '{research_topic}' and identify key insights:
        
        {json.dumps(content_for_analysis, indent=2)}
        
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
            
            # Add raw search results and original topic
            analysis["raw_competitors"] = competitors
            analysis["original_topic"] = research_topic
            # analysis["search_term"] = search_term
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error researching competitors: {e}")
            return {
                "competitor_strengths": [],
                "competitor_weaknesses": [],
                "differentiation_opportunities": [],
                "raw_competitors": competitors,
                "original_topic": research_topic,
                # "search_term": search_term
            }
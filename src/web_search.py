#!/usr/bin/env python3
"""
Web Search Module for GenerAI

This module implements internet search capabilities using the Tavily API.
It provides functions to search the web for current information to enhance
the article generation process with real-time data.
"""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from loguru import logger
from tavily import TavilyClient


class WebSearchManager:
    """Manages web searches using the Tavily API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the web search manager.
        
        Args:
            api_key: Tavily API key (if None, will try to get from environment)
        """
        # Get API key from environment if not provided
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        
        if not self.api_key:
            logger.warning("Tavily API key not found. Web search functionality will be limited.")
            self.client = None
        else:
            try:
                self.client = TavilyClient(api_key=self.api_key)
                logger.info("Web search manager initialized with Tavily API")
            except Exception as e:
                logger.error(f"Error initializing Tavily client: {e}")
                self.client = None
    
    def is_available(self) -> bool:
        """Check if web search functionality is available.
        
        Returns:
            True if the Tavily client is initialized, False otherwise
        """
        return self.client is not None
    
    def search(self, query: str, search_depth: str = "basic", max_results: int = 5) -> Dict[str, Any]:
        """Search the web for information related to the query.
        
        Args:
            query: Search query string
            search_depth: Depth of search ("basic" or "comprehensive")
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing search results and metadata
        """
        if not self.is_available():
            logger.warning("Web search unavailable: Tavily client not initialized")
            return {"results": [], "error": "Tavily client not initialized"}
        
        try:
            logger.info(f"Searching web for: {query}")
            
            # Set search parameters based on depth
            search_params = {
                "query": query,
                "search_depth": "comprehensive" if search_depth == "comprehensive" else "basic",
                "max_results": max_results
            }
            
            # Execute search
            response = self.client.search(**search_params)
            
            # Format results
            search_results = {
                "query": query,
                "results": response.get("results", []),
                "search_depth": search_depth,
                "timestamp": datetime.now().isoformat(),
                "result_count": len(response.get("results", []))
            }
            
            logger.info(f"Found {search_results['result_count']} results for query: {query}")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching web: {e}")
            return {"results": [], "error": str(e)}
    
    def search_news(self, topic: str, max_results: int = 5) -> Dict[str, Any]:
        """Search for recent news articles related to the topic.
        
        Args:
            topic: Topic to search for news about
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing news search results and metadata
        """
        # Construct a query specifically for recent news
        query = f"latest news about {topic} in the past month"
        
        # Use the general search method with the news-focused query
        return self.search(query=query, search_depth="basic", max_results=max_results)
    
    def get_topic_insights(self, topic: str) -> Dict[str, Any]:
        """Get comprehensive insights about a topic from the web.
        
        Args:
            topic: Topic to research
            
        Returns:
            Dictionary containing topic insights from web search
        """
        if not self.is_available():
            logger.warning("Web search unavailable: Tavily client not initialized")
            return {"insights": {}, "error": "Tavily client not initialized"}
        
        try:
            logger.info(f"Gathering web insights for topic: {topic}")
            
            # Search for general information about the topic
            general_query = f"comprehensive information about {topic}"
            general_results = self.search(query=general_query, search_depth="comprehensive", max_results=3)
            
            # Search for recent developments
            recent_query = f"recent developments in {topic} in the past 3 months"
            recent_results = self.search(query=recent_query, search_depth="basic", max_results=3)
            
            # Search for trending subtopics
            trending_query = f"trending subtopics within {topic}"
            trending_results = self.search(query=trending_query, search_depth="basic", max_results=3)
            
            # Combine all results into insights
            insights = {
                "topic": topic,
                "general_information": general_results.get("results", []),
                "recent_developments": recent_results.get("results", []),
                "trending_subtopics": trending_results.get("results", []),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Gathered web insights for topic: {topic}")
            return {"insights": insights, "error": None}
            
        except Exception as e:
            logger.error(f"Error gathering web insights: {e}")
            return {"insights": {}, "error": str(e)}
    
    def get_competitor_content(self, topic: str, max_results: int = 5) -> Dict[str, Any]:
        """Search for competitor content related to the topic.
        
        Args:
            topic: Topic to research competitor content for
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing competitor content search results
        """
        # Construct a query specifically for finding competitor content
        query = f"best articles about {topic}"
        
        # Use the general search method with the competitor-focused query
        return self.search(query=query, search_depth="comprehensive", max_results=max_results)
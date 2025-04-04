#!/usr/bin/env python3
"""
Web Search Module for GenerAI

This module implements internet search capabilities using the Brave Search API (default)
and Tavily API (optional). It provides functions to search the web for current information
to enhance the article generation process with real-time data.
"""

import os
import requests
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod

from loguru import logger
from tavily import TavilyClient


class SearchProvider(ABC):
    """Abstract base class for search providers."""
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if search functionality is available."""
        pass
    
    @abstractmethod
    def search(self, query: str, search_depth: str = "basic", max_results: int = 5) -> Dict[str, Any]:
        """Search the web for information related to the query."""
        pass
    
    @abstractmethod
    def search_news(self, topic: str, max_results: int = 5) -> Dict[str, Any]:
        """Search for recent news articles related to the topic."""
        pass
    
    @abstractmethod
    def get_topic_insights(self, topic: str) -> Dict[str, Any]:
        """Get comprehensive insights about a topic from the web."""
        pass
    
    @abstractmethod
    def get_competitor_content(self, topic: str, max_results: int = 5) -> Dict[str, Any]:
        """Search for competitor content related to the topic."""
        pass


class BraveSearchManager(SearchProvider):
    """Manages web searches using the Brave Search API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Brave search manager.
        
        Args:
            api_key: Brave Search API key (if None, will try to get from environment)
        """
        # Get API key from environment if not provided
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY")
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.init_error = None
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum time between requests in seconds
        self.max_retries = 3
        self.retry_delay = 2.0  # Delay between retries in seconds
        
        if not self.api_key:
            logger.warning("Brave API key not found. Web search functionality will be limited.")
            self.client = None
            self.init_error = "Brave API key not found"
        else:
            try:
                # Test the API key with a simple request
                headers = {"Accept": "application/json", "X-Subscription-Token": self.api_key}
                test_response = requests.get(
                    self.base_url,
                    headers=headers,
                    params={"q": "test", "count": 1}
                )
                
                if test_response.status_code == 200:
                    self.client = True  # Just a flag to indicate the API is working
                    logger.info("Web search manager initialized with Brave Search API")
                else:
                    error_msg = f"Error initializing Brave Search client: {test_response.status_code}"
                    logger.error(error_msg)
                    self.client = None
                    self.init_error = error_msg
            except Exception as e:
                error_msg = f"Error initializing Brave Search client: {e}"
                logger.error(error_msg)
                self.client = None
                self.init_error = str(e)
    
    def is_available(self) -> bool:
        """Check if web search functionality is available.
        
        Returns:
            True if the Brave Search client is initialized, False otherwise
        """
        return self.client is not None
    
    def _make_request(self, headers: Dict[str, str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the Brave Search API with rate limiting and retry logic.
        
        Args:
            headers: Request headers
            params: Request parameters
            
        Returns:
            Dictionary containing the API response
        """
        # Ensure minimum time between requests
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.base_url, headers=headers, params=params)
                self.last_request_time = time.time()
                
                if response.status_code == 429:  # Too Many Requests
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Rate limit hit, retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        response.raise_for_status()
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Request failed, retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    raise
    
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
            logger.warning("Web search unavailable: Brave Search client not initialized")
            # Return the original error message if available, otherwise use the default message
            error_msg = self.init_error if self.init_error else "Brave Search client not initialized"
            return {"results": [], "error": error_msg}
        
        try:
            logger.info(f"Searching web for: {query}")
            
            # Set search parameters
            headers = {"Accept": "application/json", "X-Subscription-Token": self.api_key}
            params = {
                "q": query,
                "count": max_results,
                # Use more comprehensive search if requested
                "text_detail": "paragraph" if search_depth == "comprehensive" else "snippet"
            }
            
            # Execute search with rate limiting and retry logic
            data = self._make_request(headers, params)
            
            # Format results to match the expected structure
            results = []
            for web in data.get("web", {}).get("results", []):
                results.append({
                    "title": web.get("title", ""),
                    "url": web.get("url", ""),
                    "content": web.get("description", ""),
                    "score": web.get("relevance_score", 0)
                })
            
            # Format results
            search_results = {
                "query": query,
                "results": results,
                "search_depth": search_depth,
                "timestamp": datetime.now().isoformat(),
                "result_count": len(results)
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
            logger.warning("Web search unavailable: Brave Search client not initialized")
            return {"insights": {}, "error": "Brave Search client not initialized"}
        
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


class TavilySearchManager(SearchProvider):
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


class WebSearchManager:
    """Factory class that creates and manages search providers."""
    
    def __init__(self, api_key: Optional[str] = None, provider: str = "brave"):
        """
        Initialize the web search manager with the specified provider.
        
        Args:
            api_key: API key for the search provider (if None, will try to get from environment)
            provider: Search provider to use ("brave" or "tavily")
        """
        self.provider_name = provider.lower()
        
        # Create the appropriate search provider
        if self.provider_name == "tavily":
            self.provider = TavilySearchManager(api_key=api_key)
        else:  # Default to Brave
            self.provider = BraveSearchManager(api_key=api_key)
        
        if self.provider.is_available():
            logger.info(f"Web search manager initialized with {self.provider_name.capitalize()} API")
        else:
            logger.warning(f"Web search unavailable: {self.provider_name.capitalize()} client not initialized")
    
    def is_available(self) -> bool:
        """Check if web search functionality is available.
        
        Returns:
            True if the search provider is available, False otherwise
        """
        return self.provider.is_available()
    
    def search(self, query: str, search_depth: str = "basic", max_results: int = 5) -> Dict[str, Any]:
        """Search the web for information related to the query.
        
        Args:
            query: Search query string
            search_depth: Depth of search ("basic" or "comprehensive")
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing search results and metadata
        """
        return self.provider.search(query, search_depth, max_results)
    
    def search_news(self, topic: str, max_results: int = 5) -> Dict[str, Any]:
        """Search for recent news articles related to the topic.
        
        Args:
            topic: Topic to search for news about
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing news search results and metadata
        """
        return self.provider.search_news(topic, max_results)
    
    def get_topic_insights(self, topic: str) -> Dict[str, Any]:
        """Get comprehensive insights about a topic from the web.
        
        Args:
            topic: Topic to research
            
        Returns:
            Dictionary containing topic insights from web search
        """
        return self.provider.get_topic_insights(topic)
    
    def get_competitor_content(self, topic: str, max_results: int = 5) -> Dict[str, Any]:
        """Search for competitor content related to the topic.
        
        Args:
            topic: Topic to research competitor content for
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing competitor content search results
        """
        return self.provider.get_competitor_content(topic, max_results)
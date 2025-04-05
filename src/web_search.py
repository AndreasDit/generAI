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
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod

from loguru import logger
from tavily import TavilyClient
from src.llm_client import LLMClient


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
    """Manages web searches for article research."""
    
    def __init__(self, openai_client: LLMClient, data_dir: Path):
        """Initialize the web search manager.
        
        Args:
            openai_client: LLM client for API interactions
            data_dir: Directory to store search data
        """
        self.llm_client = openai_client
        self.data_dir = data_dir
        self.search_dir = data_dir / "searches"
        self.search_dir.mkdir(parents=True, exist_ok=True)
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search the web for a query.
        
        Args:
            query: Search query
            
        Returns:
            List of search results
        """
        logger.info(f"Searching for: {query}")
        
        # Simulate web search results
        system_prompt = (
            "You are an expert web researcher who provides search results. "
            "Your task is to simulate realistic search results for a query."
        )
        
        user_prompt = f"""Generate simulated search results for the query: {query}

        Please provide 5-10 results with:
        1. Title
        2. URL
        3. Snippet
        4. Source
        5. Date
        
        Format each result as a JSON object.
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
            
            # Parse search results
            results = []
            current_result = {}
            
            for line in response.split("\n"):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    result = json.loads(line)
                    results.append(result)
                except json.JSONDecodeError:
                    continue
            
            # Save search results
            search_file = self.search_dir / f"{query.replace(' ', '_')}.json"
            with open(search_file, "w") as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Found {len(results)} results for: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
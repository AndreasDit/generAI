#!/usr/bin/env python3
"""
Cache Manager for GenerAI

This module implements a caching system to avoid redundant API calls to OpenAI.
It stores API responses based on input parameters and retrieves them when similar requests are made.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from loguru import logger


class CacheManager:
    """Manages caching of API responses to avoid redundant calls."""
    
    def __init__(self, cache_dir: str = "cache", ttl_days: int = 7):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory for storing cache files
            ttl_days: Time-to-live for cache entries in days
        """
        self.cache_dir = Path(cache_dir)
        self.ttl_days = ttl_days
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True)
        logger.info(f"Cache manager initialized with directory: {self.cache_dir}")
    
    def _generate_cache_key(self, params: Dict[str, Any]) -> str:
        """
        Generate a unique cache key based on request parameters.
        
        Args:
            params: Dictionary of request parameters
            
        Returns:
            A unique hash string to use as cache key
        """
        # Convert params to a stable string representation
        param_str = json.dumps(params, sort_keys=True)
        
        # Generate a hash of the parameters
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def get(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached response if available and not expired.
        
        Args:
            params: Dictionary of request parameters
            
        Returns:
            Cached response or None if not found or expired
        """
        cache_key = self._generate_cache_key(params)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
            
            # Check if cache has expired
            cached_time = datetime.fromisoformat(cache_data["cached_at"])
            if datetime.now() - cached_time > timedelta(days=self.ttl_days):
                logger.info(f"Cache expired for key: {cache_key}")
                return None
            
            logger.info(f"Cache hit for key: {cache_key}")
            return cache_data["response"]
            
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None
    
    def set(self, params: Dict[str, Any], response: Dict[str, Any]) -> None:
        """
        Store a response in the cache.
        
        Args:
            params: Dictionary of request parameters
            response: Response data to cache
        """
        cache_key = self._generate_cache_key(params)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            cache_data = {
                "cached_at": datetime.now().isoformat(),
                "params": params,
                "response": response
            }
            
            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
                
            logger.info(f"Cached response for key: {cache_key}")
            
        except Exception as e:
            logger.error(f"Error writing to cache: {e}")
    
    def clear_expired(self) -> int:
        """
        Clear expired cache entries.
        
        Returns:
            Number of cache entries cleared
        """
        cleared_count = 0
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r") as f:
                    cache_data = json.load(f)
                
                cached_time = datetime.fromisoformat(cache_data["cached_at"])
                if datetime.now() - cached_time > timedelta(days=self.ttl_days):
                    cache_file.unlink()
                    cleared_count += 1
            except Exception as e:
                logger.error(f"Error clearing cache file {cache_file}: {e}")
        
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} expired cache entries")
        
        return cleared_count
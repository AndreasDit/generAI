#!/usr/bin/env python3
"""
Medium Publisher for GenerAI

This module handles publishing articles to Medium platform.
"""

from typing import Dict, List, Optional, Any

import requests
from loguru import logger


class MediumPublisher:
    """Client for publishing articles to Medium."""
    
    def __init__(self, integration_token: str, author_id: Optional[str] = None):
        """Initialize the Medium publisher.
        
        Args:
            integration_token: Medium integration token
            author_id: Medium author ID (optional)
        """
        self.integration_token = integration_token
        self.author_id = author_id
        self.api_url = "https://api.medium.com/v1"
        self.headers = {
            "Authorization": f"Bearer {integration_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        # Validate the token and get user info if author_id not provided
        if not author_id:
            self._get_user_info()
        
        logger.info("Medium publisher initialized")
    
    def _get_user_info(self) -> None:
        """Get user information from Medium API."""
        try:
            response = requests.get(
                f"{self.api_url}/me",
                headers=self.headers
            )
            response.raise_for_status()
            data = response.json()
            self.author_id = data.get("data", {}).get("id")
            logger.info(f"Retrieved Medium user ID: {self.author_id}")
        except Exception as e:
            logger.error(f"Error getting Medium user info: {e}")
    
    def publish_article(self, title: str, content: str, tags: List[str] = None, 
                       publish_status: str = "draft", canonical_url: str = None) -> Dict[str, Any]:
        """Publish an article to Medium.
        
        Args:
            title: Article title
            content: Article content in Markdown format
            tags: List of tags for the article
            publish_status: Publication status (draft, public, unlisted)
            canonical_url: Original URL if this is a cross-post
        
        Returns:
            Dictionary with publication details or error information
        """
        if not self.author_id:
            logger.error("Author ID not available. Cannot publish.")
            return {"success": False, "error": "Author ID not available"}
        
        # Validate publish_status
        valid_statuses = ["draft", "public", "unlisted"]
        if publish_status not in valid_statuses:
            publish_status = "draft"
            logger.warning(f"Invalid publish status. Using 'draft' instead.")
        
        # Prepare the payload
        payload = {
            "title": title,
            "contentFormat": "markdown",
            "content": content,
            "publishStatus": publish_status,
        }
        
        if tags:
            payload["tags"] = tags[:5]  # Medium allows up to 5 tags
        
        if canonical_url:
            payload["canonicalUrl"] = canonical_url
        
        try:
            logger.info(f"Publishing article '{title}' to Medium with status: {publish_status}")
            response = requests.post(
                f"{self.api_url}/users/{self.author_id}/posts",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            post_id = data.get("data", {}).get("id")
            post_url = data.get("data", {}).get("url")
            
            logger.info(f"Article published successfully. URL: {post_url}")
            return {
                "success": True,
                "post_id": post_id,
                "url": post_url,
                "publish_status": publish_status
            }
            
        except Exception as e:
            logger.error(f"Error publishing to Medium: {e}")
            return {"success": False, "error": str(e)}
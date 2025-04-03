#!/usr/bin/env python3
"""
Article Generator and Medium Publisher

This script generates articles using OpenAI's API and publishes them to Medium.
"""

import os
import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

import openai
import requests
from dotenv import load_dotenv
from loguru import logger

# Configure logger
logger.add(
    "logs/article_generator_{time}.log",
    rotation="10 MB",
    retention="1 week",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    enqueue=True,
)

# Load environment variables
load_dotenv()

class ConfigManager:
    """Manages configuration settings from environment variables."""
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": os.getenv("OPENAI_MODEL", "gpt-4"),
            },
            "medium": {
                "integration_token": os.getenv("MEDIUM_INTEGRATION_TOKEN"),
                "author_id": os.getenv("MEDIUM_AUTHOR_ID"),
            },
            "article": {
                "default_tags": [tag.strip() for tag in os.getenv("DEFAULT_TAGS", "AI,Technology").split(",")],
                "default_status": os.getenv("DEFAULT_STATUS", "draft"),
            }
        }
        return config
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self.config


class OpenAIClient:
    """Client for interacting with OpenAI API."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
        """
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
        logger.info(f"OpenAI client initialized with model: {model}")
    
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
        
        try:
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
            
            logger.info(f"Generated article with title: '{title}'")
            return {"title": title, "content": content}
            
        except Exception as e:
            logger.error(f"Error generating article: {e}")
            return {"title": "", "content": ""}


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


def setup_argparse() -> argparse.Namespace:
    """Set up command line argument parsing.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Generate articles with OpenAI and publish to Medium")
    
    # Article generation options
    parser.add_argument("--topic", type=str, help="Topic for the article")
    parser.add_argument("--tone", type=str, default="informative", 
                        choices=["informative", "casual", "professional", "technical", "conversational"],
                        help="Tone of the article")
    parser.add_argument("--length", type=str, default="medium", 
                        choices=["short", "medium", "long"],
                        help="Length of the article")
    parser.add_argument("--outline", type=str, help="Comma-separated list of sections to include")
    
    # Medium publishing options
    parser.add_argument("--publish", action="store_true", help="Publish to Medium after generation")
    parser.add_argument("--tags", type=str, help="Comma-separated list of tags for Medium")
    parser.add_argument("--status", type=str, default="draft", 
                        choices=["draft", "public", "unlisted"],
                        help="Publication status on Medium")
    parser.add_argument("--canonical-url", type=str, help="Original URL if this is a cross-post")
    
    # Output options
    parser.add_argument("--output", type=str, help="Save article to file")
    
    return parser.parse_args()


def main():
    """Main function to run the article generator and publisher."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Parse command line arguments
    args = setup_argparse()
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Check if OpenAI API key is available
    if not config["openai"]["api_key"]:
        logger.error("OpenAI API key not found. Please set it as OPENAI_API_KEY environment variable.")
        return
    
    # Initialize OpenAI client
    openai_client = OpenAIClient(
        api_key=config["openai"]["api_key"],
        model=config["openai"]["model"]
    )
    
    # Get topic from arguments or prompt user
    topic = args.topic
    if not topic:
        topic = input("Enter the topic for your article: ")
    
    # Parse outline if provided
    outline = None
    if args.outline:
        outline = [section.strip() for section in args.outline.split(",")]
    
    # Generate the article
    article = openai_client.generate_article(
        topic=topic,
        tone=args.tone,
        length=args.length,
        outline=outline
    )
    
    if not article["title"] or not article["content"]:
        logger.error("Failed to generate article.")
        return
    
    # Save to file if requested
    if args.output:
        output_file = args.output
        try:
            with open(output_file, "w") as f:
                f.write(f"# {article['title']}\n\n{article['content']}")
            logger.info(f"Article saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving article to file: {e}")
    else:
        # Print the article to console
        print(f"\n{'=' * 80}\n{article['title']}\n{'=' * 80}\n")
        print(article["content"])
        print(f"\n{'=' * 80}\n")
    
    # Publish to Medium if requested
    if args.publish:
        # Check if Medium integration token is available
        if not config["medium"]["integration_token"]:
            logger.error("Medium integration token not found. Please set it as MEDIUM_INTEGRATION_TOKEN environment variable.")
            return
        
        # Initialize Medium publisher
        medium_publisher = MediumPublisher(
            integration_token=config["medium"]["integration_token"],
            author_id=config["medium"]["author_id"]
        )
        
        # Get tags from arguments or config
        tags = None
        if args.tags:
            tags = [tag.strip() for tag in args.tags.split(",")]
        elif config["article"]["default_tags"]:
            tags = config["article"]["default_tags"]
        
        # Get status from arguments or config
        status = args.status or config["article"]["default_status"]
        
        # Publish the article
        result = medium_publisher.publish_article(
            title=article["title"],
            content=article["content"],
            tags=tags,
            publish_status=status,
            canonical_url=args.canonical_url
        )
        
        if result["success"]:
            print(f"\nArticle published to Medium!")
            print(f"URL: {result['url']}")
            print(f"Status: {result['publish_status']}")
        else:
            print(f"\nFailed to publish to Medium: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
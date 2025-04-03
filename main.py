#!/usr/bin/env python3
"""
GenerAI - AI Article Generator and Medium Publisher

This script generates articles using OpenAI's API and publishes them to Medium.
"""

import os
from loguru import logger

from src.config_manager import ConfigManager
from src.openai_client import OpenAIClient
from src.medium_publisher import MediumPublisher
from src.utils import setup_argparse, setup_logging, parse_outline


def main():
    """Main function to run the article generator and publisher."""
    # Set up logging
    setup_logging()
    
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
    outline = parse_outline(args.outline)
    
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
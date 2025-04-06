#!/usr/bin/env python3
"""
GenerAI - AI Article Generator and Medium Publisher

This is the main entry point for the GenerAI application. It provides a unified interface
for article generation and publishing with both simple and modular pipeline approaches.

The application supports:
1. Simple article generation - Quick generation of articles on a given topic
2. Modular pipeline approach - Advanced multi-step process including:
   - Idea generation through research
   - Idea evaluation and selection
   - Project setup for selected article
   - Outline generation
   - Paragraph-by-paragraph content generation
   - Article assembly
   - Final refinement
   - SEO optimization
   - Medium publishing

Usage:
  Simple mode: python generai.py simple --topic "Your Topic" [options]
  Modular mode: python generai.py modular --run-full-pipeline [options]
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

from loguru import logger

from src.config_manager import ConfigManager
from src.llm_client import create_llm_client
from src.medium_publisher import MediumPublisher
from src.utils import setup_logging, parse_outline

# Import the new modular article pipeline
from src.article_pipeline import ArticlePipeline
from src.article_pipeline.utils import setup_pipeline_logging


def setup_argparse() -> argparse.ArgumentParser:
    """
    Set up command line argument parsing with options for both simple and modular approaches.
    
    Returns:
        ArgumentParser object with configured arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate articles with OpenAI and publish to Medium",
        epilog="GenerAI supports both simple article generation and a modular pipeline approach."
    )
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Simple mode (original article_generator.py and main.py functionality)
    simple_parser = subparsers.add_parser("simple", help="Simple article generation")
    
    # Article generation options for simple mode
    simple_parser.add_argument("--topic", type=str, help="Topic for the article")
    simple_parser.add_argument("--tone", type=str, default="informative", 
                        choices=["informative", "casual", "professional", "technical", "conversational"],
                        help="Tone of the article")
    simple_parser.add_argument("--length", type=str, default="medium", 
                        choices=["short", "medium", "long"],
                        help="Length of the article")
    simple_parser.add_argument("--outline", type=str, help="Comma-separated list of sections to include")
    
    # Medium publishing options for simple mode
    simple_parser.add_argument("--publish", action="store_true", help="Publish to Medium after generation")
    simple_parser.add_argument("--tags", type=str, help="Comma-separated list of tags for Medium")
    simple_parser.add_argument("--status", type=str, default="draft", 
                        choices=["draft", "public", "unlisted"],
                        help="Publication status on Medium")
    simple_parser.add_argument("--canonical-url", type=str, help="Original URL if this is a cross-post")
    
    # Output options for simple mode
    simple_parser.add_argument("--output", type=str, help="Save article to file")
    
    # Modular mode (modular_article_generator.py functionality)
    modular_parser = subparsers.add_parser("modular", help="Modular pipeline approach")
    
    # OpenAI configuration options
    modular_parser.add_argument("--openai-api-key", type=str, help="OpenAI API key (if not set in environment)")
    modular_parser.add_argument("--model", type=str, help="OpenAI model to use")
    modular_parser.add_argument("--temperature", type=float, help="Temperature for OpenAI API calls")
    modular_parser.add_argument("--max-tokens", type=int, help="Maximum tokens for OpenAI API calls")
    modular_parser.add_argument("--cache-dir", type=str, help="Directory for caching API responses")
    
    # Pipeline control options
    modular_parser.add_argument("--run-full-pipeline", action="store_true", 
                        help="Run the complete article generation pipeline")
    modular_parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory for storing article data")
    
    # Web search options
    modular_parser.add_argument("--search-provider", type=str, default="brave", choices=["brave", "tavily"],
                        help="Web search provider to use (default: brave)")
    modular_parser.add_argument("--brave-api-key", type=str,
                        help="API key for Brave Search (if not set in environment)")
    modular_parser.add_argument("--tavily-api-key", type=str,
                        help="API key for Tavily Search (if not set in environment)")
    
    # Caching and feedback options
    modular_parser.add_argument("--use-cache", action="store_true", dest="use_cache",
                        help="Enable API response caching (default based on config)")
    modular_parser.add_argument("--no-cache", action="store_false", dest="use_cache",
                        help="Disable API response caching")
    modular_parser.add_argument("--use-feedback", action="store_true", dest="use_feedback",
                        help="Enable feedback loop mechanism (default based on config)")
    modular_parser.add_argument("--no-feedback", action="store_false", dest="use_feedback",
                        help="Disable feedback loop mechanism")
    
    # Individual step options
    modular_parser.add_argument("--analyze-trends", action="store_true",
                        help="Analyze current trends for a topic")
    modular_parser.add_argument("--research-competitors", action="store_true",
                        help="Research competitor content for a topic")
    modular_parser.add_argument("--generate-ideas", action="store_true",
                        help="Generate article ideas based on trend analysis and competitor research")
    modular_parser.add_argument("--research-topic", type=str,
                        help="Topic to research for trend analysis, competitor research, and idea generation")
    modular_parser.add_argument("--num-ideas", type=int, default=5,
                        help="Number of ideas to generate")
    
    modular_parser.add_argument("--evaluate-ideas", action="store_true",
                        help="Evaluate existing ideas and select the best one")
    modular_parser.add_argument("--max-ideas", type=int, default=10,
                        help="Maximum number of ideas to evaluate")
    
    modular_parser.add_argument("--create-project", action="store_true",
                        help="Create a project for the next article in the queue")
    
    modular_parser.add_argument("--generate-outline", action="store_true",
                        help="Generate an outline for the current project")
    modular_parser.add_argument("--project-id", type=str,
                        help="Project ID for operations that require it")
    
    modular_parser.add_argument("--generate-paragraphs", action="store_true",
                        help="Generate paragraphs for the current project")
    
    modular_parser.add_argument("--assemble-article", action="store_true",
                        help="Assemble the article for the current project")
    
    modular_parser.add_argument("--refine-article", action="store_true",
                        help="Refine the article for the current project")
    
    modular_parser.add_argument("--optimize-seo", action="store_true",
                        help="Optimize the article for SEO for the current project")
    
    # Medium publishing options for modular mode
    modular_parser.add_argument("--publish-to-medium", action="store_true",
                        help="Publish the article to Medium")
    modular_parser.add_argument("--tags", type=str,
                        help="Comma-separated list of tags for Medium")
    modular_parser.add_argument("--status", type=str, default="draft", 
                        choices=["draft", "public", "unlisted"],
                        help="Publication status on Medium")
    
    # Metrics recording
    modular_parser.add_argument("--record-metrics", action="store_true",
                        help="Record performance metrics for the current project")
    
    # Output options for modular mode
    modular_parser.add_argument("--output", type=str,
                        help="Save final article to file")
    
    return parser


def run_simple_mode(args, config):
    """
    Run the simple article generation mode (from original article_generator.py and main.py).
    
    Args:
        args: Command line arguments
        config: Application configuration
    """
    # Check if OpenAI API key is available
    if not config["openai"]["api_key"]:
        logger.error("OpenAI API key not found. Please set it as OPENAI_API_KEY environment variable.")
        return
    
    # Initialize OpenAI client
    openai_client = LLMClient(
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
        publish_to_medium(config, article, args)


def run_modular_mode(args, config):
    """Run the article generation pipeline in modular mode."""
    # Check if OpenAI API key is available
    api_key = args.openai_api_key or config["openai"]["api_key"]
    if not api_key:
        logger.error("OpenAI API key not found. Please set it as OPENAI_API_KEY environment variable or use --openai-api-key.")
        return
    
    # Initialize LLM client
    llm_client = create_llm_client()
    
    # Initialize the pipeline
    pipeline = ArticlePipeline(
        llm_client=llm_client,
        data_dir=Path(args.data_dir or config.get("data_dir", "data"))
    )
    
    # Get default research topic from config
    default_topic = config.get("research", {}).get("default_topic", "artificial intelligence")
    
    # Execute requested pipeline steps
    if args.analyze_trends:
        research_topic = args.research_topic or config.get("research_topic", default_topic)
        trend_analysis = pipeline.analyze_trends(research_topic)
        print(f"Trend analysis for '{research_topic}':")
        print(json.dumps(trend_analysis, indent=2))
    
    if args.research_competitors:
        research_topic = args.research_topic or config.get("research_topic", default_topic)
        competitor_research = pipeline.research_competitors(research_topic)
        print(f"Competitor research for '{research_topic}':")
        print(json.dumps(competitor_research, indent=2))
    
    if args.generate_ideas:
        research_topic = args.research_topic or config.get("research_topic", default_topic)
        logger.info(f"{config.get("research", {}).get("num_ideas", 5)} {config.get("research", {})} {args.num_ideas}")
        num_ideas = args.num_ideas or config.get("research", {}).get("num_ideas", 5)
        ideas = pipeline.generate_ideas(research_topic=research_topic, num_ideas=3)
        print(f"Generated {len(ideas)} ideas for '{research_topic}':")
        for i, idea in enumerate(ideas, 1):
            print(f"{i}. {idea.get('title', 'No title')}")
            print(f"   {idea.get('description', 'No description')}")
    
    if args.evaluate_ideas:
        max_ideas = args.max_ideas or config.get("max_ideas", 10)
        selected_idea = pipeline.evaluate_ideas()
        if selected_idea:
            print(f"Selected idea: {selected_idea}")
        else:
            print("No suitable idea selected.")
    
    if args.create_project:
        project_id = pipeline.create_project(project_id=args.project_id)
        if project_id:
            print(f"Created project with ID: {project_id}")
        else:
            print("Failed to create project.")
    
    if args.generate_outline:
        if not args.project_id:
            print("Error: --project-id is required for generating an outline.")
            return
        outline = pipeline.generate_outline(args.project_id)
        if outline:
            print(f"Generated outline for project {args.project_id}:")
            for i, section in enumerate(outline, 1):
                print(f"{i}. {section}")
        else:
            print(f"Failed to generate outline for project {args.project_id}.")
    
    if args.generate_paragraphs:
        if not args.project_id:
            print("Error: --project-id is required for generating paragraphs.")
            return
        success = pipeline.generate_paragraphs(args.project_id)
        if success:
            print(f"Generated paragraphs for project {args.project_id}.")
        else:
            print(f"Failed to generate paragraphs for project {args.project_id}.")
    
    if args.assemble_article:
        if not args.project_id:
            print("Error: --project-id is required for assembling the article.")
            return
        article = pipeline.assemble_article(args.project_id)
        if article:
            print(f"Assembled article for project {args.project_id}:")
            print(f"Content length: {len(article)} characters")
        else:
            print(f"Failed to assemble article for project {args.project_id}")
    
    if args.refine_article:
        if not args.project_id:
            print("Error: --project-id is required for refining the article.")
            return
        refined_article = pipeline.refine_article(args.project_id)
        if refined_article:
            print(f"Refined article for project {args.project_id}:")
            print(f"Title: {refined_article.get('title', 'No title')}")
            print(f"Content: {refined_article.get('content', 'No content')[:200]}...")
        else:
            print(f"Failed to refine article for project {args.project_id}.")
    
    if args.optimize_seo:
        if not args.project_id:
            print("Error: --project-id is required for optimizing SEO.")
            return
        seo_article = pipeline.optimize_seo(args.project_id)
        if seo_article and isinstance(seo_article, dict):
            print(f"Optimized article for SEO (project {args.project_id}):")
            print(f"Title: {seo_article.get('title', 'No title')}")
            # Extract metadata for display
            metadata = seo_article.get('metadata', '')
            # Try to extract meta description and keywords from metadata if they exist
            print(f"Metadata: {metadata[:100]}..." if metadata else "No metadata")
        else:
            print(f"Failed to optimize article for SEO (project {args.project_id}).")
    
    if args.publish_to_medium:
        if not args.project_id:
            print("Error: --project-id is required for publishing to Medium.")
            return
        tags = args.tags.split(',') if args.tags else None
        status = args.status or "draft"
        result = pipeline.publish_to_medium(args.project_id, tags, status)
        if result.get('success'):
            print(f"Published article to Medium: {result.get('url')}")
        else:
            print(f"Failed to publish article to Medium: {result.get('error')}")
    
    if args.record_metrics:
        if not args.project_id:
            print("Error: --project-id is required for recording metrics.")
            return
        success = pipeline.record_metrics(args.project_id)
        if success:
            print(f"Recorded metrics for project {args.project_id}.")
        else:
            print(f"Failed to record metrics for project {args.project_id}.")
    
    if args.run_full_pipeline:
        research_topic = args.research_topic or config.get("research_topic", default_topic)
        num_ideas = args.num_ideas or config.get("num_ideas", 5)
        max_ideas = args.max_ideas or config.get("max_ideas", 10)
        result = pipeline.run_full_pipeline(research_topic, num_ideas, max_ideas)
        if result:
            print(f"Pipeline completed successfully. Article saved to: {result.get('file_path')}")
        else:
            print("Pipeline failed.")


def publish_to_medium(config, article, args) -> Dict[str, Any]:
    """
    Publish an article to Medium.
    
    Args:
        config: Application configuration
        article: Article to publish
        args: Command line arguments
        
    Returns:
        Dictionary with publishing result information
    """
    # Check if Medium integration token is available
    if not config["medium"]["integration_token"]:
        logger.error("Medium integration token not found. Please set it as MEDIUM_INTEGRATION_TOKEN environment variable.")
        return {"success": False, "error": "Medium integration token not found"}
    
    # Initialize Medium publisher
    medium_publisher = MediumPublisher(
        integration_token=config["medium"]["integration_token"],
        author_id=config["medium"]["author_id"]
    )
    
    # Get tags from arguments or config
    tags = []
    if args.tags:
        tags = [tag.strip() for tag in args.tags.split(",")]
    elif config["article"]["default_tags"]:
        tags = config["article"]["default_tags"]
    
    # Get status from arguments or config
    status = args.status or config["article"]["default_status"]
    
    # Use SEO-optimized title and content if available
    title = article.get("title", "")
    content = article.get("content", "")
    
    # Add SEO meta description as subtitle if available
    meta_description = article.get("meta_description", "")
    if meta_description:
        content = f"*{meta_description}*\n\n{content}"
    
    # Get canonical URL if available (only in simple mode)
    canonical_url = getattr(args, 'canonical_url', "")
    
    # Publish the article
    result = medium_publisher.publish_article(
        title=title,
        content=content,
        tags=tags,
        publish_status=status,
        canonical_url=canonical_url
    )
    
    if result["success"]:
        print(f"\nArticle published to Medium!")
        print(f"URL: {result['url']}")
        print(f"Status: {result['publish_status']}")
    else:
        print(f"\nFailed to publish to Medium: {result.get('error', 'Unknown error')}")
    
    return result


def main():
    """
    Main function that serves as the entry point for the GenerAI application.
    Provides a unified interface for both simple and modular article generation approaches.
    
    Simple mode: Quick generation of articles on a given topic
    Modular mode: Advanced multi-step pipeline for comprehensive article creation
    """
    # Set up logging
    setup_logging()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Get argument parser and parse arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Determine which mode to run based on args
    if args.mode == "simple" or args.mode is None:  # Default to simple mode if not specified
        run_simple_mode(args, config)
    elif args.mode == "modular":
        run_modular_mode(args, config)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        print(f"Unknown mode: {args.mode}. Use 'simple' or 'modular'.")
        print("For help, run: python generai.py --help")


if __name__ == "__main__":
    main()
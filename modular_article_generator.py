#!/usr/bin/env python3
"""
Modular Article Generator

This script implements a modular approach to article generation with a pipeline architecture.
It breaks down the article creation process into sequential steps:
1. Idea generation through research
2. Idea evaluation and selection
3. Project setup for selected article
4. Outline generation
5. Paragraph-by-paragraph content generation
6. Article assembly
7. Final refinement
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from loguru import logger

from src.config_manager import ConfigManager
from src.openai_client import OpenAIClient
from src.article_pipeline import ArticlePipeline
from src.medium_publisher import MediumPublisher
from src.utils import setup_logging


def setup_argparse() -> argparse.Namespace:
    """Set up command line argument parsing.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Generate articles with a modular pipeline approach")
    
    # Pipeline control options
    parser.add_argument("--run-full-pipeline", action="store_true", 
                        help="Run the complete article generation pipeline")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory for storing article data")
    
    # Caching and feedback options
    parser.add_argument("--use-cache", action="store_true", dest="use_cache",
                        help="Enable API response caching (default based on config)")
    parser.add_argument("--no-cache", action="store_false", dest="use_cache",
                        help="Disable API response caching")
    parser.add_argument("--use-feedback", action="store_true", dest="use_feedback",
                        help="Enable feedback loop mechanism (default based on config)")
    parser.add_argument("--no-feedback", action="store_false", dest="use_feedback",
                        help="Disable feedback loop mechanism")
    parser.add_argument("--record-metrics", type=str, metavar="PROJECT_ID",
                        help="Record performance metrics for a published article")
    
    # Individual step options
    parser.add_argument("--analyze-trends", action="store_true",
                        help="Analyze current trends for a topic")
    parser.add_argument("--research-competitors", action="store_true",
                        help="Research competitor content for a topic")
    parser.add_argument("--generate-ideas", action="store_true",
                        help="Generate article ideas based on trend analysis and competitor research")
    parser.add_argument("--research-topic", type=str,
                        help="Topic to research for trend analysis, competitor research, and idea generation")
    parser.add_argument("--num-ideas", type=int, default=5,
                        help="Number of ideas to generate")
    
    parser.add_argument("--evaluate-ideas", action="store_true",
                        help="Evaluate existing ideas and select the best one")
    parser.add_argument("--max-ideas", type=int, default=10,
                        help="Maximum number of ideas to evaluate")
    
    parser.add_argument("--create-project", action="store_true",
                        help="Create a project for the next article in the queue")
    
    parser.add_argument("--generate-outline", type=str, metavar="PROJECT_ID",
                        help="Generate an outline for the specified project")
    
    parser.add_argument("--generate-paragraphs", type=str, metavar="PROJECT_ID",
                        help="Generate paragraphs for the specified project")
    
    parser.add_argument("--assemble-article", type=str, metavar="PROJECT_ID",
                        help="Assemble the article for the specified project")
    
    parser.add_argument("--refine-article", type=str, metavar="PROJECT_ID",
                        help="Refine the article for the specified project")
    
    parser.add_argument("--optimize-seo", type=str, metavar="PROJECT_ID",
                        help="Optimize the article for SEO for the specified project")
    
    # Medium publishing options
    parser.add_argument("--publish", type=str, metavar="PROJECT_ID",
                        help="Publish the article for the specified project to Medium")
    parser.add_argument("--tags", type=str,
                        help="Comma-separated list of tags for Medium")
    parser.add_argument("--status", type=str, default="draft", 
                        choices=["draft", "public", "unlisted"],
                        help="Publication status on Medium")
    
    # Output options
    parser.add_argument("--output", type=str,
                        help="Save final article to file")
    
    return parser.parse_args()


def main():
    """Main function to run the modular article generator."""
    # Set up logging
    setup_logging()
    
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
    
    # Determine whether to use cache and feedback from args or config
    use_cache = args.use_cache if hasattr(args, 'use_cache') and args.use_cache is not None else config["openai"]["use_cache"]
    use_feedback = args.use_feedback if hasattr(args, 'use_feedback') and args.use_feedback is not None else config["feedback"]["enabled"]
    
    # Initialize OpenAI client with caching support
    openai_client = OpenAIClient(
        api_key=config["openai"]["api_key"],
        model=config["openai"]["model"],
        use_cache=use_cache,
        cache_ttl_days=config["cache"]["ttl_days"]
    )
    
    # Initialize article pipeline with feedback loop and web search
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        logger.warning("Tavily API key not found. Web search functionality will be limited.")
    
    pipeline = ArticlePipeline(
        openai_client, 
        data_dir=args.data_dir,
        use_feedback=use_feedback,
        tavily_api_key=tavily_api_key
    )
    
    logger.info(f"Article generator initialized with caching {'enabled' if use_cache else 'disabled'} and feedback {'enabled' if use_feedback else 'disabled'}")
    
    # Run the requested operation
    if args.run_full_pipeline:
        # Run the complete pipeline
        research_topic = args.research_topic
        if not research_topic:
            research_topic = input("Enter a research topic for idea generation: ")
        
        final_article = pipeline.run_pipeline(
            research_topic=research_topic,
            num_ideas=args.num_ideas,
            max_ideas_to_evaluate=args.max_ideas
        )
        
        if not final_article:
            logger.error("Pipeline execution failed.")
            return
        
        # Save to file if requested
        if args.output:
            try:
                with open(args.output, "w") as f:
                    f.write(f"# {final_article['title']}\n\n{final_article['content']}")
                logger.info(f"Article saved to {args.output}")
            except Exception as e:
                logger.error(f"Error saving article to file: {e}")
        else:
            # Print the article title to console
            print(f"\n{'=' * 80}\n{final_article['title']}\n{'=' * 80}\n")
            print("Article generated successfully. View it in the project directory.")
            print(f"\n{'=' * 80}\n")
            
            # Show SEO metadata if available
            if 'meta_description' in final_article or 'keywords' in final_article:
                print("\nSEO Metadata:")
                if 'meta_description' in final_article:
                    print(f"Meta Description: {final_article['meta_description']}")
                if 'keywords' in final_article:
                    print(f"Keywords: {final_article['keywords']}")
                print(f"\n{'=' * 80}\n")
        
        # Publish to Medium if requested
        if args.publish:
            result = publish_to_medium(config, final_article, args)
            
            # Record metrics if publishing was successful and feedback is enabled
            if result and result.get("success") and use_feedback and pipeline.feedback_manager:
                logger.info(f"Recording initial metrics for published article: {args.publish}")
                # Record initial metrics (these would be updated later with actual performance data)
                pipeline.feedback_manager.record_article_metrics(args.publish, {
                    "publish_status": result.get("publish_status", "unknown"),
                    "published_at": datetime.now().isoformat(),
                    "views": 0,
                    "reads": 0,
                    "claps": 0
                })
            
    else:
        # Run individual steps based on arguments
        if args.analyze_trends:
            research_topic = args.research_topic
            if not research_topic:
                research_topic = input("Enter a research topic for trend analysis: ")
            
            trend_analysis = pipeline.analyze_trends(research_topic)
            if trend_analysis:
                print(f"\nTrend Analysis for '{research_topic}':")
                if "trending_subtopics" in trend_analysis:
                    print(f"\nTrending Subtopics:\n{trend_analysis['trending_subtopics']}")
                if "key_questions" in trend_analysis:
                    print(f"\nKey Questions:\n{trend_analysis['key_questions']}")
                if "recent_developments" in trend_analysis:
                    print(f"\nRecent Developments:\n{trend_analysis['recent_developments']}")
                if "timely_considerations" in trend_analysis:
                    print(f"\nTimely Considerations:\n{trend_analysis['timely_considerations']}")
                if "popular_formats" in trend_analysis:
                    print(f"\nPopular Formats:\n{trend_analysis['popular_formats']}")
            else:
                print("Failed to analyze trends.")
        
        if args.research_competitors:
            research_topic = args.research_topic
            if not research_topic:
                research_topic = input("Enter a research topic for competitor research: ")
            
            competitor_research = pipeline.research_competitors(research_topic)
            if competitor_research:
                print(f"\nCompetitor Research for '{research_topic}':")
                if "common_themes" in competitor_research:
                    print(f"\nCommon Themes:\n{competitor_research['common_themes']}")
                if "content_gaps" in competitor_research:
                    print(f"\nContent Gaps:\n{competitor_research['content_gaps']}")
                if "typical_structures" in competitor_research:
                    print(f"\nTypical Structures:\n{competitor_research['typical_structures']}")
                if "strengths_weaknesses" in competitor_research:
                    print(f"\nStrengths and Weaknesses:\n{competitor_research['strengths_weaknesses']}")
                if "differentiation_opportunities" in competitor_research:
                    print(f"\nDifferentiation Opportunities:\n{competitor_research['differentiation_opportunities']}")
            else:
                print("Failed to research competitors.")
        
        if args.generate_ideas:
            research_topic = args.research_topic
            if not research_topic:
                research_topic = input("Enter a research topic for idea generation: ")
            
            ideas = pipeline.generate_ideas(research_topic, args.num_ideas)
            if ideas:
                print(f"\nGenerated {len(ideas)} ideas:")
                for i, idea in enumerate(ideas):
                    print(f"\n{i+1}. {idea['title']}")
                    print(f"   Description: {idea.get('description', 'Not provided')}")
                    print(f"   Audience: {idea.get('audience', 'Not specified')}")
                    print(f"   Engagement: {idea.get('engagement', 'Not specified')}")
            else:
                print("Failed to generate ideas.")
        
        if args.evaluate_ideas:
            selected_idea_id = pipeline.evaluate_ideas(max_ideas=args.max_ideas)
            if selected_idea_id:
                print(f"\nSelected idea with ID: {selected_idea_id}")
                print("The idea has been moved to the article queue.")
        
        if args.create_project:
            project_id = pipeline.create_project()
            if project_id:
                print(f"\nCreated project with ID: {project_id}")
        
        if args.generate_outline:
            outline = pipeline.generate_outline(args.generate_outline)
            if outline:
                print(f"\nGenerated outline with {len(outline)} sections:")
                for i, section in enumerate(outline):
                    print(f"{i+1}. {section}")
        
        if args.generate_paragraphs:
            success = pipeline.generate_paragraphs(args.generate_paragraphs)
            if success:
                print(f"\nGenerated paragraphs for project {args.generate_paragraphs}")
        
        if args.assemble_article:
            assembled_article = pipeline.assemble_article(args.assemble_article)
            if assembled_article:
                print(f"\nAssembled article for project {args.assemble_article}")
                print(f"Title: {assembled_article['title']}")
        
        if args.refine_article:
            final_article = pipeline.refine_article(args.refine_article)
            if final_article:
                print(f"\nRefined article for project {args.refine_article}")
                print(f"Title: {final_article['title']}")
                
                # Save to file if requested
                if args.output:
                    try:
                        with open(args.output, "w") as f:
                            f.write(f"# {final_article['title']}\n\n{final_article['content']}")
                        logger.info(f"Article saved to {args.output}")
                    except Exception as e:
                        logger.error(f"Error saving article to file: {e}")
        
        if args.optimize_seo:
            seo_article = pipeline.optimize_seo(args.optimize_seo)
            if seo_article:
                print(f"\nArticle SEO optimized successfully: {seo_article['title']}")
                print(f"View the SEO optimized article in {args.data_dir}/projects/{args.optimize_seo}/seo_optimized_article.md")
                print(f"\nSEO Metadata:")
                print(f"Meta Description: {seo_article.get('meta_description', 'Not available')}")
                print(f"Keywords: {seo_article.get('keywords', 'Not available')}")
                if 'additional_recommendations' in seo_article and seo_article['additional_recommendations']:
                    print(f"\nAdditional SEO Recommendations:\n{seo_article['additional_recommendations']}")
                
                # Save to file if requested
                if args.output:
                    try:
                        with open(args.output, "w") as f:
                            f.write(f"# {seo_article['title']}\n\n{seo_article['content']}")
                        logger.info(f"SEO optimized article saved to {args.output}")
                    except Exception as e:
                        logger.error(f"Error saving SEO optimized article to file: {e}")
        
        # Record performance metrics for a published article
        if args.record_metrics and use_feedback and pipeline.feedback_manager:
            try:
                # In a real application, these metrics would come from Medium's API or another analytics source
                # For demonstration, we'll use sample metrics or prompt the user
                print(f"\nRecording performance metrics for project: {args.record_metrics}")
                views = int(input("Enter number of views: "))
                reads = int(input("Enter number of reads: "))
                claps = int(input("Enter number of claps/likes: "))
                
                metrics = {
                    "views": views,
                    "reads": reads,
                    "claps": claps,
                    "recorded_at": datetime.now().isoformat()
                }
                
                pipeline.feedback_manager.record_article_metrics(args.record_metrics, metrics)
                print("Metrics recorded successfully!")
                
                # Show performance insights if available
                insights = pipeline.feedback_manager.get_performance_insights()
                if insights and insights.get("general_recommendations"):
                    print("\nContent Strategy Insights:")
                    for rec in insights["general_recommendations"]:
                        print(f"- {rec}")
                    
            except Exception as e:
                logger.error(f"Error recording metrics: {e}")
        
        if args.publish:
            # Load the final article from the project
            project_dir = Path(args.data_dir) / "projects" / args.publish
            try:
                with open(project_dir / "final_article.json", "r") as f:
                    import json
                    final_article = json.load(f)
                publish_to_medium(config, final_article, args)
            except Exception as e:
                logger.error(f"Error loading article for publishing: {e}")


def publish_to_medium(config, article, args) -> Dict[str, Any]:
    """Publish an article to Medium.
    
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
    
    # Use SEO-optimized title and content if available
    title = article.get("title", "")
    content = article.get("content", "")
    
    # Add SEO meta description as subtitle if available
    meta_description = article.get("meta_description", "")
    if meta_description:
        content = f"*{meta_description}*\n\n{content}"
    
    # Publish the article
    result = medium_publisher.publish_article(
        title=title,
        content=content,
        tags=tags,
        publish_status=status
    )
    
    if result["success"]:
        print(f"\nArticle published to Medium!")
        print(f"URL: {result['url']}")
        print(f"Status: {result['publish_status']}")
    else:
        print(f"\nFailed to publish to Medium: {result.get('error', 'Unknown error')}")
    
    return result


if __name__ == "__main__":
    main()
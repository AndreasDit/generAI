"""
Article Pipeline for GenerAI

This module implements a modular approach to article generation with a pipeline architecture.
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
from pathlib import Path
from typing import Dict, List, Optional, Any

from loguru import logger

from src.openai_client import OpenAIClient
from src.web_search import WebSearchManager
from src.feedback_manager import FeedbackManager
from .trend_analyzer import TrendAnalyzer
from .idea_generator import IdeaGenerator
from .project_manager import ProjectManager
from .content_generator import ContentGenerator
from .article_assembler import ArticleAssembler
from .seo_optimizer import SEOOptimizer
from .utils import setup_directory_structure

class ArticlePipeline:
    """Implements a modular pipeline for article generation."""
    
    def __init__(self, openai_client: OpenAIClient, data_dir: str = "data", use_feedback: bool = True, search_api_key: Optional[str] = None, search_provider: str = "brave"):
        """Initialize the article pipeline.
        
        Args:
            openai_client: OpenAI client for API interactions
            data_dir: Base directory for storing article data
            use_feedback: Whether to use feedback loop for content improvement
            search_api_key: API key for web search (if None, will try to get from environment)
            search_provider: Search provider to use ("brave" or "tavily", defaults to "brave")
        """
        self.openai_client = openai_client
        self.data_dir = Path(data_dir)
        self.use_feedback = use_feedback
        
        # Create necessary directory structure
        self.ideas_dir, self.article_queue_dir, self.projects_dir = setup_directory_structure(self.data_dir)
        
        # Initialize web search manager for internet connectivity
        self.web_search = WebSearchManager(api_key=search_api_key, provider=search_provider)
        if self.web_search.is_available():
            logger.info(f"Article pipeline initialized with {search_provider} web search capability")
        else:
            logger.warning("Web search capability not available - using AI-only generation")
        
        # Initialize feedback manager if feedback is enabled
        if self.use_feedback:
            self.feedback_manager = FeedbackManager(data_dir=data_dir)
            logger.info("Article pipeline initialized with feedback loop enabled")
        else:
            self.feedback_manager = None
            logger.info("Article pipeline initialized")
            
        # Initialize component modules
        self.trend_analyzer = TrendAnalyzer(openai_client, self.web_search)
        self.idea_generator = IdeaGenerator(openai_client, self.ideas_dir, self.article_queue_dir)
        self.project_manager = ProjectManager(openai_client, self.projects_dir)
        self.content_generator = ContentGenerator(openai_client, self.projects_dir)
        self.article_assembler = ArticleAssembler(openai_client, self.projects_dir)
        self.seo_optimizer = SEOOptimizer(openai_client, self.projects_dir)
    
    def run_pipeline(self, research_topic: str = None, num_ideas: int = 5, 
                    max_ideas_to_evaluate: int = 10) -> Optional[Dict[str, str]]:
        """Run the complete article generation pipeline.
        
        Args:
            research_topic: Topic to research and generate article about
            num_ideas: Number of ideas to generate
            max_ideas_to_evaluate: Maximum number of ideas to evaluate
            
        Returns:
            Dictionary containing the generated article data or None if failed
        """
        try:
            # Step 1: Analyze trends
            trend_analysis = self.trend_analyzer.analyze_trends(research_topic)
            
            # Step 2: Generate and evaluate ideas
            ideas = self.idea_generator.generate_ideas(research_topic, num_ideas)
            selected_idea = self.idea_generator.evaluate_ideas(max_ideas=max_ideas_to_evaluate)
            
            if not selected_idea:
                logger.error("No suitable idea selected")
                return None
            
            # Step 3: Create project
            project_id = self.project_manager.create_project(selected_idea)
            if not project_id:
                logger.error("Failed to create project")
                return None
            
            # Step 4: Generate outline
            outline = self.content_generator.generate_outline(project_id)
            if not outline:
                logger.error("Failed to generate outline")
                return None
            
            # Step 5: Generate paragraphs
            if not self.content_generator.generate_paragraphs(project_id):
                logger.error("Failed to generate paragraphs")
                return None
            
            # Step 6: Assemble article
            article = self.article_assembler.assemble_article(project_id)
            if not article:
                logger.error("Failed to assemble article")
                return None
            
            # Step 7: Refine article
            refined_article = self.article_assembler.refine_article(project_id)
            if not refined_article:
                logger.error("Failed to refine article")
                return None
            
            # Step 8: Optimize SEO
            final_article = self.seo_optimizer.optimize_seo(project_id)
            if not final_article:
                logger.error("Failed to optimize SEO")
                return None
            
            return final_article
            
        except Exception as e:
            logger.error(f"Error in pipeline execution: {e}")
            return None 
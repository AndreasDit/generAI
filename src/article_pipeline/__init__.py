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
    """Main class for orchestrating the article generation pipeline."""
    
    def __init__(self, openai_client: OpenAIClient, data_dir: Path):
        """Initialize the article pipeline.
        
        Args:
            openai_client: OpenAI client for API interactions
            data_dir: Base directory for data storage
        """
        self.openai_client = openai_client
        self.data_dir = data_dir
        
        # Initialize web search manager
        self.web_search = WebSearchManager()
        
        # Initialize components
        self.trend_analyzer = TrendAnalyzer(openai_client, self.web_search)
        self.idea_generator = IdeaGenerator(
            openai_client=openai_client,
            ideas_dir=data_dir / "ideas",
            article_queue_dir=data_dir / "article_queue",
            trend_analyzer=self.trend_analyzer
        )
        self.project_manager = ProjectManager(
            openai_client=openai_client,
            projects_dir=data_dir / "projects"
        )
        self.content_generator = ContentGenerator(
            openai_client=openai_client,
            projects_dir=data_dir / "projects"
        )
        self.article_assembler = ArticleAssembler(
            openai_client=openai_client,
            projects_dir=data_dir / "projects"
        )
        self.seo_optimizer = SEOOptimizer(
            openai_client=openai_client,
            projects_dir=data_dir / "projects"
        )
        self.feedback_manager = FeedbackManager(str(data_dir))
    
    def analyze_trends(self, research_topic: str) -> Dict[str, Any]:
        """Analyze trends for a research topic.
        
        Args:
            research_topic: Topic to analyze trends for
            
        Returns:
            Dictionary containing trend analysis results
        """
        return self.trend_analyzer.analyze_trends(research_topic)
    
    def research_competitors(self, research_topic: str) -> Dict[str, Any]:
        """Research competitors for a topic.
        
        Args:
            research_topic: Topic to research competitors for
            
        Returns:
            Dictionary containing competitor research results
        """
        return self.trend_analyzer.research_competitors(research_topic)
    
    def generate_ideas(self, research_topic: str, num_ideas: int = 5) -> List[Dict[str, str]]:
        """Generate article ideas based on research topic.
        
        Args:
            research_topic: Topic to generate ideas for
            num_ideas: Number of ideas to generate
            
        Returns:
            List of generated idea dictionaries
        """
        # First analyze trends and research competitors
        trend_analysis = self.analyze_trends(research_topic)
        competitor_research = self.research_competitors(research_topic)
        
        # Generate ideas using the analysis
        return self.idea_generator.generate_ideas(
            research_topic=research_topic,
            num_ideas=num_ideas,
            trend_analysis=trend_analysis,
            competitor_research=competitor_research
        )
    
    def evaluate_ideas(self, max_ideas: int = 10) -> Optional[str]:
        """Evaluate and select the best idea from the queue.
        
        Args:
            max_ideas: Maximum number of ideas to evaluate
            
        Returns:
            ID of the selected idea or None if no suitable idea found
        """
        return self.idea_generator.evaluate_ideas(max_ideas)
    
    def create_project(self) -> Optional[str]:
        """Create a new project from the selected idea.
        
        Returns:
            Project ID if successful, None otherwise
        """
        return self.project_manager.create_project()
    
    def generate_outline(self, project_id: str) -> Optional[List[str]]:
        """Generate an outline for a project.
        
        Args:
            project_id: ID of the project to generate outline for
            
        Returns:
            List of outline sections if successful, None otherwise
        """
        return self.content_generator.generate_outline(project_id)
    
    def generate_paragraphs(self, project_id: str) -> bool:
        """Generate paragraphs for a project's outline.
        
        Args:
            project_id: ID of the project to generate paragraphs for
            
        Returns:
            True if successful, False otherwise
        """
        return self.content_generator.generate_paragraphs(project_id)
    
    def assemble_article(self, project_id: str) -> Optional[Dict[str, str]]:
        """Assemble the article from generated paragraphs.
        
        Args:
            project_id: ID of the project to assemble article for
            
        Returns:
            Dictionary containing assembled article data if successful, None otherwise
        """
        return self.article_assembler.assemble_article(project_id)
    
    def refine_article(self, project_id: str) -> Optional[Dict[str, str]]:
        """Refine the assembled article.
        
        Args:
            project_id: ID of the project to refine article for
            
        Returns:
            Dictionary containing refined article data if successful, None otherwise
        """
        return self.article_assembler.refine_article(project_id)
    
    def optimize_seo(self, project_id: str) -> Optional[Dict[str, str]]:
        """Optimize the article for SEO.
        
        Args:
            project_id: ID of the project to optimize SEO for
            
        Returns:
            Dictionary containing SEO-optimized article data if successful, None otherwise
        """
        return self.seo_optimizer.optimize_seo(project_id)
    
    def publish_to_medium(self, project_id: str, tags: Optional[List[str]] = None, status: str = "draft") -> Dict[str, Any]:
        """Publish the article to Medium.
        
        Args:
            project_id: ID of the project to publish
            tags: List of tags to apply to the article
            status: Publication status ("draft", "public", or "unlisted")
            
        Returns:
            Dictionary containing publishing result data
        """
        return self.article_assembler.publish_to_medium(project_id, tags, status)
    
    def record_metrics(self, project_id: str) -> bool:
        """Record performance metrics for a published article.
        
        Args:
            project_id: ID of the project to record metrics for
            
        Returns:
            True if successful, False otherwise
        """
        if not self.feedback_manager:
            logger.warning("Feedback manager not initialized - cannot record metrics")
            return False
        return self.feedback_manager.record_article_metrics(project_id)
    
    def run_full_pipeline(self, research_topic: str = None, num_ideas: int = 5, 
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
            trend_analysis = self.analyze_trends(research_topic)
            
            # Step 2: Generate and evaluate ideas
            ideas = self.generate_ideas(research_topic, num_ideas)
            selected_idea = self.evaluate_ideas(max_ideas=max_ideas_to_evaluate)
            
            if not selected_idea:
                logger.error("No suitable idea selected")
                return None
            
            # Step 3: Create project
            project_id = self.create_project()
            if not project_id:
                logger.error("Failed to create project")
                return None
            
            # Step 4: Generate outline
            outline = self.generate_outline(project_id)
            if not outline:
                logger.error("Failed to generate outline")
                return None
            
            # Step 5: Generate paragraphs
            if not self.generate_paragraphs(project_id):
                logger.error("Failed to generate paragraphs")
                return None
            
            # Step 6: Assemble article
            article = self.assemble_article(project_id)
            if not article:
                logger.error("Failed to assemble article")
                return None
            
            # Step 7: Refine article
            refined_article = self.refine_article(project_id)
            if not refined_article:
                logger.error("Failed to refine article")
                return None
            
            # Step 8: Optimize SEO
            final_article = self.optimize_seo(project_id)
            if not final_article:
                logger.error("Failed to optimize SEO")
                return None
            
            return final_article
            
        except Exception as e:
            logger.error(f"Error in pipeline execution: {e}")
            return None 
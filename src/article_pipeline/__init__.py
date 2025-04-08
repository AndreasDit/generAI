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
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid
import os

from loguru import logger

from src.config import (
    DATA_DIR, CACHE_DIR, PROJECTS_DIR, IDEAS_DIR, ARTICLE_QUEUE_DIR,
    get_llm_config, get_cache_config, get_web_search_config,
    get_project_config, get_feedback_config
)
import os
from src.llm_client import LLMClient
from src.cache_manager import CacheManager
from src.web_search import BraveSearchManager
from src.feedback_manager import FeedbackManager
from .trend_analyzer import TrendAnalyzer
from .project_manager import ProjectManager
from .content_generator import ContentGenerator
from .article_assembler import ArticleAssembler
from .seo_optimizer import SEOOptimizer
from .utils import setup_directory_structure

class ArticlePipeline:
    """Main class for orchestrating the article generation pipeline."""
    
    def __init__(self, llm_client: LLMClient, data_dir: Path):
        """Initialize the article pipeline.
        
        Args:
            llm_client: LLM client for API interactions
            data_dir: Directory to store pipeline data
        """
        self.llm_client = llm_client
        self.data_dir = data_dir
        
        # Create required directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "ideas").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "projects").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "feedback").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "searches").mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        # Get web search configuration
        web_search_config = get_web_search_config()
        self.web_search = BraveSearchManager(api_key=web_search_config["brave"]["api_key"])
        
        # Initialize trend analyzer with BraveSearchManager
        self.trend_analyzer = TrendAnalyzer(llm_client, self.web_search)
        
        # Initialize other components
        self.project_manager = ProjectManager(llm_client, data_dir / "projects")
        self.content_generator = ContentGenerator(llm_client, data_dir / "projects")
        self.article_assembler = ArticleAssembler(llm_client, data_dir / "projects")
        self.seo_optimizer = SEOOptimizer(llm_client, data_dir / "projects")
        self.feedback_manager = FeedbackManager(data_dir / "projects")
    
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
    
    def generate_ideas(self, research_topic: Optional[str] = None, num_ideas: int = None) -> List[Dict[str, Any]]:
        """Generate article ideas.
        
        Args:
            research_topic: Optional topic to research
            num_ideas: Number of ideas to generate (default from RESEARCH_NUM_IDEAS env var)
            
        Returns:
            List of generated ideas
        """
        logger.info("Generating article ideas")
        
        # Analyze trends
        trends = self.trend_analyzer.analyze_trends(research_topic or "current trends")
        
        # Research competitors
        competitors = self.trend_analyzer.research_competitors(research_topic or "current trends")
        
        # Use the provided num_ideas or get from environment
        # Make sure we respect the value in .env file
        if num_ideas is None:
            num_ideas = int(os.getenv("RESEARCH_NUM_IDEAS", "3"))
        
        # Generate ideas using LLM
        system_prompt = (
            "You are an expert content strategist who generates article ideas. "
            "Your ideas should be unique, valuable, and based on trend analysis."
        )
        
        user_prompt = f"""Generate article ideas based on the following research:

        Trend Analysis:
        {json.dumps(trends, indent=2)}
        
        Competitor Research:
        {json.dumps(competitors, indent=2)}
        
        Generate {num_ideas} unique article ideas that:
        1. Address identified trends
        2. Fill content gaps
        3. Differentiate from competitors
        4. Provide unique value
        
        Format each idea as a JSON object with:
        - title: Article title
        - description: Brief description
        - target_audience: Target audience
        - key_points: List of key points
        - value_proposition: Unique value proposition
        - id: Universally unique identifier for the idea
        """
        
        try:
            response = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Parse ideas
            ideas = []
            
            # Try to parse the entire response as a JSON array first
            try:
                # Check if the response is a JSON array
                if response.strip().startswith('[') and response.strip().endswith(']'):
                    ideas = json.loads(response)
                else:
                    # Try to extract JSON objects from the text
                    import re
                    json_objects = re.findall(r'\{[^{}]*\}', response)
                    for json_str in json_objects:
                        try:
                            idea = json.loads(json_str)
                            ideas.append(idea)
                        except json.JSONDecodeError:
                            continue
                    
                    # If no JSON objects were found, create simple idea objects from the text
                    if not ideas:
                        lines = [line.strip() for line in response.split('\n') if line.strip()]
                        for line in lines:
                            if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')):
                                # This looks like a numbered idea
                                title = line.split('.', 1)[1].strip()
                                ideas.append({'title': title, 'description': ''})
            except Exception as e:
                logger.warning(f"Error parsing ideas as JSON: {e}")
                # Fallback: create simple idea objects
                ideas = [{'title': f"Idea {i+1}", 'description': idea_text.strip()} 
                         for i, idea_text in enumerate(response.split('\n\n')) if idea_text.strip()]
            
            # Save ideas
            ideas_dir = self.data_dir / "ideas"
            for i, idea in enumerate(ideas):
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                idea_id = f"idea_{timestamp}_{i + 1}"
                idea_file = ideas_dir / f"{idea_id}.json"
                with open(idea_file, "w") as f:
                    json.dump(idea, f, indent=2)
            
            logger.info(f"Generated {len(ideas)} article ideas")
            return ideas
            
        except Exception as e:
            logger.error(f"Error generating ideas: {e}")
            return []
    
    def evaluate_ideas(self) -> Optional[Dict[str, Any]]:
        """Evaluate generated ideas and select the best one.
        
        Returns:
            Selected idea or None if evaluation failed
        """
        logger.info("Evaluating article ideas")
        
        # Create directories for organizing ideas
        ideas_dir = self.data_dir / "ideas"
        ideas_chosen_dir = self.data_dir / "ideas_chosen"
        ideas_sorted_out_dir = self.data_dir / "ideas_sorted_out"
        
        # Create directories if they don't exist
        ideas_chosen_dir.mkdir(parents=True, exist_ok=True)
        ideas_sorted_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Load generated ideas
        ideas = []
        idea_files = []
        
        for idea_file in ideas_dir.glob("idea_*.json"):
            try:
                with open(idea_file) as f:
                    idea = json.load(f)
                    ideas.append(idea)
                    idea_files.append(idea_file)
            except Exception as e:
                logger.error(f"Error loading idea file {idea_file}: {e}")
        
        if not ideas:
            logger.error("No ideas found to evaluate")
            return None
        
        # Evaluate ideas using LLM
        system_prompt = (
            "You are an expert content strategist who evaluates article ideas. "
            "Your evaluation should consider potential impact and feasibility."
        )
        
        user_prompt = f"""Evaluate the following article ideas:

        {json.dumps(ideas, indent=2)}
        
        Select the best idea based on:
        1. Market potential
        2. Value proposition
        3. Potential for engagement
        4. Alignment with trends
        
        Format your response as a JSON object with:
        - selected_idea_index: Index of the selected idea (0-based)
        - reasoning: Explanation of the selection
        - improvements: Suggested improvements
        - worst_idea_indices: Array of indices of the 11 worst ideas (0-based)
        
        Return ONLY the JSON object, nothing else. No identifer that this is a JSON object.
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
            logger.info(f"Evaluation response")
            
            # Parse evaluation
            try:
                evaluation = json.loads(response)
                logger.info(f"Selected idea index: {evaluation}")
                selected_index = evaluation.get("selected_idea_index")
                worst_indices = evaluation.get("worst_idea_indices", [])
                logger.info(f"Selected idea index: {selected_index}, {evaluation}")
                
                if selected_index is not None and 0 <= selected_index < len(ideas):
                    selected_idea = ideas[selected_index]
                    selected_idea["evaluation"] = {
                        "reasoning": evaluation.get("reasoning", ""),
                        "improvements": evaluation.get("improvements", "")
                    }
                    
                    # Save selected idea to article_queue
                    selected_file = self.data_dir / "article_queue" / f"selected_idea_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
                    selected_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(selected_file, "w") as f:
                        json.dump(selected_idea, f, indent=2)
                    
                    # Move selected idea to ideas_chosen directory
                    chosen_file = ideas_chosen_dir / f"chosen_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
                    with open(chosen_file, "w") as f:
                        json.dump(selected_idea, f, indent=2)
                    
                    # Move the 3 worst ideas to ideas_sorted_out directory
                    for idx in worst_indices:
                        if 0 <= idx < len(ideas) and idx != selected_index:
                            worst_idea = ideas[idx]
                            worst_file = ideas_sorted_out_dir / f"sorted_out_{datetime.now().strftime('%Y%m%d%H%M%S')}_{idx}.json"
                            with open(worst_file, "w") as f:
                                json.dump(worst_idea, f, indent=2)
                            
                            # Delete the original file from ideas directory
                            if idx < len(idea_files):
                                try:
                                    idea_files[idx].unlink()
                                    logger.info(f"Deleted original file: {idea_files[idx]}")
                                except Exception as e:
                                    logger.error(f"Error deleting original file {idea_files[idx]}: {e}")
                    
                    # Delete the selected idea file from ideas directory
                    if selected_index < len(idea_files):
                        try:
                            idea_files[selected_index].unlink()
                            logger.info(f"Deleted selected idea file: {idea_files[selected_index]}")
                        except Exception as e:
                            logger.error(f"Error deleting selected idea file {idea_files[selected_index]}: {e}")
                    
                    logger.info(f"Selected idea: {selected_idea['title']}")
                    logger.info(f"Moved selected idea to {ideas_chosen_dir}")
                    logger.info(f"Moved {len(worst_indices)} worst ideas to {ideas_sorted_out_dir}")
                    
                    # Generate tweets for the selected idea
                    self._generate_tweets_for_idea(selected_idea)
                    
                    return selected_idea
                
            except json.JSONDecodeError:
                logger.error("Error parsing evaluation response")
            
        except Exception as e:
            logger.error(f"Error evaluating ideas: {e}")
        
        return None
    
    def _generate_tweets_for_idea(self, idea: Dict[str, Any]) -> None:
        """Generate tweets for the selected idea based on the content strategy.
        
        Args:
            idea: The selected idea to generate tweets for
        """
        logger.info("Generating tweets for selected idea")
        
        try:
            # Get the X_POSTS_INPUT_DIR from environment variable
            x_posts_dir = os.environ.get("X_POSTS_INPUT_DIR")
            if not x_posts_dir:
                logger.error("X_POSTS_INPUT_DIR environment variable not set")
                return
            
            # Create directory if it doesn't exist
            x_posts_path = Path(x_posts_dir)
            x_posts_path.mkdir(parents=True, exist_ok=True)
            
            # Read content strategy
            content_strategy_path = self.data_dir / "social_media" / "x_content_strategy.md"
            if not content_strategy_path.exists():
                logger.error(f"Content strategy file not found: {content_strategy_path}")
                return
            
            # Calculate tomorrow's date for scheduling tweets
            tomorrow = datetime.now() + timedelta(days=1)
            
            # Generate tweets based on content strategy
            # Morning post (9-10 AM): Announcement with article link
            morning_time = tomorrow.replace(hour=9, minute=30, second=0, microsecond=0)
            morning_tweet = {
                "platform": "twitter",
                "id": str(uuid.uuid4())[:8],
                "content": f"New article: {idea['title']}. Learn how to {idea['description'].split('.')[0].lower()}. #AI #PassiveIncome",
                "hashtags": "#MakeMoneyOnline #PassiveIncome #SideHustle",
                "datetime_for_post": morning_time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Afternoon post (1-2 PM): Share a key insight or quote from the article
            afternoon_time = tomorrow.replace(hour=13, minute=30, second=0, microsecond=0)
            key_point = idea['key_points'][0] if idea.get('key_points') and len(idea['key_points']) > 0 else idea['title']
            afternoon_tweet = {
                "platform": "twitter",
                "id": str(uuid.uuid4())[:8],
                "content": f"Key insight: {key_point} ✨",
                "hashtags": "#MakeMoneyOnline #PassiveIncome #SideHustle",
                "datetime_for_post": afternoon_time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Evening post (7-8 PM): Ask a thought-provoking question related to the article topic
            evening_time = tomorrow.replace(hour=19, minute=30, second=0, microsecond=0)
            evening_tweet = {
                "platform": "twitter",
                "id": str(uuid.uuid4())[:8],
                "content": f"Question: How would you use AI to create passive income as a solo entrepreneur? Share your thoughts! 🤔",
                "hashtags": "#MakeMoneyOnline #PassiveIncome #SideHustle",
                "datetime_for_post": evening_time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Use LLM to refine the tweets
            system_prompt = (
                "You are a social media expert who creates engaging tweets. "
                "Your tweets should be concise, engaging, and include relevant emojis."
            )
            
            user_prompt = f"""Refine the following tweets for an article about {idea['title']}:
            
            Morning tweet (announcement):
            {morning_tweet['content']}
            
            Afternoon tweet (key insight):
            {afternoon_tweet['content']}
            
            Evening tweet (question):
            {evening_tweet['content']}
            
            Make each tweet engaging, concise (under 280 characters), and include 1-2 relevant emojis.
            Format your response as a JSON object with three keys: 'morning', 'afternoon', and 'evening',
            each containing the refined tweet text.
            
            Return ONLY the JSON object, nothing else.
            """
            
            try:
                response = self.llm_client.chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                
                # Parse refined tweets
                refined_tweets = json.loads(response)
                
                # Update tweet content with refined versions
                if 'morning' in refined_tweets:
                    morning_tweet['content'] = refined_tweets['morning']
                if 'afternoon' in refined_tweets:
                    afternoon_tweet['content'] = refined_tweets['afternoon']
                if 'evening' in refined_tweets:
                    evening_tweet['content'] = refined_tweets['evening']
                    
            except Exception as e:
                logger.error(f"Error refining tweets with LLM: {e}")
                # Continue with original tweets if refinement fails
            
            # Save tweets to X_POSTS_INPUT_DIR
            morning_file = x_posts_path / f"{morning_tweet['id']}.json"
            afternoon_file = x_posts_path / f"{afternoon_tweet['id']}.json"
            evening_file = x_posts_path / f"{evening_tweet['id']}.json"
            
            with open(morning_file, "w") as f:
                json.dump(morning_tweet, f, indent=2)
            
            with open(afternoon_file, "w") as f:
                json.dump(afternoon_tweet, f, indent=2)
            
            with open(evening_file, "w") as f:
                json.dump(evening_tweet, f, indent=2)
            
            logger.info(f"Generated and saved 3 tweets for tomorrow ({tomorrow.strftime('%Y-%m-%d')})")
            
        except Exception as e:
            logger.error(f"Error generating tweets: {e}")
    
    def create_project(self, project_id: str = None) -> Optional[str]:
        """Create a project from the selected idea.
        
        Returns:
            Project ID or None if creation failed
        """
        logger.info("Creating project from selected idea")
        
        # Load selected idea
        selected_file = self.data_dir / "article_queue" / f"{project_id}.json"
        logger.info(f"Selected file: {selected_file}")
        if not selected_file.exists():
            logger.error("No selected idea found")
            return None
        
        try:
            with open(selected_file) as f:
                idea = json.load(f)
            
            # Create project
            project_id = self.project_manager.create_project(idea)
            if project_id:
                logger.info(f"Created project: {project_id}")
                
                # Move the selected idea file to the project folder
                project_dir = self.data_dir / "projects" / project_id
                logger.info(f"Project directory: {project_dir}")
                project_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy the idea to the project folder
                project_idea_file = project_dir / "idea.json"
                with open(project_idea_file, "w") as f:
                    json.dump(idea, f, indent=2)
                
                # Remove the file from article_queue
                try:
                    selected_file.unlink()
                    logger.info(f"Removed selected idea file from article_queue: {selected_file}")
                except Exception as e:
                    logger.error(f"Error removing selected idea file from article_queue: {e}")
                
                return project_id
            
        except Exception as e:
            logger.error(f"Error creating project: {e}")
        
        return None
    
    def generate_outline(self, project_id: str) -> bool:
        """Generate an outline for a project.
        
        Args:
            project_id: ID of the project
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Generating outline for project: {project_id}")
        
        try:
            outline = self.content_generator.generate_outline(project_id)
            return outline
            
        except Exception as e:
            logger.error(f"Error generating outline: {e}")
            return False
    
    def generate_paragraphs(self, project_id: str) -> bool:
        """Generate paragraphs for a project.
        
        Args:
            project_id: ID of the project
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Generating paragraphs for project: {project_id}")
        
        try:
            paragraphs = self.content_generator.generate_paragraphs(project_id)
            return bool(paragraphs)
            
        except Exception as e:
            logger.error(f"Error generating paragraphs: {e}")
            return False
    
    def assemble_article(self, project_id: str) -> Dict[str, Any]:
        """Assemble an article for a project.
        
        Args:
            project_id: ID of the project
            
        Returns:
            Dictionary containing the assembled article or empty dict if failed
        """
        logger.info(f"Assembling article for project: {project_id}")
        
        try:
            article = self.article_assembler.assemble_article(project_id)
            return article
            
        except Exception as e:
            logger.error(f"Error assembling article: {e}")
            return {}
    
    def refine_article(self, project_id: str) -> Dict[str, Any]:
        """Refine an article for a project.
        
        Args:
            project_id: ID of the project
            
        Returns:
            Dictionary containing the refined article or empty dict if failed
        """
        logger.info(f"Refining article for project: {project_id}")
        
        try:
            refined = self.article_assembler.refine_article(project_id)
            return refined
            
        except Exception as e:
            logger.error(f"Error refining article: {e}")
            return {}
    
    def optimize_seo(self, project_id: str) -> Dict[str, Any]:
        """Optimize an article for SEO.
        
        Args:
            project_id: ID of the project
            
        Returns:
            Dictionary containing the optimized article data if successful, empty dict otherwise
        """
        logger.info(f"Optimizing SEO for project: {project_id}")
        
        try:
            optimized = self.seo_optimizer.optimize_article(project_id)
            return optimized
            
        except Exception as e:
            logger.error(f"Error optimizing SEO: {e}")
            return {}
    
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
    
    def analyze_feedback(self, project_id: str, feedback: str) -> bool:
        """Analyze feedback for an article.
        
        Args:
            project_id: ID of the project
            feedback: Feedback text to analyze
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Analyzing feedback for project: {project_id}")
        
        try:
            analysis = self.feedback_manager.analyze_feedback(project_id, feedback)
            return bool(analysis)
            
        except Exception as e:
            logger.error(f"Error analyzing feedback: {e}")
            return False
    
    def process_next_article(self, project_id: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Process the next article from the queue through all pipeline steps.
        
        This method either takes the next article idea from the queue or continues processing
        an existing project through the pipeline steps: creating a project, generating an outline, 
        generating paragraphs, assembling the article, refining it, and optimizing SEO.
        
        Args:
            project_id: Optional ID of an existing project to continue processing.
                        If provided, will continue from the last successful step.
                        If None, will process the next article from the queue.
        
        Returns:
            Dictionary containing the generated article data or None if failed
        """
        try:
            # If project_id is provided, continue from the last successful step
            if project_id:
                logger.info(f"Continuing processing for existing project: {project_id}")
                
                # Check if project exists
                project_dir = self.projects_dir / project_id
                if not project_dir.exists():
                    logger.error(f"Project not found: {project_id}")
                    return None
                
                # Load project metadata to determine the last successful step
                metadata_file = project_dir / "metadata.json"
                if not metadata_file.exists():
                    logger.error(f"Project metadata not found: {project_id}")
                    return None
                
                with open(metadata_file) as f:
                    metadata = json.load(f)
                
                # Get current status
                current_status = metadata.get("status", "created")
                logger.info(f"Current project status: {current_status}")
                
                # Continue from the last successful step
                if current_status == "created":
                    # Generate outline
                    outline = self.generate_outline(project_id)
                    if not outline:
                        logger.error(f"Failed to generate outline for project {project_id}")
                        return None
                    current_status = "outline_generated"
                
                if current_status == "outline_generated":
                    # Generate paragraphs
                    if not self.generate_paragraphs(project_id):
                        logger.error(f"Failed to generate paragraphs for project {project_id}")
                        return None
                    current_status = "paragraphs_generated"
                
                if current_status == "paragraphs_generated":
                    # Assemble article
                    article = self.assemble_article(project_id)
                    if not article:
                        logger.error(f"Failed to assemble article for project {project_id}")
                        return None
                    current_status = "article_assembled"
                
                if current_status == "article_assembled":
                    # Refine article
                    refined_article = self.refine_article(project_id)
                    if not refined_article:
                        logger.error(f"Failed to refine article for project {project_id}")
                        return None
                    current_status = "article_refined"
                
                # if current_status == "article_refined":
                #     # Optimize SEO
                #     final_article = self.optimize_seo(project_id)
                #     if not final_article:
                #         logger.error(f"Failed to optimize SEO for project {project_id}")
                #         return None
                #     current_status = "article_optimized"
                
                # logger.info(f"Successfully continued processing for project {project_id}")
                
                # Return the final article if we've reached the end of the pipeline
                if current_status == "article_refined":
                    return final_article
                else:
                    # Return the current state of the project
                    return {"project_id": project_id, "status": current_status}
            
            # If no project_id is provided, process the next article from the queue
            else:
                # Get the next article from the queue
                article_queue_dir = self.data_dir / "article_queue"
                if not article_queue_dir.exists() or not any(article_queue_dir.iterdir()):
                    logger.error("No articles in the queue")
                    return None
                    
                # Get the oldest file in the queue (first in, first out)
                queue_files = sorted(article_queue_dir.glob("*.json"))
                if not queue_files:
                    logger.error("No article files found in the queue")
                    return None
                    
                selected_file = queue_files[0]
                logger.info(f"Processing article from queue: {selected_file.name}")
                
                # Load the idea
                with open(selected_file) as f:
                    idea = json.load(f)
                
                # Create project
                project_id = self.project_manager.create_project(idea)
                if not project_id:
                    logger.error("Failed to create project")
                    return None
                    
                # Remove the file from the queue
                try:
                    selected_file.unlink()
                    logger.info(f"Removed article file from queue: {selected_file}")
                except Exception as e:
                    logger.error(f"Error removing article file from queue: {e}")
                
                # Generate outline
                outline = self.generate_outline(project_id)
                if not outline:
                    logger.error(f"Failed to generate outline for project {project_id}")
                    return None
                
                # Generate paragraphs
                if not self.generate_paragraphs(project_id):
                    logger.error(f"Failed to generate paragraphs for project {project_id}")
                    return None
                
                # Assemble article
                article = self.assemble_article(project_id)
                if not article:
                    logger.error(f"Failed to assemble article for project {project_id}")
                    return None
                
                # Refine article
                refined_article = self.refine_article(project_id)
                if not refined_article:
                    logger.error(f"Failed to refine article for project {project_id}")
                    return None
                
                final_article = refined_article
                # # Optimize SEO
                # final_article = self.optimize_seo(project_id)
                # if not final_article:
                #     logger.error(f"Failed to optimize SEO for project {project_id}")
                #     return None
                
                logger.info(f"Successfully processed article from queue for project {project_id}")
                return final_article
            
        except Exception as e:
            logger.error(f"Error processing article: {e}")
            return None
    
    def run_full_pipeline(self, research_topic: str = None, num_ideas: int = None, 
                    max_ideas_to_evaluate: int = None) -> Optional[Dict[str, str]]:
        """Run the complete article generation pipeline.
        
        Args:
            research_topic: Topic to research and generate article about
            num_ideas: Number of ideas to generate (default from RESEARCH_NUM_IDEAS env var)
            max_ideas_to_evaluate: Maximum number of ideas to evaluate (default from RESEARCH_MAX_IDEAS env var)
            
        Returns:
            Dictionary containing the generated article data or None if failed
        """
        try:
            # Get max_ideas_to_evaluate from environment if not provided
            if max_ideas_to_evaluate is None:
                max_ideas_to_evaluate = int(os.getenv("RESEARCH_MAX_IDEAS", "10"))
                
            # Step 1: Analyze trends
            trend_analysis = self.analyze_trends(research_topic)
            
            # Step 2: Generate and evaluate ideas
            ideas = self.generate_ideas(research_topic, num_ideas)
            selected_idea = self.evaluate_ideas()
            
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
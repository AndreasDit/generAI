#!/usr/bin/env python3
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
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import openai
from loguru import logger

from src.openai_client import OpenAIClient
from src.web_search import WebSearchManager
from src.feedback_manager import FeedbackManager
from .article_pipeline import ArticlePipeline

__all__ = ['ArticlePipeline']

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
        self._setup_directory_structure()
        
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
    
    def _setup_directory_structure(self) -> None:
        """Set up the directory structure for the article pipeline."""
        # Create main data directory if it doesn't exist
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each stage of the pipeline
        self.ideas_dir = self.data_dir / "ideas"
        self.ideas_dir.mkdir(exist_ok=True)
        
        self.article_queue_dir = self.data_dir / "article_queue"
        self.article_queue_dir.mkdir(exist_ok=True)
        
        self.projects_dir = self.data_dir / "projects"
        self.projects_dir.mkdir(exist_ok=True)
        
        logger.info(f"Directory structure set up in {self.data_dir}")
    
    def _parse_trend_analysis(self, content: str) -> Dict[str, Any]:
        """Parse the trend analysis response from OpenAI.
        
        Args:
            content: Raw text response from OpenAI
            
        Returns:
            Dictionary containing parsed trend analysis data
        """
        # Parse the trend analysis into a structured format
        trends = []
        summary = "Summary of trends based on analysis"
        
        # Extract trending subtopics
        if "TRENDING_SUBTOPICS:" in content:
            subtopics_text = content.split("TRENDING_SUBTOPICS:")[1].split("KEY_QUESTIONS:")[0].strip()
            subtopics = [s.strip() for s in subtopics_text.split(",")]
            
            # Create trend objects
            for subtopic in subtopics:
                trends.append({
                    "name": subtopic,
                    "description": f"Description of {subtopic.lower()}"
                })
        
        return {
            "trends": trends,
            "summary": summary
        }
    
    def analyze_trends(self, research_topic: str) -> Dict[str, Any]:
        """Analyze current trends related to the research topic.
        
        Args:
            research_topic: General topic to research for trends
            
        Returns:
            Dictionary containing trend analysis data
        """
        logger.info(f"Analyzing trends for topic: {research_topic}")
        
        # Get web search results if available
        web_insights = {}
        if self.web_search and self.web_search.is_available():
            try:
                logger.info(f"Gathering web insights for trend analysis on: {research_topic}")
                web_results = self.web_search.get_topic_insights(research_topic)
                
                if not web_results.get("error"):
                    web_insights = web_results.get("insights", {})
                    logger.info(f"Successfully gathered web insights for trend analysis")
                else:
                    logger.warning(f"Error in web search: {web_results.get('error')}")
            except Exception as e:
                logger.error(f"Error gathering web insights: {e}")
        
        # Construct the prompt for trend analysis, including web insights if available
        system_prompt = (
            "You are an expert content strategist who analyzes current trends. "
            "Your analysis should identify popular topics, emerging interests, and potential content opportunities."
        )
        
        # Include web insights in the prompt if available
        web_insights_text = ""
        if web_insights:
            web_insights_text = "\n\nHere are some recent web search results to inform your analysis:\n"
            
            # Add general information
            if "general_information" in web_insights and web_insights["general_information"]:
                web_insights_text += "\nGeneral Information:\n"
                for i, result in enumerate(web_insights["general_information"][:3], 1):
                    web_insights_text += f"{i}. {result.get('title', 'No title')}: {result.get('content', 'No content')}\n"
            
            # Add recent developments
            if "recent_developments" in web_insights and web_insights["recent_developments"]:
                web_insights_text += "\nRecent Developments:\n"
                for i, result in enumerate(web_insights["recent_developments"][:3], 1):
                    web_insights_text += f"{i}. {result.get('title', 'No title')}: {result.get('content', 'No content')}\n"
            
            # Add trending subtopics
            if "trending_subtopics" in web_insights and web_insights["trending_subtopics"]:
                web_insights_text += "\nTrending Subtopics:\n"
                for i, result in enumerate(web_insights["trending_subtopics"][:3], 1):
                    web_insights_text += f"{i}. {result.get('title', 'No title')}: {result.get('content', 'No content')}\n"
        
        user_prompt = f"""Analyze current trends related to '{research_topic}'.{web_insights_text}
        
        Provide the following information:
        1. Top 3-5 trending subtopics within this area
        2. Key questions people are asking about this topic
        3. Recent developments or news in this field
        4. Seasonal or timely considerations
        5. Content formats that are performing well for this topic
        
        Format your analysis as follows:
        TRENDING_SUBTOPICS: [List of trending subtopics]
        KEY_QUESTIONS: [List of questions people are asking]
        RECENT_DEVELOPMENTS: [Summary of recent developments]
        TIMELY_CONSIDERATIONS: [Any seasonal or timely factors]
        POPULAR_FORMATS: [Content formats performing well]
        """
        
        try:
            response = self.openai_client.client.chat.completions.create(
                model=self.openai_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1500,
            )
            
            content = response.choices[0].message.content
            
            # Parse the trend analysis
            trend_analysis = {}
            current_section = None
            current_content = []
            
            lines = content.strip().split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith("TRENDING_SUBTOPICS:"):
                    current_section = "trending_subtopics"
                    current_content = [line[20:].strip()]
                elif line.startswith("KEY_QUESTIONS:"):
                    if current_section:
                        trend_analysis[current_section] = "\n".join(current_content)
                    current_section = "key_questions"
                    current_content = [line[14:].strip()]
                elif line.startswith("RECENT_DEVELOPMENTS:"):
                    if current_section:
                        trend_analysis[current_section] = "\n".join(current_content)
                    current_section = "recent_developments"
                    current_content = [line[21:].strip()]
                elif line.startswith("TIMELY_CONSIDERATIONS:"):
                    if current_section:
                        trend_analysis[current_section] = "\n".join(current_content)
                    current_section = "timely_considerations"
                    current_content = [line[22:].strip()]
                elif line.startswith("POPULAR_FORMATS:"):
                    if current_section:
                        trend_analysis[current_section] = "\n".join(current_content)
                    current_section = "popular_formats"
                    current_content = [line[17:].strip()]
                elif current_section:
                    current_content.append(line)
            
            # Add the last section
            if current_section:
                trend_analysis[current_section] = "\n".join(current_content)
            
            # Add web search metadata if available
            if web_insights:
                trend_analysis["web_search_used"] = True
                trend_analysis["web_search_timestamp"] = web_insights.get("timestamp", datetime.now().isoformat())
            else:
                trend_analysis["web_search_used"] = False
            
            # Save trend analysis to file
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            analysis_id = f"trend_analysis_{timestamp}"
            analysis_file = self.ideas_dir / f"{analysis_id}.json"
            
            trend_analysis["topic"] = research_topic
            trend_analysis["created_at"] = datetime.now().isoformat()
            
            with open(analysis_file, "w") as f:
                json.dump(trend_analysis, f, indent=2)
            
            logger.info(f"Completed trend analysis for topic: {research_topic}")
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {}
    
    def get_ideas(self) -> List[Dict[str, Any]]:
        """Get all available article ideas from the ideas directory.
        
        Returns:
            List of article ideas with metadata
        """
        logger.info("Getting available article ideas")
        
        ideas = []
        try:
            # Get all JSON files in the ideas directory that contain article ideas
            idea_files = list(self.ideas_dir.glob("idea_*.json"))
            
            for idea_file in idea_files:
                try:
                    with open(idea_file, "r") as f:
                        idea_data = json.load(f)
                        ideas.append(idea_data)
                except Exception as e:
                    logger.error(f"Error reading idea file {idea_file}: {e}")
            
            logger.info(f"Found {len(ideas)} article ideas")
            return ideas
            
        except Exception as e:
            logger.error(f"Error getting article ideas: {e}")
            return []
    
    def evaluate_ideas(self, max_ideas: int = 10) -> List[Dict[str, Any]]:
        """Evaluate article ideas and select the best one.
        
        Args:
            max_ideas: Maximum number of ideas to evaluate
            
        Returns:
            List of evaluated ideas with scores
        """
        logger.info("Evaluating article ideas")
        
        # Get available ideas
        ideas = self.get_ideas()
        
        if not ideas:
            logger.warning("No article ideas found to evaluate")
            return []
        
        # Limit the number of ideas to evaluate
        if len(ideas) > max_ideas:
            logger.info(f"Limiting evaluation to {max_ideas} ideas")
            ideas = ideas[:max_ideas]
        
        try:
            # Evaluate ideas using OpenAI
            evaluations = self.openai_client.evaluate_article_ideas(ideas)
            
            # Save evaluations
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            evaluation_file = self.ideas_dir / f"evaluation_{timestamp}.json"
            
            with open(evaluation_file, "w") as f:
                json.dump(evaluations, f, indent=2)
            
            # Sort evaluations by score (highest first)
            evaluations.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # If we have a top idea, add it to the article queue
            if evaluations and len(evaluations) > 0:
                top_idea = evaluations[0]
                idea_id = top_idea.get("id")
                
                # Find the corresponding idea data
                idea_data = next((idea for idea in ideas if idea.get("id") == idea_id), None)
                
                if idea_data:
                    # Add to article queue
                    queue_file = self.article_queue_dir / f"queued_{idea_id}.json"
                    
                    with open(queue_file, "w") as f:
                        json.dump({
                            "idea": idea_data,
                            "evaluation": top_idea,
                            "queued_at": datetime.now().isoformat()
                        }, f, indent=2)
                    
                    logger.info(f"Added top idea '{idea_data.get('title')}' to article queue")
            
            logger.info(f"Completed evaluation of {len(evaluations)} ideas")
            return evaluations
            
        except Exception as e:
            logger.error(f"Error evaluating article ideas: {e}")
            return []
    
    def generate_ideas(self, research_topic: str, num_ideas: int = 5) -> List[Dict[str, Any]]:
        """Generate article ideas based on trend analysis and competitor research.
        
        Args:
            research_topic: General topic to research for ideas
            num_ideas: Number of ideas to generate
            
        Returns:
            List of generated ideas with metadata
        """
        logger.info(f"Generating {num_ideas} ideas for topic: {research_topic}")
        
        # First perform trend analysis and competitor research
        trend_analysis = self.analyze_trends(research_topic)
        competitor_research = self.research_competitors(research_topic)
        
        try:
            # Generate ideas using OpenAI
            ideas = self.openai_client.generate_article_ideas(
                research_topic=research_topic,
                trend_analysis=trend_analysis,
                competitor_research=competitor_research,
                num_ideas=num_ideas
            )
            
            # Save ideas to files
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            
            for i, idea in enumerate(ideas):
                # Generate a unique ID for the idea
                idea_id = f"idea_{timestamp}_{i+1}"
                idea["id"] = idea_id
                idea["research_topic"] = research_topic
                idea["created_at"] = datetime.now().isoformat()
                
                # Save to file
                idea_file = self.ideas_dir / f"idea_{idea_id}.json"
                
                with open(idea_file, "w") as f:
                    json.dump(idea, f, indent=2)
            
            logger.info(f"Generated and saved {len(ideas)} ideas for topic: {research_topic}")
            return ideas
            
        except Exception as e:
            logger.error(f"Error generating article ideas: {e}")
            return []
            
    def get_idea_by_id(self, idea_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific article idea by its ID.
        
        Args:
            idea_id: ID of the idea to retrieve
            
        Returns:
            Article idea data or None if not found
        """
        logger.info(f"Getting article idea with ID: {idea_id}")
        
        try:
            # Look for the idea file in the ideas directory
            idea_file = self.ideas_dir / f"idea_{idea_id}.json"
            
            if not idea_file.exists():
                # Also check for files that might contain the ID in their name
                potential_files = list(self.ideas_dir.glob(f"*{idea_id}*.json"))
                if not potential_files:
                    logger.warning(f"No idea found with ID: {idea_id}")
                    return None
                idea_file = potential_files[0]
            
            # Read the idea data
            with open(idea_file, "r") as f:
                idea_data = json.load(f)
            
            logger.info(f"Found idea: {idea_data.get('title', 'Unknown title')}")
            return idea_data
            
        except Exception as e:
            logger.error(f"Error getting idea by ID: {e}")
            return None
    
    def create_project(self, idea_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new project based on a selected article idea.
        
        Args:
            idea_id: ID of the idea to use for the project (if None, uses the next idea in the queue)
            
        Returns:
            Project metadata
        """
        logger.info("Creating new article project")
        
        # Get the idea to use for the project
        idea_data = None
        
        if idea_id:
            # Use the specified idea
            idea_data = self.get_idea_by_id(idea_id)
            if not idea_data:
                logger.error(f"Failed to create project: Idea with ID {idea_id} not found")
                return {}
        else:
            # Use the next idea in the queue
            queue_files = list(self.article_queue_dir.glob("queued_*.json"))
            if not queue_files:
                logger.error("Failed to create project: No ideas in the queue")
                return {}
            
            # Use the first idea in the queue
            with open(queue_files[0], "r") as f:
                queue_data = json.load(f)
                idea_data = queue_data.get("idea")
        
        if not idea_data:
            logger.error("Failed to create project: No valid idea data found")
            return {}
        
        try:
            # Generate a unique project ID
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            title = idea_data.get("title", "Untitled Article")
            
            # Create a sanitized version of the title for the project ID
            filename = title.lower().replace(" ", "_")[:30]
            sanitized = "".join(c if c.isalnum() or c in "_- " else "_" for c in filename)
            project_id = f"{sanitized}_{timestamp}"
            
            # Create project directory
            project_dir = self.projects_dir / project_id
            project_dir.mkdir(exist_ok=True)
            
            # Create metadata file
            metadata = {
                "project_id": project_id,
                "title": idea_data.get("title"),
                "summary": idea_data.get("summary"),
                "audience": idea_data.get("audience"),
                "keywords": idea_data.get("keywords", []),
                "created_at": datetime.now().isoformat(),
                "status": "initialized",
                "idea_id": idea_data.get("id")
            }
            
            metadata_file = project_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Create directories for project components
            (project_dir / "outline").mkdir(exist_ok=True)
            (project_dir / "paragraphs").mkdir(exist_ok=True)
            (project_dir / "article").mkdir(exist_ok=True)
            
            logger.info(f"Created project '{project_id}' for article: {title}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error creating project: {e}")
            return {}
    
    def research_competitors(self, research_topic: str) -> Dict[str, Any]:
        """Research competitor content related to the research topic.
        
        Args:
            research_topic: General topic to research for competitor analysis
            
        Returns:
            Dictionary containing competitor research data
        """
        logger.info(f"Researching competitor content for topic: {research_topic}")
        
        # Get web search results for competitor content if available
        competitor_web_results = {}
        if self.web_search and self.web_search.is_available():
            try:
                logger.info(f"Searching web for competitor content on: {research_topic}")
                web_results = self.web_search.get_competitor_content(research_topic, max_results=5)
                
                if web_results and "results" in web_results and web_results["results"]:
                    competitor_web_results = web_results
                    logger.info(f"Found {len(web_results['results'])} competitor articles from web search")
                else:
                    logger.warning("No competitor content found from web search")
            except Exception as e:
                logger.error(f"Error searching for competitor content: {e}")
        
        # Construct the prompt for competitor research
        system_prompt = (
            "You are an expert content strategist who analyzes competitor content. "
            "Your analysis should identify content gaps, common approaches, and opportunities for differentiation."
        )
        
        # Include web search results in the prompt if available
        web_results_text = ""
        if competitor_web_results and "results" in competitor_web_results:
            web_results_text = "\n\nHere are some existing articles on this topic from web search:\n"
            for i, result in enumerate(competitor_web_results["results"][:5], 1):
                web_results_text += f"{i}. {result.get('title', 'No title')}:\n"
                web_results_text += f"   {result.get('content', 'No content')[:300]}...\n"
                if result.get('url'):
                    web_results_text += f"   Source: {result.get('url')}\n"
        
        user_prompt = f"""Research existing content related to '{research_topic}' and analyze competitor approaches.{web_results_text}
        
        Provide the following information:
        1. Common content themes and angles used by competitors
        2. Content gaps or underexplored aspects of the topic
        3. Typical content structure and formats used
        4. Strengths and weaknesses of existing content
        5. Opportunities for differentiation
        
        Format your analysis as follows:
        COMMON_THEMES: [List of common themes and angles]
        CONTENT_GAPS: [List of content gaps or underexplored aspects]
        TYPICAL_STRUCTURES: [Common content structures and formats]
        STRENGTHS_WEAKNESSES: [Strengths and weaknesses of existing content]
        DIFFERENTIATION_OPPORTUNITIES: [Opportunities for differentiation]
        """
        
        try:
            response = self.openai_client.client.chat.completions.create(
                model=self.openai_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1500,
            )
            
            content = response.choices[0].message.content
            
            # Parse the competitor research
            competitor_research = {}
            current_section = None
            current_content = []
            
            lines = content.strip().split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith("COMMON_THEMES:"):
                    current_section = "common_themes"
                    current_content = [line[14:].strip()]
                elif line.startswith("CONTENT_GAPS:"):
                    if current_section:
                        competitor_research[current_section] = "\n".join(current_content)
                    current_section = "content_gaps"
                    current_content = [line[13:].strip()]
                elif line.startswith("TYPICAL_STRUCTURES:"):
                    if current_section:
                        competitor_research[current_section] = "\n".join(current_content)
                    current_section = "typical_structures"
                    current_content = [line[19:].strip()]
                elif line.startswith("STRENGTHS_WEAKNESSES:"):
                    if current_section:
                        competitor_research[current_section] = "\n".join(current_content)
                    current_section = "strengths_weaknesses"
                    current_content = [line[21:].strip()]
                elif line.startswith("DIFFERENTIATION_OPPORTUNITIES:"):
                    if current_section:
                        competitor_research[current_section] = "\n".join(current_content)
                    current_section = "differentiation_opportunities"
                    current_content = [line[29:].strip()]
                elif current_section:
                    current_content.append(line)
            
            # Add the last section
            if current_section:
                competitor_research[current_section] = "\n".join(current_content)
            
            # Add web search metadata if available
            if competitor_web_results:
                competitor_research["web_search_used"] = True
                competitor_research["web_search_timestamp"] = datetime.now().isoformat()
                competitor_research["web_sources"] = [
                    {"title": result.get("title", "Unknown"), "url": result.get("url", "")} 
                    for result in competitor_web_results.get("results", [])
                ]
            else:
                competitor_research["web_search_used"] = False
            
            # Save competitor research to file
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            research_id = f"competitor_research_{timestamp}"
            research_file = self.ideas_dir / f"{research_id}.json"
            
            competitor_research["topic"] = research_topic
            competitor_research["created_at"] = datetime.now().isoformat()
            
            with open(research_file, "w") as f:
                json.dump(competitor_research, f, indent=2)
            
            logger.info(f"Completed competitor research for topic: {research_topic}")
            return competitor_research
            
        except Exception as e:
            logger.error(f"Error researching competitors: {e}")
            return {}
    
    def generate_ideas(self, research_topic: str, num_ideas: int = 5) -> List[Dict[str, str]]:
        """Generate article ideas based on trend analysis and competitor research.
        
        Args:
            research_topic: General topic to research for ideas
            num_ideas: Number of ideas to generate
            
        Returns:
            List of generated ideas with metadata
        """
        logger.info(f"Generating {num_ideas} ideas for topic: {research_topic}")
        
        # First perform trend analysis and competitor research
        trend_analysis = self.analyze_trends(research_topic)
        competitor_research = self.research_competitors(research_topic)
        
        # Construct the prompt for idea generation
        system_prompt = (
            "You are an expert content strategist who generates engaging article ideas. "
            "Your ideas should be specific, valuable to readers, and have potential for high engagement."
        )
        
        # Include insights from trend analysis and competitor research
        trend_insights = ""
        if trend_analysis:
            trend_insights = f"""\n\nTrend Analysis Insights:\n"""
            if "trending_subtopics" in trend_analysis:
                trend_insights += f"Trending Subtopics: {trend_analysis['trending_subtopics']}\n"
            if "key_questions" in trend_analysis:
                trend_insights += f"Key Questions: {trend_analysis['key_questions']}\n"
            if "recent_developments" in trend_analysis:
                trend_insights += f"Recent Developments: {trend_analysis['recent_developments']}\n"
            if "timely_considerations" in trend_analysis:
                trend_insights += f"Timely Considerations: {trend_analysis['timely_considerations']}\n"
            if "popular_formats" in trend_analysis:
                trend_insights += f"Popular Formats: {trend_analysis['popular_formats']}\n"
        
        competitor_insights = ""
        if competitor_research:
            competitor_insights = f"""\n\nCompetitor Research Insights:\n"""
            if "content_gaps" in competitor_research:
                competitor_insights += f"Content Gaps: {competitor_research['content_gaps']}\n"
            if "differentiation_opportunities" in competitor_research:
                competitor_insights += f"Differentiation Opportunities: {competitor_research['differentiation_opportunities']}\n"
        
        user_prompt = f"""Generate {num_ideas} article ideas related to '{research_topic}'.{trend_insights}{competitor_insights}
        
        For each idea, provide:
        1. A compelling article title
        2. A brief description (2-3 sentences) explaining what the article would cover
        3. Target audience for this article
        4. Estimated audience engagement potential (high, medium, low) with brief justification
        
        Format each idea as follows:
        TITLE: [Article Title]
        DESCRIPTION: [Brief description]
        AUDIENCE: [Target audience]
        ENGAGEMENT: [Potential] - [Justification]
        """
        
        try:
            response = self.openai_client.client.chat.completions.create(
                model=self.openai_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=2000,
            )
            
            content = response.choices[0].message.content
            
            # Parse the ideas from the response
            ideas = self._parse_ideas(content)
            
            # Save ideas to files
            saved_ideas = []
            for idea in ideas:
                idea_id = self._save_idea(idea)
                saved_ideas.append({**idea, "id": idea_id})
            
            logger.info(f"Generated and saved {len(saved_ideas)} ideas")
            return saved_ideas
            
        except Exception as e:
            logger.error(f"Error generating ideas: {e}")
            return []
    
    def _parse_ideas(self, content: str) -> List[Dict[str, str]]:
        """Parse ideas from the OpenAI response.
        
        Args:
            content: Raw text response from OpenAI
            
        Returns:
            List of parsed ideas
        """
        ideas = []
        current_idea = {}
        
        lines = content.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("TITLE:"):
                # If we have a previous idea, add it to the list
                if current_idea and "title" in current_idea:
                    ideas.append(current_idea)
                    current_idea = {}
                
                current_idea["title"] = line[6:].strip()
            elif line.startswith("DESCRIPTION:"):
                current_idea["description"] = line[12:].strip()
            elif line.startswith("AUDIENCE:"):
                current_idea["audience"] = line[9:].strip()
            elif line.startswith("ENGAGEMENT:"):
                current_idea["engagement"] = line[11:].strip()
        
        # Add the last idea if it exists
        if current_idea and "title" in current_idea:
            ideas.append(current_idea)
        
        return ideas
    
    def _save_idea(self, idea: Dict[str, str]) -> str:
        """Save an idea to a file in the ideas directory.
        
        Args:
            idea: Idea dictionary with metadata
            
        Returns:
            Idea ID (filename without extension)
        """
        # Create a timestamp-based ID for the idea
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        idea_id = f"idea_{timestamp}"
        
        # Add metadata
        idea_with_metadata = {
            **idea,
            "created_at": datetime.now().isoformat(),
            "status": "new"
        }
        
        # Save to file
        idea_file = self.ideas_dir / f"{idea_id}.json"
        with open(idea_file, "w") as f:
            json.dump(idea_with_metadata, f, indent=2)
        
        return idea_id
    
    def evaluate_ideas(self, max_ideas: int = 10, 
                      similarity_threshold: float = 0.7) -> Optional[str]:
        """Evaluate existing ideas and select the best one for the article queue.
        Uses feedback insights if available to improve idea selection.
        
        Args:
            max_ideas: Maximum number of ideas to evaluate
            similarity_threshold: Threshold for considering ideas similar
            
        Returns:
            ID of the selected idea, or None if no idea was selected
        """
        logger.info("Evaluating ideas for article queue")
        
        # Get list of idea files
        idea_files = list(self.ideas_dir.glob("*.json"))
        if not idea_files:
            logger.warning("No ideas found for evaluation")
            return None
        
        # Sort by creation time (newest first) and limit to max_ideas
        idea_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        idea_files = idea_files[:max_ideas]
        
        # Load ideas
        ideas = []
        for file in idea_files:
            try:
                with open(file, "r") as f:
                    idea = json.load(f)
                    idea["file_path"] = file
                    ideas.append(idea)
            except Exception as e:
                logger.error(f"Error loading idea from {file}: {e}")
        
        if not ideas:
            logger.warning("No valid ideas found for evaluation")
            return None
            
        # Apply feedback insights to score ideas if feedback is enabled
        feedback_insights = None
        if self.use_feedback and self.feedback_manager:
            try:
                # Score ideas based on previous performance data
                scored_ideas = self.feedback_manager.apply_insights_to_idea_evaluation(ideas)
                if scored_ideas:
                    logger.info(f"Applied feedback insights to {len(scored_ideas)} ideas")
                    
                    # Log the scoring results
                    for scored_idea in scored_ideas:
                        logger.info(f"Idea '{scored_idea['title']}' scored {scored_idea['score']}: {', '.join(scored_idea['reasons'])}")
                    
                    # Get insights for prompt enhancement
                    feedback_insights = self.feedback_manager.get_performance_insights()
            except Exception as e:
                logger.error(f"Error applying feedback insights: {e}")
        
        # Get recently published articles for similarity comparison
        recent_articles = self._get_recent_articles()
        
        # Construct the prompt for idea evaluation
        system_prompt = (
            "You are an expert content strategist who evaluates article ideas. "
            "Your goal is to select the idea with the highest potential for audience engagement "
            "and value, while avoiding topics too similar to recently published articles."
        )
        
        # Add feedback insights to system prompt if available
        if feedback_insights and any(insights for key, insights in feedback_insights.items() 
                                  if key != 'general_recommendations' and insights):
            feedback_prompt = "\n\nBased on previous article performance data, consider these insights:\n"
            
            if feedback_insights.get('top_topics'):
                feedback_prompt += f"- High-performing topics: {', '.join(feedback_insights['top_topics'])}\n"
                
            if feedback_insights.get('top_audiences'):
                feedback_prompt += f"- Engaged audiences: {', '.join(feedback_insights['top_audiences'])}\n"
                
            if feedback_insights.get('top_styles'):
                feedback_prompt += f"- Effective content styles: {', '.join(feedback_insights['top_styles'])}\n"
                
            if feedback_insights.get('general_recommendations'):
                feedback_prompt += "\nRecommendations:\n" + "\n".join([f"- {rec}" for rec in feedback_insights['general_recommendations']])
                
            system_prompt += feedback_prompt
        
        ideas_text = "\n\n".join([f"IDEA {i+1}:\nTitle: {idea['title']}\n"
                               f"Description: {idea['description']}\n"
                               f"Audience: {idea.get('audience', 'Not specified')}\n"
                               f"Engagement: {idea.get('engagement', 'Not specified')}"
                               for i, idea in enumerate(ideas)])
        
        recent_articles_text = "\n\n".join([f"ARTICLE {i+1}:\nTitle: {article['title']}\n"
                                        f"Summary: {article.get('summary', 'Not available')}"
                                        for i, article in enumerate(recent_articles)])
        
        user_prompt = f"""Evaluate the following article ideas and select the ONE with the highest 
        potential for audience engagement and value.
        
        IDEAS TO EVALUATE:\n{ideas_text}
        
        RECENTLY PUBLISHED ARTICLES (avoid similar topics):\n{recent_articles_text}
        
        Analyze each idea based on:
        1. Potential audience engagement (high/medium/low)
        2. Value provided to the audience (high/medium/low)
        3. Uniqueness compared to recent articles (avoid similarity above {similarity_threshold*100}%)
        4. Current relevance and timeliness
        
        Return your evaluation in this format:
        SELECTED IDEA: [Number of the selected idea, e.g., IDEA 3]
        JUSTIFICATION: [Detailed explanation of why this idea was selected]
        SIMILARITY CONCERNS: [Any similarity concerns with recent articles]
        IMPROVEMENT SUGGESTIONS: [Suggestions to improve the selected idea]
        """
        
        try:
            response = self.openai_client.client.chat.completions.create(
                model=self.openai_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent evaluation
                max_tokens=1500,
            )
            
            content = response.choices[0].message.content
            
            # Parse the selected idea
            selected_idea_num = None
            justification = ""
            similarity_concerns = ""
            improvement_suggestions = ""
            
            lines = content.strip().split("\n")
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith("SELECTED IDEA:"):
                    current_section = "selected_idea"
                    # Extract the idea number
                    idea_text = line[14:].strip()
                    if "IDEA" in idea_text and any(c.isdigit() for c in idea_text):
                        for c in idea_text:
                            if c.isdigit():
                                selected_idea_num = int(c) - 1  # Convert to 0-based index
                                break
                elif line.startswith("JUSTIFICATION:"):
                    current_section = "justification"
                    justification = line[14:].strip()
                elif line.startswith("SIMILARITY CONCERNS:"):
                    current_section = "similarity_concerns"
                    similarity_concerns = line[20:].strip()
                elif line.startswith("IMPROVEMENT SUGGESTIONS:"):
                    current_section = "improvement_suggestions"
                    improvement_suggestions = line[24:].strip()
                elif current_section == "justification":
                    justification += " " + line
                elif current_section == "similarity_concerns":
                    similarity_concerns += " " + line
                elif current_section == "improvement_suggestions":
                    improvement_suggestions += " " + line
            
            # If no idea was selected or the index is invalid, return None
            if selected_idea_num is None or selected_idea_num < 0 or selected_idea_num >= len(ideas):
                logger.warning("No valid idea was selected during evaluation")
                return None
            
            # Get the selected idea
            selected_idea = ideas[selected_idea_num]
            
            # Add evaluation data to the idea
            selected_idea["evaluation"] = {
                "justification": justification,
                "similarity_concerns": similarity_concerns,
                "improvement_suggestions": improvement_suggestions,
                "evaluated_at": datetime.now().isoformat()
            }
            
            # Update the idea file with evaluation data
            with open(selected_idea["file_path"], "w") as f:
                json.dump(selected_idea, f, indent=2)
            
            # Move the idea to the article queue
            self._move_to_article_queue(selected_idea)
            
            logger.info(f"Selected idea '{selected_idea['title']}' for article queue")
            return selected_idea.get("id")
            
        except Exception as e:
            logger.error(f"Error evaluating ideas: {e}")
            return ''
    
    def _get_recent_articles(self, max_articles: int = 5) -> List[Dict[str, Any]]:
        """Get information about recently published articles.
        
        Args:
            max_articles: Maximum number of recent articles to retrieve
            
        Returns:
            List of recent article metadata
        """
        # In a real implementation, this would retrieve data from a database or API
        # For now, we'll just check the projects directory for completed articles
        
        recent_articles = []
        completed_projects = list(self.projects_dir.glob("*/final_article.json"))
        
        # Sort by modification time (newest first) and limit to max_articles
        completed_projects.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        completed_projects = completed_projects[:max_articles]
        
        for project_file in completed_projects:
            try:
                with open(project_file, "r") as f:
                    article = json.load(f)
                    recent_articles.append(article)
            except Exception as e:
                logger.error(f"Error loading article from {project_file}: {e}")
        
        return recent_articles
    
    def _move_to_article_queue(self, idea: Dict[str, Any]) -> None:
        """Move a selected idea to the article queue.
        
        Args:
            idea: The idea to move to the queue
        """
        # Update idea status
        idea["status"] = "queued"
        idea["queued_at"] = datetime.now().isoformat()
        
        # Save to article queue directory
        idea_id = idea.get("id", f"idea_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        queue_file = self.article_queue_dir / f"{idea_id}.json"
        
        with open(queue_file, "w") as f:
            json.dump(idea, f, indent=2)
        
        logger.info(f"Moved idea '{idea['title']}' to article queue")
    
    def create_project(self) -> Optional[str]:
        """Create a project for the next article in the queue.
        
        Returns:
            Project ID if successful, None otherwise
        """
        logger.info("Creating project for next article in queue")
        
        # Get the next idea from the queue (oldest first)
        queue_files = list(self.article_queue_dir.glob("*.json"))
        if not queue_files:
            logger.warning("No ideas in the article queue")
            return None
        
        # Sort by creation time (oldest first)
        queue_files.sort(key=lambda x: x.stat().st_mtime)
        next_idea_file = queue_files[0]
        
        try:
            with open(next_idea_file, "r") as f:
                idea = json.load(f)
        except Exception as e:
            logger.error(f"Error loading idea from {next_idea_file}: {e}")
            return None
        
        # Create a project ID and directory
        project_id = f"project_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        project_dir = self.projects_dir / project_id
        project_dir.mkdir(exist_ok=True)
        
        # Save idea to project directory
        with open(project_dir / "idea.json", "w") as f:
            json.dump(idea, f, indent=2)
        
        # Update idea status
        idea["status"] = "in_progress"
        idea["project_id"] = project_id
        idea["project_created_at"] = datetime.now().isoformat()
        
        # Save updated idea back to queue file
        with open(next_idea_file, "w") as f:
            json.dump(idea, f, indent=2)
        
        logger.info(f"Created project '{project_id}' for idea '{idea['title']}'")
        return project_id
    
    def generate_outline(self, project_id: str) -> Optional[List[str]]:
        """Generate an outline for the article.
        
        Args:
            project_id: ID of the project
            
        Returns:
            List of outline sections if successful, None otherwise
        """
        logger.info(f"Generating outline for project {project_id}")
        
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            logger.error(f"Project directory {project_dir} does not exist")
            return None
        
        # Load the idea
        try:
            with open(project_dir / "idea.json", "r") as f:
                idea = json.load(f)
        except Exception as e:
            logger.error(f"Error loading idea for project {project_id}: {e}")
            return None
        
        # Construct the prompt for outline generation
        system_prompt = (
            "You are an expert content writer who creates well-structured article outlines. "
            "Your outlines should be comprehensive, logical, and designed to engage readers "
            "while delivering maximum value."
        )
        
        user_prompt = f"""Create a detailed outline for an article with the title: '{idea['title']}'
        
        Article description: {idea.get('description', 'Not provided')}
        Target audience: {idea.get('audience', 'Not specified')}
        
        The outline should include:
        1. An introduction section
        2. 4-7 main sections with clear, descriptive headings
        3. A conclusion section
        
        For each main section, include 2-3 bullet points describing key points to cover.
        
        Format the outline as a list of section headings only, without the bullet points.
        Each heading should be on a new line and be descriptive enough to guide the writing process.
        """
        
        try:
            response = self.openai_client.client.chat.completions.create(
                model=self.openai_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
            )
            
            content = response.choices[0].message.content
            
            # Parse the outline
            outline = [line.strip() for line in content.strip().split("\n") if line.strip()]
            
            # Save the outline to the project directory
            outline_data = {
                "title": idea["title"],
                "sections": outline,
                "generated_at": datetime.now().isoformat()
            }
            
            with open(project_dir / "outline.json", "w") as f:
                json.dump(outline_data, f, indent=2)
            
            logger.info(f"Generated outline with {len(outline)} sections for project {project_id}")
            return outline
            
        except Exception as e:
            logger.error(f"Error generating outline: {e}")
            return None
    
    def generate_paragraphs(self, project_id: str) -> bool:
        """Generate paragraphs for each section of the outline.
        
        Args:
            project_id: ID of the project
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Generating paragraphs for project {project_id}")
        
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            logger.error(f"Project directory {project_dir} does not exist")
            return False
        
        # Load the idea and outline
        try:
            with open(project_dir / "idea.json", "r") as f:
                idea = json.load(f)
                
            with open(project_dir / "outline.json", "r") as f:
                outline_data = json.load(f)
                outline = outline_data["sections"]
        except Exception as e:
            logger.error(f"Error loading project data: {e}")
            return False
        
        # Create paragraphs directory
        paragraphs_dir = project_dir / "paragraphs"
        paragraphs_dir.mkdir(exist_ok=True)
        
        # Generate paragraph for each section
        for i, section in enumerate(outline):
            paragraph = self._generate_paragraph(idea, outline, section, i)
            if paragraph:
                # Save paragraph to file
                section_filename = f"{i+1:02d}_{self._sanitize_filename(section)}.md"
                with open(paragraphs_dir / section_filename, "w") as f:
                    f.write(f"# {section}\n\n{paragraph}")
                
                logger.info(f"Generated paragraph for section '{section}'")
            else:
                logger.error(f"Failed to generate paragraph for section '{section}'")
                return False
        
        return True
    
    def _generate_paragraph(self, idea: Dict[str, Any], outline: List[str], 
                          section: str, section_index: int) -> Optional[str]:
        """Generate a paragraph for a specific section of the outline.
        
        Args:
            idea: The article idea
            outline: The full article outline
            section: The current section to generate content for
            section_index: Index of the section in the outline
            
        Returns:
            Generated paragraph text if successful, None otherwise
        """
        # Determine if this is introduction, body, or conclusion
        section_type = "introduction" if section_index == 0 else \
                      "conclusion" if section_index == len(outline) - 1 else "body"
        
        # Construct the prompt for paragraph generation
        system_prompt = (
            "You are an expert content writer who creates engaging, informative paragraphs. "
            "Your writing should be clear, valuable to readers, and maintain a consistent tone "
            "throughout the article."
        )
        
        # Include the full outline for context
        outline_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(outline)])
        
        user_prompt = f"""Write a detailed paragraph for the following section of an article:
        
        Article Title: {idea['title']}
        Article Description: {idea.get('description', 'Not provided')}
        Target Audience: {idea.get('audience', 'Not specified')}
        
        Full Article Outline:
        {outline_text}
        
        Current Section: {section} (Section {section_index+1} of {len(outline)})
        Section Type: {section_type}
        
        Guidelines:
        - Write approximately 150-300 words for this section
        - Maintain a {idea.get('tone', 'informative')} tone
        - Include specific details, examples, or data points where appropriate
        - For introduction: Hook the reader and introduce the topic
        - For body sections: Provide valuable information and insights
        - For conclusion: Summarize key points and include a call to action
        
        Write ONLY the content for this section, without the heading.
        """
        
        try:
            response = self.openai_client.client.chat.completions.create(
                model=self.openai_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
            )
            
            content = response.choices[0].message.content.strip()
            return content
            
        except Exception as e:
            logger.error(f"Error generating paragraph: {e}")
            return None
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize a string for use as a filename.
        
        Args:
            filename: The string to sanitize
            
        Returns:
            Sanitized filename
        """
        # Replace spaces with underscores and remove special characters
        sanitized = "".join(c if c.isalnum() or c in "_- " else "_" for c in filename)
        sanitized = sanitized.replace(" ", "_").lower()
        return sanitized[:50]  # Limit length
        
    def get_idea_by_id(self, idea_id: str) -> Optional[Dict[str, Any]]:
        """Get an idea by its ID.
        
        Args:
            idea_id: ID of the idea to retrieve
            
        Returns:
            Idea dictionary if found, None otherwise
        """
        logger.info(f"Getting idea with ID: {idea_id}")
        
        # Look for the idea file in the ideas directory
        idea_file = self.ideas_dir / f"{idea_id}.json"
        
        if not idea_file.exists():
            # Also check the article queue directory
            idea_file = self.article_queue_dir / f"{idea_id}.json"
            
            if not idea_file.exists():
                logger.warning(f"Idea with ID {idea_id} not found")
                return None
        
        try:
            with open(idea_file, "r") as f:
                idea = json.load(f)
                return idea
        except Exception as e:
            logger.error(f"Error loading idea with ID {idea_id}: {e}")
            return None
    
    def assemble_article(self, project_id: str) -> Optional[Dict[str, str]]:
        """Assemble the paragraphs into a complete article.
        
        Args:
            project_id: ID of the project
            
        Returns:
            Dictionary with title and content if successful, None otherwise
        """
        logger.info(f"Assembling article for project {project_id}")
        
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            logger.error(f"Project directory {project_dir} does not exist")
            return None
        
        # Load the idea and outline
        try:
            with open(project_dir / "idea.json", "r") as f:
                idea = json.load(f)
                
            with open(project_dir / "outline.json", "r") as f:
                outline_data = json.load(f)
                outline = outline_data["sections"]
        except Exception as e:
            logger.error(f"Error loading project data: {e}")
            return None
        
        # Check if paragraphs directory exists
        paragraphs_dir = project_dir / "paragraphs"
        if not paragraphs_dir.exists():
            logger.error(f"Paragraphs directory does not exist for project {project_id}")
            return None
        
        # Get paragraph files and sort them by section number
        paragraph_files = list(paragraphs_dir.glob("*.md"))
        paragraph_files.sort()
        
        if not paragraph_files:
            logger.error(f"No paragraph files found for project {project_id}")
            return None
        
        # Assemble the article
        article_content = f"# {idea['title']}\n\n"
        
        for file in paragraph_files:
            try:
                with open(file, "r") as f:
                    content = f.read()
                    
                # Extract the section heading and content
                lines = content.strip().split("\n")
                if lines and lines[0].startswith("# "):
                    heading = lines[0][2:]
                    paragraph = "\n".join(lines[2:]) if len(lines) > 2 else ""
                    
                    article_content += f"## {heading}\n\n{paragraph}\n\n"
                else:
                    article_content += f"{content}\n\n"
                    
            except Exception as e:
                logger.error(f"Error reading paragraph file {file}: {e}")
        
        # Save the assembled article
        assembled_article = {
            "title": idea["title"],
            "content": article_content.strip(),
            "assembled_at": datetime.now().isoformat()
        }
        
        with open(project_dir / "assembled_article.json", "w") as f:
            json.dump(assembled_article, f, indent=2)
        
        logger.info(f"Assembled article for project {project_id}")
        return assembled_article
    
    def refine_article(self, project_id: str) -> Optional[Dict[str, str]]:
        """Refine the assembled article for cohesiveness.
        
        Args:
            project_id: ID of the project
            
        Returns:
            Dictionary with title and content if successful, None otherwise
        """
        logger.info(f"Refining article for project {project_id}")
        
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            logger.error(f"Project directory {project_dir} does not exist")
            return None
            
        # Load the assembled article
        try:
            with open(project_dir / "assembled_article.json", "r") as f:
                assembled_article = json.load(f)
                
            with open(project_dir / "idea.json", "r") as f:
                idea = json.load(f)
        except Exception as e:
            logger.error(f"Error loading assembled article: {e}")
            return None
        
        # Construct the prompt for article refinement
        system_prompt = (
            "You are an expert editor who refines articles to make them cohesive, engaging, and valuable. "
            "Your task is to improve the flow, consistency, and overall quality of the article "
            "while preserving its core content and structure."
        )
        
        user_prompt = f"""Refine the following article to make it more cohesive and engaging.
        
        Article Title: {assembled_article['title']}
        Target Audience: {idea.get('audience', 'Not specified')}
        
        Guidelines:
        1. Improve transitions between sections to create better flow
        2. Ensure consistent tone and voice throughout
        3. Enhance clarity and readability
        4. Fix any grammatical or stylistic issues
        5. Do NOT change the overall structure or main points
        6. Preserve all section headings
        7. Maintain the same approximate length
        
        ARTICLE CONTENT:
        {assembled_article['content']}
        
        Return the refined article in Markdown format, preserving all headings and formatting.
        """
        
        try:
            response = self.openai_client.client.chat.completions.create(
                model=self.openai_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=4000,
            )
            
            refined_content = response.choices[0].message.content.strip()
            
            # Save the refined article
            final_article = {
                "title": assembled_article["title"],
                "content": refined_content,
                "original_content": assembled_article["content"],
                "refined_at": datetime.now().isoformat(),
                "idea": idea,
                "summary": idea.get("description", "")
            }
            
            with open(project_dir / "final_article.json", "w") as f:
                json.dump(final_article, f, indent=2)
                
            # Also save as markdown for easy viewing
            with open(project_dir / "final_article.md", "w") as f:
                f.write(f"# {final_article['title']}\n\n{refined_content}")
            
            logger.info(f"Refined article for project {project_id}")
            return final_article
            
        except Exception as e:
            logger.error(f"Error refining article: {e}")
            return None
            
    def optimize_seo(self, project_id: str) -> Optional[Dict[str, str]]:
        """Optimize the article for SEO before publishing.
        
        Args:
            project_id: ID of the project
            
        Returns:
            Dictionary with SEO-optimized title and content if successful, None otherwise
        """
        logger.info(f"Optimizing SEO for project {project_id}")
        
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            logger.error(f"Project directory {project_dir} does not exist")
            return None
            
        # Load the final article
        try:
            with open(project_dir / "final_article.json", "r") as f:
                final_article = json.load(f)
                
            with open(project_dir / "idea.json", "r") as f:
                idea = json.load(f)
        except Exception as e:
            logger.error(f"Error loading final article: {e}")
            return None
        
        # Construct the prompt for SEO optimization
        system_prompt = (
            "You are an SEO expert who optimizes content for search engines while maintaining readability and value. "
            "Your task is to enhance the article's discoverability without compromising its quality or reader experience."
        )
        
        user_prompt = f"""Optimize the following article for search engines.
        
        Article Title: {final_article['title']}
        Target Audience: {idea.get('audience', 'Not specified')}
        
        Guidelines:
        1. Suggest an SEO-optimized title (if needed)
        2. Recommend meta description (150-160 characters)
        3. Identify 5-7 primary and secondary keywords
        4. Optimize headings (H1, H2, H3) for keyword inclusion
        5. Suggest internal and external linking opportunities
        6. Recommend image alt text where appropriate
        7. Ensure proper keyword density (avoid keyword stuffing)
        8. Maintain readability and natural language flow
        
        ARTICLE CONTENT:
        {final_article['content']}
        
        Format your response as follows:
        SEO_TITLE: [Optimized title]
        META_DESCRIPTION: [Meta description]
        KEYWORDS: [List of keywords]
        OPTIMIZED_CONTENT: [The full optimized content in Markdown format]
        ADDITIONAL_RECOMMENDATIONS: [Any other SEO recommendations]
        """
        
        try:
            response = self.openai_client.client.chat.completions.create(
                model=self.openai_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=4000,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse the SEO optimization results
            seo_optimized = {}
            current_section = None
            current_content = []
            
            lines = content.strip().split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith("SEO_TITLE:"):
                    current_section = "seo_title"
                    current_content = [line[10:].strip()]
                elif line.startswith("META_DESCRIPTION:"):
                    if current_section:
                        seo_optimized[current_section] = "\n".join(current_content)
                    current_section = "meta_description"
                    current_content = [line[17:].strip()]
                elif line.startswith("KEYWORDS:"):
                    if current_section:
                        seo_optimized[current_section] = "\n".join(current_content)
                    current_section = "keywords"
                    current_content = [line[9:].strip()]
                elif line.startswith("OPTIMIZED_CONTENT:"):
                    if current_section:
                        seo_optimized[current_section] = "\n".join(current_content)
                    current_section = "optimized_content"
                    current_content = []
                elif line.startswith("ADDITIONAL_RECOMMENDATIONS:"):
                    if current_section:
                        seo_optimized[current_section] = "\n".join(current_content)
                    current_section = "additional_recommendations"
                    current_content = [line[28:].strip()]
                elif current_section:
                    current_content.append(line)
            
            # Add the last section
            if current_section:
                seo_optimized[current_section] = "\n".join(current_content)
            
            # Create the SEO-optimized article
            seo_article = {
                "title": seo_optimized.get("seo_title", final_article["title"]),
                "content": seo_optimized.get("optimized_content", final_article["content"]),
                "original_title": final_article["title"],
                "original_content": final_article["content"],
                "meta_description": seo_optimized.get("meta_description", ""),
                "keywords": seo_optimized.get("keywords", ""),
                "additional_recommendations": seo_optimized.get("additional_recommendations", ""),
                "seo_optimized_at": datetime.now().isoformat(),
                "idea": idea,
                "summary": idea.get("description", "")
            }
            
            # Save the SEO-optimized article
            with open(project_dir / "seo_optimized_article.json", "w") as f:
                json.dump(seo_article, f, indent=2)
                
            # Also save as markdown for easy viewing
            with open(project_dir / "seo_optimized_article.md", "w") as f:
                f.write(f"# {seo_article['title']}\n\n{seo_article['content']}")
            
            logger.info(f"SEO optimization completed for project {project_id}")
            return seo_article
            
        except Exception as e:
            logger.error(f"Error optimizing SEO: {e}")
            return None
            
    def run_pipeline(self, research_topic: str = None, num_ideas: int = 5, 
                    max_ideas_to_evaluate: int = 10) -> Optional[Dict[str, str]]:
        """Run the complete article generation pipeline.
        
        Args:
            research_topic: Topic to research for idea generation (if None, skip idea generation)
            num_ideas: Number of ideas to generate
            max_ideas_to_evaluate: Maximum number of ideas to evaluate
            
        Returns:
            Dictionary with the final article if successful, None otherwise
        """
        logger.info("Starting article generation pipeline")
        
        # Step 1a: Analyze trends related to the research topic
        if research_topic:
            logger.info(f"Step 1a: Analyzing trends for topic '{research_topic}'")
            trend_analysis = self.analyze_trends(research_topic)
            if not trend_analysis:
                logger.warning("Trend analysis produced limited results. Continuing pipeline.")
            else:
                logger.info(f"Completed trend analysis for topic '{research_topic}'")
                
            # Step 1b: Research competitor content
            logger.info(f"Step 1b: Researching competitor content for topic '{research_topic}'")
            competitor_research = self.research_competitors(research_topic)
            if not competitor_research:
                logger.warning("Competitor research produced limited results. Continuing pipeline.")
            else:
                logger.info(f"Completed competitor research for topic '{research_topic}'")
            
            # Step 1c: Generate ideas based on research
            logger.info(f"Step 1c: Generating ideas for topic '{research_topic}'")
            ideas = self.generate_ideas(research_topic, num_ideas)
            if not ideas:
                logger.error("Failed to generate ideas. Pipeline stopped.")
                return None
            logger.info(f"Generated {len(ideas)} ideas")
        else:
            logger.info("Skipping research and idea generation steps")
        
        # Step 2: Evaluate ideas and select the best one
        logger.info("Step 2: Evaluating ideas and selecting the best one")
        selected_idea_id = self.evaluate_ideas(max_ideas=max_ideas_to_evaluate)
        if not selected_idea_id:
            logger.error("Failed to select an idea. Pipeline stopped.")
            return None
        logger.info(f"Selected idea with ID: {selected_idea_id}")
        
        # Step 3: Create a project for the selected idea
        logger.info("Step 3: Creating project for the selected idea")
        project_id = self.create_project()
        if not project_id:
            logger.error("Failed to create project. Pipeline stopped.")
            return None
        logger.info(f"Created project with ID: {project_id}")
        
        # Step 4: Generate outline for the article
        logger.info("Step 4: Generating outline for the article")
        outline = self.generate_outline(project_id)
        if not outline:
            logger.error("Failed to generate outline. Pipeline stopped.")
            return None
        logger.info(f"Generated outline with {len(outline)} sections")
        
        # Step 5: Generate paragraphs for each section of the outline
        logger.info("Step 5: Generating paragraphs for each section")
        if not self.generate_paragraphs(project_id):
            logger.error("Failed to generate paragraphs. Pipeline stopped.")
            return None
        logger.info("Generated paragraphs for all sections")
        
        # Step 6: Assemble the article from the paragraphs
        logger.info("Step 6: Assembling the article")
        assembled_article = self.assemble_article(project_id)
        if not assembled_article:
            logger.error("Failed to assemble article. Pipeline stopped.")
            return None
        logger.info("Assembled article successfully")
        
        # Step 7: Refine the article for cohesiveness
        logger.info("Step 7: Refining the article")
        final_article = self.refine_article(project_id)
        if not final_article:
            logger.error("Failed to refine article. Pipeline stopped.")
            return None
        logger.info("Refined article successfully")
        
        # Step 8: Optimize the article for SEO
        logger.info("Step 8: Optimizing the article for SEO")
        seo_article = self.optimize_seo(project_id)
        if not seo_article:
            logger.error("Failed to optimize article for SEO. Using unoptimized version.")
            seo_article = final_article
        else:
            logger.info("SEO optimization completed successfully")
        
        logger.info(f"Article generation pipeline completed successfully for project {project_id}")
        return seo_article
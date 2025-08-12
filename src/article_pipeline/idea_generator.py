"""Idea generation and evaluation functionality for the article pipeline."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger

from src.openai_client import OpenAIClient
from .trend_analyzer import TrendAnalyzer

class IdeaGenerator:
    """Handles article idea generation and evaluation."""
    
    def __init__(self, openai_client: OpenAIClient, ideas_dir: Path, article_queue_dir: Path, trend_analyzer: Optional[TrendAnalyzer] = None):
        """Initialize the idea generator.
        
        Args:
            openai_client: OpenAI client for API interactions
            ideas_dir: Directory to store generated ideas
            article_queue_dir: Directory for queued articles
            trend_analyzer: Trend analyzer for research insights (optional)
        """
        self.openai_client = openai_client
        self.ideas_dir = ideas_dir
        self.article_queue_dir = article_queue_dir
        self.trend_analyzer = trend_analyzer
        self._idea_counter = 0  # Counter for generating unique idea IDs
    
    def generate_ideas(self, research_topic: str, num_ideas: int = 5, trend_analysis: Optional[Dict[str, Any]] = None, competitor_research: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        """Generate article ideas based on the research topic.
        
        Args:
            research_topic: Topic to generate ideas for
            num_ideas: Number of ideas to generate
            trend_analysis: Optional pre-existing trend analysis
            competitor_research: Optional pre-existing competitor research
            
        Returns:
            List of generated idea dictionaries
        """
        logger.info(f"Generating {num_ideas} ideas for topic: {research_topic}")
        
        # Get trend analysis if not provided
        if trend_analysis is None and self.trend_analyzer:
            logger.info(f"Analyzing trends for topic: {research_topic}")
            trend_analysis = self.trend_analyzer.analyze_trends(research_topic)
        
        # Get competitor research if not provided
        if competitor_research is None and self.trend_analyzer:
            logger.info(f"Researching competitors for topic: {research_topic}")
            competitor_research = self.trend_analyzer.research_competitors(research_topic)
        
        system_prompt = (
            "You are an expert content strategist who generates engaging article ideas. "
            "Your ideas should be unique, valuable, and aligned with current trends."
        )
        
        # Include trend analysis in the prompt if available
        trend_analysis_text = ""
        if trend_analysis:
            trend_analysis_text = "\n\nHere is the trend analysis to inform your ideas:\n"
            
            if "trending_subtopics" in trend_analysis:
                trend_analysis_text += "\nTrending Subtopics:\n"
                trend_analysis_text += trend_analysis["trending_subtopics"] + "\n"
            
            if "key_questions" in trend_analysis:
                trend_analysis_text += "\nKey Questions:\n"
                trend_analysis_text += trend_analysis["key_questions"] + "\n"
            
            if "recent_developments" in trend_analysis:
                trend_analysis_text += "\nRecent Developments:\n"
                trend_analysis_text += trend_analysis["recent_developments"] + "\n"
            
            if "timely_considerations" in trend_analysis:
                trend_analysis_text += "\nTimely Considerations:\n"
                trend_analysis_text += trend_analysis["timely_considerations"] + "\n"
            
            if "popular_formats" in trend_analysis:
                trend_analysis_text += "\nPopular Formats:\n"
                trend_analysis_text += trend_analysis["popular_formats"] + "\n"
        logger.info(f"Trend analysis: {trend_analysis}")
        
        # Include competitor research in the prompt if available
        competitor_research_text = ""
        if competitor_research:
            competitor_research_text = "\n\nHere is the competitor research to inform your ideas:\n"
            
            if "common_themes" in competitor_research:
                competitor_research_text += "\nCommon Themes:\n"
                competitor_research_text += competitor_research["common_themes"] + "\n"
            
            if "content_gaps" in competitor_research:
                competitor_research_text += "\nContent Gaps:\n"
                competitor_research_text += competitor_research["content_gaps"] + "\n"
            
            if "typical_structures" in competitor_research:
                competitor_research_text += "\nTypical Structures:\n"
                competitor_research_text += competitor_research["typical_structures"] + "\n"
            
            if "strengths_weaknesses" in competitor_research:
                competitor_research_text += "\nStrengths and Weaknesses:\n"
                competitor_research_text += competitor_research["strengths_weaknesses"] + "\n"
            
            if "differentiation_opportunities" in competitor_research:
                competitor_research_text += "\nDifferentiation Opportunities:\n"
                competitor_research_text += competitor_research["differentiation_opportunities"] + "\n"
        logger.info(f"Competitor research: {competitor_research}")
        
        user_prompt = f"""Generate {num_ideas} unique article ideas related to '{research_topic}'.{trend_analysis_text}{competitor_research_text}
        
        For each idea, provide:
        1. A compelling title, do mention the year only sometimes in the title
        2. A brief description of the content
        3. The target audience
        4. Key points to cover
        5. Potential sources or references
        
        Format each idea as follows:
        TITLE: [Article title]
        DESCRIPTION: [Brief description]
        AUDIENCE: [Target audience]
        KEY_POINTS: [List of key points]
        SOURCES: [Potential sources]
        ---
        """
        
        try:
            response = self.openai_client.client.chat.completions.create(
                model=self.openai_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1,
                max_tokens=2000,
            )
            
            content = response.choices[0].message.content
            ideas = self._parse_ideas(content)
            
            # Save each idea
            for idea in ideas:
                self._save_idea(idea)
            
            return ideas
            
        except Exception as e:
            logger.error(f"Error generating ideas: {e}")
            return []
    
    def evaluate_ideas(self, max_ideas: int = 10) -> Optional[Dict[str, Any]]:
        """Evaluate and select the best idea from the generated ideas.
        
        Args:
            max_ideas: Maximum number of ideas to evaluate
            
        Returns:
            Selected idea dictionary or None if no suitable idea found
        """
        logger.info(f"Evaluating up to {max_ideas} ideas")
        
        # Get recent ideas
        ideas = self._get_recent_ideas(max_ideas)
        if not ideas:
            logger.warning("No ideas available for evaluation")
            return None
        
        system_prompt = (
            "You are an expert content strategist who evaluates article ideas. "
            "Your evaluation should consider uniqueness, value, and potential impact."
        )
        
        # Prepare ideas for evaluation
        ideas_text = ""
        for i, idea in enumerate(ideas, 1):
            ideas_text += f"\nIdea {i}:\n"
            ideas_text += f"Title: {idea.get('title', 'No title')}\n"
            ideas_text += f"Description: {idea.get('description', 'No description')}\n"
            ideas_text += f"Audience: {idea.get('audience', 'No audience specified')}\n"
            ideas_text += f"Key Points: {idea.get('key_points', 'No key points')}\n"
            ideas_text += "---\n"
        
        user_prompt = f"""Evaluate the following article ideas and select the best one:{ideas_text}
        
        Consider:
        1. Uniqueness and originality
        2. Value to the target audience
        3. Potential impact and reach
        4. Feasibility of execution
        5. Alignment with current trends
        
        Provide your evaluation in the following format:
        SELECTED_IDEA: [Number of the selected idea]
        REASONING: [Detailed explanation of why this idea was selected]
        IMPROVEMENTS: [Suggestions for improving the selected idea]
        """
        
        try:
            response = self.openai_client.client.chat.completions.create(
                model=self.openai_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1,
                max_tokens=1000,
            )
            
            content = response.choices[0].message.content
            
            # Parse the evaluation
            selected_idea_num = None
            reasoning = ""
            improvements = ""
            
            lines = content.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("SELECTED_IDEA:"):
                    try:
                        selected_idea_num = int(line[14:].strip())
                    except ValueError:
                        pass
                elif line.startswith("REASONING:"):
                    reasoning = line[10:].strip()
                elif line.startswith("IMPROVEMENTS:"):
                    improvements = line[13:].strip()
            
            if selected_idea_num is None or selected_idea_num < 1 or selected_idea_num > len(ideas):
                logger.warning("Invalid idea selection")
                return None
            
            selected_idea = ideas[selected_idea_num - 1]
            selected_idea["evaluation"] = {
                "reasoning": reasoning,
                "improvements": improvements,
                "timestamp": datetime.now().isoformat()
            }
            
            # Move selected idea to article queue
            self._move_to_article_queue(selected_idea)
            
            return selected_idea
            
        except Exception as e:
            logger.error(f"Error evaluating ideas: {e}")
            return None
    
    def _parse_ideas(self, content: str) -> List[Dict[str, str]]:
        """Parse the generated ideas from the OpenAI response.
        
        Args:
            content: Raw text response from OpenAI
            
        Returns:
            List of parsed idea dictionaries
        """
        ideas = []
        current_idea = {}
        
        lines = content.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line == "---":
                if current_idea:
                    ideas.append(current_idea)
                    current_idea = {}
                continue
            
            if line.startswith("TITLE:"):
                current_idea["title"] = line[6:].strip()
            elif line.startswith("DESCRIPTION:"):
                current_idea["description"] = line[12:].strip()
            elif line.startswith("AUDIENCE:"):
                current_idea["audience"] = line[9:].strip()
            elif line.startswith("KEY_POINTS:"):
                current_idea["key_points"] = line[11:].strip()
            elif line.startswith("SOURCES:"):
                current_idea["sources"] = line[8:].strip()
            elif current_idea:
                # Append to the last field
                last_key = list(current_idea.keys())[-1]
                current_idea[last_key] += "\n" + line
        
        # Add the last idea if exists
        if current_idea:
            ideas.append(current_idea)
        
        return ideas
    
    def _save_idea(self, idea: Dict[str, str]) -> str:
        """Save a generated idea to a file.
        
        Args:
            idea: Idea dictionary to save
            
        Returns:
            ID of the saved idea
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self._idea_counter += 1  # Increment counter for each idea
        idea_id = f"idea_{timestamp}_{self._idea_counter}"
        
        idea["id"] = idea_id
        idea["timestamp"] = timestamp
        
        idea_file = self.ideas_dir / f"{idea_id}.json"
        logger.info(f"Write idea to {idea_file}")
        with open(idea_file, "w") as f:
            json.dump(idea, f, indent=2)
        logger.info(f"Idea saved with ID {idea_id} to file {idea_file}")
        
        return idea_id
    
    def _get_recent_ideas(self, max_ideas: int) -> List[Dict[str, Any]]:
        """Get the most recently generated ideas.
        
        Args:
            max_ideas: Maximum number of ideas to retrieve
            
        Returns:
            List of recent idea dictionaries
        """
        ideas = []
        idea_files = sorted(self.ideas_dir.glob("idea_*.json"), reverse=True)
        
        for idea_file in idea_files[:max_ideas]:
            try:
                with open(idea_file) as f:
                    idea = json.load(f)
                ideas.append(idea)
            except Exception as e:
                logger.error(f"Error reading idea file {idea_file}: {e}")
        
        return ideas
    
    def _move_to_article_queue(self, idea: Dict[str, Any]) -> None:
        """Move a selected idea to the article queue.
        
        Args:
            idea: Selected idea dictionary
        """
        try:
            # Create a new file in the article queue
            queue_file = self.article_queue_dir / f"{idea['id']}.json"
            with open(queue_file, "w") as f:
                json.dump(idea, f, indent=2)
            
            # Remove the idea from the ideas directory
            idea_file = self.ideas_dir / f"{idea['id']}.json"
            if idea_file.exists():
                idea_file.unlink()
            
            logger.info(f"Moved idea {idea['id']} to article queue")
            
        except Exception as e:
            logger.error(f"Error moving idea to article queue: {e}")
    
    def get_idea_by_id(self, idea_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an idea by its ID.
        
        Args:
            idea_id: ID of the idea to retrieve
            
        Returns:
            Idea dictionary or None if not found
        """
        # Check in ideas directory
        idea_file = self.ideas_dir / f"{idea_id}.json"
        if idea_file.exists():
            try:
                with open(idea_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading idea file {idea_file}: {e}")
        
        # Check in article queue
        queue_file = self.article_queue_dir / f"{idea_id}.json"
        if queue_file.exists():
            try:
                with open(queue_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading queue file {queue_file}: {e}")
        
        return None 
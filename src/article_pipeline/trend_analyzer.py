"""Trend analysis functionality for the article pipeline."""

from datetime import datetime
from typing import Dict, Any, Optional
from loguru import logger

from src.openai_client import OpenAIClient
from src.web_search import WebSearchManager

class TrendAnalyzer:
    """Handles trend analysis for article topics."""
    
    def __init__(self, openai_client: OpenAIClient, web_search: WebSearchManager):
        """Initialize the trend analyzer.
        
        Args:
            openai_client: OpenAI client for API interactions
            web_search: Web search manager for gathering insights
        """
        self.openai_client = openai_client
        self.web_search = web_search
    
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
        
        # Construct the prompt for trend analysis
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
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {
                "error": str(e),
                "web_search_used": bool(web_insights)
            }
    
    def research_competitors(self, research_topic: str) -> Dict[str, Any]:
        """Research competitor content for a topic.
        
        Args:
            research_topic: Topic to research competitors for
            
        Returns:
            Dictionary containing competitor research data
        """
        logger.info(f"Researching competitors for topic: {research_topic}")
        
        # Get web search results if available
        web_insights = {}
        if self.web_search and self.web_search.is_available():
            try:
                logger.info(f"Gathering competitor insights for: {research_topic}")
                web_results = self.web_search.get_competitor_content(research_topic)
                
                if not web_results.get("error"):
                    web_insights = {
                        "competitor_content": web_results.get("results", []),
                        "timestamp": datetime.now().isoformat()
                    }
                    logger.info(f"Successfully gathered competitor insights")
                else:
                    logger.warning(f"Error in web search: {web_results.get('error')}")
            except Exception as e:
                logger.error(f"Error gathering competitor insights: {e}")
        
        # Construct the prompt for competitor research
        system_prompt = (
            "You are an expert content strategist who analyzes competitor content. "
            "Your analysis should identify common themes, content gaps, and opportunities for differentiation."
        )
        
        # Include web insights in the prompt if available
        web_insights_text = ""
        if web_insights:
            web_insights_text = "\n\nHere are some competitor content examples to inform your analysis:\n"
            
            # Add competitor content
            if "competitor_content" in web_insights and web_insights["competitor_content"]:
                web_insights_text += "\nCompetitor Content Examples:\n"
                for i, result in enumerate(web_insights["competitor_content"][:5], 1):
                    web_insights_text += f"{i}. {result.get('title', 'No title')}: {result.get('content', 'No content')}\n"
            
            # Add common themes
            if "common_themes" in web_insights and web_insights["common_themes"]:
                web_insights_text += "\nCommon Themes:\n"
                for i, result in enumerate(web_insights["common_themes"][:3], 1):
                    web_insights_text += f"{i}. {result}\n"
            
            # Add content gaps
            if "content_gaps" in web_insights and web_insights["content_gaps"]:
                web_insights_text += "\nContent Gaps:\n"
                for i, result in enumerate(web_insights["content_gaps"][:3], 1):
                    web_insights_text += f"{i}. {result}\n"
        
        user_prompt = f"""Research competitor content for '{research_topic}'.{web_insights_text}
        
        Provide the following information:
        1. Common themes across competitor content
        2. Content gaps or underserved areas
        3. Typical content structures and formats
        4. Strengths and weaknesses of competitor content
        5. Opportunities for differentiation
        
        Format your analysis as follows:
        COMMON_THEMES: [List of common themes]
        CONTENT_GAPS: [List of content gaps]
        TYPICAL_STRUCTURES: [Description of typical structures]
        STRENGTHS_WEAKNESSES: [Analysis of strengths and weaknesses]
        DIFFERENTIATION_OPPORTUNITIES: [List of differentiation opportunities]
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
                    current_content = [line[28:].strip()]
                elif current_section:
                    current_content.append(line)
            
            # Add the last section
            if current_section:
                competitor_research[current_section] = "\n".join(current_content)
            
            # Add web search metadata if available
            if web_insights:
                competitor_research["web_search_used"] = True
                competitor_research["web_search_timestamp"] = web_insights.get("timestamp", datetime.now().isoformat())
            else:
                competitor_research["web_search_used"] = False
            
            return competitor_research
            
        except Exception as e:
            logger.error(f"Error in competitor research: {e}")
            return {
                "error": str(e),
                "web_search_used": bool(web_insights)
            } 
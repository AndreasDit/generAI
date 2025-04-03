#!/usr/bin/env python3
"""
Feedback Manager for GenerAI

This module implements a feedback loop mechanism to improve content quality
based on previous article performance metrics.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from loguru import logger


class FeedbackManager:
    """Manages feedback data and performance metrics for articles."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the feedback manager.
        
        Args:
            data_dir: Base directory for storing article data
        """
        self.data_dir = Path(data_dir)
        self.feedback_dir = self.data_dir / "feedback"
        self.feedback_dir.mkdir(exist_ok=True)
        
        # Create analytics file if it doesn't exist
        self.analytics_file = self.feedback_dir / "analytics.json"
        if not self.analytics_file.exists():
            self._initialize_analytics()
        
        logger.info(f"Feedback manager initialized with directory: {self.feedback_dir}")
    
    def _initialize_analytics(self) -> None:
        """
        Initialize the analytics file with default structure.
        """
        default_analytics = {
            "articles": [],
            "topic_performance": {},
            "audience_performance": {},
            "style_performance": {},
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.analytics_file, "w") as f:
            json.dump(default_analytics, f, indent=2)
    
    def record_article_metrics(self, project_id: str, metrics: Dict[str, Any]) -> None:
        """
        Record performance metrics for a published article.
        
        Args:
            project_id: ID of the article project
            metrics: Dictionary of performance metrics (views, reads, claps, etc.)
        """
        try:
            # Load project data
            project_dir = self.data_dir / "projects" / project_id
            if not project_dir.exists():
                logger.error(f"Project directory not found: {project_id}")
                return
            
            # Load article metadata
            metadata_file = project_dir / "metadata.json"
            if not metadata_file.exists():
                logger.error(f"Metadata file not found for project: {project_id}")
                return
            
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            
            # Create article performance record
            performance_record = {
                "project_id": project_id,
                "title": metadata.get("title", "Unknown"),
                "topic": metadata.get("topic", "Unknown"),
                "audience": metadata.get("audience", "General"),
                "style": metadata.get("style", "Informative"),
                "metrics": metrics,
                "recorded_at": datetime.now().isoformat()
            }
            
            # Save individual performance record
            performance_file = self.feedback_dir / f"{project_id}_performance.json"
            with open(performance_file, "w") as f:
                json.dump(performance_record, f, indent=2)
            
            # Update analytics
            self._update_analytics(performance_record)
            
            logger.info(f"Recorded performance metrics for project: {project_id}")
            
        except Exception as e:
            logger.error(f"Error recording article metrics: {e}")
    
    def _update_analytics(self, performance_record: Dict[str, Any]) -> None:
        """
        Update analytics with new performance data.
        
        Args:
            performance_record: Performance record for an article
        """
        try:
            with open(self.analytics_file, "r") as f:
                analytics = json.load(f)
            
            # Add article to list
            analytics["articles"].append({
                "project_id": performance_record["project_id"],
                "title": performance_record["title"],
                "metrics": performance_record["metrics"],
                "recorded_at": performance_record["recorded_at"]
            })
            
            # Update topic performance
            topic = performance_record["topic"]
            if topic not in analytics["topic_performance"]:
                analytics["topic_performance"][topic] = {
                    "count": 0,
                    "total_views": 0,
                    "total_reads": 0,
                    "total_claps": 0,
                    "engagement_rate": 0
                }
            
            analytics["topic_performance"][topic]["count"] += 1
            analytics["topic_performance"][topic]["total_views"] += performance_record["metrics"].get("views", 0)
            analytics["topic_performance"][topic]["total_reads"] += performance_record["metrics"].get("reads", 0)
            analytics["topic_performance"][topic]["total_claps"] += performance_record["metrics"].get("claps", 0)
            
            # Calculate engagement rate (reads/views)
            views = performance_record["metrics"].get("views", 0)
            reads = performance_record["metrics"].get("reads", 0)
            if views > 0:
                engagement = reads / views
                # Update with weighted average
                count = analytics["topic_performance"][topic]["count"]
                old_rate = analytics["topic_performance"][topic]["engagement_rate"]
                new_rate = ((old_rate * (count - 1)) + engagement) / count
                analytics["topic_performance"][topic]["engagement_rate"] = new_rate
            
            # Similar updates for audience and style performance
            self._update_category_performance(analytics, "audience_performance", 
                                             performance_record["audience"], performance_record["metrics"])
            
            self._update_category_performance(analytics, "style_performance", 
                                             performance_record["style"], performance_record["metrics"])
            
            # Update timestamp
            analytics["last_updated"] = datetime.now().isoformat()
            
            # Save updated analytics
            with open(self.analytics_file, "w") as f:
                json.dump(analytics, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error updating analytics: {e}")
    
    def _update_category_performance(self, analytics: Dict[str, Any], 
                                    category_key: str, category_value: str, 
                                    metrics: Dict[str, Any]) -> None:
        """
        Update performance metrics for a specific category (audience, style, etc.)
        
        Args:
            analytics: Analytics data dictionary
            category_key: Key in analytics for the category (e.g., "audience_performance")
            category_value: Value of the category (e.g., "Developers")
            metrics: Performance metrics
        """
        if category_value not in analytics[category_key]:
            analytics[category_key][category_value] = {
                "count": 0,
                "total_views": 0,
                "total_reads": 0,
                "total_claps": 0,
                "engagement_rate": 0
            }
        
        analytics[category_key][category_value]["count"] += 1
        analytics[category_key][category_value]["total_views"] += metrics.get("views", 0)
        analytics[category_key][category_value]["total_reads"] += metrics.get("reads", 0)
        analytics[category_key][category_value]["total_claps"] += metrics.get("claps", 0)
        
        # Calculate engagement rate
        views = metrics.get("views", 0)
        reads = metrics.get("reads", 0)
        if views > 0:
            engagement = reads / views
            # Update with weighted average
            count = analytics[category_key][category_value]["count"]
            old_rate = analytics[category_key][category_value]["engagement_rate"]
            new_rate = ((old_rate * (count - 1)) + engagement) / count
            analytics[category_key][category_value]["engagement_rate"] = new_rate
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """
        Get insights from performance data to guide content creation.
        
        Returns:
            Dictionary of insights for content improvement
        """
        try:
            with open(self.analytics_file, "r") as f:
                analytics = json.load(f)
            
            insights = {
                "top_topics": self._get_top_performers(analytics["topic_performance"], "engagement_rate", 5),
                "top_audiences": self._get_top_performers(analytics["audience_performance"], "engagement_rate", 3),
                "top_styles": self._get_top_performers(analytics["style_performance"], "engagement_rate", 3),
                "general_recommendations": []
            }
            
            # Generate general recommendations based on data
            if insights["top_topics"]:
                insights["general_recommendations"].append(
                    f"Focus on these high-performing topics: {', '.join(insights['top_topics'])}"
                )
            
            if insights["top_audiences"]:
                insights["general_recommendations"].append(
                    f"Target these engaged audiences: {', '.join(insights['top_audiences'])}"
                )
            
            if insights["top_styles"]:
                insights["general_recommendations"].append(
                    f"Use these effective content styles: {', '.join(insights['top_styles'])}"
                )
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting performance insights: {e}")
            return {"error": str(e)}
    
    def _get_top_performers(self, performance_data: Dict[str, Dict[str, Any]], 
                           metric: str, limit: int) -> List[str]:
        """
        Get top performing categories based on a specific metric.
        
        Args:
            performance_data: Dictionary of performance data by category
            metric: Metric to sort by (e.g., "engagement_rate")
            limit: Maximum number of top performers to return
            
        Returns:
            List of top performing category names
        """
        # Filter out categories with too few articles (need at least 2 for meaningful data)
        filtered_data = {k: v for k, v in performance_data.items() if v["count"] >= 2}
        
        # Sort by the specified metric
        sorted_items = sorted(filtered_data.items(), 
                             key=lambda x: x[1][metric], 
                             reverse=True)
        
        # Return top N category names
        return [item[0] for item in sorted_items[:limit]]
    
    def apply_insights_to_idea_evaluation(self, ideas: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """
        Apply performance insights to score and rank article ideas.
        
        Args:
            ideas: List of article idea dictionaries
            
        Returns:
            List of dictionaries with idea scores
        """
        try:
            # Get current insights
            insights = self.get_performance_insights()
            
            # If we don't have enough data, return ideas with neutral scores
            if not insights.get("top_topics") and not insights.get("top_audiences"):
                return [{
                    "id": idea.get("id", ""),
                    "title": idea.get("title", ""),
                    "score": 0.5,  # Neutral score
                    "reasons": ["Insufficient performance data for scoring"]
                } for idea in ideas]
            
            scored_ideas = []
            
            for idea in ideas:
                score = 0.5  # Start with neutral score
                reasons = []
                
                # Score based on topic match
                topic = idea.get("title", "").lower()
                topic_matched = False
                for top_topic in insights.get("top_topics", []):
                    if top_topic.lower() in topic:
                        score += 0.15
                        reasons.append(f"Topic matches high-performing area: {top_topic}")
                        topic_matched = True
                        break
                
                if not topic_matched and insights.get("top_topics"):
                    score -= 0.05
                    reasons.append("Topic doesn't match any high-performing areas")
                
                # Score based on audience match
                audience = idea.get("audience", "").lower()
                audience_matched = False
                for top_audience in insights.get("top_audiences", []):
                    if top_audience.lower() in audience:
                        score += 0.15
                        reasons.append(f"Audience matches high-engagement group: {top_audience}")
                        audience_matched = True
                        break
                
                if not audience_matched and insights.get("top_audiences"):
                    score -= 0.05
                    reasons.append("Audience doesn't match any high-engagement groups")
                
                # Ensure score is within bounds
                score = max(0.1, min(1.0, score))
                
                scored_ideas.append({
                    "id": idea.get("id", ""),
                    "title": idea.get("title", ""),
                    "score": score,
                    "reasons": reasons
                })
            
            # Sort by score (highest first)
            return sorted(scored_ideas, key=lambda x: x["score"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error applying insights to idea evaluation: {e}")
            # Return ideas with neutral scores on error
            return [{
                "id": idea.get("id", ""),
                "title": idea.get("title", ""),
                "score": 0.5,
                "reasons": [f"Error in scoring: {str(e)}"]
            } for idea in ideas]
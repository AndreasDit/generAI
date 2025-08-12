"""Tweet Generator Module for GenerAI.

This module handles the generation and scheduling of tweets for articles.
It can be used independently or as part of the article pipeline.
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import uuid
from typing import Dict, Any, Optional

from loguru import logger

class TweetGenerator:
    def __init__(self, llm_client, data_dir: Path):
        """Initialize the TweetGenerator.
        
        Args:
            llm_client: LLM client for generating refined tweets
            data_dir: Base directory for data files
        """
        self.llm_client = llm_client
        self.data_dir = data_dir

    def generate_tweets_for_idea(self, idea: Dict[str, Any]) -> None:
        """Generate tweets for the given idea based on the content strategy.
        
        Args:
            idea: The idea to generate tweets for
        """
        logger.info("Generating tweets for idea")
        
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
            
            Rewrite the tweets to be engaging, informative, and relevant to the article topic.
            Make each tweet engaging, concise (under 280 characters), and include 1-2 relevant emojis.
            Format your response as a JSON object with three keys: 'morning', 'afternoon', and 'evening',
            each containing the refined tweet text.
            
            Return ONLY the JSON object, nothing else. Do not mark it as a JSON object.
            """
            
            try:
                response = self.llm_client.chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=1,
                    max_tokens=500,
                    use_text_generation_model=True
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
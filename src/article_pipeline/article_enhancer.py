import json
from pathlib import Path
from typing import Dict, Any

from loguru import logger

from src.llm_client import LLMClient


class ArticleEnhancer:
    def __init__(self, llm_client: LLMClient, projects_dir: Path):
        self.llm_client = llm_client
        self.projects_dir = projects_dir

    def add_value_to_article(self, project_id: str) -> bool:
        logger.info(f"Adding value to article for project: {project_id}")
        project_dir = self.projects_dir / project_id
        article_path = project_dir / "article.md"
        if not article_path.exists():
            logger.error(f"article.md not found for project {project_id}")
            return False

        with open(article_path, "r") as f:
            article_content = f.read()

        # Prompt 1: Review article
        prompt1 = f"""Review this article on a scale from 1 to 10. Consider that it will be published on the popular site medium.com. I want my articles to contain actionable information so that I can give readers something to copy and do immediately.

        But this is only one of the relevant aspects. Act as a professional Medium editor and perform the review and then list improvements that can be added to the article in order to bring it to a 10 out of 10. No rewriting, but addition.

        Article:
        {article_content}"""
        response1 = self.llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt1}]
        )

        # Prompt 2: Choose most important suggestions
        prompt2 = f"""Those are a lot of suggestions. Such an article can get pretty lengthy - maybe even too long for medium. From your 10 suggestion choose the most important ones that keep the article from exploding in length. Keep in mind that visual flair will be added to the article later on.

        Suggestions:
        {response1}"""
        response2 = self.llm_client.chat_completion(
            messages=[
                {"role": "user", "content": prompt1},
                {"role": "assistant", "content": response1},
                {"role": "user", "content": prompt2},
            ]
        )

        # Prompt 3: Implement additions
        prompt3 = f"""Great, now implement these three additions into the article. Focus on the content, not the formatting. The article will be formatted later on.
        Return only the article content, no additional text. Formatted as markdown.

        Article:
        {article_content}

        Additions:
        {response2}"""
        enhanced_article_content = self.llm_client.chat_completion(
            messages=[
                {"role": "user", "content": prompt1},
                {"role": "assistant", "content": response1},
                {"role": "user", "content": prompt2},
                {"role": "assistant", "content": response2},
                {"role": "user", "content": prompt3},
            ]
        )

        enhanced_article_path = project_dir / "enhanced_article.md"
        with open(enhanced_article_path, "w") as f:
            f.write(enhanced_article_content)

        logger.info(f"Enhanced article saved to {enhanced_article_path}")
        return True

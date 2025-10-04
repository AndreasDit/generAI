"""Content generator for article generation."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from loguru import logger

from src.llm_client import LLMClient
from src.web_search import TavilySearchManager

class ContentGenerator:
    """Generates content for articles."""
    
    def __init__(self, openai_client: LLMClient, projects_dir: Path, web_search: TavilySearchManager):
        """Initialize the content generator.
        
        Args:
            openai_client: LLM client for API interactions
            projects_dir: Directory to store project data
        """
        self.llm_client = openai_client
        self.projects_dir = projects_dir
        self.web_search = web_search
    
    def generate_image_suggestions(self, project_id: str) -> Dict[str, Any]:
        """Generate image suggestions for a refined article.
        
        Args:
            project_id: ID of the project to generate image suggestions for
            
        Returns:
            Dictionary containing the image suggestions
        """
        logger.info(f"Generating image suggestions for project: {project_id}")
        
        # Get project data
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            logger.error(f"Project not found: {project_id}")
            return {}
        
        refined_article_file = project_dir / "refined_article.md"
        if not refined_article_file.exists():
            logger.error(f"Refined article not found for project: {project_id}")
            return {}
        
        with open(refined_article_file) as f:
            article_content = f.read()
            
        # Load idea data to provide more context to the LLM
        idea_file = project_dir / "idea.json"
        
        idea_data = {}
        
        if idea_file.exists():
            try:
                with open(idea_file) as f:
                    idea_data = json.load(f)
                logger.info(f"Loaded idea data for project: {project_id}")
            except Exception as e:
                logger.error(f"Error loading idea data: {e}")
        else:
            logger.warning(f"Idea file not found for project: {project_id}")
                    
        # Generate image suggestions using LLM
        system_prompt = (
            "You are an expert visual content strategist who suggests optimal image placements for articles. "
            "Your task is to analyze an article and suggest strategic locations where images would enhance "
            "the reader's understanding, engagement, and overall experience."
        )
        
        # Prepare idea context for the prompt
        idea_context = ""
        if idea_data:
            idea_context = f"""ARTICLE IDEA:
        Title: {idea_data.get('title', 'No title')}
        Description: {idea_data.get('description', 'No description')}
        Target Audience: {idea_data.get('target_audience', 'General audience')}
        Key Points: {', '.join(idea_data.get('key_points', []))}
        Value Proposition: {idea_data.get('value_proposition', 'No value proposition')}
        """
        
        user_prompt = f"""Analyze the following article and suggest 3-7 strategic locations where images would enhance the content:

        This is the article idea:
        {idea_context}        
                
        This is the article content:
        {article_content}
                
        For each suggested image:
        1. Identify the exact location in the article where the image should be placed (after which paragraph or section)
        2. Describe what the image should contain in detail
        3. Explain why this image would enhance the reader's experience at this location
        4. Suggest an appropriate caption for the image
        
        Format your response as a JSON array of image suggestions, where each suggestion is an object with these properties:
        - location: Description of where in the article the image should be placed
        - description: Detailed description of what the image should contain
        - rationale: Explanation of why this image enhances the content
        - caption: Suggested caption for the image
        - prompt: Prompt like a senior prompt engineer that creates this graphic
        
        Return ONLY the JSON array, nothing else.
        """
        
        try:
            response = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1,
                max_tokens=1500
            )
                        
            # Parse the image suggestions
            try:
                # Write the response to a file
                with open(project_dir / "image_suggestions.json", "w") as f:
                    f.write(response)

                # Update project metadata
                metadata_file = project_dir / "metadata.json"
                with open(metadata_file) as f:
                    metadata = json.load(f)
                
                metadata["has_image_suggestions"] = True
                metadata["updated_at"] = datetime.now().isoformat()
                
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Generated image suggestions for project: {project_id}")
                return True
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing image suggestions: {e}")
                return {"error": "Failed to parse image suggestions"}
                
        except Exception as e:
            logger.error(f"Error generating image suggestions: {e}")
            return {"error": str(e)}
    
    def generate_outline(self, project_id: str) -> Dict[str, Any]:
        """Generate an outline for a project.
        
        Args:
            project_id: ID of the project to generate outline for
            
        Returns:
            Dictionary containing the generated outline
        """
        logger.info(f"Generating outline for project: {project_id}")
        
        # Get project data
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            logger.error(f"Project not found: {project_id}")
            return {}
        
        idea_file = project_dir / "idea.json"
        if not idea_file.exists():
            logger.error(f"Project idea not found: {project_id}")
            return {}
        
        with open(idea_file) as f:
            idea = json.load(f)
        
        # Check if web search results are available
        search_results_file = project_dir / "search_results.json"
        search_results_data = ""
        if search_results_file.exists():
            try:
                with open(search_results_file) as f:
                    search_results = json.load(f)
                    # drop the raw_content from the search results
                    for result in search_results['results']:
                        result.pop('raw_content', None)
                    search_results_data = f"\n\nWeb Search Results:\n{json.dumps(search_results.get('results', []), indent=2)}"
                    
                    logger.info(f"Incorporating web search results into outline generation for project: {project_id}")
            except Exception as e:
                logger.error(f"Error loading search results: {e}")
        
        # Generate outline using LLM
        system_prompt = (
            "You are an expert content strategist who creates detailed article outlines. "
            "Your outlines should be well-structured, comprehensive, and engaging."
        )
        
        user_prompt = f"""Create a detailed outline for an article about '{idea.get('title', '')}'.

        Here is the idea data to inform your outline:
        {json.dumps(idea, indent=3)}
        Here is some research:
        {search_results_data}
        
        The outline should include:
        1. A compelling introduction
        2. Key points to cover in each section
        3. A strong conclusion
        
        Format your outline as follows:
        INTRODUCTION: [Brief description of the introduction]
        MAIN_SECTIONS:
        - Section 1 Title:
          - Subsection 1.1: [Description]
          - Subsection 1.2: [Description]
        - Section 2 Title:
          - Subsection 2.1: [Description]
          - Subsection 2.2: [Description]
        CONCLUSION: [Brief description of the conclusion]
        
        No more than 3 main sections. No more than 3 subsections per section.
        Each description should contain the following: detailed important points from the research, necessary hard facts, and necessary hard data.
        Do not use any markdown or other formatting.
        """
        
        try:
            # Log model and token usage info
            token_count = len(user_prompt + system_prompt)
            logger.info(f"Input tokens: {token_count}")
            response = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1,
                max_tokens=1000,
                use_text_generation_model=True
            )
            
            # Parse the outline
            outline = {}
            current_section = None
            current_subsections = []
            
            for line in response.split("\n"):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("INTRODUCTION:"):
                    current_section = "introduction"
                    outline[current_section] = line[14:].strip()
                elif line.startswith("CONCLUSION:"):
                    if current_section:
                        outline[current_section] = current_subsections
                    current_section = "conclusion"
                    outline[current_section] = line[11:].strip()
                elif line.startswith("- "):
                    if current_section and current_section not in ["introduction", "conclusion"]:
                        if "section" in line.lower() or "subsection" in line.lower():
                            title = line[2:].split(":")[0].strip()
                            description = line[2:].split(":")[1].strip()
                            current_subsections.append({
                                "title": title,
                                "description": description
                            })
                        else:
                            current_subsections[-1]["description"] += "\n- " + line[2:].strip()
                elif line.endswith(":"):
                    if current_section and current_section not in ["introduction", "conclusion"]:
                        outline[current_section] = current_subsections
                    current_section = line[:-1].lower().replace(" ", "_")
                    current_subsections = []
            
            # Add the last section
            if current_section and current_section not in ["introduction", "conclusion"]:
                outline[current_section] = current_subsections
            
            # Save the outline
            outline_file = project_dir / "outline.json"
            with open(outline_file, "w") as f:
                json.dump(outline, f, indent=2)
            
            # Update project metadata
            metadata_file = project_dir / "metadata.json"
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            metadata["status"] = "outline_generated"
            metadata["updated_at"] = idea.get("created_at", "")
            
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Generated outline for project: {project_id}")
            return outline
            
        except Exception as e:
            logger.error(f"Error generating outline: {e}")
            return {}
    
    def generate_paragraphs(self, project_id: str) -> List[Dict[str, Any]]:
        """Generate paragraphs for a project.
        
        Args:
            project_id: ID of the project to generate paragraphs for
            
        Returns:
            List of generated paragraphs
        """
        
        # Get project data
        project_dir = self.projects_dir / project_id
        if not project_dir.exists():
            logger.error(f"Project not found: {project_id}")
            return []
        
        outline_file = project_dir / "outline.json"
        if not outline_file.exists():
            logger.error(f"Project outline not found: {project_id}")
            return []
        
        with open(outline_file) as f:
            outline = json.load(f)
        
        # Generate paragraphs using LLM
        system_prompt = (
            "You are an expert content writer who creates engaging, informative paragraphs. "
            "Your writing should be clear, concise, and well-structured."
        )
        
        paragraphs = []
        
        # Generate introduction
        if "introduction" in outline:
            user_prompt = f"""
            Assume the role of a seasoned writer specializing in captivating introductions for Medium articles.

            Write an engaging introduction for an article about '{outline.get('introduction', '')}'.
            The introduction should hook the reader immediately, setting the tone and context of the article while intriguing them to continue reading.
            Start with a compelling hook—this could be a provocative question, a surprising fact, a vivid anecdote, or a powerful quote.
            Briefly outline the main points that will be covered in the article, establishing your credibility on the subject.
            Make sure the introduction aligns with the overall tone and style of the article, whether it's formal, conversational, or humorous.
            Include a transition at the end of the introduction that seamlessly leads into the main body of the article, ensuring a smooth reader experience.

            The introduction should:
            1. Hook the reader's attention
            2. Provide context for the topic
            3. Outline what the article will cover
            4. Be 2-3 paragraphs long
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
                
                paragraphs.append({
                    "type": "introduction",
                    "content": response.strip()
                })
                
            except Exception as e:
                logger.error(f"Error generating introduction: {e}")
        
        # Generate main sections
        for section_key, section_data in outline.items():
            if section_key in ["introduction", "conclusion"]:
                continue
            
            if isinstance(section_data, list):
                for subsection in section_data:
                    title = subsection.get("title", "")
                    description = subsection.get("description", "")
                    
                    # Transform the research topic into an effective search term
                    topic = title + " " + description
                    search_term = self.llm_client.transform_search_term(topic)
                    search_results = self.web_search.search(search_term, max_results=2)
                    extracted_contents = self.web_search.extract_content_from_search_results(search_results)
                    
                    user_prompt = f"""Write a detailed paragraph for the section '{title}' with the following description:
                    {description}
                    
                    Addidional information and research:
                    {extracted_contents}
                    
                    The paragraph should:
                    1. Be informative and engaging
                    2. Include relevant details and examples. Include necessary hard facts and hard data. Be precise, concise and above all else provide all necesasry information.
                    3. Flow naturally from previous content
                    4. Be 2-4 paragraphs long
                    
                    After the parapraph list some relevant hard data as bullet points.
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
                        
                        paragraphs.append({
                            "type": "section",
                            "title": subsection.get("title", ""),
                            "content": response.strip()
                        })
                        
                    except Exception as e:
                        logger.error(f"Error generating section paragraph: {e}")
        
        # Generate conclusion
        if "conclusion" in outline:
            user_prompt = f"""Write a strong conclusion for an article about '{outline.get('conclusion', '')}'.

            The conclusion should:
            1. Summarize key points
            2. Provide final insights
            3. Leave a lasting impression
            4. Be 2-3 paragraphs long
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
                
                paragraphs.append({
                    "type": "conclusion",
                    "content": response.strip()
                })
                
            except Exception as e:
                logger.error(f"Error generating conclusion: {e}")
        
        # Save the paragraphs
        paragraphs_file = project_dir / "paragraphs.json"
        with open(paragraphs_file, "w") as f:
            json.dump(paragraphs, f, indent=2)
        
        # Update project metadata
        metadata_file = project_dir / "metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        metadata["status"] = "paragraphs_generated"
        metadata["updated_at"] = outline.get("created_at", "")
        
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Generated paragraphs for project: {project_id}")
        return paragraphs

    def generate_article_from_idea(self, project_id: str, idea: Dict[str, Any]) -> Optional[str]:
        """Generate a complete article from an idea in a single step.

        Args:
            project_id: ID of the project.
            idea: Dictionary containing the article idea.

        Returns:
            The generated article as a string, or None if generation fails.
        """
        logger.info(f"Generating article from idea for project: {project_id}")
        
        system_prompt = (
            "You are an experienced mindfulness and mental health writer who publishes on Medium. Your specialty is creating practical, evidence-based content that helps busy professionals improve their mental well-being without overwhelming them."
        )

        user_prompt = f"""
        TARGET AUDIENCE: Adults aged 25-45 seeking actionable mental health and mindfulness strategies they can implement immediately in their daily lives.

        TASK: Write a 500-600 word Medium article on the following topic:
        
        Title: {idea.get('title', '')}
        Description: {idea.get('description', '')}
        Key Points: {', '.join(idea.get('key_points', []))}

        ARTICLE STRUCTURE:
        1. A catchy subtitle that complements the title and animates people to click on the article.
        2. Hook (50-75 words): Start with a relatable problem, question, or surprising insight that immediately resonates with the reader's experience
        3. Main Content (350-400 words): Deliver 3-4 practical strategies or insights organized with clear subheadings (use ###). Each point should include a specific, actionable example
        4. Conclusion (75-100 words): Summarize the key takeaway and include one simple action the reader can take today

        WRITING STYLE REQUIREMENTS:
        - Use short paragraphs (2-3 sentences maximum)
        - Write in a warm, conversational tone as if advising a friend
        - Use "you" and "your" to address readers directly
        - Include at least one concrete, specific example or scenario
        - Avoid clinical jargon, generic phrases like "in today's world," and corporate language
        - Focus on actionable advice, not abstract concepts
        - Use active voice throughout

        WHAT TO AVOID:
        - Fluff and filler words
        - Generic self-help clichés
        - Overly spiritual or medical language
        - Lists without context or explanation
        - Promises of quick fixes or miracle solutions

        FORMAT: Write in plain text with clear markdown headers (###) for subheadings. Make the content scannable and engaging for Medium readers.
        """

        try:
            response = self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1,
                max_tokens=3000,
                # use_text_generation_model=True,
                model_name='gpt-5'
            )

            return response.strip()

        except Exception as e:
            logger.error(f"Error generating article from idea: {e}")
            return None
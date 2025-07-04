# GenerAI - AI-Powered Article Generation Pipeline

A modular article generation pipeline that supports multiple LLM providers (OpenAI by default, with Claude as an alternative) and includes features for trend analysis, content generation, SEO optimization, and more.

## Features

- Multiple LLM providers (OpenAI by default, Claude as alternative)
- Modular pipeline architecture
- Trend analysis and competitor research
- Content generation and refinement
- SEO optimization
- Feedback analysis
- Web search integration (Brave and Tavily)
- Medium publishing integration
- Image Suggestions: Provides creative visual prompts for enhancing articles with relevant imagery, including infographics and illustrations.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/generai.git
cd generai
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure your API keys:
   - Copy `.env.example` to `.env`
   - Add your API keys:
     - OpenAI API key (required, default provider)
     - Anthropic API key (optional, for Claude)
     - Brave API key (for web search)
     - Tavily API key (optional, alternative web search)
     - Medium API key (optional, for publishing)

## Configuration

The following API keys are required:

### LLM Configuration
- `OPENAI_API_KEY`: Your OpenAI API key (required, default provider)
- `ANTHROPIC_API_KEY`: Your Anthropic API key (optional, for Claude)

### Web Search Configuration
- `BRAVE_API_KEY`: Your Brave Search API key
- `TAVILY_API_KEY`: Your Tavily API key (optional)

### Publishing Configuration (Optional)
- `MEDIUM_API_KEY`: Your Medium API key
- `MEDIUM_AUTHOR_ID`: Your Medium author ID
- `MEDIUM_PUBLICATION_ID`: Your Medium publication ID

## Usage

The application offers two main modes: `simple` for quick article generation and `modular` for a step-by-step pipeline.

### Simple Mode

For quickly generating a complete article on a given topic.

**Example:**
```bash
python generai.py simple --topic "The Future of Artificial Intelligence" --publish
```

### Modular Mode

For a more controlled, step-by-step process from idea to publication.

#### Full Pipeline Execution

To run the entire pipeline automatically:

- **Run the complete pipeline for a new topic:** This will perform trend analysis, generate ideas, and process the best one.
  ```bash
  python generai.py modular --run-full-pipeline --research-topic "SaaS business ideas"
  ```

- **Process the next article in the queue:** This will take the next available idea and run it through all the stages.
  ```bash
  python generai.py modular --process-next-article
  ```

#### Individual Module Execution

You can also execute each module of the pipeline individually. This is useful for debugging or for manual control over the process.

- **1. Generate Ideas:** Research a topic and generate article ideas.
  ```bash
  python generai.py modular --generate-ideas --research-topic "Python programming" --num-ideas 10
  ```

- **2. Evaluate Ideas:** Evaluate the generated ideas and select the best one to be processed.
  ```bash
  python generai.py modular --evaluate-ideas
  ```

- **3. Create Project:** Create a project for the selected article idea.
  ```bash
  python generai.py modular --create-project
  ```

- **4. Generate Outline:** Generate an article outline for a specific project.
  ```bash
  python generai.py modular --generate-outline --project-id <project_id>
  ```

- **5. Generate Paragraphs:** Write the content for each section of the outline.
  ```bash
  python generai.py modular --generate-paragraphs --project-id <project_id>
  ```

- **6. Assemble Article:** Combine the generated paragraphs into a full article draft.
  ```bash
  python generai.py modular --assemble-article --project-id <project_id>
  ```

- **7. Refine Article:** Review and refine the assembled article for clarity and flow.
  ```bash
  python generai.py modular --refine-article --project-id <project_id>
  ```

- **8. SEO Optimization:** Optimize the article for search engines.
  ```bash
  python generai.py modular --optimize-seo --project-id <project_id>
  ```

- **9. Suggest Images:** Generate suggestions for relevant images and illustrations.
  ```bash
  python generai.py modular --suggest-images --project-id <project_id>
  ```

- **10. Generate Social Media Content:**
  - **Hashtags:** Generate relevant hashtags for social media promotion.
    ```bash
    python generai.py modular --suggest-hashtags --project-id <project_id>
    ```
  - **Tweets:** Generate tweets to promote the article.
    ```bash
    python generai.py modular --generate-tweets --project-id <project_id>
    ```

- **11. Publish to Medium:** Publish the final article to Medium.
  ```bash
  python generai.py modular --publish-to-medium --project-id <project_id> --tags "ai,tech,writing"
  ```

## Project Structure

```
generai/
├── generai.py
├── src/
│   ├── __init__.py
│   ├── article_pipeline/
│   │   ├── __init__.py
│   │   ├── article_assembler.py
│   │   ├── content_generator.py
│   │   ├── feedback_manager.py
│   │   ├── idea_generator.py
│   │   ├── project_manager.py
│   │   ├── seo_optimizer.py
│   │   ├── trend_analyzer.py
│   │   └── utils.py
│   ├── cache_manager.py
│   ├── config.py
│   ├── config_manager.py
│   ├── feedback_manager.py
│   ├── llm_client.py
│   ├── medium_publisher.py
│   ├── openai_client.py
│   ├── utils.py
│   └── web_search.py
├── tests/
│   ├── README.md
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_article_pipeline.py
│   ├── test_brave_search.py
│   ├── test_cache_manager.py
│   ├── test_feedback_manager.py
│   ├── test_generai.py
│   ├── test_medium_publisher.py
│   ├── test_openai_client.py
│   ├── test_utils.py
│   └── test_web_search.py
├── cache/
├── data/
├── logs/
├── .env.example
├── pytest.ini
├── requirements-dev.txt
├── requirements.txt
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
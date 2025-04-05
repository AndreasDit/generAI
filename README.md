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

### Running the Pipeline

The pipeline can be run in modular mode, allowing you to execute specific steps:

```bash
python generai.py modular <command> [options]
```

Available commands:

1. Generate article ideas:
```bash
python generai.py modular --generate-ideas
```

2. Evaluate ideas:
```bash
python generai.py modular --evaluate-ideas
```

3. Create project:
```bash
python generai.py modular --create-project --idea "Your article idea"
```

4. Generate outline:
```bash
python generai.py modular --generate-outline --project-id <project_id>
```

5. Generate paragraphs:
```bash
python generai.py modular --generate-paragraphs --project-id <project_id>
```

6. Assemble article:
```bash
python generai.py modular --assemble-article --project-id <project_id>
```

7. Refine article:
```bash
python generai.py modular --refine-article --project-id <project_id>
```

8. Optimize for SEO:
```bash
python generai.py modular --optimize-seo --project-id <project_id>
```

9. Analyze feedback:
```bash
python generai.py modular --analyze-feedback --project-id <project_id>
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
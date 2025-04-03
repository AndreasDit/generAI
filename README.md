# GenerAI - AI Article Generator and Medium Publisher

## Overview

GenerAI is a powerful tool for generating high-quality articles using OpenAI's API and publishing them to Medium. The application offers two modes of operation:

1. **Simple Mode**: Quick generation of articles on a given topic
2. **Modular Pipeline Mode**: Advanced multi-step process for comprehensive article creation

## Features

- AI-powered article generation with customizable tone and length
- Modular pipeline approach for research-based content creation
- Trend analysis and competitor research
- Idea generation and evaluation
- Structured outline and paragraph generation
- SEO optimization
- Direct publishing to Medium
- Performance metrics tracking and feedback loop

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/generAI.git
cd generAI

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Usage

The application has a single entry point (`generai.py`) with two operation modes:

### Simple Mode

Quickly generate an article on a specific topic:

```bash
python generai.py simple --topic "Artificial Intelligence Trends" --tone informative --length medium --output article.md
```

Publish to Medium:

```bash
python generai.py simple --topic "Artificial Intelligence Trends" --publish --tags "AI,Technology" --status draft
```

### Modular Pipeline Mode

Run the complete article generation pipeline:

```bash
python generai.py modular --run-full-pipeline --research-topic "Artificial Intelligence Trends" --output article.md
```

Or run individual steps:

```bash
# Generate ideas
python generai.py modular --generate-ideas --research-topic "Artificial Intelligence Trends" --num-ideas 5

# Evaluate ideas and select the best one
python generai.py modular --evaluate-ideas

# Create a project for the selected idea
python generai.py modular --create-project

# Generate an outline for the project
python generai.py modular --generate-outline PROJECT_ID

# Generate paragraphs based on the outline
python generai.py modular --generate-paragraphs PROJECT_ID

# Assemble the article
python generai.py modular --assemble-article PROJECT_ID

# Refine the article
python generai.py modular --refine-article PROJECT_ID

# Optimize for SEO
python generai.py modular --optimize-seo PROJECT_ID

# Publish to Medium
python generai.py modular --publish PROJECT_ID --tags "AI,Technology" --status draft
```

## Configuration

The application uses environment variables for configuration. Copy the `.env.example` file to `.env` and set the following variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: The OpenAI model to use (default: gpt-4)
- `MEDIUM_INTEGRATION_TOKEN`: Your Medium integration token
- `MEDIUM_AUTHOR_ID`: Your Medium author ID (optional)
- `TAVILY_API_KEY`: Your Tavily API key for web search functionality

## Directory Structure

```
├── generai.py          # Main entry point
├── requirements.txt    # Dependencies
├── .env.example        # Example environment variables
├── data/               # Data directory
│   ├── article_queue/  # Queue of articles to be written
│   ├── ideas/          # Generated article ideas
│   └── projects/       # Article projects
└── src/                # Source code
    ├── article_pipeline.py   # Modular pipeline implementation
    ├── cache_manager.py      # API response caching
    ├── config_manager.py     # Configuration management
    ├── feedback_manager.py   # Performance feedback loop
    ├── medium_publisher.py   # Medium publishing functionality
    ├── openai_client.py      # OpenAI API client
    ├── utils.py              # Utility functions
    └── web_search.py         # Web search functionality
```

## License

MIT
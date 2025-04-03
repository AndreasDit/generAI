# GenerAI - AI Article Generator and Medium Publisher

GenerAI is a Python tool that uses OpenAI's API to generate high-quality articles and publish them directly to Medium. This tool streamlines the content creation process by leveraging AI to draft articles on any topic and providing an easy way to publish them to your Medium account.

## Features

- Generate well-structured articles on any topic using OpenAI's powerful language models
- Customize article tone, length, and structure
- Publish articles directly to Medium with proper formatting
- Save generated articles to local files
- Configure via environment variables or JSON configuration file
- Detailed logging for troubleshooting

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/generAI.git
   cd generAI
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your configuration:
   - Copy `.env.example` to `.env` and fill in your API keys
   - OR copy `config.json.example` to `config.json` and fill in your API keys

## Configuration

### Configuration Method

This application uses environment variables for configuration. No config.json file is needed.

#### Required API Keys

1. **OpenAI API Key**: Get this from [OpenAI's platform](https://platform.openai.com/)
2. **Medium Integration Token**: Get this from [Medium's developer settings](https://medium.com/me/settings)

#### Setting Up Environment Variables

Copy the `.env.example` file to `.env` and fill in your details:

```
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4
MEDIUM_INTEGRATION_TOKEN=your-medium-integration-token-here
MEDIUM_AUTHOR_ID=your-medium-author-id-here
DEFAULT_TAGS=AI,Technology,Writing,Content Creation
DEFAULT_STATUS=draft
```

## Usage

### Basic Usage

Generate an article and display it in the console:

```
python article_generator.py --topic "The Future of Artificial Intelligence"
```

### Customizing Article Generation

```
python article_generator.py \
    --topic "The Future of Artificial Intelligence" \
    --tone professional \
    --length long \
    --outline "Introduction,Current State of AI,Future Predictions,Ethical Considerations,Conclusion"
```

### Saving to File

```
python article_generator.py \
    --topic "The Future of Artificial Intelligence" \
    --output article.md
```

### Publishing to Medium

```
python article_generator.py \
    --topic "The Future of Artificial Intelligence" \
    --publish \
    --tags "AI,Future,Technology,Ethics" \
    --status draft
```

## Command Line Options

### Article Generation Options

- `--topic`: Topic for the article (required if not provided interactively)
- `--tone`: Tone of the article (informative, casual, professional, technical, conversational)
- `--length`: Length of the article (short, medium, long)
- `--outline`: Comma-separated list of sections to include

### Medium Publishing Options

- `--publish`: Publish to Medium after generation
- `--tags`: Comma-separated list of tags for Medium
- `--status`: Publication status on Medium (draft, public, unlisted)
- `--canonical-url`: Original URL if this is a cross-post

### Output Options

- `--output`: Save article to file

## Logs

Logs are stored in the `logs` directory with timestamps for easy troubleshooting.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
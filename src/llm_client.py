"""LLM client interface and implementations."""

import abc
from datetime import datetime
from typing import Dict, Any, List, Optional

from loguru import logger
import anthropic
import openai

from src.config import LLM_CONFIG


class LLMClient(abc.ABC):
    """Abstract base class for LLM clients."""
    
    @abc.abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Generate a chat completion.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        pass
    
    def log_token_usage(self, model: str, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        """Log token usage for an LLM API call.
        
        Args:
            model: The model name used for the API call
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            total_tokens: Total number of tokens used
        """
        logger.info(f"Token usage - Model: {model}, Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Total tokens: {total_tokens}")

    def transform_search_term(self, research_topic: str) -> str:
        """Transform a research topic into an effective search term.
        
        Args:
            research_topic: The original research topic
            
        Returns:
            A transformed search term optimized for web search
        """
        logger.info(f"Transforming search term for topic: {research_topic}")
        
        system_prompt = (
            "You are an expert web researcher who creates effective search terms. "
            "Your task is to transform a general topic into a specific, targeted search term "
            "that will yield relevant and high-quality search results."
        )
        
        user_prompt = f"""Transform the following research topic into an effective search term:

        TOPIC: {research_topic}
        
        Guidelines:
        1. Make the search term more specific and targeted
        2. Include relevant keywords that will help find high-quality content
        3. Keep the search term concise (5-10 words maximum)
        4. Focus on the most important aspects of the topic
        5. Avoid overly broad or vague terms
        
        Provide ONLY the transformed search term without any explanation or additional text.
        """
        
        try:
            response = self.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )
            
            # Clean up the response to get just the search term
            search_term = response.strip()
            
            # Add the current year to focus on recent results
            current_year = datetime.now().year
            search_term = f"{search_term} Focus on results from the year {current_year}"
            
            logger.info(f"Transformed search term with year: {search_term}")
            return search_term
            
        except Exception as e:
            logger.error(f"Error transforming search term: {e}")
            # Return the original topic if transformation fails
            return research_topic


class OpenAIClient(LLMClient):
    """OpenAI API client implementation."""
    
    def __init__(self):
        """Initialize the OpenAI client."""
        config = LLM_CONFIG["openai"]
        self.client = openai.OpenAI(api_key=config["api_key"])
        self.model = config["model"]
        self.text_generation_model = config.get("text_generation_model", self.model)
        self.max_tokens = config["max_tokens"]
        self.temperature = config["temperature"]
    
    def chat_completion(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, max_tokens: Optional[int] = None, use_text_generation_model: bool = False, model_name: str = None) -> str:
        """Generate a chat completion.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 1.0), defaults to instance value
            max_tokens: Maximum number of tokens to generate, defaults to instance value
            use_text_generation_model: Whether to use the text generation model instead of the default model
            model_name: Name of the model to use (overrides instance value if provided)
            
        Returns:
            Generated text response
        """
        logger.info("Prforming query to OpenAI")
        # Use instance values if not specified
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Choose the appropriate model
        if not model_name:
            model = self.text_generation_model if use_text_generation_model else self.model
        else:
            model = model_name

        logger.info(f"Using model {model}")
        try:
            if use_text_generation_model:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
            else:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
            
            # Log token usage if available
            if hasattr(response, 'usage'):
                self.log_token_usage(
                    model=model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {e}")
            return ""


class ClaudeClient(LLMClient):
    """Anthropic Claude API client implementation."""
    
    def __init__(self):
        """Initialize the Claude client."""
        config = LLM_CONFIG["claude"]
        self.client = anthropic.Anthropic(api_key=config["api_key"])
        self.model = config["model"]
        self.max_tokens = config["max_tokens"]
        self.temperature = config["temperature"]
    
    def chat_completion(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Generate a chat completion using Claude.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        try:
            # Convert OpenAI message format to Claude format
            system_message = next((m["content"] for m in messages if m["role"] == "system"), None)
            user_messages = [m["content"] for m in messages if m["role"] == "user"]
            assistant_messages = [m["content"] for m in messages if m["role"] == "assistant"]
            
            # Build messages list
            claude_messages = []
            if system_message:
                claude_messages.append({"role": "system", "content": system_message})
            
            for i in range(max(len(user_messages), len(assistant_messages))):
                if i < len(user_messages):
                    claude_messages.append({"role": "user", "content": user_messages[i]})
                if i < len(assistant_messages):
                    claude_messages.append({"role": "assistant", "content": assistant_messages[i]})
            
            response = self.client.messages.create(
                model=self.model,
                messages=claude_messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )
            
            # Log token usage
            if hasattr(response, 'usage'):
                self.log_token_usage(
                    model=self.model,
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens
                )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Error in Claude chat completion: {e}")
            raise


def create_llm_client(config=None) -> LLMClient:
    """Create an LLM client based on configuration.
    
    Args:
        config: Optional configuration dictionary
    
    Returns:
        LLMClient instance
    """
    provider = config.get('default_provider', LLM_CONFIG['default_provider']) if config else LLM_CONFIG['default_provider']
    
    if provider == "openai":
        return OpenAIClient()
    elif provider == "claude":
        return ClaudeClient()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
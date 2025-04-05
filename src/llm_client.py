"""LLM client interface and implementations."""

import abc
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


class OpenAIClient(LLMClient):
    """OpenAI API client implementation."""
    
    def __init__(self):
        """Initialize the OpenAI client."""
        config = LLM_CONFIG["openai"]
        self.client = openai.OpenAI(api_key=config["api_key"])
        self.model = config["model"]
        self.max_tokens = config["max_tokens"]
        self.temperature = config["temperature"]
    
    def chat_completion(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Generate a chat completion using OpenAI.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )
            
            # Log token usage
            if hasattr(response, 'usage'):
                self.log_token_usage(
                    model=self.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in OpenAI chat completion: {e}")
            raise


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


def create_llm_client() -> LLMClient:
    """Create an LLM client based on configuration.
    
    Returns:
        LLMClient instance
    """
    provider = LLM_CONFIG["default_provider"]
    
    if provider == "openai":
        return OpenAIClient()
    elif provider == "claude":
        return ClaudeClient()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
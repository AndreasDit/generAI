"""LLM client interface and implementations."""

import abc
from typing import Dict, Any, List, Optional

from loguru import logger
import anthropic
import openai
import tiktoken

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
        self.text_generation_model = config.get("text_generation_model", self.model)
        self.max_tokens = config["max_tokens"]
        self.temperature = config["temperature"]
        
    def _count_tokens(self, messages: List[Dict[str, str]], model: str) -> int:
        """Count the number of tokens in the messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: The model name to use for encoding
            
        Returns:
            Number of tokens in the messages
        """
        try:
            # Get the encoding for the model
            encoding = tiktoken.encoding_for_model(model)
            
            # Default to cl100k_base encoding if model-specific encoding not found
            if not encoding:
                encoding = tiktoken.get_encoding("cl100k_base")
                
            num_tokens = 0
            for message in messages:
                # Add tokens for message format (3 tokens per message)
                num_tokens += 3
                
                # Add tokens for content
                if "content" in message and message["content"]:
                    num_tokens += len(encoding.encode(message["content"]))
                    
                # Add tokens for role (1 token)
                if "role" in message:
                    num_tokens += 1
                    
            # Add tokens for message format (3 tokens at the end)
            num_tokens += 3
            
            return num_tokens
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}. Using a rough estimate instead.")
            # Fallback to a rough estimate if tiktoken fails
            return sum(len(m.get("content", "")) // 4 for m in messages)
            
    def _trim_messages(self, messages: List[Dict[str, str]], model: str, token_limit: int) -> List[Dict[str, str]]:
        """Trim messages to fit within token limit while preserving system messages and recent messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: The model name to use for encoding
            token_limit: Maximum number of tokens allowed
            
        Returns:
            Trimmed list of messages that fits within the token limit
        """
        # Separate system messages from other messages
        system_messages = [m for m in messages if m.get("role") == "system"]
        non_system_messages = [m for m in messages if m.get("role") != "system"]
        
        # If we only have system messages and they're already over the limit,
        # we need to truncate the content of the system messages
        if not non_system_messages and system_messages:
            system_tokens = self._count_tokens(system_messages, model)
            if system_tokens > token_limit:
                logger.warning("System messages alone exceed token limit. Truncating system message content.")
                # Keep only the first system message and truncate it
                encoding = tiktoken.encoding_for_model(model) or tiktoken.get_encoding("cl100k_base")
                first_system = system_messages[0]
                content = first_system["content"]
                # Approximate number of tokens to keep (leaving room for message format tokens)
                tokens_to_keep = token_limit - 10
                encoded_content = encoding.encode(content)[:tokens_to_keep]
                first_system["content"] = encoding.decode(encoded_content)
                return [first_system]
        
        # Start with system messages as they're typically important context
        trimmed_messages = system_messages.copy()
        remaining_token_budget = token_limit - self._count_tokens(trimmed_messages, model)
        
        # Add messages from most recent to oldest until we hit the token limit
        for message in reversed(non_system_messages):
            message_tokens = self._count_tokens([message], model)
            if message_tokens <= remaining_token_budget:
                trimmed_messages.insert(len(system_messages), message)  # Insert after system messages
                remaining_token_budget -= message_tokens
            else:
                # If we can't add even one message, we might need to truncate the last message
                if not trimmed_messages or len(trimmed_messages) == len(system_messages):
                    # Try to add a truncated version of the message if it's the only one
                    try:
                        encoding = tiktoken.encoding_for_model(model) or tiktoken.get_encoding("cl100k_base")
                        content = message["content"]
                        # Leave room for message format tokens
                        tokens_to_keep = remaining_token_budget - 5
                        if tokens_to_keep > 0:
                            encoded_content = encoding.encode(content)[:tokens_to_keep]
                            message["content"] = encoding.decode(encoded_content)
                            trimmed_messages.insert(len(system_messages), message)
                            logger.warning("Added truncated message to fit within token limit.")
                    except Exception as e:
                        logger.warning(f"Failed to truncate message: {e}")
                break
        
        # If we couldn't add any non-system messages, log a warning
        if len(trimmed_messages) == len(system_messages):
            logger.warning("Could not add any non-system messages within token limit.")
        else:
            logger.info(f"Trimmed messages from {len(messages)} to {len(trimmed_messages)} to fit within token limit.")
        
        return trimmed_messages
    
    def chat_completion(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, max_tokens: Optional[int] = None, use_text_generation_model: bool = False) -> str:
        """Generate a chat completion.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 1.0), defaults to instance value
            max_tokens: Maximum number of tokens to generate, defaults to instance value
            use_text_generation_model: Whether to use the text generation model instead of the default model
            
        Returns:
            Generated text response
        """
        # Use instance values if not specified
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Choose the appropriate model
        model = self.text_generation_model if use_text_generation_model else self.model
                
        # Check if we need to apply token limits
        if use_text_generation_model:
            token_limit = 128000 # 128k tokens for GPT-4o
        else:
            token_limit = 16000 # 16k tokens for GPT-4o-mini
        token_count = self._count_tokens(messages, model)
        
        # If messages exceed token limit, trim them
        if token_count > token_limit:
            logger.warning(f"Messages exceed token limit for {model} ({token_count} > {token_limit}). Trimming messages.")
            messages = self._trim_messages(messages, model, token_limit)
        
        try:
            logger.info(f"Calling OpenAI API with model: {model} and a total token input count from messages of {token_count}.")
            if use_text_generation_model:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
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
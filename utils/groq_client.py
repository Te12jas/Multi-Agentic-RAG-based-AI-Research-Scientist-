"""
Groq API client wrapper for Agentic RAG system.
Provides unified interface for LLM calls with token tracking.
"""

import json
import time
from typing import Optional, Dict, Any, List
from groq import Groq
from config import config


class GroqClient:
    """Wrapper for Groq API with token usage tracking."""
    
    def __init__(self):
        """Initialize Groq client with API key from config."""
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.total_tokens_used = 0
        self.call_count = 0
    
    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
        json_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a completion using Groq API.
        
        Args:
            prompt: The user prompt
            model: Model to use (defaults to FAST_MODEL)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            json_mode: If True, request JSON response format
            
        Returns:
            Dict with 'content', 'tokens_used', and 'model'
        """
        model = model or config.FAST_MODEL
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        start_time = time.time()
        response = self.client.chat.completions.create(**kwargs)
        latency = time.time() - start_time
        
        content = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if response.usage else 0
        
        self.total_tokens_used += tokens_used
        self.call_count += 1
        
        return {
            "content": content,
            "tokens_used": tokens_used,
            "model": model,
            "latency": latency
        }
    
    def complete_json(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a JSON completion and parse the response.
        
        Returns:
            Parsed JSON dict with metadata
        """
        result = self.complete(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            json_mode=True
        )
        
        try:
            parsed = json.loads(result["content"])
            result["parsed"] = parsed
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            content = result["content"]
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    parsed = json.loads(content[start:end])
                    result["parsed"] = parsed
                except json.JSONDecodeError:
                    result["parsed"] = None
                    result["parse_error"] = "Failed to parse JSON response"
            else:
                result["parsed"] = None
                result["parse_error"] = "No JSON found in response"
        
        return result
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return {
            "total_tokens_used": self.total_tokens_used,
            "call_count": self.call_count
        }
    
    def reset_stats(self):
        """Reset usage statistics."""
        self.total_tokens_used = 0
        self.call_count = 0


# Global client instance
groq_client = GroqClient()

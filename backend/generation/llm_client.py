"""
Unified LLM client supporting both Ollama and OpenAI.
Provides consistent interface for different LLM providers.
"""

from typing import Dict, Any, Optional, List
import asyncio
from ..core.logger import app_logger
from ..core.config import settings

class LLMClient:
    """Unified client for different LLM providers."""
    
    def __init__(self):
        self.logger = app_logger
        self.provider = settings.llm_provider
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client."""
        try:
            if self.provider == "openai":
                self._init_openai()
            elif self.provider == "ollama":
                self._init_ollama()
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
            
            self.logger.info(f"LLM client initialized with provider: {self.provider}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {str(e)}")
            raise
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            import openai
            self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
            self.model = "gpt-3.5-turbo"  # Default model
            self.logger.info("OpenAI client initialized")
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        except Exception as e:
            raise Exception(f"Failed to initialize OpenAI client: {str(e)}")
    
    def _init_ollama(self):
        """Initialize Ollama client."""
        try:
            import ollama
            self.ollama_client = ollama
            self.model = settings.ollama_model
            self.logger.info(f"Ollama client initialized with model: {self.model}")
        except ImportError:
            raise ImportError("Ollama package not installed. Run: pip install ollama")
        except Exception as e:
            raise Exception(f"Failed to initialize Ollama client: {str(e)}")
    
    async def generate_response(self, prompt: str, 
                              context: str = "",
                              max_tokens: int = 1000,
                              temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate response using the configured LLM provider.
        
        Args:
            prompt: Input prompt
            context: Additional context for the prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Response dictionary with generated text and metadata
        """
        try:
            if self.provider == "openai":
                return await self._generate_openai(prompt, context, max_tokens, temperature)
            elif self.provider == "ollama":
                return await self._generate_ollama(prompt, context, max_tokens, temperature)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                "text": f"Error generating response: {str(e)}",
                "provider": self.provider,
                "error": str(e)
            }
    
    async def _generate_openai(self, prompt: str, context: str, 
                            max_tokens: int, temperature: float) -> Dict[str, Any]:
        """Generate response using OpenAI."""
        try:
            # Construct full prompt with context
            full_prompt = self._construct_prompt(prompt, context)
            
            # Make API call
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that provides accurate and helpful responses based on the given context."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            generated_text = response.choices[0].message.content
            
            return {
                "text": generated_text,
                "provider": "openai",
                "model": self.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "finish_reason": response.choices[0].finish_reason
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI generation error: {str(e)}")
            raise
    
    async def _generate_ollama(self, prompt: str, context: str, 
                             max_tokens: int, temperature: float) -> Dict[str, Any]:
        """Generate response using Ollama."""
        try:
            # Construct full prompt with context
            full_prompt = self._construct_prompt(prompt, context)
            
            # Make API call
            response = self.ollama_client.generate(
                model=self.model,
                prompt=full_prompt,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            )
            
            generated_text = response["response"]
            
            return {
                "text": generated_text,
                "provider": "ollama",
                "model": self.model,
                "usage": {
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0),
                    "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
                },
                "finish_reason": "stop"
            }
            
        except Exception as e:
            self.logger.error(f"Ollama generation error: {str(e)}")
            raise
    
    def _construct_prompt(self, prompt: str, context: str) -> str:
        """Construct full prompt with context."""
        if not context:
            return prompt
        
        return f"""Context:
{context}

Question: {prompt}

Please provide a helpful and accurate response based on the context above. If the context doesn't contain enough information to answer the question, please say so."""
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "provider": self.provider,
            "model": self.model,
            "is_openai": self.provider == "openai",
            "is_ollama": self.provider == "ollama"
        }
    
    def switch_provider(self, new_provider: str) -> bool:
        """Switch to a different LLM provider."""
        try:
            old_provider = self.provider
            self.provider = new_provider
            self._initialize_client()
            
            self.logger.info(f"Switched LLM provider from {old_provider} to {new_provider}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to switch provider to {new_provider}: {str(e)}")
            return False
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to the LLM provider."""
        try:
            if self.provider == "openai":
                return self._test_openai()
            elif self.provider == "ollama":
                return self._test_ollama()
            else:
                return {"success": False, "error": f"Unknown provider: {self.provider}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _test_openai(self) -> Dict[str, Any]:
        """Test OpenAI connection."""
        try:
            # Simple test call
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            
            return {
                "success": True,
                "provider": "openai",
                "model": self.model,
                "response": response.choices[0].message.content
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _test_ollama(self) -> Dict[str, Any]:
        """Test Ollama connection."""
        try:
            # Simple test call
            response = self.ollama_client.generate(
                model=self.model,
                prompt="Hello",
                options={"num_predict": 10}
            )
            
            return {
                "success": True,
                "provider": "ollama",
                "model": self.model,
                "response": response["response"]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

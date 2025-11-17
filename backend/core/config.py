"""
Configuration management for the AI Knowledge Search Platform.
Handles environment variables and application settings.
"""

import os
from pathlib import Path
from typing import Literal
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv("config.env")

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # LLM Configuration
    llm_provider: Literal["ollama", "openai"] = Field(
        default="ollama", 
        env="LLM_PROVIDER"
    )
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    ollama_model: str = Field(default="llama2", env="OLLAMA_MODEL")
    
    # Embedding Configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")
    
    # Database Configuration
    chroma_persist_directory: str = Field(
        default="./data/chroma_index",
        env="CHROMA_PERSIST_DIRECTORY"
    )
    sqlite_db_path: str = Field(
        default="./data/knowledge_search.db",
        env="SQLITE_DB_PATH"
    )
    
    # Document Processing
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    max_chunks_per_query: int = Field(default=10, env="MAX_CHUNKS_PER_QUERY")
    
    # Retrieval Configuration
    bm25_k1: float = Field(default=1.2, env="BM25_K1")
    bm25_b: float = Field(default=0.75, env="BM25_B")
    hybrid_alpha: float = Field(default=0.7, env="HYBRID_ALPHA")
    
    # Context Optimization
    similarity_threshold: float = Field(default=0.9, env="SIMILARITY_THRESHOLD")
    max_context_tokens: int = Field(default=4000, env="MAX_CONTEXT_TOKENS")
    
    # Redis Cache Configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: str = Field(default="", env="REDIS_PASSWORD")
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    
    # Load Balancing
    workers: int = Field(default=4, env="WORKERS")
    
    class Config:
        env_file = "config.env"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure data directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            Path(self.chroma_persist_directory),
            Path(self.sqlite_db_path).parent,
            Path("./data/raw"),
            Path("./data/processed"),
            Path("./logs")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def is_openai_enabled(self) -> bool:
        """Check if OpenAI is properly configured."""
        return (
            self.llm_provider == "openai" and 
            self.openai_api_key and 
            self.openai_api_key != "your_openai_api_key_here"
        )
    
    @property
    def is_ollama_enabled(self) -> bool:
        """Check if Ollama is properly configured."""
        return self.llm_provider == "ollama"

# Global settings instance
settings = Settings()

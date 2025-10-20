"""
Embedding generation using sentence-transformers.
Handles text-to-vector conversion for semantic search.
"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from ..core.config import settings
from ..core.logger import app_logger

class EmbeddingGenerator:
    """Generates embeddings for text using sentence-transformers."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        self.logger = app_logger
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.logger.info("Embedding model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array containing the embedding vector
        """
        if not text or not text.strip():
            self.logger.warning("Empty text provided for embedding")
            return np.zeros(settings.embedding_dimension)
        
        try:
            # Clean and truncate text if too long
            text = self._preprocess_text(text)
            
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            return np.zeros(settings.embedding_dimension)
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of numpy arrays containing embeddings
        """
        if not texts:
            return []
        
        try:
            # Preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Generate embeddings in batch
            embeddings = self.model.encode(processed_texts, convert_to_numpy=True)
            
            # Convert to list of arrays
            return [embedding for embedding in embeddings]
            
        except Exception as e:
            self.logger.error(f"Error generating batch embeddings: {str(e)}")
            # Return zero vectors as fallback
            return [np.zeros(settings.embedding_dimension) for _ in texts]
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding generation."""
        # Remove excessive whitespace
        text = text.strip()
        
        # Truncate if too long (sentence-transformers have limits)
        max_length = 512  # Conservative limit for most models
        if len(text) > max_length:
            text = text[:max_length]
            self.logger.debug(f"Text truncated to {max_length} characters")
        
        return text
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if self.model is None:
            return settings.embedding_dimension
        
        try:
            # Generate a test embedding to get dimension
            test_embedding = self.generate_embedding("test")
            return len(test_embedding)
        except Exception:
            return settings.embedding_dimension
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         candidate_embeddings: List[np.ndarray], 
                         top_k: int = 5) -> List[tuple]:
        """
        Find most similar embeddings to query.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.similarity(query_embedding, candidate)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]

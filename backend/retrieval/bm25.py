"""
BM25 retrieval implementation for keyword-based search.
Provides traditional lexical search capabilities.
"""

import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from ..core.logger import app_logger
from ..core.config import settings
from ..core.utils import clean_text

class BM25Retriever:
    """BM25-based keyword retrieval system."""
    
    def __init__(self, k1: float = None, b: float = None):
        self.k1 = k1 or settings.bm25_k1
        self.b = b or settings.bm25_b
        self.logger = app_logger
        
        # BM25 model and corpus
        self.bm25_model = None
        self.corpus = []
        self.metadata = []
        self.is_initialized = False
    
    def initialize(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Initialize BM25 model with documents.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                self.logger.warning("No documents provided for BM25 initialization")
                return False
            
            # Extract texts and metadata
            texts = []
            metadata_list = []
            
            for doc in documents:
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                
                if not content.strip():
                    continue
                
                # Tokenize text
                tokens = self._tokenize(content)
                if tokens:
                    texts.append(tokens)
                    metadata_list.append(metadata)
            
            if not texts:
                self.logger.warning("No valid texts found for BM25 initialization")
                return False
            
            # Initialize BM25 model
            self.bm25_model = BM25Okapi(texts, k1=self.k1, b=self.b)
            self.corpus = texts
            self.metadata = metadata_list
            self.is_initialized = True
            
            self.logger.info(f"BM25 model initialized with {len(texts)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing BM25 model: {str(e)}")
            return False
    
    def search(self, query: str, top_k: int = 10, 
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search using BM25 algorithm.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results with BM25 scores
        """
        if not self.is_initialized:
            self.logger.error("BM25 model not initialized")
            return []
        
        try:
            # Tokenize query
            query_tokens = self._tokenize(query)
            if not query_tokens:
                self.logger.warning("Empty query tokens")
                return []
            
            # Get BM25 scores
            scores = self.bm25_model.get_scores(query_tokens)
            
            # Create results with scores and metadata
            results = []
            for i, (score, metadata) in enumerate(zip(scores, self.metadata)):
                # Apply metadata filters if provided
                if filter_metadata and not self._matches_filter(metadata, filter_metadata):
                    continue
                
                # Get original text (reconstruct from tokens)
                original_text = self._reconstruct_text(self.corpus[i])
                
                results.append({
                    "content": original_text,
                    "metadata": metadata,
                    "score": float(score),
                    "method": "bm25"
                })
            
            # Sort by score (descending)
            results.sort(key=lambda x: x["score"], reverse=True)
            
            # Return top-k results
            top_results = results[:top_k]
            
            self.logger.info(f"BM25 search: '{query}' -> {len(top_results)} results")
            return top_results
            
        except Exception as e:
            self.logger.error(f"Error in BM25 search: {str(e)}")
            return []
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 processing."""
        # Clean text
        text = clean_text(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Simple tokenization (can be enhanced with stemming, stopwords, etc.)
        tokens = re.findall(r'\b\w+\b', text)
        
        # Filter out very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def _reconstruct_text(self, tokens: List[str]) -> str:
        """Reconstruct text from tokens (approximate)."""
        return " ".join(tokens)
    
    def _matches_filter(self, metadata: Dict[str, Any], 
                       filter_metadata: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_metadata.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get BM25 model statistics."""
        if not self.is_initialized:
            return {"initialized": False}
        
        return {
            "initialized": True,
            "document_count": len(self.corpus),
            "total_tokens": sum(len(tokens) for tokens in self.corpus),
            "k1": self.k1,
            "b": self.b,
            "vocabulary_size": len(self.bm25_model.idf) if self.bm25_model else 0
        }
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add new documents to existing BM25 model."""
        if not self.is_initialized:
            return self.initialize(documents)
        
        try:
            # Extract new texts
            new_texts = []
            new_metadata = []
            
            for doc in documents:
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                
                if not content.strip():
                    continue
                
                tokens = self._tokenize(content)
                if tokens:
                    new_texts.append(tokens)
                    new_metadata.append(metadata)
            
            if not new_texts:
                return False
            
            # Add to existing corpus
            self.corpus.extend(new_texts)
            self.metadata.extend(new_metadata)
            
            # Reinitialize BM25 model with updated corpus
            self.bm25_model = BM25Okapi(self.corpus, k1=self.k1, b=self.b)
            
            self.logger.info(f"Added {len(new_texts)} documents to BM25 model")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding documents to BM25: {str(e)}")
            return False

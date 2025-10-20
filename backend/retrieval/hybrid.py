"""
Hybrid retrieval combining BM25 and semantic search.
Provides balanced precision and recall through weighted combination.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ..core.logger import app_logger
from ..core.config import settings
from .bm25 import BM25Retriever
from .semantic import SemanticRetriever

class HybridRetriever:
    """Hybrid retrieval combining BM25 and semantic search."""
    
    def __init__(self, alpha: float = None, 
                 bm25_weight: float = None, 
                 semantic_weight: float = None):
        """
        Initialize hybrid retriever.
        
        Args:
            alpha: Weight for semantic search (0.0 = pure BM25, 1.0 = pure semantic)
            bm25_weight: Explicit BM25 weight (overrides alpha)
            semantic_weight: Explicit semantic weight (overrides alpha)
        """
        self.alpha = alpha or settings.hybrid_alpha
        self.bm25_weight = bm25_weight or (1 - self.alpha)
        self.semantic_weight = semantic_weight or self.alpha
        
        self.logger = app_logger
        
        # Initialize component retrievers
        self.bm25_retriever = BM25Retriever()
        self.semantic_retriever = SemanticRetriever()
        
        # Ensure weights sum to 1
        total_weight = self.bm25_weight + self.semantic_weight
        if total_weight > 0:
            self.bm25_weight /= total_weight
            self.semantic_weight /= total_weight
        
        self.logger.info(f"Hybrid retriever initialized: BM25={self.bm25_weight:.2f}, Semantic={self.semantic_weight:.2f}")
    
    def initialize(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Initialize both BM25 and semantic retrievers.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            True if both retrievers initialized successfully
        """
        try:
            # Initialize BM25
            bm25_success = self.bm25_retriever.initialize(documents)
            
            # Semantic retriever uses ChromaDB (already initialized)
            semantic_success = True  # ChromaDB is persistent
            
            if bm25_success and semantic_success:
                self.logger.info("Hybrid retriever initialized successfully")
                return True
            else:
                self.logger.error("Failed to initialize hybrid retriever components")
                return False
                
        except Exception as e:
            self.logger.error(f"Error initializing hybrid retriever: {str(e)}")
            return False
    
    def search(self, query: str, top_k: int = 10,
               filter_metadata: Optional[Dict[str, Any]] = None,
               normalize_scores: bool = True) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining BM25 and semantic results.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            filter_metadata: Optional metadata filters
            normalize_scores: Whether to normalize scores before combining
            
        Returns:
            List of hybrid search results
        """
        try:
            # Get results from both retrievers
            bm25_results = self.bm25_retriever.search(
                query=query,
                top_k=top_k * 2,  # Get more results for better combination
                filter_metadata=filter_metadata
            )
            
            semantic_results = self.semantic_retriever.search(
                query=query,
                top_k=top_k * 2,
                filter_metadata=filter_metadata
            )
            
            # Combine results
            combined_results = self._combine_results(
                bm25_results, semantic_results, normalize_scores
            )
            
            # Return top-k results
            final_results = combined_results[:top_k]
            
            self.logger.info(f"Hybrid search: '{query}' -> {len(final_results)} results")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error in hybrid search: {str(e)}")
            return []
    
    def _combine_results(self, bm25_results: List[Dict[str, Any]], 
                        semantic_results: List[Dict[str, Any]],
                        normalize_scores: bool = True) -> List[Dict[str, Any]]:
        """Combine BM25 and semantic results with weighted scoring."""
        
        # Create content-to-result mapping
        content_to_result = {}
        
        # Process BM25 results
        for result in bm25_results:
            content = result["content"]
            if content not in content_to_result:
                content_to_result[content] = {
                    "content": content,
                    "metadata": result["metadata"],
                    "bm25_score": result["score"],
                    "semantic_score": 0.0,
                    "combined_score": 0.0,
                    "methods": []
                }
            content_to_result[content]["bm25_score"] = result["score"]
            content_to_result[content]["methods"].append("bm25")
        
        # Process semantic results
        for result in semantic_results:
            content = result["content"]
            if content not in content_to_result:
                content_to_result[content] = {
                    "content": content,
                    "metadata": result["metadata"],
                    "bm25_score": 0.0,
                    "semantic_score": result["score"],
                    "combined_score": 0.0,
                    "methods": []
                }
            content_to_result[content]["semantic_score"] = result["score"]
            content_to_result[content]["methods"].append("semantic")
        
        # Normalize scores if requested
        if normalize_scores:
            content_to_result = self._normalize_scores(content_to_result)
        
        # Calculate combined scores
        for content, result_data in content_to_result.items():
            combined_score = (
                self.bm25_weight * result_data["bm25_score"] +
                self.semantic_weight * result_data["semantic_score"]
            )
            result_data["combined_score"] = combined_score
        
        # Convert to list and sort by combined score
        combined_results = list(content_to_result.values())
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Format final results
        final_results = []
        for result in combined_results:
            final_results.append({
                "content": result["content"],
                "metadata": result["metadata"],
                "score": result["combined_score"],
                "method": "hybrid",
                "bm25_score": result["bm25_score"],
                "semantic_score": result["semantic_score"],
                "methods_used": result["methods"],
                "weights": {
                    "bm25": self.bm25_weight,
                    "semantic": self.semantic_weight
                }
            })
        
        return final_results
    
    def _normalize_scores(self, content_to_result: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Normalize BM25 and semantic scores to [0, 1] range."""
        
        # Collect all scores
        bm25_scores = [result["bm25_score"] for result in content_to_result.values()]
        semantic_scores = [result["semantic_score"] for result in content_to_result.values()]
        
        # Calculate min-max normalization
        if bm25_scores:
            bm25_min, bm25_max = min(bm25_scores), max(bm25_scores)
            bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1
        else:
            bm25_min, bm25_range = 0, 1
        
        if semantic_scores:
            semantic_min, semantic_max = min(semantic_scores), max(semantic_scores)
            semantic_range = semantic_max - semantic_min if semantic_max > semantic_min else 1
        else:
            semantic_min, semantic_range = 0, 1
        
        # Normalize scores
        for result in content_to_result.values():
            if bm25_scores:
                result["bm25_score"] = (result["bm25_score"] - bm25_min) / bm25_range
            if semantic_scores:
                result["semantic_score"] = (result["semantic_score"] - semantic_min) / semantic_range
        
        return content_to_result
    
    def search_with_reranking(self, query: str, top_k: int = 10,
                            rerank_top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Perform hybrid search with reranking of top results.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            rerank_top_k: Number of results to rerank
            
        Returns:
            List of reranked hybrid search results
        """
        try:
            # Get initial hybrid results
            initial_results = self.search(query, top_k=rerank_top_k)
            
            if not initial_results:
                return []
            
            # Simple reranking based on query-document similarity
            reranked_results = self._rerank_results(query, initial_results)
            
            # Return top-k reranked results
            final_results = reranked_results[:top_k]
            
            # Add reranking metadata
            for result in final_results:
                result["reranked"] = True
            
            self.logger.info(f"Hybrid search with reranking: '{query}' -> {len(final_results)} results")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error in hybrid search with reranking: {str(e)}")
            return []
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank results based on query-document similarity."""
        try:
            # Calculate semantic similarity for reranking
            for result in results:
                similarity = self.semantic_retriever.get_embedding_similarity(
                    query, result["content"]
                )
                
                # Combine original score with similarity
                result["rerank_score"] = 0.7 * result["score"] + 0.3 * similarity
            
            # Sort by rerank score
            results.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error reranking results: {str(e)}")
            return results  # Return original results if reranking fails
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hybrid retriever statistics."""
        try:
            bm25_stats = self.bm25_retriever.get_stats()
            semantic_stats = self.semantic_retriever.get_stats()
            
            return {
                "bm25_weight": self.bm25_weight,
                "semantic_weight": self.semantic_weight,
                "alpha": self.alpha,
                "bm25_stats": bm25_stats,
                "semantic_stats": semantic_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error getting hybrid retriever stats: {str(e)}")
            return {"error": str(e)}

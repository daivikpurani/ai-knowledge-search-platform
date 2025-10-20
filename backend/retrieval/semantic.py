"""
Semantic retrieval using vector embeddings and similarity search.
Provides semantic understanding and concept-based search.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from ..core.logger import app_logger
from ..core.config import settings
from ..embedding.index_manager import IndexManager
from ..embedding.embedder import EmbeddingGenerator

class SemanticRetriever:
    """Semantic retrieval using vector embeddings."""
    
    def __init__(self, similarity_threshold: float = None):
        self.similarity_threshold = similarity_threshold or 0.7
        self.logger = app_logger
        
        # Initialize components
        self.index_manager = IndexManager()
        self.embedder = EmbeddingGenerator()
    
    def search(self, query: str, top_k: int = 10, 
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results with similarity scores
        """
        try:
            # Use ChromaDB for semantic search
            results = self.index_manager.search_similar(
                query=query,
                n_results=top_k,
                filter_metadata=filter_metadata
            )
            
            # Filter by similarity threshold
            filtered_results = []
            for result in results:
                if result.get("score", 0) >= self.similarity_threshold:
                    result["method"] = "semantic"
                    filtered_results.append(result)
            
            self.logger.info(f"Semantic search: '{query}' -> {len(filtered_results)} results (threshold: {self.similarity_threshold})")
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {str(e)}")
            return []
    
    def search_with_expansion(self, query: str, top_k: int = 10,
                            expand_query: bool = True) -> List[Dict[str, Any]]:
        """
        Search with query expansion for better recall.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            expand_query: Whether to expand the query
            
        Returns:
            List of search results with expanded context
        """
        try:
            if expand_query:
                # Simple query expansion (can be enhanced)
                expanded_query = self._expand_query(query)
                self.logger.debug(f"Expanded query: '{query}' -> '{expanded_query}'")
            else:
                expanded_query = query
            
            # Perform semantic search
            results = self.search(expanded_query, top_k)
            
            # Add expansion metadata
            for result in results:
                result["query_expansion"] = expand_query
                result["original_query"] = query
                result["expanded_query"] = expanded_query
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in expanded semantic search: {str(e)}")
            return []
    
    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms."""
        # Simple expansion - can be enhanced with:
        # - Synonym dictionaries
        # - Word embeddings for related terms
        # - Query reformulation
        
        expansions = {
            # Common synonyms
            "help": "assistance support guide",
            "problem": "issue error bug trouble",
            "how": "method way process steps",
            "what": "definition meaning explanation",
            "why": "reason cause purpose",
            "when": "time date schedule",
            "where": "location place position",
            "who": "person individual user",
        }
        
        expanded_terms = []
        words = query.lower().split()
        
        for word in words:
            expanded_terms.append(word)
            if word in expansions:
                expanded_terms.append(expansions[word])
        
        return " ".join(expanded_terms)
    
    def find_similar_documents(self, document_content: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to a given document."""
        try:
            # Use document content as query
            results = self.search(document_content, top_k)
            
            # Add similarity metadata
            for result in results:
                result["similarity_type"] = "document_similarity"
                result["reference_document"] = document_content[:100] + "..."
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error finding similar documents: {str(e)}")
            return []
    
    def get_embedding_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        try:
            # Generate embeddings
            embedding1 = self.embedder.generate_embedding(text1)
            embedding2 = self.embedder.generate_embedding(text2)
            
            # Calculate cosine similarity
            similarity = self.embedder.similarity(embedding1, embedding2)
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error calculating embedding similarity: {str(e)}")
            return 0.0
    
    def cluster_similar_results(self, results: List[Dict[str, Any]], 
                              cluster_threshold: float = 0.8) -> List[List[Dict[str, Any]]]:
        """Cluster similar results together."""
        if not results:
            return []
        
        try:
            clusters = []
            used_indices = set()
            
            for i, result in enumerate(results):
                if i in used_indices:
                    continue
                
                # Start new cluster
                cluster = [result]
                used_indices.add(i)
                
                # Find similar results
                for j, other_result in enumerate(results[i+1:], i+1):
                    if j in used_indices:
                        continue
                    
                    # Calculate similarity
                    similarity = self.get_embedding_similarity(
                        result["content"], 
                        other_result["content"]
                    )
                    
                    if similarity >= cluster_threshold:
                        cluster.append(other_result)
                        used_indices.add(j)
                
                clusters.append(cluster)
            
            self.logger.info(f"Clustered {len(results)} results into {len(clusters)} clusters")
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error clustering results: {str(e)}")
            return [results]  # Return original results if clustering fails
    
    def get_stats(self) -> Dict[str, Any]:
        """Get semantic retriever statistics."""
        try:
            collection_stats = self.index_manager.get_collection_stats()
            
            return {
                "similarity_threshold": self.similarity_threshold,
                "embedding_model": self.embedder.model_name,
                "embedding_dimension": self.embedder.get_embedding_dimension(),
                "collection_stats": collection_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error getting semantic retriever stats: {str(e)}")
            return {"error": str(e)}

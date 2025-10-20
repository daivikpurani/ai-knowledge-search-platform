"""
Context deduplication to remove redundant information.
Clusters similar chunks and keeps the most relevant ones.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from ..core.logger import app_logger
from ..core.config import settings
from ..core.utils import count_tokens, calculate_similarity_score
from ..embedding.embedder import EmbeddingGenerator

class ContextDeduplicator:
    """Removes duplicate and highly similar content from search results."""
    
    def __init__(self, similarity_threshold: float = None):
        self.similarity_threshold = similarity_threshold or settings.similarity_threshold
        self.logger = app_logger
        self.embedder = EmbeddingGenerator()
    
    def deduplicate(self, results: List[Dict[str, Any]], 
                   max_results: int = None) -> List[Dict[str, Any]]:
        """
        Remove duplicate and highly similar results.
        
        Args:
            results: List of search results
            max_results: Maximum number of results to return
            
        Returns:
            Deduplicated list of results
        """
        if not results:
            return []
        
        try:
            # Step 1: Remove exact duplicates
            unique_results = self._remove_exact_duplicates(results)
            
            # Step 2: Remove highly similar content
            deduplicated_results = self._remove_similar_content(unique_results)
            
            # Step 3: Limit results if specified
            if max_results and len(deduplicated_results) > max_results:
                deduplicated_results = deduplicated_results[:max_results]
            
            # Add deduplication metadata
            for result in deduplicated_results:
                result["deduplicated"] = True
            
            self.logger.info(f"Deduplication: {len(results)} -> {len(deduplicated_results)} results")
            return deduplicated_results
            
        except Exception as e:
            self.logger.error(f"Error in deduplication: {str(e)}")
            return results  # Return original results if deduplication fails
    
    def _remove_exact_duplicates(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove exact duplicate content."""
        seen_content = set()
        unique_results = []
        
        for result in results:
            content = result.get("content", "").strip()
            if content and content not in seen_content:
                seen_content.add(content)
                unique_results.append(result)
        
        self.logger.debug(f"Removed {len(results) - len(unique_results)} exact duplicates")
        return unique_results
    
    def _remove_similar_content(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove highly similar content using clustering."""
        if len(results) <= 1:
            return results
        
        try:
            # Generate embeddings for all results
            contents = [result["content"] for result in results]
            embeddings = self.embedder.generate_embeddings_batch(contents)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Find similar pairs
            similar_pairs = self._find_similar_pairs(similarity_matrix)
            
            # Remove duplicates based on similarity
            deduplicated_results = self._remove_similar_pairs(results, similar_pairs)
            
            return deduplicated_results
            
        except Exception as e:
            self.logger.error(f"Error removing similar content: {str(e)}")
            return results
    
    def _find_similar_pairs(self, similarity_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
        """Find pairs of results with high similarity."""
        similar_pairs = []
        n = similarity_matrix.shape[0]
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                if similarity >= self.similarity_threshold:
                    similar_pairs.append((i, j, similarity))
        
        # Sort by similarity (descending)
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return similar_pairs
    
    def _remove_similar_pairs(self, results: List[Dict[str, Any]], 
                            similar_pairs: List[Tuple[int, int, float]]) -> List[Dict[str, Any]]:
        """Remove one result from each similar pair."""
        removed_indices = set()
        
        for i, j, similarity in similar_pairs:
            if i in removed_indices or j in removed_indices:
                continue
            
            # Keep the result with higher score
            if results[i]["score"] >= results[j]["score"]:
                removed_indices.add(j)
                self.logger.debug(f"Removed similar result {j} (similarity: {similarity:.3f})")
            else:
                removed_indices.add(i)
                self.logger.debug(f"Removed similar result {i} (similarity: {similarity:.3f})")
        
        # Return results not in removed_indices
        deduplicated_results = [
            result for i, result in enumerate(results) 
            if i not in removed_indices
        ]
        
        return deduplicated_results
    
    def cluster_results(self, results: List[Dict[str, Any]], 
                       min_cluster_size: int = 2) -> List[List[Dict[str, Any]]]:
        """
        Cluster similar results together.
        
        Args:
            results: List of search results
            min_cluster_size: Minimum size for a cluster
            
        Returns:
            List of clusters (each cluster is a list of results)
        """
        if len(results) < min_cluster_size:
            return [results] if results else []
        
        try:
            # Generate embeddings
            contents = [result["content"] for result in results]
            embeddings = self.embedder.generate_embeddings_batch(contents)
            
            # Perform clustering
            clustering = DBSCAN(
                eps=1 - self.similarity_threshold,  # Convert similarity to distance
                min_samples=min_cluster_size,
                metric='cosine'
            )
            
            cluster_labels = clustering.fit_predict(embeddings)
            
            # Group results by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(results[i])
            
            # Convert to list and sort clusters by average score
            cluster_list = []
            for label, cluster_results in clusters.items():
                if label == -1:  # Noise points (no cluster)
                    # Add each noise point as individual cluster
                    for result in cluster_results:
                        cluster_list.append([result])
                else:
                    # Sort cluster by score
                    cluster_results.sort(key=lambda x: x["score"], reverse=True)
                    cluster_list.append(cluster_results)
            
            # Sort clusters by their best result's score
            cluster_list.sort(key=lambda x: x[0]["score"], reverse=True)
            
            self.logger.info(f"Clustered {len(results)} results into {len(cluster_list)} clusters")
            return cluster_list
            
        except Exception as e:
            self.logger.error(f"Error clustering results: {str(e)}")
            return [results]  # Return original results if clustering fails
    
    def get_deduplication_stats(self, original_results: List[Dict[str, Any]], 
                              deduplicated_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about deduplication process."""
        if not original_results:
            return {"error": "No original results"}
        
        original_count = len(original_results)
        deduplicated_count = len(deduplicated_results)
        removed_count = original_count - deduplicated_count
        
        # Calculate token savings
        original_tokens = sum(count_tokens(result["content"]) for result in original_results)
        deduplicated_tokens = sum(count_tokens(result["content"]) for result in deduplicated_results)
        token_savings = original_tokens - deduplicated_tokens
        token_reduction_percent = (token_savings / original_tokens * 100) if original_tokens > 0 else 0
        
        return {
            "original_count": original_count,
            "deduplicated_count": deduplicated_count,
            "removed_count": removed_count,
            "reduction_percent": (removed_count / original_count * 100) if original_count > 0 else 0,
            "original_tokens": original_tokens,
            "deduplicated_tokens": deduplicated_tokens,
            "token_savings": token_savings,
            "token_reduction_percent": token_reduction_percent,
            "similarity_threshold": self.similarity_threshold
        }

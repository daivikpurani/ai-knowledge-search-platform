"""
Context compression to reduce token usage while maintaining quality.
Implements text summarization and compression techniques.
"""

from typing import List, Dict, Any, Optional
import re
from ..core.logger import app_logger
from ..core.config import settings
from ..core.utils import count_tokens, clean_text

class ContextCompressor:
    """Compresses context to reduce token usage while maintaining quality."""
    
    def __init__(self, max_tokens: int = None, compression_ratio: float = 0.7):
        self.max_tokens = max_tokens or settings.max_context_tokens
        self.compression_ratio = compression_ratio  # Target compression ratio
        self.logger = app_logger
    
    def compress(self, results: List[Dict[str, Any]], 
                target_tokens: int = None) -> List[Dict[str, Any]]:
        """
        Compress context to fit within token limits.
        
        Args:
            results: List of search results
            target_tokens: Target token count (defaults to max_tokens)
            
        Returns:
            Compressed list of results
        """
        if not results:
            return []
        
        target_tokens = target_tokens or self.max_tokens
        
        try:
            # Calculate current token count
            current_tokens = sum(count_tokens(result["content"]) for result in results)
            
            if current_tokens <= target_tokens:
                self.logger.info(f"Context already within token limit: {current_tokens}/{target_tokens}")
                return results
            
            # Apply compression strategies
            compressed_results = self._apply_compression_strategies(results, target_tokens)
            
            # Add compression metadata
            final_tokens = sum(count_tokens(result["content"]) for result in compressed_results)
            compression_ratio = final_tokens / current_tokens if current_tokens > 0 else 1.0
            
            for result in compressed_results:
                result["compressed"] = True
                result["compression_ratio"] = compression_ratio
            
            self.logger.info(f"Context compressed: {current_tokens} -> {final_tokens} tokens ({compression_ratio:.2f})")
            return compressed_results
            
        except Exception as e:
            self.logger.error(f"Error in context compression: {str(e)}")
            return results
    
    def _apply_compression_strategies(self, results: List[Dict[str, Any]], 
                                    target_tokens: int) -> List[Dict[str, Any]]:
        """Apply various compression strategies."""
        compressed_results = []
        current_tokens = 0
        
        # Sort results by score (descending)
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        for result in sorted_results:
            content = result["content"]
            content_tokens = count_tokens(content)
            
            # If adding this result would exceed target, compress it
            if current_tokens + content_tokens > target_tokens:
                # Try to compress the content
                compressed_content = self._compress_text(content, target_tokens - current_tokens)
                compressed_tokens = count_tokens(compressed_content)
                
                if compressed_tokens > 0 and current_tokens + compressed_tokens <= target_tokens:
                    # Use compressed content
                    compressed_result = result.copy()
                    compressed_result["content"] = compressed_content
                    compressed_result["original_tokens"] = content_tokens
                    compressed_result["compressed_tokens"] = compressed_tokens
                    compressed_results.append(compressed_result)
                    current_tokens += compressed_tokens
                else:
                    # Skip this result
                    break
            else:
                # Add result as-is
                compressed_results.append(result)
                current_tokens += content_tokens
        
        return compressed_results
    
    def _compress_text(self, text: str, max_tokens: int) -> str:
        """Compress text using various strategies."""
        if count_tokens(text) <= max_tokens:
            return text
        
        # Strategy 1: Remove low-relevance sentences
        compressed_text = self._remove_low_relevance_sentences(text, max_tokens)
        
        # Strategy 2: Summarize if still too long
        if count_tokens(compressed_text) > max_tokens:
            compressed_text = self._extractive_summarization(compressed_text, max_tokens)
        
        # Strategy 3: Truncate if still too long
        if count_tokens(compressed_text) > max_tokens:
            compressed_text = self._truncate_text(compressed_text, max_tokens)
        
        return compressed_text
    
    def _remove_low_relevance_sentences(self, text: str, max_tokens: int) -> str:
        """Remove sentences that are likely less relevant."""
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return text
        
        # Score sentences based on various factors
        scored_sentences = []
        for sentence in sentences:
            score = self._score_sentence(sentence)
            scored_sentences.append((sentence, score))
        
        # Sort by score (descending)
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Keep sentences until token limit
        kept_sentences = []
        current_tokens = 0
        
        for sentence, score in scored_sentences:
            sentence_tokens = count_tokens(sentence)
            if current_tokens + sentence_tokens <= max_tokens:
                kept_sentences.append(sentence)
                current_tokens += sentence_tokens
            else:
                break
        
        return " ".join(kept_sentences)
    
    def _score_sentence(self, sentence: str) -> float:
        """Score a sentence based on relevance indicators."""
        score = 0.0
        
        # Length bonus (not too short, not too long)
        length = len(sentence.split())
        if 5 <= length <= 30:
            score += 0.3
        elif length > 30:
            score += 0.1
        
        # Question bonus
        if '?' in sentence:
            score += 0.2
        
        # Definition/explanation bonus
        definition_words = ['is', 'are', 'means', 'refers to', 'defined as', 'known as']
        if any(word in sentence.lower() for word in definition_words):
            score += 0.3
        
        # Technical terms bonus
        technical_patterns = [r'\b[A-Z]{2,}\b', r'\b\d+\b', r'\b\w+ing\b']
        for pattern in technical_patterns:
            if re.search(pattern, sentence):
                score += 0.1
        
        # Avoid filler sentences
        filler_words = ['however', 'moreover', 'furthermore', 'in addition']
        if any(word in sentence.lower() for word in filler_words):
            score -= 0.1
        
        return max(score, 0.0)
    
    def _extractive_summarization(self, text: str, max_tokens: int) -> str:
        """Simple extractive summarization."""
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 2:
            return text
        
        # Score sentences
        scored_sentences = [(sentence, self._score_sentence(sentence)) for sentence in sentences]
        
        # Sort by score
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top sentences until token limit
        kept_sentences = []
        current_tokens = 0
        
        for sentence, score in scored_sentences:
            sentence_tokens = count_tokens(sentence)
            if current_tokens + sentence_tokens <= max_tokens:
                kept_sentences.append(sentence)
                current_tokens += sentence_tokens
            else:
                break
        
        return " ".join(kept_sentences)
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit token limit."""
        words = text.split()
        current_tokens = 0
        kept_words = []
        
        for word in words:
            word_tokens = count_tokens(word)
            if current_tokens + word_tokens <= max_tokens:
                kept_words.append(word)
                current_tokens += word_tokens
            else:
                break
        
        return " ".join(kept_words)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def merge_overlapping_chunks(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge overlapping chunks to reduce redundancy."""
        if len(results) <= 1:
            return results
        
        merged_results = []
        used_indices = set()
        
        for i, result in enumerate(results):
            if i in used_indices:
                continue
            
            content = result["content"]
            merged_content = content
            merged_score = result["score"]
            merged_metadata = result["metadata"].copy()
            
            # Find overlapping chunks
            for j, other_result in enumerate(results[i+1:], i+1):
                if j in used_indices:
                    continue
                
                other_content = other_result["content"]
                
                # Check for overlap
                if self._has_overlap(content, other_content):
                    # Merge content
                    merged_content = self._merge_content(merged_content, other_content)
                    merged_score = max(merged_score, other_result["score"])
                    used_indices.add(j)
            
            # Create merged result
            merged_result = {
                "content": merged_content,
                "metadata": merged_metadata,
                "score": merged_score,
                "method": result.get("method", "unknown"),
                "merged": True,
                "original_count": len([r for r in results if r not in used_indices])
            }
            
            merged_results.append(merged_result)
            used_indices.add(i)
        
        self.logger.info(f"Merged {len(results)} chunks into {len(merged_results)} chunks")
        return merged_results
    
    def _has_overlap(self, text1: str, text2: str, min_overlap: int = 20) -> bool:
        """Check if two texts have significant overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        overlap = len(words1.intersection(words2))
        return overlap >= min_overlap
    
    def _merge_content(self, content1: str, content2: str) -> str:
        """Merge two pieces of content, removing duplicates."""
        words1 = content1.split()
        words2 = content2.split()
        
        # Simple merge - concatenate and remove duplicate sentences
        merged = content1 + " " + content2
        
        # Remove duplicate sentences
        sentences = self._split_into_sentences(merged)
        unique_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if sentence_lower not in seen_sentences:
                unique_sentences.append(sentence)
                seen_sentences.add(sentence_lower)
        
        return " ".join(unique_sentences)
    
    def get_compression_stats(self, original_results: List[Dict[str, Any]], 
                            compressed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about compression process."""
        if not original_results:
            return {"error": "No original results"}
        
        original_tokens = sum(count_tokens(result["content"]) for result in original_results)
        compressed_tokens = sum(count_tokens(result["content"]) for result in compressed_results)
        
        token_savings = original_tokens - compressed_tokens
        compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        
        return {
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "token_savings": token_savings,
            "compression_ratio": compression_ratio,
            "compression_percent": (1 - compression_ratio) * 100,
            "original_count": len(original_results),
            "compressed_count": len(compressed_results),
            "max_tokens": self.max_tokens
        }

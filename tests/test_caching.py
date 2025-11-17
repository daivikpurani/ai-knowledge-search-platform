"""
Test suite for caching functionality.
Tests cache hit/miss behavior and Redis integration.
"""

import sys
from pathlib import Path
import time

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
from backend.core.cache import CacheManager
from backend.generation.rag_pipeline import RAGPipeline

class TestCaching:
    """Test caching functionality."""
    
    @pytest.fixture
    def cache_manager(self):
        """Create cache manager instance."""
        return CacheManager()
    
    @pytest.fixture
    def rag_pipeline(self):
        """Create RAG pipeline instance."""
        return RAGPipeline()
    
    def test_cache_set_get(self, cache_manager):
        """Test basic cache set and get operations."""
        if not cache_manager.is_available():
            pytest.skip("Redis cache is not available")
        
        test_key = "test:key:123"
        test_value = {"data": "test", "number": 42}
        
        # Set value
        success = cache_manager.set(test_key, test_value, ttl=60)
        assert success, "Cache set should succeed"
        
        # Get value
        cached_value = cache_manager.get(test_key)
        assert cached_value == test_value, "Cached value should match original"
        
        # Cleanup
        cache_manager.delete(test_key)
    
    def test_cache_miss(self, cache_manager):
        """Test cache miss behavior."""
        if not cache_manager.is_available():
            pytest.skip("Redis cache is not available")
        
        # Try to get non-existent key
        cached_value = cache_manager.get("test:nonexistent:key")
        assert cached_value is None, "Non-existent key should return None"
    
    def test_cache_ttl(self, cache_manager):
        """Test cache TTL expiration."""
        if not cache_manager.is_available():
            pytest.skip("Redis cache is not available")
        
        test_key = "test:ttl:key"
        test_value = {"data": "ttl_test"}
        
        # Set with short TTL
        cache_manager.set(test_key, test_value, ttl=1)
        
        # Should be available immediately
        cached_value = cache_manager.get(test_key)
        assert cached_value == test_value, "Value should be available immediately"
        
        # Wait for expiration
        time.sleep(2)
        
        # Should be expired
        cached_value = cache_manager.get(test_key)
        assert cached_value is None, "Value should be expired after TTL"
    
    def test_query_result_caching(self, cache_manager):
        """Test query result caching."""
        if not cache_manager.is_available():
            pytest.skip("Redis cache is not available")
        
        query = "What is artificial intelligence?"
        strategy = "semantic"
        max_results = 10
        test_result = {
            "query": query,
            "answer": "AI is...",
            "metadata": {"cached": False}
        }
        
        # Cache query result
        success = cache_manager.cache_query_result(query, strategy, max_results, test_result)
        assert success, "Query result caching should succeed"
        
        # Retrieve cached result
        cached_result = cache_manager.get_cached_query_result(query, strategy, max_results)
        assert cached_result is not None, "Cached query result should be retrievable"
        assert cached_result["query"] == query, "Cached query should match"
        
        # Cleanup - use public method to generate key
        import hashlib
        key_string = f"query:{query}:{strategy}:{max_results}"
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        cache_key = f"query:{key_hash}"
        cache_manager.delete(cache_key)
    
    def test_intent_classification_caching(self, cache_manager):
        """Test intent classification caching."""
        if not cache_manager.is_available():
            pytest.skip("Redis cache is not available")
        
        query = "What is machine learning?"
        intent = "factoid"
        confidence = 0.85
        explanation = {"reason": "test"}
        
        # Cache intent classification
        success = cache_manager.cache_intent_classification(query, intent, confidence, explanation)
        assert success, "Intent classification caching should succeed"
        
        # Retrieve cached classification
        cached_classification = cache_manager.get_cached_intent_classification(query)
        assert cached_classification is not None, "Cached classification should be retrievable"
        assert cached_classification["intent"] == intent, "Cached intent should match"
        assert cached_classification["confidence"] == confidence, "Cached confidence should match"
        
        # Cleanup - use public method to generate key
        import hashlib
        key_string = f"intent:{query}"
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        cache_key = f"intent:{key_hash}"
        cache_manager.delete(cache_key)
    
    @pytest.mark.asyncio
    async def test_cache_hit_performance(self, rag_pipeline, cache_manager):
        """Test that cache hits improve performance."""
        if not cache_manager.is_available():
            pytest.skip("Redis cache is not available")
        
        query = "What is artificial intelligence?"
        
        # First request (cache miss)
        start_time = time.time()
        response1 = await rag_pipeline.process_query(query=query, max_results=10)
        time1 = time.time() - start_time
        
        # Second request (cache hit)
        start_time = time.time()
        response2 = await rag_pipeline.process_query(query=query, max_results=10)
        time2 = time.time() - start_time
        
        # Cache hit should be faster
        assert time2 < time1, "Cache hit should be faster than cache miss"
        assert response2["metadata"].get("cached", False), "Second response should be cached"
        
        print(f"\nCache miss time: {time1:.3f}s")
        print(f"Cache hit time: {time2:.3f}s")
        print(f"Speedup: {time1/time2:.2f}x")
    
    def test_cache_stats(self, cache_manager):
        """Test cache statistics."""
        if not cache_manager.is_available():
            pytest.skip("Redis cache is not available")
        
        stats = cache_manager.get_cache_stats()
        assert "enabled" in stats, "Stats should include enabled status"
        assert "status" in stats, "Stats should include status"
        
        if stats["enabled"]:
            assert stats["status"] in ["connected", "error"], "Status should be connected or error"
    
    def test_cache_graceful_degradation(self, cache_manager):
        """Test that cache gracefully degrades when unavailable."""
        # Even if Redis is unavailable, operations should not raise exceptions
        test_key = "test:graceful:degradation"
        test_value = {"data": "test"}
        
        # These should not raise exceptions even if cache is unavailable
        cache_manager.set(test_key, test_value)
        cached_value = cache_manager.get(test_key)
        cache_manager.delete(test_key)
        
        # Should complete without errors
        assert True, "Cache operations should complete without errors"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


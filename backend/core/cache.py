"""
Redis cache manager for query results and intent classification.
Provides caching layer with TTL and graceful degradation.
"""

import json
import hashlib
from typing import Dict, Any, Optional, List
from ..core.logger import app_logger
from ..core.config import settings

class CacheManager:
    """Manages Redis caching for query results and other data."""
    
    def __init__(self):
        self.logger = app_logger
        self.redis_client = None
        self.enabled = settings.cache_enabled
        self.ttl = settings.cache_ttl
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis client connection."""
        if not self.enabled:
            self.logger.info("Caching is disabled")
            return
        
        try:
            import redis
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password if settings.redis_password else None,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2
            )
            
            # Test connection
            self.redis_client.ping()
            self.logger.info(f"Redis cache initialized: {settings.redis_host}:{settings.redis_port}")
            
        except ImportError:
            self.logger.warning("Redis package not installed. Caching disabled.")
            self.enabled = False
            self.redis_client = None
        except Exception as e:
            self.logger.warning(f"Failed to connect to Redis: {str(e)}. Caching disabled.")
            self.enabled = False
            self.redis_client = None
    
    def _generate_cache_key(self, prefix: str, *args) -> str:
        """Generate a cache key from arguments."""
        key_string = ":".join(str(arg) for arg in args)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            cached_value = self.redis_client.get(key)
            if cached_value:
                self.logger.debug(f"Cache hit: {key}")
                return json.loads(cached_value)
            else:
                self.logger.debug(f"Cache miss: {key}")
                return None
        except Exception as e:
            self.logger.warning(f"Error reading from cache: {str(e)}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (defaults to configured TTL)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            ttl = ttl or self.ttl
            serialized_value = json.dumps(value)
            self.redis_client.setex(key, ttl, serialized_value)
            self.logger.debug(f"Cached: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            self.logger.warning(f"Error writing to cache: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            self.redis_client.delete(key)
            self.logger.debug(f"Deleted from cache: {key}")
            return True
        except Exception as e:
            self.logger.warning(f"Error deleting from cache: {str(e)}")
            return False
    
    def cache_query_result(self, query: str, strategy: str, max_results: int, result: Dict[str, Any]) -> bool:
        """
        Cache a query result.
        
        Args:
            query: Query string
            strategy: Retrieval strategy used
            max_results: Maximum results requested
            result: Query result to cache
            
        Returns:
            True if cached successfully
        """
        cache_key = self._generate_cache_key("query", query, strategy, max_results)
        return self.set(cache_key, result)
    
    def get_cached_query_result(self, query: str, strategy: str, max_results: int) -> Optional[Dict[str, Any]]:
        """
        Get cached query result.
        
        Args:
            query: Query string
            strategy: Retrieval strategy used
            max_results: Maximum results requested
            
        Returns:
            Cached result or None
        """
        cache_key = self._generate_cache_key("query", query, strategy, max_results)
        return self.get(cache_key)
    
    def cache_intent_classification(self, query: str, intent: str, confidence: float, explanation: Dict[str, Any]) -> bool:
        """
        Cache intent classification result.
        
        Args:
            query: Query string
            intent: Classified intent
            confidence: Confidence score
            explanation: Classification explanation
            
        Returns:
            True if cached successfully
        """
        cache_key = self._generate_cache_key("intent", query)
        value = {
            "intent": intent,
            "confidence": confidence,
            "explanation": explanation
        }
        return self.set(cache_key, value, ttl=3600)  # Cache intents for 1 hour
    
    def get_cached_intent_classification(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached intent classification.
        
        Args:
            query: Query string
            
        Returns:
            Cached classification or None
        """
        cache_key = self._generate_cache_key("intent", query)
        return self.get(cache_key)
    
    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries matching a pattern.
        
        Args:
            pattern: Redis pattern (e.g., "query:*" or "intent:*")
            
        Returns:
            Number of keys deleted
        """
        if not self.enabled or not self.redis_client:
            return 0
        
        try:
            if pattern:
                keys = self.redis_client.keys(pattern)
            else:
                keys = self.redis_client.keys("*")
            
            if keys:
                deleted = self.redis_client.delete(*keys)
                self.logger.info(f"Cleared {deleted} cache entries matching pattern: {pattern or '*'}")
                return deleted
            return 0
        except Exception as e:
            self.logger.warning(f"Error clearing cache: {str(e)}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled or not self.redis_client:
            return {
                "enabled": False,
                "status": "disabled"
            }
        
        try:
            info = self.redis_client.info("stats")
            return {
                "enabled": True,
                "status": "connected",
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "total_keys": len(self.redis_client.keys("*")),
                "hit_rate": (
                    info.get("keyspace_hits", 0) / 
                    (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1)) * 100
                ) if (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0)) > 0 else 0
            }
        except Exception as e:
            return {
                "enabled": True,
                "status": "error",
                "error": str(e)
            }
    
    def is_available(self) -> bool:
        """Check if cache is available."""
        if not self.enabled:
            return False
        
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.ping()
            return True
        except Exception:
            return False


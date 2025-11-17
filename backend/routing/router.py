"""
Query router that determines the best retrieval strategy based on intent.
Routes queries to BM25, semantic, hybrid, or conversational strategies.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from ..core.logger import app_logger
from ..core.config import settings
from ..core.cache import CacheManager
from .intent_classifier import QueryIntent, IntentClassifier

class RetrievalStrategy(Enum):
    """Available retrieval strategies."""
    BM25 = "bm25"                    # Keyword-based search
    SEMANTIC = "semantic"            # Vector similarity search
    HYBRID = "hybrid"                # Combined BM25 + semantic
    CONVERSATIONAL = "conversational" # Semantic + expanded context

class StrategyConfig:
    """Configuration for retrieval strategies."""
    
    def __init__(self):
        self.configs = {
            RetrievalStrategy.BM25: {
                "max_results": 15,
                "k1": settings.bm25_k1,
                "b": settings.bm25_b,
                "description": "Keyword-based search using BM25 algorithm"
            },
            RetrievalStrategy.SEMANTIC: {
                "max_results": 12,
                "similarity_threshold": 0.7,
                "description": "Semantic search using vector embeddings"
            },
            RetrievalStrategy.HYBRID: {
                "max_results": 10,
                "alpha": settings.hybrid_alpha,  # Weight for semantic vs BM25
                "bm25_weight": 1 - settings.hybrid_alpha,
                "semantic_weight": settings.hybrid_alpha,
                "description": "Combined BM25 and semantic search"
            },
            RetrievalStrategy.CONVERSATIONAL: {
                "max_results": 8,
                "context_expansion": True,
                "similarity_threshold": 0.6,
                "description": "Semantic search with expanded context for conversational queries"
            }
        }
    
    def get_config(self, strategy: RetrievalStrategy) -> Dict[str, Any]:
        """Get configuration for a specific strategy."""
        return self.configs.get(strategy, {})
    
    def get_max_results(self, strategy: RetrievalStrategy) -> int:
        """Get maximum results for a strategy."""
        config = self.get_config(strategy)
        return config.get("max_results", 10)

class QueryRouter:
    """Routes queries to optimal retrieval strategies based on intent."""
    
    def __init__(self):
        self.logger = app_logger
        self.intent_classifier = IntentClassifier()
        self.strategy_config = StrategyConfig()
        self.cache_manager = CacheManager()
        
        # Intent to strategy mapping
        self.intent_strategy_map = {
            QueryIntent.FACTOID: RetrievalStrategy.BM25,
            QueryIntent.SEMANTIC: RetrievalStrategy.SEMANTIC,
            QueryIntent.ANALYTICAL: RetrievalStrategy.HYBRID,
            QueryIntent.CONVERSATIONAL: RetrievalStrategy.CONVERSATIONAL
        }
    
    def route_query(self, query: str, custom_strategy: Optional[RetrievalStrategy] = None) -> Dict[str, Any]:
        """
        Route a query to the optimal retrieval strategy.
        
        Args:
            query: Input query text
            custom_strategy: Override automatic strategy selection
            
        Returns:
            Routing decision with strategy and parameters
        """
        try:
            # Check cache for intent classification
            cached_intent = self.cache_manager.get_cached_intent_classification(query)
            if cached_intent:
                intent = QueryIntent(cached_intent["intent"])
                confidence = cached_intent["confidence"]
                explanation = cached_intent["explanation"]
                self.logger.debug(f"Using cached intent classification for: '{query}'")
            else:
                # Classify intent
                intent, confidence, explanation = self.intent_classifier.classify_intent(query)
                # Cache the classification
                self.cache_manager.cache_intent_classification(
                    query, intent.value, confidence, explanation
                )
            
            # Select strategy
            if custom_strategy:
                strategy = custom_strategy
                self.logger.info(f"Using custom strategy: {strategy.value}")
            else:
                strategy = self.intent_strategy_map.get(intent, RetrievalStrategy.HYBRID)
            
            # Get strategy configuration
            config = self.strategy_config.get_config(strategy)
            max_results = self.strategy_config.get_max_results(strategy)
            
            # Create routing decision
            routing_decision = {
                "query": query,
                "intent": {
                    "type": intent.value,
                    "confidence": confidence,
                    "description": self.intent_classifier.get_intent_description(intent)
                },
                "strategy": {
                    "type": strategy.value,
                    "description": config.get("description", ""),
                    "max_results": max_results,
                    "parameters": config
                },
                "explanation": explanation,
                "routing_confidence": confidence
            }
            
            self.logger.info(f"Query routed: '{query}' -> {strategy.value} (confidence: {confidence:.2f})")
            
            return routing_decision
            
        except Exception as e:
            self.logger.error(f"Error routing query '{query}': {str(e)}")
            
            # Fallback to hybrid strategy
            fallback_strategy = RetrievalStrategy.HYBRID
            config = self.strategy_config.get_config(fallback_strategy)
            
            return {
                "query": query,
                "intent": {
                    "type": "unknown",
                    "confidence": 0.0,
                    "description": "Intent classification failed"
                },
                "strategy": {
                    "type": fallback_strategy.value,
                    "description": config.get("description", ""),
                    "max_results": self.strategy_config.get_max_results(fallback_strategy),
                    "parameters": config
                },
                "explanation": {"error": str(e)},
                "routing_confidence": 0.0
            }
    
    def get_available_strategies(self) -> List[Dict[str, Any]]:
        """Get list of all available retrieval strategies."""
        strategies = []
        
        for strategy in RetrievalStrategy:
            config = self.strategy_config.get_config(strategy)
            strategies.append({
                "name": strategy.value,
                "description": config.get("description", ""),
                "max_results": config.get("max_results", 10),
                "parameters": config
            })
        
        return strategies
    
    def get_strategy_for_intent(self, intent: QueryIntent) -> RetrievalStrategy:
        """Get the recommended strategy for a specific intent."""
        return self.intent_strategy_map.get(intent, RetrievalStrategy.HYBRID)
    
    def validate_strategy(self, strategy_name: str) -> bool:
        """Validate if a strategy name is valid."""
        try:
            RetrievalStrategy(strategy_name)
            return True
        except ValueError:
            return False
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get statistics about routing decisions."""
        return {
            "available_strategies": len(RetrievalStrategy),
            "intent_types": len(QueryIntent),
            "strategy_mappings": {
                intent.value: strategy.value 
                for intent, strategy in self.intent_strategy_map.items()
            },
            "confidence_threshold": self.intent_classifier.confidence_threshold
        }

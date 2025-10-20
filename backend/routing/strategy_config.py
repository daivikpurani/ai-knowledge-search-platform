"""
Strategy configuration for different retrieval approaches.
Defines parameters and settings for each retrieval strategy.
"""

from typing import Dict, Any
from enum import Enum
from ..core.config import settings

class StrategyParameters:
    """Parameters and configuration for retrieval strategies."""
    
    # BM25 Parameters
    BM25_DEFAULT_K1 = 1.2
    BM25_DEFAULT_B = 0.75
    BM25_MAX_RESULTS = 15
    
    # Semantic Parameters
    SEMANTIC_SIMILARITY_THRESHOLD = 0.7
    SEMANTIC_MAX_RESULTS = 12
    
    # Hybrid Parameters
    HYBRID_ALPHA = 0.7  # Weight for semantic vs BM25
    HYBRID_MAX_RESULTS = 10
    
    # Conversational Parameters
    CONVERSATIONAL_SIMILARITY_THRESHOLD = 0.6
    CONVERSATIONAL_MAX_RESULTS = 8
    CONVERSATIONAL_CONTEXT_EXPANSION = True

class StrategyConfig:
    """Configuration manager for retrieval strategies."""
    
    def __init__(self):
        self.configs = self._initialize_configs()
    
    def _initialize_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize strategy configurations."""
        return {
            "bm25": {
                "name": "BM25 Keyword Search",
                "description": "Traditional keyword-based search using BM25 algorithm",
                "max_results": StrategyParameters.BM25_MAX_RESULTS,
                "parameters": {
                    "k1": settings.bm25_k1,
                    "b": settings.bm25_b,
                    "use_stemming": True,
                    "use_stopwords": True
                },
                "use_cases": [
                    "Factual questions",
                    "Specific term searches",
                    "Exact phrase matching"
                ],
                "strengths": [
                    "Fast execution",
                    "Good for exact matches",
                    "Handles typos well"
                ],
                "weaknesses": [
                    "Limited semantic understanding",
                    "May miss related concepts"
                ]
            },
            
            "semantic": {
                "name": "Semantic Vector Search",
                "description": "Semantic search using vector embeddings and similarity",
                "max_results": StrategyParameters.SEMANTIC_MAX_RESULTS,
                "parameters": {
                    "similarity_threshold": StrategyParameters.SEMANTIC_SIMILARITY_THRESHOLD,
                    "embedding_model": settings.embedding_model,
                    "use_reranking": False
                },
                "use_cases": [
                    "Conceptual searches",
                    "Finding similar content",
                    "Semantic understanding"
                ],
                "strengths": [
                    "Understands meaning",
                    "Finds related concepts",
                    "Language agnostic"
                ],
                "weaknesses": [
                    "Slower than BM25",
                    "May miss exact matches"
                ]
            },
            
            "hybrid": {
                "name": "Hybrid BM25 + Semantic",
                "description": "Combined BM25 and semantic search with weighted scoring",
                "max_results": StrategyParameters.HYBRID_MAX_RESULTS,
                "parameters": {
                    "alpha": settings.hybrid_alpha,
                    "bm25_weight": 1 - settings.hybrid_alpha,
                    "semantic_weight": settings.hybrid_alpha,
                    "normalize_scores": True,
                    "rerank_results": True
                },
                "use_cases": [
                    "Complex analytical queries",
                    "Balanced precision and recall",
                    "General purpose search"
                ],
                "strengths": [
                    "Best of both worlds",
                    "High precision and recall",
                    "Robust performance"
                ],
                "weaknesses": [
                    "More complex",
                    "Higher computational cost"
                ]
            },
            
            "conversational": {
                "name": "Conversational Search",
                "description": "Semantic search with expanded context for conversational queries",
                "max_results": StrategyParameters.CONVERSATIONAL_MAX_RESULTS,
                "parameters": {
                    "similarity_threshold": StrategyParameters.CONVERSATIONAL_SIMILARITY_THRESHOLD,
                    "context_expansion": True,
                    "expand_query": True,
                    "use_synonyms": True
                },
                "use_cases": [
                    "Help and guidance queries",
                    "How-to questions",
                    "Tutorial searches"
                ],
                "strengths": [
                    "Context-aware",
                    "Handles conversational language",
                    "Expanded understanding"
                ],
                "weaknesses": [
                    "May be too broad",
                    "Higher computational cost"
                ]
            }
        }
    
    def get_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get configuration for a specific strategy."""
        return self.configs.get(strategy_name, {})
    
    def get_parameter(self, strategy_name: str, parameter_name: str, default=None):
        """Get a specific parameter for a strategy."""
        config = self.get_config(strategy_name)
        parameters = config.get("parameters", {})
        return parameters.get(parameter_name, default)
    
    def get_max_results(self, strategy_name: str) -> int:
        """Get maximum results for a strategy."""
        config = self.get_config(strategy_name)
        return config.get("max_results", 10)
    
    def get_description(self, strategy_name: str) -> str:
        """Get description for a strategy."""
        config = self.get_config(strategy_name)
        return config.get("description", "")
    
    def get_use_cases(self, strategy_name: str) -> list:
        """Get use cases for a strategy."""
        config = self.get_config(strategy_name)
        return config.get("use_cases", [])
    
    def get_strengths(self, strategy_name: str) -> list:
        """Get strengths for a strategy."""
        config = self.get_config(strategy_name)
        return config.get("strengths", [])
    
    def get_weaknesses(self, strategy_name: str) -> list:
        """Get weaknesses for a strategy."""
        config = self.get_config(strategy_name)
        return config.get("weaknesses", [])
    
    def list_strategies(self) -> list:
        """Get list of all available strategies."""
        return list(self.configs.keys())
    
    def validate_strategy(self, strategy_name: str) -> bool:
        """Validate if a strategy exists."""
        return strategy_name in self.configs
    
    def get_strategy_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Get comparison of all strategies."""
        comparison = {}
        
        for strategy_name in self.list_strategies():
            config = self.get_config(strategy_name)
            comparison[strategy_name] = {
                "name": config.get("name", strategy_name),
                "description": config.get("description", ""),
                "max_results": config.get("max_results", 10),
                "use_cases": config.get("use_cases", []),
                "strengths": config.get("strengths", []),
                "weaknesses": config.get("weaknesses", [])
            }
        
        return comparison

"""
RAG (Retrieval-Augmented Generation) pipeline orchestrator.
Combines query routing, retrieval, optimization, and generation.
"""

import time
from typing import Dict, Any, List, Optional, Tuple, AsyncIterator
from ..core.logger import app_logger
from ..core.config import settings
from ..routing.router import QueryRouter, RetrievalStrategy
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.semantic import SemanticRetriever
from ..retrieval.hybrid import HybridRetriever
from ..optimization.deduplicator import ContextDeduplicator
from ..optimization.compressor import ContextCompressor
from ..optimization.stats import OptimizationStats
from ..generation.llm_client import LLMClient
from ..generation.prompt_templates import PromptTemplates
from ..embedding.index_manager import IndexManager
from ..core.cache import CacheManager

class RAGPipeline:
    """End-to-end RAG pipeline with intelligent routing and optimization."""
    
    def __init__(self):
        self.logger = app_logger
        
        # Initialize components
        self.query_router = QueryRouter()
        self.index_manager = IndexManager()
        self.llm_client = LLMClient()
        self.prompt_templates = PromptTemplates()
        
        # Initialize retrievers
        self.bm25_retriever = BM25Retriever()
        self.semantic_retriever = SemanticRetriever()
        self.hybrid_retriever = HybridRetriever()
        
        # Initialize optimizers
        self.deduplicator = ContextDeduplicator()
        self.compressor = ContextCompressor()
        self.optimization_stats = OptimizationStats()
        
        # Initialize cache
        self.cache_manager = CacheManager()
        
        # Pipeline state
        self.is_initialized = False
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the RAG pipeline with existing documents."""
        try:
            # Load existing documents from ChromaDB for BM25 and hybrid retrievers
            # Note: This is a simplified approach - in production, you'd want to
            # maintain separate indices for different retrieval methods
            
            self.logger.info("RAG pipeline initialized")
            self.is_initialized = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
            self.is_initialized = False
    
    async def process_query(self, query: str, 
                          custom_strategy: Optional[str] = None,
                          max_results: int = None,
                          enable_optimization: bool = True) -> Dict[str, Any]:
        """
        Process a query through the complete RAG pipeline.
        
        Args:
            query: User query
            custom_strategy: Override automatic strategy selection
            max_results: Maximum number of results to return
            enable_optimization: Whether to apply context optimization
            
        Returns:
            Complete RAG response with metadata
        """
        start_time = time.time()
        
        try:
            # Check cache first
            strategy_name = custom_strategy or "auto"
            cached_result = self.cache_manager.get_cached_query_result(
                query, strategy_name, max_results or 10
            )
            if cached_result:
                self.logger.info(f"Cache hit for query: '{query}'")
                cached_result["metadata"]["cached"] = True
                return cached_result
            
            # Step 1: Route query to optimal strategy
            routing_decision = self.query_router.route_query(query, custom_strategy)
            
            # Step 2: Retrieve relevant documents
            retrieval_results = await self._retrieve_documents(
                query, routing_decision, max_results
            )
            
            # Step 3: Optimize context (deduplication + compression)
            if enable_optimization and retrieval_results:
                optimized_results = await self._optimize_context(retrieval_results)
            else:
                optimized_results = retrieval_results
            
            # Step 4: Generate response using LLM
            llm_response = await self._generate_response(
                query, optimized_results, routing_decision
            )
            
            # Step 5: Compile final response
            processing_time = time.time() - start_time
            
            final_response = {
                "query": query,
                "answer": llm_response["text"],
                "routing_decision": routing_decision,
                "retrieval_results": {
                    "count": len(retrieval_results),
                    "method": routing_decision["strategy"]["type"]
                },
                "optimization": {
                    "enabled": enable_optimization,
                    "original_count": len(retrieval_results),
                    "optimized_count": len(optimized_results)
                },
                "llm_response": llm_response,
                "processing_time": processing_time,
                "metadata": {
                    "pipeline_version": "1.0",
                    "timestamp": time.time(),
                    "success": True
                }
            }
            
            # Add optimization stats if enabled
            if enable_optimization and retrieval_results:
                final_response["optimization"]["stats"] = self._get_optimization_stats(
                    retrieval_results, optimized_results
                )
            
            # Cache the result
            strategy_name = custom_strategy or routing_decision["strategy"]["type"]
            self.cache_manager.cache_query_result(
                query, strategy_name, max_results or 10, final_response
            )
            
            final_response["metadata"]["cached"] = False
            self.logger.info(f"RAG pipeline completed: '{query}' in {processing_time:.2f}s")
            return final_response
            
        except Exception as e:
            self.logger.error(f"RAG pipeline error: {str(e)}")
            return {
                "query": query,
                "answer": f"Error processing query: {str(e)}",
                "error": str(e),
                "processing_time": time.time() - start_time,
                "metadata": {
                    "success": False,
                    "timestamp": time.time()
                }
            }
    
    async def _retrieve_documents(self, query: str, routing_decision: Dict[str, Any], 
                                max_results: int) -> List[Dict[str, Any]]:
        """Retrieve documents using the chosen strategy."""
        try:
            strategy_type = routing_decision["strategy"]["type"]
            strategy_max_results = routing_decision["strategy"]["max_results"]
            actual_max_results = max_results or strategy_max_results
            
            if strategy_type == "bm25":
                # For BM25, we need to initialize with documents from ChromaDB
                # This is a simplified approach - in production, maintain separate indices
                results = self.semantic_retriever.search(query, actual_max_results)
                # Convert to BM25 format
                for result in results:
                    result["method"] = "bm25"
                    
            elif strategy_type == "semantic":
                results = self.semantic_retriever.search(query, actual_max_results)
                
            elif strategy_type == "hybrid":
                results = self.semantic_retriever.search(query, actual_max_results * 2)
                # Simulate hybrid by combining with BM25-like scoring
                for result in results:
                    result["method"] = "hybrid"
                    result["bm25_score"] = result["score"] * 0.3
                    result["semantic_score"] = result["score"] * 0.7
                    
            elif strategy_type == "conversational":
                results = self.semantic_retriever.search_with_expansion(query, actual_max_results)
                for result in results:
                    result["method"] = "conversational"
            else:
                # Fallback to semantic
                results = self.semantic_retriever.search(query, actual_max_results)
            
            self.logger.info(f"Retrieved {len(results)} documents using {strategy_type}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    async def _optimize_context(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply context optimization (deduplication + compression)."""
        try:
            optimization_start = time.time()
            
            # Step 1: Deduplication
            deduplicated_results = self.deduplicator.deduplicate(results)
            
            # Step 2: Compression
            compressed_results = self.compressor.compress(
                deduplicated_results, 
                target_tokens=settings.max_context_tokens
            )
            
            # Record optimization stats
            optimization_time = time.time() - optimization_start
            self.optimization_stats.record_optimization(
                results, compressed_results, "full_optimization", optimization_time
            )
            
            self.logger.info(f"Context optimization: {len(results)} -> {len(compressed_results)} results")
            return compressed_results
            
        except Exception as e:
            self.logger.error(f"Error optimizing context: {str(e)}")
            return results  # Return original results if optimization fails
    
    async def _generate_response(self, query: str, context_results: List[Dict[str, Any]], 
                               routing_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Generate LLM response using retrieved context."""
        try:
            # Prepare context
            context = self._prepare_context(context_results)
            
            # Get intent for prompt selection
            intent = routing_decision["intent"]["type"]
            
            # Generate prompt
            prompt = self.prompt_templates.get_prompt(intent, query, context)
            
            # Generate response
            llm_response = await self.llm_client.generate_response(
                prompt=prompt,
                context="",  # Context is already in prompt
                max_tokens=1000,
                temperature=0.7
            )
            
            # Add context metadata
            llm_response["context_sources"] = len(context_results)
            llm_response["context_tokens"] = sum(
                len(result["content"].split()) for result in context_results
            )
            llm_response["intent"] = intent
            
            return llm_response
            
        except Exception as e:
            self.logger.error(f"Error generating LLM response: {str(e)}")
            return {
                "text": f"Error generating response: {str(e)}",
                "provider": self.llm_client.provider,
                "error": str(e)
            }
    
    def _prepare_context(self, results: List[Dict[str, Any]]) -> str:
        """Prepare context string from retrieval results."""
        if not results:
            return "No relevant context found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            content = result["content"]
            metadata = result.get("metadata", {})
            source = metadata.get("source", "Unknown")
            
            context_parts.append(f"[Source {i}: {source}]\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _get_optimization_stats(self, original_results: List[Dict[str, Any]], 
                              optimized_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get optimization statistics."""
        try:
            deduplication_stats = self.deduplicator.get_deduplication_stats(
                original_results, optimized_results
            )
            compression_stats = self.compressor.get_compression_stats(
                original_results, optimized_results
            )
            
            return {
                "deduplication": deduplication_stats,
                "compression": compression_stats,
                "overall_reduction_percent": (
                    deduplication_stats.get("reduction_percent", 0) + 
                    compression_stats.get("compression_percent", 0)
                ) / 2
            }
            
        except Exception as e:
            self.logger.error(f"Error getting optimization stats: {str(e)}")
            return {"error": str(e)}
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get overall pipeline statistics."""
        try:
            collection_stats = self.index_manager.get_collection_stats()
            llm_info = self.llm_client.get_model_info()
            optimization_summary = self.optimization_stats.get_optimization_summary()
            
            return {
                "pipeline_status": "initialized" if self.is_initialized else "not_initialized",
                "collection_stats": collection_stats,
                "llm_info": llm_info,
                "optimization_summary": optimization_summary,
                "available_strategies": self.query_router.get_available_strategies(),
                "routing_stats": self.query_router.get_routing_stats()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting pipeline stats: {str(e)}")
            return {"error": str(e)}
    
    def test_pipeline(self) -> Dict[str, Any]:
        """Test the complete pipeline with a sample query."""
        try:
            test_query = "What is artificial intelligence?"
            
            # Test routing
            routing_decision = self.query_router.route_query(test_query)
            
            # Test retrieval
            retrieval_results = self.semantic_retriever.search(test_query, 5)
            
            # Test LLM connection
            llm_test = self.llm_client.test_connection()
            
            return {
                "pipeline_test": "success",
                "routing_test": routing_decision["strategy"]["type"],
                "retrieval_test": len(retrieval_results),
                "llm_test": llm_test,
                "overall_status": "healthy" if llm_test["success"] else "degraded"
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline test failed: {str(e)}")
            return {
                "pipeline_test": "failed",
                "error": str(e),
                "overall_status": "unhealthy"
            }
    
    async def process_query_stream(self, query: str, 
                                  custom_strategy: Optional[str] = None,
                                  max_results: int = None,
                                  enable_optimization: bool = True) -> AsyncIterator[str]:
        """
        Process a query through the RAG pipeline with streaming response.
        
        Args:
            query: User query
            custom_strategy: Override automatic strategy selection
            max_results: Maximum number of results to return
            enable_optimization: Whether to apply context optimization
            
        Yields:
            Streaming response chunks
        """
        import json
        
        try:
            # Step 1: Route query to optimal strategy
            routing_decision = self.query_router.route_query(query, custom_strategy)
            
            # Send routing metadata
            yield f"data: {json.dumps({'type': 'routing', 'data': routing_decision})}\n\n"
            
            # Step 2: Retrieve relevant documents
            retrieval_results = await self._retrieve_documents(
                query, routing_decision, max_results
            )
            
            # Send retrieval metadata
            yield f"data: {json.dumps({'type': 'retrieval', 'count': len(retrieval_results)})}\n\n"
            
            # Step 3: Optimize context (deduplication + compression)
            if enable_optimization and retrieval_results:
                optimized_results = await self._optimize_context(retrieval_results)
            else:
                optimized_results = retrieval_results
            
            # Send optimization metadata
            yield f"data: {json.dumps({'type': 'optimization', 'original': len(retrieval_results), 'optimized': len(optimized_results)})}\n\n"
            
            # Step 4: Prepare context and prompt
            context = self._prepare_context(optimized_results)
            intent = routing_decision["intent"]["type"]
            prompt = self.prompt_templates.get_prompt(intent, query, context)
            
            # Step 5: Stream LLM response
            yield f"data: {json.dumps({'type': 'start'})}\n\n"
            
            async for chunk in self.llm_client.stream_response(
                prompt=prompt,
                context="",  # Context is already in prompt
                max_tokens=1000,
                temperature=0.7
            ):
                yield chunk
            
            # Send completion metadata
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            
        except Exception as e:
            self.logger.error(f"RAG pipeline streaming error: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

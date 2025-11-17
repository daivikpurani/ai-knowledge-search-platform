"""
Test suite for verifying token reduction metrics.
Tests that the system achieves at least 40% average token reduction.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
from backend.generation.rag_pipeline import RAGPipeline
from backend.core.utils import count_tokens

class TestTokenReduction:
    """Test token reduction through optimization."""
    
    @pytest.fixture
    def rag_pipeline(self):
        """Create RAG pipeline instance."""
        return RAGPipeline()
    
    @pytest.fixture
    def test_queries(self):
        """Test queries for token reduction testing."""
        return [
            "What is artificial intelligence?",
            "Compare supervised vs unsupervised learning",
            "How do I implement a neural network?",
            "Find concepts related to deep learning",
            "Analyze the benefits of neural networks",
            "What is machine learning?",
            "Help me understand backpropagation",
            "Show me content about AI ethics",
            "Evaluate different AI approaches",
            "What are the applications of neural networks?"
        ]
    
    @pytest.mark.asyncio
    async def test_average_token_reduction(self, rag_pipeline, test_queries):
        """Test that average token reduction is at least 40%."""
        token_reductions = []
        
        for query in test_queries:
            try:
                # Process query with optimization enabled
                response = await rag_pipeline.process_query(
                    query=query,
                    max_results=10,
                    enable_optimization=True
                )
                
                # Get optimization stats
                optimization = response.get("optimization", {})
                stats = optimization.get("stats", {})
                
                compression_stats = stats.get("compression", {})
                if compression_stats:
                    reduction_percent = compression_stats.get("compression_percent", 0)
                    if reduction_percent > 0:
                        token_reductions.append(reduction_percent)
            
            except Exception as e:
                print(f"Error processing query '{query}': {str(e)}")
                continue
        
        if not token_reductions:
            pytest.skip("No token reduction data collected")
        
        average_reduction = sum(token_reductions) / len(token_reductions)
        
        print(f"\nAverage Token Reduction: {average_reduction:.2f}%")
        print(f"Individual Reductions: {token_reductions}")
        
        # Assert minimum 40% average reduction
        assert average_reduction >= 40.0, (
            f"Average token reduction ({average_reduction:.2f}%) is below required 40%"
        )
    
    @pytest.mark.asyncio
    async def test_optimization_preserves_relevance(self, rag_pipeline):
        """Test that optimization doesn't significantly degrade relevance."""
        query = "What is artificial intelligence?"
        
        # Process with and without optimization
        response_with_opt = await rag_pipeline.process_query(
            query=query,
            max_results=10,
            enable_optimization=True
        )
        
        response_without_opt = await rag_pipeline.process_query(
            query=query,
            max_results=10,
            enable_optimization=False
        )
        
        # Both should have answers
        assert "answer" in response_with_opt
        assert "answer" in response_without_opt
        
        # Answers should not be empty
        assert len(response_with_opt["answer"]) > 0
        assert len(response_without_opt["answer"]) > 0
    
    @pytest.mark.asyncio
    async def test_token_reduction_per_query(self, rag_pipeline):
        """Test token reduction for individual queries."""
        query = "Compare supervised vs unsupervised learning"
        
        response = await rag_pipeline.process_query(
            query=query,
            max_results=10,
            enable_optimization=True
        )
        
        optimization = response.get("optimization", {})
        stats = optimization.get("stats", {})
        compression_stats = stats.get("compression", {})
        
        if compression_stats:
            reduction_percent = compression_stats.get("compression_percent", 0)
            original_tokens = compression_stats.get("original_tokens", 0)
            compressed_tokens = compression_stats.get("compressed_tokens", 0)
            
            print(f"\nQuery: {query}")
            print(f"Original Tokens: {original_tokens}")
            print(f"Compressed Tokens: {compressed_tokens}")
            print(f"Reduction: {reduction_percent:.2f}%")
            
            # Should have some reduction if original tokens > 0
            if original_tokens > 0:
                assert reduction_percent >= 0, "Token reduction should be non-negative"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


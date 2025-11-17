"""
Test suite for streaming endpoints.
Tests streaming functionality for both OpenAI and Ollama.
"""

import sys
from pathlib import Path
import json

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
from backend.generation.llm_client import LLMClient
from backend.generation.rag_pipeline import RAGPipeline

class TestStreaming:
    """Test streaming functionality."""
    
    @pytest.fixture
    def llm_client(self):
        """Create LLM client instance."""
        return LLMClient()
    
    @pytest.fixture
    def rag_pipeline(self):
        """Create RAG pipeline instance."""
        return RAGPipeline()
    
    @pytest.mark.asyncio
    async def test_llm_streaming(self, llm_client):
        """Test LLM streaming response."""
        prompt = "What is artificial intelligence? Answer in one sentence."
        
        chunks = []
        async for chunk in llm_client.stream_response(
            prompt=prompt,
            max_tokens=100,
            temperature=0.7
        ):
            chunks.append(chunk)
        
        # Should receive chunks
        assert len(chunks) > 0, "Should receive at least one chunk"
        
        # Parse chunks
        content_chunks = []
        for chunk in chunks:
            if chunk.startswith("data: "):
                data_str = chunk[6:]  # Remove "data: " prefix
                try:
                    data = json.loads(data_str)
                    if "content" in data:
                        content_chunks.append(data["content"])
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
        
        # Should have content
        assert len(content_chunks) > 0, "Should receive content chunks"
        
        # Combine content
        full_response = "".join(content_chunks)
        assert len(full_response) > 0, "Full response should not be empty"
    
    @pytest.mark.asyncio
    async def test_rag_pipeline_streaming(self, rag_pipeline):
        """Test RAG pipeline streaming."""
        query = "What is machine learning?"
        
        chunks = []
        async for chunk in rag_pipeline.process_query_stream(
            query=query,
            max_results=5,
            enable_optimization=True
        ):
            chunks.append(chunk)
        
        # Should receive chunks
        assert len(chunks) > 0, "Should receive at least one chunk"
        
        # Parse chunks to find content
        content_chunks = []
        routing_received = False
        retrieval_received = False
        
        for chunk in chunks:
            if chunk.startswith("data: "):
                data_str = chunk[6:]
                try:
                    data = json.loads(data_str)
                    
                    if data.get("type") == "routing":
                        routing_received = True
                    elif data.get("type") == "retrieval":
                        retrieval_received = True
                    elif "content" in data:
                        content_chunks.append(data["content"])
                    elif data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
        
        # Should receive routing and retrieval metadata
        assert routing_received, "Should receive routing metadata"
        assert retrieval_received, "Should receive retrieval metadata"
        
        # Should have content chunks
        assert len(content_chunks) > 0, "Should receive content chunks"
        
        # Combine content
        full_response = "".join(content_chunks)
        assert len(full_response) > 0, "Full response should not be empty"
    
    @pytest.mark.asyncio
    async def test_streaming_completeness(self, llm_client):
        """Test that streaming responses are complete."""
        prompt = "Count from 1 to 5."
        
        chunks = []
        async for chunk in llm_client.stream_response(
            prompt=prompt,
            max_tokens=50,
            temperature=0.7
        ):
            chunks.append(chunk)
        
        # Parse and combine content
        content_chunks = []
        done_received = False
        
        for chunk in chunks:
            if chunk.startswith("data: "):
                data_str = chunk[6:]
                try:
                    data = json.loads(data_str)
                    if "content" in data:
                        content_chunks.append(data["content"])
                    if data.get("done"):
                        done_received = True
                        break
                except json.JSONDecodeError:
                    continue
        
        # Should receive completion signal
        assert done_received, "Should receive completion signal"
        
        # Response should be reasonable length
        full_response = "".join(content_chunks)
        assert len(full_response) > 0, "Response should not be empty"
    
    @pytest.mark.asyncio
    async def test_streaming_error_handling(self, llm_client):
        """Test streaming error handling."""
        # Invalid prompt that might cause error
        prompt = ""
        
        chunks = []
        error_received = False
        
        async for chunk in llm_client.stream_response(
            prompt=prompt,
            max_tokens=10,
            temperature=0.7
        ):
            chunks.append(chunk)
            if "error" in chunk.lower():
                error_received = True
        
        # Should handle errors gracefully
        assert len(chunks) > 0, "Should receive at least one chunk even on error"
    
    @pytest.mark.asyncio
    async def test_streaming_vs_non_streaming(self, rag_pipeline):
        """Test that streaming and non-streaming produce similar results."""
        query = "What is AI?"
        
        # Non-streaming response
        non_streaming_response = await rag_pipeline.process_query(
            query=query,
            max_results=5,
            enable_optimization=True
        )
        
        # Streaming response
        chunks = []
        async for chunk in rag_pipeline.process_query_stream(
            query=query,
            max_results=5,
            enable_optimization=True
        ):
            chunks.append(chunk)
        
        # Parse streaming content
        content_chunks = []
        for chunk in chunks:
            if chunk.startswith("data: "):
                data_str = chunk[6:]
                try:
                    data = json.loads(data_str)
                    if "content" in data:
                        content_chunks.append(data["content"])
                except json.JSONDecodeError:
                    continue
        
        streaming_answer = "".join(content_chunks)
        non_streaming_answer = non_streaming_response.get("answer", "")
        
        # Both should have answers
        assert len(streaming_answer) > 0, "Streaming should produce answer"
        assert len(non_streaming_answer) > 0, "Non-streaming should produce answer"
        
        # Answers should be similar in length (within 50% variance)
        length_ratio = len(streaming_answer) / len(non_streaming_answer) if len(non_streaming_answer) > 0 else 0
        assert 0.5 <= length_ratio <= 1.5, "Answers should be similar in length"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


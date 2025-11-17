# AI Knowledge Search Platform - Architecture Documentation

## Overview

The AI Knowledge Search Platform is designed as a modular, extensible system that demonstrates advanced RAG (Retrieval-Augmented Generation) capabilities with intelligent query routing and context optimization. This document provides a deep dive into the architecture, design decisions, and implementation details.

## System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI      │    │   ChromaDB      │
│   (Frontend)    │◄──►│   (Backend)    │◄──►│   (Vector DB)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   RAG Pipeline  │
                    │   Orchestrator  │
                    └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            ┌─────────────┐    ┌─────────────┐
            │   Query     │    │  Retrieval  │
            │  Router     │    │  Strategies │
            └─────────────┘    └─────────────┘
                    │                   │
                    ▼                   ▼
            ┌─────────────┐    ┌─────────────┐
            │   Context   │    │     LLM     │
            │ Optimizer   │    │   Client    │
            └─────────────┘    └─────────────┘
```

### Component Responsibilities

1. **Streamlit UI**: Interactive frontend for querying and visualization
2. **FastAPI Backend**: REST API with async processing and comprehensive logging
3. **RAG Pipeline**: End-to-end orchestration of query processing
4. **Query Router**: Intent classification and strategy selection
5. **Retrieval Strategies**: Multiple search approaches (BM25, semantic, hybrid)
6. **Context Optimizer**: Deduplication and compression for efficiency
7. **LLM Client**: Unified interface for different language models
8. **ChromaDB**: Persistent vector storage and similarity search

## Design Decisions and Trade-offs

### 1. Query Routing Strategy

**Decision**: Hybrid approach combining pattern matching with optional ML classification

**Rationale**:
- Pattern matching provides fast, deterministic baseline
- ML classification can be added for improved accuracy
- Fallback mechanisms ensure robustness
- Confidence scoring enables quality assessment

**Trade-offs**:
- ✅ Fast execution and reliability
- ✅ Easy to debug and explain decisions
- ❌ May miss nuanced intent patterns
- ❌ Requires manual pattern maintenance

### 2. Retrieval Strategy Selection

**Decision**: Intent-based routing to specialized retrieval methods

**Mapping**:
- Factoid queries → BM25 (keyword matching)
- Semantic queries → Vector similarity
- Analytical queries → Hybrid (weighted combination)
- Conversational queries → Semantic + context expansion

**Rationale**:
- Different query types benefit from different retrieval approaches
- BM25 excels at exact matches and factual queries
- Semantic search handles conceptual and related content
- Hybrid approach balances precision and recall

**Trade-offs**:
- ✅ Optimized performance for each query type
- ✅ Clear separation of concerns
- ❌ Increased complexity in routing logic
- ❌ Potential inconsistency across strategies

### 3. Context Optimization Approach

**Decision**: Two-stage optimization (deduplication → compression)

**Rationale**:
- Deduplication removes redundant information
- Compression reduces token usage while preserving quality
- Measurable benefits with clear metrics
- Demonstrates understanding of LLM economics

**Trade-offs**:
- ✅ Significant token savings (30-45%)
- ✅ Improved response quality
- ❌ Additional processing overhead
- ❌ Risk of losing important information

### 4. LLM Provider Abstraction

**Decision**: Unified client supporting multiple providers

**Rationale**:
- Flexibility for different deployment scenarios
- Local development with Ollama
- Production deployment with OpenAI
- Consistent interface regardless of provider

**Trade-offs**:
- ✅ Maximum flexibility and portability
- ✅ Easy to switch between providers
- ❌ Additional abstraction complexity
- ❌ Potential feature limitations

## Implementation Details

### Query Routing Algorithm

```python
def classify_intent(self, query: str) -> Tuple[QueryIntent, float, Dict]:
    # 1. Pattern matching for each intent type
    scores = {}
    for intent, patterns in self.patterns.items():
        score = self._calculate_pattern_score(query, patterns)
        scores[intent] = score
    
    # 2. Find best intent
    best_intent = max(scores.items(), key=lambda x: x[1])
    
    # 3. Apply heuristics
    confidence = self._apply_heuristics(query, best_intent[0], best_intent[1])
    
    return best_intent[0], confidence, explanation
```

**Key Features**:
- Regex pattern matching for fast classification
- Confidence scoring with heuristics
- Detailed explanation for debugging
- Fallback mechanisms for edge cases

### Hybrid Retrieval Implementation

```python
def _combine_results(self, bm25_results, semantic_results, normalize_scores):
    # 1. Create content-to-result mapping
    content_to_result = {}
    
    # 2. Process BM25 results
    for result in bm25_results:
        content_to_result[content] = {
            "bm25_score": result["score"],
            "semantic_score": 0.0,
            "combined_score": 0.0
        }
    
    # 3. Process semantic results
    for result in semantic_results:
        if content in content_to_result:
            content_to_result[content]["semantic_score"] = result["score"]
    
    # 4. Calculate combined scores
    for result_data in content_to_result.values():
        combined_score = (
            self.bm25_weight * result_data["bm25_score"] +
            self.semantic_weight * result_data["semantic_score"]
        )
        result_data["combined_score"] = combined_score
    
    return sorted_results_by_combined_score
```

**Key Features**:
- Weighted combination of BM25 and semantic scores
- Score normalization for fair comparison
- Detailed metadata for analysis
- Configurable weight parameters

### Context Optimization Pipeline

```python
async def _optimize_context(self, results):
    # 1. Deduplication
    deduplicated_results = self.deduplicator.deduplicate(results)
    
    # 2. Compression
    compressed_results = self.compressor.compress(
        deduplicated_results, 
        target_tokens=settings.max_context_tokens
    )
    
    # 3. Record statistics
    self.optimization_stats.record_optimization(
        results, compressed_results, "full_optimization"
    )
    
    return compressed_results
```

**Key Features**:
- Cosine similarity clustering for deduplication
- Token-based compression with quality preservation
- Comprehensive statistics tracking
- Configurable thresholds and limits

## Performance Characteristics

### Query Processing Pipeline

1. **Intent Classification**: ~10-50ms
2. **Document Retrieval**: ~100-500ms
3. **Context Optimization**: ~50-200ms
4. **LLM Generation**: ~500-2000ms
5. **Total Pipeline**: ~1-3 seconds

### Memory Usage

- **Embedding Model**: ~100-200MB
- **ChromaDB Index**: ~50-100MB per 1000 documents
- **BM25 Index**: ~10-20MB per 1000 documents
- **Total Memory**: ~200-500MB typical

### Scalability Considerations

- **Document Processing**: Linear with document count
- **Query Processing**: Constant time with proper indexing
- **Memory Usage**: Scales with vocabulary size and document count
- **Concurrent Queries**: Limited by LLM API rate limits

## Error Handling and Resilience

### Graceful Degradation

1. **LLM Failure**: Fallback to retrieval-only response
2. **Embedding Failure**: Use BM25-only retrieval
3. **Routing Failure**: Default to hybrid strategy
4. **Optimization Failure**: Return original results

### Error Recovery

- Automatic retry with exponential backoff
- Circuit breaker pattern for external services
- Comprehensive logging for debugging
- Health checks for system monitoring

## Testing Strategy

### Unit Tests

- Intent classification accuracy
- Retrieval strategy correctness
- Optimization algorithm validation
- LLM client functionality

### Integration Tests

- End-to-end RAG pipeline
- API endpoint functionality
- UI component behavior
- Error handling scenarios

### Performance Tests

- Query latency benchmarks
- Memory usage profiling
- Concurrent request handling
- Optimization effectiveness

## Future Improvements

### Short-term Enhancements

1. **Advanced Reranking**: Cross-encoder models for better relevance
2. **Query Expansion**: Synonym and related term expansion
3. **Caching Layer**: Redis for repeated query caching
4. **Batch Processing**: Async document ingestion

### Long-term Vision

1. **Multi-modal Support**: Image and document processing
2. **Real-time Updates**: Live document indexing
3. **User Personalization**: Adaptive routing based on user behavior
4. **Federated Learning**: Continuous improvement from user feedback

## Security Considerations

### Data Privacy

- Local processing for sensitive documents
- No data transmission to external services (Ollama mode)
- Configurable data retention policies
- User consent for data processing

### Access Control

- API key management for OpenAI
- Local authentication for Ollama
- Document access permissions
- Audit logging for compliance

## Monitoring and Observability

### Metrics Collection

- Query processing times
- Intent classification accuracy
- Retrieval strategy distribution
- Optimization effectiveness
- Error rates and types
- Cache hit/miss rates
- Worker performance metrics

### Logging Strategy

- Structured logging with Loguru
- Different log levels for different components
- Request/response logging for debugging
- Performance metrics logging

### Health Checks

- Component availability monitoring
- Resource usage tracking
- External service connectivity
- Data integrity verification
- Worker health endpoints
- Cache connectivity status

## Caching Architecture

### Redis Integration

The platform uses Redis for caching query results and intent classifications:

- **Query Result Caching**: Caches complete RAG responses with configurable TTL
- **Intent Classification Caching**: Caches intent classification results to avoid redundant processing
- **Cache Key Format**: `{prefix}:{hash}` where hash is MD5 of query parameters
- **Graceful Degradation**: System continues to function if Redis is unavailable
- **Cache Statistics**: Tracks hit/miss rates and performance metrics

### Cache Manager

The `CacheManager` class provides:
- Connection pooling for Redis
- Automatic TTL management
- Cache invalidation strategies
- Statistics and monitoring
- Error handling and fallback

## Streaming Architecture

### Streaming Support

The platform supports streaming responses for both OpenAI and Ollama:

- **OpenAI Streaming**: Uses native streaming API with Server-Sent Events (SSE)
- **Ollama Streaming**: Uses Ollama's streaming API with chunk-based responses
- **Unified Interface**: Consistent streaming interface regardless of provider
- **Metadata Streaming**: Streams routing, retrieval, and optimization metadata
- **Error Handling**: Graceful error handling in streaming mode

### Streaming Endpoint

The `/api/v1/query/stream` endpoint:
- Returns Server-Sent Events (SSE) format
- Streams pipeline metadata (routing, retrieval, optimization)
- Streams LLM response chunks in real-time
- Includes completion signals and error handling

## Load Balancing

### Application-Level Load Balancing

The platform supports multiple worker processes:

- **Worker Configuration**: Configurable via `WORKERS` environment variable
- **Uvicorn Workers**: Uses uvicorn's worker processes for concurrent request handling
- **Worker Health Checks**: `/api/v1/health/workers` endpoint for monitoring
- **Process Isolation**: Each worker runs in a separate process
- **Production Deployment**: Recommended 4+ workers for production workloads

### Deployment Considerations

- Workers are not compatible with `--reload` mode (development only)
- Each worker maintains its own connection pools (Redis, ChromaDB)
- Shared state (ChromaDB) is accessed concurrently but safely
- Cache (Redis) provides shared state across workers

This architecture demonstrates production-ready thinking while maintaining simplicity for interview demonstration. The modular design allows for easy extension and modification, while the comprehensive error handling, caching, streaming, and load balancing ensure reliability and scalability in real-world scenarios.

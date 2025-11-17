"""
Health check endpoints for monitoring application status.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import os
from ..core.logger import app_logger
from ..core.config import settings
from ..embedding.index_manager import IndexManager

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "message": "AI Knowledge Search Platform is running"
    }

@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with component status."""
    health_status = {
        "status": "healthy",
        "timestamp": None,
        "components": {}
    }
    
    try:
        # Check ChromaDB connection
        index_manager = IndexManager()
        stats = index_manager.get_collection_stats()
        
        health_status["components"]["chromadb"] = {
            "status": "healthy",
            "documents": stats.get("total_documents", 0),
            "collection": stats.get("collection_name", "unknown")
        }
        
    except Exception as e:
        app_logger.error(f"ChromaDB health check failed: {str(e)}")
        health_status["components"]["chromadb"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check embedding model
    try:
        from ..embedding.embedder import EmbeddingGenerator
        embedder = EmbeddingGenerator()
        dimension = embedder.get_embedding_dimension()
        
        health_status["components"]["embeddings"] = {
            "status": "healthy",
            "model": embedder.model_name,
            "dimension": dimension
        }
        
    except Exception as e:
        app_logger.error(f"Embedding model health check failed: {str(e)}")
        health_status["components"]["embeddings"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check LLM provider
    try:
        from ..core.config import settings
        
        if settings.llm_provider == "openai":
            health_status["components"]["llm"] = {
                "status": "healthy" if settings.is_openai_enabled else "unhealthy",
                "provider": "openai",
                "configured": settings.is_openai_enabled
            }
        else:
            health_status["components"]["llm"] = {
                "status": "healthy",
                "provider": "ollama",
                "model": settings.ollama_model
            }
            
    except Exception as e:
        app_logger.error(f"LLM health check failed: {str(e)}")
        health_status["components"]["llm"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    return health_status

@router.get("/health/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check for Kubernetes-style deployments."""
    try:
        # Check if all critical components are ready
        index_manager = IndexManager()
        stats = index_manager.get_collection_stats()
        
        if stats.get("total_documents", 0) == 0:
            return {
                "status": "not_ready",
                "message": "No documents indexed yet"
            }
        
        return {
            "status": "ready",
            "message": "All components are ready"
        }
        
    except Exception as e:
        app_logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service not ready")

@router.get("/health/workers")
async def worker_health_check() -> Dict[str, Any]:
    """Worker health check for load balancing."""
    try:
        worker_id = os.getpid()
        worker_count = settings.workers
        
        return {
            "status": "healthy",
            "worker_id": worker_id,
            "configured_workers": worker_count,
            "message": f"Worker {worker_id} is running"
        }
    except Exception as e:
        app_logger.error(f"Worker health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

"""
FastAPI main application for AI Knowledge Search Platform.
Provides REST API endpoints for document ingestion and querying.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import os
from pathlib import Path

from .core.config import settings
from .core.logger import app_logger
from .api.query_routes import router as query_router
from .api.health import router as health_router

# Initialize FastAPI app
app = FastAPI(
    title="AI Knowledge Search Platform",
    description="Intelligent document search with query routing and context optimization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(query_router, prefix="/api/v1")
app.include_router(health_router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    app_logger.info("Starting AI Knowledge Search Platform")
    app_logger.info(f"LLM Provider: {settings.llm_provider}")
    app_logger.info(f"Embedding Model: {settings.embedding_model}")
    
    # Ensure data directories exist
    Path("./data").mkdir(exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    app_logger.info("Shutting down AI Knowledge Search Platform")

@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "AI Knowledge Search Platform",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/api/v1/info")
async def get_info():
    """Get application information and configuration."""
    return {
        "app_name": "AI Knowledge Search Platform",
        "version": "1.0.0",
        "llm_provider": settings.llm_provider,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "max_chunks_per_query": settings.max_chunks_per_query,
        "data_directory": settings.chroma_persist_directory
    }

if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Get worker count from environment or settings
    workers = settings.workers if "--workers" not in sys.argv else None
    
    # Check if running in production mode (no reload)
    reload = "--reload" in sys.argv or "--dev" in sys.argv
    
    # Parse workers from command line if provided
    if "--workers" in sys.argv:
        try:
            workers_idx = sys.argv.index("--workers")
            workers = int(sys.argv[workers_idx + 1])
        except (IndexError, ValueError):
            workers = settings.workers
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=reload,
        workers=workers if not reload else 1,  # Workers not compatible with reload
        log_level="info"
    )

"""
Query and ingestion API routes.
Handles document upload, processing, and search queries.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import os
import tempfile
from pathlib import Path

from ..core.logger import app_logger
from ..core.utils import validate_query
from ..ingestion.loader import DocumentLoader
from ..ingestion.chunker import DocumentChunker
from ..embedding.index_manager import IndexManager
from ..generation.rag_pipeline import RAGPipeline

router = APIRouter(tags=["query"])

# Initialize components
document_loader = DocumentLoader()
document_chunker = DocumentChunker()
index_manager = IndexManager()
rag_pipeline = RAGPipeline()

class QueryRequest(BaseModel):
    """Request model for search queries."""
    query: str
    max_results: Optional[int] = 10
    include_metadata: Optional[bool] = True

class QueryResponse(BaseModel):
    """Response model for search queries."""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    processing_time: float
    metadata: Dict[str, Any]

class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    success: bool
    message: str
    documents_processed: int
    chunks_created: int
    errors: List[str]

@router.post("/query", response_model=QueryResponse)
async def search_documents(request: QueryRequest):
    """
    Process query through complete RAG pipeline with intelligent routing.
    
    Args:
        request: Query request with search text and parameters
        
    Returns:
        Complete RAG response with answer and metadata
    """
    start_time = time.time()
    
    try:
        # Validate query
        if not validate_query(request.query):
            raise HTTPException(
                status_code=400,
                detail="Invalid query. Query must be 3-1000 characters long."
            )
        
        # Process through RAG pipeline
        rag_response = await rag_pipeline.process_query(
            query=request.query,
            max_results=request.max_results,
            enable_optimization=True
        )
        
        processing_time = time.time() - start_time
        
        # Format response for backward compatibility
        response = QueryResponse(
            query=request.query,
            results=rag_response.get("retrieval_results", {}),
            total_results=rag_response.get("retrieval_results", {}).get("count", 0),
            processing_time=processing_time,
            metadata={
                "rag_response": rag_response,
                "routing_decision": rag_response.get("routing_decision", {}),
                "optimization": rag_response.get("optimization", {}),
                "llm_response": rag_response.get("llm_response", {})
            }
        )
        
        app_logger.info(f"RAG query completed: '{request.query}' in {processing_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"RAG query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@router.post("/query/rag")
async def rag_query(request: QueryRequest):
    """
    Process query through complete RAG pipeline (full response).
    
    Args:
        request: Query request with search text and parameters
        
    Returns:
        Complete RAG response with answer, routing, and optimization details
    """
    try:
        # Validate query
        if not validate_query(request.query):
            raise HTTPException(
                status_code=400,
                detail="Invalid query. Query must be 3-1000 characters long."
            )
        
        # Process through RAG pipeline
        rag_response = await rag_pipeline.process_query(
            query=request.query,
            max_results=request.max_results,
            enable_optimization=True
        )
        
        return rag_response
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"RAG query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAG processing failed: {str(e)}")

@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """
    Ingest a single document file.
    
    Args:
        file: Uploaded file (PDF, Markdown, or Text)
        
    Returns:
        Ingestion status and statistics
    """
    errors = []
    documents_processed = 0
    chunks_created = 0
    
    try:
        # Validate file type
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ['.pdf', '.md', '.txt', '.markdown']:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Load document
            document = document_loader.load_document(temp_file_path)
            if not document:
                raise HTTPException(status_code=400, detail="Failed to load document")
            
            # Chunk document
            chunks = document_chunker.chunk_document(document)
            if not chunks:
                raise HTTPException(status_code=400, detail="Failed to chunk document")
            
            # Add to index
            success = index_manager.add_documents(chunks)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to add documents to index")
            
            documents_processed = 1
            chunks_created = len(chunks)
            
            app_logger.info(f"Successfully ingested file: {file.filename} -> {chunks_created} chunks")
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
        
        return IngestResponse(
            success=True,
            message=f"Successfully processed {file.filename}",
            documents_processed=documents_processed,
            chunks_created=chunks_created,
            errors=errors
        )
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"File ingestion error: {str(e)}")
        return IngestResponse(
            success=False,
            message=f"Failed to process {file.filename}",
            documents_processed=documents_processed,
            chunks_created=chunks_created,
            errors=[str(e)]
        )

@router.post("/ingest/directory", response_model=IngestResponse)
async def ingest_directory(directory_path: str = Form(...)):
    """
    Ingest all supported documents from a directory.
    
    Args:
        directory_path: Path to directory containing documents
        
    Returns:
        Ingestion status and statistics
    """
    errors = []
    documents_processed = 0
    chunks_created = 0
    
    try:
        # Validate directory path
        if not os.path.exists(directory_path):
            raise HTTPException(status_code=400, detail="Directory does not exist")
        
        if not os.path.isdir(directory_path):
            raise HTTPException(status_code=400, detail="Path is not a directory")
        
        # Load all documents from directory
        documents = document_loader.load_directory(directory_path, recursive=True)
        
        if not documents:
            return IngestResponse(
                success=True,
                message="No supported documents found in directory",
                documents_processed=0,
                chunks_created=0,
                errors=[]
            )
        
        # Process each document
        all_chunks = []
        for doc in documents:
            try:
                chunks = document_chunker.chunk_document(doc)
                all_chunks.extend(chunks)
                documents_processed += 1
            except Exception as e:
                error_msg = f"Failed to chunk document {doc.get('metadata', {}).get('filename', 'unknown')}: {str(e)}"
                errors.append(error_msg)
                app_logger.error(error_msg)
        
        # Add all chunks to index
        if all_chunks:
            success = index_manager.add_documents(all_chunks)
            if success:
                chunks_created = len(all_chunks)
            else:
                errors.append("Failed to add chunks to index")
        
        app_logger.info(f"Directory ingestion completed: {documents_processed} docs -> {chunks_created} chunks")
        
        return IngestResponse(
            success=len(errors) == 0,
            message=f"Processed {documents_processed} documents from {directory_path}",
            documents_processed=documents_processed,
            chunks_created=chunks_created,
            errors=errors
        )
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Directory ingestion error: {str(e)}")
        return IngestResponse(
            success=False,
            message=f"Failed to process directory {directory_path}",
            documents_processed=documents_processed,
            chunks_created=chunks_created,
            errors=[str(e)]
        )

@router.get("/stats")
async def get_stats():
    """Get comprehensive pipeline statistics."""
    try:
        pipeline_stats = rag_pipeline.get_pipeline_stats()
        return {
            "pipeline_stats": pipeline_stats,
            "app_config": {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "max_chunks_per_query": settings.max_chunks_per_query,
                "embedding_model": settings.embedding_model,
                "llm_provider": settings.llm_provider
            }
        }
    except Exception as e:
        app_logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@router.get("/test")
async def test_pipeline():
    """Test the complete RAG pipeline."""
    try:
        test_results = rag_pipeline.test_pipeline()
        return test_results
    except Exception as e:
        app_logger.error(f"Pipeline test error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline test failed: {str(e)}")

@router.delete("/clear")
async def clear_index():
    """Clear all documents from the index."""
    try:
        success = index_manager.clear_collection()
        if success:
            return {"message": "Index cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear index")
    except Exception as e:
        app_logger.error(f"Error clearing index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear index: {str(e)}")

# Import time for processing time calculation
import time

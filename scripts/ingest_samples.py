#!/usr/bin/env python3
"""
Script to ingest sample documents for demonstration.
Processes all sample documents and makes them available for search.
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.core.logger import app_logger
from backend.ingestion.loader import DocumentLoader
from backend.ingestion.chunker import DocumentChunker
from backend.embedding.index_manager import IndexManager

def ingest_sample_documents():
    """Ingest all sample documents from the samples directory."""
    try:
        app_logger.info("Starting sample document ingestion")
        
        # Get samples directory
        samples_dir = Path(__file__).parent / "data" / "samples"
        
        if not samples_dir.exists():
            app_logger.error(f"Samples directory not found: {samples_dir}")
            return False
        
        # Initialize components
        loader = DocumentLoader()
        chunker = DocumentChunker()
        index_manager = IndexManager()
        
        # Load all sample documents
        documents = loader.load_directory(str(samples_dir), recursive=False)
        
        if not documents:
            app_logger.warning("No sample documents found")
            return False
        
        app_logger.info(f"Found {len(documents)} sample documents")
        
        # Process documents
        total_chunks = 0
        processed_docs = 0
        
        for doc in documents:
            try:
                # Chunk document
                chunks = chunker.chunk_document(doc)
                
                if chunks:
                    # Add to index
                    success = index_manager.add_documents(chunks)
                    if success:
                        total_chunks += len(chunks)
                        processed_docs += 1
                        app_logger.info(f"Processed: {doc['metadata']['filename']} -> {len(chunks)} chunks")
                    else:
                        app_logger.error(f"Failed to index: {doc['metadata']['filename']}")
                else:
                    app_logger.warning(f"No chunks created for: {doc['metadata']['filename']}")
                    
            except Exception as e:
                app_logger.error(f"Error processing {doc['metadata']['filename']}: {str(e)}")
        
        app_logger.info(f"Sample ingestion completed: {processed_docs}/{len(documents)} documents, {total_chunks} total chunks")
        
        # Show collection stats
        stats = index_manager.get_collection_stats()
        app_logger.info(f"Collection stats: {stats}")
        
        return processed_docs > 0
        
    except Exception as e:
        app_logger.error(f"Sample ingestion failed: {str(e)}")
        return False

def main():
    """Main function."""
    print("🔍 AI Knowledge Search Platform - Sample Document Ingestion")
    print("=" * 60)
    
    success = ingest_sample_documents()
    
    if success:
        print("\n✅ Sample documents ingested successfully!")
        print("📚 You can now search through the sample documents")
        print("🚀 Start the Streamlit UI to begin searching")
    else:
        print("\n❌ Sample document ingestion failed")
        print("🔧 Check the logs for detailed error information")
        sys.exit(1)

if __name__ == "__main__":
    main()

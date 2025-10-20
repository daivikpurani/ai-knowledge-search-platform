#!/usr/bin/env python3
"""
CLI script for ingesting documents into the AI Knowledge Search Platform.
Supports batch processing of documents from directories.
"""

import argparse
import sys
from pathlib import Path
import time

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.core.logger import app_logger
from backend.ingestion.loader import DocumentLoader
from backend.ingestion.chunker import DocumentChunker
from backend.embedding.index_manager import IndexManager

def ingest_documents(directory_path: str, recursive: bool = True) -> bool:
    """
    Ingest all documents from a directory.
    
    Args:
        directory_path: Path to directory containing documents
        recursive: Whether to search subdirectories
        
    Returns:
        True if successful, False otherwise
    """
    try:
        app_logger.info(f"Starting document ingestion from: {directory_path}")
        
        # Initialize components
        loader = DocumentLoader()
        chunker = DocumentChunker()
        index_manager = IndexManager()
        
        # Load documents
        documents = loader.load_directory(directory_path, recursive=recursive)
        
        if not documents:
            app_logger.warning("No supported documents found")
            return False
        
        app_logger.info(f"Found {len(documents)} documents to process")
        
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
        
        app_logger.info(f"Ingestion completed: {processed_docs}/{len(documents)} documents, {total_chunks} total chunks")
        
        # Show collection stats
        stats = index_manager.get_collection_stats()
        app_logger.info(f"Collection stats: {stats}")
        
        return processed_docs > 0
        
    except Exception as e:
        app_logger.error(f"Ingestion failed: {str(e)}")
        return False

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into AI Knowledge Search Platform"
    )
    
    parser.add_argument(
        "directory",
        help="Directory containing documents to ingest"
    )
    
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories recursively"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Validate directory
    directory_path = Path(args.directory)
    if not directory_path.exists():
        print(f"Error: Directory '{args.directory}' does not exist")
        sys.exit(1)
    
    if not directory_path.is_dir():
        print(f"Error: '{args.directory}' is not a directory")
        sys.exit(1)
    
    # Start ingestion
    start_time = time.time()
    
    success = ingest_documents(
        directory_path=str(directory_path),
        recursive=not args.no_recursive
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    if success:
        print(f"\n✅ Ingestion completed successfully in {duration:.2f} seconds")
        print(f"📁 Directory: {args.directory}")
        print(f"🔍 Recursive: {not args.no_recursive}")
        print(f"⏱️  Duration: {duration:.2f}s")
    else:
        print(f"\n❌ Ingestion failed after {duration:.2f} seconds")
        sys.exit(1)

if __name__ == "__main__":
    main()

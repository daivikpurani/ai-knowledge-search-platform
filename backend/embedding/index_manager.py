"""
ChromaDB index manager for storing and retrieving document embeddings.
Handles vector database operations and metadata management.
"""

import uuid
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from ..core.config import settings
from ..core.logger import app_logger
from .embedder import EmbeddingGenerator

class IndexManager:
    """Manages ChromaDB vector index for document embeddings."""
    
    def __init__(self, collection_name: str = "knowledge_search"):
        self.collection_name = collection_name
        self.logger = app_logger
        self.client = None
        self.collection = None
        self.embedder = EmbeddingGenerator()
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                self.logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "AI Knowledge Search Platform document embeddings"}
                )
                self.logger.info(f"Created new collection: {self.collection_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector index.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
            
        Returns:
            True if successful, False otherwise
        """
        if not documents:
            self.logger.warning("No documents provided for indexing")
            return False
        
        try:
            # Extract texts and metadata
            texts = []
            metadatas = []
            ids = []
            
            for doc in documents:
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                
                if not content.strip():
                    self.logger.warning("Skipping document with empty content")
                    continue
                
                # Generate unique ID
                doc_id = str(uuid.uuid4())
                
                # Prepare metadata for ChromaDB
                chroma_metadata = self._prepare_metadata(metadata)
                
                texts.append(content)
                metadatas.append(chroma_metadata)
                ids.append(doc_id)
            
            if not texts:
                self.logger.warning("No valid texts found for indexing")
                return False
            
            # Add to collection
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"Successfully indexed {len(texts)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding documents to index: {str(e)}")
            return False
    
    def search_similar(self, query: str, n_results: int = 10, 
                      filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of similar documents with scores
        """
        try:
            # Perform semantic search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance
                    
                    formatted_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'score': similarity_score,
                        'rank': i + 1
                    })
            
            self.logger.info(f"Found {len(formatted_results)} similar documents for query")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error searching similar documents: {str(e)}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by its ID."""
        try:
            results = self.collection.get(ids=[doc_id])
            
            if results['documents'] and results['documents'][0]:
                return {
                    'content': results['documents'][0],
                    'metadata': results['metadatas'][0] if results['metadatas'] else {},
                    'id': doc_id
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting document by ID {doc_id}: {str(e)}")
            return None
    
    def update_document(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> bool:
        """Update an existing document."""
        try:
            chroma_metadata = self._prepare_metadata(metadata)
            
            self.collection.update(
                ids=[doc_id],
                documents=[content],
                metadatas=[chroma_metadata]
            )
            
            self.logger.info(f"Updated document: {doc_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating document {doc_id}: {str(e)}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the index."""
        try:
            self.collection.delete(ids=[doc_id])
            self.logger.info(f"Deleted document: {doc_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'embedding_dimension': self.embedder.get_embedding_dimension()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {str(e)}")
            return {'error': str(e)}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "AI Knowledge Search Platform document embeddings"}
            )
            
            self.logger.info("Cleared collection and recreated")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing collection: {str(e)}")
            return False
    
    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB storage."""
        # ChromaDB has restrictions on metadata values
        prepared = {}
        
        for key, value in metadata.items():
            # Convert to string if not already
            if isinstance(value, (str, int, float, bool)):
                prepared[key] = str(value)
            elif isinstance(value, list):
                # Convert list to string representation
                prepared[key] = str(value)
            else:
                # Convert other types to string
                prepared[key] = str(value)
        
        return prepared

"""
Text chunking utilities for splitting documents into manageable pieces.
Implements smart chunking with overlap and semantic boundaries.
"""

import re
from typing import List, Dict, Any
from ..core.config import settings
from ..core.logger import app_logger
from ..core.utils import count_tokens

class DocumentChunker:
    """Splits documents into chunks with overlap for better retrieval."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.logger = app_logger
        
        # Ensure overlap is not larger than chunk size
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = self.chunk_size // 4
            self.logger.warning(f"Chunk overlap adjusted to {self.chunk_overlap}")
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a document into chunks with metadata.
        
        Args:
            document: Document dictionary with 'content' and 'metadata'
            
        Returns:
            List of chunk dictionaries
        """
        content = document.get("content", "")
        metadata = document.get("metadata", {})
        
        if not content.strip():
            self.logger.warning("Empty document content, skipping chunking")
            return []
        
        # Split content into sentences first
        sentences = self._split_into_sentences(content)
        
        # Create chunks from sentences
        chunks = self._create_chunks_from_sentences(sentences, metadata)
        
        self.logger.info(f"Created {len(chunks)} chunks from document: {metadata.get('filename', 'unknown')}")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        # Enhanced sentence splitting regex
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter out very short fragments
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _create_chunks_from_sentences(self, sentences: List[str], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks from sentences with overlap."""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk_metadata = self._create_chunk_metadata(chunk_text, metadata, len(chunks))
                chunks.append({
                    "content": chunk_text,
                    "metadata": chunk_metadata
                })
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk if it has content
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_metadata = self._create_chunk_metadata(chunk_text, metadata, len(chunks))
            chunks.append({
                "content": chunk_text,
                "metadata": chunk_metadata
            })
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap from the end of current chunk."""
        if not sentences:
            return []
        
        overlap_text = ' '.join(sentences)
        overlap_length = 0
        overlap_sentences = []
        
        # Add sentences from the end until we reach overlap size
        for sentence in reversed(sentences):
            if overlap_length + len(sentence) <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_length += len(sentence)
            else:
                break
        
        return overlap_sentences
    
    def _create_chunk_metadata(self, chunk_text: str, doc_metadata: Dict[str, Any], chunk_index: int) -> Dict[str, Any]:
        """Create metadata for a chunk."""
        chunk_metadata = doc_metadata.copy()
        chunk_metadata.update({
            "chunk_index": chunk_index,
            "chunk_length": len(chunk_text),
            "chunk_tokens": count_tokens(chunk_text),
            "is_chunk": True
        })
        
        # Add chunk-specific metadata
        chunk_metadata["chunk_id"] = f"{doc_metadata.get('filename', 'doc')}_chunk_{chunk_index}"
        
        return chunk_metadata
    
    def chunk_by_tokens(self, document: Dict[str, Any], max_tokens: int = None) -> List[Dict[str, Any]]:
        """
        Alternative chunking method based on token count instead of character count.
        
        Args:
            document: Document dictionary
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of token-based chunks
        """
        max_tokens = max_tokens or settings.max_context_tokens // 4  # Conservative chunk size
        content = document.get("content", "")
        metadata = document.get("metadata", {})
        
        sentences = self._split_into_sentences(content)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = count_tokens(sentence)
            
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk_metadata = self._create_chunk_metadata(chunk_text, metadata, len(chunks))
                chunk_metadata["chunk_method"] = "token_based"
                chunks.append({
                    "content": chunk_text,
                    "metadata": chunk_metadata
                })
                
                # Start new chunk
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_metadata = self._create_chunk_metadata(chunk_text, metadata, len(chunks))
            chunk_metadata["chunk_method"] = "token_based"
            chunks.append({
                "content": chunk_text,
                "metadata": chunk_metadata
            })
        
        return chunks

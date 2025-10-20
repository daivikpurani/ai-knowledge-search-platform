"""
Utility functions for the AI Knowledge Search Platform.
"""

import hashlib
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import tiktoken

def generate_document_id(file_path: str, content: str) -> str:
    """Generate a unique document ID based on file path and content hash."""
    file_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    file_name = Path(file_path).stem
    return f"{file_name}_{file_hash}"

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    return text.strip()

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # Fallback to cl100k_base encoding
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

def extract_metadata_from_filename(file_path: str) -> Dict[str, Any]:
    """Extract metadata from filename."""
    path = Path(file_path)
    
    metadata = {
        "filename": path.name,
        "file_extension": path.suffix.lower(),
        "file_size": path.stat().st_size if path.exists() else 0,
        "directory": str(path.parent),
    }
    
    # Extract potential document type from filename
    filename_lower = path.stem.lower()
    if any(keyword in filename_lower for keyword in ["manual", "guide", "doc"]):
        metadata["document_type"] = "manual"
    elif any(keyword in filename_lower for keyword in ["paper", "research", "study"]):
        metadata["document_type"] = "research"
    elif any(keyword in filename_lower for keyword in ["report", "analysis"]):
        metadata["document_type"] = "report"
    elif any(keyword in filename_lower for keyword in ["faq", "help", "support"]):
        metadata["document_type"] = "faq"
    else:
        metadata["document_type"] = "general"
    
    return metadata

def validate_query(query: str) -> bool:
    """Validate if query is suitable for processing."""
    if not query or not query.strip():
        return False
    
    # Check minimum length
    if len(query.strip()) < 3:
        return False
    
    # Check for excessive length
    if len(query) > 1000:
        return False
    
    return True

def format_search_results(results: List[Dict[str, Any]], max_results: int = 10) -> List[Dict[str, Any]]:
    """Format search results for consistent output."""
    formatted_results = []
    
    for i, result in enumerate(results[:max_results]):
        formatted_result = {
            "rank": i + 1,
            "content": result.get("content", ""),
            "metadata": result.get("metadata", {}),
            "score": result.get("score", 0.0),
            "source": result.get("metadata", {}).get("source", "unknown")
        }
        formatted_results.append(formatted_result)
    
    return formatted_results

def calculate_similarity_score(text1: str, text2: str) -> float:
    """Calculate simple text similarity using Jaccard similarity."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

"""
Document loader for various file formats (PDF, Markdown, Text).
Handles file parsing and content extraction.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import PyPDF2
import markdown
from ..core.logger import app_logger
from ..core.utils import clean_text, extract_metadata_from_filename

class DocumentLoader:
    """Loads and parses documents from various file formats."""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.md', '.txt', '.markdown'}
    
    def __init__(self):
        self.logger = app_logger
    
    def load_document(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load a single document and extract its content.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing document content and metadata, or None if failed
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                self.logger.error(f"File not found: {file_path}")
                return None
            
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                self.logger.warning(f"Unsupported file format: {file_path.suffix}")
                return None
            
            # Extract content based on file type
            content = self._extract_content(file_path)
            if not content:
                self.logger.error(f"Failed to extract content from: {file_path}")
                return None
            
            # Extract metadata
            metadata = extract_metadata_from_filename(str(file_path))
            metadata.update({
                "source": str(file_path),
                "content_length": len(content),
                "file_type": file_path.suffix.lower()
            })
            
            self.logger.info(f"Successfully loaded document: {file_path.name}")
            
            return {
                "content": content,
                "metadata": metadata,
                "file_path": str(file_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error loading document {file_path}: {str(e)}")
            return None
    
    def load_directory(self, directory_path: str, recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search subdirectories
            
        Returns:
            List of loaded documents
        """
        directory = Path(directory_path)
        if not directory.exists():
            self.logger.error(f"Directory not found: {directory_path}")
            return []
        
        documents = []
        
        # Find all supported files
        pattern = "**/*" if recursive else "*"
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                doc = self.load_document(file_path)
                if doc:
                    documents.append(doc)
        
        self.logger.info(f"Loaded {len(documents)} documents from {directory_path}")
        return documents
    
    def _extract_content(self, file_path: Path) -> Optional[str]:
        """Extract content from a file based on its extension."""
        try:
            if file_path.suffix.lower() == '.pdf':
                return self._extract_pdf_content(file_path)
            elif file_path.suffix.lower() in ['.md', '.markdown']:
                return self._extract_markdown_content(file_path)
            elif file_path.suffix.lower() == '.txt':
                return self._extract_text_content(file_path)
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error extracting content from {file_path}: {str(e)}")
            return None
    
    def _extract_pdf_content(self, file_path: Path) -> str:
        """Extract text content from PDF file."""
        content = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        content.append(f"[Page {page_num + 1}]\n{page_text}")
                except Exception as e:
                    self.logger.warning(f"Error extracting page {page_num + 1} from {file_path}: {str(e)}")
        
        return clean_text('\n\n'.join(content))
    
    def _extract_markdown_content(self, file_path: Path) -> str:
        """Extract text content from Markdown file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            markdown_content = file.read()
        
        # Convert markdown to plain text
        html = markdown.markdown(markdown_content)
        
        # Simple HTML tag removal (for basic markdown)
        import re
        text = re.sub(r'<[^>]+>', '', html)
        
        return clean_text(text)
    
    def _extract_text_content(self, file_path: Path) -> str:
        """Extract content from plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return clean_text(content)
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                return clean_text(content)
            except Exception as e:
                self.logger.error(f"Failed to read text file {file_path}: {str(e)}")
                return ""

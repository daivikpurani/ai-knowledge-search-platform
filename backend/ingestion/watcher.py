"""
Document watcher for monitoring file system changes.
Automatically processes new documents when they are added.
"""

import time
from pathlib import Path
from typing import Callable, List, Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ..core.logger import app_logger
from .loader import DocumentLoader

class DocumentWatcher(FileSystemEventHandler):
    """Watches for file system changes and triggers document processing."""
    
    def __init__(self, callback: Callable[[str], None], supported_extensions: Set[str] = None):
        self.callback = callback
        self.supported_extensions = supported_extensions or {'.pdf', '.md', '.txt', '.markdown'}
        self.logger = app_logger
        self.processed_files: Set[str] = set()
    
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            self._process_file(event.src_path)
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            self._process_file(event.src_path)
    
    def _process_file(self, file_path: str):
        """Process a file if it's supported and not already processed."""
        path = Path(file_path)
        
        if path.suffix.lower() in self.supported_extensions:
            if file_path not in self.processed_files:
                self.logger.info(f"New file detected: {file_path}")
                self.processed_files.add(file_path)
                
                # Small delay to ensure file is fully written
                time.sleep(1)
                
                try:
                    self.callback(file_path)
                except Exception as e:
                    self.logger.error(f"Error processing file {file_path}: {str(e)}")

class DocumentWatcherManager:
    """Manages document watchers for multiple directories."""
    
    def __init__(self):
        self.observers: List[Observer] = []
        self.logger = app_logger
    
    def start_watching(self, directory_path: str, callback: Callable[[str], None]):
        """Start watching a directory for document changes."""
        directory = Path(directory_path)
        if not directory.exists():
            self.logger.error(f"Directory does not exist: {directory_path}")
            return
        
        observer = Observer()
        event_handler = DocumentWatcher(callback)
        observer.schedule(event_handler, directory_path, recursive=True)
        observer.start()
        
        self.observers.append(observer)
        self.logger.info(f"Started watching directory: {directory_path}")
    
    def stop_all(self):
        """Stop all active watchers."""
        for observer in self.observers:
            observer.stop()
            observer.join()
        
        self.observers.clear()
        self.logger.info("Stopped all document watchers")
    
    def is_watching(self) -> bool:
        """Check if any watchers are active."""
        return len(self.observers) > 0

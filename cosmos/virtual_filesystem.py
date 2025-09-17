"""
Intelligent Virtual File System for Redis-backed repository access.

This module provides a virtual file system that uses an indexing system
to efficiently access files from content.md without parsing the entire file.
Uses indexing_tree.txt for fast file lookups.
"""

import os
import re
import time
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple, Any
import logging

try:
    from .content_indexer import ContentIndexer
except ImportError:
    from content_indexer import ContentIndexer

logger = logging.getLogger(__name__)


class IntelligentVirtualFileSystem:
    """
    Creates a smart virtual directory structure that efficiently accesses files
    using an indexing system instead of parsing the entire content.md file.
    """
    
    def __init__(self, content_md: str, tree_txt: str, repo_name: str = "", repo_storage_dir: str = None):
        """
        Initialize the virtual file system with indexed access.
        
        Args:
            content_md: The complete content.md file content (for fallback)
            tree_txt: The tree.txt file content showing directory structure
            repo_name: Name of the repository for logging/debugging
            repo_storage_dir: Directory where repository files are stored (for indexing)
        """
        self.content_md = content_md
        self.tree_txt = tree_txt
        self.repo_name = repo_name
        self.repo_storage_dir = repo_storage_dir
        
        # Indexing system
        self.indexer = None
        if repo_storage_dir and repo_name:
            self.indexer = ContentIndexer(repo_storage_dir, repo_name)
            self.indexer.ensure_index()
        
        # Parsed data structures (for fallback and metadata)
        self._files = {}  # file_path -> file_content (cached)
        self._file_metadata = {}  # file_path -> metadata dict
        self._directory_structure = {}  # Hierarchical directory structure
        self._tracked_files = set()  # All files that exist in the repo
        
        # Initialize the system
        self._initialize_filesystem()
        
        logger.info(f"Initialized virtual filesystem for {repo_name} with indexing: {self.indexer is not None}")
    
    def _initialize_filesystem(self) -> None:
        """
        Initialize the filesystem using indexing or fallback to full parsing.
        """
        if self.indexer and self.indexer.is_index_valid():
            # Use indexing system
            self._tracked_files = set(self.indexer.get_all_files())
            logger.info(f"Using indexed access for {len(self._tracked_files)} files")
        else:
            # Fallback to full parsing
            logger.info(f"Using fallback parsing for {self.repo_name}")
            self._parse_content_md()
        
        # Always parse tree structure for directory operations
        self._parse_tree_txt()
        self._generate_file_metadata()
    
    def smart_parse_content(self) -> Dict[str, str]:
        """
        Intelligent parsing that understands cosmos file access patterns.
        Uses indexing for efficient access when available.
        
        Returns:
            Dictionary mapping file paths to their content
        """
        if self.indexer:
            # Return all files from index (lazy loading)
            return {file_path: "" for file_path in self._tracked_files}
        else:
            # Return cached files from full parsing
            return self._files.copy()
    
    def _parse_content_md(self) -> None:
        """
        Advanced algorithm to parse content.md and extract individual file boundaries.
        
        The content.md format uses:
        ================================================
        FILE: path/to/file.ext
        ================================================
        [file content]
        
        This method handles various edge cases and encoding issues.
        """
        if not self.content_md:
            logger.warning(f"Empty content.md for {self.repo_name}")
            return
            
        # Split by file boundary markers
        file_boundary_pattern = r'^={48}\nFILE: (.+?)\n={48}\n'
        
        # Split content into sections
        sections = re.split(file_boundary_pattern, self.content_md, flags=re.MULTILINE)
        
        # First section is usually empty or contains header info
        if len(sections) < 3:
            logger.warning(f"Invalid content.md format for {self.repo_name}")
            return
            
        # Process file sections (skip first empty section)
        for i in range(1, len(sections), 2):
            if i + 1 >= len(sections):
                break
                
            file_path = sections[i].strip()
            file_content = sections[i + 1]
            
            # Clean up file path (remove any leading/trailing whitespace)
            file_path = file_path.strip()
            
            # Handle encoding issues and normalize content
            try:
                # Remove any trailing newlines that might be artifacts
                if file_content.endswith('\n\n'):
                    file_content = file_content.rstrip('\n') + '\n'
                elif not file_content.endswith('\n') and file_content:
                    # Ensure files end with newline for consistency
                    file_content += '\n'
                    
                self._files[file_path] = file_content
                self._tracked_files.add(file_path)
                
            except Exception as e:
                logger.error(f"Error processing file {file_path} in {self.repo_name}: {e}")
                # Store as-is if there are encoding issues
                self._files[file_path] = file_content
                self._tracked_files.add(file_path)
        
        logger.info(f"Parsed {len(self._files)} files from content.md")
    

    
    def get_file_content(self, file_path: str) -> Optional[str]:
        """
        Get file content - alias for extract_file_with_context for compatibility.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string, or None if file not found
        """
        return self.extract_file_with_context(file_path)
    
    def file_exists(self, file_path: str) -> bool:
        """
        Check if file exists using efficient indexing.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file exists
        """
        # Normalize path separators
        normalized_path = file_path.replace('\\', '/')
        
        if self.indexer:
            # Try exact match first
            if self.indexer.file_exists(normalized_path):
                return True
            
            # Try without leading slash
            if normalized_path.startswith('/'):
                alt_path = normalized_path[1:]
                if self.indexer.file_exists(alt_path):
                    return True
            
            # Try with leading slash
            if not normalized_path.startswith('/'):
                alt_path = '/' + normalized_path
                if self.indexer.file_exists(alt_path):
                    return True
        
        # Fallback to cached files
        return (normalized_path in self._files or 
                (normalized_path.startswith('/') and normalized_path[1:] in self._files) or
                (not normalized_path.startswith('/') and '/' + normalized_path in self._files))
    
    def get_tracked_files(self) -> List[str]:
        """
        Get list of all tracked files efficiently.
        
        Returns:
            List of file paths
        """
        if self.indexer:
            return self.indexer.get_all_files()
        
        return list(self._tracked_files)
    
    def _parse_tree_txt(self) -> None:
        """
        Create smart directory structure parsing from tree.txt with cosmos-compatible format.
        
        Parses the tree structure and creates a hierarchical representation
        that matches cosmos expectations for directory traversal.
        """
        if not self.tree_txt:
            logger.warning(f"Empty tree.txt for {self.repo_name}")
            return
            
        lines = self.tree_txt.split('\n')
        current_path_stack = []
        
        # Find the root directory name
        root_dir = None
        for line in lines:
            if line.strip() and not line.startswith(' ') and '/' in line:
                root_dir = line.strip().rstrip('/')
                break
        
        if not root_dir:
            logger.warning(f"Could not find root directory in tree.txt for {self.repo_name}")
            return
            
        self._directory_structure = {
            'name': root_dir,
            'type': 'directory',
            'children': {},
            'files': [],
            'path': ''
        }
        
        for line in lines:
            if not line.strip():
                continue
                
            # Skip the "Directory structure:" header and root directory line
            if 'Directory structure:' in line or (line.strip() == root_dir or line.strip() == f"└── {root_dir}/"):
                continue
            
            # Parse indentation level and content
            indent_match = re.match(r'^(\s*[│├└─\s]*)', line)
            if not indent_match:
                continue
                
            indent = indent_match.group(1)
            content = line[len(indent):].strip()
            
            if not content:
                continue
            
            # Calculate depth based on tree characters
            depth = len(re.findall(r'[├└]', indent))
            
            # Clean up the content (remove trailing /)
            is_directory = content.endswith('/')
            clean_name = content.rstrip('/')
            
            # Build the full path
            if depth == 0:
                continue  # Skip root level
            
            # Adjust path stack to current depth
            while len(current_path_stack) >= depth:
                current_path_stack.pop()
            
            current_path_stack.append(clean_name)
            full_path = '/'.join(current_path_stack)
            
            if is_directory:
                # Add directory to structure
                self._add_directory_to_structure(current_path_stack.copy())
            else:
                # Add file to structure
                self._add_file_to_structure(current_path_stack.copy())
        
        logger.info(f"Parsed directory structure with {len(self._get_all_directories())} directories")
    
    def _add_directory_to_structure(self, path_parts: List[str]) -> None:
        """Add a directory to the hierarchical structure."""
        current = self._directory_structure
        
        for part in path_parts:
            if part not in current['children']:
                current['children'][part] = {
                    'name': part,
                    'type': 'directory',
                    'children': {},
                    'files': [],
                    'path': '/'.join(path_parts[:path_parts.index(part) + 1])
                }
            current = current['children'][part]
    
    def _add_file_to_structure(self, path_parts: List[str]) -> None:
        """Add a file to the hierarchical structure."""
        if not path_parts:
            return
            
        file_name = path_parts[-1]
        dir_parts = path_parts[:-1]
        
        # Ensure parent directories exist
        if dir_parts:
            self._add_directory_to_structure(dir_parts)
        
        # Find parent directory
        current = self._directory_structure
        for part in dir_parts:
            current = current['children'][part]
        
        # Add file to parent directory
        if file_name not in current['files']:
            current['files'].append(file_name)
    
    def _generate_file_metadata(self) -> None:
        """
        Generate file metadata that matches cosmos expectations.
        
        Creates metadata for each file including size, mtime, and git status
        to simulate a real file system.
        """
        current_time = time.time()
        
        for file_path, content in self._files.items():
            # Calculate file size
            size = len(content.encode('utf-8'))
            
            # Generate realistic mtime (slightly randomized but consistent)
            # Use hash of file path for consistency
            path_hash = hash(file_path) % 86400  # Within 24 hours
            mtime = current_time - path_hash
            
            self._file_metadata[file_path] = {
                'size': size,
                'mtime': mtime,
                'git_status': 'tracked',  # All files in content.md are tracked
                'exists': True,
                'is_file': True,
                'is_dir': False,
                'path': file_path
            }
    
    def extract_file_with_context(self, file_path: str) -> Optional[str]:
        """
        Extract file content efficiently using indexing system.
        
        Args:
            file_path: Path to the file to extract
            
        Returns:
            File content as string, or None if file not found
        """
        # Normalize path separators
        normalized_path = file_path.replace('\\', '/')
        
        # First try indexed access
        if self.indexer:
            content = self.indexer.get_file_content(normalized_path)
            if content is not None:
                # Cache the content for future access
                self._files[normalized_path] = content
                return content
        
        # Try exact match first in cached files
        if normalized_path in self._files:
            return self._files[normalized_path]
        
        # Try without leading slash
        if normalized_path.startswith('/'):
            alt_path = normalized_path[1:]
            if alt_path in self._files:
                return self._files[alt_path]
            # Also try indexed access with alt_path
            if self.indexer:
                content = self.indexer.get_file_content(alt_path)
                if content is not None:
                    self._files[alt_path] = content
                    return content
        
        # Try with leading slash
        if not normalized_path.startswith('/'):
            alt_path = '/' + normalized_path
            if alt_path in self._files:
                return self._files[alt_path]
            # Also try indexed access with alt_path
            if self.indexer:
                content = self.indexer.get_file_content(alt_path)
                if content is not None:
                    self._files[alt_path] = content
                    return content
        
        logger.warning(f"File not found: {file_path} in {self.repo_name}")
        return None
    
    def get_cosmos_compatible_tree(self) -> Dict[str, Any]:
        """
        Get tree structure exactly as cosmos expects.
        
        Returns:
            Dictionary representing the directory tree in cosmos format
        """
        return self._directory_structure
    
    def resolve_cosmos_path(self, relative_path: str) -> str:
        """
        Path resolution matching cosmos behavior.
        
        Args:
            relative_path: Relative path to resolve
            
        Returns:
            Resolved absolute path in virtual filesystem
        """
        # Normalize path separators
        normalized = relative_path.replace('\\', '/')
        
        # Remove leading slash if present
        if normalized.startswith('/'):
            normalized = normalized[1:]
        
        return normalized
    
    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get file stats that cosmos expects.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file metadata (size, mtime, etc.)
        """
        normalized_path = self.resolve_cosmos_path(file_path)
        
        if normalized_path in self._file_metadata:
            return self._file_metadata[normalized_path].copy()
        
        # Return default metadata for non-existent files
        return {
            'size': 0,
            'mtime': 0,
            'git_status': 'untracked',
            'exists': False,
            'is_file': False,
            'is_dir': False,
            'path': file_path
        }
    
    def simulate_git_operations(self) -> Dict[str, Any]:
        """
        Simulate git-like operations cosmos uses.
        
        Returns:
            Dictionary with git operation simulation data
        """
        return {
            'tracked_files': list(self._tracked_files),
            'ignored_patterns': ['.git/', '__pycache__/', '*.pyc', '.DS_Store'],
            'repo_root': f'/virtual/{self.repo_name}',
            'is_git_repo': True,
            'has_uncommitted_changes': False,
            'current_branch': 'main'
        }
    
    def get_tracked_files(self) -> List[str]:
        """
        Get list of all tracked files.
        
        Returns:
            List of file paths that are tracked in the repository
        """
        return list(self._tracked_files)
    

    
    def is_directory(self, path: str) -> bool:
        """
        Check if a path is a directory.
        
        Args:
            path: Path to check
            
        Returns:
            True if path is a directory, False otherwise
        """
        normalized_path = self.resolve_cosmos_path(path)
        
        # Check if any file starts with this path + /
        for file_path in self._files:
            if file_path.startswith(normalized_path + '/'):
                return True
        
        return False
    
    def list_directory(self, dir_path: str = "") -> List[str]:
        """
        List contents of a directory.
        
        Args:
            dir_path: Directory path to list (empty for root)
            
        Returns:
            List of file and directory names in the directory
        """
        normalized_path = self.resolve_cosmos_path(dir_path)
        
        if normalized_path and not normalized_path.endswith('/'):
            normalized_path += '/'
        
        contents = set()
        
        for file_path in self._files:
            if normalized_path == "" or file_path.startswith(normalized_path):
                # Get the relative part after the directory path
                relative_part = file_path[len(normalized_path):] if normalized_path else file_path
                
                # Get the first component (file or directory name)
                if '/' in relative_part:
                    # It's in a subdirectory
                    first_component = relative_part.split('/')[0]
                    contents.add(first_component)
                else:
                    # It's a direct file
                    contents.add(relative_part)
        
        return sorted(list(contents))
    
    def _get_all_directories(self) -> List[str]:
        """Get all directory paths in the virtual filesystem."""
        directories = set()
        
        for file_path in self._files:
            parts = file_path.split('/')
            for i in range(len(parts) - 1):
                dir_path = '/'.join(parts[:i + 1])
                directories.add(dir_path)
        
        return list(directories)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the virtual filesystem.
        
        Returns:
            Dictionary with filesystem statistics
        """
        return {
            'total_files': len(self._files),
            'total_directories': len(self._get_all_directories()),
            'total_size': sum(len(content.encode('utf-8')) for content in self._files.values()),
            'repo_name': self.repo_name
        }
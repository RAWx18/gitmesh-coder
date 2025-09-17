"""
Virtual IO wrapper for seamless Redis integration.

This module provides an IO wrapper that can transparently route file operations
through a virtual file system when working with Redis-backed repositories.
"""

import os
from pathlib import Path
from typing import Optional, Any

from cosmos.io import InputOutput


class VirtualIOWrapper:
    """
    IO wrapper that routes file operations through virtual file system when appropriate.
    
    This wrapper intercepts file read operations and routes them through the
    repository's virtual file system if the file is part of a Redis-backed repository.
    """
    
    def __init__(self, base_io: InputOutput, repo_manager=None):
        """
        Initialize virtual IO wrapper.
        
        Args:
            base_io: Base InputOutput instance
            repo_manager: Repository manager (GitRepo or RedisRepoManager)
        """
        self.base_io = base_io
        self.repo_manager = repo_manager
        
        # Delegate all attributes to base_io by default
        for attr in dir(base_io):
            if not attr.startswith('_') and not hasattr(self, attr):
                setattr(self, attr, getattr(base_io, attr))
    
    def __getattr__(self, name):
        """Delegate unknown attributes to base_io."""
        return getattr(self.base_io, name)
    
    def read_text(self, filename, silent=False):
        """
        Read text file with virtual file system support.
        
        Args:
            filename: Path to file
            silent: Whether to suppress error messages
            
        Returns:
            File content or None if not found
        """
        # Convert to string and normalize path
        filename_str = str(filename)
        
        # Check if we have a Redis-backed repository and the file is tracked
        if (self.repo_manager and 
            hasattr(self.repo_manager, 'virtual_fs') and 
            self.repo_manager.virtual_fs and
            self._is_tracked_file(filename_str)):
            
            try:
                # Use virtual file system for tracked files
                relative_path = self._get_relative_path(filename_str)
                content = self.repo_manager.virtual_fs.get_file_content(relative_path)
                
                if content is not None:
                    return content
                    
            except Exception as e:
                if not silent:
                    self.tool_error(f"Virtual filesystem error for {filename}: {e}")
                # Fall back to regular file system
        
        # Fall back to regular file system read
        return self.base_io.read_text(filename, silent)
    
    def _is_tracked_file(self, filename: str) -> bool:
        """
        Check if file is tracked by the repository.
        
        Args:
            filename: File path to check
            
        Returns:
            True if file is tracked, False otherwise
        """
        if not self.repo_manager or not hasattr(self.repo_manager, 'get_tracked_files'):
            return False
            
        try:
            relative_path = self._get_relative_path(filename)
            tracked_files = self.repo_manager.get_tracked_files()
            return relative_path in tracked_files
        except Exception:
            return False
    
    def _get_relative_path(self, filename: str) -> str:
        """
        Get relative path from repository root.
        
        Args:
            filename: Absolute or relative file path
            
        Returns:
            Relative path from repository root
        """
        if not self.repo_manager or not hasattr(self.repo_manager, 'root'):
            return filename
            
        try:
            file_path = Path(filename).resolve()
            repo_root = Path(self.repo_manager.root).resolve()
            
            # Try to get relative path
            try:
                relative_path = file_path.relative_to(repo_root)
                return str(relative_path)
            except ValueError:
                # File is outside repository, return as-is
                return filename
                
        except Exception:
            return filename


def create_virtual_io(base_io: InputOutput, repo_manager=None) -> InputOutput:
    """
    Create virtual IO wrapper if needed.
    
    Args:
        base_io: Base InputOutput instance
        repo_manager: Repository manager instance
        
    Returns:
        VirtualIOWrapper if Redis repo, otherwise base_io
    """
    # Only wrap if we have a Redis-backed repository
    if (repo_manager and 
        hasattr(repo_manager, 'virtual_fs') and 
        repo_manager.virtual_fs is not None):
        return VirtualIOWrapper(base_io, repo_manager)
    
    return base_io
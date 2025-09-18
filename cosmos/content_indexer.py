"""
Content Indexer for optimized virtual file system access.

This module creates an indexing system that maps file paths to their starting
line numbers in content.md, enabling efficient file access without parsing
the entire content.md file.
"""

import re
import os
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ContentIndexer:
    """
    Creates and manages an index of file locations in content.md for fast access.
    """
    
    def __init__(self, repo_storage_dir: str, repo_name: str):
        """
        Initialize the content indexer.
        
        Args:
            repo_storage_dir: Directory where repository files are stored
            repo_name: Name of the repository
        """
        self.repo_storage_dir = Path(repo_storage_dir)
        self.repo_name = repo_name
        self.repo_dir = self.repo_storage_dir / repo_name
        
        # File paths
        self.content_md_path = self.repo_dir / "content.md"
        self.tree_txt_path = self.repo_dir / "tree.txt"
        self.indexing_tree_path = self.repo_dir / "indexing_tree.txt"
        
        # Index data
        self.file_index: Dict[str, Tuple[int, int]] = {}  # file_path -> (start_line, end_line)
        self.total_lines = 0
        
    def create_index(self) -> bool:
        """
        Create the indexing_tree.txt file by parsing content.md.
        
        Returns:
            True if index was created successfully, False otherwise
        """
        try:
            if not self.content_md_path.exists():
                logger.warning(f"content.md not found for {self.repo_name}")
                return False
            
            logger.info(f"Creating index for {self.repo_name}...")
            
            # Parse content.md to find file boundaries
            self._parse_content_boundaries()
            
            # Write the index file
            self._write_index_file()
            
            logger.info(f"Created index with {len(self.file_index)} files for {self.repo_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index for {self.repo_name}: {e}")
            return False
    
    def _parse_content_boundaries(self) -> None:
        """
        Parse content.md to find file boundaries and line numbers.
        """
        self.file_index.clear()
        
        with open(self.content_md_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        self.total_lines = len(lines)
        
        # File boundary pattern: ================================================
        #                        FILE: path/to/file.ext
        #                        ================================================
        file_boundary_pattern = r'^={48}$'
        file_path_pattern = r'^FILE: (.+)$'
        
        current_file = None
        current_start_line = None
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Check for file boundary
            if re.match(file_boundary_pattern, line):
                # If we were tracking a file, close it
                if current_file and current_start_line:
                    self.file_index[current_file] = (current_start_line, line_num - 1)
                
                # Look for FILE: line in next line
                if line_num < len(lines):
                    next_line = lines[line_num].strip()
                    file_match = re.match(file_path_pattern, next_line)
                    if file_match:
                        current_file = file_match.group(1).strip()
                        # Content starts after the second boundary (line_num + 2)
                        current_start_line = line_num + 2
        
        # Handle the last file
        if current_file and current_start_line:
            self.file_index[current_file] = (current_start_line, self.total_lines)
    
    def _write_index_file(self) -> None:
        """
        Write the indexing_tree.txt file with file path to line number mappings.
        """
        with open(self.indexing_tree_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"# Content Index for {self.repo_name}\n")
            f.write(f"# Format: file_path:start_line:end_line\n")
            f.write(f"# Total lines in content.md: {self.total_lines}\n")
            f.write(f"# Total files indexed: {len(self.file_index)}\n")
            f.write("\n")
            
            # Write file mappings sorted by file path
            for file_path in sorted(self.file_index.keys()):
                start_line, end_line = self.file_index[file_path]
                f.write(f"{file_path}:{start_line}:{end_line}\n")
    
    def load_index(self) -> bool:
        """
        Load the existing indexing_tree.txt file.
        
        Returns:
            True if index was loaded successfully, False otherwise
        """
        try:
            if not self.indexing_tree_path.exists():
                logger.info(f"No existing index found for {self.repo_name}")
                return False
            
            self.file_index.clear()
            
            with open(self.indexing_tree_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if line.startswith('#') or not line:
                        continue
                    
                    # Parse file mapping: file_path:start_line:end_line
                    parts = line.split(':')
                    if len(parts) >= 3:
                        file_path = ':'.join(parts[:-2])  # Handle file paths with colons
                        start_line = int(parts[-2])
                        end_line = int(parts[-1])
                        self.file_index[file_path] = (start_line, end_line)
            
            logger.info(f"Loaded index with {len(self.file_index)} files for {self.repo_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index for {self.repo_name}: {e}")
            return False
    
    def get_file_content(self, file_path: str) -> Optional[str]:
        """
        Get file content using the index for efficient access.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string, or None if file not found
        """
        if file_path not in self.file_index:
            return None
        
        try:
            start_line, end_line = self.file_index[file_path]
            
            with open(self.content_md_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Skip to start line
                for _ in range(start_line - 1):
                    f.readline()
                
                # Read file content
                content_lines = []
                for line_num in range(start_line, end_line + 1):
                    line = f.readline()
                    if not line:  # EOF
                        break
                    content_lines.append(line)
                
                # Join and clean up content
                content = ''.join(content_lines)
                
                # Remove boundary markers and cleanup
                content = self._clean_file_content(content)
                
                return content
                
        except Exception as e:
            logger.error(f"Failed to get content for {file_path} in {self.repo_name}: {e}")
            return None
    
    def _clean_file_content(self, content: str) -> str:
        """
        Clean file content by removing boundary markers and normalizing format.
        
        Args:
            content: Raw file content that may contain boundary markers
            
        Returns:
            Cleaned file content
        """
        if not content:
            return ''
        
        lines = content.splitlines(keepends=True)
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Skip boundary marker lines (48 equals signs)
            if stripped == '=' * 48:
                continue
            # Skip FILE: header lines
            if stripped.startswith('FILE: '):
                continue
            cleaned_lines.append(line)
        
        # Join back and ensure proper line ending
        cleaned_content = ''.join(cleaned_lines)
        
        # Remove any leading/trailing boundary markers that might have been missed
        while cleaned_content.startswith('=' * 48):
            lines = cleaned_content.splitlines(keepends=True)
            if lines:
                cleaned_content = ''.join(lines[1:])
            else:
                break
                
        while cleaned_content.endswith('=' * 48 + '\n'):
            cleaned_content = cleaned_content[:-49]
        
        # Ensure file ends with newline if it has content
        if cleaned_content and not cleaned_content.endswith('\n'):
            cleaned_content += '\n'
            
        return cleaned_content
    
    def file_exists(self, file_path: str) -> bool:
        """
        Check if file exists in the index.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file exists in index
        """
        return file_path in self.file_index
    
    def get_all_files(self) -> List[str]:
        """
        Get list of all files in the index.
        
        Returns:
            List of file paths
        """
        return list(self.file_index.keys())
    
    def get_file_line_range(self, file_path: str) -> Optional[Tuple[int, int]]:
        """
        Get the line range for a file in content.md.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (start_line, end_line) or None if file not found
        """
        return self.file_index.get(file_path)
    
    def is_index_valid(self) -> bool:
        """
        Check if the current index is valid and up-to-date.
        
        Returns:
            True if index is valid, False if it needs to be recreated
        """
        try:
            # Check if all required files exist
            if not all([
                self.content_md_path.exists(),
                self.indexing_tree_path.exists()
            ]):
                return False
            
            # Check if content.md is newer than index
            content_mtime = self.content_md_path.stat().st_mtime
            index_mtime = self.indexing_tree_path.stat().st_mtime
            
            if content_mtime > index_mtime:
                logger.info(f"Index outdated for {self.repo_name}, content.md is newer")
                return False
            
            # Check if index can be loaded
            if not self.file_index:
                return self.load_index()
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating index for {self.repo_name}: {e}")
            return False
    
    def ensure_index(self) -> bool:
        """
        Ensure that a valid index exists, creating it if necessary.
        
        Returns:
            True if valid index is available, False otherwise
        """
        if self.is_index_valid():
            return True
        
        logger.info(f"Creating new index for {self.repo_name}")
        return self.create_index()
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            'repo_name': self.repo_name,
            'total_files': len(self.file_index),
            'total_lines': self.total_lines,
            'index_exists': self.indexing_tree_path.exists(),
            'content_exists': self.content_md_path.exists(),
            'tree_exists': self.tree_txt_path.exists()
        }


def create_repository_index(repo_storage_dir: str, repo_name: str) -> bool:
    """
    Create index for a repository.
    
    Args:
        repo_storage_dir: Directory where repository files are stored
        repo_name: Name of the repository
        
    Returns:
        True if index was created successfully
    """
    indexer = ContentIndexer(repo_storage_dir, repo_name)
    return indexer.ensure_index()


def get_file_content_indexed(repo_storage_dir: str, repo_name: str, file_path: str) -> Optional[str]:
    """
    Get file content using the index system.
    
    Args:
        repo_storage_dir: Directory where repository files are stored
        repo_name: Name of the repository
        file_path: Path to the file
        
    Returns:
        File content as string, or None if file not found
    """
    indexer = ContentIndexer(repo_storage_dir, repo_name)
    
    if not indexer.ensure_index():
        return None
    
    return indexer.get_file_content(file_path)


if __name__ == "__main__":
    """Test the content indexer."""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python content_indexer.py <repo_storage_dir> <repo_name>")
        sys.exit(1)
    
    repo_storage_dir = sys.argv[1]
    repo_name = sys.argv[2]
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create index
    indexer = ContentIndexer(repo_storage_dir, repo_name)
    
    if indexer.ensure_index():
        stats = indexer.get_stats()
        print(f"Index created successfully:")
        print(f"  Repository: {stats['repo_name']}")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Total lines: {stats['total_lines']}")
        
        # Test file access
        files = indexer.get_all_files()
        if files:
            test_file = files[0]
            content = indexer.get_file_content(test_file)
            print(f"  Test file: {test_file}")
            print(f"  Content length: {len(content) if content else 0} characters")
    else:
        print("Failed to create index")
        sys.exit(1)
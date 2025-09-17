"""
Redis-based repository manager that provides GitRepo-compatible interface.

This module implements RedisRepoManager which mimics the GitRepo interface
while using Redis as the backend storage and a virtual file system for
file operations.
"""

import os
import time
from pathlib import Path, PurePosixPath
from typing import List, Optional, Dict, Any
import logging

from cosmos.virtual_filesystem import IntelligentVirtualFileSystem
from cosmos.redis_cache import SmartRedisCache
from cosmos.tier_access_control import (
    TierAccessController, 
    TierAccessDeniedError,
    get_tier_access_controller
)
from cosmos import utils


class VirtualFileSystemIO:
    """
    IO wrapper that can read from virtual filesystem for Redis mode.
    
    This class wraps the original IO object and intercepts file reading
    operations to use the virtual filesystem when files don't exist
    on the actual filesystem.
    """
    
    def __init__(self, original_io, virtual_fs, repo_root):
        self.original_io = original_io
        self.virtual_fs = virtual_fs
        self.repo_root = Path(repo_root)
        
        # Delegate all other attributes to the original IO
        for attr in dir(original_io):
            if not attr.startswith('_') and attr not in ['read_text', 'read_image']:
                setattr(self, attr, getattr(original_io, attr))
    
    def read_text(self, filename, silent=False):
        """
        Read text from virtual filesystem if file doesn't exist on disk.
        
        Args:
            filename: Path to the file
            silent: Whether to suppress error messages
            
        Returns:
            File content as string or None if not found
        """
        try:
            # First try to read from the actual filesystem
            return self.original_io.read_text(filename, silent=True)
        except (FileNotFoundError, OSError):
            # If file doesn't exist on disk, try virtual filesystem
            if self.virtual_fs:
                try:
                    # Convert absolute path to relative path for virtual filesystem
                    file_path = Path(filename)
                    if file_path.is_absolute():
                        try:
                            rel_path = file_path.relative_to(self.repo_root)
                            content = self.virtual_fs.extract_file_with_context(str(rel_path))
                            if content:
                                return content
                        except ValueError:
                            # Path is not relative to repo root, try as-is
                            pass
                    
                    # Try the path as-is
                    content = self.virtual_fs.extract_file_with_context(str(filename))
                    if content:
                        return content
                        
                except Exception as e:
                    if not silent:
                        logger.warning(f"Error reading from virtual filesystem {filename}: {e}")
            
            # If all else fails, use original IO (which will show appropriate error)
            return self.original_io.read_text(filename, silent)
    
    def read_image(self, filename):
        """
        Read image from virtual filesystem if file doesn't exist on disk.
        
        Args:
            filename: Path to the image file
            
        Returns:
            Base64 encoded image content or None if not found
        """
        try:
            # First try to read from the actual filesystem
            return self.original_io.read_image(filename)
        except (FileNotFoundError, OSError):
            # For now, images are not supported in virtual filesystem
            # Could be extended in the future if needed
            return self.original_io.read_image(filename)

logger = logging.getLogger(__name__)


class RedisRepoManager:
    """
    Redis-backed implementation that mirrors the GitRepo interface.
    
    This class provides the same methods as GitRepo to ensure seamless
    integration with existing cosmos codebase while using Redis storage
    and virtual file system.
    """
    
    def __init__(self, io, fnames, git_dname, repo_url: str, user_tier: str = None, 
                 redis_client=None, cosmos_ignore_file=None, models=None,
                 attribute_author=True, attribute_committer=True,
                 attribute_commit_message_author=False,
                 attribute_commit_message_committer=False,
                 commit_prompt=None, subtree_only=False,
                 git_commit_verify=True, attribute_co_authored_by=False):
        """
        Initialize RedisRepoManager with GitRepo-compatible interface.
        
        Args:
            io: Input/output handler
            fnames: List of filenames
            git_dname: Git directory name
            repo_url: URL of the repository
            user_tier: User tier (free, pro, enterprise) - optional, reads from env if None
            redis_client: Redis client instance
            cosmos_ignore_file: Cosmos ignore file path
            models: Models configuration
            attribute_author: Whether to attribute author
            attribute_committer: Whether to attribute committer
            attribute_commit_message_author: Whether to attribute commit message author
            attribute_commit_message_committer: Whether to attribute commit message committer
            commit_prompt: Commit prompt template
            subtree_only: Whether to use subtree only
            git_commit_verify: Whether to verify git commits
            attribute_co_authored_by: Whether to attribute co-authored by
        """
        self.original_io = io
        self.models = models
        self.repo_url = repo_url
        self.user_tier = user_tier
        
        # GitRepo compatibility attributes
        self.normalized_path = {}
        self.tree_files = {}
        self.attribute_author = attribute_author
        self.attribute_committer = attribute_committer
        self.attribute_commit_message_author = attribute_commit_message_author
        self.attribute_commit_message_committer = attribute_commit_message_committer
        self.attribute_co_authored_by = attribute_co_authored_by
        self.commit_prompt = commit_prompt
        self.subtree_only = subtree_only
        self.git_commit_verify = git_commit_verify
        self.ignore_file_cache = {}
        
        # GitRepo compatibility - simulate repo object
        self.repo = None  # Will be set to self for compatibility
        self.git_repo_error = None
        
        # Cosmos ignore handling
        self.cosmos_ignore_file = None
        self.cosmos_ignore_spec = None
        self.cosmos_ignore_ts = 0
        self.cosmos_ignore_last_check = 0
        
        if cosmos_ignore_file:
            self.cosmos_ignore_file = Path(cosmos_ignore_file)
        
        # Redis and virtual filesystem
        self.redis_cache = redis_client or SmartRedisCache()
        self.virtual_fs = None
        self.repo_name = self._extract_repo_name(repo_url)
        
        # Set up virtual root path - use temp directory for cache files
        import tempfile
        temp_dir = tempfile.gettempdir()
        self.root = utils.safe_abs_path(f"{temp_dir}/cosmos_redis_cache/{self.repo_name}")
        
        # Ensure the directory exists for cache files
        os.makedirs(self.root, exist_ok=True)
        
        # Tier access control
        self.tier_controller = get_tier_access_controller()
        
        # Validate tier access before loading repository data
        self._validate_tier_access()
        
        # Load repository data from Redis
        self._load_repository_data()
        
        # Set up virtual filesystem IO wrapper
        self.io = VirtualFileSystemIO(self.original_io, self.virtual_fs, self.root)
        
        # Set repo to self for GitRepo compatibility
        self.repo = self
        
        logger.info(f"Initialized RedisRepoManager for {self.repo_name}")
    
    def _extract_repo_name(self, repo_url: str) -> str:
        """Extract repository name from URL."""
        if not repo_url:
            return "unknown-repo"
        
        # Handle direct repository names (e.g., "octocat/Hello-World")
        if "/" in repo_url and not repo_url.startswith(('http', 'git@')):
            # This is likely a direct repo name like "octocat/Hello-World"
            return repo_url.strip()
        
        # Handle GitHub URLs
        if "github.com" in repo_url:
            # Handle both HTTPS and SSH URLs
            if repo_url.startswith('git@'):
                # SSH format: git@github.com:user/repo.git
                parts = repo_url.split(':')[-1].split('/')
            else:
                # HTTPS format: https://github.com/user/repo
                parts = repo_url.rstrip('/').split('/')
            
            if len(parts) >= 2:
                user = parts[-2]
                repo = parts[-1].replace('.git', '')
                return f"{user}/{repo}"
        
        # Fallback to simple name extraction
        return repo_url.split('/')[-1].replace('.git', '')
    
    def _validate_tier_access(self) -> None:
        """Validate tier access for this repository."""
        try:
            # Get repository data to estimate token count
            repo_data = self.redis_cache.get_repository_data_cached(self.repo_name)
            
            if not repo_data:
                # If no data exists, allow access (will be validated during fetch)
                return
            
            # Estimate token count from content
            content_md = repo_data.get('content', '')
            estimated_tokens = len(content_md.split()) * 1.3  # Rough token estimation
            
            # Validate access
            is_allowed, message, _ = self.tier_controller.validate_repository_access(
                self.repo_url, int(estimated_tokens), self.user_tier
            )
            
            if not is_allowed:
                raise TierAccessDeniedError(message)
                
        except TierAccessDeniedError:
            raise
        except Exception as e:
            logger.warning(f"Could not validate tier access for {self.repo_name}: {e}")
            # Allow access if validation fails (graceful degradation)
    
    def _load_repository_data(self) -> None:
        """Load repository data from Redis and initialize virtual filesystem."""
        try:
            repo_data = self.redis_cache.get_repository_data_cached(self.repo_name)
            
            if not repo_data:
                logger.warning(f"No repository data found for {self.repo_name}")
                # Initialize empty virtual filesystem
                self.virtual_fs = IntelligentVirtualFileSystem("", "", self.repo_name)
                return
            
            content_md = repo_data.get('content', '')
            tree_txt = repo_data.get('tree', '')
            
            # Get storage directory from environment or use default
            storage_dir = os.getenv('STORAGE_DIR', '/tmp/repo_storage')
            
            # Initialize virtual filesystem with indexing support
            self.virtual_fs = IntelligentVirtualFileSystem(
                content_md, 
                tree_txt, 
                self.repo_name,
                repo_storage_dir=storage_dir
            )
            
            # Extract files to disk for repo-map compatibility
            self._extract_files_to_disk()
            
            logger.info(f"Loaded repository data for {self.repo_name}")
            
        except Exception as e:
            logger.error(f"Error loading repository data for {self.repo_name}: {e}")
            # Initialize empty virtual filesystem as fallback
            self.virtual_fs = IntelligentVirtualFileSystem("", "", self.repo_name)
    
    def _extract_files_to_disk(self) -> None:
        """
        Extract files from virtual filesystem to disk for repo-map compatibility.
        
        This creates physical files in the repository root directory so that
        repo-map and other file system operations can access them.
        """
        if not self.virtual_fs:
            return
        
        try:
            # Get all tracked files from virtual filesystem
            tracked_files = self.virtual_fs.get_tracked_files()
            
            if not tracked_files:
                logger.warning(f"No tracked files found in virtual filesystem for {self.repo_name}")
                return
            
            # Create repository directory
            repo_dir = Path(self.root)
            repo_dir.mkdir(parents=True, exist_ok=True)
            
            files_extracted = 0
            for file_path in tracked_files:
                try:
                    # Get file content from virtual filesystem
                    content = self.virtual_fs.extract_file_with_context(file_path)
                    
                    if content is not None:
                        # Create physical file
                        physical_path = repo_dir / file_path
                        physical_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Write content to disk
                        with open(physical_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        files_extracted += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to extract file {file_path}: {e}")
                    continue
            
            logger.info(f"Extracted {files_extracted} files to disk for {self.repo_name}")
            
        except Exception as e:
            logger.error(f"Error extracting files to disk for {self.repo_name}: {e}")
    
    # GitRepo interface compatibility methods
    
    def commit(self, fnames=None, context=None, message=None, cosmos_edits=False, coder=None):
        """
        Simulate commit operation (no-op for Redis backend).
        
        Args:
            fnames: List of filenames to commit
            context: Commit context
            message: Commit message
            cosmos_edits: Whether these are cosmos edits
            coder: Coder instance
        """
        # Redis backend doesn't support commits - this is a no-op
        if message:
            logger.info(f"Simulated commit for {self.repo_name}: {message}")
        return None
    
    def get_rel_repo_dir(self) -> str:
        """Get relative repository directory."""
        return f".redis/{self.repo_name}"
    
    def get_commit_message(self, diffs, context, user_language=None) -> str:
        """
        Generate commit message (simulated for Redis backend).
        
        Args:
            diffs: Diff content
            context: Commit context
            user_language: User language preference
            
        Returns:
            Generated commit message
        """
        return f"Update files in {self.repo_name}"
    
    def get_diffs(self, fnames=None) -> str:
        """
        Get diffs (simulated for Redis backend).
        
        Args:
            fnames: List of filenames
            
        Returns:
            Empty string (no diffs in Redis backend)
        """
        return ""
    
    def diff_commits(self, pretty, from_commit, to_commit) -> str:
        """
        Diff between commits (simulated for Redis backend).
        
        Args:
            pretty: Whether to use pretty format
            from_commit: From commit hash
            to_commit: To commit hash
            
        Returns:
            Empty string (no commit history in Redis backend)
        """
        return ""
    
    def get_tracked_files(self) -> List[str]:
        """
        Get list of tracked files from virtual filesystem.
        
        Returns:
            List of tracked file paths
        """
        if not self.virtual_fs:
            return []
        
        return self.virtual_fs.get_tracked_files()
    
    def normalize_path(self, path: str) -> str:
        """
        Normalize file path for virtual filesystem (GitRepo compatible).
        
        Args:
            path: Path to normalize
            
        Returns:
            Normalized path
        """
        if not path:
            return path
            
        orig_path = path
        
        # Use cached result if available
        if orig_path in self.normalized_path:
            return self.normalized_path[orig_path]
        
        # Normalize path similar to GitRepo
        try:
            path = str(Path(PurePosixPath((Path(self.root) / path).relative_to(self.root))))
        except (ValueError, OSError):
            # If path normalization fails, use virtual filesystem fallback
            if self.virtual_fs:
                path = self.virtual_fs.resolve_cosmos_path(path)
        
        self.normalized_path[orig_path] = path
        return path
    
    def refresh_cosmos_ignore(self) -> None:
        """Refresh cosmos ignore patterns (no-op for Redis backend)."""
        pass
    
    def git_ignored_file(self, path: str) -> bool:
        """
        Check if file is git ignored (simulated for Redis backend).
        
        Args:
            path: File path to check
            
        Returns:
            False (no git ignore in Redis backend)
        """
        return False
    
    def ignored_file(self, fname: str) -> bool:
        """
        Check if file is ignored by cosmos ignore patterns.
        
        Args:
            fname: Filename to check
            
        Returns:
            True if file should be ignored
        """
        # Basic ignore patterns for Redis backend
        ignore_patterns = ['.git/', '__pycache__/', '*.pyc', '.DS_Store', '*.log']
        
        fname_str = str(fname)
        for pattern in ignore_patterns:
            if pattern.endswith('/') and fname_str.startswith(pattern):
                return True
            elif pattern.startswith('*.') and fname_str.endswith(pattern[1:]):
                return True
            elif pattern in fname_str:
                return True
        
        return False
    
    def ignored_file_raw(self, fname: str) -> bool:
        """
        Raw file ignore check.
        
        Args:
            fname: Filename to check
            
        Returns:
            Result of ignored_file check
        """
        return self.ignored_file(fname)
    
    def path_in_repo(self, path: str) -> bool:
        """
        Check if path is in repository.
        
        Args:
            path: Path to check
            
        Returns:
            True if path exists in virtual filesystem
        """
        if not self.virtual_fs:
            return False
        
        normalized_path = self.normalize_path(path)
        return self.virtual_fs.file_exists(normalized_path)
    
    def abs_root_path(self, path: str) -> Path:
        """
        Get absolute root path for virtual filesystem.
        
        Args:
            path: Relative path
            
        Returns:
            Absolute path in virtual filesystem
        """
        return Path(self.root) / path
    
    def get_dirty_files(self) -> List[str]:
        """
        Get dirty files (no dirty files in Redis backend).
        
        Returns:
            Empty list (no dirty files concept in Redis backend)
        """
        return []
    
    def is_dirty(self, path=None) -> bool:
        """
        Check if repository or path is dirty.
        
        Args:
            path: Optional path to check
            
        Returns:
            False (no dirty state in Redis backend)
        """
        return False
    
    def get_head_commit(self):
        """
        Get head commit (simulated for Redis backend).
        
        Returns:
            None (no commits in Redis backend)
        """
        return None
    
    def get_head_commit_sha(self, short=False) -> Optional[str]:
        """
        Get head commit SHA (simulated for Redis backend).
        
        Args:
            short: Whether to return short SHA
            
        Returns:
            None (no commits in Redis backend)
        """
        return None
    
    def get_head_commit_message(self, default=None) -> Optional[str]:
        """
        Get head commit message (simulated for Redis backend).
        
        Args:
            default: Default message if no commit
            
        Returns:
            Default value (no commits in Redis backend)
        """
        return default
    
    # Smart file operations for cosmos integration
    
    def read_text(self, file_path: str, encoding: str = 'utf-8') -> str:
        """
        Read file content as text (cosmos-compatible file reading).
        
        Args:
            file_path: Path to the file
            encoding: Text encoding (default: utf-8)
            
        Returns:
            File content as string
        """
        if not self.virtual_fs:
            return ""
        
        try:
            content = self.virtual_fs.extract_file_with_context(file_path)
            return content if content else ""
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            return ""
    
    def get_file_content(self, file_path: str) -> str:
        """
        Get file content from virtual filesystem.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string
        """
        return self.read_text(file_path)
    
    def read_file_safe(self, file_path: str, default: str = "") -> str:
        """
        Safely read file content with fallback.
        
        Args:
            file_path: Path to the file
            default: Default content if file doesn't exist
            
        Returns:
            File content or default
        """
        if not self.file_exists(file_path):
            return default
        
        return self.read_text(file_path)
    
    def get_file_lines(self, file_path: str) -> List[str]:
        """
        Get file content as list of lines.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of file lines
        """
        content = self.read_text(file_path)
        if not content:
            return []
        
        return content.splitlines()
    
    def cosmos_file_exists(self, file_path: str) -> bool:
        """
        Check if file exists (cosmos-compatible).
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file exists
        """
        return self.file_exists(file_path)
    
    def cosmos_is_file(self, file_path: str) -> bool:
        """
        Check if path is a file (cosmos-compatible).
        
        Args:
            file_path: Path to check
            
        Returns:
            True if path is a file
        """
        if not self.virtual_fs:
            return False
        
        metadata = self.virtual_fs.get_file_metadata(file_path)
        return metadata.get('is_file', False)
    
    def cosmos_is_dir(self, file_path: str) -> bool:
        """
        Check if path is a directory (cosmos-compatible).
        
        Args:
            file_path: Path to check
            
        Returns:
            True if path is a directory
        """
        return self.is_directory(file_path)
    
    def cosmos_iterdir(self, dir_path: str = "") -> List[str]:
        """
        Iterate directory contents (cosmos-compatible).
        
        Args:
            dir_path: Directory path to iterate
            
        Returns:
            List of directory contents with full paths
        """
        if not self.virtual_fs:
            return []
        
        try:
            contents = self.virtual_fs.list_directory(dir_path)
            # Return full paths for cosmos compatibility
            if dir_path:
                return [f"{dir_path.rstrip('/')}/{item}" for item in contents]
            else:
                return contents
        except Exception as e:
            logger.warning(f"Could not iterate directory {dir_path}: {e}")
            return []
    
    def cosmos_glob(self, pattern: str, recursive: bool = False) -> List[str]:
        """
        Find files matching pattern (cosmos-compatible glob).
        
        Args:
            pattern: Glob pattern to match
            recursive: Whether to search recursively
            
        Returns:
            List of matching file paths
        """
        if not self.virtual_fs:
            return []
        
        try:
            all_files = self.get_tracked_files()
            
            # Simple pattern matching (basic implementation)
            import fnmatch
            matching_files = []
            
            for file_path in all_files:
                if fnmatch.fnmatch(file_path, pattern):
                    matching_files.append(file_path)
                elif recursive and '**' in pattern:
                    # Handle recursive patterns
                    simple_pattern = pattern.replace('**/', '').replace('**', '*')
                    if fnmatch.fnmatch(file_path, simple_pattern):
                        matching_files.append(file_path)
            
            return matching_files
        except Exception as e:
            logger.warning(f"Could not glob pattern {pattern}: {e}")
            return []
    
    def cosmos_walk(self, top_dir: str = "") -> List[tuple]:
        """
        Walk directory tree (cosmos-compatible os.walk).
        
        Args:
            top_dir: Top directory to start walking from
            
        Returns:
            List of (dirpath, dirnames, filenames) tuples
        """
        if not self.virtual_fs:
            return []
        
        try:
            # Get directory structure from virtual filesystem
            structure = self.virtual_fs.get_cosmos_compatible_tree()
            
            # Convert to os.walk format
            walk_results = []
            
            def _walk_recursive(current_path: str, tree_node: dict):
                dirnames = []
                filenames = []
                
                for name, node in tree_node.items():
                    if isinstance(node, dict):
                        dirnames.append(name)
                        # Recursively walk subdirectories
                        subdir_path = f"{current_path}/{name}" if current_path else name
                        _walk_recursive(subdir_path, node)
                    else:
                        filenames.append(name)
                
                walk_results.append((current_path, dirnames, filenames))
            
            if isinstance(structure, dict):
                _walk_recursive(top_dir, structure)
            
            return walk_results
        except Exception as e:
            logger.warning(f"Could not walk directory {top_dir}: {e}")
            return []
    
    def cosmos_relative_to(self, path: str, base_path: str = None) -> str:
        """
        Get relative path (cosmos-compatible Path.relative_to).
        
        Args:
            path: Path to make relative
            base_path: Base path (defaults to repo root)
            
        Returns:
            Relative path string
        """
        if base_path is None:
            base_path = self.root
        
        try:
            path_obj = Path(path)
            base_obj = Path(base_path)
            return str(path_obj.relative_to(base_obj))
        except (ValueError, OSError):
            # If relative_to fails, return normalized path
            return self.normalize_path(path)
    
    def cosmos_resolve_path(self, path: str) -> str:
        """
        Resolve path to absolute form (cosmos-compatible).
        
        Args:
            path: Path to resolve
            
        Returns:
            Resolved absolute path
        """
        if Path(path).is_absolute():
            return path
        
        # Make relative to virtual root
        return str(Path(self.root) / path)
    
    def cosmos_stat(self, file_path: str) -> Dict[str, Any]:
        """
        Get file statistics (cosmos-compatible os.stat).
        
        Args:
            file_path: Path to get stats for
            
        Returns:
            Dictionary with stat-like information
        """
        if not self.virtual_fs:
            return {
                'st_size': 0,
                'st_mtime': 0,
                'st_mode': 0o644,
                'exists': False
            }
        
        metadata = self.virtual_fs.get_file_metadata(file_path)
        
        # Convert to os.stat-like format
        return {
            'st_size': metadata.get('size', 0),
            'st_mtime': metadata.get('mtime', 0),
            'st_mode': 0o755 if metadata.get('is_dir', False) else 0o644,
            'exists': metadata.get('exists', False)
        }
    
    def get_directory_structure(self) -> Dict[str, Any]:
        """
        Get directory structure from virtual filesystem.
        
        Returns:
            Directory structure dictionary
        """
        if not self.virtual_fs:
            return {}
        
        return self.virtual_fs.get_cosmos_compatible_tree()
    
    def check_tier_access(self) -> bool:
        """
        Check if user tier allows access to this repository.
        
        Returns:
            True if access is allowed
        """
        # This will be implemented in the TierManager integration
        # For now, allow all access
        return True
    
    def get_virtual_root(self) -> str:
        """
        Get virtual root path.
        
        Returns:
            Virtual root path string
        """
        return self.root
    
    def list_directory(self, dir_path: str = "") -> List[str]:
        """
        List directory contents.
        
        Args:
            dir_path: Directory path to list
            
        Returns:
            List of directory contents
        """
        if not self.virtual_fs:
            return []
        
        return self.virtual_fs.list_directory(dir_path)
    
    def file_exists(self, file_path: str) -> bool:
        """
        Check if file exists in virtual filesystem.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file exists
        """
        if not self.virtual_fs:
            return False
        
        return self.virtual_fs.file_exists(file_path)
    
    def is_directory(self, path: str) -> bool:
        """
        Check if path is a directory.
        
        Args:
            path: Path to check
            
        Returns:
            True if path is a directory
        """
        if not self.virtual_fs:
            return False
        
        return self.virtual_fs.is_directory(path)
    
    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get file metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File metadata dictionary
        """
        if not self.virtual_fs:
            return {
                'size': 0,
                'mtime': 0,
                'exists': False,
                'is_file': False,
                'is_dir': False
            }
        
        return self.virtual_fs.get_file_metadata(file_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get repository statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.virtual_fs:
            return {
                'total_files': 0,
                'total_directories': 0,
                'total_size': 0,
                'repo_name': self.repo_name
            }
        
        return self.virtual_fs.get_stats()
    
    # Cosmos-specific directory access patterns
    
    def cosmos_find_files(self, extensions: List[str] = None, exclude_patterns: List[str] = None) -> List[str]:
        """
        Find files matching cosmos criteria.
        
        Args:
            extensions: List of file extensions to include (e.g., ['.py', '.js'])
            exclude_patterns: List of patterns to exclude
            
        Returns:
            List of matching file paths
        """
        if not self.virtual_fs:
            return []
        
        all_files = self.get_tracked_files()
        matching_files = []
        
        for file_path in all_files:
            # Skip if already ignored
            if self.ignored_file(file_path):
                continue
            
            # Check extensions
            if extensions:
                file_ext = Path(file_path).suffix.lower()
                if file_ext not in [ext.lower() for ext in extensions]:
                    continue
            
            # Check exclude patterns
            if exclude_patterns:
                excluded = False
                for pattern in exclude_patterns:
                    if pattern in file_path or file_path.endswith(pattern):
                        excluded = True
                        break
                if excluded:
                    continue
            
            matching_files.append(file_path)
        
        return matching_files
    
    def cosmos_get_source_files(self) -> List[str]:
        """
        Get source code files for cosmos processing.
        
        Returns:
            List of source file paths
        """
        source_extensions = [
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
            '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala',
            '.clj', '.hs', '.ml', '.fs', '.vb', '.pl', '.sh', '.bash',
            '.ps1', '.r', '.m', '.mm', '.sql', '.html', '.css', '.scss',
            '.less', '.vue', '.svelte', '.dart', '.lua', '.nim', '.zig'
        ]
        
        return self.cosmos_find_files(extensions=source_extensions)
    
    def cosmos_get_config_files(self) -> List[str]:
        """
        Get configuration files for cosmos processing.
        
        Returns:
            List of configuration file paths
        """
        config_patterns = [
            'package.json', 'requirements.txt', 'cargo.toml', 'pom.xml',
            'build.gradle', 'cmakelists.txt', 'makefile', 'dockerfile',
            '.gitignore', '.env', 'config.', 'settings.', '.yml', '.yaml',
            '.toml', '.ini', '.conf', '.cfg', '.json', '.xml'
        ]
        
        all_files = self.get_tracked_files()
        config_files = []
        
        for file_path in all_files:
            if self.ignored_file(file_path):
                continue
            
            file_name = Path(file_path).name.lower()
            for pattern in config_patterns:
                if pattern in file_name or file_name.endswith(pattern):
                    config_files.append(file_path)
                    break
        
        return config_files
    
    def cosmos_get_documentation_files(self) -> List[str]:
        """
        Get documentation files for cosmos processing.
        
        Returns:
            List of documentation file paths
        """
        doc_extensions = ['.md', '.rst', '.txt', '.adoc', '.org']
        doc_names = ['readme', 'changelog', 'license', 'contributing', 'docs']
        
        all_files = self.get_tracked_files()
        doc_files = []
        
        for file_path in all_files:
            if self.ignored_file(file_path):
                continue
            
            file_name = Path(file_path).name.lower()
            file_ext = Path(file_path).suffix.lower()
            
            # Check by extension
            if file_ext in doc_extensions:
                doc_files.append(file_path)
                continue
            
            # Check by name patterns
            for doc_name in doc_names:
                if doc_name in file_name:
                    doc_files.append(file_path)
                    break
        
        return doc_files
    
    def cosmos_get_file_tree(self, max_depth: int = None) -> Dict[str, Any]:
        """
        Get file tree structure for cosmos display.
        
        Args:
            max_depth: Maximum depth to traverse (None for unlimited)
            
        Returns:
            Nested dictionary representing file tree
        """
        if not self.virtual_fs:
            return {}
        
        try:
            tree = self.virtual_fs.get_cosmos_compatible_tree()
            
            if max_depth is not None:
                # Limit tree depth
                def _limit_depth(node: dict, current_depth: int) -> dict:
                    if current_depth >= max_depth:
                        return {}
                    
                    limited_node = {}
                    for key, value in node.items():
                        if isinstance(value, dict):
                            limited_node[key] = _limit_depth(value, current_depth + 1)
                        else:
                            limited_node[key] = value
                    
                    return limited_node
                
                tree = _limit_depth(tree, 0)
            
            return tree
        except Exception as e:
            logger.warning(f"Could not get file tree: {e}")
            return {}
    
    def cosmos_search_files(self, query: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """
        Search for files containing query text.
        
        Args:
            query: Text to search for
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            List of dictionaries with file path and match information
        """
        if not self.virtual_fs:
            return []
        
        results = []
        search_query = query if case_sensitive else query.lower()
        
        for file_path in self.get_tracked_files():
            if self.ignored_file(file_path):
                continue
            
            try:
                content = self.read_text(file_path)
                if not content:
                    continue
                
                search_content = content if case_sensitive else content.lower()
                
                if search_query in search_content:
                    # Find line numbers with matches
                    lines = content.splitlines()
                    matching_lines = []
                    
                    for i, line in enumerate(lines, 1):
                        search_line = line if case_sensitive else line.lower()
                        if search_query in search_line:
                            matching_lines.append({
                                'line_number': i,
                                'line_content': line.strip(),
                                'match_start': search_line.find(search_query)
                            })
                    
                    results.append({
                        'file_path': file_path,
                        'matches': len(matching_lines),
                        'matching_lines': matching_lines[:10]  # Limit to first 10 matches
                    })
            
            except Exception as e:
                logger.warning(f"Could not search file {file_path}: {e}")
                continue
        
        return results
    
    # Additional GitRepo compatibility methods
    
    def is_dirty(self, path=None) -> bool:
        """
        Check if repository or path is dirty (GitRepo compatible).
        
        Args:
            path: Optional path to check
            
        Returns:
            False (no dirty state in Redis backend)
        """
        if path and not self.path_in_repo(path):
            return True
        return False
    
    def ignored(self, path: str) -> bool:
        """
        Check if path is ignored (GitRepo git.ignored compatibility).
        
        Args:
            path: Path to check
            
        Returns:
            True if path is ignored
        """
        return self.git_ignored_file(path)
    
    # Properties for GitRepo compatibility
    
    @property
    def working_tree_dir(self) -> str:
        """Get working tree directory (virtual root)."""
        return self.root
    
    @property
    def git_dir(self) -> str:
        """Get git directory (simulated)."""
        return f"{self.root}/.git"
    
    @property
    def head(self):
        """Simulate git head for compatibility."""
        return self
    
    @property
    def commit_obj(self):
        """Simulate commit object for compatibility."""
        return self
    
    @property
    def hexsha(self) -> str:
        """Simulate commit hexsha."""
        hash_str = str(abs(hash(self.repo_name)))
        return "redis-virtual-commit-" + hash_str[:7].zfill(7)
    
    @property
    def message(self) -> str:
        """Simulate commit message."""
        return f"Virtual commit for {self.repo_name}"
    
    @property
    def index(self):
        """Simulate git index for compatibility."""
        return self
    
    @property
    def entries(self) -> Dict:
        """Simulate index entries."""
        if not self.virtual_fs:
            return {}
        
        # Return tracked files as index entries
        tracked_files = self.get_tracked_files()
        return {(fname, 0): None for fname in tracked_files}
    
    @property
    def git(self):
        """Simulate git command interface."""
        return self
    
    # Git command simulation methods
    
    def config(self, *args) -> str:
        """Simulate git config command."""
        if args == ("--get", "user.name"):
            return os.getenv("GIT_AUTHOR_NAME", "Redis User")
        return ""
    
    def add(self, *args) -> None:
        """Simulate git add command (no-op for Redis backend)."""
        pass
    
    def diff(self, *args, **kwargs) -> str:
        """Simulate git diff command."""
        return ""  # No diffs in Redis backend
    
    def iter_commits(self, *args, **kwargs):
        """Simulate git commit iteration."""
        return iter([])  # No commits in Redis backend
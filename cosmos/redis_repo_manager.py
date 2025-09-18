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
from cosmos.repomap import RepoMap
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
        
        # Track modified files for PR creation
        self.modified_files = {}
        
        # Delegate all other attributes to the original IO
        for attr in dir(original_io):
            if not attr.startswith('_') and attr not in ['read_text', 'read_image', 'get_mtime', 'write_text']:
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
            # First try to read from the actual filesystem (for extracted files)
            result = self.original_io.read_text(filename, silent=True)
            if result is not None:
                return result
        except (FileNotFoundError, OSError):
            pass
        
        # If file doesn't exist on disk, try virtual filesystem
        if self.virtual_fs:
            try:
                # Convert absolute path to relative path for virtual filesystem
                file_path = Path(filename)
                if file_path.is_absolute():
                    try:
                        rel_path = file_path.relative_to(self.repo_root)
                        content = self.virtual_fs.extract_file_with_context(str(rel_path))
                        if content is not None:
                            return content
                    except ValueError:
                        # Path is not relative to repo root, try as-is
                        pass
                
                # Try the path as-is
                content = self.virtual_fs.extract_file_with_context(str(filename))
                if content is not None:
                    return content
                    
            except Exception as e:
                if not silent:
                    logger.warning(f"Error reading from virtual filesystem {filename}: {e}")
        
        # If all else fails, return None for RepoMap compatibility
        if silent:
            return None
        
        # For non-silent calls, try original IO one more time to get proper error handling
        try:
            return self.original_io.read_text(filename, silent=False)
        except:
            return None
    
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
    
    def get_file_content(self, filename):
        """
        Get file content from virtual filesystem - alias for read_text.
        
        Args:
            filename: Path to the file
            
        Returns:
            File content as string or None if not found
        """
        return self.read_text(filename, silent=True)
    
    def get_file_content_for_pr(self, filename):
        """
        Get file content for PR creation, prioritizing modified content.
        
        Args:
            filename: Path to the file
            
        Returns:
            File content as string (modified if available, otherwise original)
        """
        # First check if we have modified content
        modified_content = self.get_modified_content(filename)
        if modified_content is not None:
            return modified_content
        
        # Fall back to original content from virtual filesystem
        return self.read_text(filename, silent=True)
    
    def write_text(self, filename, content, encoding='utf-8'):
        """
        Write text to file and store modified content for PR creation.
        
        Args:
            filename: Path to the file
            content: Content to write
            encoding: Text encoding (default: utf-8)
        """
        try:
            # Write to actual filesystem first - the original IO doesn't accept encoding as parameter
            result = self.original_io.write_text(filename, content)
            
            # Store modified content for PR creation
            # Convert to relative path for consistent tracking
            file_path = Path(filename)
            if file_path.is_absolute():
                try:
                    rel_path = file_path.relative_to(self.repo_root)
                    self.modified_files[str(rel_path)] = content
                except ValueError:
                    # Path is not relative to repo root, store as-is
                    self.modified_files[str(filename)] = content
            else:
                self.modified_files[str(filename)] = content
            
            return result
        except Exception as e:
            logger.warning(f"Error writing file {filename}: {e}")
            # Still store the content for PR creation even if write fails
            file_path = Path(filename)
            if file_path.is_absolute():
                try:
                    rel_path = file_path.relative_to(self.repo_root)
                    self.modified_files[str(rel_path)] = content
                except ValueError:
                    self.modified_files[str(filename)] = content
            else:
                self.modified_files[str(filename)] = content
            # Don't re-raise the exception since we've stored the content for PR creation
            logger.info(f"Content stored in buffer for {filename} despite write error")
    
    def get_modified_content(self, filename):
        """
        Get modified content for a file if it has been changed.
        
        Args:
            filename: Path to the file
            
        Returns:
            Modified content as string or None if not modified
        """
        # Try different path formats
        paths_to_try = [str(filename)]
        
        file_path = Path(filename)
        if file_path.is_absolute():
            try:
                rel_path = file_path.relative_to(self.repo_root)
                paths_to_try.append(str(rel_path))
            except ValueError:
                pass
        else:
            abs_path = self.repo_root / filename
            paths_to_try.append(str(abs_path))
        
        for path in paths_to_try:
            if path in self.modified_files:
                return self.modified_files[path]
        
        return None
    
    def get_mtime(self, filename):
        """
        Get file modification time for RepoMap caching.
        
        Args:
            filename: Path to the file
            
        Returns:
            Modification time or current time for virtual files
        """
        try:
            # Try to get actual file mtime first (for extracted files)
            return os.path.getmtime(filename)
        except (FileNotFoundError, OSError):
            # For virtual files, return a consistent timestamp based on repo data
            # This ensures RepoMap caching works properly
            file_path = Path(filename)
            
            # Check if it's a virtual file
            if self.virtual_fs:
                # Convert absolute path to relative if needed
                if file_path.is_absolute():
                    try:
                        rel_path = file_path.relative_to(self.repo_root)
                        if self.virtual_fs.file_exists(str(rel_path)):
                            # Use a hash of the filename to create a consistent "mtime"
                            return abs(hash(f"{self.repo_name}:{rel_path}")) % (2**31)
                    except ValueError:
                        pass
                
                # Try as-is
                if self.virtual_fs.file_exists(str(filename)):
                    return abs(hash(f"{self.repo_name}:{filename}")) % (2**31)
            
            return None
    
    def get_mtime(self, filename):
        """
        Get file modification time for RepoMap caching (VirtualFileSystemIO method).
        
        Args:
            filename: Path to the file
            
        Returns:
            Modification time or consistent hash for virtual files
        """
        try:
            # Try to get actual file mtime first (for extracted files)
            return os.path.getmtime(filename)
        except (FileNotFoundError, OSError):
            # For virtual files, return a consistent timestamp
            file_path = Path(filename)
            
            if self.virtual_fs:
                # Convert absolute path to relative if needed
                if file_path.is_absolute():
                    try:
                        rel_path = file_path.relative_to(self.repo_root)
                        if self.virtual_fs.file_exists(str(rel_path)):
                            # Use a hash of the filename to create a consistent "mtime"
                            return abs(hash(f"virtual:{rel_path}")) % (2**31)
                    except ValueError:
                        pass
                
                # Try as-is
                if self.virtual_fs.file_exists(str(filename)):
                    return abs(hash(f"virtual:{filename}")) % (2**31)
            
            return None
    
    def path_exists(self, filename):
        """
        Check if path exists for RepoMap compatibility.
        
        Args:
            filename: Path to check
            
        Returns:
            True if path exists in filesystem or virtual filesystem
        """
        try:
            # Check actual filesystem first
            if Path(filename).exists():
                return True
        except (OSError, ValueError):
            pass
        
        # Check virtual filesystem
        if self.virtual_fs:
            # Convert absolute path to relative if needed
            file_path = Path(filename)
            if file_path.is_absolute():
                try:
                    rel_path = file_path.relative_to(self.repo_root)
                    return self.virtual_fs.file_exists(str(rel_path))
                except ValueError:
                    pass
            
            return self.virtual_fs.file_exists(str(filename))
        
        return False

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
                 git_commit_verify=True, attribute_co_authored_by=False,
                 create_pull_request=False, pr_base_branch='main', pr_draft=False,
                 auto_cleanup=True, github_token=None):
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
            create_pull_request: Whether to create pull requests instead of direct commits
            pr_base_branch: Base branch for pull requests
            pr_draft: Whether to create draft pull requests
            auto_cleanup: Whether to automatically cleanup temporary files after operations
            github_token: GitHub personal access token for PR operations
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
        
        # PR-related attributes
        self.create_pull_request = create_pull_request
        self.pr_base_branch = pr_base_branch
        self.pr_draft = pr_draft
        self.github_token = github_token
        
        # PR creation tracking to prevent duplicates
        self.pr_created_this_session = False
        
        # Cleanup attributes
        self.auto_cleanup = auto_cleanup
        self._extracted_files = set()
        self._temp_dirs = set()
        
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
        
        # Set up virtual root path - use current working directory with repo-specific subdirectory
        # This allows repo-map to create its cache files properly
        current_dir = Path.cwd()
        safe_repo_name = self.repo_name.replace('/', '_').replace(':', '_')
        self.root = utils.safe_abs_path(f"{current_dir}/.redis_repos/{safe_repo_name}")
        
        # Ensure the directory exists for cache files
        os.makedirs(self.root, exist_ok=True)
        
        # Tier access control
        self.tier_controller = get_tier_access_controller()
        
        # Validate tier access before loading repository data
        self._validate_tier_access()
        
        # Load repository data from Redis
        self._load_repository_data()
        
        # Extract files to local directory for repo-map compatibility
        self._extract_files_to_local()
        
        # Set up virtual filesystem IO wrapper
        self.io = VirtualFileSystemIO(self.original_io, self.virtual_fs, self.root)
        
        # Set repo to self for GitRepo compatibility
        self.repo = self
        
        # Initialize RepoMap for cloud-based implementation
        self._init_repomap()
        
        # Ensure files are properly extracted for RepoMap
        self._ensure_files_for_repomap()
        
        # Build RepoMap cache by triggering initial scan
        self._build_repomap_cache()
        
        logger.info(f"Initialized RedisRepoManager for {self.repo_name}")
    
    def get_file_content_for_pr(self, filename):
        """
        Get file content for PR creation, prioritizing modified content from IO wrapper.
        
        Args:
            filename: Path to the file
            
        Returns:
            File content as string (modified if available, otherwise original)
        """
        # Check if we have modified content in the IO wrapper
        if hasattr(self.io, 'get_modified_content'):
            modified_content = self.io.get_modified_content(filename)
            if modified_content is not None:
                return modified_content
        
        # Fall back to virtual filesystem content
        if self.virtual_fs:
            # Convert absolute path to relative if needed
            file_path = Path(filename)
            if file_path.is_absolute():
                try:
                    rel_path = file_path.relative_to(self.root)
                    return self.virtual_fs.extract_file_with_context(str(rel_path))
                except ValueError:
                    pass
            
            return self.virtual_fs.extract_file_with_context(str(filename))
        
        return None
    
    def _init_repomap(self) -> None:
        """Initialize RepoMap for the cloud-based repository."""
        try:
            # Initialize RepoMap with the repository root and IO wrapper
            main_model = None
            if self.models:
                # Try to get the main model from models object
                if hasattr(self.models, 'main_model'):
                    main_model = self.models.main_model
                elif hasattr(self.models, 'model'):
                    main_model = self.models.model
            
            self.repo_map = RepoMap(
                map_tokens=1024,  # Default token limit for repo map
                root=self.root,
                main_model=main_model,
                io=self.io,
                verbose=False,
                refresh="auto"
            )
            
            # Override RepoMap's get_mtime method to work with virtual filesystem
            original_get_mtime = self.repo_map.get_mtime
            def custom_get_mtime(fname):
                try:
                    return self.io.get_mtime(fname)
                except:
                    return original_get_mtime(fname)
            
            self.repo_map.get_mtime = custom_get_mtime
            logger.info(f"Initialized RepoMap for {self.repo_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize RepoMap for {self.repo_name}: {e}")
            self.repo_map = None
    
    def _ensure_files_for_repomap(self) -> None:
        """Ensure all files are properly extracted for RepoMap to access."""
        if not self.virtual_fs:
            return
        
        try:
            tracked_files = self.get_tracked_files()
            extracted_count = 0
            
            for file_path in tracked_files:
                local_file_path = Path(self.root) / file_path
                
                # Only extract if file doesn't exist locally
                if not local_file_path.exists():
                    try:
                        content = self.virtual_fs.extract_file_with_context(file_path)
                        if content is not None:
                            # Ensure directory exists
                            local_file_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            # Write file content
                            with open(local_file_path, 'w', encoding='utf-8') as f:
                                f.write(content)
                            
                            extracted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to extract file {file_path} for RepoMap: {e}")
                        continue
            
            if extracted_count > 0:
                logger.info(f"Extracted {extracted_count} additional files for RepoMap")
                
        except Exception as e:
            logger.error(f"Error ensuring files for RepoMap: {e}")
    
    def _build_repomap_cache(self) -> None:
        """Build RepoMap cache by triggering initial file scan."""
        if not self.repo_map:
            return
        
        try:
            tracked_files = self.get_tracked_files()
            if not tracked_files:
                logger.info("No tracked files found, skipping RepoMap cache build")
                return
            
            # Convert to absolute paths for RepoMap
            abs_files = []
            for file_path in tracked_files:
                abs_path = str(Path(self.root) / file_path)
                if Path(abs_path).exists():
                    abs_files.append(abs_path)
            
            if not abs_files:
                logger.info("No physical files found for RepoMap cache build")
                return
            
            logger.info(f"Building RepoMap cache for {len(abs_files)} files...")
            
            # Trigger RepoMap to scan files and build cache by calling get_ranked_tags
            # This will force RepoMap to process files and create .cosmos.tags.cache
            try:
                self.repo_map.get_ranked_tags(
                    chat_fnames=[],  # No chat files for initial scan
                    other_fnames=abs_files[:50],  # Limit to first 50 files to avoid timeout
                    mentioned_fnames=set(),
                    mentioned_idents=set()
                )
                logger.info(f"Successfully built RepoMap cache for {self.repo_name}")
            except Exception as e:
                logger.warning(f"Error during RepoMap cache build: {e}")
                # Try a smaller subset if the full scan fails
                if len(abs_files) > 10:
                    logger.info("Retrying with smaller file set...")
                    self.repo_map.get_ranked_tags(
                        chat_fnames=[],
                        other_fnames=abs_files[:10],
                        mentioned_fnames=set(),
                        mentioned_idents=set()
                    )
                    logger.info(f"Built RepoMap cache with reduced file set for {self.repo_name}")
                else:
                    raise
            
        except Exception as e:
            logger.warning(f"Failed to build RepoMap cache for {self.repo_name}: {e}")
            # Don't fail initialization if cache building fails
    
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
        """Load repository data from Redis and initialize virtual filesystem with auto-recovery."""
        max_retries = 2
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                repo_data = self.redis_cache.get_repository_data_cached(self.repo_name)
                
                if not repo_data:
                    if retry_count == 0:
                        logger.warning(f"No repository data found for {self.repo_name}, attempting auto-recovery")
                        self._auto_recover_repository()
                        retry_count += 1
                        continue
                    else:
                        logger.warning(f"No repository data found for {self.repo_name} after recovery attempt")
                        # Initialize empty virtual filesystem
                        self.virtual_fs = IntelligentVirtualFileSystem("", "", self.repo_name)
                        return
                
                content_md = repo_data.get('content', '')
                tree_txt = repo_data.get('tree', '')
                
                # Validate that we have meaningful content
                if not content_md or not content_md.strip():
                    if retry_count == 0:
                        logger.warning(f"Empty or missing content.md for {self.repo_name}, attempting auto-recovery")
                        self._auto_recover_repository()
                        retry_count += 1
                        continue
                    else:
                        logger.error(f"Empty content.md for {self.repo_name} after recovery attempt")
                        # Initialize empty virtual filesystem as fallback
                        self.virtual_fs = IntelligentVirtualFileSystem("", "", self.repo_name)
                        return
                
                # Get storage directory from environment or use default
                storage_dir = os.getenv('STORAGE_DIR', '/tmp/repo_storage')
                
                # Initialize virtual filesystem with indexing support
                self.virtual_fs = IntelligentVirtualFileSystem(
                    content_md, 
                    tree_txt, 
                    self.repo_name,
                    repo_storage_dir=storage_dir
                )
                
                # Validate virtual filesystem was created successfully
                if not self.virtual_fs or not hasattr(self.virtual_fs, 'get_tracked_files'):
                    if retry_count == 0:
                        logger.warning(f"Failed to initialize virtual filesystem for {self.repo_name}, attempting auto-recovery")
                        self._auto_recover_repository()
                        retry_count += 1
                        continue
                    else:
                        logger.error(f"Failed to initialize virtual filesystem for {self.repo_name} after recovery")
                        self.virtual_fs = IntelligentVirtualFileSystem("", "", self.repo_name)
                        return
                
                # Extract files to disk for repo-map compatibility
                self._extract_files_to_disk()
                
                logger.info(f"Successfully loaded repository data for {self.repo_name}")
                return
                
            except Exception as e:
                if retry_count == 0:
                    logger.warning(f"Error loading repository data for {self.repo_name}: {e}, attempting auto-recovery")
                    self._auto_recover_repository()
                    retry_count += 1
                    continue
                else:
                    logger.error(f"Error loading repository data for {self.repo_name} after recovery: {e}")
                    # Initialize empty virtual filesystem as fallback
                    self.virtual_fs = IntelligentVirtualFileSystem("", "", self.repo_name)
                    return
    
    def _auto_recover_repository(self) -> None:
        """Auto-recover repository by clearing cache and re-fetching from GitHub."""
        try:
            if self.original_io:
                self.original_io.tool_output(f"🔄 Auto-recovering repository {self.repo_name}...")
                self.original_io.tool_output("   Clearing corrupted cache and re-fetching from GitHub...")
            
            logger.info(f"Starting auto-recovery for {self.repo_name}")
            
            # Step 1: Clear the corrupted cache
            if hasattr(self.redis_cache, 'smart_invalidate'):
                success = self.redis_cache.smart_invalidate(self.repo_name)
                if success:
                    logger.info(f"Successfully cleared cache for {self.repo_name}")
                else:
                    logger.warning(f"Failed to clear cache for {self.repo_name}")
            
            # Step 2: Clear local cache files if they exist
            import shutil
            local_cache_dir = f"/tmp/repo_storage/{self.repo_name}"
            if os.path.exists(local_cache_dir):
                try:
                    shutil.rmtree(local_cache_dir)
                    logger.info(f"Cleared local cache directory: {local_cache_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clear local cache directory {local_cache_dir}: {e}")
            
            # Step 3: Re-fetch repository from GitHub
            try:
                from cosmos.repo_fetch import fetch_and_store_repo
                
                # Convert repo_name to URL format if needed
                repo_url = self.repo_url
                if not repo_url or repo_url == "unknown":
                    if '/' in self.repo_name:
                        repo_url = f"https://github.com/{self.repo_name}"
                    else:
                        logger.error(f"Cannot determine GitHub URL for {self.repo_name}")
                        return
                
                if self.original_io:
                    self.original_io.tool_output(f"   Re-fetching {repo_url} from GitHub...")
                
                success = fetch_and_store_repo(repo_url)
                if success:
                    logger.info(f"Successfully re-fetched repository {self.repo_name}")
                    if self.original_io:
                        self.original_io.tool_output(f"✅ Repository {self.repo_name} recovered successfully!")
                else:
                    logger.error(f"Failed to re-fetch repository {self.repo_name}")
                    if self.original_io:
                        self.original_io.tool_error(f"❌ Failed to recover repository {self.repo_name}")
                        
            except ImportError:
                logger.error("repo_fetch module not available for auto-recovery")
                if self.original_io:
                    self.original_io.tool_error("❌ Auto-recovery module not available")
            except Exception as e:
                logger.error(f"Error during repository re-fetch: {e}")
                if self.original_io:
                    self.original_io.tool_error(f"❌ Error during recovery: {e}")
                    
        except Exception as e:
            logger.error(f"Error during auto-recovery for {self.repo_name}: {e}")
            if self.original_io:
                self.original_io.tool_error(f"❌ Auto-recovery failed: {e}")
    
    def _validate_repository_data(self, repo_data: dict) -> bool:
        """Validate that repository data is complete and usable."""
        if not repo_data:
            return False
        
        content_md = repo_data.get('content', '')
        tree_txt = repo_data.get('tree', '')
        
        # Check if content.md exists and has meaningful content
        if not content_md or not content_md.strip():
            logger.warning(f"Missing or empty content.md for {self.repo_name}")
            return False
        
        # Check if content.md has the expected structure
        if '# ' not in content_md and '## ' not in content_md:
            logger.warning(f"content.md appears to be malformed for {self.repo_name}")
            return False
        
        # Check if tree.txt exists
        if not tree_txt or not tree_txt.strip():
            logger.warning(f"Missing or empty tree.txt for {self.repo_name}")
            return False
        
        # Basic validation passed
        return True
    
    def get_repository_health(self) -> dict:
        """Get health status of the repository cache."""
        health = {
            'repo_name': self.repo_name,
            'cache_healthy': False,
            'virtual_fs_healthy': False,
            'files_extracted': 0,
            'issues': []
        }
        
        try:
            # Check cache data
            repo_data = self.redis_cache.get_repository_data_cached(self.repo_name)
            if self._validate_repository_data(repo_data):
                health['cache_healthy'] = True
            else:
                health['issues'].append('Cache data is missing or corrupted')
            
            # Check virtual filesystem
            if self.virtual_fs and hasattr(self.virtual_fs, 'get_tracked_files'):
                try:
                    tracked_files = self.virtual_fs.get_tracked_files()
                    health['files_extracted'] = len(tracked_files)
                    health['virtual_fs_healthy'] = len(tracked_files) > 0
                    if len(tracked_files) == 0:
                        health['issues'].append('No files found in virtual filesystem')
                except Exception as e:
                    health['issues'].append(f'Virtual filesystem error: {e}')
            else:
                health['issues'].append('Virtual filesystem not initialized')
            
        except Exception as e:
            health['issues'].append(f'Health check error: {e}')
        
        return health
    
    def _extract_files_to_local(self) -> None:
        """Extract files from virtual filesystem to local directory for repo-map compatibility."""
        if not self.virtual_fs:
            return
        
        try:
            # Get all tracked files
            tracked_files = self.virtual_fs.get_tracked_files()
            
            for file_path in tracked_files:
                try:
                    # Get file content from virtual filesystem
                    content = self.virtual_fs.extract_file_with_context(file_path)
                    if content is not None:
                        # Create local file path
                        local_file_path = Path(self.root) / file_path
                        
                        # Ensure directory exists
                        local_file_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Track the directory for cleanup
                        if self.auto_cleanup:
                            self._temp_dirs.add(local_file_path.parent)
                        
                        # Write file content
                        with open(local_file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        # Track the file for cleanup
                        if self.auto_cleanup:
                            self._extracted_files.add(local_file_path)
                            
                except Exception as e:
                    logger.warning(f"Failed to extract file {file_path}: {e}")
                    continue
            
            logger.info(f"Extracted {len(tracked_files)} files to local directory for repo-map")
            
        except Exception as e:
            logger.error(f"Error extracting files to local directory: {e}")
    
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
                        
                        # Track the directory for cleanup
                        if self.auto_cleanup:
                            self._temp_dirs.add(physical_path.parent)
                        
                        # Write content to disk
                        with open(physical_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        # Track the file for cleanup
                        if self.auto_cleanup:
                            self._extracted_files.add(physical_path)
                        
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
        Buffer-based commit operation for Redis backend.
        
        In Redis Cloud mode, this method stages changes in a buffer and creates
        a pull request if PR mode is enabled, otherwise just logs the changes.
        
        Args:
            fnames: List of filenames to commit
            context: Commit context
            message: Commit message
            cosmos_edits: Whether these are cosmos edits
            coder: Coder instance
            
        Returns:
            tuple: (commit_hash, commit_message) if successful, None otherwise
        """
        # Check if we have any modified files in the buffer
        modified_files = []
        if hasattr(self.io, 'modified_files') and self.io.modified_files:
            modified_files = list(self.io.modified_files.keys())
        
        if not modified_files and not fnames:
            if self.original_io:
                self.original_io.tool_output("No changes to commit in buffer.")
            return None
        
        # Generate commit message if not provided
        if not message:
            if context and coder:
                try:
                    message = self.get_commit_message(
                        diffs="Changes made via cosmos chat",
                        context=context,
                        user_language=getattr(coder, 'commit_language', None)
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate commit message: {e}")
                    message = f"Update files in {self.repo_name}"
            else:
                message = f"Update files in {self.repo_name}"
        
        # If PR mode is enabled, create a pull request (only for own repos)
        if self.create_pull_request:
            result = self.commit_and_create_pr(fnames, context, message, cosmos_edits, coder)
            if result and result.get('commit_hash'):
                return result['commit_hash'], result['commit_message']
            return None
        
        # Otherwise, just stage changes in buffer and log
        commit_hash = f"buffer_{int(time.time())}"
        
        if self.original_io:
            self.original_io.tool_output(f"📋 Changes staged in buffer: {message}", bold=True)
            if modified_files:
                self.original_io.tool_output(f"📁 Modified files ({len(modified_files)}):")
                for i, file_path in enumerate(modified_files[:5]):
                    self.original_io.tool_output(f"   {i+1}. {file_path}")
                if len(modified_files) > 5:
                    self.original_io.tool_output(f"   ... and {len(modified_files) - 5} more files")
            
            # Check if this is an external repository and provide appropriate guidance
            if self.repo_name and '/' in self.repo_name:
                owner = self.repo_name.split('/')[0]
                github_token = self.github_token or os.getenv('GITHUB_TOKEN')
                
                if github_token:
                    try:
                        import requests
                        headers = {
                            'Authorization': f'token {github_token}',
                            'Accept': 'application/vnd.github.v3+json'
                        }
                        
                        user_response = requests.get("https://api.github.com/user", headers=headers)
                        if user_response.status_code == 200:
                            current_user = user_response.json()['login']
                            
                            if owner.lower() != current_user.lower():
                                self.original_io.tool_output(f"\n⚠️  Note: {self.repo_name} belongs to {owner}, not you ({current_user})")
                                self.original_io.tool_output("💡 Next steps for external repositories:")
                                self.original_io.tool_output("   • Use /buffer to view all staged changes")
                                self.original_io.tool_output("   • Copy the changes to your own repository")
                                self.original_io.tool_output("   • Create PRs only in repositories you own")
                                return commit_hash, message
                    except:
                        pass  # Ignore errors in ownership check for buffer mode
            
            self.original_io.tool_output("\n💡 Next steps:")
            self.original_io.tool_output("   • Use /buffer to view all staged changes")
            self.original_io.tool_output("   • Use /pr on to enable pull request mode (own repos only)")
            self.original_io.tool_output("   • Use /commit to create a PR with your changes")
        
        logger.info(f"Staged changes in buffer for {self.repo_name}: {message}")
        return commit_hash, message
    
    def get_buffer_status(self):
        """
        Get the current status of the buffer (modified files).
        
        Returns:
            dict: Buffer status information
        """
        status = {
            'modified_files': [],
            'file_count': 0,
            'pr_mode_enabled': self.create_pull_request,
            'pr_base_branch': self.pr_base_branch,
            'pr_draft': self.pr_draft
        }
        
        if hasattr(self.io, 'modified_files') and self.io.modified_files:
            status['modified_files'] = list(self.io.modified_files.keys())
            status['file_count'] = len(self.io.modified_files)
        
        return status
    
    def clear_buffer(self):
        """
        Clear the buffer of modified files.
        """
        if hasattr(self.io, 'modified_files'):
            self.io.modified_files.clear()
            if self.original_io:
                self.original_io.tool_output("Buffer cleared.")
        
        logger.info(f"Cleared buffer for {self.repo_name}")

    def commit_and_create_pr(self, fnames=None, context=None, message=None, cosmos_edits=False, coder=None):
        """
        Create a GitHub pull request for Redis-based repositories.
        
        Args:
            fnames: List of filenames to commit
            context: Commit context  
            message: Commit message
            cosmos_edits: Whether these are cosmos edits
            coder: Coder instance
            
        Returns:
            dict: Contains commit info and PR details if successful
        """
        # Check if PR was already created this session to prevent duplicates
        if self.pr_created_this_session:
            logger.info(f"PR already created this session for {self.repo_name}, skipping duplicate")
            return {
                'commit_hash': 'duplicate_pr_prevented',
                'commit_message': message or f"Update files in {self.repo_name}",
                'pr_info': {'duplicate': True}
            }
            
        if not self.create_pull_request:
            # If PR mode is disabled, just return a simulated commit
            if message:
                logger.info(f"Simulated commit for {self.repo_name}: {message}")
            return {
                'commit_hash': 'simulated_hash',
                'commit_message': message or f"Update files in {self.repo_name}",
                'pr_info': None
            }
        
        # EARLY OWNERSHIP CHECK - Check repository ownership before attempting PR
        if self.repo_name and '/' in self.repo_name:
            owner = self.repo_name.split('/')[0]
            
            # Check if we have GitHub token
            github_token = self.github_token or os.getenv('GITHUB_TOKEN')
            if not github_token:
                self.original_io.tool_error("❌ GitHub token required for PR creation. Set GITHUB_TOKEN environment variable")
                return {
                    'commit_hash': 'error_no_token',
                    'commit_message': message or f"Update files in {self.repo_name}",
                    'pr_info': None
                }
            
            # Quick ownership check using GitHub API
            try:
                import requests
                headers = {
                    'Authorization': f'token {github_token}',
                    'Accept': 'application/vnd.github.v3+json'
                }
                
                user_response = requests.get("https://api.github.com/user", headers=headers)
                if user_response.status_code == 200:
                    current_user = user_response.json()['login']
                    
                    if owner.lower() != current_user.lower():
                        self.original_io.tool_error(f"❌ Cannot create PR: Repository {self.repo_name} belongs to {owner}, but you are {current_user}")
                        self.original_io.tool_output("💡 You can only create pull requests for your own repositories")
                        self.original_io.tool_output("📋 Your changes are safely stored in the buffer")
                        self.original_io.tool_output("🔄 Use /buffer to view changes, then copy them to your own repository")
                        return {
                            'commit_hash': 'error_not_owner',
                            'commit_message': message or f"Update files in {self.repo_name}",
                            'pr_info': None
                        }
                else:
                    self.original_io.tool_error("❌ Failed to verify GitHub user. Check your GITHUB_TOKEN")
                    return {
                        'commit_hash': 'error_token_invalid',
                        'commit_message': message or f"Update files in {self.repo_name}",
                        'pr_info': None
                    }
                    
            except Exception as e:
                self.original_io.tool_error(f"❌ Error checking repository ownership: {e}")
                return {
                    'commit_hash': 'error_ownership_check',
                    'commit_message': message or f"Update files in {self.repo_name}",
                    'pr_info': None
                }
        
        # Import the GitHub PR module
        try:
            from cosmos.github_pr import create_pull_request_workflow
        except ImportError as e:
            self.original_io.tool_error(f"GitHub PR module not available: {e}")
            return {
                'commit_hash': 'error_no_module',
                'commit_message': message or f"Update files in {self.repo_name}",
                'pr_info': None
            }
        
        # Generate commit message if not provided
        if not message:
            if context and coder:
                try:
                    # Get changed files for commit message generation
                    changed_files = []
                    if fnames:
                        changed_files = [str(fname) for fname in fnames]
                    
                    message = self.get_commit_message(
                        diffs="Changes made via cosmos chat",
                        context=context,
                        user_language=getattr(coder, 'commit_language', None)
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate commit message: {e}")
                    message = f"Update files in {self.repo_name}"
            else:
                message = f"Update files in {self.repo_name}"
        
        # Get list of changed files
        changed_files = []
        if fnames:
            changed_files = [str(fname) for fname in fnames]
        elif self.virtual_fs:
            # Get all tracked files as potential changes
            try:
                changed_files = self.virtual_fs.get_tracked_files()[:10]  # Limit to first 10 files
            except Exception as e:
                logger.warning(f"Failed to get changed files: {e}")
                changed_files = ["Updated files"]
        
        try:
            # Create the pull request using the workflow
            pr_info = create_pull_request_workflow(
                repo=self,
                commit_message=message,
                changed_files=changed_files,
                io=self.original_io,
                base_branch=self.pr_base_branch,
                draft=self.pr_draft,
                github_token=github_token
            )
            
            if pr_info:
                # Mark that PR was created successfully
                self.pr_created_this_session = True
                
                commit_hash = f"pr_{pr_info.get('number', 'unknown')}"
                result = {
                    'commit_hash': commit_hash,
                    'commit_message': message,
                    'pr_info': pr_info
                }
                
                # Log successful PR creation
                logger.info(f"Created PR #{pr_info.get('number')} for {self.repo_name}: {message}")
                
                return result
            else:
                self.original_io.tool_error("Failed to create pull request")
                return {
                    'commit_hash': 'error_pr_failed',
                    'commit_message': message,
                    'pr_info': None
                }
                
        except Exception as e:
            error_msg = f"Error creating pull request: {str(e)}"
            logger.error(error_msg)
            self.original_io.tool_error(error_msg)
            return {
                'commit_hash': 'error_exception',
                'commit_message': message,
                'pr_info': None
            }
    
    def get_file_content_for_pr(self, file_path: str) -> Optional[str]:
        """
        Get file content for PR creation, prioritizing modified content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string, or None if not found
        """
        try:
            # First check if we have modified content from VirtualFileSystemIO
            if hasattr(self, 'io') and hasattr(self.io, 'get_modified_content'):
                modified_content = self.io.get_modified_content(file_path)
                if modified_content is not None:
                    logger.debug(f"Found modified content for {file_path}")
                    return modified_content
            
            # Try to read from physical filesystem (where edits are written)
            try:
                physical_path = Path(self.root) / file_path
                if physical_path.exists():
                    with open(physical_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    logger.debug(f"Read content from physical file: {file_path}")
                    return content
            except Exception as e:
                logger.debug(f"Could not read from physical file {file_path}: {e}")
            
            # Try virtual filesystem as fallback
            if self.virtual_fs:
                content = self.virtual_fs.extract_file_with_context(file_path)
                if content is not None:
                    logger.debug(f"Read content from virtual filesystem: {file_path}")
                    return content
            
            # Try content indexer as last resort
            if self.content_indexer:
                content = self.content_indexer.get_file_content(file_path)
                if content is not None:
                    logger.debug(f"Read content from content indexer: {file_path}")
                    return content
            
            # Try IO wrapper
            if hasattr(self, 'io'):
                content = self.io.read_text(file_path, silent=True)
                if content is not None:
                    logger.debug(f"Read content from IO wrapper: {file_path}")
                    return content
            
            logger.warning(f"Could not get content for file: {file_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting file content for {file_path}: {e}")
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
        
        # Try both the original path and normalized path
        return (self.virtual_fs.file_exists(path) or 
                self.virtual_fs.file_exists(self.normalize_path(path)))
    
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
            return content if content is not None else ""
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
    
    # RepoMap integration methods
    
    def get_repo_map(self, chat_files=None, other_files=None, mentioned_fnames=None, 
                     mentioned_idents=None, force_refresh=False):
        """
        Get repository map using RepoMap for cloud-based implementation.
        
        Args:
            chat_files: List of files currently in chat context
            other_files: List of other files to consider for the map
            mentioned_fnames: Set of mentioned filenames
            mentioned_idents: Set of mentioned identifiers
            force_refresh: Whether to force refresh the map
            
        Returns:
            Repository map as string or None if not available
        """
        if not self.repo_map:
            return None
        
        try:
            # Ensure we have file lists
            if chat_files is None:
                chat_files = []
            if other_files is None:
                other_files = self.get_tracked_files()
            
            # Convert relative paths to absolute paths for RepoMap
            # Ensure files exist in the extracted directory structure
            abs_chat_files = []
            abs_other_files = []
            
            for f in chat_files:
                if f:
                    abs_path = str(Path(self.root) / f)
                    if Path(abs_path).exists() or self.file_exists(f):
                        abs_chat_files.append(abs_path)
            
            for f in other_files:
                if f:
                    abs_path = str(Path(self.root) / f)
                    if Path(abs_path).exists() or self.file_exists(f):
                        abs_other_files.append(abs_path)
            
            # Get the repository map
            repo_map = self.repo_map.get_repo_map(
                chat_files=abs_chat_files,
                other_files=abs_other_files,
                mentioned_fnames=mentioned_fnames,
                mentioned_idents=mentioned_idents,
                force_refresh=force_refresh
            )
            
            return repo_map
            
        except Exception as e:
            logger.warning(f"Error generating repo map for {self.repo_name}: {e}")
            return None
    
    def get_ranked_tags_map(self, chat_files=None, other_files=None, max_map_tokens=None,
                           mentioned_fnames=None, mentioned_idents=None, force_refresh=False):
        """
        Get ranked tags map using RepoMap.
        
        Args:
            chat_files: List of files currently in chat context
            other_files: List of other files to consider
            max_map_tokens: Maximum tokens for the map
            mentioned_fnames: Set of mentioned filenames
            mentioned_idents: Set of mentioned identifiers
            force_refresh: Whether to force refresh
            
        Returns:
            Ranked tags map as string or None if not available
        """
        if not self.repo_map:
            return None
        
        try:
            # Ensure we have file lists
            if chat_files is None:
                chat_files = []
            if other_files is None:
                other_files = self.get_tracked_files()
            
            # Convert relative paths to absolute paths for RepoMap
            # Ensure files exist in the extracted directory structure
            abs_chat_files = []
            abs_other_files = []
            
            for f in chat_files:
                if f:
                    abs_path = str(Path(self.root) / f)
                    if Path(abs_path).exists() or self.file_exists(f):
                        abs_chat_files.append(abs_path)
            
            for f in other_files:
                if f:
                    abs_path = str(Path(self.root) / f)
                    if Path(abs_path).exists() or self.file_exists(f):
                        abs_other_files.append(abs_path)
            
            # Get the ranked tags map
            ranked_map = self.repo_map.get_ranked_tags_map(
                chat_fnames=abs_chat_files,
                other_fnames=abs_other_files,
                max_map_tokens=max_map_tokens,
                mentioned_fnames=mentioned_fnames,
                mentioned_idents=mentioned_idents,
                force_refresh=force_refresh
            )
            
            return ranked_map
            
        except Exception as e:
            logger.warning(f"Error generating ranked tags map for {self.repo_name}: {e}")
            return None
    
    def refresh_repo_map(self):
        """Refresh the repository map cache."""
        if self.repo_map:
            try:
                # Clear the map cache to force refresh
                self.repo_map.map_cache.clear()
                self.repo_map.tree_cache.clear()
                self.repo_map.tree_context_cache.clear()
                
                # Clear the tags cache to force rebuild
                if hasattr(self.repo_map, 'TAGS_CACHE'):
                    if hasattr(self.repo_map.TAGS_CACHE, 'clear'):
                        self.repo_map.TAGS_CACHE.clear()
                
                # Re-extract files to ensure RepoMap has latest content
                self._ensure_files_for_repomap()
                
                # Rebuild the cache
                self._build_repomap_cache()
                
                logger.info(f"Refreshed repo map cache for {self.repo_name}")
            except Exception as e:
                logger.warning(f"Error refreshing repo map for {self.repo_name}: {e}")
    
    def get_repomap_cache_path(self):
        """Get the path to the RepoMap cache directory."""
        if self.repo_map:
            return Path(self.root) / self.repo_map.TAGS_CACHE_DIR
        return None
    
    def repomap_cache_exists(self):
        """Check if RepoMap cache directory exists."""
        cache_path = self.get_repomap_cache_path()
        return cache_path and cache_path.exists()
    
    def force_rebuild_repomap_cache(self):
        """Force rebuild of RepoMap cache."""
        if not self.repo_map:
            return False
        
        try:
            # Remove existing cache directory
            cache_path = self.get_repomap_cache_path()
            if cache_path and cache_path.exists():
                import shutil
                shutil.rmtree(cache_path)
                logger.info(f"Removed existing RepoMap cache at {cache_path}")
            
            # Reload the cache
            self.repo_map.load_tags_cache()
            
            # Rebuild cache
            self._build_repomap_cache()
            
            logger.info(f"Force rebuilt RepoMap cache for {self.repo_name}")
            return True
        except Exception as e:
            logger.error(f"Error force rebuilding RepoMap cache: {e}")
            return False
    
    def set_repo_map_tokens(self, max_tokens):
        """Set the maximum tokens for repository map."""
        if self.repo_map:
            self.repo_map.max_map_tokens = max_tokens
            # Clear cache to ensure new token limit is applied
            self.repo_map.map_cache.clear()
            logger.info(f"Set repo map tokens to {max_tokens} for {self.repo_name}")
    
    def configure_repo_map(self, **kwargs):
        """Configure RepoMap settings."""
        if not self.repo_map:
            return False
        
        try:
            updated = False
            
            if 'max_tokens' in kwargs:
                self.repo_map.max_map_tokens = kwargs['max_tokens']
                updated = True
            
            if 'refresh_mode' in kwargs:
                self.repo_map.refresh = kwargs['refresh_mode']
                updated = True
            
            if 'verbose' in kwargs:
                self.repo_map.verbose = kwargs['verbose']
                updated = True
            
            if 'map_mul_no_files' in kwargs:
                self.repo_map.map_mul_no_files = kwargs['map_mul_no_files']
                updated = True
            
            if updated:
                # Clear cache when configuration changes
                self.repo_map.map_cache.clear()
                logger.info(f"Updated RepoMap configuration for {self.repo_name}")
            
            return updated
        except Exception as e:
            logger.warning(f"Error configuring RepoMap for {self.repo_name}: {e}")
            return False
    
    def get_repo_map_stats(self):
        """Get repository map statistics."""
        if not self.repo_map:
            return None
        
        try:
            stats = {
                'max_map_tokens': self.repo_map.max_map_tokens,
                'cache_size': len(getattr(self.repo_map, 'map_cache', {})),
                'tree_cache_size': len(getattr(self.repo_map, 'tree_cache', {})),
                'processing_time': getattr(self.repo_map, 'map_processing_time', 0),
                'root': self.repo_map.root,
                'refresh_mode': self.repo_map.refresh,
                'repo_name': self.repo_name,
                'tracked_files_count': len(self.get_tracked_files()),
                'extracted_files_exist': any(Path(self.root).glob('*'))
            }
            return stats
        except Exception as e:
            logger.warning(f"Error getting repo map stats for {self.repo_name}: {e}")
            return None
    
    def is_repomap_available(self):
        """Check if RepoMap is available and properly initialized."""
        return self.repo_map is not None
    
    def get_repomap_supported_files(self):
        """Get list of files that RepoMap can process (have language support)."""
        if not self.repo_map:
            return []
        
        try:
            from grep_ast import filename_to_lang
            
            tracked_files = self.get_tracked_files()
            supported_files = []
            
            for file_path in tracked_files:
                abs_path = str(Path(self.root) / file_path)
                if filename_to_lang(abs_path):
                    supported_files.append(file_path)
            
            return supported_files
        except Exception as e:
            logger.warning(f"Error getting RepoMap supported files: {e}")
            return []
    
    def verify_repomap_integration(self):
        """
        Verify that RepoMap integration is working correctly.
        
        Returns:
            Dict with verification results
        """
        results = {
            'repomap_available': False,
            'cache_exists': False,
            'files_extracted': False,
            'tags_extracted': False,
            'map_generated': False,
            'errors': []
        }
        
        try:
            # Check if RepoMap is available
            results['repomap_available'] = self.is_repomap_available()
            
            if not results['repomap_available']:
                results['errors'].append("RepoMap not initialized")
                return results
            
            # Check if cache exists
            results['cache_exists'] = self.repomap_cache_exists()
            
            # Check if files are extracted
            tracked_files = self.get_tracked_files()
            if tracked_files:
                extracted_count = 0
                for file_path in tracked_files:
                    local_path = Path(self.root) / file_path
                    if local_path.exists():
                        extracted_count += 1
                
                results['files_extracted'] = extracted_count > 0
                results['extracted_count'] = extracted_count
                results['total_files'] = len(tracked_files)
            
            # Test tag extraction
            if tracked_files and results['files_extracted']:
                try:
                    # Test with first Python file
                    python_files = [f for f in tracked_files if f.endswith('.py')]
                    if python_files:
                        test_file = python_files[0]
                        abs_path = str(Path(self.root) / test_file)
                        rel_path = self.repo_map.get_rel_fname(abs_path)
                        tags = self.repo_map.get_tags(abs_path, rel_path)
                        results['tags_extracted'] = len(tags) > 0
                        results['sample_tags_count'] = len(tags)
                except Exception as e:
                    results['errors'].append(f"Tag extraction failed: {e}")
            
            # Test map generation
            if tracked_files:
                try:
                    repo_map = self.get_repo_map(
                        chat_files=[],
                        other_files=tracked_files[:5]  # Test with first 5 files
                    )
                    results['map_generated'] = repo_map is not None and len(repo_map) > 0
                    if results['map_generated']:
                        results['map_length'] = len(repo_map)
                except Exception as e:
                    results['errors'].append(f"Map generation failed: {e}")
            
        except Exception as e:
            results['errors'].append(f"Verification failed: {e}")
        
        return results
    
    # Cleanup methods
    
    def cleanup_extracted_files(self) -> None:
        """
        Clean up all extracted temporary files and directories.
        
        This method removes all files and directories that were created
        during the file extraction process.
        """
        if not self.auto_cleanup:
            logger.debug("Auto cleanup is disabled")
            return
        
        try:
            # Remove extracted files
            files_removed = 0
            for file_path in self._extracted_files:
                try:
                    if file_path.exists():
                        file_path.unlink()
                        files_removed += 1
                        logger.debug(f"Removed extracted file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove file {file_path}: {e}")
            
            # Remove empty directories (in reverse order to remove nested dirs first)
            dirs_removed = 0
            sorted_dirs = sorted(self._temp_dirs, key=lambda x: str(x), reverse=True)
            for dir_path in sorted_dirs:
                try:
                    if dir_path.exists() and dir_path.is_dir():
                        # Only remove if directory is empty
                        if not any(dir_path.iterdir()):
                            dir_path.rmdir()
                            dirs_removed += 1
                            logger.debug(f"Removed empty directory: {dir_path}")
                except Exception as e:
                    logger.debug(f"Could not remove directory {dir_path}: {e}")
            
            # Clear tracking sets
            self._extracted_files.clear()
            self._temp_dirs.clear()
            
            if files_removed > 0 or dirs_removed > 0:
                logger.info(f"Cleanup complete: removed {files_removed} files and {dirs_removed} directories")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - performs automatic cleanup."""
        if self.auto_cleanup:
            self.cleanup_extracted_files()
        
    def __del__(self):
        """Destructor - performs cleanup if auto_cleanup is enabled."""
        try:
            if hasattr(self, 'auto_cleanup') and self.auto_cleanup:
                self.cleanup_extracted_files()
        except Exception:
            # Suppress all exceptions in destructor
            pass
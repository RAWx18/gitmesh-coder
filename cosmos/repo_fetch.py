import os
import time
import logging
import re
import traceback
from typing import Dict, Any, Optional, Tuple
from urllib.parse import urlparse
from dotenv import load_dotenv
from gitingest import ingest

from .redis_cache import SmartRedisCache
from .tier_manager import TierManager, TierValidationError, TierAccessDeniedError
from .content_indexer import ContentIndexer
from .config import initialize_configuration

# Load environment variables from the .env file when the module is imported.
load_dotenv()

# Configure comprehensive logging
logger = logging.getLogger(__name__)

# Set up detailed logging format if not already configured
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- Configuration ---
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# Error categories for monitoring
class ErrorCategory:
    NETWORK = "network"
    REDIS = "redis"
    TIER_ACCESS = "tier_access"
    VALIDATION = "validation"
    GITHUB_API = "github_api"
    PARSING = "parsing"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class RepoFetchError(Exception):
    """Base exception for repository fetch operations."""
    
    def __init__(self, message: str, category: str = ErrorCategory.UNKNOWN, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.category = category
        self.details = details or {}
        self.timestamp = time.time()


class NetworkError(RepoFetchError):
    """Network-related errors during repository fetch."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.NETWORK, details)


class RedisError(RepoFetchError):
    """Redis-related errors during repository operations."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.REDIS, details)


class GitHubAPIError(RepoFetchError):
    """GitHub API-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.GITHUB_API, details)

def _create_repository_files_and_index(storage_dir: str, repo_name: str, content: str, tree: str) -> None:
    """
    Create repository files and indexing for efficient access.
    
    Args:
        storage_dir: Directory to store repository files
        repo_name: Name of the repository
        content: Content.md file content
        tree: Tree.txt file content
    """
    try:
        # Create repository directory
        repo_dir = os.path.join(storage_dir, repo_name)
        os.makedirs(repo_dir, exist_ok=True)
        
        # Write content.md file
        content_path = os.path.join(repo_dir, "content.md")
        with open(content_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Write tree.txt file
        tree_path = os.path.join(repo_dir, "tree.txt")
        with open(tree_path, 'w', encoding='utf-8') as f:
            f.write(tree)
        
        # Create indexing file
        indexer = ContentIndexer(storage_dir, repo_name)
        indexer.create_index()
        
        logger.info(f"Created repository files and index for {repo_name} in {repo_dir}")
        
    except Exception as e:
        logger.error(f"Failed to create repository files for {repo_name}: {e}")
        raise


def _log_operation_metrics(operation: str, duration: float, success: bool, details: Optional[Dict[str, Any]] = None) -> None:
    """
    Log operation metrics for monitoring and debugging.
    
    Args:
        operation: Name of the operation
        duration: Operation duration in seconds
        success: Whether the operation succeeded
        details: Additional operation details
    """
    try:
        status = "SUCCESS" if success else "FAILURE"
        log_data = {
            "operation": operation,
            "duration_seconds": round(duration, 3),
            "status": status,
            "timestamp": time.time()
        }
        
        if details:
            log_data.update(details)
        
        if success:
            logger.info(f"METRICS: {operation} completed successfully in {duration:.3f}s", extra=log_data)
        else:
            logger.warning(f"METRICS: {operation} failed after {duration:.3f}s", extra=log_data)
            
    except Exception as e:
        logger.error(f"Error logging metrics for {operation}: {e}")


def _handle_redis_connection_failure(redis_cache: SmartRedisCache, operation: str) -> bool:
    """
    Handle Redis connection failures with graceful recovery.
    
    Args:
        redis_cache: Redis cache instance
        operation: Operation that failed
        
    Returns:
        True if recovery successful, False otherwise
    """
    try:
        logger.warning(f"Redis connection failure detected during {operation}")
        
        # Attempt health check and recovery
        if redis_cache.health_check(force=True):
            logger.info(f"Redis connection recovered for {operation}")
            return True
        else:
            logger.error(f"Redis connection recovery failed for {operation}")
            return False
            
    except Exception as e:
        logger.error(f"Error during Redis connection recovery: {e}")
        return False


def _create_error_response(error: Exception, operation: str, repo_url: str) -> Dict[str, Any]:
    """
    Create standardized error response for monitoring.
    
    Args:
        error: Exception that occurred
        operation: Operation that failed
        repo_url: Repository URL
        
    Returns:
        Standardized error response dictionary
    """
    error_info = {
        "operation": operation,
        "repo_url": repo_url,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": time.time(),
        "traceback": traceback.format_exc()
    }
    
    # Add category if it's a RepoFetchError
    if isinstance(error, RepoFetchError):
        error_info["category"] = error.category
        error_info["details"] = error.details
    
    return error_info


def _validate_environment_config() -> Tuple[bool, str]:
    """
    Validate required environment configuration.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        required_vars = []
        optional_vars = ["GITHUB_TOKEN", "STORAGE_DIR"]  # STORAGE_DIR now optional for Redis mode
        
        # Check Redis configuration
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            return False, "REDIS_URL environment variable is required for Redis mode"
        
        # Validate tier plan
        tier_plan = os.getenv("TIER_PLAN", "free").lower()
        valid_tiers = ["free", "pro", "enterprise"]
        if tier_plan not in valid_tiers:
            return False, f"TIER_PLAN must be one of {valid_tiers}, got '{tier_plan}'"
        
        logger.info("Environment configuration validation passed")
        return True, "Configuration valid"
        
    except Exception as e:
        return False, f"Configuration validation error: {e}"


def _extract_estimated_tokens_from_summary(summary: str) -> int:
    """
    Extract estimated token count from summary text.
    
    Args:
        summary: Summary text containing token information
        
    Returns:
        Estimated token count, defaults to 0 if not found
    """
    try:
        # Look for patterns like "Estimated tokens: 12345" or "tokens: 12345"
        patterns = [
            r'estimated\s+tokens?\s*:?\s*(\d+)',
            r'tokens?\s*:?\s*(\d+)',
            r'token\s+count\s*:?\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, summary, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        logger.warning("Could not extract token count from summary, defaulting to 0")
        return 0
        
    except Exception as e:
        logger.error(f"Error extracting token count from summary: {e}")
        return 0


def _get_repo_name_from_url(repo_url: str) -> str:
    """
    Extract repository name from GitHub URL.
    
    Args:
        repo_url: GitHub repository URL
        
    Returns:
        Repository name
        
    Raises:
        ValueError: If repository name cannot be determined
    """
    try:
        path = urlparse(repo_url).path
        # Handle various URL formats: /owner/repo, /owner/repo.git, /owner/repo/
        path_parts = [part for part in path.strip('/').split('/') if part]
        
        if len(path_parts) < 2:
            raise ValueError("URL must contain owner and repository name")
        
        # Take the repository name (second part) and remove .git suffix
        repo_name = path_parts[1].replace('.git', '')
        
        if not repo_name:
            raise ValueError("Repository name cannot be empty")
        
        # Create full repo identifier with owner for uniqueness
        full_repo_name = f"{path_parts[0]}/{repo_name}"
        return full_repo_name
        
    except Exception as e:
        raise ValueError(f"Could not determine repository name from URL '{repo_url}': {e}")


def fetch_and_store_repo(repo_url: str) -> bool:
    """
    Fetches a GitHub repository using gitingest and stores its data in Redis.

    This function implements comprehensive error handling, monitoring, tier validation,
    Redis connection recovery, and detailed logging for production deployment.

    Args:
        repo_url: The full URL of the GitHub repository to import.

    Returns:
        True if the import was successful, False otherwise.
    """
    start_time = time.time()
    operation = "fetch_and_store_repo"
    
    logger.info(f"Starting Redis-based import for: {repo_url}")
    print(f"Starting import for: {repo_url}")

    # Initialize variables for cleanup and monitoring
    redis_cache = None
    tier_manager = None
    repo_name = None
    
    try:
        # 1. Initialize configuration first
        try:
            initialize_configuration()
            logger.info("Configuration initialized successfully")
        except Exception as e:
            error = RepoFetchError(f"Configuration initialization failed: {e}", ErrorCategory.CONFIGURATION)
            logger.error(f"Configuration initialization failed: {e}")
            print(f"Configuration Error: {e}")
            _log_operation_metrics(operation, time.time() - start_time, False, 
                                 {"error": str(e), "repo_url": repo_url})
            return False

        # 2. Validate environment configuration
        config_valid, config_error = _validate_environment_config()
        if not config_valid:
            error = RepoFetchError(config_error, ErrorCategory.CONFIGURATION)
            logger.error(f"Configuration validation failed: {config_error}")
            print(f"Configuration Error: {config_error}")
            _log_operation_metrics(operation, time.time() - start_time, False, 
                                 {"error": config_error, "repo_url": repo_url})
            return False

        # 3. Initialize Redis cache and tier manager with error handling
        init_start = time.time()
        try:
            redis_cache = SmartRedisCache()
            tier_manager = TierManager()
            
            # Test Redis connection
            if not redis_cache.health_check(force=True):
                raise RedisError("Redis health check failed during initialization")
            
            _log_operation_metrics("redis_initialization", time.time() - init_start, True)
            logger.info("Successfully initialized Redis cache and tier manager")
            
        except Exception as e:
            _log_operation_metrics("redis_initialization", time.time() - init_start, False, 
                                 {"error": str(e)})
            
            if "redis" in str(e).lower() or "connection" in str(e).lower():
                error = RedisError(f"Redis initialization failed: {e}", {"repo_url": repo_url})
            else:
                error = RepoFetchError(f"Service initialization failed: {e}", 
                                     ErrorCategory.CONFIGURATION, {"repo_url": repo_url})
            
            logger.error(f"Initialization failed: {e}")
            print(f"Error: Failed to initialize services: {e}")
            return False

        # 4. Get and validate user tier
        user_tier = os.getenv("TIER_PLAN", "free").lower().strip()
        logger.info(f"User tier: {user_tier}")

        # 5. Get GitHub token (optional)
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            logger.info("GitHub token available for authenticated requests")
        else:
            logger.info("No GitHub token - will use public API limits")

        # 6. Extract and validate repository name from URL
        try:
            repo_name = _get_repo_name_from_url(repo_url)
            logger.info(f"Repository name: {repo_name}")
        except ValueError as e:
            error = RepoFetchError(str(e), ErrorCategory.VALIDATION, {"repo_url": repo_url})
            logger.error(f"Invalid repository URL: {e}")
            print(f"Error: {e}")
            _log_operation_metrics(operation, time.time() - start_time, False, 
                                 {"error": str(e), "repo_url": repo_url})
            return False

        # 7. Check if repository already exists in Redis with error recovery
        existence_check_start = time.time()
        try:
            existence_info = redis_cache.exists_with_metadata(repo_name)
            _log_operation_metrics("redis_existence_check", time.time() - existence_check_start, True)
            
            if existence_info['exists']:
                logger.info(f"Repository {repo_name} already exists in Redis cache")
                print(f"Repository {repo_name} already cached. Skipping fetch.")
                _log_operation_metrics(operation, time.time() - start_time, True, 
                                     {"cached": True, "repo_name": repo_name})
                return True
            elif existence_info['partial']:
                logger.warning(f"Partial data found for {repo_name}, cleaning up and re-fetching")
                redis_cache.smart_invalidate(repo_name)
                
        except Exception as e:
            _log_operation_metrics("redis_existence_check", time.time() - existence_check_start, False, 
                                 {"error": str(e)})
            
            # Try to recover Redis connection
            if not _handle_redis_connection_failure(redis_cache, "existence_check"):
                error = RedisError(f"Redis connection failed and recovery unsuccessful: {e}")
                logger.error(f"Redis connection failure: {e}")
                print(f"Error: Redis connection failed: {e}")
                return False
            
            logger.warning(f"Redis existence check failed but connection recovered: {e}")

        # 8. Fetch repository data with comprehensive retry and error handling
        fetch_start = time.time()
        summary, tree, content = None, None, None
        
        for attempt in range(MAX_RETRIES):
            attempt_start = time.time()
            try:
                logger.info(f"Fetching repository data (Attempt {attempt + 1}/{MAX_RETRIES})")
                print(f"Attempting to fetch... (Attempt {attempt + 1}/{MAX_RETRIES})")
                
                # Prepare ingest arguments
                ingest_kwargs = {}
                if github_token and github_token.strip():
                    logger.info("Using GitHub token for authenticated request")
                    print("Found GitHub token. Using it for the request.")
                    ingest_kwargs['token'] = github_token
                else:
                    logger.info("No GitHub token found, proceeding without authentication")
                    print("No GitHub token found. Proceeding without authentication (for public repos).")

                # Call gitingest with timeout handling
                summary, tree, content = ingest(repo_url, **ingest_kwargs)
                
                _log_operation_metrics("github_fetch", time.time() - attempt_start, True, 
                                     {"attempt": attempt + 1, "repo_name": repo_name})
                logger.info("Successfully fetched repository data from GitHub")
                print("Successfully fetched repository data.")
                break
                
            except Exception as e:
                _log_operation_metrics("github_fetch", time.time() - attempt_start, False, 
                                     {"attempt": attempt + 1, "error": str(e)})
                
                # Categorize the error
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "403" in error_msg:
                    error_category = ErrorCategory.GITHUB_API
                    logger.warning(f"GitHub API rate limit or permission error: {e}")
                elif "network" in error_msg or "connection" in error_msg or "timeout" in error_msg:
                    error_category = ErrorCategory.NETWORK
                    logger.warning(f"Network error during GitHub fetch: {e}")
                else:
                    error_category = ErrorCategory.GITHUB_API
                    logger.warning(f"GitHub fetch error: {e}")
                
                print(f"Warning: Failed to fetch on attempt {attempt + 1}. Error: {e}")
                
                if attempt < MAX_RETRIES - 1:
                    # Exponential backoff for retries
                    retry_delay = RETRY_DELAY_SECONDS * (2 ** attempt)
                    logger.info(f"Retrying in {retry_delay} seconds with exponential backoff...")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    error = GitHubAPIError(f"All {MAX_RETRIES} fetch attempts failed. Last error: {e}", 
                                         {"repo_url": repo_url, "attempts": MAX_RETRIES})
                    logger.error("All retry attempts failed for repository fetch")
                    print("Error: All retry attempts failed. Aborting import for this repo.")
                    _log_operation_metrics(operation, time.time() - start_time, False, 
                                         {"error": "fetch_failed", "attempts": MAX_RETRIES})
                    return False

        _log_operation_metrics("github_fetch_total", time.time() - fetch_start, True, 
                             {"repo_name": repo_name})

        # 9. Validate tier access before storing with detailed error handling
        tier_check_start = time.time()
        try:
            estimated_tokens = _extract_estimated_tokens_from_summary(summary)
            logger.info(f"Estimated tokens for repository: {estimated_tokens}")
            
            is_allowed, access_message = tier_manager.validate_access(user_tier, estimated_tokens)
            
            _log_operation_metrics("tier_validation", time.time() - tier_check_start, is_allowed, 
                                 {"user_tier": user_tier, "estimated_tokens": estimated_tokens})
            
            if not is_allowed:
                error = TierAccessDeniedError(access_message)
                logger.warning(f"Tier access denied: {access_message}")
                print(f"Access denied: {access_message}")
                return False
            
            logger.info(f"Tier access granted: {access_message}")
            print(f"Access granted: {access_message}")
            
        except TierValidationError as e:
            _log_operation_metrics("tier_validation", time.time() - tier_check_start, False, 
                                 {"error": str(e)})
            logger.error(f"Tier validation error: {e}")
            print(f"Error: Tier validation failed: {e}")
            return False
        except Exception as e:
            _log_operation_metrics("tier_validation", time.time() - tier_check_start, False, 
                                 {"error": str(e)})
            logger.error(f"Unexpected tier validation error: {e}")
            print(f"Error: Tier validation failed unexpectedly: {e}")
            return False

        # 10. Store repository data in Redis with comprehensive error handling
        storage_start = time.time()
        try:
            logger.info(f"Storing repository data in Redis for: {repo_name}")
            print(f"Storing repository data in Redis...")
            
            # Prepare data for batch storage with metadata
            repo_data = {
                'content': content,
                'tree': tree,
                'summary': summary,
                'metadata': f"stored_at:{time.time()},estimated_tokens:{estimated_tokens},user_tier:{user_tier},repo_url:{repo_url}"
            }
            
            # Store using optimized batch operation with retry
            success = False
            for storage_attempt in range(2):  # Allow one retry for storage
                try:
                    success = redis_cache.store_repository_batch(repo_name, repo_data)
                    if success:
                        break
                except Exception as storage_error:
                    logger.warning(f"Storage attempt {storage_attempt + 1} failed: {storage_error}")
                    if storage_attempt == 0:
                        # Try to recover connection and retry once
                        if _handle_redis_connection_failure(redis_cache, "storage"):
                            continue
                    raise storage_error
            
            _log_operation_metrics("redis_storage", time.time() - storage_start, success, 
                                 {"repo_name": repo_name, "data_size": len(content) + len(tree) + len(summary)})
            
            if success:
                logger.info(f"Successfully stored repository {repo_name} in Redis")
                print("Import complete. Repository data stored in Redis.")
                
                # Create indexing files for efficient access
                try:
                    storage_dir = os.getenv('STORAGE_DIR', '/tmp/repo_storage')
                    _create_repository_files_and_index(storage_dir, repo_name, content, tree)
                    logger.info(f"Created indexing files for {repo_name}")
                except Exception as index_error:
                    logger.warning(f"Failed to create indexing files for {repo_name}: {index_error}")
                    # Don't fail the entire operation if indexing fails
                
                _log_operation_metrics(operation, time.time() - start_time, True, 
                                     {"repo_name": repo_name, "estimated_tokens": estimated_tokens})
                return True
            else:
                error = RedisError(f"Failed to store repository {repo_name} in Redis")
                logger.error(f"Failed to store repository {repo_name} in Redis")
                print("Error: Failed to store repository data in Redis.")
                return False
                
        except Exception as e:
            _log_operation_metrics("redis_storage", time.time() - storage_start, False, 
                                 {"error": str(e)})
            
            if "redis" in str(e).lower() or "connection" in str(e).lower():
                error = RedisError(f"Redis storage failed: {e}", {"repo_name": repo_name})
            else:
                error = RepoFetchError(f"Storage failed: {e}", ErrorCategory.REDIS, {"repo_name": repo_name})
            
            logger.error(f"Error storing repository data in Redis: {e}")
            print(f"Error: Could not store repository data in Redis: {e}")
            return False

    except Exception as e:
        # Catch-all for any unexpected errors
        error_info = _create_error_response(e, operation, repo_url)
        logger.error(f"Unexpected error during repository fetch and store: {e}", extra=error_info)
        print(f"Error: Unexpected error occurred: {e}")
        
        _log_operation_metrics(operation, time.time() - start_time, False, 
                             {"error": "unexpected", "error_type": type(e).__name__})
        return False
    
    finally:
        # Cleanup and final logging
        total_duration = time.time() - start_time
        logger.info(f"Repository fetch operation completed in {total_duration:.3f}s for {repo_url}")
        
        # Log Redis connection info for monitoring
        if redis_cache:
            try:
                conn_info = redis_cache.get_connection_info()
                logger.debug(f"Redis connection info: {conn_info}")
            except Exception as e:
                logger.warning(f"Could not retrieve Redis connection info: {e}")
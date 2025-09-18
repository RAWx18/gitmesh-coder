"""
Repository factory for creating appropriate repository instances.

This module provides a factory function that creates either GitRepo or RedisRepoManager
instances based on configuration, allowing seamless integration of Redis-backed
repositories while maintaining backward compatibility.
"""

import os
from typing import Optional, List, Any

from cosmos.repo import GitRepo
from cosmos.redis_repo_manager import RedisRepoManager


def create_repo(
    io,
    fnames: List[str],
    git_dname: Optional[str],
    cosmos_ignore_file: Optional[str] = None,
    models: Optional[Any] = None,
    attribute_author: bool = True,
    attribute_committer: bool = True,
    attribute_commit_message_author: bool = False,
    attribute_commit_message_committer: bool = False,
    commit_prompt: Optional[str] = None,
    subtree_only: bool = False,
    git_commit_verify: bool = True,
    attribute_co_authored_by: bool = False,
    repo_url: Optional[str] = None,
    force_redis: bool = False,
    create_pull_request: bool = False,  # New parameter for PR mode
    pr_base_branch: str = 'main',  # Base branch for PRs
    pr_draft: bool = False,  # Whether to create draft PRs
    auto_cleanup: bool = True,  # Whether to automatically cleanup temporary files
    github_token: Optional[str] = None,  # GitHub token for PR operations
):
    """
    Create appropriate repository instance based on configuration.
    
    This factory function determines whether to create a GitRepo or RedisRepoManager
    instance based on environment configuration and parameters.
    
    Args:
        io: Input/output handler
        fnames: List of filenames
        git_dname: Git directory name
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
        repo_url: Repository URL (required for Redis mode)
        force_redis: Force Redis mode even without repo_url
        create_pull_request: Whether to create pull requests instead of direct commits
        pr_base_branch: Base branch for pull requests
        pr_draft: Whether to create draft pull requests
        auto_cleanup: Whether to automatically cleanup temporary files after operations
        github_token: GitHub personal access token for PR operations
        
    Returns:
        GitRepo or RedisRepoManager instance
        
    Raises:
        ValueError: If Redis mode is requested but repo_url is not provided
    """
    # Check if Redis mode should be used
    use_redis = (
        force_redis or 
        repo_url is not None or 
        os.getenv('USE_REDIS_REPO', '').lower() in ('true', '1', 'yes')
    )
    
    if use_redis:
        if not repo_url and not force_redis:
            # Try to extract repo_url from git_dname or fnames if it looks like a URL
            if git_dname and ('github.com' in git_dname or 'gitlab.com' in git_dname):
                repo_url = git_dname
            elif fnames and len(fnames) == 1 and ('github.com' in fnames[0] or 'gitlab.com' in fnames[0]):
                repo_url = fnames[0]
        
        if not repo_url and force_redis:
            raise ValueError("Redis mode requested but no repo_url provided")
            
        if repo_url or force_redis:
            # Enable PR mode by default for Redis repositories since direct commits aren't supported
            redis_create_pr = create_pull_request or True  # Default to True for Redis mode
            
            return RedisRepoManager(
                io=io,
                fnames=fnames,
                git_dname=git_dname,
                repo_url=repo_url or "unknown",  # Allow empty repo_url for cached repos
                cosmos_ignore_file=cosmos_ignore_file,
                models=models,
                attribute_author=attribute_author,
                attribute_committer=attribute_committer,
                attribute_commit_message_author=attribute_commit_message_author,
                attribute_commit_message_committer=attribute_commit_message_committer,
                commit_prompt=commit_prompt,
                subtree_only=subtree_only,
                git_commit_verify=git_commit_verify,
                attribute_co_authored_by=attribute_co_authored_by,
                create_pull_request=redis_create_pr,  # Use the default-enabled value
                pr_base_branch=pr_base_branch,
                pr_draft=pr_draft,
                auto_cleanup=auto_cleanup,
                github_token=github_token
            )
    
    # Default to GitRepo for local repositories
    return GitRepo(
        io=io,
        fnames=fnames,
        git_dname=git_dname,
        cosmos_ignore_file=cosmos_ignore_file,
        models=models,
        attribute_author=attribute_author,
        attribute_committer=attribute_committer,
        attribute_commit_message_author=attribute_commit_message_author,
        attribute_commit_message_committer=attribute_commit_message_committer,
        commit_prompt=commit_prompt,
        subtree_only=subtree_only,
        git_commit_verify=git_commit_verify,
        attribute_co_authored_by=attribute_co_authored_by,
        create_pull_request=create_pull_request,
        pr_base_branch=pr_base_branch,
        pr_draft=pr_draft
    )


def is_redis_mode_available() -> bool:
    """
    Check if Redis mode is available based on configuration.
    
    Returns:
        True if Redis configuration is available, False otherwise
    """
    redis_url = os.getenv('REDIS_URL')
    return redis_url is not None and redis_url.strip() != ''


def get_repo_url_from_context(git_dname: Optional[str], fnames: List[str]) -> Optional[str]:
    """
    Extract repository URL from context if available.
    
    Args:
        git_dname: Git directory name
        fnames: List of filenames
        
    Returns:
        Repository URL if found, None otherwise
    """
    # Check git_dname for URL patterns
    if git_dname and ('github.com' in git_dname or 'gitlab.com' in git_dname):
        return git_dname
        
    # Check fnames for URL patterns
    if fnames:
        for fname in fnames:
            if 'github.com' in fname or 'gitlab.com' in fname:
                return fname
                
    return None
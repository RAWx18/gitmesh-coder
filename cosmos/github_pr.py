"""
GitHub Pull Request Integration Module

This module provides functionality to create pull requests directly from cosmos
when making code changes, allowing for automated GitHub workflow integration.
"""

import os
import json
import logging
import requests
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class GitHubPRError(Exception):
    """Exception raised for GitHub PR operations."""
    pass


class GitHubPRManager:
    """
    Manages GitHub pull request creation and operations.
    """
    
    def __init__(self, repo, github_token=None, io=None):
        """
        Initialize GitHub PR manager.
        
        Args:
            repo: GitRepo instance
            github_token: GitHub personal access token
            io: InputOutput instance for user feedback
        """
        self.repo = repo
        self.io = io
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        
        if not self.github_token:
            raise GitHubPRError("GitHub token not found. Set GITHUB_TOKEN environment variable.")
        
        self._repo_info = None
        self._headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json'
        }
    
    @property
    def repo_info(self) -> Tuple[str, str]:
        """
        Get repository owner and name from git remote.
        
        Returns:
            Tuple of (owner, repo_name)
        """
        if self._repo_info:
            return self._repo_info
            
        try:
            # Get the remote URL
            remote_url = self.repo.repo.remotes.origin.url
            
            # Parse GitHub URL
            if remote_url.startswith('git@github.com:'):
                # SSH format: git@github.com:owner/repo.git
                repo_path = remote_url.split(':')[1].replace('.git', '')
            elif 'github.com' in remote_url:
                # HTTPS format: https://github.com/owner/repo.git
                parsed = urlparse(remote_url)
                repo_path = parsed.path.strip('/').replace('.git', '')
            else:
                raise GitHubPRError(f"Not a GitHub repository: {remote_url}")
            
            owner, repo_name = repo_path.split('/')
            self._repo_info = (owner, repo_name)
            return self._repo_info
            
        except Exception as e:
            raise GitHubPRError(f"Could not determine GitHub repository info: {e}")
    
    def create_branch(self, branch_name: str, base_branch: str = 'main') -> bool:
        """
        Create a new branch for the pull request.
        
        Args:
            branch_name: Name of the new branch
            base_branch: Base branch to create from (default: main)
            
        Returns:
            True if branch was created successfully
        """
        try:
            # Check if branch already exists locally
            existing_branches = [ref.name for ref in self.repo.repo.heads]
            if f"refs/heads/{branch_name}" in existing_branches or branch_name in existing_branches:
                if self.io:
                    self.io.tool_output(f"Branch {branch_name} already exists locally")
                return True
            
            # Create new branch
            self.repo.repo.git.checkout('-b', branch_name)
            if self.io:
                self.io.tool_output(f"Created new branch: {branch_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create branch {branch_name}: {e}")
            return False
    
    def push_branch(self, branch_name: str) -> bool:
        """
        Push the current branch to GitHub.
        
        Args:
            branch_name: Name of the branch to push
            
        Returns:
            True if push was successful
        """
        try:
            # Push the branch to origin
            self.repo.repo.git.push('origin', branch_name, set_upstream=True)
            if self.io:
                self.io.tool_output(f"Pushed branch {branch_name} to GitHub")
            return True
            
        except Exception as e:
            logger.error(f"Failed to push branch {branch_name}: {e}")
            if self.io:
                self.io.tool_error(f"Failed to push branch: {e}")
            return False
    
    def create_pull_request(self, 
                          title: str, 
                          body: str, 
                          head_branch: str, 
                          base_branch: str = 'main',
                          draft: bool = False) -> Optional[Dict[str, Any]]:
        """
        Create a pull request on GitHub.
        
        Args:
            title: PR title
            body: PR description
            head_branch: Source branch
            base_branch: Target branch (default: main)
            draft: Whether to create as draft PR
            
        Returns:
            PR data dict if successful, None otherwise
        """
        owner, repo_name = self.repo_info
        
        pr_data = {
            'title': title,
            'body': body,
            'head': head_branch,
            'base': base_branch,
            'draft': draft
        }
        
        url = f'https://api.github.com/repos/{owner}/{repo_name}/pulls'
        
        try:
            response = requests.post(url, headers=self._headers, json=pr_data)
            
            if response.status_code == 201:
                pr_info = response.json()
                if self.io:
                    self.io.tool_output(f"✅ Pull request created successfully!")
                    self.io.tool_output(f"PR #{pr_info['number']}: {title}")
                    self.io.tool_output(f"URL: {pr_info['html_url']}")
                return pr_info
            else:
                error_msg = f"Failed to create PR. Status: {response.status_code}"
                if response.content:
                    try:
                        error_data = response.json()
                        error_msg += f", Error: {error_data.get('message', 'Unknown error')}"
                    except:
                        error_msg += f", Response: {response.text}"
                
                raise GitHubPRError(error_msg)
                
        except requests.RequestException as e:
            raise GitHubPRError(f"Network error creating PR: {e}")
    
    def generate_pr_title(self, commit_message: str) -> str:
        """
        Generate a PR title from commit message.
        
        Args:
            commit_message: The commit message
            
        Returns:
            Formatted PR title
        """
        # Take the first line and clean it up
        first_line = commit_message.split('\n')[0].strip()
        
        # Remove 'cosmos:' prefix if present
        if first_line.startswith('cosmos: '):
            first_line = first_line[8:]
        
        # Capitalize first letter if not already
        if first_line and first_line[0].islower():
            first_line = first_line[0].upper() + first_line[1:]
        
        return first_line or "Code changes via cosmos"
    
    def generate_pr_body(self, 
                        commit_message: str, 
                        changed_files: List[str], 
                        diffs: str = None) -> str:
        """
        Generate a PR body with details about the changes.
        
        Args:
            commit_message: The commit message
            changed_files: List of changed file paths
            diffs: Optional diff content
            
        Returns:
            Formatted PR body
        """
        body_parts = []
        
        # Add commit message details
        lines = commit_message.split('\n')
        if len(lines) > 1:
            # Add additional commit message content if available
            additional_content = '\n'.join(lines[1:]).strip()
            if additional_content:
                body_parts.append("## Description")
                body_parts.append(additional_content)
        
        # Add changed files
        if changed_files:
            body_parts.append("## Changed Files")
            for file_path in sorted(changed_files):
                body_parts.append(f"- `{file_path}`")
        
        # Add cosmos attribution
        body_parts.append("---")
        body_parts.append("*This pull request was created automatically by [cosmos](https://github.com/paul-gauthier/aider) AI pair programming.*")
        
        return '\n\n'.join(body_parts)
    
    def generate_branch_name(self, commit_message: str) -> str:
        """
        Generate a branch name from commit message.
        
        Args:
            commit_message: The commit message
            
        Returns:
            Branch name suitable for GitHub
        """
        # Take first line of commit message
        first_line = commit_message.split('\n')[0].strip()
        
        # Remove 'cosmos:' prefix if present
        if first_line.startswith('cosmos: '):
            first_line = first_line[8:]
        
        # Clean up for branch name
        import re
        branch_name = re.sub(r'[^a-zA-Z0-9\s\-_]', '', first_line)
        branch_name = re.sub(r'\s+', '-', branch_name)
        branch_name = branch_name.strip('-').lower()
        
        # Limit length and add timestamp for uniqueness
        if len(branch_name) > 40:
            branch_name = branch_name[:40].rstrip('-')
        
        timestamp = datetime.now().strftime('%m%d-%H%M')
        return f"cosmos/{branch_name}-{timestamp}"


def create_pull_request_workflow(repo, 
                                commit_message: str, 
                                changed_files: List[str],
                                io=None, 
                                base_branch: str = 'main',
                                draft: bool = False,
                                github_token: str = None) -> Optional[Dict[str, Any]]:
    """
    Complete workflow to create a pull request from changes.
    
    Args:
        repo: GitRepo or RedisRepoManager instance
        commit_message: Commit message for the changes
        changed_files: List of files that were changed
        io: InputOutput instance for user feedback
        base_branch: Base branch for PR (default: main)
        draft: Whether to create as draft PR
        github_token: GitHub API token (for Redis repos)
        
    Returns:
        PR info dict if successful, None otherwise
    """
    try:
        # Check if this is a Redis repository
        is_redis_repo = hasattr(repo, 'redis_cache') and hasattr(repo, 'repo_name')
        
        if is_redis_repo:
            # Handle Redis repository PR creation
            return create_redis_pull_request(repo, commit_message, changed_files, io, base_branch, draft, github_token)
        else:
            # Handle standard GitRepo PR creation
            pr_manager = GitHubPRManager(repo, io=io)
            
            # Generate branch name and PR details
            branch_name = pr_manager.generate_branch_name(commit_message)
            pr_title = pr_manager.generate_pr_title(commit_message)
            pr_body = pr_manager.generate_pr_body(commit_message, changed_files)
            
            if io:
                io.tool_output(f"Creating pull request workflow...")
                io.tool_output(f"Branch: {branch_name}")
                io.tool_output(f"Title: {pr_title}")
            
            # Create and switch to new branch
            if not pr_manager.create_branch(branch_name, base_branch):
                return None
            
            # Note: The actual file changes and commit should have already been made
            # by the calling code before this function is called
            
            # Push the branch to GitHub
            if not pr_manager.push_branch(branch_name):
                return None
            
            # Create the pull request
            pr_info = pr_manager.create_pull_request(
                title=pr_title,
                body=pr_body,
                head_branch=branch_name,
                base_branch=base_branch,
                draft=draft
            )
            
            return pr_info
        
    except GitHubPRError as e:
        if io:
            io.tool_error(f"GitHub PR Error: {e}")
        logger.error(f"GitHub PR error: {e}")
        return None
    except Exception as e:
        if io:
            io.tool_error(f"Unexpected error creating PR: {e}")
        logger.error(f"Unexpected error in PR workflow: {e}")
        return None


def create_redis_pull_request(repo, commit_message: str, changed_files: List[str], 
                              io=None, base_branch: str = 'main', draft: bool = False,
                              github_token: str = None) -> Optional[Dict[str, Any]]:
    """
    Create a real pull request for Redis-based repositories using GitHub API directly.
    
    Args:
        repo: RedisRepoManager instance
        commit_message: Commit message for the changes
        changed_files: List of files that were changed
        io: InputOutput instance for user feedback
        base_branch: Base branch for PR (default: main)
        draft: Whether to create as draft PR
        github_token: GitHub API token
        
    Returns:
        PR info dict if successful, None otherwise
    """
    try:
        import requests
        import json
        import time
        import base64
        
        # Get GitHub repository info from repo_url or repo_name
        repo_url = getattr(repo, 'repo_url', '')
        repo_name = getattr(repo, 'repo_name', '')
        
        # Extract owner and repo from URL or name
        if 'github.com' in repo_url:
            # Extract from URL like "https://github.com/owner/repo"
            parts = repo_url.rstrip('/').split('/')
            if len(parts) >= 2:
                owner = parts[-2]
                repo_name_clean = parts[-1].replace('.git', '')
            else:
                raise GitHubPRError(f"Invalid GitHub URL format: {repo_url}")
        elif '/' in repo_name:
            # Format like "owner/repo"
            owner, repo_name_clean = repo_name.split('/', 1)
        else:
            raise GitHubPRError(f"Cannot determine repository owner and name from: {repo_url or repo_name}")
        
        # Get GitHub token
        token = github_token or getattr(repo, 'github_token', None) or os.getenv('GITHUB_TOKEN')
        if not token:
            raise GitHubPRError("GitHub token required for PR creation")
        
        headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json'
        }
        
        # Generate branch name and PR details
        timestamp = int(time.time())
        branch_name = f"cosmos-changes-{timestamp}"
        pr_title = commit_message.split('\n')[0][:72]  # First line, max 72 chars
        
        # Create PR body
        pr_body = f"{commit_message}\n\n"
        if changed_files:
            pr_body += "## Changed files:\n"
            for file in changed_files[:10]:  # Limit to first 10 files
                pr_body += f"- {file}\n"
        pr_body += "\n*This PR was created automatically by cosmos.*"
        
        if io:
            io.tool_output(f"Creating pull request for {owner}/{repo_name_clean}...")
            io.tool_output(f"Branch: {branch_name}")
            io.tool_output(f"Title: {pr_title}")
        
        # Step 1: Get the reference to the base branch
        ref_url = f"https://api.github.com/repos/{owner}/{repo_name_clean}/git/refs/heads/{base_branch}"
        ref_response = requests.get(ref_url, headers=headers)
        
        if ref_response.status_code != 200:
            raise GitHubPRError(f"Failed to get base branch reference: {ref_response.status_code} - {ref_response.text}")
        
        base_sha = ref_response.json()['object']['sha']
        
        # Step 2: Create a new branch reference
        create_ref_url = f"https://api.github.com/repos/{owner}/{repo_name_clean}/git/refs"
        create_ref_data = {
            "ref": f"refs/heads/{branch_name}",
            "sha": base_sha
        }
        
        ref_create_response = requests.post(create_ref_url, headers=headers, json=create_ref_data)
        
        if ref_create_response.status_code != 201:
            raise GitHubPRError(f"Failed to create branch: {ref_create_response.status_code} - {ref_create_response.text}")
        
        if io:
            io.tool_output(f"✅ Created branch: {branch_name}")
        
        # Step 3: Get current file contents and create commits for changed files
        commits_created = []
        
        for file_path in changed_files:
            try:
                # Get the file content using the repo's method
                if hasattr(repo, 'get_file_content_for_pr'):
                    file_content = repo.get_file_content_for_pr(file_path)
                elif hasattr(repo, 'virtual_fs') and repo.virtual_fs:
                    file_content = repo.virtual_fs.extract_file_with_context(file_path)
                elif hasattr(repo, 'content_indexer') and repo.content_indexer:
                    file_content = repo.content_indexer.get_file_content(file_path)
                else:
                    if io:
                        io.tool_warning(f"Could not get content for {file_path}")
                    continue
                
                if not file_content:
                    if io:
                        io.tool_warning(f"Empty content for {file_path}")
                    continue
                
                # Encode content to base64
                content_encoded = base64.b64encode(file_content.encode('utf-8')).decode('utf-8')
                
                # Check if file exists in the repository
                file_url = f"https://api.github.com/repos/{owner}/{repo_name_clean}/contents/{file_path}"
                file_response = requests.get(file_url, headers=headers, params={'ref': base_branch})
                
                # Create or update the file
                update_data = {
                    "message": f"Update {file_path}",
                    "content": content_encoded,
                    "branch": branch_name
                }
                
                if file_response.status_code == 200:
                    # File exists, update it
                    existing_file = file_response.json()
                    update_data["sha"] = existing_file["sha"]
                # If file doesn't exist (404), we'll create it (no SHA needed)
                
                update_response = requests.put(file_url, headers=headers, json=update_data)
                
                if update_response.status_code in [200, 201]:
                    commits_created.append(file_path)
                    if io:
                        io.tool_output(f"✅ Updated {file_path}")
                else:
                    if io:
                        io.tool_warning(f"Failed to update {file_path}: {update_response.status_code}")
                    
            except Exception as e:
                if io:
                    io.tool_warning(f"Error processing {file_path}: {e}")
                continue
        
        if not commits_created:
            # Clean up the branch if no commits were made
            delete_ref_url = f"https://api.github.com/repos/{owner}/{repo_name_clean}/git/refs/heads/{branch_name}"
            requests.delete(delete_ref_url, headers=headers)
            raise GitHubPRError("No files could be updated in the repository")
        
        # Step 4: Create the pull request
        pr_url = f"https://api.github.com/repos/{owner}/{repo_name_clean}/pulls"
        pr_data = {
            "title": pr_title,
            "body": pr_body,
            "head": branch_name,
            "base": base_branch,
            "draft": draft
        }
        
        pr_response = requests.post(pr_url, headers=headers, json=pr_data)
        
        if pr_response.status_code == 201:
            pr_info = pr_response.json()
            if io:
                io.tool_output(f"✅ Created pull request #{pr_info['number']}")
                io.tool_output(f"URL: {pr_info['html_url']}")
                io.tool_output(f"Files updated: {', '.join(commits_created)}")
            
            return {
                'number': pr_info['number'],
                'html_url': pr_info['html_url'],
                'title': pr_info['title'],
                'body': pr_info['body'],
                'state': pr_info['state'],
                'head': pr_info['head'],
                'base': pr_info['base'],
                'commits_created': commits_created
            }
        else:
            # Clean up the branch if PR creation failed
            delete_ref_url = f"https://api.github.com/repos/{owner}/{repo_name_clean}/git/refs/heads/{branch_name}"
            requests.delete(delete_ref_url, headers=headers)
            
            error_msg = f"Failed to create pull request: {pr_response.status_code} - {pr_response.text}"
            if io:
                io.tool_error(error_msg)
            raise GitHubPRError(error_msg)
            
    except requests.exceptions.RequestException as e:
        error_msg = f"GitHub API request failed: {e}"
        if io:
            io.tool_error(error_msg)
        logger.error(error_msg)
        return None
    except Exception as e:
        error_msg = f"Error creating Redis pull request: {e}"
        if io:
            io.tool_error(error_msg)
        logger.error(error_msg)
        return None

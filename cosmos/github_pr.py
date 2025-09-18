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
        elif '_' in repo_name:
            # Handle format like "owner_repo" (convert to "owner/repo")
            owner, repo_name_clean = repo_name.split('_', 1)
            if io:
                io.tool_output(f"Converted repository name from {repo_name} to {owner}/{repo_name_clean}")
        else:
            raise GitHubPRError(f"Cannot determine repository owner and name from: {repo_url or repo_name}")
        
        # Debug logging
        if io:
            io.tool_output(f"Extracted repository: {owner}/{repo_name_clean}")
            io.tool_output(f"From repo_url: {repo_url}, repo_name: {repo_name}")
        
        # Get GitHub token
        token = github_token or getattr(repo, 'github_token', None) or os.getenv('GITHUB_TOKEN')
        if not token:
            raise GitHubPRError("GitHub token required for PR creation")
        
        headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json'
        }
        
        # Verify token permissions
        user_url = "https://api.github.com/user"
        user_response = requests.get(user_url, headers=headers)
        
        if user_response.status_code == 401:
            raise GitHubPRError("Invalid GitHub token. Please check your GITHUB_TOKEN environment variable.")
        elif user_response.status_code != 200:
            raise GitHubPRError(f"Failed to verify GitHub token: {user_response.status_code}")
        
        user_info = user_response.json()
        if io:
            io.tool_output(f"GitHub token verified for user: {user_info.get('login', 'unknown')}")
        
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
        
        # Step 1: Verify repository access
        repo_check_url = f"https://api.github.com/repos/{owner}/{repo_name_clean}"
        repo_check_response = requests.get(repo_check_url, headers=headers)
        
        if repo_check_response.status_code == 404:
            raise GitHubPRError(f"Repository {owner}/{repo_name_clean} not found or not accessible with provided token")
        elif repo_check_response.status_code != 200:
            raise GitHubPRError(f"Failed to access repository: {repo_check_response.status_code} - {repo_check_response.text}")
        
        repo_info = repo_check_response.json()
        if io:
            io.tool_output(f"Repository access confirmed: {repo_info.get('full_name')}")
            io.tool_output(f"Default branch: {repo_info.get('default_branch', 'unknown')}")
        
        # Get current user info to check ownership
        user_response = requests.get("https://api.github.com/user", headers=headers)
        if user_response.status_code != 200:
            raise GitHubPRError("Failed to get user information")
        
        current_user = user_response.json()['login']
        
        # Check if this is the user's own repository
        is_own_repo = (owner.lower() == current_user.lower())
        
        if not is_own_repo:
            raise GitHubPRError(f"❌ Pull requests are only allowed for your own repositories.\n"
                              f"Repository {owner}/{repo_name_clean} belongs to {owner}, but you are {current_user}.\n"
                              f"Please work with your own repositories or use the buffer to stage changes locally.")
        
        # Check if user has write access to their own repo
        permissions = repo_info.get('permissions', {})
        has_write_access = permissions.get('push', False)
        
        if not has_write_access:
            raise GitHubPRError(f"❌ No write access to your repository {owner}/{repo_name_clean}.\n"
                              f"Please check your repository permissions.")
        
        if io:
            io.tool_output(f"✅ Confirmed ownership and write access to {owner}/{repo_name_clean}")
        
        # Use the original repository (no forking needed for own repos)
        fork_owner = owner
        fork_repo = repo_name_clean
        
        # Use the repository's default branch if base_branch is 'main' but doesn't exist
        actual_base_branch = base_branch
        if base_branch == 'main' and repo_info.get('default_branch') != 'main':
            actual_base_branch = repo_info.get('default_branch', 'main')
            if io:
                io.tool_output(f"Using repository default branch: {actual_base_branch}")
        
        # Step 2: Get the reference to the base branch (from original repo)
        ref_url = f"https://api.github.com/repos/{owner}/{repo_name_clean}/git/refs/heads/{actual_base_branch}"
        ref_response = requests.get(ref_url, headers=headers)
        
        if ref_response.status_code != 200:
            # Try to list available branches
            branches_url = f"https://api.github.com/repos/{owner}/{repo_name_clean}/branches"
            branches_response = requests.get(branches_url, headers=headers)
            available_branches = []
            if branches_response.status_code == 200:
                available_branches = [b['name'] for b in branches_response.json()]
            
            error_msg = f"Failed to get base branch '{actual_base_branch}' reference: {ref_response.status_code}"
            if available_branches:
                error_msg += f"\nAvailable branches: {', '.join(available_branches[:5])}"
            raise GitHubPRError(error_msg)
        
        base_sha = ref_response.json()['object']['sha']
        
        # Step 3: Create a new branch reference (in fork if we're using one)
        create_ref_url = f"https://api.github.com/repos/{fork_owner}/{fork_repo}/git/refs"
        create_ref_data = {
            "ref": f"refs/heads/{branch_name}",
            "sha": base_sha
        }
        
        ref_create_response = requests.post(create_ref_url, headers=headers, json=create_ref_data)
        
        if ref_create_response.status_code != 201:
            error_detail = ref_create_response.text
            try:
                error_json = ref_create_response.json()
                if 'message' in error_json:
                    error_detail = error_json['message']
            except:
                pass
            raise GitHubPRError(f"Failed to create branch in {fork_owner}/{fork_repo}: {ref_create_response.status_code} - {error_detail}")
        
        if io:
            io.tool_output(f"✅ Created branch: {branch_name} in {fork_owner}/{fork_repo}")
        
        if io:
            io.tool_output(f"✅ Created branch: {branch_name}")
        
        # Step 3: Get current file contents and create commits for changed files
        commits_created = []
        
        # Limit the number of files to avoid API rate limits
        max_files = 10
        files_to_process = changed_files[:max_files]
        
        if len(changed_files) > max_files:
            if io:
                io.tool_warning(f"Limiting to first {max_files} files to avoid API limits")
        
        for file_path in files_to_process:
            try:
                # Get the file content using the repo's method
                file_content = None
                
                if hasattr(repo, 'get_file_content_for_pr'):
                    file_content = repo.get_file_content_for_pr(file_path)
                elif hasattr(repo, 'virtual_fs') and repo.virtual_fs:
                    file_content = repo.virtual_fs.extract_file_with_context(file_path)
                elif hasattr(repo, 'content_indexer') and repo.content_indexer:
                    file_content = repo.content_indexer.get_file_content(file_path)
                
                if not file_content:
                    if io:
                        io.tool_warning(f"Could not get content for {file_path}, skipping")
                    continue
                
                # Validate content is not too large (GitHub has a 100MB limit)
                if len(file_content.encode('utf-8')) > 50 * 1024 * 1024:  # 50MB limit for safety
                    if io:
                        io.tool_warning(f"File {file_path} too large, skipping")
                    continue
                
                # Encode content to base64
                try:
                    content_encoded = base64.b64encode(file_content.encode('utf-8')).decode('utf-8')
                except UnicodeEncodeError:
                    if io:
                        io.tool_warning(f"Could not encode {file_path}, skipping")
                    continue
                
                # Check if file exists in the fork (use actual_base_branch)
                file_url = f"https://api.github.com/repos/{fork_owner}/{fork_repo}/contents/{file_path}"
                file_response = requests.get(file_url, headers=headers, params={'ref': actual_base_branch})
                
                # Create or update the file
                update_data = {
                    "message": f"Update {file_path} via cosmos",
                    "content": content_encoded,
                    "branch": branch_name
                }
                
                if file_response.status_code == 200:
                    # File exists, update it
                    existing_file = file_response.json()
                    update_data["sha"] = existing_file["sha"]
                elif file_response.status_code == 404:
                    # File doesn't exist, create it (no SHA needed)
                    pass
                else:
                    if io:
                        io.tool_warning(f"Could not check {file_path} status: {file_response.status_code}")
                    continue
                
                # Make the update/create request (to the fork)
                update_response = requests.put(file_url, headers=headers, json=update_data)
                
                if update_response.status_code in [200, 201]:
                    commits_created.append(file_path)
                    if io:
                        io.tool_output(f"✅ Updated {file_path}")
                else:
                    error_detail = update_response.text
                    try:
                        error_json = update_response.json()
                        if 'message' in error_json:
                            error_detail = error_json['message']
                    except:
                        pass
                    
                    if io:
                        io.tool_warning(f"Failed to update {file_path}: {update_response.status_code} - {error_detail}")
                    
            except Exception as e:
                if io:
                    io.tool_warning(f"Error processing {file_path}: {e}")
                continue
        
        if not commits_created:
            # Clean up the branch if no commits were made
            delete_ref_url = f"https://api.github.com/repos/{fork_owner}/{fork_repo}/git/refs/heads/{branch_name}"
            requests.delete(delete_ref_url, headers=headers)
            raise GitHubPRError("No files could be updated in the repository")
        
        # Step 4: Create the pull request (from fork to original repo)
        pr_url = f"https://api.github.com/repos/{owner}/{repo_name_clean}/pulls"
        
        # If we're using a fork, the head should be "fork_owner:branch_name"
        head_ref = branch_name if fork_owner == owner else f"{fork_owner}:{branch_name}"
        
        pr_data = {
            "title": pr_title,
            "body": pr_body,
            "head": head_ref,
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
            delete_ref_url = f"https://api.github.com/repos/{fork_owner}/{fork_repo}/git/refs/heads/{branch_name}"
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

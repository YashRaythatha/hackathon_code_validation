#!/usr/bin/env python3
"""
GitHub Repository Analyzer

This handles all the GitHub API stuff - fetching repos, files, etc.
Nothing fancy, just gets the data we need.
"""

import requests
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
import base64


class GitHubAnalyzer:
    """Takes care of GitHub API calls"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token
        self.session = requests.Session()
        # Add auth header if we have a token
        if token:
            self.session.headers.update({'Authorization': f'token {token}'})
    
    def get_repo_info(self, repo_url: str, branch: str = 'main') -> Dict[str, Any]:
        """Extract repository information from URL"""
        # Parse GitHub URL
        if 'github.com' not in repo_url:
            raise ValueError("Only GitHub repositories are supported")
        
        # Extract owner/repo from URL
        parsed = urlparse(repo_url)
        path_parts = parsed.path.strip('/').split('/')
        if len(path_parts) < 2:
            raise ValueError("Invalid GitHub URL format")
        
        owner, repo = path_parts[0], path_parts[1]
        if repo.endswith('.git'):
            repo = repo[:-4]
        
        return {
            'owner': owner,
            'repo': repo,
            'branch': branch,
            'api_url': f'https://api.github.com/repos/{owner}/{repo}'
        }
    
    def get_file_tree(self, repo_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get repository file tree"""
        try:
            url = f"{repo_info['api_url']}/git/trees/{repo_info['branch']}?recursive=1"
            response = self.session.get(url)
            response.raise_for_status()
            return response.json().get('tree', [])
        except Exception as e:
            print(f"Warning: Could not fetch file tree: {e}")
            return []
    
    def get_readme(self, repo_info: Dict[str, Any]) -> str:
        """Get README content"""
        try:
            url = f"{repo_info['api_url']}/readme"
            response = self.session.get(url)
            response.raise_for_status()
            content = response.json().get('content', '')
            return base64.b64decode(content).decode('utf-8')
        except Exception as e:
            print(f"Warning: Could not fetch README: {e}")
            return ""
    
    def get_file_content(self, repo_info: Dict[str, Any], file_path: str) -> str:
        """Get content of a specific file"""
        try:
            url = f"{repo_info['api_url']}/contents/{file_path}"
            response = self.session.get(url)
            response.raise_for_status()
            content = response.json().get('content', '')
            return base64.b64decode(content).decode('utf-8')
        except Exception as e:
            print(f"Warning: Could not fetch {file_path}: {e}")
            return ""


"""
Core utilities and configuration for the hackathon grader.
This module contains all the essential components needed for the application.
"""

import os
import sys
import time
import json
import logging
import subprocess
import shlex
import signal
import threading
import psutil
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Union
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration classes for different parts of the application

@dataclass
class GitHubConfig:
    token: Optional[str] = os.getenv("GITHUB_TOKEN")
    base_url: str = "https://api.github.com"

@dataclass
class UIConfig:
    headless: bool = os.getenv("UI_HEADLESS", "True").lower() == "true"
    timeout: int = int(os.getenv("UI_TIMEOUT", "30"))

@dataclass
class CacheConfig:
    max_size: int = int(os.getenv("CACHE_MAX_SIZE", "100"))

@dataclass
class SecurityConfig:
    max_timeout: int = int(os.getenv("SECURITY_MAX_TIMEOUT", "120"))

@dataclass
class WebConfig:
    secret_key: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    max_content_length: int = int(os.getenv("MAX_CONTENT_LENGTH", "16777216"))  # 16MB

@dataclass
class AppConfig:
    debug: bool = os.getenv("FLASK_ENV", "development") == "development"
    port: int = int(os.getenv("PORT", "5000"))
    github: GitHubConfig = field(default_factory=GitHubConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    web: WebConfig = field(default_factory=WebConfig)

config = AppConfig()

# Custom exceptions for better error handling

class AnalysisError(Exception):
    """Base exception for analysis-related errors."""
    pass

class GitHubAPIError(AnalysisError):
    """Raised when there's an issue with GitHub API calls."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code

class UIExecutionError(AnalysisError):
    """Raised when UI execution fails."""
    pass

class UIExecutionTimeoutError(UIExecutionError):
    """Raised when UI execution times out."""
    pass

class UIExecutionFailedError(UIExecutionError):
    """Raised when UI execution fails for other reasons."""
    pass

class CommandExecutionError(AnalysisError):
    """Raised when an external command fails to execute."""
    pass

class CommandTimeoutError(CommandExecutionError):
    """Raised when an external command times out."""
    pass

class ValidationError(AnalysisError):
    """Raised when input validation fails."""
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message)
        self.field = field

class RepositoryValidationError(ValidationError):
    """Raised when repository validation fails."""
    def __init__(self, repo_url: str):
        super().__init__(f"Invalid repository URL: {repo_url}", "repo_url")
        self.repo_url = repo_url

# Logging setup for the application

def get_logger(name: str) -> logging.Logger:
    """Get a logger with consistent configuration"""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # File handler
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Security handler
    security_handler = logging.FileHandler(
        os.path.join(log_dir, f"security_{datetime.now().strftime('%Y%m%d')}.log")
    )
    security_handler.setLevel(logging.WARNING)
    security_handler.setFormatter(file_formatter)
    
    # Configure logger
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(security_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger

# Cache implementation for storing analysis results

class AnalysisCache:
    """LRU cache for analysis results with memory monitoring"""
    
    def __init__(self, max_size: Optional[int] = None):
        self.max_size = max_size if max_size is not None else config.cache.max_size
        self.cache = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
        self.logger = get_logger(__name__)
        self.logger.info(f"AnalysisCache initialized with max_size={self.max_size}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key in self.cache:
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.hit_count += 1
            self.logger.debug(f"Cache hit for key: {key}")
            return value
        
        self.miss_count += 1
        self.logger.debug(f"Cache miss for key: {key}")
        return None
    
    def set(self, key: str, value: Any):
        """Set item in cache"""
        if key in self.cache:
            # Update existing item
            self.cache.pop(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used item
            oldest_key = next(iter(self.cache))
            self.cache.pop(oldest_key)
            self.logger.debug(f"Evicted oldest cache entry: {oldest_key}")
        
        self.cache[key] = value
        self.logger.debug(f"Cached item with key: {key}")
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        self.logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'memory_usage': self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss': memory_info.rss,
                'vms': memory_info.vms,
                'percent': process.memory_percent()
            }
        except Exception as e:
            self.logger.warning(f"Could not get memory usage: {e}")
            return {'error': str(e)}
    
    def get_analysis(self, repo_url: str, branch: str, selected_agents: List[str]) -> Optional[Any]:
        """Get cached analysis result for specific parameters"""
        cache_key = f"{repo_url}:{branch}:{':'.join(sorted(selected_agents))}"
        return self.get(cache_key)
    
    def set_analysis(self, repo_url: str, branch: str, selected_agents: List[str], result: Any):
        """Set cached analysis result for specific parameters"""
        cache_key = f"{repo_url}:{branch}:{':'.join(sorted(selected_agents))}"
        self.set(cache_key, result)

# Secure subprocess execution with safety checks

class SecureExecutor:
    """Secure subprocess execution with timeout and cleanup"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.allowed_commands = {
            'npm', 'node', 'npx', 'python', 'pip', 'git', 'ls', 'cat', 'echo',
            'cd', 'pwd', 'mkdir', 'rm', 'cp', 'mv', 'chmod', 'chown'
        }
    
    def execute_safe(self, command: List[str], timeout: int = 60, cwd: Optional[str] = None) -> Dict[str, Any]:
        """Execute command safely with timeout"""
        if not command:
            raise CommandExecutionError("Empty command provided")
        
        # Validate command
        if command[0] not in self.allowed_commands:
            raise CommandExecutionError(f"Command '{command[0]}' not in allowed list")
        
        # Sanitize arguments
        sanitized_args = [shlex.quote(arg) for arg in command[1:]]
        full_command = [command[0]] + sanitized_args
        
        self.logger.info(f"Executing safe command: {' '.join(full_command)}")
        
        try:
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                cwd=cwd,
                shell=True  # Use shell to access system PATH
            )
            
            return {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'command': full_command
            }
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Command timed out after {timeout}s: {' '.join(full_command)}")
            raise CommandTimeoutError(f"Command timed out after {timeout}s")
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            raise CommandExecutionError(f"Command execution failed: {e}")
    
    def execute_with_cleanup(self, command: List[str], timeout: int = 60, cwd: Optional[str] = None) -> Dict[str, Any]:
        """Execute command with process cleanup"""
        if not command:
            raise CommandExecutionError("Empty command provided")
        
        # Validate command
        if command[0] not in self.allowed_commands:
            raise CommandExecutionError(f"Command '{command[0]}' not in allowed list")
        
        # Sanitize arguments
        sanitized_args = [shlex.quote(arg) for arg in command[1:]]
        full_command = [command[0]] + sanitized_args
        
        self.logger.info(f"Executing command with cleanup: {' '.join(full_command)}")
        
        process = None
        try:
            process = subprocess.Popen(
                full_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,  # Use shell to access system PATH
                cwd=cwd
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                return {
                    'success': process.returncode == 0,
                    'returncode': process.returncode,
                    'stdout': stdout,
                    'stderr': stderr,
                    'command': full_command
                }
            except subprocess.TimeoutExpired:
                self.logger.warning(f"Command timed out, terminating process: {' '.join(full_command)}")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.logger.warning("Process didn't terminate gracefully, killing it")
                    process.kill()
                    process.wait()
                raise CommandTimeoutError(f"Command timed out after {timeout}s")
                
        except Exception as e:
            if process and process.poll() is None:
                self.logger.warning(f"Cleaning up running process: {e}")
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except:
                    process.kill()
            raise CommandExecutionError(f"Command execution failed: {e}")

# Global instances for use throughout the application
analysis_cache = AnalysisCache()
secure_executor = SecureExecutor()

# Utility functions

def log_analysis_progress(repo_url: str, message: str, progress: int = 0):
    """Log analysis progress"""
    logger = get_logger(__name__)
    logger.info(f"Analysis Progress [{progress}%] - {repo_url}: {message}")

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    return analysis_cache.get_stats()

def clear_cache():
    """Clear analysis cache"""
    analysis_cache.clear()

def get_config() -> AppConfig:
    """Get application configuration"""
    return config

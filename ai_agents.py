#!/usr/bin/env python3
"""
AI Agents for Hackathon Grader

This module contains the AI agents that analyze hackathon submissions.
Each agent focuses on a specific aspect like code quality, architecture, etc.
"""

import json
import re
import os
import pickle
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import requests
from pathlib import Path
from collections import defaultdict, Counter
import math

# Some constants we'll use throughout
DEFAULT_CONFIDENCE = 0.5
MAX_PATTERNS = 1000
MIN_SAMPLES_FOR_LEARNING = 3


@dataclass
class AgentAnalysis:
    """Container for agent analysis results - keeps things organized"""
    agent_name: str
    score: int
    confidence: float
    evidence: List[str]
    recommendations: List[str]
    insights: List[str]
    risks: List[str]


class BaseAIAgent(ABC):
    """Base class for all our AI agents - keeps the interface consistent"""
    
    def __init__(self, name: str, api_key: Optional[str] = None):
        self.name = name
        self.api_key = api_key
        self.analysis_history = []  # Keep track of what we've analyzed
    
    @abstractmethod
    def analyze(self, context: Dict[str, Any]) -> AgentAnalysis:
        """Each agent needs to implement this - the main analysis method"""
        pass
    
    def get_confidence_score(self, evidence_count: int, quality_indicators: int) -> float:
        """Figure out how confident we are in our analysis"""
        if evidence_count == 0:
            return 0.0  # No evidence = no confidence
        
        # More evidence = more confidence (up to a point)
        base_confidence = min(evidence_count / 10.0, 1.0)
        
        # Quality indicators give us a little boost
        quality_boost = min(quality_indicators / 5.0, 0.3)
        
        return min(base_confidence + quality_boost, 1.0)
    
    def extract_code_patterns(self, code_content: str) -> Dict[str, List[str]]:
        """Extract common code patterns and anti-patterns"""
        patterns = {
            'good_patterns': [],
            'bad_patterns': [],
            'security_concerns': [],
            'performance_issues': []
        }
        
        # Good patterns
        if re.search(r'async\s+def|await\s+', code_content):
            patterns['good_patterns'].append('Async/await usage')
        
        if re.search(r'try:\s*.*except\s+', code_content, re.DOTALL):
            patterns['good_patterns'].append('Error handling')
        
        if re.search(r'def\s+\w+\([^)]*\):\s*""".*"""', code_content, re.DOTALL):
            patterns['good_patterns'].append('Function documentation')
        
        # Bad patterns
        if re.search(r'print\s*\(', code_content):
            patterns['bad_patterns'].append('Debug print statements')
        
        if re.search(r'password\s*=\s*["\'][^"\']+["\']', code_content, re.IGNORECASE):
            patterns['security_concerns'].append('Hardcoded passwords')
        
        if re.search(r'eval\s*\(', code_content):
            patterns['security_concerns'].append('Use of eval() function')
        
        if re.search(r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(', code_content):
            patterns['performance_issues'].append('Inefficient loop patterns')
        
        return patterns


class CodeAnalysisAgent(BaseAIAgent):
    """AI agent for deep code analysis"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("CodeAnalysisAgent", api_key)
    
    def analyze(self, context: Dict[str, Any]) -> AgentAnalysis:
        """Analyze code quality, patterns, and best practices"""
        file_tree = context.get('file_tree', [])
        readme = context.get('readme', '')
        artifacts = context.get('artifacts', {})
        
        evidence = []
        recommendations = []
        insights = []
        risks = []
        score = 0
        
        # Analyze code files
        code_files = self._get_code_files(file_tree)
        evidence.append(f"Found {len(code_files)} code files")
        
        if code_files:
            score += 2
        
        # Analyze code patterns (simulated - in real implementation, would analyze actual code)
        patterns = self._analyze_code_patterns(code_files, artifacts)
        
        # Good patterns boost score
        if patterns['good_patterns']:
            score += len(patterns['good_patterns'])
            evidence.extend([f"Good pattern: {pattern}" for pattern in patterns['good_patterns'][:3]])
        
        # Bad patterns reduce score
        if patterns['bad_patterns']:
            score -= len(patterns['bad_patterns'])
            risks.extend([f"Code smell: {pattern}" for pattern in patterns['bad_patterns'][:3]])
            recommendations.append("Refactor code to remove anti-patterns")
        
        # Security concerns
        if patterns['security_concerns']:
            score -= 2
            risks.extend([f"Security issue: {pattern}" for pattern in patterns['security_concerns']])
            recommendations.append("Address security vulnerabilities")
        
        # Performance issues
        if patterns['performance_issues']:
            score -= 1
            recommendations.append("Optimize performance bottlenecks")
        
        # Test analysis
        test_analysis = self._analyze_tests(file_tree, artifacts)
        score += test_analysis['score']
        evidence.extend(test_analysis['evidence'])
        recommendations.extend(test_analysis['recommendations'])
        
        # Documentation analysis
        doc_analysis = self._analyze_documentation(readme, file_tree)
        score += doc_analysis['score']
        evidence.extend(doc_analysis['evidence'])
        
        # Generate insights
        insights = self._generate_code_insights(patterns, test_analysis, doc_analysis)
        
        # Calculate confidence
        confidence = self.get_confidence_score(len(evidence), len(patterns['good_patterns']))
        
        # Normalize score to 0-10
        score = max(0, min(score, 10))
        
        return AgentAnalysis(
            agent_name=self.name,
            score=score,
            confidence=confidence,
            evidence=evidence,
            recommendations=recommendations,
            insights=insights,
            risks=risks
        )
    
    def _get_code_files(self, file_tree: List[Dict[str, Any]]) -> List[str]:
        """Extract code files from file tree"""
        code_extensions = ['.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c', '.php', '.rb', '.swift', '.kt']
        code_files = []
        
        for file_info in file_tree:
            if file_info.get('type') == 'blob':
                file_path = file_info.get('path', '')
                if any(file_path.endswith(ext) for ext in code_extensions):
                    code_files.append(file_path)
        
        return code_files
    
    def _analyze_code_patterns(self, code_files: List[str], artifacts: Dict[str, Any]) -> Dict[str, List[str]]:
        """Analyze code patterns (simulated analysis)"""
        patterns = {
            'good_patterns': [],
            'bad_patterns': [],
            'security_concerns': [],
            'performance_issues': []
        }
        
        # Simulate pattern analysis based on file names and structure
        for file_path in code_files:
            file_name = file_path.lower()
            
            # Good patterns based on file structure
            if 'test' in file_name:
                patterns['good_patterns'].append('Test files present')
            if 'config' in file_name:
                patterns['good_patterns'].append('Configuration management')
            if 'util' in file_name or 'helper' in file_name:
                patterns['good_patterns'].append('Utility functions')
            if 'service' in file_name:
                patterns['good_patterns'].append('Service layer pattern')
            if 'model' in file_name:
                patterns['good_patterns'].append('Data modeling')
            
            # Bad patterns
            if 'temp' in file_name or 'tmp' in file_name:
                patterns['bad_patterns'].append('Temporary files in repo')
            if 'backup' in file_name:
                patterns['bad_patterns'].append('Backup files in repo')
            if 'old' in file_name:
                patterns['bad_patterns'].append('Outdated files')
        
        # Analyze lint results if available
        if artifacts.get('lint_results'):
            lint_results = artifacts['lint_results'].lower()
            if 'no issues' in lint_results:
                patterns['good_patterns'].append('Clean linting results')
            elif 'error' in lint_results:
                patterns['bad_patterns'].append('Linting errors present')
        
        return patterns
    
    def _analyze_tests(self, file_tree: List[Dict[str, Any]], artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test coverage and quality"""
        result = {
            'score': 0,
            'evidence': [],
            'recommendations': []
        }
        
        # Find test files
        test_files = [f for f in file_tree if 'test' in f.get('path', '').lower()]
        
        if test_files:
            result['score'] += 2
            result['evidence'].append(f"Test files found: {len(test_files)}")
        
        # Analyze test results
        if artifacts.get('test_results'):
            test_results = artifacts['test_results']
            if 'passed' in test_results.lower():
                result['score'] += 2
                result['evidence'].append("Tests are passing")
            
            if 'coverage' in test_results.lower():
                result['score'] += 1
                result['evidence'].append("Coverage reporting present")
        else:
            result['recommendations'].append("Add comprehensive test suite")
        
        return result
    
    def _analyze_documentation(self, readme: str, file_tree: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze code documentation"""
        result = {
            'score': 0,
            'evidence': []
        }
        
        if readme:
            result['score'] += 1
            result['evidence'].append("README present")
            
            # Check for code examples
            if '```' in readme:
                result['score'] += 1
                result['evidence'].append("Code examples in README")
        
        # Check for additional documentation
        doc_files = [f for f in file_tree if f.get('path', '').endswith(('.md', '.rst'))]
        if len(doc_files) > 1:  # More than just README
            result['score'] += 1
            result['evidence'].append("Additional documentation files")
        
        return result
    
    def _generate_code_insights(self, patterns: Dict, test_analysis: Dict, doc_analysis: Dict) -> List[str]:
        """Generate insights about the codebase"""
        insights = []
        
        if patterns['good_patterns']:
            insights.append(f"Codebase shows {len(patterns['good_patterns'])} good practices")
        
        if test_analysis['score'] >= 3:
            insights.append("Strong testing culture evident")
        
        if doc_analysis['score'] >= 2:
            insights.append("Good documentation practices")
        
        if not patterns['security_concerns']:
            insights.append("No obvious security issues detected")
        
        return insights


class ArchitectureAgent(BaseAIAgent):
    """AI agent for architecture analysis"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("ArchitectureAgent", api_key)
    
    def analyze(self, context: Dict[str, Any]) -> AgentAnalysis:
        """Analyze architecture and design patterns"""
        file_tree = context.get('file_tree', [])
        readme = context.get('readme', '')
        artifacts = context.get('artifacts', {})
        
        evidence = []
        recommendations = []
        insights = []
        risks = []
        score = 0
        
        # Analyze project structure
        structure_analysis = self._analyze_structure(file_tree)
        score += structure_analysis['score']
        evidence.extend(structure_analysis['evidence'])
        recommendations.extend(structure_analysis['recommendations'])
        
        # Analyze design patterns
        pattern_analysis = self._analyze_design_patterns(file_tree, readme)
        score += pattern_analysis['score']
        evidence.extend(pattern_analysis['evidence'])
        insights.extend(pattern_analysis['insights'])
        
        # Analyze scalability
        scalability_analysis = self._analyze_scalability(file_tree, artifacts)
        score += scalability_analysis['score']
        evidence.extend(scalability_analysis['evidence'])
        recommendations.extend(scalability_analysis['recommendations'])
        
        # Analyze security architecture
        security_analysis = self._analyze_security_architecture(file_tree, artifacts)
        score += security_analysis['score']
        evidence.extend(security_analysis['evidence'])
        risks.extend(security_analysis['risks'])
        
        # Calculate confidence
        confidence = self.get_confidence_score(len(evidence), len(insights))
        
        # Normalize score
        score = max(0, min(score, 10))
        
        return AgentAnalysis(
            agent_name=self.name,
            score=score,
            confidence=confidence,
            evidence=evidence,
            recommendations=recommendations,
            insights=insights,
            risks=risks
        )
    
    def _analyze_structure(self, file_tree: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze project structure"""
        result = {
            'score': 0,
            'evidence': [],
            'recommendations': []
        }
        
        # Count directories and files
        directories = [f for f in file_tree if f.get('type') == 'tree']
        files = [f for f in file_tree if f.get('type') == 'blob']
        
        result['evidence'].append(f"Project structure: {len(directories)} directories, {len(files)} files")
        
        # Check for common good practices
        good_dirs = ['src', 'lib', 'app', 'components', 'utils', 'config', 'tests', 'docs']
        found_dirs = set()
        
        for dir_info in directories:
            dir_name = dir_info.get('path', '').split('/')[0]
            if dir_name in good_dirs:
                found_dirs.add(dir_name)
        
        if len(found_dirs) >= 3:
            result['score'] += 3
            result['evidence'].append(f"Good directory structure: {', '.join(found_dirs)}")
        elif len(found_dirs) >= 1:
            result['score'] += 1
            result['evidence'].append(f"Basic directory structure: {', '.join(found_dirs)}")
        else:
            result['recommendations'].append("Improve project organization with proper directory structure")
        
        return result
    
    def _analyze_design_patterns(self, file_tree: List[Dict[str, Any]], readme: str) -> Dict[str, Any]:
        """Analyze design patterns and architecture"""
        result = {
            'score': 0,
            'evidence': [],
            'insights': []
        }
        
        # Look for architectural patterns
        patterns = {
            'mvc': ['controller', 'model', 'view'],
            'microservices': ['service', 'api', 'gateway'],
            'layered': ['presentation', 'business', 'data'],
            'component': ['component', 'module', 'widget']
        }
        
        found_patterns = []
        for pattern_name, keywords in patterns.items():
            for file_info in file_tree:
                file_path = file_info.get('path', '').lower()
                if any(keyword in file_path for keyword in keywords):
                    found_patterns.append(pattern_name)
                    break
        
        if found_patterns:
            result['score'] += 2
            result['evidence'].append(f"Architectural patterns detected: {', '.join(found_patterns)}")
            result['insights'].append(f"Uses {found_patterns[0]} architecture pattern")
        
        # Check README for architecture documentation
        if readme and any(keyword in readme.lower() for keyword in ['architecture', 'design', 'pattern']):
            result['score'] += 1
            result['evidence'].append("Architecture documented in README")
        
        return result
    
    def _analyze_scalability(self, file_tree: List[Dict[str, Any]], artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze scalability considerations"""
        result = {
            'score': 0,
            'evidence': [],
            'recommendations': []
        }
        
        # Check for configuration management
        config_files = ['.env', 'config', 'settings', 'docker']
        found_config = False
        
        for file_info in file_tree:
            file_path = file_info.get('path', '').lower()
            if any(config in file_path for config in config_files):
                found_config = True
                break
        
        if found_config:
            result['score'] += 1
            result['evidence'].append("Configuration management present")
        
        # Check for containerization
        if any('dockerfile' in f.get('path', '').lower() for f in file_tree):
            result['score'] += 2
            result['evidence'].append("Containerization (Docker) present")
        
        # Check for CI/CD
        if any('.github' in f.get('path', '') for f in file_tree):
            result['score'] += 1
            result['evidence'].append("CI/CD pipeline present")
        
        # Check for performance notes
        if artifacts.get('perf_notes'):
            result['score'] += 1
            result['evidence'].append("Performance considerations documented")
        
        return result
    
    def _analyze_security_architecture(self, file_tree: List[Dict[str, Any]], artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze security architecture"""
        result = {
            'score': 0,
            'evidence': [],
            'risks': []
        }
        
        # Check for security-related files
        security_files = ['security', 'auth', 'middleware', 'guard', 'jwt']
        found_security = False
        
        for file_info in file_tree:
            file_path = file_info.get('path', '').lower()
            if any(sec in file_path for sec in security_files):
                found_security = True
                result['evidence'].append(f"Security file: {file_info.get('path')}")
                break
        
        if found_security:
            result['score'] += 2
        else:
            result['risks'].append("No obvious security implementation found")
        
        # Check for environment configuration
        if any('.env.example' in f.get('path', '') for f in file_tree):
            result['score'] += 1
            result['evidence'].append("Environment variables template present")
        
        # Check for security analysis results
        if artifacts.get('sast_results'):
            result['score'] += 1
            result['evidence'].append("Security analysis performed")
        
        return result


class UIUXAgent(BaseAIAgent):
    """AI agent for UI/UX analysis"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("UIUXAgent", api_key)
    
    def analyze(self, context: Dict[str, Any]) -> AgentAnalysis:
        """Analyze UI/UX quality and accessibility"""
        file_tree = context.get('file_tree', [])
        readme = context.get('readme', '')
        artifacts = context.get('artifacts', {})
        
        evidence = []
        recommendations = []
        insights = []
        risks = []
        score = 0
        
        # Analyze UI files
        ui_analysis = self._analyze_ui_files(file_tree)
        score += ui_analysis['score']
        evidence.extend(ui_analysis['evidence'])
        recommendations.extend(ui_analysis['recommendations'])
        
        # Analyze accessibility
        a11y_analysis = self._analyze_accessibility(file_tree, artifacts)
        score += a11y_analysis['score']
        evidence.extend(a11y_analysis['evidence'])
        recommendations.extend(a11y_analysis['recommendations'])
        
        # Analyze responsive design
        responsive_analysis = self._analyze_responsive_design(file_tree)
        score += responsive_analysis['score']
        evidence.extend(responsive_analysis['evidence'])
        
        # Analyze demo/screenshots
        demo_analysis = self._analyze_demo(artifacts)
        score += demo_analysis['score']
        evidence.extend(demo_analysis['evidence'])
        
        # Calculate confidence
        confidence = self.get_confidence_score(len(evidence), len(insights))
        
        # Normalize score
        score = max(0, min(score, 10))
        
        return AgentAnalysis(
            agent_name=self.name,
            score=score,
            confidence=confidence,
            evidence=evidence,
            recommendations=recommendations,
            insights=insights,
            risks=risks
        )
    
    def _analyze_ui_files(self, file_tree: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze UI-related files"""
        result = {
            'score': 0,
            'evidence': [],
            'recommendations': []
        }
        
        ui_extensions = ['.html', '.jsx', '.tsx', '.vue', '.svelte', '.css', '.scss', '.sass']
        ui_files = []
        
        for file_info in file_tree:
            if file_info.get('type') == 'blob':
                file_path = file_info.get('path', '')
                if any(file_path.endswith(ext) for ext in ui_extensions):
                    ui_files.append(file_path)
        
        if ui_files:
            result['score'] += 2
            result['evidence'].append(f"UI files found: {len(ui_files)}")
            
            # Check for component structure
            components = [f for f in ui_files if 'component' in f.lower()]
            if components:
                result['score'] += 1
                result['evidence'].append("Component-based architecture")
        else:
            result['recommendations'].append("Add UI components and styling")
        
        return result
    
    def _analyze_accessibility(self, file_tree: List[Dict[str, Any]], artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze accessibility features"""
        result = {
            'score': 0,
            'evidence': [],
            'recommendations': []
        }
        
        # Check for accessibility files
        a11y_files = ['accessibility', 'a11y', 'aria', 'semantic']
        found_a11y = False
        
        for file_info in file_tree:
            file_path = file_info.get('path', '').lower()
            if any(a11y in file_path for a11y in a11y_files):
                found_a11y = True
                result['evidence'].append(f"Accessibility file: {file_info.get('path')}")
                break
        
        if found_a11y:
            result['score'] += 2
        else:
            result['recommendations'].append("Add accessibility considerations")
        
        # Check for accessibility notes in artifacts
        if artifacts.get('accessibility_notes'):
            result['score'] += 2
            result['evidence'].append("Accessibility compliance documented")
        
        return result
    
    def _analyze_responsive_design(self, file_tree: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze responsive design implementation"""
        result = {
            'score': 0,
            'evidence': []
        }
        
        # Look for CSS files that might contain responsive design
        css_files = [f for f in file_tree if f.get('path', '').endswith(('.css', '.scss', '.sass'))]
        
        if css_files:
            result['score'] += 1
            result['evidence'].append("Styling files present")
        
        # Check for mobile-specific files
        mobile_files = [f for f in file_tree if 'mobile' in f.get('path', '').lower()]
        if mobile_files:
            result['score'] += 1
            result['evidence'].append("Mobile-specific files found")
        
        return result
    
    def _analyze_demo(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze demo and screenshots"""
        result = {
            'score': 0,
            'evidence': []
        }
        
        if artifacts.get('screenshots_or_demo'):
            result['score'] += 2
            result['evidence'].append("Demo/screenshots provided")
        
        return result


class SecurityAgent(BaseAIAgent):
    """AI agent for security analysis"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("SecurityAgent", api_key)
    
    def analyze(self, context: Dict[str, Any]) -> AgentAnalysis:
        """Analyze security posture"""
        file_tree = context.get('file_tree', [])
        readme = context.get('readme', '')
        artifacts = context.get('artifacts', {})
        
        evidence = []
        recommendations = []
        insights = []
        risks = []
        score = 0
        
        # Analyze secrets management
        secrets_analysis = self._analyze_secrets_management(file_tree)
        score += secrets_analysis['score']
        evidence.extend(secrets_analysis['evidence'])
        risks.extend(secrets_analysis['risks'])
        recommendations.extend(secrets_analysis['recommendations'])
        
        # Analyze dependencies
        deps_analysis = self._analyze_dependencies(file_tree, artifacts)
        score += deps_analysis['score']
        evidence.extend(deps_analysis['evidence'])
        risks.extend(deps_analysis['risks'])
        
        # Analyze input validation
        validation_analysis = self._analyze_input_validation(file_tree)
        score += validation_analysis['score']
        evidence.extend(validation_analysis['evidence'])
        recommendations.extend(validation_analysis['recommendations'])
        
        # Analyze authentication
        auth_analysis = self._analyze_authentication(file_tree)
        score += auth_analysis['score']
        evidence.extend(auth_analysis['evidence'])
        recommendations.extend(auth_analysis['recommendations'])
        
        # Calculate confidence
        confidence = self.get_confidence_score(len(evidence), len(insights))
        
        # Normalize score
        score = max(0, min(score, 10))
        
        return AgentAnalysis(
            agent_name=self.name,
            score=score,
            confidence=confidence,
            evidence=evidence,
            recommendations=recommendations,
            insights=insights,
            risks=risks
        )
    
    def _analyze_secrets_management(self, file_tree: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze secrets management"""
        result = {
            'score': 0,
            'evidence': [],
            'risks': [],
            'recommendations': []
        }
        
        # Check for .env.example
        if any('.env.example' in f.get('path', '') for f in file_tree):
            result['score'] += 2
            result['evidence'].append("Environment variables template present")
        else:
            result['risks'].append("No environment variables template found")
            result['recommendations'].append("Add .env.example file")
        
        # Check for .env in gitignore
        gitignore_files = [f for f in file_tree if f.get('path', '').endswith('.gitignore')]
        if gitignore_files:
            result['score'] += 1
            result['evidence'].append(".gitignore file present")
        
        return result
    
    def _analyze_dependencies(self, file_tree: List[Dict[str, Any]], artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dependency security"""
        result = {
            'score': 0,
            'evidence': [],
            'risks': []
        }
        
        # Check for package files
        package_files = ['package.json', 'requirements.txt', 'pom.xml', 'Cargo.toml']
        found_packages = False
        
        for file_info in file_tree:
            if file_info.get('path', '').split('/')[-1] in package_files:
                found_packages = True
                result['evidence'].append(f"Package file: {file_info.get('path')}")
                break
        
        if found_packages:
            result['score'] += 1
        
        # Check for security scan results
        if artifacts.get('sast_results'):
            sast_results = artifacts['sast_results'].lower()
            if 'no critical' in sast_results:
                result['score'] += 2
                result['evidence'].append("Security scan passed")
            elif 'vulnerability' in sast_results:
                result['risks'].append("Security vulnerabilities detected")
        
        return result
    
    def _analyze_input_validation(self, file_tree: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze input validation"""
        result = {
            'score': 0,
            'evidence': [],
            'recommendations': []
        }
        
        # Look for validation files
        validation_files = ['validator', 'validation', 'schema', 'middleware']
        found_validation = False
        
        for file_info in file_tree:
            file_path = file_info.get('path', '').lower()
            if any(val in file_path for val in validation_files):
                found_validation = True
                result['evidence'].append(f"Validation file: {file_info.get('path')}")
                break
        
        if found_validation:
            result['score'] += 2
        else:
            result['recommendations'].append("Add input validation")
        
        return result
    
    def _analyze_authentication(self, file_tree: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze authentication implementation"""
        result = {
            'score': 0,
            'evidence': [],
            'recommendations': []
        }
        
        # Look for auth-related files
        auth_files = ['auth', 'login', 'jwt', 'oauth', 'session']
        found_auth = False
        
        for file_info in file_tree:
            file_path = file_info.get('path', '').lower()
            if any(auth in file_path for auth in auth_files):
                found_auth = True
                result['evidence'].append(f"Authentication file: {file_info.get('path')}")
                break
        
        if found_auth:
            result['score'] += 2
        else:
            result['recommendations'].append("Implement authentication system")
        
        return result


class LearningAgent(BaseAIAgent):
    """The learning agent - gets smarter over time"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("LearningAgent", api_key)
        # Store what we've learned
        self.learning_data = []
        self.pattern_weights = defaultdict(float)
        self.technology_patterns = defaultdict(list)
        self.score_predictions = {}
        self.quality_indicators = defaultdict(int)
        self.learning_model = None
        self.model_file = "learning_model.pkl"
        self.data_file = "learning_data.json"
        # Load any previous learning
        self._load_learning_data()
    
    def analyze(self, context: Dict[str, Any]) -> AgentAnalysis:
        """Use what we've learned to analyze this submission"""
        evidence = []
        recommendations = []
        insights = []
        risks = []
        
        # First, let's extract the important features
        features = self._extract_features(context)
        
        # Now predict the score based on what we've seen before
        predicted_score = self._predict_score(features)
        
        # Generate some insights based on patterns we've learned
        pattern_insights = self._generate_pattern_insights(features)
        insights.extend(pattern_insights)
        
        # Make recommendations based on what usually works
        ml_recommendations = self._generate_ml_recommendations(features)
        recommendations.extend(ml_recommendations)
        
        # Look for potential issues we've seen before
        risk_patterns = self._identify_risk_patterns(features)
        risks.extend(risk_patterns)
        
        # How confident are we in this analysis?
        confidence = self._calculate_ml_confidence(features)
        
        # Store this analysis so we can learn from it
        self._store_analysis(context, features, predicted_score, confidence)
        
        # Update our learning model with this new data
        self._update_learning_model()
        
        # Add some evidence about our learning process
        evidence.append(f"Learned from {len(self.learning_data)} previous analyses")
        evidence.append(f"Recognizing {len(self.pattern_weights)} different patterns")
        
        return AgentAnalysis(
            agent_name=self.name,
            score=predicted_score,
            confidence=confidence,
            evidence=evidence,
            recommendations=recommendations,
            insights=insights,
            risks=risks
        )
    
    def _extract_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract ML features from context"""
        features = {
            'file_count': 0,
            'code_file_count': 0,
            'test_file_count': 0,
            'doc_file_count': 0,
            'config_file_count': 0,
            'has_readme': False,
            'has_dockerfile': False,
            'has_ci_cd': False,
            'has_tests': False,
            'has_docs': False,
            'technology_stack': [],
            'architecture_patterns': [],
            'security_files': 0,
            'ui_files': 0,
            'complexity_score': 0.0
        }
        
        file_tree = context.get('file_tree', [])
        readme = context.get('readme', '')
        
        # Count files by type
        for file_info in file_tree:
            if file_info.get('type') == 'blob':
                file_path = file_info.get('path', '').lower()
                features['file_count'] += 1
                
                # Code files
                if any(file_path.endswith(ext) for ext in ['.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c']):
                    features['code_file_count'] += 1
                
                # Test files
                if 'test' in file_path:
                    features['test_file_count'] += 1
                    features['has_tests'] = True
                
                # Documentation files
                if file_path.endswith(('.md', '.rst')):
                    features['doc_file_count'] += 1
                    features['has_docs'] = True
                
                # Configuration files
                if any(config in file_path for config in ['.env', 'config', 'settings', 'package.json', 'requirements.txt']):
                    features['config_file_count'] += 1
                
                # Docker
                if 'dockerfile' in file_path:
                    features['has_dockerfile'] = True
                
                # CI/CD
                if '.github' in file_path:
                    features['has_ci_cd'] = True
                
                # Security files
                if any(sec in file_path for sec in ['auth', 'security', 'jwt', 'middleware']):
                    features['security_files'] += 1
                
                # UI files
                if any(file_path.endswith(ext) for ext in ['.html', '.jsx', '.tsx', '.vue', '.css', '.scss']):
                    features['ui_files'] += 1
        
        # README analysis
        features['has_readme'] = bool(readme)
        
        # Technology stack detection
        features['technology_stack'] = self._detect_technology_stack(file_tree, readme)
        
        # Architecture patterns
        features['architecture_patterns'] = self._detect_architecture_patterns(file_tree)
        
        # Complexity score
        features['complexity_score'] = self._calculate_complexity_score(features)
        
        return features
    
    def _detect_technology_stack(self, file_tree: List[Dict], readme: str) -> List[str]:
        """Detect technology stack from files and README"""
        stack = []
        
        # File-based detection
        for file_info in file_tree:
            file_path = file_info.get('path', '').lower()
            
            if file_path.endswith('.py'):
                stack.append('Python')
            elif file_path.endswith(('.js', '.jsx')):
                stack.append('JavaScript')
            elif file_path.endswith(('.ts', '.tsx')):
                stack.append('TypeScript')
            elif file_path.endswith('.java'):
                stack.append('Java')
            elif file_path.endswith('.go'):
                stack.append('Go')
            elif file_path.endswith('.rs'):
                stack.append('Rust')
            elif file_path.endswith(('.cpp', '.c')):
                stack.append('C++')
            elif file_path.endswith('.php'):
                stack.append('PHP')
            elif file_path.endswith('.rb'):
                stack.append('Ruby')
            elif file_path.endswith('.swift'):
                stack.append('Swift')
            elif file_path.endswith('.kt'):
                stack.append('Kotlin')
        
        # README-based detection
        if readme:
            readme_lower = readme.lower()
            if 'react' in readme_lower:
                stack.append('React')
            if 'vue' in readme_lower:
                stack.append('Vue')
            if 'angular' in readme_lower:
                stack.append('Angular')
            if 'django' in readme_lower:
                stack.append('Django')
            if 'flask' in readme_lower:
                stack.append('Flask')
            if 'express' in readme_lower:
                stack.append('Express')
            if 'spring' in readme_lower:
                stack.append('Spring')
            if 'rails' in readme_lower:
                stack.append('Rails')
        
        return list(set(stack))  # Remove duplicates
    
    def _detect_architecture_patterns(self, file_tree: List[Dict]) -> List[str]:
        """Detect architecture patterns from file structure"""
        patterns = []
        
        # MVC pattern
        if any('controller' in f.get('path', '').lower() for f in file_tree):
            patterns.append('MVC')
        
        # Microservices pattern
        if any('service' in f.get('path', '').lower() for f in file_tree):
            patterns.append('Microservices')
        
        # Component-based pattern
        if any('component' in f.get('path', '').lower() for f in file_tree):
            patterns.append('Component-based')
        
        # Layered architecture
        if any(keyword in f.get('path', '').lower() for f in file_tree for keyword in ['presentation', 'business', 'data']):
            patterns.append('Layered')
        
        return patterns
    
    def _calculate_complexity_score(self, features: Dict) -> float:
        """Calculate project complexity score"""
        complexity = 0.0
        
        # File count complexity
        complexity += min(features['file_count'] / 50.0, 1.0) * 0.3
        
        # Code file ratio
        if features['file_count'] > 0:
            code_ratio = features['code_file_count'] / features['file_count']
            complexity += code_ratio * 0.3
        
        # Technology stack diversity
        complexity += min(len(features['technology_stack']) / 5.0, 1.0) * 0.2
        
        # Architecture pattern complexity
        complexity += min(len(features['architecture_patterns']) / 3.0, 1.0) * 0.2
        
        return min(complexity, 1.0)
    
    def _predict_score(self, features: Dict) -> int:
        """Predict score using learned patterns"""
        if not self.learning_data:
            return 5  # Default score
        
        # Simple linear regression-like prediction
        score = 5.0  # Base score
        
        # File count bonus
        if features['file_count'] > 10:
            score += 1.0
        if features['file_count'] > 50:
            score += 1.0
        
        # Test coverage bonus
        if features['has_tests']:
            score += 1.5
        
        # Documentation bonus
        if features['has_docs']:
            score += 1.0
        
        # Configuration management bonus
        if features['config_file_count'] > 0:
            score += 0.5
        
        # Docker bonus
        if features['has_dockerfile']:
            score += 1.0
        
        # CI/CD bonus
        if features['has_ci_cd']:
            score += 1.0
        
        # Security bonus
        if features['security_files'] > 0:
            score += 1.0
        
        # Technology stack bonus
        if len(features['technology_stack']) > 1:
            score += 0.5
        
        # Architecture pattern bonus
        if features['architecture_patterns']:
            score += 1.0
        
        # Apply learned weights
        for pattern, weight in self.pattern_weights.items():
            if pattern in str(features):
                score += weight
        
        # Normalize to 0-10 range
        return max(0, min(int(round(score)), 10))
    
    def _generate_pattern_insights(self, features: Dict) -> List[str]:
        """Generate insights based on learned patterns"""
        insights = []
        
        # Technology stack insights
        if features['technology_stack']:
            stack_str = ', '.join(features['technology_stack'])
            insights.append(f"Uses modern technology stack: {stack_str}")
        
        # Architecture insights
        if features['architecture_patterns']:
            arch_str = ', '.join(features['architecture_patterns'])
            insights.append(f"Implements {arch_str} architecture pattern")
        
        # Complexity insights
        if features['complexity_score'] > 0.7:
            insights.append("High complexity project with multiple components")
        elif features['complexity_score'] < 0.3:
            insights.append("Simple project structure")
        
        # File organization insights
        if features['file_count'] > 20:
            insights.append("Well-organized project with multiple files")
        
        # Testing insights
        if features['has_tests']:
            insights.append("Good testing practices evident")
        
        return insights
    
    def _generate_ml_recommendations(self, features: Dict) -> List[str]:
        """Generate recommendations based on ML analysis"""
        recommendations = []
        
        # Missing tests
        if not features['has_tests'] and features['code_file_count'] > 5:
            recommendations.append("Add comprehensive test suite for better code quality")
        
        # Missing documentation
        if not features['has_docs'] and features['file_count'] > 10:
            recommendations.append("Add documentation to improve project maintainability")
        
        # Missing Docker
        if not features['has_dockerfile'] and features['file_count'] > 15:
            recommendations.append("Consider adding Docker for easier deployment")
        
        # Missing CI/CD
        if not features['has_ci_cd'] and features['file_count'] > 20:
            recommendations.append("Add CI/CD pipeline for automated testing and deployment")
        
        # Security improvements
        if features['security_files'] == 0 and features['code_file_count'] > 10:
            recommendations.append("Implement security measures and authentication")
        
        # Architecture improvements
        if not features['architecture_patterns'] and features['file_count'] > 15:
            recommendations.append("Consider implementing clear architectural patterns")
        
        return recommendations
    
    def _identify_risk_patterns(self, features: Dict) -> List[str]:
        """Identify risks based on learned patterns"""
        risks = []
        
        # High complexity without tests
        if features['complexity_score'] > 0.7 and not features['has_tests']:
            risks.append("High complexity project without adequate testing")
        
        # Large codebase without documentation
        if features['code_file_count'] > 20 and not features['has_docs']:
            risks.append("Large codebase without proper documentation")
        
        # No configuration management
        if features['file_count'] > 10 and features['config_file_count'] == 0:
            risks.append("No configuration management detected")
        
        # No security measures
        if features['code_file_count'] > 15 and features['security_files'] == 0:
            risks.append("No security implementation detected")
        
        return risks
    
    def _calculate_ml_confidence(self, features: Dict) -> float:
        """Calculate confidence based on ML model certainty"""
        if not self.learning_data:
            return 0.5
        
        # Base confidence from data size
        data_confidence = min(len(self.learning_data) / 100.0, 1.0)
        
        # Feature completeness confidence
        feature_confidence = 0.0
        if features['file_count'] > 0:
            feature_confidence += 0.2
        if features['has_readme']:
            feature_confidence += 0.2
        if features['technology_stack']:
            feature_confidence += 0.2
        if features['architecture_patterns']:
            feature_confidence += 0.2
        if features['has_tests']:
            feature_confidence += 0.2
        
        # Pattern recognition confidence
        pattern_confidence = min(len(self.pattern_weights) / 50.0, 1.0)
        
        # Combine confidences
        total_confidence = (data_confidence * 0.4 + feature_confidence * 0.4 + pattern_confidence * 0.2)
        
        return min(max(total_confidence, 0.1), 1.0)
    
    def _store_analysis(self, context: Dict, features: Dict, score: int, confidence: float):
        """Store analysis data for learning"""
        analysis_data = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'score': score,
            'confidence': confidence,
            'context_keys': list(context.keys()),
            'file_count': features['file_count'],
            'code_file_count': features['code_file_count'],
            'technology_stack': features['technology_stack'],
            'architecture_patterns': features['architecture_patterns']
        }
        
        self.learning_data.append(analysis_data)
        
        # Keep only recent data (last 1000 analyses)
        if len(self.learning_data) > 1000:
            self.learning_data = self.learning_data[-1000:]
        
        # Update pattern weights based on successful patterns
        self._update_pattern_weights(features, score)
        
        # Save data periodically
        if len(self.learning_data) % 10 == 0:
            self._save_learning_data()
    
    def _update_pattern_weights(self, features: Dict, score: int):
        """Update pattern weights based on score"""
        # Positive patterns (high scores)
        if score >= 7:
            for tech in features['technology_stack']:
                self.pattern_weights[f"tech_{tech}"] += 0.1
            
            for pattern in features['architecture_patterns']:
                self.pattern_weights[f"arch_{pattern}"] += 0.1
            
            if features['has_tests']:
                self.pattern_weights['has_tests'] += 0.1
            
            if features['has_docs']:
                self.pattern_weights['has_docs'] += 0.1
        
        # Negative patterns (low scores)
        elif score <= 3:
            if not features['has_tests']:
                self.pattern_weights['no_tests'] -= 0.1
            
            if not features['has_docs']:
                self.pattern_weights['no_docs'] -= 0.1
        
        # Normalize weights
        for pattern in self.pattern_weights:
            self.pattern_weights[pattern] = max(-1.0, min(1.0, self.pattern_weights[pattern]))
    
    def _update_learning_model(self):
        """Update the learning model with new data"""
        if len(self.learning_data) < 10:
            return
        
        # Simple model update - in production, this would use scikit-learn or similar
        recent_data = self.learning_data[-50:]  # Last 50 analyses
        
        # Calculate average scores by technology stack
        tech_scores = defaultdict(list)
        for data in recent_data:
            for tech in data['technology_stack']:
                tech_scores[tech].append(data['score'])
        
        # Update technology pattern scores
        for tech, scores in tech_scores.items():
            if len(scores) >= 3:  # Minimum 3 samples
                avg_score = sum(scores) / len(scores)
                self.technology_patterns[tech] = avg_score
    
    def _load_learning_data(self):
        """Load learning data from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.learning_data = data.get('learning_data', [])
                    self.pattern_weights = defaultdict(float, data.get('pattern_weights', {}))
                    self.technology_patterns = defaultdict(list, data.get('technology_patterns', {}))
        except Exception as e:
            print(f"Warning: Could not load learning data: {e}")
            self.learning_data = []
            self.pattern_weights = defaultdict(float)
            self.technology_patterns = defaultdict(list)
    
    def _save_learning_data(self):
        """Save learning data to file"""
        try:
            data = {
                'learning_data': self.learning_data,
                'pattern_weights': dict(self.pattern_weights),
                'technology_patterns': dict(self.technology_patterns),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save learning data: {e}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        if not self.learning_data:
            return {"message": "No learning data available"}
        
        recent_data = self.learning_data[-100:]  # Last 100 analyses
        
        return {
            "total_analyses": len(self.learning_data),
            "recent_analyses": len(recent_data),
            "learned_patterns": len(self.pattern_weights),
            "technology_patterns": len(self.technology_patterns),
            "average_score": sum(d['score'] for d in recent_data) / len(recent_data),
            "confidence_trend": sum(d['confidence'] for d in recent_data) / len(recent_data),
            "most_common_tech": Counter([tech for d in recent_data for tech in d['technology_stack']]).most_common(5),
            "pattern_weights": dict(sorted(self.pattern_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
        }


class AgentOrchestrator:
    """Coordinates all our AI agents - the conductor of the orchestra"""
    
    def __init__(self, api_key: Optional[str] = None):
        # Set up all our agents
        self.agents = {
            'code': CodeAnalysisAgent(api_key),
            'architecture': ArchitectureAgent(api_key),
            'ui_ux': UIUXAgent(api_key),
            'security': SecurityAgent(api_key),
            'learning': LearningAgent(api_key)
        }
        self.analysis_history = []  # Keep track of what we've done
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run all our agents and put their results together"""
        agent_results = {}
        
        # Let each agent do their thing
        for agent_name, agent in self.agents.items():
            try:
                result = agent.analyze(context)
                agent_results[agent_name] = result
            except Exception as e:
                print(f"Warning: Agent {agent_name} failed: {e}")
                # If an agent fails, give it a default result
                agent_results[agent_name] = AgentAnalysis(
                    agent_name=agent_name,
                    score=0,
                    confidence=0.0,
                    evidence=[],
                    recommendations=[],
                    insights=[],
                    risks=[f"Agent failed: {str(e)}"]
                )
        
        # Combine everything into a final result
        combined_result = self._combine_agent_results(agent_results)
        
        # Remember this analysis for learning
        self.analysis_history.append({
            'context': context,
            'results': agent_results,
            'combined': combined_result
        })
        
        return combined_result
    
    def _combine_agent_results(self, agent_results: Dict[str, AgentAnalysis]) -> Dict[str, Any]:
        """Combine results from all agents"""
        # Calculate weighted scores based on agent confidence
        total_score = 0
        total_weight = 0
        
        all_evidence = []
        all_recommendations = []
        all_insights = []
        all_risks = []
        
        for agent_name, result in agent_results.items():
            weight = result.confidence
            total_score += result.score * weight
            total_weight += weight
            
            all_evidence.extend([f"{agent_name}: {evidence}" for evidence in result.evidence])
            all_recommendations.extend([f"{agent_name}: {rec}" for rec in result.recommendations])
            all_insights.extend([f"{agent_name}: {insight}" for insight in result.insights])
            all_risks.extend([f"{agent_name}: {risk}" for risk in result.risks])
        
        # Calculate final score
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0
        
        return {
            'total_score': round(final_score, 1),
            'agent_scores': {name: result.score for name, result in agent_results.items()},
            'confidence_scores': {name: result.confidence for name, result in agent_results.items()},
            'evidence': all_evidence[:20],  # Limit to top 20
            'recommendations': all_recommendations[:15],  # Limit to top 15
            'insights': all_insights[:10],  # Limit to top 10
            'risks': all_risks[:10],  # Limit to top 10
            'agent_count': len(agent_results)
        }

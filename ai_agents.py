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

# Import UI rendering capabilities
try:
    from ui_renderer import UIRenderer, UIExecutionAgent
    UI_RENDERING_AVAILABLE = True
except ImportError:
    UI_RENDERING_AVAILABLE = False
    print("Warning: UI rendering capabilities not available. Install selenium, opencv-python, and pillow.")

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
            # Deduplicate bad patterns before adding to risks
            unique_bad_patterns = list(set(patterns['bad_patterns'][:3]))
            risks.extend([f"Code smell: {pattern}" for pattern in unique_bad_patterns])
            recommendations.extend([
                "🔴 HIGH PRIORITY: Fix code issues to improve quality",
                "🟡 MEDIUM PRIORITY: Set up code review process",
                "🟢 LOW PRIORITY: Consider using automated code analysis tools"
            ])
        
        # Security concerns
        if patterns['security_concerns']:
            score -= 2
            # Deduplicate security concerns before adding to risks
            unique_security_concerns = list(set(patterns['security_concerns']))
            risks.extend([f"Security vulnerability: {pattern}" for pattern in unique_security_concerns])
            recommendations.extend([
                "🔴 HIGH PRIORITY: Fix security issues immediately",
                "🔴 HIGH PRIORITY: Implement secure coding practices",
                "🟡 MEDIUM PRIORITY: Add automated security scanning",
                "🟡 MEDIUM PRIORITY: Review authentication systems"
            ])
        
        # Performance issues
        if patterns['performance_issues']:
            score -= 1
            recommendations.extend([
                "🟡 MEDIUM PRIORITY: Fix performance issues",
                "🟢 LOW PRIORITY: Add performance monitoring tools",
                "Consider caching strategies for frequently accessed data",
                "Review database queries and optimize slow operations"
            ])
        
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
        normalized_score = max(0, min(score, 10))
        
        # Generate detailed scoring explanation with normalized score
        scoring_explanation = self._generate_scoring_explanation(normalized_score, patterns, test_analysis, doc_analysis, code_files)
        insights.append(scoring_explanation)
        
        return AgentAnalysis(
            agent_name=self.name,
            score=normalized_score,
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
        """Comprehensive code pattern analysis following evaluation parameters"""
        patterns = {
            'good_patterns': [],
            'bad_patterns': [],
            'security_concerns': [],
            'performance_issues': [],
            'naming_conventions': [],
            'code_quality': [],
            'algorithmic_efficiency': [],
            'maintainability': []
        }
        
        # Enhanced pattern analysis based on comprehensive evaluation parameters
        for file_path in code_files:
            file_name = file_path.lower()
            
            # Code Quality & Readability Analysis
            if self._check_naming_conventions(file_path):
                patterns['naming_conventions'].append('Consistent naming conventions detected')
                patterns['good_patterns'].append('Proper naming conventions')
            
            if self._check_code_organization(file_path):
                patterns['code_quality'].append('Well-organized code structure')
                patterns['good_patterns'].append('Good code organization')
            
            # Code Organization & Structure Analysis
            if 'test' in file_name:
                patterns['good_patterns'].append('Test files present')
                patterns['maintainability'].append('Test coverage for maintainability')
            if 'config' in file_name:
                patterns['good_patterns'].append('Configuration management')
                patterns['code_quality'].append('Parameterized configuration')
            if 'util' in file_name or 'helper' in file_name:
                patterns['good_patterns'].append('Utility functions')
                patterns['code_quality'].append('DRY principle - reusable utilities')
            if 'service' in file_name:
                patterns['good_patterns'].append('Service layer pattern')
                patterns['code_quality'].append('Separation of concerns')
            if 'model' in file_name:
                patterns['good_patterns'].append('Data modeling')
                patterns['code_quality'].append('Clear data abstraction')
            
            # Algorithmic Efficiency Analysis
            if self._check_algorithmic_efficiency(file_path):
                patterns['algorithmic_efficiency'].append('Efficient data structures detected')
                patterns['good_patterns'].append('Optimized algorithms')
            
            # Scalability & Maintainability Analysis
            if self._check_maintainability(file_path):
                patterns['maintainability'].append('Low coupling, high cohesion')
                patterns['good_patterns'].append('Maintainable code structure')
            
            # Bad patterns and code smells
            if 'temp' in file_name or 'tmp' in file_name:
                patterns['bad_patterns'].append('Temporary files in repo')
                patterns['code_quality'].append('Code smell: temporary files')
            if 'backup' in file_name:
                patterns['bad_patterns'].append('Backup files in repo')
                patterns['code_quality'].append('Code smell: backup files')
            if 'old' in file_name:
                patterns['bad_patterns'].append('Outdated files')
                patterns['maintainability'].append('Maintainability issue: outdated code')
            
            # Security and Performance Analysis
            if self._check_security_concerns(file_path):
                patterns['security_concerns'].append('Potential security vulnerability')
            if self._check_performance_issues(file_path):
                patterns['performance_issues'].append('Performance bottleneck detected')
        
        # Enhanced lint results analysis
        if artifacts.get('lint_results'):
            lint_results = artifacts['lint_results'].lower()
            if 'no issues' in lint_results:
                patterns['good_patterns'].append('Clean linting results')
                patterns['code_quality'].append('Adherence to coding standards')
            elif 'error' in lint_results:
                patterns['bad_patterns'].append('Linting errors present')
                patterns['code_quality'].append('Code quality issues detected')
        
        return patterns
    
    def _check_naming_conventions(self, file_path: str) -> bool:
        """Check for consistent naming conventions"""
        # Check for consistent naming patterns
        if any(pattern in file_path.lower() for pattern in ['camelcase', 'snake_case', 'kebab-case']):
            return True
        # Check for proper file naming
        if file_path.count('_') > 0 or file_path.count('-') > 0:
            return True
        return False
    
    def _check_code_organization(self, file_path: str) -> bool:
        """Check for good code organization patterns"""
        # Check for modular structure
        if any(folder in file_path.lower() for folder in ['src', 'lib', 'app', 'components', 'utils']):
            return True
        return False
    
    def _check_algorithmic_efficiency(self, file_path: str) -> bool:
        """Check for algorithmic efficiency indicators"""
        # Check for efficient data structures
        if any(ds in file_path.lower() for ds in ['hash', 'map', 'set', 'tree', 'graph', 'queue', 'stack']):
            return True
        return False
    
    def _check_maintainability(self, file_path: str) -> bool:
        """Check for maintainability indicators"""
        # Check for loose coupling indicators
        if any(pattern in file_path.lower() for pattern in ['interface', 'abstract', 'base', 'contract']):
            return True
        return False
    
    def _check_security_concerns(self, file_path: str) -> bool:
        """Check for security concerns"""
        # Check for potential security issues
        if any(concern in file_path.lower() for concern in ['password', 'secret', 'key', 'token']):
            return True
        return False
    
    def _check_performance_issues(self, file_path: str) -> bool:
        """Check for performance issues"""
        # Check for potential performance bottlenecks
        if any(issue in file_path.lower() for issue in ['loop', 'recursive', 'nested', 'heavy']):
            return True
        return False
    
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
            result['recommendations'].extend([
                "Consider adding tests for better code reliability (optional)",
                "Implement test-driven development (TDD) practices (optional)",
                "Set up automated testing in CI/CD pipeline (optional)",
                "Aim for good code coverage (optional)",
                "Include performance and security testing (optional)"
            ])
        
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
        """Generate human-friendly insights about the codebase"""
        insights = []
        
        # Overall assessment (simplified)
        if patterns['good_patterns']:
            insights.append("Follows good coding practices and standards")
        else:
            insights.append("Could benefit from better coding practices")
        
        # Testing insights (human-friendly)
        if test_analysis['score'] >= 3:
            insights.append("Includes comprehensive testing to prevent bugs")
        elif test_analysis['score'] >= 1:
            insights.append("Some testing present but could be expanded")
        else:
            insights.append("No testing found - this could lead to bugs in production")
        
        # Documentation insights (simplified)
        if doc_analysis['score'] >= 2:
            insights.append("Well-documented code that's easy to understand and use")
        elif doc_analysis['score'] >= 1:
            insights.append("Basic documentation present but could be improved")
        else:
            insights.append("Limited documentation - may be hard for others to understand")
        
        # Security insights (simplified)
        if not patterns['security_concerns']:
            insights.append("No major security issues detected")
        else:
            insights.append(f"Some security concerns found that need attention")
        
        # Performance insights (simplified)
        if patterns['performance_issues']:
            insights.append("Some performance issues that could slow down the application")
        else:
            insights.append("Code appears to run efficiently")
        
        # Code quality insights (simplified)
        if patterns['bad_patterns']:
            insights.append(f"Some code issues found that could be improved")
        else:
            insights.append("Code is well-written with good practices")
        
        return insights
    
    def _generate_scoring_explanation(self, score: int, patterns: Dict[str, List[str]], 
                                     test_analysis: Dict[str, Any], doc_analysis: Dict[str, Any], 
                                     code_files: List[str]) -> str:
        """Generate human-friendly explanation of code analysis scoring"""
        
        # Human-friendly assessment
        if score >= 8:
            assessment = "Excellent code quality with strong practices"
        elif score >= 6:
            assessment = "Good code quality with some areas for improvement"
        elif score >= 4:
            assessment = "Fair code quality with several issues"
        else:
            assessment = "Poor code quality requiring significant improvements"
        
        explanation = f"Code Quality: {score}/10 - {assessment}\n\n"
        
        # Why this score is high (if high)
        if score >= 8:
            explanation += "🎯 Why this score is high:\n"
            if len(code_files) > 0:
                explanation += f"  • Code files: {len(code_files)} files found - Good codebase size\n"
            if patterns['naming_conventions']:
                explanation += f"  • Naming conventions: {len(patterns['naming_conventions'])} good patterns - Clear and consistent naming\n"
            if patterns['code_quality']:
                explanation += f"  • Code quality: {len(patterns['code_quality'])} good patterns - Well-written code\n"
            if patterns['good_patterns']:
                explanation += f"  • Good patterns: {len(patterns['good_patterns'])} patterns - Strong code organization\n"
            if patterns['algorithmic_efficiency']:
                explanation += f"  • Algorithmic efficiency: {len(patterns['algorithmic_efficiency'])} patterns - Efficient algorithms\n"
        
        # Why some points were deducted (if low)
        if score < 8:
            explanation += "⚠️ Why some points were deducted:\n"
            if len(code_files) == 0:
                explanation += f"  • Code files: No code files found - No codebase to analyze\n"
            if len(patterns['naming_conventions']) < 5:
                if not patterns['naming_conventions']:
                    explanation += f"  • Naming conventions: No good patterns found - Poor naming conventions\n"
                else:
                    explanation += f"  • Naming conventions: Only {len(patterns['naming_conventions'])} patterns found - Could improve consistency\n"
            if len(patterns['code_quality']) < 5:
                if not patterns['code_quality']:
                    explanation += f"  • Code quality: No good patterns found - Poor code quality\n"
                else:
                    explanation += f"  • Code quality: Only {len(patterns['code_quality'])} patterns found - Some quality issues exist\n"
            if len(patterns['good_patterns']) < 3:
                if not patterns['good_patterns']:
                    explanation += f"  • Good patterns: No positive patterns found - Poor code organization\n"
                else:
                    explanation += f"  • Good patterns: Only {len(patterns['good_patterns'])} patterns found - Code organization could be better\n"
            if len(patterns['algorithmic_efficiency']) < 3:
                if not patterns['algorithmic_efficiency']:
                    explanation += f"  • Algorithmic efficiency: No efficient patterns found - Poor performance\n"
                else:
                    explanation += f"  • Algorithmic efficiency: Only {len(patterns['algorithmic_efficiency'])} patterns found - Performance could be improved\n"
        
        # Key Reasons for the Score
        explanation += f"\n📊 Key Reasons for the Score:\n"
        explanation += f"  • Code files: {len(code_files)} files - How many code files are present\n"
        explanation += f"  • Naming conventions: {len(patterns['naming_conventions'])} patterns - How clear the naming is\n"
        explanation += f"  • Code quality: {len(patterns['code_quality'])} patterns - How well-written the code is\n"
        explanation += f"  • Good patterns: {len(patterns['good_patterns'])} patterns - How well-organized the code is\n"
        explanation += f"  • Algorithmic efficiency: {len(patterns['algorithmic_efficiency'])} patterns - How efficient the algorithms are\n"
        
        # In Simple Terms
        explanation += f"\n💡 In Simple Terms:\n"
        if score >= 8:
            explanation += "This code is excellent! It's well-written, well-organized, and follows good practices. It's easy to read, maintain, and extend."
        elif score >= 6:
            explanation += "This code is good overall. It has some good practices but could be improved in a few areas. It's generally well-written and maintainable."
        elif score >= 4:
            explanation += "This code is okay but needs improvement. It has some good aspects but also has issues that should be addressed for better quality."
        else:
            explanation += "This code needs significant improvement. It has many issues that should be fixed for better quality, maintainability, and performance."
        
        return explanation


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
        
        # Analyze code-to-architecture alignment
        alignment_analysis = self._analyze_code_architecture_alignment(file_tree, readme)
        score += alignment_analysis['score']
        evidence.extend(alignment_analysis['evidence'])
        insights.extend(alignment_analysis['insights'])
        
        # Calculate confidence
        confidence = self.get_confidence_score(len(evidence), len(insights))
        
        # Normalize score
        normalized_score = max(0, min(score, 10))
        
        # Generate detailed scoring explanation with normalized score
        scoring_explanation = self._generate_architecture_scoring_explanation(
            normalized_score, structure_analysis, pattern_analysis, scalability_analysis, security_analysis
        )
        insights.append(scoring_explanation)
        
        return AgentAnalysis(
            agent_name=self.name,
            score=normalized_score,
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
        """Comprehensive design pattern and architecture analysis following evaluation parameters"""
        result = {
            'score': 0,
            'evidence': [],
            'insights': []
        }
        
        # Enhanced architectural pattern detection
        patterns = {
            'mvc': ['controller', 'model', 'view', 'mvc'],
            'microservices': ['service', 'api', 'gateway', 'microservice'],
            'layered': ['presentation', 'business', 'data', 'layer'],
            'component': ['component', 'module', 'widget', 'ui'],
            'event_driven': ['event', 'listener', 'handler', 'publisher'],
            'hexagonal': ['port', 'adapter', 'hexagon'],
            'clean_architecture': ['domain', 'application', 'infrastructure']
        }
        
        found_patterns = []
        layer_analysis = self._analyze_layer_separation(file_tree)
        data_flow_analysis = self._analyze_data_flow(file_tree)
        
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
        
        # Layer separation analysis
        if layer_analysis['score'] > 0:
            result['score'] += layer_analysis['score']
            result['evidence'].extend(layer_analysis['evidence'])
            result['insights'].extend(layer_analysis['insights'])
        
        # Data flow analysis
        if data_flow_analysis['score'] > 0:
            result['score'] += data_flow_analysis['score']
            result['evidence'].extend(data_flow_analysis['evidence'])
            result['insights'].extend(data_flow_analysis['insights'])
        
        # Check README for architecture documentation
        if readme and any(keyword in readme.lower() for keyword in ['architecture', 'design', 'pattern']):
            result['score'] += 1
            result['evidence'].append("Architecture documented in README")
        
        return result
    
    def _analyze_layer_separation(self, file_tree: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze layer separation in architecture"""
        result = {
            'score': 0,
            'evidence': [],
            'insights': []
        }
        
        # Check for clear layer separation
        layers = {
            'presentation': ['ui', 'view', 'component', 'page', 'screen'],
            'business': ['service', 'logic', 'business', 'domain'],
            'data': ['model', 'entity', 'repository', 'dao', 'database'],
            'infrastructure': ['config', 'util', 'helper', 'common']
        }
        
        found_layers = set()
        for layer_name, keywords in layers.items():
            for file_info in file_tree:
                file_path = file_info.get('path', '').lower()
                if any(keyword in file_path for keyword in keywords):
                    found_layers.add(layer_name)
                    break
        
        if len(found_layers) >= 3:
            result['score'] += 2
            result['evidence'].append(f"Clear layer separation: {', '.join(found_layers)}")
            result['insights'].append("Good separation of concerns with multiple architectural layers")
        elif len(found_layers) >= 2:
            result['score'] += 1
            result['evidence'].append(f"Basic layer separation: {', '.join(found_layers)}")
            result['insights'].append("Some layer separation present")
        
        return result
    
    def _analyze_data_flow(self, file_tree: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze data flow and control flow in architecture"""
        result = {
            'score': 0,
            'evidence': [],
            'insights': []
        }
        
        # Check for data flow indicators
        data_flow_indicators = ['api', 'endpoint', 'route', 'controller', 'service', 'repository']
        control_flow_indicators = ['middleware', 'interceptor', 'filter', 'guard', 'decorator']
        
        data_flow_count = 0
        control_flow_count = 0
        
        for file_info in file_tree:
            file_path = file_info.get('path', '').lower()
            if any(indicator in file_path for indicator in data_flow_indicators):
                data_flow_count += 1
            if any(indicator in file_path for indicator in control_flow_indicators):
                control_flow_count += 1
        
        if data_flow_count >= 3:
            result['score'] += 1
            result['evidence'].append(f"Clear data flow with {data_flow_count} data flow components")
            result['insights'].append("Well-defined data flow architecture")
        
        if control_flow_count >= 2:
            result['score'] += 1
            result['evidence'].append(f"Control flow management with {control_flow_count} control components")
            result['insights'].append("Proper control flow handling")
        
        return result
    
    def _analyze_scalability(self, file_tree: List[Dict[str, Any]], artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive scalability and performance readiness analysis"""
        result = {
            'score': 0,
            'evidence': [],
            'recommendations': []
        }
        
        # Horizontal/Vertical Scalability Design
        scalability_analysis = self._analyze_scalability_design(file_tree)
        result['score'] += scalability_analysis['score']
        result['evidence'].extend(scalability_analysis['evidence'])
        result['recommendations'].extend(scalability_analysis['recommendations'])
        
        # Caching Strategy Analysis
        caching_analysis = self._analyze_caching_strategy(file_tree)
        result['score'] += caching_analysis['score']
        result['evidence'].extend(caching_analysis['evidence'])
        
        # Async Processing Implementation
        async_analysis = self._analyze_async_processing(file_tree)
        result['score'] += async_analysis['score']
        result['evidence'].extend(async_analysis['evidence'])
        
        # Load Handling Capability
        load_analysis = self._analyze_load_handling(file_tree)
        result['score'] += load_analysis['score']
        result['evidence'].extend(load_analysis['evidence'])
        
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
    
    def _analyze_scalability_design(self, file_tree: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze horizontal/vertical scalability design"""
        result = {
            'score': 0,
            'evidence': [],
            'recommendations': []
        }
        
        # Check for horizontal scalability indicators
        horizontal_indicators = ['load_balancer', 'cluster', 'distributed', 'microservice', 'api_gateway']
        vertical_indicators = ['optimization', 'performance', 'memory', 'cpu', 'resource']
        
        horizontal_count = 0
        vertical_count = 0
        
        for file_info in file_tree:
            file_path = file_info.get('path', '').lower()
            if any(indicator in file_path for indicator in horizontal_indicators):
                horizontal_count += 1
            if any(indicator in file_path for indicator in vertical_indicators):
                vertical_count += 1
        
        if horizontal_count >= 2:
            result['score'] += 2
            result['evidence'].append(f"Horizontal scalability design with {horizontal_count} indicators")
        elif horizontal_count >= 1:
            result['score'] += 1
            result['evidence'].append("Basic horizontal scalability considerations")
        
        if vertical_count >= 2:
            result['score'] += 1
            result['evidence'].append(f"Vertical scalability optimization with {vertical_count} indicators")
        
        return result
    
    def _analyze_caching_strategy(self, file_tree: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze caching strategy implementation"""
        result = {
            'score': 0,
            'evidence': []
        }
        
        # Check for caching indicators
        caching_indicators = ['cache', 'redis', 'memcached', 'session', 'storage']
        caching_count = 0
        
        for file_info in file_tree:
            file_path = file_info.get('path', '').lower()
            if any(indicator in file_path for indicator in caching_indicators):
                caching_count += 1
        
        if caching_count >= 2:
            result['score'] += 2
            result['evidence'].append(f"Comprehensive caching strategy with {caching_count} caching components")
        elif caching_count >= 1:
            result['score'] += 1
            result['evidence'].append("Basic caching implementation")
        
        return result
    
    def _analyze_async_processing(self, file_tree: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze async processing implementation"""
        result = {
            'score': 0,
            'evidence': []
        }
        
        # Check for async indicators
        async_indicators = ['async', 'await', 'promise', 'future', 'callback', 'queue', 'worker']
        async_count = 0
        
        for file_info in file_tree:
            file_path = file_info.get('path', '').lower()
            if any(indicator in file_path for indicator in async_indicators):
                async_count += 1
        
        if async_count >= 2:
            result['score'] += 2
            result['evidence'].append(f"Async processing implementation with {async_count} async components")
        elif async_count >= 1:
            result['score'] += 1
            result['evidence'].append("Basic async processing")
        
        return result
    
    def _analyze_load_handling(self, file_tree: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze load handling capability"""
        result = {
            'score': 0,
            'evidence': []
        }
        
        # Check for load handling indicators
        load_indicators = ['rate_limit', 'throttle', 'circuit_breaker', 'bulkhead', 'timeout']
        load_count = 0
        
        for file_info in file_tree:
            file_path = file_info.get('path', '').lower()
            if any(indicator in file_path for indicator in load_indicators):
                load_count += 1
        
        if load_count >= 2:
            result['score'] += 2
            result['evidence'].append(f"Advanced load handling with {load_count} load management components")
        elif load_count >= 1:
            result['score'] += 1
            result['evidence'].append("Basic load handling implementation")
        
        return result
    
    def _analyze_code_architecture_alignment(self, file_tree: List[Dict[str, Any]], readme: str) -> Dict[str, Any]:
        """Analyze code-to-architecture alignment"""
        result = {
            'score': 0,
            'evidence': [],
            'insights': []
        }
        
        # Check for architectural consistency
        consistency_analysis = self._check_architectural_consistency(file_tree)
        result['score'] += consistency_analysis['score']
        result['evidence'].extend(consistency_analysis['evidence'])
        result['insights'].extend(consistency_analysis['insights'])
        
        # Check for implementation gaps
        gap_analysis = self._check_implementation_gaps(file_tree, readme)
        result['score'] += gap_analysis['score']
        result['evidence'].extend(gap_analysis['evidence'])
        result['insights'].extend(gap_analysis['insights'])
        
        return result
    
    def _check_architectural_consistency(self, file_tree: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for architectural consistency between design and implementation"""
        result = {
            'score': 0,
            'evidence': [],
            'insights': []
        }
        
        # Check for consistent architectural patterns
        pattern_consistency = 0
        architectural_files = []
        
        for file_info in file_tree:
            file_path = file_info.get('path', '').lower()
            if any(pattern in file_path for pattern in ['controller', 'service', 'model', 'repository', 'component']):
                architectural_files.append(file_path)
                pattern_consistency += 1
        
        if pattern_consistency >= 4:
            result['score'] += 2
            result['evidence'].append(f"High architectural consistency with {pattern_consistency} architectural components")
            result['insights'].append("Implementation closely follows architectural design")
        elif pattern_consistency >= 2:
            result['score'] += 1
            result['evidence'].append(f"Moderate architectural consistency with {pattern_consistency} components")
            result['insights'].append("Some architectural consistency present")
        
        return result
    
    def _check_implementation_gaps(self, file_tree: List[Dict[str, Any]], readme: str) -> Dict[str, Any]:
        """Check for gaps between architecture and implementation"""
        result = {
            'score': 0,
            'evidence': [],
            'insights': []
        }
        
        # Check for missing architectural components
        expected_components = ['controller', 'service', 'model', 'repository', 'component', 'api']
        found_components = set()
        
        for file_info in file_tree:
            file_path = file_info.get('path', '').lower()
            for component in expected_components:
                if component in file_path:
                    found_components.add(component)
        
        coverage_ratio = len(found_components) / len(expected_components)
        
        if coverage_ratio >= 0.8:
            result['score'] += 2
            result['evidence'].append(f"High implementation coverage: {len(found_components)}/{len(expected_components)} components")
            result['insights'].append("Minimal architectural gaps detected")
        elif coverage_ratio >= 0.5:
            result['score'] += 1
            result['evidence'].append(f"Moderate implementation coverage: {len(found_components)}/{len(expected_components)} components")
            result['insights'].append("Some architectural gaps present")
        else:
            result['evidence'].append(f"Low implementation coverage: {len(found_components)}/{len(expected_components)} components")
            result['insights'].append("Significant architectural gaps detected")
        
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
    
    def _generate_architecture_scoring_explanation(self, score: int, structure_analysis: Dict[str, Any], 
                                                 pattern_analysis: Dict[str, Any], scalability_analysis: Dict[str, Any], 
                                                 security_analysis: Dict[str, Any]) -> str:
        """Generate human-friendly explanation of architecture scoring"""
        
        # Get individual scores
        structure_score = structure_analysis.get('score', 0)
        pattern_score = pattern_analysis.get('score', 0)
        scalability_score = scalability_analysis.get('score', 0)
        security_score = security_analysis.get('score', 0)
        
        # Human-friendly assessment
        if score >= 8:
            assessment = "Excellent architecture with strong patterns and scalability"
        elif score >= 6:
            assessment = "Good architecture with some areas for improvement"
        elif score >= 4:
            assessment = "Fair architecture with several architectural concerns"
        else:
            assessment = "Poor architecture requiring significant redesign"
        
        explanation = f"Architecture Analysis: {score}/10 - {assessment}\n\n"
        
        # Why this score is high (if high)
        if score >= 8:
            explanation += "🎯 Why this score is high:\n"
            if structure_score >= 7:
                explanation += f"  • Project structure: {structure_score}/10 - Well-organized codebase\n"
            if pattern_score >= 7:
                explanation += f"  • Design patterns: {pattern_score}/10 - Good architectural patterns implemented\n"
            if scalability_score >= 7:
                explanation += f"  • Scalability: {scalability_score}/10 - Architecture designed for scale\n"
            if security_score >= 7:
                explanation += f"  • Security architecture: {security_score}/10 - Strong security considerations\n"
        
        # Why some points were deducted (if low)
        if score < 8:
            explanation += "⚠️ Why some points were deducted:\n"
            if structure_score < 8:
                if structure_score < 5:
                    explanation += f"  • Project structure: {structure_score}/10 - Poor organization or missing structure\n"
                else:
                    explanation += f"  • Project structure: {structure_score}/10 - Good but could be better organized\n"
            if pattern_score < 8:
                if pattern_score < 5:
                    explanation += f"  • Design patterns: {pattern_score}/10 - No design patterns detected\n"
                else:
                    explanation += f"  • Design patterns: {pattern_score}/10 - Some patterns used but not comprehensive\n"
            if scalability_score < 8:
                if scalability_score < 5:
                    explanation += f"  • Scalability: {scalability_score}/10 - Architecture not designed for scale\n"
                else:
                    explanation += f"  • Scalability: {scalability_score}/10 - Basic scalability but not optimal\n"
            if security_score < 8:
                if security_score < 5:
                    explanation += f"  • Security architecture: {security_score}/10 - No security architecture detected\n"
                else:
                    explanation += f"  • Security architecture: {security_score}/10 - Some security considerations but not comprehensive\n"
        
        # Key Reasons for the Score
        explanation += f"\n📊 Key Reasons for the Score:\n"
        explanation += f"  • Project structure: {structure_score}/10 - How well the codebase is organized\n"
        explanation += f"  • Design patterns: {pattern_score}/10 - How many architectural patterns are used\n"
        explanation += f"  • Scalability: {scalability_score}/10 - How well the architecture can scale\n"
        explanation += f"  • Security architecture: {security_score}/10 - How well security is built into the architecture\n"
        
        # In Simple Terms
        explanation += f"\n💡 In Simple Terms:\n"
        if score >= 8:
            explanation += "This project has excellent architecture! The code is well-organized, uses good design patterns, can scale well, and has strong security. It's built to last and grow."
        elif score >= 6:
            explanation += "This project has good architecture. The code is reasonably organized, uses some design patterns, and has decent scalability. It's solid but could be improved."
        elif score >= 4:
            explanation += "This project has basic architecture. The code organization is okay, but it lacks design patterns and scalability considerations. It works but needs improvement."
        else:
            explanation += "This project has poor architecture. The code is poorly organized, lacks design patterns, and has no scalability or security considerations. It needs significant redesign."
        
        return explanation


class UIUXAgent(BaseAIAgent):
    """AI agent for UI/UX analysis with UI rendering capabilities"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("UIUXAgent", api_key)
        self.ui_renderer = UIRenderer() if UI_RENDERING_AVAILABLE else None
    
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
        
        # NEW: Execute and analyze UI if possible
        ui_execution_analysis = self._analyze_ui_execution(context)
        if ui_execution_analysis:
            score += ui_execution_analysis['score']
            evidence.extend(ui_execution_analysis['evidence'])
            recommendations.extend(ui_execution_analysis['recommendations'])
            insights.extend(ui_execution_analysis['insights'])
        
        # Add basic insights
        if ui_analysis['score'] > 0:
            insights.append("UI files and components detected")
        if a11y_analysis['score'] > 0:
            insights.append("Accessibility features implemented")
        if responsive_analysis['score'] > 0:
            insights.append("Responsive design patterns found")
        if demo_analysis['score'] > 0:
            insights.append("Demo/screenshots available for evaluation")
        if ui_execution_analysis and ui_execution_analysis['score'] > 0:
            insights.append("UI execution and interaction testing completed")
        
        # Calculate confidence
        confidence = self.get_confidence_score(len(evidence), len(insights))
        
        # Normalize score
        normalized_score = max(0, min(score, 10))
        
        # Generate detailed scoring explanation with normalized score
        scoring_explanation = self._generate_ui_scoring_explanation(
            normalized_score, ui_analysis, a11y_analysis, responsive_analysis, demo_analysis, ui_execution_analysis
        )
        insights.append(scoring_explanation)
        
        return AgentAnalysis(
            agent_name=self.name,
            score=normalized_score,
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
            result['score'] += 3  # Increased base score
            result['evidence'].append(f"UI files found: {len(ui_files)}")
            
            # Check for modern frameworks
            react_files = [f for f in ui_files if 'react' in f.lower() or 'jsx' in f.lower()]
            if react_files:
                result['score'] += 2
                result['evidence'].append("React framework detected - modern UI development")
            
            vue_files = [f for f in ui_files if 'vue' in f.lower()]
            if vue_files:
                result['score'] += 2
                result['evidence'].append("Vue framework detected - modern UI development")
            
            angular_files = [f for f in ui_files if 'angular' in f.lower()]
            if angular_files:
                result['score'] += 2
                result['evidence'].append("Angular framework detected - enterprise UI development")
            
            # Check for component structure
            components = [f for f in ui_files if 'component' in f.lower()]
            if components:
                result['score'] += 1
                result['evidence'].append(f"Component-based architecture ({len(components)} components)")
            
            # Check for CSS frameworks
            css_frameworks = [f for f in ui_files if any(fw in f.lower() for fw in ['bootstrap', 'tailwind', 'material', 'bulma'])]
            if css_frameworks:
                result['score'] += 1
                result['evidence'].append("CSS framework detected - improved styling")
            
            # Recommendations based on findings
            if result['score'] < 6:
                result['recommendations'].append("Consider using a modern UI framework (React, Vue, Angular)")
                result['recommendations'].append("Add CSS framework for better styling (Bootstrap, Tailwind)")
        else:
            result['recommendations'].append("Add UI components and styling")
            result['recommendations'].append("Create HTML/CSS/JS files for user interface")
        
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
    
    def _analyze_ui_execution(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute and analyze the UI if possible"""
        if not self.ui_renderer:
            return None
        
        try:
            # Get project path from context
            project_path = context.get('project_path', '.')
            if not os.path.exists(project_path):
                return None
            
            # Try to execute the application
            execution_result = self.ui_renderer.execute_web_app(project_path)
            
            if not execution_result['success']:
                errors = execution_result.get('errors', ['Unknown error'])
                error_msg = errors[0] if errors else "Unknown error"
                
                # Provide more helpful analysis even when execution fails
                return {
                    'score': 4,  # Give more points for having UI files
                    'evidence': [
                        f'UI execution attempted but failed: {error_msg}',
                        'Falling back to static analysis of UI files',
                        'Repository contains web application files but needs setup',
                        'This is common for complex applications requiring specific environment setup'
                    ],
                    'recommendations': [
                        'Install Node.js and npm for JavaScript/React applications',
                        'Ensure the application can be run locally with npm start or similar',
                        'Add proper startup files (package.json with scripts, requirements.txt, etc.)',
                        'Verify all dependencies are installed and compatible',
                        'Consider adding a simple index.html for static hosting',
                        'Test the application locally before submission',
                        'Add setup instructions to README.md'
                    ],
                    'insights': [
                        'Static analysis only - no runtime UI evaluation',
                        'UI execution would provide more accurate scoring',
                        'This is common for complex applications that need setup',
                        'Consider providing a live demo URL or screenshots'
                    ]
                }
            
            # Analyze the captured UI
            ui_analysis = execution_result.get('analysis', {})
            screenshots = execution_result.get('screenshots', [])
            
            evidence = []
            recommendations = []
            insights = []
            score = 0
            
            # Visual quality analysis
            visual_quality = ui_analysis.get('visual_quality', 0)
            if visual_quality > 7:
                evidence.append(f"Excellent visual quality (score: {visual_quality:.1f})")
                score += 2
            elif visual_quality > 5:
                evidence.append(f"Good visual quality (score: {visual_quality:.1f})")
                score += 1
            else:
                evidence.append(f"Visual quality needs improvement (score: {visual_quality:.1f})")
                recommendations.append("Improve visual design and layout")
            
            # Accessibility analysis
            accessibility = ui_analysis.get('accessibility', 0)
            if accessibility > 7:
                evidence.append(f"Excellent accessibility (score: {accessibility:.1f})")
                score += 2
            elif accessibility > 5:
                evidence.append(f"Good accessibility (score: {accessibility:.1f})")
                score += 1
            else:
                evidence.append(f"Accessibility needs improvement (score: {accessibility:.1f})")
                recommendations.append("Enhance accessibility features (contrast, text size, keyboard navigation)")
            
            # Responsiveness analysis
            responsiveness = ui_analysis.get('responsiveness', 0)
            if responsiveness > 7:
                evidence.append(f"Excellent responsiveness (score: {responsiveness:.1f})")
                score += 2
            elif responsiveness > 5:
                evidence.append(f"Good responsiveness (score: {responsiveness:.1f})")
                score += 1
            else:
                evidence.append(f"Responsiveness needs improvement (score: {responsiveness:.1f})")
                recommendations.append("Improve responsive design for different screen sizes")
            
            # Interactivity analysis
            interactivity = ui_analysis.get('interactivity', 0)
            if interactivity > 7:
                evidence.append(f"Excellent interactivity (score: {interactivity:.1f})")
                score += 2
            elif interactivity > 5:
                evidence.append(f"Good interactivity (score: {interactivity:.1f})")
                score += 1
            else:
                evidence.append(f"Interactivity needs improvement (score: {interactivity:.1f})")
                recommendations.append("Add more interactive elements and user feedback")
            
            # Screenshot analysis
            if screenshots:
                evidence.append(f"Captured {len(screenshots)} UI screenshots")
                insights.append("Runtime UI analysis performed with visual feedback")
                score += 1
            
            # Issues and recommendations from UI analysis
            issues = ui_analysis.get('issues', [])
            if issues:
                evidence.append(f"Found {len(issues)} UI issues")
                recommendations.extend(issues[:3])  # Top 3 issues
            
            ui_recommendations = ui_analysis.get('recommendations', [])
            recommendations.extend(ui_recommendations[:3])  # Top 3 recommendations
            
            return {
                'score': min(score, 5),  # Cap at 5 points for UI execution
                'evidence': evidence,
                'recommendations': recommendations,
                'insights': insights
            }
            
        except Exception as e:
            return {
                'score': 0,
                'evidence': [f'UI execution analysis failed: {str(e)}'],
                'recommendations': ['Ensure application dependencies are installed'],
                'insights': ['Static analysis fallback used']
            }
    
    def _generate_ui_scoring_explanation(self, score: int, ui_analysis: Dict[str, Any], 
                                       a11y_analysis: Dict[str, Any], responsive_analysis: Dict[str, Any], 
                                       demo_analysis: Dict[str, Any], ui_execution_analysis: Optional[Dict[str, Any]]) -> str:
        """Generate human-friendly explanation of UI/UX scoring"""
        
        # Get individual scores
        ui_score = ui_analysis.get('score', 0)
        a11y_score = a11y_analysis.get('score', 0)
        responsive_score = responsive_analysis.get('score', 0)
        demo_score = demo_analysis.get('score', 0)
        execution_score = ui_execution_analysis.get('score', 0) if ui_execution_analysis else 0
        
        # Human-friendly assessment
        if score >= 8:
            assessment = "Excellent UI/UX with comprehensive implementation"
        elif score >= 6:
            assessment = "Good UI/UX with some areas for improvement"
        elif score >= 4:
            assessment = "Fair UI/UX with several areas needing attention"
        else:
            assessment = "Poor UI/UX requiring significant improvements"
        
        explanation = f"UI/UX Analysis: {score}/10 - {assessment}\n\n"
        
        # Why this score is high (if high)
        if score >= 8:
            explanation += "🎯 Why this score is high:\n"
            if ui_score >= 7:
                explanation += f"  • UI files: {ui_score}/10 - Good UI implementation\n"
            if a11y_score >= 7:
                explanation += f"  • Accessibility: {a11y_score}/10 - Strong accessibility features\n"
            if responsive_score >= 7:
                explanation += f"  • Responsive design: {responsive_score}/10 - Good responsive implementation\n"
            if demo_score >= 7:
                explanation += f"  • Demo/screenshots: {demo_score}/10 - Clear visual demonstration\n"
            if execution_score >= 7:
                explanation += f"  • UI execution: {execution_score}/10 - Successful UI execution\n"
        
        # Why some points were deducted (if low)
        if score < 8:
            explanation += "⚠️ Why some points were deducted:\n"
            if ui_score < 5:
                explanation += f"  • UI files: {ui_score}/10 - No UI files detected\n"
            if a11y_score < 5:
                explanation += f"  • Accessibility: {a11y_score}/10 - No accessibility features detected\n"
            if responsive_score < 5:
                explanation += f"  • Responsive design: {responsive_score}/10 - No responsive design detected\n"
            if demo_score < 5:
                explanation += f"  • Demo/screenshots: {demo_score}/10 - No visual demonstration provided\n"
            if execution_score < 5:
                explanation += f"  • UI execution: {execution_score}/10 - UI execution failed or not attempted\n"
        
        # Key Reasons for the Score
        explanation += f"\n📊 Key Reasons for the Score:\n"
        explanation += f"  • UI files: {ui_score}/10 - How many UI files are implemented\n"
        explanation += f"  • Accessibility: {a11y_score}/10 - How accessible the UI is\n"
        explanation += f"  • Responsive design: {responsive_score}/10 - How well it works on different devices\n"
        explanation += f"  • Demo/screenshots: {demo_score}/10 - How well the UI is demonstrated\n"
        explanation += f"  • UI execution: {execution_score}/10 - How well the UI actually runs\n"
        
        # In Simple Terms
        explanation += f"\n💡 In Simple Terms:\n"
        if score >= 8:
            explanation += "This project has excellent UI/UX! The interface is well-designed, accessible, works on all devices, and has a great demo. It's a pleasure to use."
        elif score >= 6:
            explanation += "This project has good UI/UX. The interface is decent, mostly accessible, and works well on different devices. It's user-friendly but could be improved."
        elif score >= 4:
            explanation += "This project has basic UI/UX. The interface is okay but lacks accessibility features and responsive design. It works but needs improvement."
        else:
            explanation += "This project has poor UI/UX. The interface is difficult to use, not accessible, and doesn't work well on different devices. It needs significant improvements."
        
        return explanation


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
        normalized_score = max(0, min(score, 10))
        
        # Generate detailed scoring explanation with normalized score
        scoring_explanation = self._generate_security_scoring_explanation(
            normalized_score, secrets_analysis, deps_analysis, validation_analysis, auth_analysis
        )
        insights.append(scoring_explanation)
        
        return AgentAnalysis(
            agent_name=self.name,
            score=normalized_score,
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
    
    def _generate_security_scoring_explanation(self, score: int, secrets_analysis: Dict[str, Any], 
                                             deps_analysis: Dict[str, Any], validation_analysis: Dict[str, Any], 
                                             auth_analysis: Dict[str, Any]) -> str:
        """Generate human-friendly explanation of security scoring"""
        
        # Get individual scores
        secrets_score = secrets_analysis.get('score', 0)
        deps_score = deps_analysis.get('score', 0)
        validation_score = validation_analysis.get('score', 0)
        auth_score = auth_analysis.get('score', 0)
        
        # Human-friendly assessment
        if score >= 8:
            assessment = "Excellent security implementation with comprehensive protection"
        elif score >= 6:
            assessment = "Good security with some areas for improvement"
        elif score >= 4:
            assessment = "Fair security with several vulnerabilities"
        else:
            assessment = "Poor security requiring immediate attention"
        
        explanation = f"Security Analysis: {score}/10 - {assessment}\n\n"
        
        # Why this score is high (if high)
        if score >= 8:
            explanation += "🎯 Why this score is high:\n"
            if secrets_score >= 7:
                explanation += f"  • Secrets management: {secrets_score}/10 - Secure secrets handling\n"
            if deps_score >= 7:
                explanation += f"  • Dependencies: {deps_score}/10 - Secure dependency management\n"
            if validation_score >= 7:
                explanation += f"  • Input validation: {validation_score}/10 - Strong input validation\n"
            if auth_score >= 7:
                explanation += f"  • Authentication: {auth_score}/10 - Robust authentication system\n"
        
        # Why some points were deducted (if low)
        if score < 8:
            explanation += "⚠️ Why some points were deducted:\n"
            if secrets_score < 5:
                explanation += f"  • Secrets management: {secrets_score}/10 - No secure secrets management detected\n"
            if deps_score < 5:
                explanation += f"  • Dependencies: {deps_score}/10 - Dependency security issues found\n"
            if validation_score < 5:
                explanation += f"  • Input validation: {validation_score}/10 - No input validation detected\n"
            if auth_score < 5:
                explanation += f"  • Authentication: {auth_score}/10 - No authentication system detected\n"
        
        # Key Reasons for the Score
        explanation += f"\n📊 Key Reasons for the Score:\n"
        explanation += f"  • Secrets management: {secrets_score}/10 - How securely secrets are handled\n"
        explanation += f"  • Dependencies: {deps_score}/10 - How secure the dependencies are\n"
        explanation += f"  • Input validation: {validation_score}/10 - How well inputs are validated\n"
        explanation += f"  • Authentication: {auth_score}/10 - How robust the authentication system is\n"
        
        # In Simple Terms
        explanation += f"\n💡 In Simple Terms:\n"
        if score >= 8:
            explanation += "This project has excellent security! It handles secrets properly, uses secure dependencies, validates inputs well, and has a robust authentication system. It's very secure."
        elif score >= 6:
            explanation += "This project has good security. It handles most security aspects well but has some areas that could be improved. It's reasonably secure."
        elif score >= 4:
            explanation += "This project has basic security. It has some security measures but also has vulnerabilities that need to be addressed. It's somewhat secure."
        else:
            explanation += "This project has poor security. It lacks proper security measures and has significant vulnerabilities. It needs immediate security improvements."
        
        return explanation


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
        
        # Normalize score to 0-10
        normalized_score = max(0, min(predicted_score, 10))
        
        # Generate detailed scoring explanation with normalized score
        scoring_explanation = self._generate_learning_scoring_explanation(normalized_score, features, pattern_insights)
        insights.append(scoring_explanation)
        
        return AgentAnalysis(
            agent_name=self.name,
            score=normalized_score,
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
    
    def _generate_learning_scoring_explanation(self, score: float, features: Dict[str, Any], pattern_insights: List[str]) -> str:
        """Generate human-friendly explanation of learning-based scoring"""
        
        # Human-friendly assessment
        if score >= 8:
            assessment = "Excellent project based on learned patterns"
        elif score >= 6:
            assessment = "Good project with some learned patterns"
        elif score >= 4:
            assessment = "Fair project with limited learned patterns"
        else:
            assessment = "Poor project based on learned patterns"
        
        explanation = f"Learning Analysis: {score:.1f}/10 - {assessment}\n\n"
        
        # Why this score is high (if high)
        if score >= 8:
            explanation += "🎯 Why this score is high:\n"
            if features.get('has_tests', False):
                explanation += f"  • Testing: Tests found - Good development practices\n"
            if features.get('has_docs', False):
                explanation += f"  • Documentation: Documentation found - Good project structure\n"
            if features.get('has_ci_cd', False):
                explanation += f"  • CI/CD: Automated workflows found - Professional development\n"
            if len(features.get('technology_stack', [])) >= 3:
                explanation += f"  • Tech stack: {len(features['technology_stack'])} technologies - Diverse technology usage\n"
        
        # Why some points were deducted (if low)
        if score < 8:
            explanation += "⚠️ Why some points were deducted:\n"
            if not features.get('has_tests', False):
                explanation += f"  • Testing: No tests found - Missing quality assurance\n"
            if not features.get('has_docs', False):
                explanation += f"  • Documentation: No documentation found - Poor project structure\n"
            if not features.get('has_ci_cd', False):
                explanation += f"  • CI/CD: No automated workflows - Missing professional practices\n"
            if len(features.get('technology_stack', [])) < 2:
                explanation += f"  • Tech stack: Limited technology diversity\n"
        
        # Key Reasons for the Score
        explanation += f"\n📊 Key Reasons for the Score:\n"
        explanation += f"  • File count: {features.get('file_count', 0)} files - Project size\n"
        explanation += f"  • Code files: {features.get('code_file_count', 0)} files - Codebase size\n"
        explanation += f"  • Testing: {'Yes' if features.get('has_tests', False) else 'No'} - Quality assurance\n"
        explanation += f"  • Documentation: {'Yes' if features.get('has_docs', False) else 'No'} - Project structure\n"
        explanation += f"  • CI/CD: {'Yes' if features.get('has_ci_cd', False) else 'No'} - Professional practices\n"
        explanation += f"  • Tech stack: {len(features.get('technology_stack', []))} technologies - Technology diversity\n"
        
        # In Simple Terms
        explanation += f"\n💡 In Simple Terms:\n"
        if score >= 8:
            explanation += "This project shows excellent patterns based on what we've learned from previous analyses. It has good structure, testing, documentation, and uses diverse technologies."
        elif score >= 6:
            explanation += "This project shows good patterns based on our learning. It has some good practices but could be improved in a few areas."
        elif score >= 4:
            explanation += "This project shows basic patterns but is missing some important practices we've learned are valuable."
        else:
            explanation += "This project doesn't follow the good patterns we've learned from previous analyses. It needs significant improvement."
        
        return explanation


# Specialized AI agents for advanced analysis

class BaseSpecializedAgent(ABC):
    """Base class for specialized AI agents"""
    
    def __init__(self, name: str, api_key: Optional[str] = None):
        self.name = name
        self.api_key = api_key
        self.analysis_history = []
    
    @abstractmethod
    def analyze(self, context: Dict[str, Any]) -> AgentAnalysis:
        """Each agent needs to implement this - the main analysis method"""
        pass
    
    def run_agent(self, model, inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent using the provided model and inputs"""
        # This would be implemented with actual LLM calls
        # For now, we'll simulate the analysis
        return self._simulate_analysis(inputs_dict)
    
    @abstractmethod
    def _simulate_analysis(self, inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate analysis for demonstration purposes"""
        pass


class InnovationCreativityAgent(BaseSpecializedAgent):
    """
    Innovation & Creativity Judge for hackathon projects.
    Evaluates how novel, inventive, and future-facing the solution is.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("InnovationCreativityAgent", api_key)
        self.prompt = """
You are the Innovation & Creativity Judge for a hackathon project.
Your job is to evaluate how novel, inventive, and future-facing the solution is.

## Inputs (you will receive them from the caller)
- project_summary: Short text describing the project in plain English (from README/pitch).
- problem_statement: The specific problem they claim to solve (from README).
- features_list: Bullet list of key features (from README or derived).
- tech_stack_detected: [strings] (e.g., React, Flask, OpenAI, LangChain, WebRTC).
- comparator_landscape: (optional) One-liners about similar tools/products if mentioned in README.
- constraints: (optional) Provided hackathon theme, required APIs, time limitations.
- demo_reference: (optional) Link or short description of demo.

## What to evaluate (be strict, no fluff)
1) Originality of the idea vs. typical CRUD demos
2) Creative use of technology (novel combinations, non-obvious patterns)
3) Problem-solution fit (is the solution clever for THIS problem?)
4) "Aha!" features (surprising, delightful, or emerging-tech showcases)
5) Future potential (extensibility, platformability)

## Scoring rubric (0–10)
- 0–2: Commonplace idea; minimal novelty.
- 3–4: Familiar idea with one fresh angle or minor twist.
- 5–6: Solid creativity; noticeable new combination or clever feature.
- 7–8: Strong originality; multiple inventive aspects or standout feature.
- 9–10: Breakthrough feel; rethinks the category or tech usage.

## Confidence (0.0–1.0)
Compute as: base = min(#evidence_points/6, 1.0).
If project_summary and features_list are both present, +0.1 (cap at 1.0).

## Output JSON (STRICT)
{
  "headline_verdict": "very short phrase (e.g., 'Fresh twist on X with Y')",
  "metric_scores": {
    "originality": 0-10,
    "creative_tech_use": 0-10,
    "problem_fit_creativity": 0-10,
    "aha_factor": 0-10,
    "future_potential": 0-10
  },
  "overall_numeric": 0-10,        // weighted mean you choose and state in rationale
  "evidence_points": [ "bullet, concrete, from inputs", "..."],
  "specific_opportunities": [ "short, surgical suggestions to raise originality", "..."],
  "risks_or_caveats": [ "risk framed as hypothesis, not fact", "..."],
  "confidence_estimate": 0.0-1.0,
  "calculation_notes": "one sentence on how you weighted sub-scores"
}

## Rules
- Use ONLY given inputs—no web lookups or imagined competitors.
- Be specific; no generic praise. Each evidence point must map to input text.
- Keep lists to 3–6 items max.
"""
    
    def analyze(self, context: Dict[str, Any]) -> AgentAnalysis:
        """Analyze innovation and creativity aspects"""
        # Extract inputs from context
        inputs_dict = self._extract_inputs(context)
        
        # Run the agent analysis
        result = self.run_agent(None, inputs_dict)
        
        # Normalize score to 0-10
        normalized_score = max(0, min(result.get('overall_numeric', 5), 10))
        
        # Create a modified result dict with normalized score for the explanation method
        result_normalized = result.copy()
        result_normalized['overall_numeric'] = normalized_score
        
        # Generate detailed scoring explanation with normalized score
        scoring_explanation = self._generate_innovation_scoring_explanation(result_normalized, inputs_dict)
        
        # Convert to AgentAnalysis format
        insights = [result.get('headline_verdict', ''), result.get('calculation_notes', '')]
        insights.append(scoring_explanation)
        
        return AgentAnalysis(
            agent_name=self.name,
            score=normalized_score,
            confidence=result.get('confidence_estimate', 0.5),
            evidence=result.get('evidence_points', []),
            recommendations=result.get('specific_opportunities', []),
            insights=insights,
            risks=result.get('risks_or_caveats', [])
        )
    
    def _extract_inputs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant inputs from context"""
        readme = context.get('readme', '')
        file_tree = context.get('file_tree', [])
        artifacts = context.get('artifacts', {})
        
        # Extract project summary from README
        project_summary = self._extract_project_summary(readme)
        
        # Extract problem statement
        problem_statement = self._extract_problem_statement(readme)
        
        # Extract features list
        features_list = self._extract_features_list(readme, file_tree)
        
        # Detect tech stack
        tech_stack_detected = self._detect_tech_stack(file_tree, readme)
        
        # Extract comparator landscape
        comparator_landscape = self._extract_comparator_landscape(readme)
        
        # Extract constraints
        constraints = self._extract_constraints(readme, artifacts)
        
        # Extract demo reference
        demo_reference = artifacts.get('screenshots_or_demo', '')
        
        return {
            'project_summary': project_summary,
            'problem_statement': problem_statement,
            'features_list': features_list,
            'tech_stack_detected': tech_stack_detected,
            'comparator_landscape': comparator_landscape,
            'constraints': constraints,
            'demo_reference': demo_reference
        }
    
    def _extract_project_summary(self, readme: str) -> str:
        """Extract project summary from README"""
        if not readme:
            return ""
        
        # Look for common patterns
        patterns = [
            r'##\s*About.*?\n(.*?)(?=\n##|\n#|\Z)',
            r'##\s*Project.*?\n(.*?)(?=\n##|\n#|\Z)',
            r'##\s*Description.*?\n(.*?)(?=\n##|\n#|\Z)',
            r'^#\s*.*?\n(.*?)(?=\n##|\n#|\Z)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, readme, re.DOTALL | re.IGNORECASE)
            if match:
                summary = match.group(1).strip()
                if len(summary) > 50:  # Ensure it's substantial
                    return summary[:500]  # Limit length
        
        # Fallback to first paragraph
        lines = readme.split('\n')
        for line in lines:
            if line.strip() and not line.startswith('#'):
                return line.strip()[:500]
        
        return ""
    
    def _extract_problem_statement(self, readme: str) -> str:
        """Extract problem statement from README"""
        if not readme:
            return ""
        
        patterns = [
            r'##\s*Problem.*?\n(.*?)(?=\n##|\n#|\Z)',
            r'##\s*Challenge.*?\n(.*?)(?=\n##|\n#|\Z)',
            r'##\s*Issue.*?\n(.*?)(?=\n##|\n#|\Z)',
            r'problem[s]?\s+is\s+(.*?)(?=\n|\.)',
            r'challenge[s]?\s+is\s+(.*?)(?=\n|\.)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, readme, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()[:300]
        
        return ""
    
    def _extract_features_list(self, readme: str, file_tree: List[Dict]) -> List[str]:
        """Extract features list from README and file structure"""
        features = []
        
        if readme:
            # Look for bullet points or numbered lists
            bullet_patterns = [
                r'[-*]\s+(.+?)(?=\n[-*]|\n\n|\Z)',
                r'\d+\.\s+(.+?)(?=\n\d+\.|\n\n|\Z)'
            ]
            
            for pattern in bullet_patterns:
                matches = re.findall(pattern, readme, re.MULTILINE)
                features.extend([match.strip() for match in matches[:10]])
        
        # Extract features from file structure
        feature_files = ['feature', 'component', 'module', 'service']
        for file_info in file_tree:
            if file_info.get('type') == 'blob':
                path = file_info.get('path', '').lower()
                if any(f in path for f in feature_files):
                    features.append(f"Feature: {file_info.get('path')}")
        
        return features[:10]  # Limit to 10 features
    
    def _detect_tech_stack(self, file_tree: List[Dict], readme: str) -> List[str]:
        """Detect technology stack from files and README"""
        tech_stack = []
        
        # File-based detection
        tech_mapping = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.jsx': 'React',
            '.tsx': 'React TypeScript',
            '.vue': 'Vue.js',
            '.java': 'Java',
            '.go': 'Go',
            '.rs': 'Rust',
            '.cpp': 'C++',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.html': 'HTML',
            '.css': 'CSS',
            '.scss': 'SCSS',
            '.sass': 'Sass'
        }
        
        for file_info in file_tree:
            if file_info.get('type') == 'blob':
                path = file_info.get('path', '')
                for ext, tech in tech_mapping.items():
                    if path.endswith(ext):
                        tech_stack.append(tech)
        
        # README-based detection
        if readme:
            readme_lower = readme.lower()
            frameworks = ['react', 'vue', 'angular', 'django', 'flask', 'express', 'spring', 'rails', 'laravel']
            for framework in frameworks:
                if framework in readme_lower:
                    tech_stack.append(framework.title())
        
        return list(set(tech_stack))  # Remove duplicates
    
    def _extract_comparator_landscape(self, readme: str) -> List[str]:
        """Extract competitor/comparison mentions from README"""
        if not readme:
            return []
        
        # Look for comparison patterns
        patterns = [
            r'vs\.?\s+(.+?)(?=\n|\.)',
            r'compared\s+to\s+(.+?)(?=\n|\.)',
            r'unlike\s+(.+?)(?=\n|\.)',
            r'similar\s+to\s+(.+?)(?=\n|\.)'
        ]
        
        comparisons = []
        for pattern in patterns:
            matches = re.findall(pattern, readme, re.IGNORECASE)
            comparisons.extend(matches)
        
        return comparisons[:5]  # Limit to 5 comparisons
    
    def _extract_constraints(self, readme: str, artifacts: Dict) -> List[str]:
        """Extract constraints from README and artifacts"""
        constraints = []
        
        if readme:
            # Look for constraint patterns
            constraint_patterns = [
                r'constraint[s]?\s*:?\s*(.+?)(?=\n|\.)',
                r'limitation[s]?\s*:?\s*(.+?)(?=\n|\.)',
                r'requirement[s]?\s*:?\s*(.+?)(?=\n|\.)',
                r'time\s+limit[s]?\s*:?\s*(.+?)(?=\n|\.)'
            ]
            
            for pattern in constraint_patterns:
                matches = re.findall(pattern, readme, re.IGNORECASE)
                constraints.extend(matches)
        
        return constraints[:5]
    
    def _simulate_analysis(self, inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate analysis for demonstration purposes"""
        project_summary = inputs_dict.get('project_summary', '')
        features_list = inputs_dict.get('features_list', [])
        tech_stack = inputs_dict.get('tech_stack_detected', [])
        
        # Calculate evidence points
        evidence_points = []
        if project_summary:
            evidence_points.append(f"Project summary provided: {project_summary[:100]}...")
        if features_list:
            evidence_points.append(f"Features identified: {len(features_list)} items")
        if tech_stack:
            evidence_points.append(f"Tech stack: {', '.join(tech_stack)}")
        
        # Calculate scores based on inputs
        originality = min(7, len(tech_stack) + len(features_list))
        creative_tech_use = min(8, len(tech_stack) * 2)
        problem_fit = 6 if project_summary else 3
        aha_factor = min(7, len(features_list))
        future_potential = min(6, len(tech_stack))
        
        overall = int((originality + creative_tech_use + problem_fit + aha_factor + future_potential) / 5)
        
        # Calculate confidence
        evidence_count = len(evidence_points)
        confidence = min(evidence_count / 6.0, 1.0)
        if project_summary and features_list:
            confidence = min(confidence + 0.1, 1.0)
        
        # Generate detailed recommendations based on analysis
        opportunities = []
        if originality < 6:
            opportunities.extend([
                "Explore novel approaches to the problem domain",
                "Consider unconventional technology combinations",
                "Research emerging technologies that could differentiate your solution"
            ])
        
        if creative_tech_use < 6:
            opportunities.extend([
                "Experiment with cutting-edge frameworks and libraries",
                "Integrate AI/ML capabilities for enhanced functionality",
                "Consider blockchain, IoT, or AR/VR technologies"
            ])
        
        if problem_fit < 6:
            opportunities.extend([
                "Conduct deeper user research to understand pain points",
                "Validate problem-solution fit with target users",
                "Refine the value proposition to be more compelling"
            ])
        
        if aha_factor < 6:
            opportunities.extend([
                "Add surprising or delightful features that wow users",
                "Implement gamification or interactive elements",
                "Create memorable user experiences"
            ])
        
        if future_potential < 6:
            opportunities.extend([
                "Design for scalability and extensibility",
                "Plan for platform expansion and API development",
                "Consider monetization and business model innovation"
            ])
        
        # Generate risks based on analysis
        risks = []
        if overall < 5:
            risks.extend([
                "Solution may lack differentiation from existing alternatives",
                "Limited innovation could impact competitive advantage",
                "May struggle to attract users or investors"
            ])
        
        if not project_summary:
            risks.append("Lack of clear project description makes evaluation difficult")
        
        if not features_list:
            risks.append("No clear feature set defined - consider documenting key features")
        
        return {
            "headline_verdict": f"Creative use of {', '.join(tech_stack[:2]) if tech_stack else 'technology'} with {overall}/10 innovation score",
            "metric_scores": {
                "originality": originality,
                "creative_tech_use": creative_tech_use,
                "problem_fit_creativity": problem_fit,
                "aha_factor": aha_factor,
                "future_potential": future_potential
            },
            "overall_numeric": overall,
            "evidence_points": evidence_points,
            "specific_opportunities": opportunities[:6],  # Limit to 6 recommendations
            "risks_or_caveats": risks[:4],  # Limit to 4 risks
            "confidence_estimate": confidence,
            "calculation_notes": f"Weighted average with emphasis on originality ({originality}) and creative tech use ({creative_tech_use})"
        }
    
    def _generate_innovation_scoring_explanation(self, result: Dict[str, Any], inputs_dict: Dict[str, Any]) -> str:
        """Generate human-friendly explanation of innovation scoring"""
        
        # Get metric scores
        metric_scores = result.get('metric_scores', {})
        originality = metric_scores.get('originality', 0)
        creative_tech_use = metric_scores.get('creative_tech_use', 0)
        problem_fit = metric_scores.get('problem_fit_creativity', 0)
        aha_factor = metric_scores.get('aha_factor', 0)
        future_potential = metric_scores.get('future_potential', 0)
        overall_score = result.get('overall_numeric', 0)
        
        # Human-friendly assessment
        if overall_score >= 8:
            assessment = "Excellent innovation with breakthrough potential"
        elif overall_score >= 6:
            assessment = "Good innovation with creative elements"
        elif overall_score >= 4:
            assessment = "Fair innovation with some creative aspects"
        else:
            assessment = "Poor innovation requiring more creativity"
        
        explanation = f"Innovation & Creativity: {overall_score}/10 - {assessment}\n\n"
        
        # Why this score is high (if high)
        if overall_score >= 8:
            explanation += "🎯 Why this score is high:\n"
            if originality >= 7:
                explanation += f"  • Originality: {originality}/10 - Unique and breakthrough idea\n"
            if creative_tech_use >= 7:
                explanation += f"  • Creative tech use: {creative_tech_use}/10 - Innovative technology combinations\n"
            if problem_fit >= 7:
                explanation += f"  • Problem fit: {problem_fit}/10 - Excellent solution for the problem\n"
            if aha_factor >= 7:
                explanation += f"  • Aha factor: {aha_factor}/10 - Surprising and delightful features\n"
            if future_potential >= 7:
                explanation += f"  • Future potential: {future_potential}/10 - High scalability potential\n"
        
        # Why some points were deducted (if low)
        if overall_score < 8:
            explanation += "⚠️ Why some points were deducted:\n"
            if originality < 8:
                if originality < 6:
                    explanation += f"  • Originality: {originality}/10 - Limited novelty, commonplace idea\n"
                else:
                    explanation += f"  • Originality: {originality}/10 - Good but not exceptional innovation\n"
            if creative_tech_use < 8:
                if creative_tech_use < 6:
                    explanation += f"  • Creative tech use: {creative_tech_use}/10 - Basic or outdated technologies\n"
                else:
                    explanation += f"  • Creative tech use: {creative_tech_use}/10 - Solid but not groundbreaking tech use\n"
            if problem_fit < 8:
                if problem_fit < 6:
                    explanation += f"  • Problem fit: {problem_fit}/10 - Weak problem-solution alignment\n"
                else:
                    explanation += f"  • Problem fit: {problem_fit}/10 - Good solution but not perfect fit\n"
            if aha_factor < 8:
                if aha_factor < 6:
                    explanation += f"  • Aha factor: {aha_factor}/10 - No standout features\n"
                else:
                    explanation += f"  • Aha factor: {aha_factor}/10 - Some interesting features but not surprising\n"
            if future_potential < 8:
                if future_potential < 6:
                    explanation += f"  • Future potential: {future_potential}/10 - No clear future potential\n"
                else:
                    explanation += f"  • Future potential: {future_potential}/10 - Decent potential but not exceptional\n"
        
        # Key Reasons for the Score
        explanation += f"\n📊 Key Reasons for the Score:\n"
        explanation += f"  • Originality: {originality}/10 - How unique and novel the idea is\n"
        explanation += f"  • Creative tech use: {creative_tech_use}/10 - Innovative use of technology\n"
        explanation += f"  • Problem fit: {problem_fit}/10 - How well it solves the problem\n"
        explanation += f"  • Aha factor: {aha_factor}/10 - Surprising and delightful elements\n"
        explanation += f"  • Future potential: {future_potential}/10 - Scalability and extensibility\n"
        
        # In Simple Terms
        explanation += f"\n💡 In Simple Terms:\n"
        if overall_score >= 8:
            explanation += "This is a truly innovative project with breakthrough potential. It's original, uses technology creatively, and has great future potential."
        elif overall_score >= 6:
            explanation += "This project shows good innovation with creative elements. It has some unique aspects and good potential for growth."
        elif overall_score >= 4:
            explanation += "This project has some creative aspects but could be more innovative. It's a decent solution but not groundbreaking."
        else:
            explanation += "This project needs more innovation and creativity. It's too similar to existing solutions and lacks unique value."
        
        return explanation


class FunctionalityCompletenessAgent(BaseSpecializedAgent):
    """
    Functionality & Completeness Judge.
    Assesses whether the project works end-to-end and how complete the core flows are.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("FunctionalityCompletenessAgent", api_key)
        self.prompt = """
You are the Functionality & Completeness Judge.
Assess whether the project works end-to-end and how complete the core flows are.

## Inputs
- features_list: Declared features.
- user_flows: ["register", "login", "upload", "analyze", "report", ...] if available.
- demo_reference: URL or description of the demo/screens or CLI transcript.
- api_endpoints: (optional) List of routes with brief descriptions.
- test_summary: (optional) e.g., "12 passed, 3 skipped, coverage 68%".
- known_limitations: (optional) What the team says is unfinished.
- env_and_deploy: (optional) notes about Docker/CI/CD/hosting that affect demoability.

## Evaluate
1) Core feature implementation (are key flows actually implemented?)
2) Breadth vs. depth (surface area vs. fully working subset)
3) Error handling/user feedback (visible indications of failure/success)
4) Demo reliability (link works, multiple steps shown)
5) Test/done-ness signals (tests, coverage, CI passing)

## Scoring rubric (0–10)
- 0–2: Mostly concept; little is working.
- 3–4: Demo runs but brittle; minimal flows complete.
- 5–6: Core use case works; a few rough edges.
- 7–8: Multiple flows complete; resilient enough for a live demo.
- 9–10: Polished, reliable; covers edge cases; strong tests.

## Confidence (0.0–1.0)
base = 0.2
+0.2 if demo_reference provided
+0.2 if api_endpoints provided
+0.2 if test_summary provided
+0.2 if user_flows provided
Cap at 1.0.

## Output JSON (STRICT)
{
  "working_claim": "one-liner stating what concretely works",
  "flow_coverage_table": [
    {"flow":"<name>","implemented":true/false,"notes":"short"},
    {"flow":"...", "implemented":...}
  ],
  "quality_signals": ["tests/coverage/CI or demo reliability notes"],
  "gaps_or_edge_cases": ["clear, checkable gaps"],
  "overall_numeric": 0-10,
  "confidence_estimate": 0.0-1.0
}

## Rules
- Only use provided inputs; do not assume hidden functionality.
- If inputs are missing, state that in gaps_or_edge_cases.
- Keep tables to max 8 rows.
"""
    
    def analyze(self, context: Dict[str, Any]) -> AgentAnalysis:
        """Analyze functionality and completeness"""
        inputs_dict = self._extract_inputs(context)
        result = self.run_agent(None, inputs_dict)
        
        # Normalize score to 0-10
        normalized_score = max(0, min(result.get('overall_numeric', 5), 10))
        
        # Create a modified result dict with normalized score for the explanation method
        result_normalized = result.copy()
        result_normalized['overall_numeric'] = normalized_score
        
        # Generate detailed scoring explanation with normalized score
        scoring_explanation = self._generate_functionality_scoring_explanation(result_normalized, inputs_dict)
        
        # Convert flow coverage table to evidence
        evidence = result.get('quality_signals', [])
        flow_table = result.get('flow_coverage_table', [])
        for flow in flow_table:
            status = "✓" if flow.get('implemented', False) else "✗"
            evidence.append(f"{status} {flow.get('flow', 'Unknown')}: {flow.get('notes', '')}")
        
        insights = [result.get('working_claim', '')]
        insights.append(scoring_explanation)
        
        return AgentAnalysis(
            agent_name=self.name,
            score=normalized_score,
            confidence=result.get('confidence_estimate', 0.5),
            evidence=evidence,
            recommendations=result.get('gaps_or_edge_cases', []),
            insights=insights,
            risks=result.get('gaps_or_edge_cases', [])
        )
    
    def _extract_inputs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant inputs from context"""
        readme = context.get('readme', '')
        file_tree = context.get('file_tree', [])
        artifacts = context.get('artifacts', {})
        
        # Extract features list
        features_list = self._extract_features_list(readme, file_tree)
        
        # Extract user flows
        user_flows = self._extract_user_flows(readme, file_tree)
        
        # Extract demo reference
        demo_reference = artifacts.get('screenshots_or_demo', '')
        
        # Extract API endpoints
        api_endpoints = self._extract_api_endpoints(file_tree, readme)
        
        # Extract test summary
        test_summary = artifacts.get('test_results', '')
        
        # Extract known limitations
        known_limitations = self._extract_limitations(readme)
        
        # Extract deployment info
        env_and_deploy = self._extract_deployment_info(file_tree, readme)
        
        return {
            'features_list': features_list,
            'user_flows': user_flows,
            'demo_reference': demo_reference,
            'api_endpoints': api_endpoints,
            'test_summary': test_summary,
            'known_limitations': known_limitations,
            'env_and_deploy': env_and_deploy
        }
    
    def _extract_features_list(self, readme: str, file_tree: List[Dict]) -> List[str]:
        """Extract features list from README and file structure"""
        features = []
        
        if readme:
            # Look for bullet points or numbered lists
            bullet_patterns = [
                r'[-*]\s+(.+?)(?=\n[-*]|\n\n|\Z)',
                r'\d+\.\s+(.+?)(?=\n\d+\.|\n\n|\Z)'
            ]
            
            for pattern in bullet_patterns:
                matches = re.findall(pattern, readme, re.MULTILINE)
                features.extend([match.strip() for match in matches[:10]])
        
        return features[:10]
    
    def _extract_user_flows(self, readme: str, file_tree: List[Dict]) -> List[str]:
        """Extract user flows from README and file structure"""
        flows = []
        
        if readme:
            # Look for flow patterns
            flow_patterns = [
                r'user\s+can\s+(.+?)(?=\n|\.)',
                r'flow[s]?\s*:?\s*(.+?)(?=\n|\.)',
                r'step[s]?\s*:?\s*(.+?)(?=\n|\.)'
            ]
            
            for pattern in flow_patterns:
                matches = re.findall(pattern, readme, re.IGNORECASE)
                flows.extend([match.strip() for match in matches[:10]])
        
        # Extract from file structure
        flow_files = ['route', 'controller', 'handler', 'api']
        for file_info in file_tree:
            if file_info.get('type') == 'blob':
                path = file_info.get('path', '').lower()
                if any(f in path for f in flow_files):
                    flows.append(f"API endpoint: {file_info.get('path')}")
        
        return flows[:10]
    
    def _extract_api_endpoints(self, file_tree: List[Dict], readme: str) -> List[str]:
        """Extract API endpoints from file structure and README"""
        endpoints = []
        
        # Look for route files
        for file_info in file_tree:
            if file_info.get('type') == 'blob':
                path = file_info.get('path', '')
                if any(keyword in path.lower() for keyword in ['route', 'api', 'endpoint', 'controller']):
                    endpoints.append(f"Route file: {path}")
        
        # Look for API documentation in README
        if readme:
            api_patterns = [
                r'GET\s+/(.+?)(?=\n|$)',
                r'POST\s+/(.+?)(?=\n|$)',
                r'PUT\s+/(.+?)(?=\n|$)',
                r'DELETE\s+/(.+?)(?=\n|$)'
            ]
            
            for pattern in api_patterns:
                matches = re.findall(pattern, readme, re.MULTILINE)
                endpoints.extend([f"API: {match.strip()}" for match in matches[:5]])
        
        return endpoints[:10]
    
    def _extract_limitations(self, readme: str) -> List[str]:
        """Extract known limitations from README"""
        if not readme:
            return []
        
        limitation_patterns = [
            r'limitation[s]?\s*:?\s*(.+?)(?=\n|\.)',
            r'known\s+issue[s]?\s*:?\s*(.+?)(?=\n|\.)',
            r'todo[s]?\s*:?\s*(.+?)(?=\n|\.)',
            r'not\s+implemented\s*:?\s*(.+?)(?=\n|\.)'
        ]
        
        limitations = []
        for pattern in limitation_patterns:
            matches = re.findall(pattern, readme, re.IGNORECASE)
            limitations.extend([match.strip() for match in matches[:5]])
        
        return limitations
    
    def _extract_deployment_info(self, file_tree: List[Dict], readme: str) -> List[str]:
        """Extract deployment information"""
        deploy_info = []
        
        # Look for deployment files
        deploy_files = ['dockerfile', 'docker-compose', 'deploy', 'ci', 'github']
        for file_info in file_tree:
            if file_info.get('type') == 'blob':
                path = file_info.get('path', '').lower()
                if any(f in path for f in deploy_files):
                    deploy_info.append(f"Deployment file: {file_info.get('path')}")
        
        return deploy_info
    
    def _simulate_analysis(self, inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate analysis for demonstration purposes"""
        features_list = inputs_dict.get('features_list', [])
        user_flows = inputs_dict.get('user_flows', [])
        demo_reference = inputs_dict.get('demo_reference', '')
        api_endpoints = inputs_dict.get('api_endpoints', [])
        test_summary = inputs_dict.get('test_summary', '')
        
        # Calculate confidence
        confidence = 0.2
        if demo_reference:
            confidence += 0.2
        if api_endpoints:
            confidence += 0.2
        if test_summary:
            confidence += 0.2
        if user_flows:
            confidence += 0.2
        
        # Create flow coverage table
        flow_coverage_table = []
        for i, feature in enumerate(features_list[:8]):  # Max 8 rows
            implemented = i < len(features_list) * 0.7  # Simulate 70% implementation
            flow_coverage_table.append({
                "flow": feature[:50],  # Truncate long names
                "implemented": implemented,
                "notes": "Working" if implemented else "Incomplete"
            })
        
        # Calculate quality signals
        quality_signals = []
        if test_summary:
            quality_signals.append(f"Test results: {test_summary}")
        if demo_reference:
            quality_signals.append("Demo available")
        if api_endpoints:
            quality_signals.append(f"API endpoints: {len(api_endpoints)} found")
        
        # Calculate gaps
        gaps = []
        if not test_summary:
            gaps.append("No test results provided")
        if not demo_reference:
            gaps.append("No demo reference provided")
        if not api_endpoints:
            gaps.append("No API endpoints identified")
        
        # Calculate overall score
        implemented_flows = sum(1 for flow in flow_coverage_table if flow['implemented'])
        total_flows = len(flow_coverage_table)
        implementation_ratio = implemented_flows / total_flows if total_flows > 0 else 0
        
        base_score = implementation_ratio * 8  # 0-8 based on implementation
        if test_summary:
            base_score += 1
        if demo_reference:
            base_score += 1
        
        overall_score = min(int(base_score), 10)
        
        return {
            "working_claim": f"Core functionality implemented with {implemented_flows}/{total_flows} flows working",
            "flow_coverage_table": flow_coverage_table,
            "quality_signals": quality_signals,
            "gaps_or_edge_cases": gaps,
            "overall_numeric": overall_score,
            "confidence_estimate": min(confidence, 1.0)
        }
    
    def _generate_functionality_scoring_explanation(self, result: Dict[str, Any], inputs_dict: Dict[str, Any]) -> str:
        """Generate human-friendly explanation of functionality scoring"""
        
        # Get flow coverage analysis
        flow_table = result.get('flow_coverage_table', [])
        implemented_flows = sum(1 for flow in flow_table if flow.get('implemented', False))
        total_flows = len(flow_table)
        coverage_percentage = (implemented_flows / total_flows) * 100 if total_flows > 0 else 0
        
        # Get other metrics
        quality_signals = result.get('quality_signals', [])
        demo_reference = inputs_dict.get('demo_reference', '')
        test_summary = inputs_dict.get('test_summary', '')
        api_endpoints = inputs_dict.get('api_endpoints', [])
        overall_score = result.get('overall_numeric', 0)
        
        # Human-friendly assessment
        if overall_score >= 8:
            assessment = "Excellent functionality with comprehensive implementation"
        elif overall_score >= 6:
            assessment = "Good functionality with some areas for improvement"
        elif overall_score >= 4:
            assessment = "Fair functionality with several gaps"
        else:
            assessment = "Poor functionality requiring significant work"
        
        explanation = f"Functionality & Completeness: {overall_score}/10 - {assessment}\n\n"
        
        # Why this score is high (if high)
        if overall_score >= 8:
            explanation += "🎯 Why this score is high:\n"
            if coverage_percentage >= 80:
                explanation += f"  • Flow coverage: {implemented_flows}/{total_flows} flows ({coverage_percentage:.0f}%) - Excellent coverage of user flows\n"
            if len(quality_signals) >= 5:
                explanation += f"  • Quality signals: {len(quality_signals)} positive indicators - Strong evidence of quality\n"
            if demo_reference:
                explanation += f"  • Demo reference: Demo/screenshots provided - Clear demonstration of functionality\n"
            if len(api_endpoints) >= 3:
                explanation += f"  • API endpoints: {len(api_endpoints)} endpoints - Good API coverage\n"
        
        # Why some points were deducted (if low or medium)
        if overall_score < 8:
            explanation += "⚠️ Why some points were deducted:\n"
            if coverage_percentage < 60:
                explanation += f"  • Flow coverage: {implemented_flows}/{total_flows} flows ({coverage_percentage:.0f}%) - Poor coverage of user flows\n"
            elif coverage_percentage < 80:
                explanation += f"  • Flow coverage: {implemented_flows}/{total_flows} flows ({coverage_percentage:.0f}%) - Some user flows incomplete\n"
            if len(quality_signals) < 3:
                explanation += f"  • Quality signals: {len(quality_signals)} positive indicators - Limited quality evidence\n"
            if not demo_reference:
                explanation += f"  • Demo reference: No demo provided - Cannot verify functionality\n"
            if len(api_endpoints) < 2:
                explanation += f"  • API endpoints: {len(api_endpoints)} endpoints - Limited API coverage\n"
            elif len(api_endpoints) < 5:
                explanation += f"  • API endpoints: {len(api_endpoints)} endpoints - Could benefit from more API coverage\n"
        
        # Key Reasons for the Score
        explanation += f"\n📊 Key Reasons for the Score:\n"
        explanation += f"  • Flow coverage: {implemented_flows}/{total_flows} flows ({coverage_percentage:.0f}%) - How many user flows are actually working\n"
        explanation += f"  • Quality signals: {len(quality_signals)} indicators - Evidence of good implementation quality\n"
        explanation += f"  • Demo reference: {'Yes' if demo_reference else 'No'} - Whether there's a working demo to verify functionality\n"
        explanation += f"  • API endpoints: {len(api_endpoints)} endpoints - How many API endpoints are available\n"
        if test_summary and 'passed' in test_summary.lower():
            explanation += f"  • Test coverage: Tests found - Bonus points for having tests (not required)\n"
        
        # In Simple Terms
        explanation += f"\n💡 In Simple Terms:\n"
        if overall_score >= 8:
            explanation += "This project works really well! All the main user flows are implemented, there's good evidence of quality, and there's a working demo. It's a complete, functional solution."
        elif overall_score >= 6:
            explanation += "This project works well for the most part. Most user flows are implemented and working, though there might be some gaps or missing pieces."
        elif overall_score >= 4:
            explanation += "This project has some working functionality but is incomplete. Some user flows work while others are missing or broken."
        else:
            explanation += "This project needs a lot more work to be functional. Most user flows are missing, broken, or not working properly."
        
        return explanation


class TechnicalComplexityAgent(BaseSpecializedAgent):
    """
    Technical Complexity Judge.
    Estimates how technically demanding the solution is and how well trade-offs were handled.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("TechnicalComplexityAgent", api_key)
        self.prompt = """
You are the Technical Complexity Judge.
Estimate how technically demanding the solution is and how well trade-offs were handled.

## Inputs
- tech_stack_detected: [frameworks, languages, infra].
- architecture_notes: (optional) summarized from README (services, queues, DB, caching).
- data_flows: (optional) descriptions of pipelines, models, or multi-service orchestration.
- constraints: (optional) time limits, required APIs.
- performance_scalability_signals: (optional) caching, pagination, async jobs, batching, indexing, queues.
- security_privacy_signals: (optional) authN/Z, token flows, secrets mgmt.

## Evaluate
1) Architectural difficulty (e.g., microservices, event-driven, streaming)
2) Integration difficulty (multiple APIs, SDKs, auth flows)
3) Data & ML complexity (training/inference pipelines, vector DBs, eval loops)
4) Performance & scalability strategies (caching, batching, queuing, shards)
5) Security/privacy complexity (roles, JWT/OAuth, secret rotation)

## Scoring rubric (0–10)
- 0–2: Simple single-tier stack, limited integrations.
- 3–4: Basic client+API, a couple external services.
- 5–6: Solid multi-component design with some advanced pieces.
- 7–8: High complexity (orchestration, queues, ML pipelines, non-trivial infra).
- 9–10: Expert-level architecture and integrations with clear trade-off reasoning.

## Confidence (0.0–1.0)
Start at min(#distinct_signals/6, 1.0) where distinct_signals are the unique items across:
tech_stack_detected + performance_scalability_signals + security_privacy_signals + non-empty data_flows/architecture_notes (count as 1 each).

## Output JSON (STRICT)
{
  "complexity_drivers": ["bullet list of the top 3–6 technical elements creating complexity"],
  "tradeoff_rationale": "one sentence on main trade-offs acknowledged or implied",
  "architecture_style": "one of: single-tier | client-server | microservices | event-driven | data-pipeline | hybrid",
  "integration_surface_area": 0-10,    // count-style score you define in rationale
  "overall_numeric": 0-10,
  "confidence_estimate": 0.0-1.0
}

## Rules
- Prefer concise, high-signal bullets.
- No invented infra—only what inputs justify.
- If signals are weak, lower both complexity and confidence and say why.
"""
    
    def analyze(self, context: Dict[str, Any]) -> AgentAnalysis:
        """Analyze technical complexity"""
        inputs_dict = self._extract_inputs(context)
        result = self.run_agent(None, inputs_dict)
        
        # Normalize score to 0-10
        normalized_score = max(0, min(result.get('overall_numeric', 5), 10))
        
        # Create a modified result dict with normalized score for the explanation method
        result_normalized = result.copy()
        result_normalized['overall_numeric'] = normalized_score
        
        # Generate detailed scoring explanation with normalized score
        scoring_explanation = self._generate_technical_scoring_explanation(result_normalized, inputs_dict)
        
        # Convert to evidence format
        evidence = result.get('complexity_drivers', [])
        evidence.append(f"Architecture: {result.get('architecture_style', 'Unknown')}")
        evidence.append(f"Integration surface: {result.get('integration_surface_area', 0)}/10")
        
        insights = [result.get('tradeoff_rationale', '')]
        insights.append(scoring_explanation)
        
        return AgentAnalysis(
            agent_name=self.name,
            score=normalized_score,
            confidence=result.get('confidence_estimate', 0.5),
            evidence=evidence,
            recommendations=[],  # No specific recommendations in this agent
            insights=insights,
            risks=[]  # No specific risks in this agent
        )
    
    def _extract_inputs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant inputs from context"""
        readme = context.get('readme', '')
        file_tree = context.get('file_tree', [])
        artifacts = context.get('artifacts', {})
        
        # Detect tech stack
        tech_stack_detected = self._detect_tech_stack(file_tree, readme)
        
        # Extract architecture notes
        architecture_notes = self._extract_architecture_notes(readme)
        
        # Extract data flows
        data_flows = self._extract_data_flows(readme, file_tree)
        
        # Extract constraints
        constraints = self._extract_constraints(readme, artifacts)
        
        # Extract performance/scalability signals
        performance_signals = self._extract_performance_signals(file_tree, readme)
        
        # Extract security/privacy signals
        security_signals = self._extract_security_signals(file_tree, readme)
        
        return {
            'tech_stack_detected': tech_stack_detected,
            'architecture_notes': architecture_notes,
            'data_flows': data_flows,
            'constraints': constraints,
            'performance_scalability_signals': performance_signals,
            'security_privacy_signals': security_signals
        }
    
    def _detect_tech_stack(self, file_tree: List[Dict], readme: str) -> List[str]:
        """Detect technology stack from files and README"""
        tech_stack = []
        
        # File-based detection
        tech_mapping = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.jsx': 'React',
            '.tsx': 'React TypeScript',
            '.vue': 'Vue.js',
            '.java': 'Java',
            '.go': 'Go',
            '.rs': 'Rust',
            '.cpp': 'C++',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.html': 'HTML',
            '.css': 'CSS',
            '.scss': 'SCSS',
            '.sass': 'Sass',
            '.sql': 'SQL',
            '.json': 'JSON',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.toml': 'TOML',
            '.xml': 'XML'
        }
        
        for file_info in file_tree:
            if file_info.get('type') == 'blob':
                path = file_info.get('path', '')
                for ext, tech in tech_mapping.items():
                    if path.endswith(ext):
                        tech_stack.append(tech)
        
        # README-based detection
        if readme:
            readme_lower = readme.lower()
            frameworks = ['react', 'vue', 'angular', 'django', 'flask', 'express', 'spring', 'rails', 'laravel', 'fastapi', 'nextjs', 'nuxt']
            databases = ['postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'cassandra']
            cloud = ['aws', 'azure', 'gcp', 'heroku', 'vercel', 'netlify']
            
            for framework in frameworks:
                if framework in readme_lower:
                    tech_stack.append(framework.title())
            
            for db in databases:
                if db in readme_lower:
                    tech_stack.append(db.title())
            
            for cloud_provider in cloud:
                if cloud_provider in readme_lower:
                    tech_stack.append(cloud_provider.upper())
        
        return list(set(tech_stack))  # Remove duplicates
    
    def _extract_architecture_notes(self, readme: str) -> str:
        """Extract architecture notes from README"""
        if not readme:
            return ""
        
        patterns = [
            r'##\s*Architecture.*?\n(.*?)(?=\n##|\n#|\Z)',
            r'##\s*Design.*?\n(.*?)(?=\n##|\n#|\Z)',
            r'##\s*Structure.*?\n(.*?)(?=\n##|\n#|\Z)',
            r'##\s*System.*?\n(.*?)(?=\n##|\n#|\Z)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, readme, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()[:500]
        
        return ""
    
    def _extract_data_flows(self, readme: str, file_tree: List[Dict]) -> List[str]:
        """Extract data flow descriptions"""
        flows = []
        
        if readme:
            # Look for data flow patterns
            flow_patterns = [
                r'data\s+flow[s]?\s*:?\s*(.+?)(?=\n|\.)',
                r'pipeline[s]?\s*:?\s*(.+?)(?=\n|\.)',
                r'processing\s*:?\s*(.+?)(?=\n|\.)'
            ]
            
            for pattern in flow_patterns:
                matches = re.findall(pattern, readme, re.IGNORECASE)
                flows.extend([match.strip() for match in matches[:5]])
        
        # Look for data processing files
        data_files = ['pipeline', 'processor', 'etl', 'transform', 'model']
        for file_info in file_tree:
            if file_info.get('type') == 'blob':
                path = file_info.get('path', '').lower()
                if any(f in path for f in data_files):
                    flows.append(f"Data processing: {file_info.get('path')}")
        
        return flows[:5]
    
    def _extract_constraints(self, readme: str, artifacts: Dict) -> List[str]:
        """Extract constraints from README and artifacts"""
        constraints = []
        
        if readme:
            constraint_patterns = [
                r'constraint[s]?\s*:?\s*(.+?)(?=\n|\.)',
                r'limitation[s]?\s*:?\s*(.+?)(?=\n|\.)',
                r'requirement[s]?\s*:?\s*(.+?)(?=\n|\.)',
                r'time\s+limit[s]?\s*:?\s*(.+?)(?=\n|\.)'
            ]
            
            for pattern in constraint_patterns:
                matches = re.findall(pattern, readme, re.IGNORECASE)
                constraints.extend([match.strip() for match in matches[:5]])
        
        return constraints
    
    def _extract_performance_signals(self, file_tree: List[Dict], readme: str) -> List[str]:
        """Extract performance and scalability signals"""
        signals = []
        
        # Look for performance-related files
        perf_files = ['cache', 'redis', 'queue', 'worker', 'async', 'batch', 'index']
        for file_info in file_tree:
            if file_info.get('type') == 'blob':
                path = file_info.get('path', '').lower()
                if any(f in path for f in perf_files):
                    signals.append(f"Performance file: {file_info.get('path')}")
        
        # Look for performance mentions in README
        if readme:
            perf_patterns = [
                r'cache[s]?\s*:?\s*(.+?)(?=\n|\.)',
                r'queue[s]?\s*:?\s*(.+?)(?=\n|\.)',
                r'async[s]?\s*:?\s*(.+?)(?=\n|\.)',
                r'scalability\s*:?\s*(.+?)(?=\n|\.)'
            ]
            
            for pattern in perf_patterns:
                matches = re.findall(pattern, readme, re.IGNORECASE)
                signals.extend([match.strip() for match in matches[:5]])
        
        return signals
    
    def _extract_security_signals(self, file_tree: List[Dict], readme: str) -> List[str]:
        """Extract security and privacy signals"""
        signals = []
        
        # Look for security-related files
        security_files = ['auth', 'security', 'jwt', 'oauth', 'middleware', 'guard', 'permission']
        for file_info in file_tree:
            if file_info.get('type') == 'blob':
                path = file_info.get('path', '').lower()
                if any(f in path for f in security_files):
                    signals.append(f"Security file: {file_info.get('path')}")
        
        # Look for security mentions in README
        if readme:
            security_patterns = [
                r'auth[entication]?\s*:?\s*(.+?)(?=\n|\.)',
                r'security\s*:?\s*(.+?)(?=\n|\.)',
                r'jwt\s*:?\s*(.+?)(?=\n|\.)',
                r'oauth\s*:?\s*(.+?)(?=\n|\.)'
            ]
            
            for pattern in security_patterns:
                matches = re.findall(pattern, readme, re.IGNORECASE)
                signals.extend([match.strip() for match in matches[:5]])
        
        return signals
    
    def _simulate_analysis(self, inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate analysis for demonstration purposes"""
        tech_stack = inputs_dict.get('tech_stack_detected', [])
        architecture_notes = inputs_dict.get('architecture_notes', '')
        data_flows = inputs_dict.get('data_flows', [])
        performance_signals = inputs_dict.get('performance_scalability_signals', [])
        security_signals = inputs_dict.get('security_privacy_signals', [])
        
        # Calculate complexity drivers
        complexity_drivers = []
        
        if len(tech_stack) > 5:
            complexity_drivers.append(f"Multi-technology stack: {len(tech_stack)} technologies")
        
        if architecture_notes:
            complexity_drivers.append("Architecture documented with design decisions")
        
        if data_flows:
            complexity_drivers.append(f"Data processing pipelines: {len(data_flows)} flows")
        
        if performance_signals:
            complexity_drivers.append(f"Performance optimization: {len(performance_signals)} signals")
        
        if security_signals:
            complexity_drivers.append(f"Security implementation: {len(security_signals)} components")
        
        # Determine architecture style
        if 'microservices' in str(architecture_notes).lower():
            architecture_style = "microservices"
        elif 'event' in str(architecture_notes).lower():
            architecture_style = "event-driven"
        elif 'pipeline' in str(data_flows).lower():
            architecture_style = "data-pipeline"
        elif len(tech_stack) > 3:
            architecture_style = "hybrid"
        elif len(tech_stack) > 1:
            architecture_style = "client-server"
        else:
            architecture_style = "single-tier"
        
        # Calculate integration surface area
        integration_score = min(len(tech_stack) + len(performance_signals) + len(security_signals), 10)
        
        # Calculate overall score
        base_score = len(complexity_drivers) * 2
        if architecture_notes:
            base_score += 1
        if data_flows:
            base_score += 1
        if performance_signals:
            base_score += 1
        if security_signals:
            base_score += 1
        
        overall_score = min(base_score, 10)
        
        # Calculate confidence
        distinct_signals = len(set(tech_stack + performance_signals + security_signals))
        if architecture_notes:
            distinct_signals += 1
        if data_flows:
            distinct_signals += 1
        
        confidence = min(distinct_signals / 6.0, 1.0)
        
        return {
            "complexity_drivers": complexity_drivers[:6],  # Max 6 items
            "tradeoff_rationale": f"Balanced {architecture_style} architecture with {len(tech_stack)} technologies",
            "architecture_style": architecture_style,
            "integration_surface_area": integration_score,
            "overall_numeric": overall_score,
            "confidence_estimate": confidence
        }
    
    def _generate_technical_scoring_explanation(self, result: Dict[str, Any], inputs_dict: Dict[str, Any]) -> str:
        """Generate human-friendly explanation of technical complexity scoring"""
        
        # Get metrics
        architecture_style = result.get('architecture_style', 'Unknown')
        integration_score = result.get('integration_surface_area', 0)
        complexity_drivers = result.get('complexity_drivers', [])
        tech_stack = inputs_dict.get('tech_stack_detected', [])
        overall_score = result.get('overall_numeric', 0)
        
        # Human-friendly assessment
        if overall_score >= 8:
            assessment = "Excellent technical complexity with sophisticated architecture"
        elif overall_score >= 6:
            assessment = "Good technical complexity with solid architecture"
        elif overall_score >= 4:
            assessment = "Fair technical complexity with basic architecture"
        else:
            assessment = "Poor technical complexity requiring architectural improvements"
        
        explanation = f"Technical Complexity: {overall_score}/10 - {assessment}\n\n"
        
        # Why this score is high (if high)
        if overall_score >= 8:
            explanation += "🎯 Why this score is high:\n"
            if architecture_style in ['Microservices', 'Event-driven', 'Distributed']:
                explanation += f"  • Architecture: {architecture_style} - High complexity architecture\n"
            if integration_score >= 7:
                explanation += f"  • Integration: {integration_score}/10 - Complex integration surface\n"
            if len(complexity_drivers) >= 5:
                explanation += f"  • Complexity drivers: {len(complexity_drivers)} factors - Multiple complexity factors\n"
            if len(tech_stack) >= 5:
                explanation += f"  • Tech stack: {len(tech_stack)} technologies - High technology diversity\n"
        
        # Why some points were deducted (if low)
        if overall_score < 8:
            explanation += "⚠️ Why some points were deducted:\n"
            if architecture_style not in ['Microservices', 'Event-driven', 'Distributed']:
                explanation += f"  • Architecture: {architecture_style} - Basic or simple architecture\n"
            if integration_score < 5:
                explanation += f"  • Integration: {integration_score}/10 - Simple integration\n"
            if len(complexity_drivers) < 3:
                explanation += f"  • Complexity drivers: {len(complexity_drivers)} factors - Limited complexity factors\n"
            if len(tech_stack) < 3:
                explanation += f"  • Tech stack: {len(tech_stack)} technologies - Limited technology diversity\n"
        
        # Key Reasons for the Score
        explanation += f"\n📊 Key Reasons for the Score:\n"
        explanation += f"  • Architecture: {architecture_style} - How complex the system architecture is\n"
        explanation += f"  • Integration: {integration_score}/10 - How complex the integrations are\n"
        explanation += f"  • Complexity drivers: {len(complexity_drivers)} factors - What makes the system complex\n"
        explanation += f"  • Tech stack: {len(tech_stack)} technologies - How many different technologies are used\n"
        
        # In Simple Terms
        explanation += f"\n💡 In Simple Terms:\n"
        if overall_score >= 8:
            explanation += "This is a very technically complex project! It uses advanced architecture, complex integrations, and many different technologies. It's impressive from a technical standpoint."
        elif overall_score >= 6:
            explanation += "This project has good technical complexity. It uses solid architecture and reasonable integrations. It's technically sound but not overly complex."
        elif overall_score >= 4:
            explanation += "This project has basic technical complexity. It uses simple architecture and basic integrations. It's technically straightforward but not very complex."
        else:
            explanation += "This project has very low technical complexity. It uses simple architecture and basic integrations. It's technically simple and straightforward."
        
        return explanation


class UIUXPolishAgent(BaseSpecializedAgent):
    """
    UI/UX Polish Judge.
    Analyzes screenshots (if provided) and/or textual cues for visual design quality.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("UIUXPolishAgent", api_key)
        self.ui_renderer = UIRenderer() if UI_RENDERING_AVAILABLE else None
        self.prompt = """
You are the UI/UX Polish Judge. Analyze screenshots (if provided) and/or textual cues.

## Inputs
- screenshots: [urls or binary refs] (optional but preferred)
- ui_routes_or_pages: (optional) ["Home","Dashboard","Settings",...]
- brand_or_theme_notes: (optional) from README
- copy_examples: (optional) small snippets of headings/buttons/tooltips

## Evaluate visually (if screenshots provided)
1) Visual polish & consistency (component harmony, states, affordances)
2) Spacing & alignment (grids, padding, vertical rhythm)
3) Hierarchy & layout (scan-ability, F-pattern/Z-pattern, prominence)
4) Color & contrast (readability, accessible contrast)
5) Typography (scale, leading, weights, pairing)
6) Micro-interactions & feedback hints (hover, active, validation messaging if visible)

If NO screenshots: infer from names and copy; score conservatively and note the limitation.

## Scoring rubric (0–10)
- 0–2: Prototype-looking, misaligned, poor contrast.
- 3–4: Basic layout with inconsistencies.
- 5–6: Usable and reasonably consistent; some spacing/contrast issues.
- 7–8: Polished, balanced spacing, clear hierarchy, accessible choices.
- 9–10: Production-grade visual system; crisp typography, consistent components, strong contrast.

## Confidence (0.0–1.0)
If screenshots >= 2: base 0.8; else if screenshots=1: base 0.6; else base 0.3.
+0.1 if ui_routes_or_pages present; +0.1 if copy_examples present. Cap at 1.0.

## Output JSON (STRICT)
{
  "quick_take": "one-line aesthetic verdict (neutral tone)",
  "subscores": {
    "visual_polish": 0-10,
    "spacing_alignment": 0-10,
    "hierarchy_layout": 0-10,
    "color_contrast": 0-10,
    "typography": 0-10
  },
  "overall_numeric": 0-10, 
  "strengths": ["3–5 concrete strengths observed"],
  "fix_first": ["3–5 precise, visual fixes (e.g., 'increase line-height on body to 1.6')"],
  "confidence_estimate": 0.0-1.0,
  "limitations": "state if screenshots were missing or partial"
}

## Rules
- Do not guess unseen screens.
- Use objective language (avoid 'beautiful', prefer 'consistent spacing and clear contrast').
- Keep strengths/fix_first actionable and specific.
"""
    
    def analyze(self, context: Dict[str, Any]) -> AgentAnalysis:
        """Analyze UI/UX polish with UI execution capabilities"""
        inputs_dict = self._extract_inputs(context)
        
        # Add UI execution analysis if available
        ui_execution_analysis = self._analyze_ui_execution(context)
        if ui_execution_analysis:
            # Enhance inputs with UI execution data
            inputs_dict.update(ui_execution_analysis)
        
        result = self.run_agent(None, inputs_dict)
        
        # Normalize score to 0-10
        normalized_score = max(0, min(result.get('overall_numeric', 5), 10))
        
        # Create a modified result dict with normalized score for the explanation method
        result_normalized = result.copy()
        result_normalized['overall_numeric'] = normalized_score
        
        # Generate detailed scoring explanation with normalized score
        scoring_explanation = self._generate_ui_polish_scoring_explanation(result_normalized, inputs_dict)
        
        # Convert to evidence format
        evidence = result.get('strengths', [])
        
        # Add UI execution evidence if available
        if ui_execution_analysis:
            evidence.extend(ui_execution_analysis.get('evidence', []))
        
        # Add basic insights
        insights = []
        if result.get('quick_take'):
            insights.append(result.get('quick_take', ''))
        
        # Add insights based on subscores
        subscores = result.get('subscores', {})
        if subscores.get('visual_polish', 0) > 0:
            insights.append("Visual polish and component consistency detected")
        if subscores.get('spacing_alignment', 0) > 0:
            insights.append("Good spacing and alignment patterns found")
        if subscores.get('hierarchy_layout', 0) > 0:
            insights.append("Clear visual hierarchy implemented")
        if subscores.get('color_contrast', 0) > 0:
            insights.append("Good color contrast and accessibility")
        if subscores.get('typography', 0) > 0:
            insights.append("Consistent typography and styling")
        
        # Add detailed scoring explanation
        insights.append(scoring_explanation)
        
        # If no basic insights were added, add a default one
        if len(insights) == 1:  # Only the detailed explanation
            insights.insert(0, "UI/UX Polish analysis completed")
        
        return AgentAnalysis(
            agent_name=self.name,
            score=normalized_score,
            confidence=result.get('confidence_estimate', 0.5),
            evidence=evidence,
            recommendations=result.get('fix_first', []),
            insights=insights,
            risks=[result.get('limitations', '')] if result.get('limitations') else []
        )
    
    def _generate_ui_polish_scoring_explanation(self, result: Dict[str, Any], inputs_dict: Dict[str, Any]) -> str:
        """Generate human-friendly explanation of UI/UX Polish scoring"""
        
        # Get subscores
        subscores = result.get('subscores', {})
        visual_polish = subscores.get('visual_polish', 0)
        spacing_alignment = subscores.get('spacing_alignment', 0)
        hierarchy_layout = subscores.get('hierarchy_layout', 0)
        color_contrast = subscores.get('color_contrast', 0)
        typography = subscores.get('typography', 0)
        overall_score = result.get('overall_numeric', 0)
        
        # Human-friendly assessment
        if overall_score >= 8:
            assessment = "Excellent UI/UX polish with production-grade visual system"
        elif overall_score >= 6:
            assessment = "Good UI/UX polish with polished, balanced design"
        elif overall_score >= 4:
            assessment = "Fair UI/UX polish with basic layout and some inconsistencies"
        else:
            assessment = "Poor UI/UX polish requiring significant visual improvements"
        
        explanation = f"UI/UX Polish: {overall_score}/10 - {assessment}\n\n"
        
        # Why this score is high (if high)
        if overall_score >= 8:
            explanation += "🎯 Why this score is high:\n"
            if visual_polish >= 7:
                explanation += f"  • Visual polish: {visual_polish}/10 - Excellent component harmony and consistency\n"
            if spacing_alignment >= 7:
                explanation += f"  • Spacing & alignment: {spacing_alignment}/10 - Balanced spacing and clear alignment\n"
            if hierarchy_layout >= 7:
                explanation += f"  • Hierarchy & layout: {hierarchy_layout}/10 - Clear visual hierarchy and layout\n"
            if color_contrast >= 7:
                explanation += f"  • Color & contrast: {color_contrast}/10 - Excellent readability and contrast\n"
            if typography >= 7:
                explanation += f"  • Typography: {typography}/10 - Crisp typography and consistent styling\n"
        
        # Why some points were deducted (if low)
        if overall_score < 8:
            explanation += "⚠️ Why some points were deducted:\n"
            if visual_polish < 5:
                explanation += f"  • Visual polish: {visual_polish}/10 - Prototype-looking with poor consistency\n"
            if spacing_alignment < 5:
                explanation += f"  • Spacing & alignment: {spacing_alignment}/10 - Misaligned elements and poor spacing\n"
            if hierarchy_layout < 5:
                explanation += f"  • Hierarchy & layout: {hierarchy_layout}/10 - Poor visual hierarchy and layout\n"
            if color_contrast < 5:
                explanation += f"  • Color & contrast: {color_contrast}/10 - Poor contrast and readability\n"
            if typography < 5:
                explanation += f"  • Typography: {typography}/10 - Inconsistent typography and poor styling\n"
        
        # Key Reasons for the Score
        explanation += f"\n📊 Key Reasons for the Score:\n"
        explanation += f"  • Visual polish: {visual_polish}/10 - How polished and consistent the visual design is\n"
        explanation += f"  • Spacing & alignment: {spacing_alignment}/10 - How well elements are spaced and aligned\n"
        explanation += f"  • Hierarchy & layout: {hierarchy_layout}/10 - How clear the visual hierarchy is\n"
        explanation += f"  • Color & contrast: {color_contrast}/10 - How readable and accessible the colors are\n"
        explanation += f"  • Typography: {typography}/10 - How consistent and well-designed the typography is\n"
        
        # In Simple Terms
        explanation += f"\n💡 In Simple Terms:\n"
        if overall_score >= 8:
            explanation += "This project has excellent UI/UX polish! The design is production-grade with consistent components, clear hierarchy, and excellent typography. It looks professional and polished."
        elif overall_score >= 6:
            explanation += "This project has good UI/UX polish. The design is polished with balanced spacing and clear hierarchy. It looks professional but could be improved."
        elif overall_score >= 4:
            explanation += "This project has basic UI/UX polish. The design is usable but has some inconsistencies and spacing issues. It works but needs improvement."
        else:
            explanation += "This project has poor UI/UX polish. The design looks prototype-like with misaligned elements and poor contrast. It needs significant visual improvements."
        
        return explanation
    
    def _extract_inputs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant inputs from context"""
        readme = context.get('readme', '')
        file_tree = context.get('file_tree', [])
        artifacts = context.get('artifacts', {})
        
        # Extract screenshots
        screenshots = []
        if artifacts.get('screenshots_or_demo'):
            screenshots.append(artifacts['screenshots_or_demo'])
        
        # Extract UI routes/pages
        ui_routes_or_pages = self._extract_ui_routes(file_tree, readme)
        
        # Extract brand/theme notes
        brand_or_theme_notes = self._extract_brand_notes(readme)
        
        # Extract copy examples
        copy_examples = self._extract_copy_examples(readme, file_tree)
        
        return {
            'screenshots': screenshots,
            'ui_routes_or_pages': ui_routes_or_pages,
            'brand_or_theme_notes': brand_or_theme_notes,
            'copy_examples': copy_examples
        }
    
    def _extract_ui_routes(self, file_tree: List[Dict], readme: str) -> List[str]:
        """Extract UI routes/pages from file structure and README"""
        routes = []
        
        # Look for UI files
        ui_files = ['.html', '.jsx', '.tsx', '.vue', '.svelte']
        for file_info in file_tree:
            if file_info.get('type') == 'blob':
                path = file_info.get('path', '')
                if any(path.endswith(ext) for ext in ui_files):
                    # Extract page name from path
                    page_name = path.split('/')[-1].split('.')[0]
                    routes.append(page_name.title())
        
        # Look for route definitions in README
        if readme:
            route_patterns = [
                r'page[s]?\s*:?\s*(.+?)(?=\n|\.)',
                r'route[s]?\s*:?\s*(.+?)(?=\n|\.)',
                r'screen[s]?\s*:?\s*(.+?)(?=\n|\.)'
            ]
            
            for pattern in route_patterns:
                matches = re.findall(pattern, readme, re.IGNORECASE)
                routes.extend([match.strip() for match in matches[:10]])
        
        return list(set(routes))[:10]  # Remove duplicates and limit
    
    def _extract_brand_notes(self, readme: str) -> str:
        """Extract brand/theme notes from README"""
        if not readme:
            return ""
        
        patterns = [
            r'##\s*Design.*?\n(.*?)(?=\n##|\n#|\Z)',
            r'##\s*Theme.*?\n(.*?)(?=\n##|\n#|\Z)',
            r'##\s*Brand.*?\n(.*?)(?=\n##|\n#|\Z)',
            r'##\s*UI.*?\n(.*?)(?=\n##|\n#|\Z)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, readme, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()[:300]
        
        return ""
    
    def _extract_copy_examples(self, readme: str, file_tree: List[Dict]) -> List[str]:
        """Extract copy examples from README and UI files"""
        examples = []
        
        if readme:
            # Look for button text, headings, etc.
            copy_patterns = [
                r'button[s]?\s*:?\s*(.+?)(?=\n|\.)',
                r'heading[s]?\s*:?\s*(.+?)(?=\n|\.)',
                r'title[s]?\s*:?\s*(.+?)(?=\n|\.)'
            ]
            
            for pattern in copy_patterns:
                matches = re.findall(pattern, readme, re.IGNORECASE)
                examples.extend([match.strip() for match in matches[:5]])
        
        return examples[:5]
    
    def _simulate_analysis(self, inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate analysis for demonstration purposes"""
        screenshots = inputs_dict.get('screenshots', [])
        ui_routes = inputs_dict.get('ui_routes_or_pages', [])
        brand_notes = inputs_dict.get('brand_or_theme_notes', '')
        copy_examples = inputs_dict.get('copy_examples', [])
        
        # Calculate confidence
        confidence = 0.3  # Base confidence
        if len(screenshots) >= 2:
            confidence = 0.8
        elif len(screenshots) == 1:
            confidence = 0.6
        
        if ui_routes:
            confidence = min(confidence + 0.1, 1.0)
        if copy_examples:
            confidence = min(confidence + 0.1, 1.0)
        
        # Calculate subscores based on available information
        visual_polish = 6 if screenshots else 4
        spacing_alignment = 6 if ui_routes else 4
        hierarchy_layout = 6 if brand_notes else 4
        color_contrast = 5 if screenshots else 3
        typography = 6 if copy_examples else 4
        
        overall = int((visual_polish + spacing_alignment + hierarchy_layout + color_contrast + typography) / 5)
        
        # Generate strengths
        strengths = []
        if screenshots:
            strengths.append("Visual assets provided for evaluation")
        if ui_routes:
            strengths.append(f"Multiple UI pages identified: {len(ui_routes)}")
        if brand_notes:
            strengths.append("Design system considerations documented")
        if copy_examples:
            strengths.append("UI copy examples available")
        
        if not strengths:
            strengths.append("Basic UI structure detected")
        
        # Generate fix suggestions
        fix_first = []
        if not screenshots:
            fix_first.append("Provide screenshots for visual evaluation")
        if not ui_routes:
            fix_first.append("Document UI page structure")
        if not brand_notes:
            fix_first.append("Add design system documentation")
        if not copy_examples:
            fix_first.append("Include UI copy examples")
        
        # Determine limitations
        limitations = ""
        if not screenshots:
            limitations = "No screenshots provided - evaluation based on structure only"
        
        return {
            "quick_take": f"UI structure with {len(ui_routes)} pages, {'visual assets available' if screenshots else 'no visual assets'}",
            "subscores": {
                "visual_polish": visual_polish,
                "spacing_alignment": spacing_alignment,
                "hierarchy_layout": hierarchy_layout,
                "color_contrast": color_contrast,
                "typography": typography
            },
            "overall_numeric": overall,
            "strengths": strengths[:5],
            "fix_first": fix_first[:5],
            "confidence_estimate": confidence,
            "limitations": limitations
        }
    
    def _analyze_ui_execution(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute and analyze the UI if possible"""
        if not self.ui_renderer:
            return None
        
        try:
            # Get project path from context
            project_path = context.get('project_path', '.')
            if not os.path.exists(project_path):
                return None
            
            # Try to execute the application
            execution_result = self.ui_renderer.execute_web_app(project_path)
            
            if not execution_result['success']:
                return {
                    'evidence': ['UI execution failed - application could not be started'],
                    'screenshots': [],
                    'visual_analysis': {}
                }
            
            # Analyze the captured UI
            ui_analysis = execution_result.get('analysis', {})
            screenshots = execution_result.get('screenshots', [])
            
            evidence = []
            visual_analysis = {}
            
            # Visual quality analysis
            visual_quality = ui_analysis.get('visual_quality', 0)
            if visual_quality > 7:
                evidence.append(f"Excellent visual quality (score: {visual_quality:.1f})")
            elif visual_quality > 5:
                evidence.append(f"Good visual quality (score: {visual_quality:.1f})")
            else:
                evidence.append(f"Visual quality needs improvement (score: {visual_quality:.1f})")
            
            visual_analysis['visual_quality'] = visual_quality
            
            # Accessibility analysis
            accessibility = ui_analysis.get('accessibility', 0)
            if accessibility > 7:
                evidence.append(f"Excellent accessibility (score: {accessibility:.1f})")
            elif accessibility > 5:
                evidence.append(f"Good accessibility (score: {accessibility:.1f})")
            else:
                evidence.append(f"Accessibility needs improvement (score: {accessibility:.1f})")
            
            visual_analysis['accessibility'] = accessibility
            
            # Responsiveness analysis
            responsiveness = ui_analysis.get('responsiveness', 0)
            if responsiveness > 7:
                evidence.append(f"Excellent responsiveness (score: {responsiveness:.1f})")
            elif responsiveness > 5:
                evidence.append(f"Good responsiveness (score: {responsiveness:.1f})")
            else:
                evidence.append(f"Responsiveness needs improvement (score: {responsiveness:.1f})")
            
            visual_analysis['responsiveness'] = responsiveness
            
            # Interactivity analysis
            interactivity = ui_analysis.get('interactivity', 0)
            if interactivity > 7:
                evidence.append(f"Excellent interactivity (score: {interactivity:.1f})")
            elif interactivity > 5:
                evidence.append(f"Good interactivity (score: {interactivity:.1f})")
            else:
                evidence.append(f"Interactivity needs improvement (score: {interactivity:.1f})")
            
            visual_analysis['interactivity'] = interactivity
            
            # Screenshot analysis
            if screenshots:
                evidence.append(f"Captured {len(screenshots)} UI screenshots")
                visual_analysis['screenshot_count'] = len(screenshots)
            
            return {
                'evidence': evidence,
                'screenshots': [s['path'] for s in screenshots],
                'visual_analysis': visual_analysis
            }
            
        except Exception as e:
            return {
                'evidence': [f'UI execution analysis failed: {str(e)}'],
                'screenshots': [],
                'visual_analysis': {}
            }


def run_new_agent(model, prompt: str, inputs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a new agent with the provided model and inputs.
    
    Args:
        model: The LLM model to use (could be text or vision LLM)
        prompt: The agent prompt
        inputs_dict: Dictionary of inputs for the agent
    
    Returns:
        Dictionary containing the agent's analysis results
    """
    # IMPORTANT: Always ask for valid JSON only.
    final_prompt = (
        prompt
        + "\n\nReturn ONLY the JSON object. Do not add prose before/after."
        + "\n\nInputs:\n" + json.dumps(inputs_dict, ensure_ascii=False)
    )
    
    # In a real implementation, this would call the model
    # raw = model.generate(final_prompt)
    # result = json.loads(raw)
    
    # For demonstration, return a mock result
    return {
        "overall_numeric": 7,
        "confidence_estimate": 0.8,
        "evidence_points": ["Mock evidence"],
        "specific_opportunities": ["Mock opportunity"],
        "risks_or_caveats": ["Mock risk"]
    }
    
class AgentOrchestrator:
    """Coordinates all our AI agents - the conductor of the orchestra"""
    
    def __init__(self, api_key: Optional[str] = None, use_specialized_agents: bool = False):
        # Set up ALL 9 agents - no differentiation
        self.agents = {
            # Core Analysis Agents
            'code': CodeAnalysisAgent(api_key),
            'architecture': ArchitectureAgent(api_key),
            'ui_ux': UIUXAgent(api_key),
            'security': SecurityAgent(api_key),
            
            # Specialized Analysis Agents
            'innovation': InnovationCreativityAgent(api_key),
            'functionality': FunctionalityCompletenessAgent(api_key),
            'technical': TechnicalComplexityAgent(api_key),
            'ui_ux_polish': UIUXPolishAgent(api_key),
            
            # Learning Agent
            'learning': LearningAgent(api_key)
        }
        
        self.analysis_history = []  # Keep track of what we have done
        self.use_specialized_agents = False  # No longer needed - all agents available
    
    def analyze(self, context: Dict[str, Any], selected_agents: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run all our agents and put their results together"""
        agent_results = {}
        
        # Filter agents based on selection
        agents_to_run = self.agents
        if selected_agents:
            # Map agent numbers to agent names - ALL 9 AGENTS AVAILABLE
            agent_mapping = {
                '1': 'code',                    # Code Analysis
                '2': 'architecture',            # Architecture Analysis
                '3': 'ui_ux',                   # UI/UX Analysis
                '4': 'security',                # Security Analysis
                '5': 'innovation',              # Innovation & Creativity
                '6': 'functionality',           # Functionality & Completeness
                '7': 'technical',               # Technical Complexity
                '8': 'ui_ux_polish',            # UI/UX Polish
                '9': 'learning'                 # Learning Agent
            }
            
            # Filter to only selected agents
            selected_agent_names = [agent_mapping.get(num) for num in selected_agents if num in agent_mapping]
            agents_to_run = {name: agent for name, agent in self.agents.items() if name in selected_agent_names}
        
        # Let each selected agent do their thing
        for agent_name, agent in agents_to_run.items():
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
        
        # Add specialized agent mapping if using specialized agents
        agent_mapping = {}
        if self.use_specialized_agents:
            agent_mapping = {
                'innovation': 'Innovation & Creativity',
                'functionality': 'Functionality & Completeness', 
                'technical': 'Technical Complexity',
                'ui_ux_polish': 'UI/UX Polish',
                'learning': 'Learning Agent'
            }
        
        return {
            'total_score': round(final_score, 1),
            'agent_scores': {name: result.score for name, result in agent_results.items()},
            'confidence_scores': {name: result.confidence for name, result in agent_results.items()},
            'evidence': all_evidence[:20],  # Limit to top 20
            'recommendations': all_recommendations[:15],  # Limit to top 15
            'insights': all_insights,  # Don't limit - we need all insights for detailed human-friendly output
            'risks': all_risks[:10],  # Limit to top 10
            'agent_count': len(agent_results),
            'agent_mapping': agent_mapping,
            'use_specialized_agents': self.use_specialized_agents
        }

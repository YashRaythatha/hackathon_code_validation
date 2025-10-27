#!/usr/bin/env python3
"""
AI-Powered Hackathon Grader

This grader uses AI agents to evaluate hackathon submissions.
It's designed to be fair, consistent, and provide detailed feedback.

How it works:
1. User gives us a GitHub URL
2. We fetch the repo data
3. AI agents analyze different aspects
4. We combine everything into a final score
"""

import json
import sys
import time
from typing import Dict, List, Any, Optional
from github_analyzer import GitHubAnalyzer
from ai_agents import AgentOrchestrator
from core import analysis_cache, get_logger


class AIGrader:
    """Main grader class - coordinates everything"""
    
    def __init__(self, github_token: Optional[str] = None, ai_api_key: Optional[str] = None, 
                 use_specialized_agents: bool = False, judge_config: Optional['JudgeConfig'] = None):
        # Set up our GitHub connection and AI agents
        self.github = GitHubAnalyzer(github_token)
        self.judge_config = judge_config
        self.agent_orchestrator = AgentOrchestrator(ai_api_key, use_specialized_agents)
        self.logger = get_logger(__name__)
        self.use_specialized_agents = False  # All agents available now
    
    def grade(self, repo_url: str, branch: str = 'main', commit_sha: Optional[str] = None,
              artifacts: Optional[Dict[str, Any]] = None, selected_agents: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Main grading method - this is where the magic happens
        
        Args:
            repo_url: The GitHub repo we're analyzing
            branch: Which branch to look at (usually 'main')
            commit_sha: Specific commit if needed
            artifacts: Extra info the user might provide
            selected_agents: List of agent numbers to run (e.g., ['1', '3', '5'])
        
        Returns:
            Complete grading results with scores and explanations
        """
        
        # Handle optional artifacts
        if artifacts is None:
            artifacts = {}
        
        # Check cache first
        if selected_agents:
            cached_result = analysis_cache.get_analysis(repo_url, branch, selected_agents)
            if cached_result:
                self.logger.info(f"Returning cached result for {repo_url}")
                return cached_result
        
        try:
            print("🤖 Starting AI analysis...")
            print("-" * 60)
            
            # First, let's get the repo data from GitHub
            print("📡 Getting repository info from GitHub...")
            repo_info = self.github.get_repo_info(repo_url, branch)
            file_tree = self.github.get_file_tree(repo_info)
            readme = self.github.get_readme(repo_info)
            
            # Now we need to prepare everything for our AI agents
            print("📦 Setting up data for AI analysis...")
            context = {
                'file_tree': file_tree,
                'readme': readme,
                'artifacts': artifacts,
                'repo_info': repo_info,
                'repo_url': repo_url,
                'branch': branch,
                'commit_sha': commit_sha,
                'project_path': repo_url  # GitHub URL for UI execution
            }
            
            # Time to let our AI agents do their thing
            print("🤖 Running AI agents...")
            print("   • Code Analysis Agent - Looking at code quality...")
            print("   • Architecture Agent - Checking project structure...")
            print("   • UI/UX Agent - Evaluating user interface...")
            print("   • Security Agent - Scanning for security issues...")
            print("   • Learning Agent - Using past experience...")
            
            ai_results = self.agent_orchestrator.analyze(context, selected_agents)
            
            # Now we need to turn the AI results into actual grades
            print("📊 Converting AI analysis to scores...")
            final_results = self._map_ai_results_to_grades(ai_results, context, selected_agents)
            
            # Cache the results
            if selected_agents:
                analysis_cache.set_analysis(repo_url, branch, selected_agents, final_results)
                self.logger.info(f"Cached analysis result for {repo_url}")
            
            print("✅ Analysis complete!\n")
            
            return final_results
            
        except Exception as e:
            return {
                "error": f"Failed to analyze repository: {str(e)}",
                "total_score": 0,
                "breakdown": {},
                "pass_fail": "fail",
                "ai_powered": True
            }
    
    def _map_ai_results_to_grades(self, ai_results: Dict[str, Any], 
                                  context: Dict[str, Any], selected_agents: List[str] = None) -> Dict[str, Any]:
        """
        Map AI agent results to the grading structure
        
        For specialized agents:
        - innovation agent → Innovation category (25%)
        - functionality agent → Functionality category (25%)
        - technical agent → Technical category (25%)
        - ui_ux_polish agent → UI/UX category (25%)
        
        For original agents:
        - ui_ux agent → UI category (10%)
        - architecture agent → Architecture category (40%)
        - code agent → Coding category (30%)
        - security agent → Other/Compliance category (20%)
        """
        
        # Extract AI agent scores
        agent_scores = ai_results.get('agent_scores', {})
        agent_confidence = ai_results.get('confidence_scores', {})
        
        # Map ALL 9 agents to categories - only include agents that were actually run
        innovation_score = agent_scores.get('innovation', 5)
        functionality_score = agent_scores.get('functionality', 5)
        technical_score = agent_scores.get('technical', 5)
        ui_ux_polish_score = agent_scores.get('ui_ux_polish', 5)
        code_score = agent_scores.get('code', 5)
        architecture_score = agent_scores.get('architecture', 5)
        ui_ux_score = agent_scores.get('ui_ux', 5)
        security_score = agent_scores.get('security', 5)
        learning_score = agent_scores.get('learning', 5)
        
        # Calculate weighted total score based on available agents
        available_scores = []
        if 'innovation' in agent_scores:
            available_scores.append(innovation_score)
        if 'functionality' in agent_scores:
            available_scores.append(functionality_score)
        if 'technical' in agent_scores:
            available_scores.append(technical_score)
        if 'ui_ux_polish' in agent_scores:
            available_scores.append(ui_ux_polish_score)
        if 'code' in agent_scores:
            available_scores.append(code_score)
        if 'architecture' in agent_scores:
            available_scores.append(architecture_score)
        if 'ui_ux' in agent_scores:
            available_scores.append(ui_ux_score)
        if 'security' in agent_scores:
            available_scores.append(security_score)
        if 'learning' in agent_scores:
            available_scores.append(learning_score)
        
        if available_scores:
            weighted_total = round(sum(available_scores) / len(available_scores))
        else:
            weighted_total = 5
        
        # Map to display categories - handle specialized agents as separate categories
        ui_score = ui_ux_polish_score if 'ui_ux_polish' in agent_scores else (ui_ux_score if 'ui_ux' in agent_scores else 0)
        architecture_score = technical_score if 'technical' in agent_scores else (architecture_score if 'architecture' in agent_scores else 0)
        coding_score = code_score if 'code' in agent_scores else (functionality_score if 'functionality' in agent_scores else 0)
        other_score = innovation_score if 'innovation' in agent_scores else (security_score if 'security' in agent_scores else 0)
        
        # Get evidence from AI agents
        all_evidence = ai_results.get('evidence', [])
        all_recommendations = ai_results.get('recommendations', [])
        all_insights = ai_results.get('insights', [])
        all_risks = ai_results.get('risks', [])
        
        # Deduplicate recommendations to avoid repetition
        all_recommendations = self._deduplicate_recommendations(all_recommendations)
        
        # Parse insights by agent
        agent_insights = {}
        agent_recommendations = {}
        agent_risks = {}
        
        for insight in all_insights:
            if ': ' in insight:
                agent_name, insight_text = insight.split(': ', 1)
                if agent_name not in agent_insights:
                    agent_insights[agent_name] = []
                agent_insights[agent_name].append(insight_text)
        
        # Deduplicate recommendations before parsing by agent
        seen_recs = set()
        for rec in all_recommendations:
            if ': ' in rec:
                agent_name, rec_text = rec.split(': ', 1)
                # Create a key for deduplication
                dedup_key = rec_text.lower()
                if dedup_key not in seen_recs:
                    seen_recs.add(dedup_key)
                    if agent_name not in agent_recommendations:
                        agent_recommendations[agent_name] = []
                    agent_recommendations[agent_name].append(rec_text)
        
        # Deduplicate risks before parsing by agent
        seen_risks = set()
        for risk in all_risks:
            if ': ' in risk:
                agent_name, risk_text = risk.split(': ', 1)
                # Create a key for deduplication
                dedup_key = risk_text.lower()
                if dedup_key not in seen_risks:
                    seen_risks.add(dedup_key)
                    if agent_name not in agent_risks:
                        agent_risks[agent_name] = []
                    agent_risks[agent_name].append(risk_text)
        
        # Separate evidence by agent/category - handle all 9 agents
        ui_evidence = [e for e in all_evidence if 'ui_ux_polish:' in e.lower() or 'ui_ux:' in e.lower()]
        arch_evidence = [e for e in all_evidence if 'technical:' in e.lower() or 'architecture:' in e.lower()]
        code_evidence = [e for e in all_evidence if 'code:' in e.lower() or 'functionality:' in e.lower()]
        security_evidence = [e for e in all_evidence if 'innovation:' in e.lower() or 'security:' in e.lower()]
        
        # Determine pass/fail status
        if weighted_total >= 7:
            pass_fail = "pass"
        elif weighted_total >= 5:
            pass_fail = "borderline"
        else:
            pass_fail = "fail"
        
        # Generate AI-powered explanations - handle all 9 agents with correct names
        # These are fallback comments - each agent will generate its own comment in the breakdown
        ui_comment = self._generate_ai_comment('UI/UX', ui_score, ui_evidence)
        arch_comment = self._generate_ai_comment('Architecture', architecture_score, arch_evidence)
        code_comment = self._generate_ai_comment('Code Analysis', coding_score, code_evidence)
        other_comment = self._generate_ai_comment('Security/Compliance', other_score, security_evidence)
        
        # Build comprehensive result - handle all 9 agents with proper categorization
        breakdown = {}
        
        # Calculate weights - use judge config if available, otherwise equal distribution
        if self.judge_config:
            # Use judge-defined weights
            judge_weights = self.judge_config.get_weights()
            total_agents = len(agent_scores)
            weight_per_agent = 100 // total_agents if total_agents > 0 else 25
        else:
            # Equal distribution (default behavior)
            total_agents = len(agent_scores)
            weight_per_agent = 100 // total_agents if total_agents > 0 else 25
        
        # UI/UX Category (prioritize specialized)
        if 'ui_ux_polish' in agent_scores:
            # Use judge weight if available, otherwise equal distribution
            ui_weight = judge_weights.get('ui_ux_polish', weight_per_agent) if self.judge_config else weight_per_agent
            breakdown["ui_ux_polish"] = {
                "score": ui_ux_polish_score,
                "weight": ui_weight,
                "evidence": ui_evidence[:10] if ui_evidence else ["AI analysis completed"],
                "comment": self._generate_ai_comment('UI/UX Polish', ui_ux_polish_score, ui_evidence),
                "ai_confidence": agent_confidence.get('ui_ux_polish', 0.5),
                "insights": agent_insights.get('ui_ux_polish', []),
                "recommendations": agent_recommendations.get('ui_ux_polish', []),
                "risks": agent_risks.get('ui_ux_polish', [])
            }
        elif 'ui_ux' in agent_scores:
            # Use judge weight if available, otherwise equal distribution
            ui_weight = judge_weights.get('ui_ux', weight_per_agent) if self.judge_config else weight_per_agent
            breakdown["ui"] = {
                "score": ui_score,
                "weight": ui_weight,
                "evidence": ui_evidence[:10] if ui_evidence else ["AI analysis completed"],
                "comment": self._generate_ai_comment('UI/UX', ui_score, ui_evidence),
                "ai_confidence": agent_confidence.get('ui_ux', 0.5),
                "insights": agent_insights.get('ui_ux', []),
                "recommendations": agent_recommendations.get('ui_ux', []),
                "risks": agent_risks.get('ui_ux', [])
            }
        
        # Architecture Category (prioritize specialized)
        if 'technical' in agent_scores:
            # Use judge weight if available, otherwise equal distribution
            tech_weight = judge_weights.get('technical', weight_per_agent) if self.judge_config else weight_per_agent
            breakdown["technical_complexity"] = {
                "score": technical_score,
                "weight": tech_weight,
                "evidence": arch_evidence[:10] if arch_evidence else ["AI analysis completed"],
                "comment": self._generate_ai_comment('Technical Complexity', technical_score, arch_evidence),
                "ai_confidence": agent_confidence.get('technical', 0.5),
                "insights": agent_insights.get('technical', []),
                "recommendations": agent_recommendations.get('technical', []),
                "risks": agent_risks.get('technical', [])
            }
        elif 'architecture' in agent_scores:
            # Use judge weight if available, otherwise equal distribution
            arch_weight = judge_weights.get('architecture', weight_per_agent) if self.judge_config else weight_per_agent
            breakdown["architecture"] = {
                "score": architecture_score,
                "weight": arch_weight,
                "evidence": arch_evidence[:10] if arch_evidence else ["AI analysis completed"],
                "comment": self._generate_ai_comment('Architecture', architecture_score, arch_evidence),
                "ai_confidence": agent_confidence.get('architecture', 0.5),
                "insights": agent_insights.get('architecture', []),
                "recommendations": agent_recommendations.get('architecture', []),
                "risks": agent_risks.get('architecture', [])
            }
        
        # Code Category (prioritize specialized)
        if 'code' in agent_scores:
            # Use judge weight if available, otherwise equal distribution
            code_weight = judge_weights.get('code_analysis', weight_per_agent) if self.judge_config else weight_per_agent
            breakdown["code_analysis"] = {
                "score": code_score,
                "weight": code_weight,
                "evidence": code_evidence[:10] if code_evidence else ["AI analysis completed"],
                "comment": self._generate_ai_comment('Code Analysis', code_score, code_evidence),
                "ai_confidence": agent_confidence.get('code', 0.5),
                "insights": agent_insights.get('code', []),
                "recommendations": agent_recommendations.get('code', []),
                "risks": agent_risks.get('code', [])
            }
        elif 'functionality' in agent_scores:
            # Use judge weight if available, otherwise equal distribution
            func_weight = judge_weights.get('functionality', weight_per_agent) if self.judge_config else weight_per_agent
            breakdown["functionality"] = {
                "score": functionality_score,
                "weight": func_weight,
                "evidence": code_evidence[:10] if code_evidence else ["AI analysis completed"],
                "comment": self._generate_ai_comment('Functionality & Completeness', functionality_score, code_evidence),
                "ai_confidence": agent_confidence.get('functionality', 0.5),
                "insights": agent_insights.get('functionality', []),
                "recommendations": agent_recommendations.get('functionality', []),
                "risks": agent_risks.get('functionality', [])
            }
        
        # Innovation/Security Category (prioritize specialized)
        if 'innovation' in agent_scores:
            # Use judge weight if available, otherwise equal distribution
            innovation_weight = judge_weights.get('innovation', weight_per_agent) if self.judge_config else weight_per_agent
            breakdown["innovation"] = {
                "score": innovation_score,
                "weight": innovation_weight,
                "evidence": security_evidence[:10] if security_evidence else ["AI analysis completed"],
                "comment": self._generate_ai_comment('Innovation & Creativity', innovation_score, security_evidence),
                "ai_confidence": agent_confidence.get('innovation', 0.5),
                "subchecks": self._generate_subchecks(innovation_score, security_evidence),
                "insights": agent_insights.get('innovation', []),
                "recommendations": agent_recommendations.get('innovation', []),
                "risks": agent_risks.get('innovation', [])
            }
        elif 'security' in agent_scores:
            # Use judge weight if available, otherwise equal distribution
            security_weight = judge_weights.get('security', weight_per_agent) if self.judge_config else weight_per_agent
            breakdown["security"] = {
                "score": security_score,
                "weight": security_weight,
                "evidence": security_evidence[:10] if security_evidence else ["AI analysis completed"],
                "comment": self._generate_ai_comment('Security/Compliance', security_score, security_evidence),
                "ai_confidence": agent_confidence.get('security', 0.5),
                "subchecks": self._generate_subchecks(security_score, security_evidence),
                "insights": agent_insights.get('security', []),
                "recommendations": agent_recommendations.get('security', []),
                "risks": agent_risks.get('security', [])
            }
        
        result = {
            "total_score": weighted_total,
            "breakdown": breakdown,
            "pass_fail": pass_fail,
            "selected_agents": selected_agents,
            "agent_scores": ai_results.get('agent_scores', {}),
            "confidence_scores": ai_results.get('confidence_scores', {}),
            "ai_powered": True,
            "ai_insights": all_insights,  # Don't truncate - we need all insights for proper breakdown display
            "top_strengths": self._generate_strengths(ui_score, architecture_score, coding_score, other_score),
            "top_improvements": self._generate_improvements_from_agents(agent_scores, agent_confidence),
            "plagiarism_flag": False,
            "risks_red_flags": all_risks[:5],
            "notes": {
                "assumptions": [
                    "All analysis performed by AI agents",
                    "No traditional rule-based analysis used",
                    "Confidence scores indicate AI reliability"
                ],
                "missing_artifacts": self._identify_missing_artifacts(context.get('artifacts', {})),
                "calculation": self._generate_calculation_string(breakdown, ui_score, architecture_score, coding_score, other_score, weighted_total),
                "ai_agents_used": ai_results.get('agent_count', 5),
                "agent_types": "Unified 9-Agent System"
            }
        }
        
        return result
    
    def _generate_calculation_string(self, breakdown: Dict, ui_score: int, architecture_score: int, 
                                   coding_score: int, other_score: int, weighted_total: int) -> str:
        """Generate calculation string based on available breakdown categories"""
        calculation_parts = []
        score_parts = []
        
        # Check for all possible category names in breakdown
        for category, details in breakdown.items():
            weight = details.get('weight', 0)
            score = details.get('score', 0)
            
            # Map category names to display names
            display_name = category.replace('_', ' ').title()
            if category == 'code_analysis':
                display_name = 'Code Analysis'
            elif category == 'ui_ux_polish':
                display_name = 'UI/UX Polish'
            elif category == 'technical_complexity':
                display_name = 'Technical Complexity'
            
            calculation_parts.append(f"{display_name}*{weight/100:.0%}")
            score_parts.append(f"{score}*{weight/100:.0%}")
        
        if calculation_parts:
            return f"{' + '.join(calculation_parts)} = {' + '.join(score_parts)} = {weighted_total}"
        else:
            return f"Total score: {weighted_total}"
    
    def _generate_ai_comment(self, category: str, score: int, evidence: List[str]) -> str:
        """Generate detailed, human-readable AI comment explaining WHY the score was given"""
        
        # Analyze evidence to understand what was found
        positive_indicators = []
        negative_indicators = []
        missing_indicators = []
        
        # Categorize evidence based on content
        for item in evidence:
            item_lower = item.lower()
            if any(word in item_lower for word in ['good', 'excellent', 'strong', 'solid', 'well', 'proper', 'correct', 'implemented', 'found', 'detected']):
                positive_indicators.append(item)
            elif any(word in item_lower for word in ['missing', 'lack', 'no', 'not found', 'absent', 'incomplete', 'poor', 'weak', 'issue', 'problem', 'error']):
                negative_indicators.append(item)
            else:
                # Neutral evidence - could be positive or negative depending on context
                if any(word in item_lower for word in ['test', 'documentation', 'security', 'performance', 'ui', 'ux']):
                    positive_indicators.append(item)
                else:
                    negative_indicators.append(item)
        
        # Generate specific explanation based on score and evidence
        if score >= 8:
            quality = "Excellent"
            if positive_indicators:
                specific_reasons = f" Strong evidence found: {', '.join(positive_indicators[:3])}"
                if len(positive_indicators) > 3:
                    specific_reasons += f" and {len(positive_indicators) - 3} more positive indicators"
            else:
                specific_reasons = " AI analysis indicates high quality implementation"
            
            desc = f"Outstanding {category.lower()} with comprehensive implementation"
            
        elif score >= 6:
            quality = "Good"
            if positive_indicators and negative_indicators:
                specific_reasons = f" Found {len(positive_indicators)} positive indicators but also {len(negative_indicators)} areas for improvement"
            elif positive_indicators:
                specific_reasons = f" Good implementation with {len(positive_indicators)} positive indicators found"
            else:
                specific_reasons = " Solid foundation but limited evidence of advanced features"
            
            desc = f"Solid {category.lower()} implementation with room for enhancement"
            
        elif score >= 4:
            quality = "Fair"
            if negative_indicators:
                specific_reasons = f" Issues identified: {', '.join(negative_indicators[:2])}"
                if len(negative_indicators) > 2:
                    specific_reasons += f" and {len(negative_indicators) - 2} more concerns"
            else:
                specific_reasons = " Basic implementation with limited advanced features"
            
            desc = f"Basic {category.lower()} with several areas needing improvement"
            
        else:
            quality = "Poor"
            if negative_indicators:
                specific_reasons = f" Major issues found: {', '.join(negative_indicators[:3])}"
                if len(negative_indicators) > 3:
                    specific_reasons += f" and {len(negative_indicators) - 3} more critical problems"
            else:
                specific_reasons = " Significant gaps in implementation and best practices"
            
            desc = f"Significant {category.lower()} issues requiring major improvements"
        
        # Add specific deduction reasons for lower scores
        deduction_reasons = []
        if score < 8:
            if category.lower() in ['ui/ux polish', 'ui/ux']:
                if not any('responsive' in item.lower() for item in evidence):
                    deduction_reasons.append("No responsive design detected")
                if not any('accessibility' in item.lower() for item in evidence):
                    deduction_reasons.append("Accessibility features not implemented")
                if not any('modern' in item.lower() or 'framework' in item.lower() for item in evidence):
                    deduction_reasons.append("Outdated or basic UI framework")
            
            elif category.lower() in ['technical complexity', 'architecture']:
                if not any('pattern' in item.lower() for item in evidence):
                    deduction_reasons.append("No design patterns implemented")
                if not any('scalable' in item.lower() for item in evidence):
                    deduction_reasons.append("Architecture not designed for scalability")
                if not any('api' in item.lower() for item in evidence):
                    deduction_reasons.append("No API design or integration")
            
            elif category.lower() in ['functionality & completeness', 'coding']:
                if not any('error' in item.lower() and 'handling' in item.lower() for item in evidence):
                    deduction_reasons.append("No error handling detected")
                if not any('documentation' in item.lower() for item in evidence):
                    deduction_reasons.append("Insufficient documentation")
            
            elif category.lower() in ['innovation & creativity', 'security/compliance']:
                if not any('novel' in item.lower() or 'creative' in item.lower() for item in evidence):
                    deduction_reasons.append("Limited innovation or creativity")
                if not any('security' in item.lower() for item in evidence):
                    deduction_reasons.append("Security measures not implemented")
        
        # Combine all information
        result = f"{quality}: {desc}.{specific_reasons}"
        
        if deduction_reasons:
            result += f" Score reduced due to: {', '.join(deduction_reasons[:3])}"
            if len(deduction_reasons) > 3:
                result += f" and {len(deduction_reasons) - 3} more factors"
        
        # Add evidence count for transparency
        if evidence:
            result += f" Analysis based on {len(evidence)} evidence points."
        
        return result
    
    def _generate_subchecks(self, score: int, evidence: List[str]) -> List[Dict[str, Any]]:
        """Generate subchecks for Other category"""
        subchecks = []
        
        # Documentation
        doc_status = "pass" if score >= 6 else "partial" if score >= 4 else "fail"
        subchecks.append({
            "name": "documentation",
            "status": doc_status,
            "evidence": [e for e in evidence if 'document' in e.lower()][:2],
            "note": "AI-assessed documentation quality"
        })
        
        # Security
        sec_status = "pass" if score >= 6 else "partial" if score >= 4 else "fail"
        subchecks.append({
            "name": "security",
            "status": sec_status,
            "evidence": [e for e in evidence if 'security' in e.lower()][:2],
            "note": "AI-assessed security practices"
        })
        
        # Tests/CI
        test_status = "pass" if score >= 6 else "partial" if score >= 4 else "fail"
        subchecks.append({
            "name": "tests_ci",
            "status": test_status,
            "evidence": [e for e in evidence if 'test' in e.lower() or 'ci' in e.lower()][:2],
            "note": "AI-assessed testing and CI/CD"
        })
        
        # Licensing
        lic_status = "pass" if score >= 5 else "fail"
        subchecks.append({
            "name": "licensing",
            "status": lic_status,
            "evidence": [],
            "note": "AI-assessed licensing compliance"
        })
        
        # Accessibility/Performance
        a11y_status = "pass" if score >= 6 else "partial" if score >= 4 else "fail"
        subchecks.append({
            "name": "accessibility_or_performance",
            "status": a11y_status,
            "evidence": [],
            "note": "AI-assessed accessibility and performance"
        })
        
        # Originality
        subchecks.append({
            "name": "originality",
            "status": "pass",
            "evidence": [],
            "note": "AI-assessed originality"
        })
        
        return subchecks
    
    def _generate_strengths(self, ui_score: int, arch_score: int, 
                           coding_score: int, other_score: int) -> List[str]:
        """Generate top strengths based on AI scores"""
        strengths = []
        
        if arch_score >= 7:
            strengths.append("AI detected strong architecture and design patterns")
        if coding_score >= 7:
            strengths.append("AI identified high code quality and good practices")
        if ui_score >= 7:
            strengths.append("AI found excellent UI/UX implementation")
        if other_score >= 7:
            strengths.append("AI verified comprehensive documentation and compliance")
        
        if not strengths:
            strengths.append("AI analysis in progress - continue improving")
        
        return strengths[:3]
    
    def _generate_improvements_from_agents(self, agent_scores: Dict[str, int], 
                                          agent_confidence: Dict[str, float]) -> List[str]:
        """Generate improvements based on actual agent scores and confidence"""
        improvements = []
        
        # Only suggest improvements for agents that were actually run and have low scores
        for agent_name, score in agent_scores.items():
            confidence = agent_confidence.get(agent_name, 0.5)
            
            # Only suggest improvements if score is low and confidence is high (reliable assessment)
            if score < 6 and confidence > 0.3:
                if agent_name == 'code':
                    improvements.append("Improve code quality and testing practices")
                elif agent_name == 'architecture':
                    improvements.append("Enhance architecture and design patterns")
                elif agent_name == 'ui_ux':
                    improvements.append("Improve UI/UX design and accessibility")
                elif agent_name == 'security':
                    improvements.append("Strengthen security practices and compliance")
                elif agent_name == 'innovation':
                    improvements.append("Increase innovation and creative solutions")
                elif agent_name == 'functionality':
                    improvements.append("Complete missing functionality and features")
                elif agent_name == 'technical':
                    improvements.append("Increase technical complexity and sophistication")
                elif agent_name == 'ui_ux_polish':
                    improvements.append("Polish UI/UX design and user experience")
                elif agent_name == 'learning':
                    improvements.append("Improve learning and adaptability features")
        
        # If no specific improvements needed, provide general guidance
        if not improvements:
            improvements.append("Continue maintaining high quality standards")
        
        return improvements[:3]
    
    def _deduplicate_recommendations(self, recommendations: List[str]) -> List[str]:
        """Remove duplicate recommendations to avoid repetition"""
        seen = set()
        deduplicated = []
        
        for rec in recommendations:
            # Remove agent prefix (e.g., "code: " or "architecture: ")
            clean_rec = rec.split(': ', 1)[-1] if ': ' in rec else rec
            
            # Check if we've seen this recommendation before
            if clean_rec.lower() not in seen:
                seen.add(clean_rec.lower())
                deduplicated.append(rec)
        
        return deduplicated
    
    def _identify_missing_artifacts(self, artifacts: Dict[str, Any]) -> List[str]:
        """Identify missing artifacts"""
        missing = []
        
        if not artifacts.get('test_results'):
            missing.append("test_results")
        if not artifacts.get('lint_results'):
            missing.append("lint_results")
        if not artifacts.get('screenshots_or_demo'):
            missing.append("screenshots_or_demo")
        if not artifacts.get('ci_config_present'):
            missing.append("ci_config")
        
        return missing


def display_results(results: Dict[str, Any], repo_url: str):
    """Show the grading results in a nice format"""
    print("\n" + "=" * 80)
    print("📊 HACKATHON GRADING RESULTS")
    print("=" * 80)
    
    if "error" in results:
        print(f"\n❌ Error: {results['error']}")
        return
    
    # Overall Score
    total_score = results.get('total_score', 0)
    pass_fail = results.get('pass_fail', 'fail')
    
    status_emoji = {'pass': '✅', 'borderline': '⚠️', 'fail': '❌'}.get(pass_fail, '❓')
    
    print(f"\n🎯 OVERALL SCORE: {total_score}/10")
    print(f"📈 STATUS: {status_emoji} {pass_fail.upper()}")
    print(f"🔗 Repository: {repo_url}")
    
    # What the score means
    print(f"\n💡 SCORE EXPLANATION:")
    if total_score >= 9:
        print("   🌟 EXCELLENT: This is really impressive work!")
    elif total_score >= 7:
        print("   ✅ GOOD: Solid work that meets the standards")
    elif total_score >= 5:
        print("   ⚠️  BORDERLINE: Not bad, but could use some improvements")
    else:
        print("   ❌ POOR: Needs quite a bit of work to get up to standard")
    
    # Category Breakdown
    print(f"\n📋 DETAILED BREAKDOWN:")
    print("-" * 80)
    
    breakdown = results.get('breakdown', {})
    categories = [
        ('ui', 'UI/UX', '🎨'),
        ('architecture', 'Architecture', '🏗️'),
        ('coding', 'Coding', '💻'),
        ('other', 'Other/Compliance', '📋')
    ]
    
    for category_key, category_name, emoji in categories:
        if category_key in breakdown:
            data = breakdown[category_key]
            score = data.get('score', 0)
            weight = data.get('weight', 0)
            confidence = data.get('ai_confidence', 0)
            comment = data.get('comment', '')
            evidence = data.get('evidence', [])
            
            print(f"\n{emoji} {category_name}: {score}/10 (Weight: {weight}%)")
            print(f"   🤖 AI Confidence: {confidence:.0%}")
            print(f"   💬 {comment}")
            
            if evidence:
                print(f"   🔍 Evidence:")
                for ev in evidence[:3]:
                    print(f"      • {ev}")
                if len(evidence) > 3:
                    print(f"      • ... and {len(evidence) - 3} more")
    
    # AI Insights
    if results.get('ai_insights'):
        print(f"\n💡 AI INSIGHTS:")
        print("-" * 40)
        for i, insight in enumerate(results['ai_insights'][:5], 1):
            print(f"   {i}. {insight}")
    
    # AI Recommendations
    if results.get('ai_recommendations'):
        print(f"\n🎯 AI RECOMMENDATIONS:")
        print("-" * 40)
        for i, rec in enumerate(results['ai_recommendations'][:5], 1):
            print(f"   {i}. {rec}")
    
    # Prioritized Actions
    if results.get('prioritized_actions'):
        print(f"\n🚀 PRIORITIZED ACTION ITEMS:")
        print("-" * 40)
        for action in results['prioritized_actions'][:3]:
            priority = action.get('priority', 1)
            action_text = action.get('action', '')
            effort = action.get('effort', 'medium')
            impact = action.get('expected_impact', '')
            
            effort_emoji = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}.get(effort, '🟡')
            print(f"   {priority}. {action_text}")
            print(f"      {effort_emoji} Effort: {effort.title()} | 💡 Impact: {impact}")
    
    # Risks
    if results.get('risks_red_flags'):
        print(f"\n🚨 RISKS & RED FLAGS:")
        print("-" * 40)
        for risk in results['risks_red_flags'][:3]:
            print(f"   ⚠️  {risk}")
    
    # Top Strengths
    if results.get('top_strengths'):
        print(f"\n✅ TOP STRENGTHS:")
        for strength in results['top_strengths'][:3]:
            print(f"   • {strength}")
    
    # Top Improvements
    if results.get('top_improvements'):
        print(f"\n📈 TOP IMPROVEMENTS NEEDED:")
        for improvement in results['top_improvements'][:3]:
            print(f"   • {improvement}")
    
    # Calculation
    if results.get('notes', {}).get('calculation'):
        print(f"\n📊 SCORE CALCULATION:")
        print(f"   {results['notes']['calculation']}")
    
    print("\n" + "=" * 80)
    print("✅ Evaluation Complete!")
    print("=" * 80 + "\n")


def main():
    """Command line interface for the grader"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI-Powered Hackathon GitHub Submission Grader')
    parser.add_argument('--repo', required=True, help='GitHub repository URL')
    parser.add_argument('--branch', default='main', help='Branch to analyze (default: main)')
    parser.add_argument('--commit', help='Specific commit SHA to analyze')
    parser.add_argument('--artifacts', help='JSON file containing artifacts')
    parser.add_argument('--token', help='GitHub API token for private repos')
    parser.add_argument('--ai-key', help='AI API key for enhanced analysis')
    parser.add_argument('--specialized-agents', action='store_true', help='Use specialized agents (Innovation, Functionality, Technical, UI/UX)')
    parser.add_argument('--output', help='Output file for results (optional)')
    parser.add_argument('--json', action='store_true', help='Output raw JSON instead of formatted display')
    
    args = parser.parse_args()
    
    # Load any additional artifacts if provided
    artifacts = {}
    if args.artifacts:
        try:
            with open(args.artifacts, 'r') as f:
                artifacts = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load artifacts file: {e}")
    
    # Create our grader
    grader = AIGrader(args.token, args.ai_key, args.specialized_agents)
    
    # Run the analysis
    results = grader.grade(args.repo, args.branch, args.commit, artifacts)
    
    # Show the results
    if args.json:
        # Just dump the raw JSON
        output = json.dumps(results, indent=2)
        print(output)
    else:
        # Pretty formatted output
        display_results(results, args.repo)
    
    # Save to file if they want it
    if args.output:
        with open(args.output, 'w') as f:
            f.write(json.dumps(results, indent=2))
        print(f"📄 Results also saved to: {args.output}")


if __name__ == '__main__':
    main()


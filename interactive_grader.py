#!/usr/bin/env python3
"""
Interactive Hackathon Grader

A simple command-line interface that walks you through
analyzing a GitHub repository step by step.
"""

import json
import sys
import time
from typing import Dict, List, Any, Optional
from ai_grader import AIGrader


class InteractiveGrader:
    """Handles the interactive grading workflow"""
    
    def __init__(self, github_token: Optional[str] = None, ai_api_key: Optional[str] = None):
        # Set up our grader
        self.grader = AIGrader(github_token=github_token, ai_api_key=ai_api_key)
    
    def run_workflow(self):
        """Main workflow execution"""
        self._print_welcome()
        
        # Step 1: Get GitHub URL
        repo_url = self._get_github_url()
        if not repo_url:
            print("❌ No URL provided. Exiting...")
            return
        
        # Step 2: Get additional options
        branch, artifacts = self._get_additional_options()
        
        # Step 3: Run evaluations
        print(f"\n🚀 Starting evaluation of {repo_url}")
        results = self._run_evaluations(repo_url, branch, artifacts)
        
        # Step 4: Display results
        self._display_results(results)
    
    def _print_welcome(self):
        """Print welcome message and instructions"""
        print("=" * 80)
        print("🤖 AI-POWERED HACKATHON SUBMISSION GRADER")
        print("=" * 80)
        print()
        print("This tool uses AI AGENTS to analyze your GitHub repository:")
        print("• 🤖 AI-driven comprehensive grading across all categories")
        print("• 🧠 Intelligent pattern recognition and analysis")
        print("• 💡 AI-generated explanations for each score")
        print("• 🎯 Smart recommendations for improvement")
        print("• 📈 Continuous learning from previous evaluations")
        print()
        print("Categories evaluated by AI agents:")
        print("• UI/UX (10%) - Design, usability, accessibility")
        print("• Architecture (40%) - Structure, patterns, scalability")
        print("• Coding (30%) - Quality, tests, standards")
        print("• Other (20%) - Documentation, security, compliance")
        print()
    
    def _get_github_url(self) -> Optional[str]:
        """Get GitHub URL from user"""
        print("📋 STEP 1: Repository Information")
        print("-" * 40)
        
        while True:
            url = input("🔗 Enter GitHub repository URL: ").strip()
            
            if not url:
                print("❌ Please enter a valid GitHub URL")
                continue
            
            # Validate GitHub URL
            if not self._validate_github_url(url):
                print("❌ Invalid GitHub URL format. Please use: https://github.com/username/repository")
                continue
            
            # Confirm URL
            print(f"\n✅ Repository: {url}")
            confirm = input("Is this correct? (y/n): ").strip().lower()
            
            if confirm in ['y', 'yes', '']:
                return url
            elif confirm in ['n', 'no']:
                continue
            else:
                print("Please enter 'y' or 'n'")
    
    def _validate_github_url(self, url: str) -> bool:
        """Validate GitHub URL format"""
        return (url.startswith('https://github.com/') and 
                len(url.split('/')) >= 5 and
                not url.endswith('/'))
    
    def _get_additional_options(self) -> tuple[str, Dict[str, Any]]:
        """Get additional options from user"""
        print("\n📋 STEP 2: Additional Options")
        print("-" * 40)
        
        # Get branch
        branch = input("🌿 Branch to analyze (default: main): ").strip()
        if not branch:
            branch = "main"
        
        # Ask about artifacts
        print("\n📁 Do you have additional artifacts to provide?")
        print("   (test results, screenshots, CI configs, etc.)")
        has_artifacts = input("   Enter 'y' for yes, 'n' for no (default: n): ").strip().lower()
        
        artifacts = {}
        if has_artifacts in ['y', 'yes']:
            artifacts = self._get_artifacts()
        
        return branch, artifacts
    
    def _get_artifacts(self) -> Dict[str, Any]:
        """Get artifacts from user"""
        artifacts = {}
        
        print("\n📄 Artifacts Input:")
        print("   You can provide the following (press Enter to skip):")
        
        # README text
        readme = input("   📖 README content (paste or press Enter to skip): ").strip()
        if readme:
            artifacts['readme_text'] = readme
        
        # Test results
        test_results = input("   🧪 Test results (paste or press Enter to skip): ").strip()
        if test_results:
            artifacts['test_results'] = test_results
        
        # Lint results
        lint_results = input("   🔍 Lint results (paste or press Enter to skip): ").strip()
        if lint_results:
            artifacts['lint_results'] = lint_results
        
        # Demo/screenshots
        demo = input("   🖼️  Demo URL or screenshot paths (press Enter to skip): ").strip()
        if demo:
            artifacts['screenshots_or_demo'] = demo
        
        # CI config
        ci_config = input("   ⚙️  CI configuration present? (y/n): ").strip().lower()
        if ci_config in ['y', 'yes']:
            artifacts['ci_config_present'] = True
        
        # License
        license_present = input("   📜 License file present? (y/n): ").strip().lower()
        if license_present in ['y', 'yes']:
            artifacts['license_text_present'] = True
        
        return artifacts
    
    def _run_evaluations(self, repo_url: str, branch: str, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Run all evaluations with progress indicators"""
        print(f"\n🔍 STEP 3: Running AI Agent Evaluations")
        print("-" * 40)
        
        # Progress indicators
        steps = [
            "📡 Fetching repository data from GitHub...",
            "📦 Preparing context for AI agents...",
            "🤖 Code Analysis Agent - Analyzing code quality...",
            "🏗️  Architecture Agent - Evaluating design patterns...",
            "🎨 UI/UX Agent - Assessing interface quality...",
            "🔒 Security Agent - Checking security practices...",
            "🧠 Learning Agent - Applying learned patterns...",
            "📊 Calculating AI-powered scores...",
            "📝 Generating AI feedback and recommendations..."
        ]
        
        for i, step in enumerate(steps, 1):
            print(f"   {i}/{len(steps)} {step}")
            time.sleep(0.3)  # Simulate processing time
        
        # Run actual evaluation using AI agents
        try:
            results = self.grader.grade(repo_url, branch, artifacts=artifacts)
            
            print("   ✅ AI evaluation completed successfully!")
            return results
            
        except Exception as e:
            print(f"   ❌ AI evaluation failed: {str(e)}")
            return {
                "error": str(e),
                "total_score": 0,
                "breakdown": {},
                "pass_fail": "fail",
                "ai_powered": True
            }
    
    def _display_results(self, results: Dict[str, Any]):
        """Display comprehensive results with explanations"""
        print(f"\n📊 STEP 4: Results & Analysis")
        print("=" * 80)
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        
        # Overall score and status
        total_score = results.get('total_score', 0)
        pass_fail = results.get('pass_fail', 'fail')
        
        print(f"\n🎯 OVERALL SCORE: {total_score}/10")
        print(f"📈 STATUS: {self._get_status_emoji(pass_fail)} {pass_fail.upper()}")
        
        # Score explanation
        self._explain_overall_score(total_score, pass_fail)
        
        # Detailed breakdown
        self._display_detailed_breakdown(results)
        
        # AI insights (if available)
        if results.get('ai_enhanced') and results.get('ai_insights'):
            self._display_ai_insights(results)
        
        # Recommendations
        self._display_recommendations(results)
        
        # Next steps
        self._display_next_steps(results)
    
    def _get_status_emoji(self, status: str) -> str:
        """Get emoji for status"""
        status_emojis = {
            'pass': '✅',
            'borderline': '⚠️',
            'fail': '❌'
        }
        return status_emojis.get(status, '❓')
    
    def _explain_overall_score(self, score: int, status: str):
        """Explain the overall score"""
        print(f"\n💡 SCORE EXPLANATION:")
        
        if score >= 9:
            print("   🌟 EXCELLENT: Outstanding submission with exceptional quality across all areas")
        elif score >= 8:
            print("   🏆 VERY GOOD: High-quality submission with minor areas for improvement")
        elif score >= 7:
            print("   ✅ GOOD: Solid submission that meets most quality standards")
        elif score >= 6:
            print("   👍 FAIR: Decent submission with several areas needing improvement")
        elif score >= 5:
            print("   ⚠️  BORDERLINE: Below average submission requiring significant improvements")
        else:
            print("   ❌ POOR: Low-quality submission needing major improvements")
        
        print(f"   📊 Score breakdown: UI(10%) + Architecture(40%) + Coding(30%) + Other(20%)")
    
    def _display_detailed_breakdown(self, results: Dict[str, Any]):
        """Display detailed category breakdown"""
        print(f"\n📋 DETAILED BREAKDOWN:")
        print("-" * 50)
        
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
                comment = data.get('comment', 'No comment available')
                evidence = data.get('evidence', [])
                
                print(f"\n{emoji} {category_name}: {score}/10")
                print(f"   💬 {comment}")
                
                if evidence:
                    print(f"   🔍 Evidence:")
                    for evidence_item in evidence[:3]:  # Show top 3
                        print(f"      • {evidence_item}")
                    if len(evidence) > 3:
                        print(f"      • ... and {len(evidence) - 3} more")
    
    def _display_ai_insights(self, results: Dict[str, Any]):
        """Display AI insights"""
        print(f"\n🤖 AI INSIGHTS:")
        print("-" * 30)
        
        insights = results.get('ai_insights', [])
        if insights:
            for i, insight in enumerate(insights[:5], 1):
                print(f"   {i}. {insight}")
        else:
            print("   No AI insights available")
    
    def _display_recommendations(self, results: Dict[str, Any]):
        """Display recommendations"""
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 30)
        
        # Top improvements
        top_improvements = results.get('top_improvements', [])
        if top_improvements:
            print("   🎯 Top Areas for Improvement:")
            for i, improvement in enumerate(top_improvements[:3], 1):
                print(f"      {i}. {improvement}")
        
        # Prioritized actions
        prioritized_actions = results.get('prioritized_actions', [])
        if prioritized_actions:
            print("\n   📋 Prioritized Action Items:")
            for action in prioritized_actions[:3]:
                priority = action.get('priority', 1)
                action_text = action.get('action', '')
                effort = action.get('effort', 'medium')
                impact = action.get('expected_impact', '')
                
                effort_emoji = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}.get(effort, '🟡')
                print(f"      {priority}. {action_text}")
                print(f"         {effort_emoji} Effort: {effort.title()} | 💡 Impact: {impact}")
    
    def _display_next_steps(self, results: Dict[str, Any]):
        """Display next steps"""
        print(f"\n🚀 NEXT STEPS:")
        print("-" * 20)
        
        total_score = results.get('total_score', 0)
        
        if total_score >= 8:
            print("   🎉 Congratulations! Your submission is excellent.")
            print("   📝 Consider documenting your approach for others to learn from.")
        elif total_score >= 6:
            print("   👍 Good work! Focus on the recommended improvements.")
            print("   🔄 Consider iterating based on the feedback provided.")
        else:
            print("   🔧 Focus on the high-priority action items first.")
            print("   📚 Consider reviewing best practices in the weak areas.")
        
        print("\n   💾 Results can be saved to a file for future reference.")
        print("   🔄 Re-run this tool after making improvements to track progress.")


def main():
    """Main entry point"""
    print("🤖 AI-Powered Interactive Hackathon Grader")
    print("=" * 50)
    print()
    print("This grader uses AI AGENTS for ALL evaluations.")
    print("No traditional analysis - pure AI-driven assessment!")
    print()
    
    # Optional: Ask for API keys
    github_token = input("GitHub API token (optional, press Enter to skip): ").strip()
    if not github_token:
        github_token = None
    
    ai_key = input("AI API key (optional, press Enter to skip): ").strip()
    if not ai_key:
        ai_key = None
    
    # Create and run AI-only grader
    grader = InteractiveGrader(github_token=github_token, ai_api_key=ai_key)
    
    try:
        grader.run_workflow()
    except KeyboardInterrupt:
        print("\n\n👋 Grading cancelled by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please try again or contact support if the issue persists.")


if __name__ == '__main__':
    main()

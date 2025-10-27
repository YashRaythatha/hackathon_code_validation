#!/usr/bin/env python3
"""
Web Interface for the Hackathon Grader

Simple web interface that lets users analyze GitHub repos.
Just paste a URL and get instant feedback from our AI agents.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
from ai_grader import AIGrader
from judge_config import JudgeConfig
import threading
import time
from datetime import datetime

# Import core utilities
from core import analysis_cache, config, get_logger, log_analysis_progress
from core import *

app = Flask(__name__)
CORS(app)

# Configure Flask
app.config['SECRET_KEY'] = config.web.secret_key
app.config['MAX_CONTENT_LENGTH'] = config.web.max_content_length

logger = get_logger(__name__)


def combine_mixed_results(specialized_results, original_results, selected_agent_ids):
    """Combine results from both specialized and original agents - only for selected agents"""
    
    # Agent ID to category mapping - CORRECTED MAPPING
    agent_category_mapping = {
        '1': 'Code Analysis',  # Core Analysis
        '2': 'Architecture Analysis',  # Core Analysis
        '3': 'UI/UX Analysis',  # Core Analysis
        '4': 'Security Analysis',  # Core Analysis
        '5': 'Innovation & Creativity',  # Specialized
        '6': 'Functionality & Completeness',  # Specialized
        '7': 'Technical Complexity',  # Specialized
        '8': 'UI/UX Polish',  # Specialized
        '9': 'Learning Agent'  # Learning
    }
    
    # Determine which agents are specialized vs original - CORRECTED CLASSIFICATION
    specialized_agents = ['5', '6', '7', '8']  # Innovation, Functionality, Technical, UI/UX Polish
    original_agents = ['1', '2', '3', '4']  # Code, Architecture, UI/UX, Security
    learning_agent = ['9']
    
    selected_specialized = [aid for aid in selected_agent_ids if aid in specialized_agents]
    selected_original = [aid for aid in selected_agent_ids if aid in original_agents]
    selected_learning = [aid for aid in selected_agent_ids if aid in learning_agent]
    
    # Combine scores and evidence
    combined_breakdown = {}
    total_score = 0
    total_weight = 0
    
    # Process specialized results - only for selected specialized agents
    if 'breakdown' in specialized_results and selected_specialized:
        for category, data in specialized_results['breakdown'].items():
            # Map specialized categories to display names - UPDATED FOR NEW CATEGORIES
            display_name = None
            if category == 'ui_ux_polish':
                display_name = 'UI/UX Polish'
            elif category == 'technical_complexity':
                display_name = 'Technical Complexity'
            elif category == 'functionality':
                display_name = 'Functionality & Completeness'
            elif category == 'innovation':
                display_name = 'Innovation & Creativity'
            elif category == 'code_analysis':
                display_name = 'Code Analysis'
            elif category == 'architecture':
                display_name = 'Architecture Analysis'
            elif category == 'ui':
                display_name = 'UI/UX Analysis'
            elif category == 'security':
                display_name = 'Security Analysis'
            
            # Only include if this agent was selected and we have a valid display name
            if display_name and display_name in [agent_category_mapping[aid] for aid in selected_specialized]:
                combined_breakdown[display_name] = {
                    'score': data['score'],
                    'weight': data['weight'],
                    'evidence': data['evidence'],
                    'comment': data['comment'],
                    'ai_confidence': data.get('ai_confidence', 0.5),
                    'agent_type': 'Specialized'
                }
                
                total_score += data['score'] * data['weight']
                total_weight += data['weight']
    
    # Process original results - only for selected original agents
    if 'breakdown' in original_results and selected_original:
        for category, data in original_results['breakdown'].items():
            # Map original categories to display names
            display_name = None
            if category == 'ui':
                display_name = 'UI/UX'
            elif category == 'architecture':
                display_name = 'Architecture'
            elif category == 'coding':
                display_name = 'Code Analysis'
            elif category == 'other':
                display_name = 'Security'
            
            # Only include if this agent was selected and we have a valid display name
            if display_name and display_name in [agent_category_mapping[aid] for aid in selected_original]:
                combined_breakdown[display_name] = {
                    'score': data['score'],
                    'weight': data['weight'],
                    'evidence': data['evidence'],
                    'comment': data['comment'],
                    'ai_confidence': data.get('ai_confidence', 0.5),
                    'agent_type': 'Original'
                }
                
                total_score += data['score'] * data['weight']
                total_weight += data['weight']
    
    # Calculate final score
    final_score = round(total_score / total_weight) if total_weight > 0 else 0
    
    # Determine pass/fail status
    if final_score >= 7:
        pass_fail = "pass"
    elif final_score >= 5:
        pass_fail = "borderline"
    else:
        pass_fail = "fail"
    
    # Combine insights and recommendations - only from selected agents
    all_insights = []
    all_recommendations = []
    all_risks = []
    
    # Add insights from specialized agents if any were selected
    if selected_specialized:
        all_insights.extend(specialized_results.get('ai_insights', []))
        all_recommendations.extend(specialized_results.get('ai_recommendations', []))
        all_risks.extend(specialized_results.get('ai_risks', []))
    
    # Add insights from original agents if any were selected
    if selected_original:
        all_insights.extend(original_results.get('ai_insights', []))
        all_recommendations.extend(original_results.get('ai_recommendations', []))
        all_risks.extend(original_results.get('ai_risks', []))
    
    # Create combined result
    combined_result = {
        'total_score': final_score,
        'breakdown': combined_breakdown,
        'pass_fail': pass_fail,
        'selected_agents': selected_agent_ids,
        'ai_powered': True,
        'ai_insights': all_insights[:10],
        'ai_recommendations': all_recommendations[:10],
        'ai_risks': all_risks[:10],
        'top_strengths': (specialized_results.get('top_strengths', []) if selected_specialized else []) + (original_results.get('top_strengths', []) if selected_original else []),
        'top_improvements': (specialized_results.get('top_improvements', []) if selected_specialized else []) + (original_results.get('top_improvements', []) if selected_original else []),
        'prioritized_actions': (specialized_results.get('prioritized_actions', []) if selected_specialized else []) + (original_results.get('prioritized_actions', []) if selected_original else []),
        'plagiarism_flag': False,
        'risks_red_flags': all_risks[:5],
        'notes': {
            'assumptions': [
                "Mixed agent analysis performed",
                "Combined specialized and original agent results",
                "Confidence scores indicate AI reliability"
            ],
            'missing_artifacts': specialized_results.get('notes', {}).get('missing_artifacts', []),
            'calculation': f"Combined weighted scores from {len(combined_breakdown)} agent categories",
            'ai_agents_used': len(selected_agent_ids),
            'agent_types': 'Mixed (Specialized + Original)'
        }
    }
    
    return combined_result


@app.route('/')
def index():
    """Show the main page"""
    return render_template('index.html')

@app.route('/judge-config')
def judge_config_page():
    """Judge configuration page"""
    return render_template('judge_config.html')

@app.route('/api/judge-config', methods=['GET', 'POST'])
def judge_config_api():
    """API endpoint for judge configuration"""
    if request.method == 'GET':
        # Get current configuration
        judge_config = JudgeConfig()
        return jsonify({
            'success': True,
            'weights': judge_config.get_weights(),
            'criteria_info': judge_config.get_criteria_info(),
            'summary': judge_config.get_weight_summary()
        })
    
    elif request.method == 'POST':
        # Update configuration
        try:
            data = request.get_json()
            weights = data.get('weights', {})
            
            judge_config = JudgeConfig()
            success, message = judge_config.set_weights(weights)
            
            return jsonify({
                'success': success,
                'message': message,
                'weights': judge_config.get_weights() if success else None
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })


@app.route('/api/analyze', methods=['POST'])
def analyze_repository():
    """Handle repository analysis requests with improved error handling"""
    start_time = time.time()
    
    try:
        data = request.json or {}
        
        # Get the input values (with some safety checks)
        repo_url = (data.get('repo_url') or '').strip()
        branch = (data.get('branch') or 'main').strip()
        github_token = (data.get('github_token') or '').strip()
        selected_agent_ids = data.get('selected_agents', [])
        
        # Clean up the token
        if not github_token:
            github_token = None
        
        # Enhanced validation
        if not repo_url:
            raise ValidationError("Repository URL is required", "repo_url")
        
        if 'github.com' not in repo_url:
            raise RepositoryValidationError(repo_url)
        
        if not selected_agent_ids:
            raise ValidationError("At least one agent must be selected", "selected_agents")
        
        # Check cache first
        cached_result = analysis_cache.get_analysis(repo_url, branch, selected_agent_ids)
        if cached_result:
            logger.info(f"Returning cached result for {repo_url}")
            return jsonify({
                'success': True,
                'results': cached_result,
                'cached': True,
                'cache_hit': True
            })
        
        logger.info(f"Starting analysis for {repo_url} with agents {selected_agent_ids}")
        log_analysis_progress(repo_url, "Starting analysis", 0)
        
        # Determine agent mode and map selected agents
        use_specialized_agents = False
        selected_agents = []
        
        if selected_agent_ids:
            # UNIFIED SYSTEM - All agents are handled the same way
            # Use the unified grader for all agents
            # Initialize grader with judge configuration
            judge_config = JudgeConfig()
            unified_grader = AIGrader(github_token=github_token, use_specialized_agents=True, judge_config=judge_config)
            results = unified_grader.grade(repo_url, branch, selected_agents=selected_agent_ids)
        else:
            # No agents selected - return error
            results = {'error': 'No agents selected'}
        
        # Cache the results
        analysis_cache.set_analysis(repo_url, branch, selected_agent_ids, results)
        
        # Log performance
        duration = time.time() - start_time
        logger.info(f"Analysis completed for {repo_url} in {duration:.2f}s")
        
        return jsonify({
            'success': True,
            'results': results,
            'cached': False,
            'cache_hit': False,
            'duration': duration
        })
        
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': 'validation'
        }), 400
        
    except GitHubAPIError as e:
        logger.error(f"GitHub API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': 'github_api',
            'status_code': getattr(e, 'status_code', None)
        }), 502
        
    except UIExecutionError as e:
        logger.error(f"UI execution error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': 'ui_execution'
        }), 500
        
    except AnalysisError as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': 'analysis'
        }), 500
        
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error during analysis: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred. Please try again.',
            'error_type': 'unexpected'
        }), 500


@app.route('/api/agents', methods=['GET'])
def get_available_agents():
    """Get all available agents in a unified list"""
    try:
        # All agents with their descriptions
        agents = {
            '1': {'id': 'innovation', 'name': 'Innovation & Creativity', 'description': 'Evaluates novelty and creative use of technology'},
            '2': {'id': 'functionality', 'name': 'Functionality & Completeness', 'description': 'Assesses working features and completeness'},
            '3': {'id': 'technical', 'name': 'Technical Complexity', 'description': 'Analyzes technical difficulty and architecture'},
            '4': {'id': 'ui_ux_polish', 'name': 'UI/UX Polish', 'description': 'Evaluates visual design and user experience'},
            '5': {'id': 'code', 'name': 'Code Analysis', 'description': 'Analyzes code quality and patterns'},
            '6': {'id': 'architecture', 'name': 'Architecture', 'description': 'Evaluates design patterns and structure'},
            '7': {'id': 'ui_ux', 'name': 'UI/UX', 'description': 'Assesses interface quality and usability'},
            '8': {'id': 'security', 'name': 'Security', 'description': 'Checks security practices and compliance'},
            '9': {'id': 'learning', 'name': 'Learning Agent', 'description': 'Applies patterns from past analyses'}
        }
        
        return jsonify({
            'success': True,
            'agents': agents
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'AI Grader is running'})


@app.route('/api/learning-stats', methods=['GET'])
def get_learning_stats():
    """Get machine learning statistics"""
    try:
        from ai_agents import AgentOrchestrator
        orchestrator = AgentOrchestrator()
        learning_agent = orchestrator.agents['learning']
        stats = learning_agent.get_learning_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    print("🚀 Starting AI-Powered Hackathon Grader Web Interface...")
    print(f"📍 Access the application at: http://localhost:{port}")
    print("🤖 AI agents ready for evaluation!")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)


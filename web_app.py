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
import threading

app = Flask(__name__)
CORS(app)

# Keep track of recent analyses (simple cache)
analysis_cache = {}


@app.route('/')
def index():
    """Show the main page"""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_repository():
    """Handle repository analysis requests"""
    try:
        data = request.json or {}
        
        # Get the input values (with some safety checks)
        repo_url = (data.get('repo_url') or '').strip()
        branch = (data.get('branch') or 'main').strip()
        github_token = (data.get('github_token') or '').strip()
        
        # Clean up the token
        if not github_token:
            github_token = None
        
        # Basic validation
        if not repo_url:
            return jsonify({'error': 'Repository URL is required'}), 400
        
        if 'github.com' not in repo_url:
            return jsonify({'error': 'Please provide a valid GitHub URL'}), 400
        
        # Create our grader and run the analysis
        grader = AIGrader(github_token=github_token)
        results = grader.grade(repo_url, branch)
        
        # Store the results for later (in case someone wants to see them again)
        cache_key = f"{repo_url}:{branch}"
        analysis_cache[cache_key] = results
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        # Log the error for debugging
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during analysis: {error_details}")
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


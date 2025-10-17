#!/usr/bin/env python3
"""
Learning Statistics Viewer

Quick tool to see what our AI agents have learned.
Shows patterns, statistics, and learning progress.
"""

import json
import os
from datetime import datetime
from ai_agents import AgentOrchestrator

def display_learning_stats():
    """Show what our AI has learned so far"""
    print("🧠 AI Learning Statistics Dashboard")
    print("=" * 50)
    
    try:
        # Get access to our learning agent
        orchestrator = AgentOrchestrator()
        learning_agent = orchestrator.agents['learning']
        
        # Pull the current stats
        stats = learning_agent.get_learning_stats()
        
        if 'message' in stats:
            print(f"📊 {stats['message']}")
            print("\n💡 The AI will start learning after analyzing some repositories.")
            return
        
        # Display basic statistics
        print(f"📈 Total Analyses: {stats.get('total_analyses', 0)}")
        print(f"🎯 Patterns Learned: {stats.get('learned_patterns', 0)}")
        print(f"📊 Average Score: {stats.get('average_score', 0):.1f}/10")
        print(f"🎪 Confidence Trend: {stats.get('confidence_trend', 0):.1%}")
        print(f"🔧 Technology Patterns: {stats.get('technology_patterns', 0)}")
        
        # Display most common technologies
        if stats.get('most_common_tech'):
            print(f"\n🔧 Most Common Technologies:")
            for tech, count in stats['most_common_tech']:
                print(f"   • {tech}: {count} times")
        
        # Display top learned patterns
        if stats.get('pattern_weights'):
            print(f"\n🎯 Top Learned Patterns:")
            sorted_patterns = sorted(stats['pattern_weights'].items(), 
                                   key=lambda x: abs(x[1]), reverse=True)
            for pattern, weight in sorted_patterns[:10]:
                sign = "+" if weight > 0 else ""
                print(f"   • {pattern}: {sign}{weight:.3f}")
        
        # Display learning insights
        print(f"\n💡 Learning Insights:")
        print(f"   • The AI has analyzed {stats.get('total_analyses', 0)} repositories")
        print(f"   • Learned {stats.get('learned_patterns', 0)} patterns from successful projects")
        print(f"   • Average confidence: {stats.get('confidence_trend', 0):.1%}")
        
        if stats.get('total_analyses', 0) > 0:
            print(f"   • Learning is active and improving over time")
        else:
            print(f"   • No learning data available yet")
        
        # Display data file info
        data_file = "learning_data.json"
        if os.path.exists(data_file):
            file_size = os.path.getsize(data_file)
            mod_time = datetime.fromtimestamp(os.path.getmtime(data_file))
            print(f"\n📁 Data File: {data_file}")
            print(f"   • Size: {file_size:,} bytes")
            print(f"   • Last Updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Error loading learning statistics: {e}")
        print("💡 Make sure the AI agents have been used at least once.")

def reset_learning_data():
    """Reset learning data (for testing purposes)"""
    data_file = "learning_data.json"
    if os.path.exists(data_file):
        os.remove(data_file)
        print("🗑️ Learning data reset successfully")
    else:
        print("📊 No learning data to reset")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='View AI Learning Statistics')
    parser.add_argument('--reset', action='store_true', 
                       help='Reset learning data (for testing)')
    
    args = parser.parse_args()
    
    if args.reset:
        reset_learning_data()
    else:
        display_learning_stats()

if __name__ == '__main__':
    main()

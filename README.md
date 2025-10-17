# 🤖 AI-Powered Hackathon Grader

A pure AI-driven tool for grading hackathon submissions. Uses **ONLY AI AGENTS** for all evaluations!

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Option 1: Web Interface (Recommended - User Friendly!)
```bash
python web_app.py
```
Then open your browser and go to: **http://localhost:5000**

Simply paste your GitHub URL and click "Analyze Repository" - that's it! 🎉

### Option 2: Command Line
```bash
# Analyze any GitHub repository
python ai_grader.py --repo https://github.com/username/repository

# Specify branch
python ai_grader.py --repo https://github.com/username/repo --branch main

# With GitHub token (for private repos)
python ai_grader.py --repo https://github.com/username/repo --token ghp_xxxxx
```

## 📁 Files

### **Core Files:**
- **`web_app.py`** - Web interface (RECOMMENDED - User Friendly!)
- **`ai_grader.py`** - Main AI grading engine
- **`ai_agents.py`** - 5 specialized AI agents with ML capabilities
- **`github_analyzer.py`** - GitHub API handler
- **`interactive_grader.py`** - Interactive CLI workflow

### **Supporting Files:**
- **`templates/index.html`** - Web UI template
- **`requirements.txt`** - Dependencies
- **`sample_artifacts.json`** - Example artifacts
- **`learning_stats.py`** - ML statistics viewer
- **`README.md`** - Complete documentation

## 🤖 AI Agents

The system uses 5 specialized AI agents:

1. **Code Analysis Agent** (30%) - Code quality, patterns, tests
2. **Architecture Agent** (40%) - Design patterns, scalability
3. **UI/UX Agent** (10%) - Interface quality, accessibility
4. **Security Agent** (20%) - Security practices, vulnerabilities
5. **Learning Agent** - Continuous improvement

## 📊 Scoring

- **9-10**: Excellent
- **7-8**: Very Good (PASS)
- **5-6**: Good (BORDERLINE)
- **3-4**: Fair
- **0-2**: Poor (FAIL)

**Formula:**
```
Final Score = UI(10%) + Architecture(40%) + Coding(30%) + Other(20%)
```

## 🎯 What You Get

The AI grader provides:
- ✅ Overall score (0-10)
- ✅ Detailed category breakdown
- ✅ AI confidence scores
- ✅ Evidence for each score
- ✅ AI-generated insights
- ✅ Actionable recommendations
- ✅ Prioritized action items
- ✅ Risk assessment

## 📝 Example Output

```
================================================================================
📊 HACKATHON GRADING RESULTS
================================================================================

🎯 OVERALL SCORE: 8/10
📈 STATUS: ✅ PASS
🔗 Repository: https://github.com/username/repo

💡 SCORE EXPLANATION:
   ✅ GOOD: Solid submission that meets quality standards

📋 DETAILED BREAKDOWN:
--------------------------------------------------------------------------------

🎨 UI/UX: 7/10 (Weight: 10%)
   🤖 AI Confidence: 85%
   💬 Good: AI agents found solid UI/UX implementation

🏗️ Architecture: 8/10 (Weight: 40%)
   🤖 AI Confidence: 92%
   💬 Excellent: AI detected outstanding architecture

💻 Coding: 9/10 (Weight: 30%)
   🤖 AI Confidence: 100%
   💬 Excellent: High code quality detected

📋 Other/Compliance: 6/10 (Weight: 20%)
   🤖 AI Confidence: 75%
   💬 Good: Security and compliance practices present

💡 AI INSIGHTS:
   1. Strong test coverage detected
   2. Scalable design patterns implemented
   3. Proper secrets management found

🎯 AI RECOMMENDATIONS:
   1. Add comprehensive test suite
   2. Implement accessibility testing
   3. Add API documentation

🚀 PRIORITIZED ACTION ITEMS:
   1. Add unit tests for critical functions
      🟡 Effort: Medium | 💡 Impact: Increase code reliability
```

## 🔧 Advanced Options

### Save Results to File
```bash
python ai_grader.py --repo <url> --output results.json
```

### Get Raw JSON Output
```bash
python ai_grader.py --repo <url> --json
```

### With Artifacts
```bash
python ai_grader.py --repo <url> --artifacts artifacts.json
```

### Interactive Mode
```bash
python interactive_grader.py
```

## 📦 Artifacts (Optional)

You can provide additional information to enhance AI analysis:

```json
{
  "test_results": "All tests passed (15/15)",
  "lint_results": "ESLint: No issues found",
  "screenshots_or_demo": "https://demo.example.com",
  "ci_config_present": true,
  "license_text_present": true,
  "accessibility_notes": "WCAG 2.1 AA compliant",
  "perf_notes": "Lighthouse score: 95/100"
}
```

See `sample_artifacts.json` for a complete example.

## 🎓 How It Works

1. **Fetch Repository Data** - Gets file tree and README from GitHub
2. **AI Agents Analyze** - 5 specialized agents evaluate all aspects
3. **Calculate Scores** - Weighted scoring with confidence levels
4. **Generate Feedback** - AI creates insights and recommendations
5. **Display Results** - Beautiful formatted output in terminal

## 💡 Key Features

✅ **Pure AI Evaluation** - No rule-based analysis
✅ **5 Specialized Agents** - Each expert in their domain
✅ **Confidence Scoring** - Know how reliable each assessment is
✅ **Continuous Learning** - System improves over time
✅ **Pattern Recognition** - Detects code patterns and anti-patterns
✅ **Detailed Explanations** - AI explains WHY each score was given
✅ **Actionable Feedback** - Prioritized recommendations

## 🆘 Troubleshooting

**Issue**: Branch not found (404 error)
```bash
python ai_grader.py --repo <url> --branch master
```

**Issue**: Rate limit exceeded
```bash
python ai_grader.py --repo <url> --token ghp_xxxxx
```

**Issue**: Unicode/emoji errors on Windows
```powershell
$env:PYTHONIOENCODING="utf-8"
python ai_grader.py --repo <url>
```

## 📄 Dependencies

- **requests** >= 2.25.0 - GitHub API calls
- **numpy** >= 1.21.0 - AI calculations
- **scikit-learn** >= 1.0.0 - Learning agent

## 📚 Examples

### Analyze a Public Repository
```bash
python ai_grader.py --repo https://github.com/facebook/react --branch main
```

### Analyze Your Repository
```bash
python ai_grader.py --repo https://github.com/YourUsername/your-repo
```

### Batch Analysis
```bash
# Analyze multiple repositories
python ai_grader.py --repo https://github.com/user/repo1
python ai_grader.py --repo https://github.com/user/repo2
python ai_grader.py --repo https://github.com/user/repo3
```

## 🧠 Machine Learning Features

### **Advanced Learning Agent**
The system now includes a sophisticated Learning Agent that continuously improves through machine learning:

#### **Pattern Recognition**
- **Technology Stack Learning**: Identifies successful patterns in different tech stacks
- **Architecture Pattern Recognition**: Learns from successful architectural decisions
- **Quality Indicator Learning**: Adapts scoring based on successful project patterns
- **Risk Pattern Detection**: Identifies common failure patterns to avoid

#### **Adaptive Scoring**
- **Historical Data Analysis**: Uses past evaluations to improve current scoring
- **Confidence-Based Learning**: Higher confidence in patterns seen more frequently
- **Technology-Specific Scoring**: Different scoring criteria for different tech stacks
- **Continuous Model Updates**: Learning model updates after each analysis

#### **Learning Dashboard**
Access the learning dashboard at: `http://localhost:5000` (after starting web app)
- **Real-time Statistics**: View learning progress and pattern recognition
- **Technology Trends**: See most common technologies and their success rates
- **Pattern Weights**: View learned positive and negative patterns
- **Confidence Metrics**: Track AI confidence improvements over time

#### **Learning Data Persistence**
- **Automatic Data Storage**: All analyses are stored for learning
- **Pattern Weight Updates**: Positive/negative patterns are weighted based on success
- **Technology Pattern Tracking**: Success rates by technology stack
- **Rolling Data Window**: Keeps recent data while maintaining performance

### **View Learning Statistics**
```bash
# View current learning statistics
python learning_stats.py

# Reset learning data (for testing)
python learning_stats.py --reset
```

## 🎯 Summary

This AI-powered hackathon grader:
- ✅ Uses ONLY AI agents for evaluation
- ✅ Follows exact workflow: URL → AI evaluates → Results
- ✅ Provides detailed explanations for every score
- ✅ **Continuously learns and improves through ML**
- ✅ Gives confidence scores for reliability
- ✅ Generates smart, actionable recommendations
- ✅ **Adapts scoring based on learned patterns**

**Built with ❤️ for the hackathon community**

---

## 📞 Support

For issues or questions, check:
1. This README for common solutions
2. The example output format above
3. `sample_artifacts.json` for testing

## 📄 License

MIT License - Free to use and modify
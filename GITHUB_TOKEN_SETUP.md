# GitHub Token Setup Instructions

## Issue: "0/10" scores with "0 files found"

The system is showing "0/10" scores because it's hitting GitHub's API rate limits. Without authentication, GitHub API has a rate limit of only 60 requests/hour.

## Solution: Add GitHub Token

### Step 1: Create GitHub Token
1. Go to https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name like "Hackathon Grader"
4. Select the "repo" scope (full repository access)
5. Click "Generate token"
6. Copy the token (you won't see it again!)

### Step 2: Set Environment Variable
Create a `.env` file in the project root with:
```
GITHUB_TOKEN=your_token_here
```

### Step 3: Restart the Application
The application will automatically use the token and increase the rate limit from 60 to 5000 requests/hour.

## Benefits
- **Rate limit**: 60 → 5000 requests/hour
- **Access**: Full repository data
- **Reliability**: No more "0 files found" errors

## Alternative: Wait for Rate Limit Reset
If you don't want to set up a token, the rate limit resets every hour. You can wait and try again later.

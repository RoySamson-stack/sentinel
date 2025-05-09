# main.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Optional
from pydantic import BaseModel
import pandas as pd
from datetime import datetime, timedelta

from reddit_scraper import RedditScraper
from sentiment_analyzer import SentimentAnalyzer

app = FastAPI(title="Reddit Sentiment Analysis API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
scraper = RedditScraper()
analyzer = SentimentAnalyzer(use_transformers=False)  # Set to True if you want to use transformers

# Pydantic models for response types
class Topic(BaseModel):
    topic: str
    count: int

class SentimentData(BaseModel):
    compound: float
    positive: float
    neutral: float
    negative: float

class Post(BaseModel):
    id: str
    title: str
    text: Optional[str]
    score: int
    num_comments: int
    created_utc: str
    url: str
    author: str
    subreddit: str
    sentiment_category: str

class PostAnalysisResponse(BaseModel):
    posts: List[Post]
    topics: List[Topic]
    overall_sentiment: SentimentData

# API endpoints
@app.get("/")
def read_root():
    return {"message": "Reddit Sentiment Analysis API"}

@app.get("/trending")
def get_trending_topics(limit: int = Query(10, ge=1, le=50)):
    """Get trending subreddits"""
    try:
        topics = scraper.get_trending_topics(limit=limit)
        return {"trending_subreddits": topics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/subreddit/{subreddit_name}")
def analyze_subreddit(
    subreddit_name: str, 
    limit: int = Query(25, ge=1, le=100)
):
    """Analyze posts from a specific subreddit"""
    try:
        posts = scraper.get_hot_posts(subreddit_name, limit=limit)
        analysis = analyzer.analyze_posts(posts)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
def search_and_analyze(
    query: str,
    limit: int = Query(50, ge=1, le=200)
):
    """Search posts and analyze sentiment"""
    try:
        posts = scraper.search_topics(query, limit=limit)
        analysis = analyzer.analyze_posts(posts)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/post/{post_id}/comments")
def get_post_comments(
    post_id: str,
    limit: int = Query(100, ge=1, le=500)
):
    """Get and analyze comments from a specific post"""
    try:
        comments = scraper.get_post_comments(post_id, limit=limit)
        
        # Add sentiment analysis to each comment
        for comment in comments:
            sentiment = analyzer.analyze_text(comment.get('text', ''))
            comment['sentiment'] = sentiment
            
            # Add sentiment category
            if sentiment['compound'] >= 0.05:
                comment['sentiment_category'] = 'positive'
            elif sentiment['compound'] <= -0.05:
                comment['sentiment_category'] = 'negative'
            else:
                comment['sentiment_category'] = 'neutral'
        
        # Calculate overall sentiment
        overall_sentiment = analyzer.calculate_overall_sentiment(comments)
        
        return {
            "comments": comments,
            "overall_sentiment": overall_sentiment
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
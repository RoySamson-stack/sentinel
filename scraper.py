# reddit_scraper.py
import praw
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

class RedditScraper:
    def __init__(self):
        # Initialize the Reddit API client
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent="sentiment_analysis_dashboard/1.0"
        )
    
    def get_trending_topics(self, limit=10):
        """Get trending subreddits"""
        trending_subreddits = self.reddit.trending_subreddits()
        return list(trending_subreddits)[:limit]
    
    def get_hot_posts(self, subreddit_name, limit=25):
        """Get hot posts from a subreddit"""
        subreddit = self.reddit.subreddit(subreddit_name)
        posts = []
        
        for post in subreddit.hot(limit=limit):
            posts.append({
                'id': post.id,
                'title': post.title,
                'text': post.selftext,
                'score': post.score,
                'num_comments': post.num_comments,
                'created_utc': datetime.fromtimestamp(post.created_utc).isoformat(),
                'url': post.url,
                'author': str(post.author),
                'subreddit': subreddit_name
            })
        
        return posts
    
    def get_post_comments(self, post_id, limit=100):
        """Get comments from a specific post"""
        submission = self.reddit.submission(id=post_id)
        submission.comments.replace_more(limit=0)  # Flatten comment tree
        
        comments = []
        for comment in submission.comments.list()[:limit]:
            comments.append({
                'id': comment.id,
                'text': comment.body,
                'score': comment.score,
                'created_utc': datetime.fromtimestamp(comment.created_utc).isoformat(),
                'author': str(comment.author)
            })
        
        return comments
    
    def search_topics(self, query, limit=100):
        """Search posts by query term"""
        posts = []
        for submission in self.reddit.subreddit("all").search(query, limit=limit):
            posts.append({
                'id': submission.id,
                'title': submission.title,
                'text': submission.selftext,
                'score': submission.score,
                'num_comments': submission.num_comments,
                'created_utc': datetime.fromtimestamp(submission.created_utc).isoformat(),
                'url': submission.url,
                'author': str(submission.author),
                'subreddit': submission.subreddit.display_name
            })
        
        return posts
# sentiment_analyzer.py
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd
from collections import Counter
import re
from transformers import pipeline
import spacy

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

class SentimentAnalyzer:
    def __init__(self, use_transformers=False):
        self.sid = SentimentIntensityAnalyzer()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        
        # Optional: more advanced sentiment model
        self.use_transformers = use_transformers
        if use_transformers:
            self.transformer_model = pipeline("sentiment-analysis")
        
        # For topic extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # If model isn't downloaded yet
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def analyze_text(self, text):
        """Analyze sentiment of a single text"""
        if not text or text.strip() == "":
            return {"compound": 0, "pos": 0, "neu": 0, "neg": 0}
        
        # VADER sentiment analysis
        sentiment = self.sid.polarity_scores(text)
        
        # Optional: Use more advanced transformer model
        if self.use_transformers:
            try:
                result = self.transformer_model(text[:512])[0]  # Truncate to avoid token limit
                sentiment['transformer_label'] = result['label']
                sentiment['transformer_score'] = result['score']
            except Exception as e:
                print(f"Transformer model error: {e}")
        
        return sentiment
    
    def preprocess_text(self, text):
        """Clean text for topic extraction"""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def extract_topics(self, texts, top_n=10):
        """Extract key topics from a collection of texts"""
        all_text = " ".join([self.preprocess_text(text) for text in texts])
        doc = self.nlp(all_text)
        
        # Extract noun phrases and named entities
        topics = []
        for chunk in doc.noun_chunks:
            if chunk.text.lower() not in self.stopwords and len(chunk.text.split()) <= 3:
                topics.append(chunk.text)
        
        for ent in doc.ents:
            if ent.text.lower() not in self.stopwords:
                topics.append(ent.text)
        
        # Count frequencies
        topic_counter = Counter(topics)
        
        # Return top N topics
        return [{"topic": topic, "count": count} 
                for topic, count in topic_counter.most_common(top_n)]
    
    def analyze_posts(self, posts):
        """Analyze a list of posts"""
        if not posts:
            return []
        
        for post in posts:
            # Analyze title sentiment
            title_sentiment = self.analyze_text(post.get('title', ''))
            
            # Analyze content sentiment
            content_sentiment = self.analyze_text(post.get('text', ''))
            
            # Calculate weighted sentiment (title has higher weight)
            compound_sentiment = 0.7 * title_sentiment['compound'] + 0.3 * content_sentiment['compound']
            
            # Add sentiment data
            post['sentiment'] = {
                'title': title_sentiment,
                'content': content_sentiment,
                'compound': compound_sentiment
            }
            
            # Add sentiment category
            if compound_sentiment >= 0.05:
                post['sentiment_category'] = 'positive'
            elif compound_sentiment <= -0.05:
                post['sentiment_category'] = 'negative'
            else:
                post['sentiment_category'] = 'neutral'
        
        # Extract topics from all posts
        all_titles = [post.get('title', '') for post in posts]
        all_texts = [post.get('text', '') for post in posts]
        topics = self.extract_topics(all_titles + all_texts)
        
        return {
            'posts': posts,
            'topics': topics,
            'overall_sentiment': self.calculate_overall_sentiment(posts)
        }
    
    def calculate_overall_sentiment(self, posts):
        """Calculate overall sentiment stats from posts"""
        if not posts:
            return {"compound": 0, "positive": 0, "neutral": 0, "negative": 0}
        
        sentiment_values = [post['sentiment']['compound'] for post in posts]
        sentiment_categories = [post['sentiment_category'] for post in posts]
        
        return {
            "compound": sum(sentiment_values) / len(sentiment_values),
            "positive": sentiment_categories.count('positive') / len(sentiment_categories),
            "neutral": sentiment_categories.count('neutral') / len(sentiment_categories),
            "negative": sentiment_categories.count('negative') / len(sentiment_categories)
        }
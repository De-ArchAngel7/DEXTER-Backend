import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import tweepy
import praw
import asyncio
import aiohttp
import structlog
from datetime import datetime, timedelta
import re
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

logger = structlog.get_logger()

class SentimentAnalyzer:
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.vader_analyzer = SentimentIntensityAnalyzer()
        except:
            self.vader_analyzer = None
            
    def analyze_text(self, text: str) -> Dict[str, Any]:
        try:
            result = self.sentiment_pipeline(text[:512])[0]
            
            sentiment_mapping = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral', 
                'LABEL_2': 'positive'
            }
            
            sentiment = sentiment_mapping.get(result['label'], 'neutral')
            confidence = result['score']
            
            if self.vader_analyzer:
                vader_scores = self.vader_analyzer.polarity_scores(text)
                compound_score = vader_scores['compound']
            else:
                compound_score = 0.0
                
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'compound_score': compound_score,
                'text': text,
                'analyzed_at': datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'compound_score': 0.0,
                'text': text,
                'analyzed_at': datetime.utcnow(),
                'error': str(e)
            }

class TwitterSentimentAnalyzer:
    def __init__(self, bearer_token: str):
        self.bearer_token = bearer_token
        self.client = tweepy.Client(bearer_token=bearer_token)
        
    def search_tweets(self, query: str, max_results: int = 100, lang: str = 'en') -> List[Dict[str, Any]]:
        try:
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                tweet_fields=['created_at', 'public_metrics', 'lang'],
                lang=lang
            )
            
            if not tweets.data:
                return []
                
            return [
                {
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'metrics': tweet.public_metrics,
                    'lang': tweet.lang
                }
                for tweet in tweets.data
            ]
        except Exception as e:
            logger.error(f"Error searching tweets: {e}")
            return []
            
    def get_user_tweets(self, username: str, max_results: int = 100) -> List[Dict[str, Any]]:
        try:
            user = self.client.get_user(username=username)
            if not user.data:
                return []
                
            tweets = self.client.get_users_tweets(
                user.data.id,
                max_results=max_results,
                tweet_fields=['created_at', 'public_metrics']
            )
            
            if not tweets.data:
                return []
                
            return [
                {
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'metrics': tweet.public_metrics
                }
                for tweet in tweets.data
            ]
        except Exception as e:
            logger.error(f"Error getting user tweets: {e}")
            return []

class RedditSentimentAnalyzer:
    def __init__(self, client_id: str, client_secret: str, user_agent: str = "TradingBot/1.0"):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
    def get_subreddit_posts(self, subreddit_name: str, limit: int = 100, time_filter: str = 'day') -> List[Dict[str, Any]]:
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = []
            
            for post in subreddit.top(time_filter=time_filter, limit=limit):
                posts.append({
                    'id': post.id,
                    'title': post.title,
                    'text': post.selftext,
                    'score': post.score,
                    'upvote_ratio': post.upvote_ratio,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'num_comments': post.num_comments,
                    'url': post.url
                })
                
            return posts
        except Exception as e:
            logger.error(f"Error getting subreddit posts: {e}")
            return []
            
    def search_posts(self, query: str, subreddit: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            if subreddit:
                subreddit_obj = self.reddit.subreddit(subreddit)
                search_results = subreddit_obj.search(query, limit=limit)
            else:
                search_results = self.reddit.subreddit('all').search(query, limit=limit)
                
            posts = []
            for post in search_results:
                posts.append({
                    'id': post.id,
                    'title': post.title,
                    'text': post.selftext,
                    'score': post.score,
                    'subreddit': post.subreddit.display_name,
                    'created_utc': datetime.fromtimestamp(post.created_utc)
                })
                
            return posts
        except Exception as e:
            logger.error(f"Error searching posts: {e}")
            return []

class TelegramSentimentAnalyzer:
    def __init__(self, bot_token: str):
        self.bot_token = bot_token
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
    async def get_channel_messages(self, channel_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/getUpdates"
                params = {
                    'chat_id': channel_id,
                    'limit': limit
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data['ok']:
                            return [
                                {
                                    'id': update['message']['message_id'],
                                    'text': update['message'].get('text', ''),
                                    'date': datetime.fromtimestamp(update['message']['date']),
                                    'chat_id': update['message']['chat']['id']
                                }
                                for update in data['result']
                                if 'message' in update and 'text' in update['message']
                            ]
            return []
        except Exception as e:
            logger.error(f"Error getting Telegram messages: {e}")
            return []

class SentimentAggregator:
    def __init__(self, sentiment_analyzer: SentimentAnalyzer):
        self.sentiment_analyzer = sentiment_analyzer
        
    def aggregate_sentiment(self, texts: List[str]) -> Dict[str, Any]:
        if not texts:
            return {
                'overall_sentiment': 'neutral',
                'confidence': 0.0,
                'compound_score': 0.0,
                'sentiment_distribution': {},
                'total_texts': 0
            }
            
        results = []
        for text in texts:
            result = self.sentiment_analyzer.analyze_text(text)
            results.append(result)
            
        sentiment_counts = {}
        total_confidence = 0
        total_compound = 0
        
        for result in results:
            sentiment = result['sentiment']
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            total_confidence += result['confidence']
            total_compound += result['compound_score']
            
        avg_confidence = total_confidence / len(results)
        avg_compound = total_compound / len(results)
        
        overall_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
        
        return {
            'overall_sentiment': overall_sentiment,
            'confidence': avg_confidence,
            'compound_score': avg_compound,
            'sentiment_distribution': sentiment_counts,
            'total_texts': len(texts),
            'individual_results': results
        }
        
    def get_crypto_sentiment(self, symbol: str, sources: List[str] = None) -> Dict[str, Any]:
        if sources is None:
            sources = ['twitter', 'reddit', 'telegram']
            
        all_texts = []
        source_results = {}
        
        for source in sources:
            if source == 'twitter':
                analyzer = TwitterSentimentAnalyzer("your_bearer_token")
                tweets = analyzer.search_tweets(f"#{symbol} OR ${symbol}")
                texts = [tweet['text'] for tweet in tweets]
                source_results[source] = texts
                all_texts.extend(texts)
            elif source == 'reddit':
                analyzer = RedditSentimentAnalyzer("your_client_id", "your_client_secret")
                posts = analyzer.search_posts(symbol, limit=50)
                texts = [post['title'] + ' ' + post['text'] for post in posts]
                source_results[source] = texts
                all_texts.extend(texts)
                
        overall_sentiment = self.aggregate_sentiment(all_texts)
        overall_sentiment['symbol'] = symbol
        overall_sentiment['sources'] = list(source_results.keys())
        overall_sentiment['source_breakdown'] = {
            source: len(texts) for source, texts in source_results.items()
        }
        
        return overall_sentiment

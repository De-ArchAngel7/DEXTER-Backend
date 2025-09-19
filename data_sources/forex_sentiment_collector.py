#!/usr/bin/env python3
"""
📰 DEXTER Forex Sentiment Data Collector
============================================================
Collects forex sentiment data from multiple sources:
- Financial news APIs
- Reddit/Twitter forex discussions
- Kaggle datasets
- GDELT news database
"""

import pandas as pd
import numpy as np
import requests
import json
import os
import structlog
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger()

class ForexSentimentCollector:
    """
    📰 Forex Sentiment Data Collector
    Collects and processes forex sentiment data for FinBERT training
    """
    
    def __init__(self, data_dir: str = "data/forex/sentiment"):
        self.data_dir = data_dir
        self.ensure_data_directory()
        
        # Currency pairs for sentiment analysis
        self.currency_pairs = [
            "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
            "AUD/USD", "USD/CAD", "NZD/USD", "EUR/JPY",
            "GBP/JPY", "EUR/GBP", "AUD/JPY", "EUR/AUD"
        ]
        
        # Forex-related keywords for sentiment analysis
        self.forex_keywords = [
            'forex', 'fx', 'currency', 'exchange rate', 'pip', 'spread',
            'central bank', 'interest rate', 'inflation', 'GDP', 'unemployment',
            'FOMC', 'ECB', 'BOE', 'BOJ', 'RBA', 'BOC', 'RBNZ',
            'bullish', 'bearish', 'support', 'resistance', 'breakout',
            'trend', 'momentum', 'volatility', 'liquidity'
        ]
        
        logger.info(f"📰 Forex Sentiment Collector initialized for {len(self.currency_pairs)} pairs")
    
    def ensure_data_directory(self):
        """Ensure sentiment data directory exists"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(f"{self.data_dir}/raw", exist_ok=True)
        os.makedirs(f"{self.data_dir}/processed", exist_ok=True)
        os.makedirs(f"{self.data_dir}/labeled", exist_ok=True)
    
    def create_sample_news_dataset(self) -> pd.DataFrame:
        """
        📰 Create sample forex news dataset
        In production, you would collect real news from APIs
        """
        try:
            logger.info("📰 Creating sample forex news dataset")
            
            # Sample forex news headlines with sentiment labels
            sample_news = [
                # Positive sentiment
                ("EUR/USD breaks above key resistance level with strong bullish momentum", "positive"),
                ("Federal Reserve signals dovish stance, USD weakens across the board", "positive"),
                ("ECB hints at rate cuts, Euro gains against major currencies", "positive"),
                ("GBP/USD rallies on positive Brexit developments", "positive"),
                ("Strong economic data boosts AUD/USD to new highs", "positive"),
                ("USD/JPY breaks resistance with increased risk appetite", "positive"),
                ("Central bank intervention supports currency stability", "positive"),
                ("Inflation data beats expectations, currency strengthens", "positive"),
                
                # Negative sentiment
                ("EUR/USD faces headwinds from political uncertainty", "negative"),
                ("USD strengthens as Fed signals hawkish monetary policy", "negative"),
                ("GBP/USD drops on disappointing economic indicators", "negative"),
                ("Risk-off sentiment weighs on commodity currencies", "negative"),
                ("Central bank rate hike concerns pressure currency markets", "negative"),
                ("Political instability causes currency volatility", "negative"),
                ("Economic recession fears impact forex trading", "negative"),
                ("Trade war tensions escalate, affecting currency pairs", "negative"),
                
                # Neutral sentiment
                ("EUR/USD consolidates near key support levels", "neutral"),
                ("Currency markets await central bank policy decisions", "neutral"),
                ("Mixed signals from economic data create uncertainty", "neutral"),
                ("Forex markets show sideways movement with low volatility", "neutral"),
                ("Technical analysis suggests range-bound trading", "neutral"),
                ("Currency pairs trade within established channels", "neutral"),
                ("Market participants await key economic releases", "neutral"),
                ("Forex volatility remains subdued in quiet trading", "neutral"),
                
                # More specific forex news
                ("FOMC meeting minutes reveal dovish stance, USD weakens", "positive"),
                ("ECB maintains current interest rates, Euro stable", "neutral"),
                ("Bank of England hints at future rate increases", "positive"),
                ("Bank of Japan continues ultra-loose monetary policy", "negative"),
                ("RBA cuts interest rates, AUD/USD falls sharply", "negative"),
                ("Bank of Canada signals potential rate hikes", "positive"),
                ("RBNZ maintains dovish stance, NZD under pressure", "negative"),
                ("Swiss National Bank intervenes to weaken CHF", "negative"),
                
                # Economic indicators
                ("US GDP growth exceeds expectations, USD gains", "positive"),
                ("Eurozone inflation data shows mixed signals", "neutral"),
                ("UK employment data surprises to the upside", "positive"),
                ("Japanese manufacturing PMI indicates contraction", "negative"),
                ("Australian trade balance improves significantly", "positive"),
                ("Canadian retail sales data disappoints markets", "negative"),
                ("New Zealand dairy prices show recovery", "positive"),
                ("Swiss economic indicators point to stability", "neutral"),
                
                # Technical analysis
                ("EUR/USD forms bullish flag pattern on daily chart", "positive"),
                ("GBP/USD breaks below key support, bearish outlook", "negative"),
                ("USD/JPY tests resistance at 150.00 level", "neutral"),
                ("AUD/USD shows strong momentum above 0.70", "positive"),
                ("USD/CAD consolidates near 1.35 resistance", "neutral"),
                ("NZD/USD faces selling pressure at 0.65", "negative"),
                ("EUR/JPY breaks out of triangle formation", "positive"),
                ("GBP/JPY shows mixed signals on technical charts", "neutral"),
            ]
            
            # Create DataFrame
            data = []
            for i, (text, sentiment) in enumerate(sample_news):
                # Extract currency pair from text
                currency_pair = self._extract_currency_pair(text)
                
                data.append({
                    'id': i + 1,
                    'text': text,
                    'sentiment': sentiment,
                    'currency_pair': currency_pair,
                    'source': 'sample_news',
                    'timestamp': datetime.utcnow().isoformat(),
                    'keywords': self._extract_keywords(text),
                    'length': len(text.split())
                })
            
            df = pd.DataFrame(data)
            
            # Save sample dataset
            sample_path = f"{self.data_dir}/raw/forex_news_sample.csv"
            df.to_csv(sample_path, index=False)
            
            logger.info(f"✅ Created sample news dataset: {len(df)} records")
            logger.info(f"📊 Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error creating sample news dataset: {e}")
            return pd.DataFrame()
    
    def create_reddit_sample_dataset(self) -> pd.DataFrame:
        """
        📱 Create sample Reddit forex discussion dataset
        In production, you would scrape real Reddit posts
        """
        try:
            logger.info("📱 Creating sample Reddit forex dataset")
            
            # Sample Reddit-style forex discussions
            reddit_posts = [
                # Positive sentiment
                ("Just went long EUR/USD at 1.0850, feeling bullish about the breakout!", "positive"),
                ("GBP/USD looking strong after the BOE announcement. Moon mission! 🚀", "positive"),
                ("USD/JPY finally broke resistance. This is going to be epic!", "positive"),
                ("AUD/USD showing great momentum. Risk-on sentiment is back!", "positive"),
                ("Made 50 pips on EUR/GBP today. The trend is your friend!", "positive"),
                
                # Negative sentiment
                ("EUR/USD getting destroyed by the Fed. This is brutal 😭", "negative"),
                ("GBP/USD tanking hard. Brexit fears are back in full force", "negative"),
                ("USD/JPY looks like it's going to crash. Risk-off mode activated", "negative"),
                ("Lost 100 pips on AUD/USD. This market is impossible to predict", "negative"),
                ("EUR/GBP breaking down. Time to cut losses and run", "negative"),
                
                # Neutral sentiment
                ("EUR/USD consolidating around 1.08. Waiting for direction", "neutral"),
                ("GBP/USD in a tight range. Not much happening today", "neutral"),
                ("USD/JPY choppy action. Need to see a clear breakout", "neutral"),
                ("AUD/USD sideways movement. Market seems indecisive", "neutral"),
                ("EUR/GBP range-bound. Patience is key in this market", "neutral"),
            ]
            
            # Create DataFrame
            data = []
            for i, (text, sentiment) in enumerate(reddit_posts):
                currency_pair = self._extract_currency_pair(text)
                
                data.append({
                    'id': i + 1,
                    'text': text,
                    'sentiment': sentiment,
                    'currency_pair': currency_pair,
                    'source': 'reddit_sample',
                    'timestamp': datetime.utcnow().isoformat(),
                    'keywords': self._extract_keywords(text),
                    'length': len(text.split()),
                    'platform': 'reddit'
                })
            
            df = pd.DataFrame(data)
            
            # Save Reddit dataset
            reddit_path = f"{self.data_dir}/raw/forex_reddit_sample.csv"
            df.to_csv(reddit_path, index=False)
            
            logger.info(f"✅ Created Reddit sample dataset: {len(df)} records")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error creating Reddit sample dataset: {e}")
            return pd.DataFrame()
    
    def create_kaggle_style_dataset(self) -> pd.DataFrame:
        """
        📊 Create Kaggle-style forex sentiment dataset
        Combines multiple sources into a comprehensive dataset
        """
        try:
            logger.info("📊 Creating Kaggle-style forex sentiment dataset")
            
            # Load existing datasets
            news_df = self.create_sample_news_dataset()
            reddit_df = self.create_reddit_sample_dataset()
            
            # Combine datasets
            combined_data = []
            
            # Add news data
            for _, row in news_df.iterrows():
                combined_data.append({
                    'text': row['text'],
                    'sentiment': row['sentiment'],
                    'currency_pair': row['currency_pair'],
                    'source': 'news',
                    'platform': 'news_api',
                    'timestamp': row['timestamp'],
                    'keywords': row['keywords'],
                    'text_length': row['length']
                })
            
            # Add Reddit data
            for _, row in reddit_df.iterrows():
                combined_data.append({
                    'text': row['text'],
                    'sentiment': row['sentiment'],
                    'currency_pair': row['currency_pair'],
                    'source': 'social',
                    'platform': 'reddit',
                    'timestamp': row['timestamp'],
                    'keywords': row['keywords'],
                    'text_length': row['length']
                })
            
            # Create comprehensive dataset
            df = pd.DataFrame(combined_data)
            
            # Add additional features
            df['sentiment_numeric'] = df['sentiment'].map({'negative': -1, 'neutral': 0, 'positive': 1})
            df['has_currency_pair'] = df['currency_pair'].notna()
            df['forex_relevance'] = df['keywords'].apply(lambda x: len(x) > 0)
            
            # Save comprehensive dataset
            kaggle_path = f"{self.data_dir}/processed/forex_sentiment_kaggle_style.csv"
            df.to_csv(kaggle_path, index=False)
            
            logger.info(f"✅ Created Kaggle-style dataset: {len(df)} records")
            logger.info(f"📊 Sources: {df['source'].value_counts().to_dict()}")
            logger.info(f"📊 Sentiment: {df['sentiment'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error creating Kaggle-style dataset: {e}")
            return pd.DataFrame()
    
    def _extract_currency_pair(self, text: str) -> Optional[str]:
        """Extract currency pair from text"""
        text_upper = text.upper()
        
        # Common currency pair patterns
        patterns = [
            'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF',
            'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/JPY',
            'GBP/JPY', 'EUR/GBP', 'AUD/JPY', 'EUR/AUD',
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF',
            'AUDUSD', 'USDCAD', 'NZDUSD', 'EURJPY',
            'GBPJPY', 'EURGBP', 'AUDJPY', 'EURAUD'
        ]
        
        for pattern in patterns:
            if pattern in text_upper:
                # Normalize to standard format
                if '/' not in pattern:
                    # Convert EURUSD to EUR/USD
                    if len(pattern) == 6:
                        return f"{pattern[:3]}/{pattern[3:]}"
                return pattern
        
        return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract forex-related keywords from text"""
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in self.forex_keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def create_training_dataset(self) -> pd.DataFrame:
        """Create final training dataset for FinBERT fine-tuning"""
        try:
            logger.info("🏋️ Creating forex sentiment training dataset")
            
            # Create comprehensive dataset
            df = self.create_kaggle_style_dataset()
            
            if df.empty:
                logger.error("❌ No data available for training dataset")
                return pd.DataFrame()
            
            # Prepare data for FinBERT training
            training_data = []
            
            for _, row in df.iterrows():
                # Create training examples in FinBERT format
                training_data.append({
                    'text': row['text'],
                    'label': row['sentiment_numeric'],
                    'sentiment': row['sentiment'],
                    'currency_pair': row['currency_pair'],
                    'source': row['source'],
                    'keywords': ','.join(row['keywords']) if row['keywords'] else '',
                    'text_length': row['text_length']
                })
            
            training_df = pd.DataFrame(training_data)
            
            # Save training dataset
            training_path = f"{self.data_dir}/labeled/forex_sentiment_training.csv"
            training_df.to_csv(training_path, index=False)
            
            # Create train/validation split
            from sklearn.model_selection import train_test_split
            
            train_df, val_df = train_test_split(
                training_df, 
                test_size=0.2, 
                random_state=42, 
                stratify=training_df['sentiment']
            )
            
            # Save splits
            train_path = f"{self.data_dir}/labeled/forex_sentiment_train.csv"
            val_path = f"{self.data_dir}/labeled/forex_sentiment_val.csv"
            
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            
            logger.info(f"✅ Training dataset created: {len(training_df)} records")
            logger.info(f"📊 Train: {len(train_df)} records")
            logger.info(f"📊 Validation: {len(val_df)} records")
            logger.info(f"📊 Sentiment distribution: {training_df['sentiment'].value_counts().to_dict()}")
            
            return training_df
            
        except Exception as e:
            logger.error(f"❌ Error creating training dataset: {e}")
            return pd.DataFrame()
    
    def get_sentiment_summary(self) -> Dict[str, Any]:
        """Get summary of sentiment data collection"""
        try:
            summary = {
                "data_directory": self.data_dir,
                "currency_pairs": self.currency_pairs,
                "forex_keywords": self.forex_keywords,
                "raw_files": [],
                "processed_files": [],
                "labeled_files": [],
                "total_records": 0
            }
            
            # Count files in each directory
            for subdir in ['raw', 'processed', 'labeled']:
                dir_path = f"{self.data_dir}/{subdir}"
                if os.path.exists(dir_path):
                    files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
                    summary[f"{subdir}_files"] = files
                    
                    # Count total records
                    for file in files:
                        try:
                            data = pd.read_csv(f"{dir_path}/{file}")
                            summary["total_records"] += len(data)
                        except:
                            continue
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting sentiment summary: {e}")
            return {"error": str(e)}

def main():
    """Test the forex sentiment collector"""
    print("📰 Testing DEXTER Forex Sentiment Collector")
    print("=" * 50)
    
    # Initialize collector
    collector = ForexSentimentCollector()
    
    # Test 1: Sample news dataset
    print("\n📰 Test 1: Sample News Dataset")
    news_df = collector.create_sample_news_dataset()
    print(f"✅ News dataset: {len(news_df)} records")
    
    # Test 2: Reddit sample dataset
    print("\n📱 Test 2: Reddit Sample Dataset")
    reddit_df = collector.create_reddit_sample_dataset()
    print(f"✅ Reddit dataset: {len(reddit_df)} records")
    
    # Test 3: Kaggle-style dataset
    print("\n📊 Test 3: Kaggle-style Dataset")
    kaggle_df = collector.create_kaggle_style_dataset()
    print(f"✅ Kaggle-style dataset: {len(kaggle_df)} records")
    
    # Test 4: Training dataset
    print("\n🏋️ Test 4: Training Dataset")
    training_df = collector.create_training_dataset()
    print(f"✅ Training dataset: {len(training_df)} records")
    
    # Test 5: Summary
    print("\n📋 Test 5: Sentiment Data Summary")
    summary = collector.get_sentiment_summary()
    print(f"📁 Data directory: {summary['data_directory']}")
    print(f"📊 Raw files: {len(summary['raw_files'])}")
    print(f"📊 Processed files: {len(summary['processed_files'])}")
    print(f"📊 Labeled files: {len(summary['labeled_files'])}")
    print(f"📊 Total records: {summary['total_records']}")
    
    print("\n✅ Forex sentiment collector testing completed!")

if __name__ == "__main__":
    main()

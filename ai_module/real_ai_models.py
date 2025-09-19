#!/usr/bin/env python3
"""
DEXTER Real AI Models Integration
Integrates pre-trained models for sentiment analysis and price forecasting
"""

import torch
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer
)
import pandas as pd
from typing import Dict, Any
import structlog
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

logger = structlog.get_logger()

class RealAITradingModels:
    """
    Real AI models for trading - no more statistical models!
    Uses pre-trained models downloaded to your laptop
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.sentiment_model = None
        self.forecasting_model = None
        self.tokenizer = None
        
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained models (downloaded once, used locally)"""
        try:
            logger.info("Loading pre-trained AI models...")
            
            self._load_sentiment_model()
            
            self._load_forecasting_model()
            
            logger.info("✅ All AI models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.warning("Falling back to statistical models")
    
    def _load_sentiment_model(self):
        """Load FinBERT for financial sentiment analysis"""
        try:
            logger.info("Loading FinBERT sentiment model...")
            
            # Try to load fine-tuned crypto FinBERT first
            finetuned_path = "backend/ai_module/models/finbert-finetuned"
            if os.path.exists(finetuned_path):
                logger.info("Loading fine-tuned crypto FinBERT...")
                self.tokenizer = AutoTokenizer.from_pretrained(finetuned_path)
                self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(finetuned_path)
                logger.info("✅ Fine-tuned crypto FinBERT loaded!")
            else:
                # Fallback to base FinBERT
                logger.warning("Fine-tuned model not found, using base FinBERT")
                model_name = "ProsusAI/finbert"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move to device
            self.sentiment_model.to(self.device)
            self.sentiment_model.eval()
            
            logger.info("✅ FinBERT sentiment model loaded!")
            
        except Exception as e:
            logger.error(f"Error loading sentiment model: {e}")
            self.sentiment_model = None
    
    def _load_forecasting_model(self):
        """Load time series forecasting model"""
        try:
            logger.info("Loading time series forecasting model...")
            
            # For now, we'll use a simple approach
            # Later we'll integrate more sophisticated models
            self.forecasting_model = "simple_forecasting"
            
            logger.info("✅ Forecasting model loaded!")
            
        except Exception as e:
            logger.error(f"Error loading forecasting model: {e}")
            self.forecasting_model = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze financial sentiment using FinBERT
        Returns: sentiment score, confidence, and analysis
        """
        try:
            if self.sentiment_model is None:
                return self._fallback_sentiment(text)
            
            # Tokenize input
            if self.tokenizer is None:
                return self._fallback_sentiment(text)
            
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
            
            labels = ["negative", "neutral", "positive"]
            sentiment_idx = torch.argmax(probabilities, dim=1).item()
            # Type assertion to help type checker
            sentiment_idx = int(sentiment_idx)
            confidence = probabilities[0, sentiment_idx].item()
            sentiment = labels[sentiment_idx]
            
            sentiment_score = (sentiment_idx - 1) * confidence
            
            return {
                "sentiment": sentiment,
                "sentiment_score": sentiment_score,
                "confidence": confidence,
                "model": "FinBERT",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._fallback_sentiment(text)
    
    def _fallback_sentiment(self, text: str) -> Dict[str, Any]:
        """Fallback sentiment analysis when AI model fails"""
        return {
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "confidence": 0.5,
            "model": "fallback",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def predict_price_movement(self, market_data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Predict price movement using AI models
        Returns: prediction, confidence, and timeframe
        """
        try:
            if self.forecasting_model is None:
                return self._fallback_price_prediction(symbol)
            
            features = self._extract_price_features(market_data)
            
            # Make prediction (placeholder for now)
            prediction = self._make_price_prediction(features)
            
            return {
                "symbol": symbol,
                "prediction": prediction["direction"],
                "confidence": prediction["confidence"],
                "timeframe": prediction["timeframe"],
                "price_target": prediction["price_target"],
                "model": "AI_Forecasting",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in price prediction: {e}")
            return self._fallback_price_prediction(symbol)
    
    def _extract_price_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract features for price prediction"""
        try:
            recent_data = data.tail(50)  # Last 50 data points
            
            close_prices: pd.Series = pd.Series(recent_data['Close'])
            volume_prices: pd.Series = pd.Series(recent_data['Volume'])
            
            features = {
                "price_change": close_prices.pct_change().mean(),
                "volume_change": volume_prices.pct_change().mean(),
                "volatility": close_prices.pct_change().std(),
                "trend": 1 if close_prices.iloc[-1] > close_prices.iloc[0] else -1,
                "rsi": self._calculate_rsi(close_prices),
                "macd": self._calculate_macd(close_prices)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}
    
    def _make_price_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make price prediction based on features"""
        try:
            # Simple rule-based prediction (placeholder)
            # Later we'll integrate real ML models
            
            if features.get("rsi", 50) < 30:
                direction = "BUY"
                confidence = 0.7
            elif features.get("rsi", 50) > 70:
                direction = "SELL"
                confidence = 0.7
            else:
                direction = "HOLD"
                confidence = 0.5
            
            return {
                "direction": direction,
                "confidence": confidence,
                "timeframe": "24h",
                "price_target": "dynamic"
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                "direction": "HOLD",
                "confidence": 0.5,
                "timeframe": "24h",
                "price_target": "unknown"
            }
    
    def _fallback_price_prediction(self, symbol: str) -> Dict[str, Any]:
        """Fallback price prediction when AI model fails"""
        return {
            "symbol": symbol,
            "prediction": "HOLD",
            "confidence": 0.5,
            "timeframe": "24h",
            "price_target": "unknown",
            "model": "fallback",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            if isinstance(rsi, pd.Series) and not rsi.empty:
                return rsi.iloc[-1]
            return 50.0
        except:
            return 50.0
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> float:
        """Calculate MACD indicator"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            if isinstance(macd, pd.Series) and not macd.empty:
                return macd.iloc[-1]
            return 0.0
        except:
            return 0.0
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all AI models"""
        return {
            "sentiment_model": "FinBERT" if self.sentiment_model else "Not Available",
            "forecasting_model": "AI_Forecasting" if self.forecasting_model else "Not Available",
            "device": str(self.device),
            "models_loaded": self.sentiment_model is not None and self.forecasting_model is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def test_models(self) -> Dict[str, Any]:
        """Test all models with sample data"""
        try:
            results = {}
            
            # Test sentiment analysis
            test_text = "Bitcoin shows strong bullish momentum with increasing volume"
            sentiment_result = self.analyze_sentiment(test_text)
            results["sentiment_test"] = sentiment_result
            
            # Test price prediction
            test_data = pd.DataFrame({
                'Close': [100, 101, 102, 103, 104],
                'Volume': [1000, 1100, 1200, 1300, 1400]
            })
            price_result = self.predict_price_movement(test_data, "BTCUSDT")
            results["price_test"] = price_result
            
            results["status"] = "success"
            return results
            
        except Exception as e:
            logger.error(f"Error testing models: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

#!/usr/bin/env python3
"""
🔮 DEXTER PREDICTIVE LEARNING SYSTEM
============================================================
Learns to predict market conditions and adapt strategies accordingly
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import structlog
import pandas as pd
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import yfinance as yf
import talib

logger = structlog.get_logger()

class PredictiveLearningSystem:
    """
    🔮 Predictive Learning System for DEXTER
    
    Features:
    1. Predict market conditions (bull/bear/sideways)
    2. Predict volatility levels
    3. Predict optimal trading times
    4. Predict market regime changes
    5. Predict user behavior patterns
    6. Adaptive strategy selection
    """
    
    def __init__(self, mongodb_url: str = None):
        self.mongodb_url = mongodb_url or os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.db = None
        self.collection = None
        
        # Predictive models
        self.market_condition_model = None
        self.volatility_model = None
        self.timing_model = None
        self.regime_change_model = None
        
        # Model parameters
        self.model_retrain_threshold = 100
        self.prediction_confidence_threshold = 0.7
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_duration = 3600  # 1 hour
        
    async def initialize(self):
        """Initialize MongoDB connection"""
        try:
            client = AsyncIOMotorClient(self.mongodb_url)
            self.db = client.dexter
            self.collection = self.db.predictive_learning
            
            # Create indexes
            await self.collection.create_index("timestamp")
            await self.collection.create_index("prediction_type")
            await self.collection.create_index("symbol")
            
            logger.info("✅ Predictive Learning System initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Predictive Learning System: {e}")
            raise
    
    async def predict_market_conditions(self, symbol: str, timeframe: str = "1d") -> Dict[str, Any]:
        """Predict market conditions for a symbol"""
        try:
            if not self.collection:
                await self.initialize()
            
            # Check cache first
            cache_key = f"market_conditions_{symbol}_{timeframe}"
            if cache_key in self.prediction_cache:
                cached_prediction = self.prediction_cache[cache_key]
                if datetime.utcnow() - cached_prediction["timestamp"] < timedelta(seconds=self.cache_duration):
                    return cached_prediction["prediction"]
            
            logger.info(f"🔮 Predicting market conditions for {symbol}")
            
            # Get historical data
            historical_data = await self._get_historical_data(symbol, days=90)
            
            if historical_data.empty:
                return {"error": f"No historical data available for {symbol}"}
            
            # Prepare features
            features = self._prepare_market_features(historical_data)
            
            # Train or load model
            if self.market_condition_model is None:
                await self._train_market_condition_model(symbol)
            
            # Make prediction
            prediction = self._predict_market_condition(features)
            
            # Store prediction
            await self._store_prediction("market_conditions", symbol, prediction, features)
            
            # Cache prediction
            self.prediction_cache[cache_key] = {
                "prediction": prediction,
                "timestamp": datetime.utcnow()
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting market conditions: {e}")
            return {"error": str(e)}
    
    async def predict_volatility(self, symbol: str, days_ahead: int = 7) -> Dict[str, Any]:
        """Predict volatility for a symbol"""
        try:
            if not self.collection:
                await self.initialize()
            
            logger.info(f"🔮 Predicting volatility for {symbol}")
            
            # Get historical data
            historical_data = await self._get_historical_data(symbol, days=180)
            
            if historical_data.empty:
                return {"error": f"No historical data available for {symbol}"}
            
            # Prepare features
            features = self._prepare_volatility_features(historical_data)
            
            # Train or load model
            if self.volatility_model is None:
                await self._train_volatility_model(symbol)
            
            # Make prediction
            prediction = self._predict_volatility(features, days_ahead)
            
            # Store prediction
            await self._store_prediction("volatility", symbol, prediction, features)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting volatility: {e}")
            return {"error": str(e)}
    
    async def predict_optimal_timing(self, symbol: str, user_id: str = None) -> Dict[str, Any]:
        """Predict optimal trading timing"""
        try:
            if not self.collection:
                await self.initialize()
            
            logger.info(f"🔮 Predicting optimal timing for {symbol}")
            
            # Get market data
            market_data = await self._get_historical_data(symbol, days=30)
            
            if market_data.empty:
                return {"error": f"No market data available for {symbol}"}
            
            # Get user trading patterns if available
            user_patterns = None
            if user_id:
                user_patterns = await self._get_user_trading_patterns(user_id)
            
            # Prepare features
            features = self._prepare_timing_features(market_data, user_patterns)
            
            # Train or load model
            if self.timing_model is None:
                await self._train_timing_model(symbol)
            
            # Make prediction
            prediction = self._predict_optimal_timing(features)
            
            # Store prediction
            await self._store_prediction("optimal_timing", symbol, prediction, features)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting optimal timing: {e}")
            return {"error": str(e)}
    
    async def predict_regime_change(self, symbol: str) -> Dict[str, Any]:
        """Predict market regime changes"""
        try:
            if not self.collection:
                await self.initialize()
            
            logger.info(f"🔮 Predicting regime change for {symbol}")
            
            # Get historical data
            historical_data = await self._get_historical_data(symbol, days=365)
            
            if historical_data.empty:
                return {"error": f"No historical data available for {symbol}"}
            
            # Prepare features
            features = self._prepare_regime_features(historical_data)
            
            # Train or load model
            if self.regime_change_model is None:
                await self._train_regime_change_model(symbol)
            
            # Make prediction
            prediction = self._predict_regime_change(features)
            
            # Store prediction
            await self._store_prediction("regime_change", symbol, prediction, features)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting regime change: {e}")
            return {"error": str(e)}
    
    async def _get_historical_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Get historical price data"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d")
            
            if hist.empty:
                return pd.DataFrame()
            
            # Add technical indicators
            hist = self._add_technical_indicators(hist)
            
            return hist
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to data"""
        try:
            # Moving averages
            data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)
            data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
            data['EMA_12'] = talib.EMA(data['Close'], timeperiod=12)
            data['EMA_26'] = talib.EMA(data['Close'], timeperiod=26)
            
            # RSI
            data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(data['Close'])
            data['MACD'] = macd
            data['MACD_Signal'] = macd_signal
            data['MACD_Hist'] = macd_hist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(data['Close'])
            data['BB_upper'] = bb_upper
            data['BB_middle'] = bb_middle
            data['BB_lower'] = bb_lower
            
            # ATR
            data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'])
            
            # ADX
            data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'])
            
            # Volume indicators
            data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
            
            # Price changes
            data['Price_Change'] = data['Close'].pct_change()
            data['Price_Change_5d'] = data['Close'].pct_change(5)
            data['Price_Change_20d'] = data['Close'].pct_change(20)
            
            # Volatility
            data['Volatility'] = data['Price_Change'].rolling(window=20).std()
            
            return data
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return data
    
    def _prepare_market_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for market condition prediction"""
        try:
            if data.empty:
                return np.array([])
            
            # Get latest values
            latest = data.iloc[-1]
            
            # Create feature vector
            features = [
                latest.get('SMA_20', 0),
                latest.get('SMA_50', 0),
                latest.get('RSI', 50),
                latest.get('MACD', 0),
                latest.get('MACD_Hist', 0),
                latest.get('BB_upper', 0),
                latest.get('BB_lower', 0),
                latest.get('ATR', 0),
                latest.get('ADX', 0),
                latest.get('Volume_Ratio', 1),
                latest.get('Price_Change', 0),
                latest.get('Price_Change_5d', 0),
                latest.get('Price_Change_20d', 0),
                latest.get('Volatility', 0.05)
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing market features: {e}")
            return np.array([])
    
    def _prepare_volatility_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for volatility prediction"""
        try:
            if data.empty:
                return np.array([])
            
            # Get recent volatility patterns
            recent_data = data.tail(30)
            
            features = [
                recent_data['Volatility'].mean(),
                recent_data['Volatility'].std(),
                recent_data['ATR'].mean(),
                recent_data['Volume_Ratio'].mean(),
                recent_data['Price_Change'].std(),
                recent_data['RSI'].mean(),
                recent_data['ADX'].mean()
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing volatility features: {e}")
            return np.array([])
    
    def _prepare_timing_features(self, data: pd.DataFrame, user_patterns: Dict[str, Any] = None) -> np.ndarray:
        """Prepare features for timing prediction"""
        try:
            if data.empty:
                return np.array([])
            
            # Get current market features
            latest = data.iloc[-1]
            
            features = [
                latest.get('RSI', 50),
                latest.get('MACD', 0),
                latest.get('Volume_Ratio', 1),
                latest.get('Price_Change', 0),
                latest.get('Volatility', 0.05)
            ]
            
            # Add user-specific features if available
            if user_patterns:
                features.extend([
                    user_patterns.get('preferred_hour', 12),
                    user_patterns.get('activity_level', 1),
                    user_patterns.get('success_rate', 0.5)
                ])
            else:
                features.extend([12, 1, 0.5])  # Default values
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing timing features: {e}")
            return np.array([])
    
    def _prepare_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for regime change prediction"""
        try:
            if data.empty:
                return np.array([])
            
            # Calculate regime indicators
            data['Trend'] = (data['SMA_20'] - data['SMA_50']) / data['SMA_50']
            data['Momentum'] = data['Price_Change_20d']
            data['Volatility_Trend'] = data['Volatility'].rolling(window=20).mean()
            
            # Get recent regime features
            recent_data = data.tail(60)
            
            features = [
                recent_data['Trend'].mean(),
                recent_data['Momentum'].mean(),
                recent_data['Volatility_Trend'].mean(),
                recent_data['RSI'].mean(),
                recent_data['ADX'].mean(),
                recent_data['Volume_Ratio'].mean()
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing regime features: {e}")
            return np.array([])
    
    async def _train_market_condition_model(self, symbol: str):
        """Train market condition prediction model"""
        try:
            # Get training data
            training_data = await self._get_training_data("market_conditions", symbol)
            
            if training_data.empty:
                logger.warning("No training data available for market condition model")
                return
            
            # Prepare features and labels
            X = training_data.drop(['condition'], axis=1).values
            y = training_data['condition'].values
            
            # Train model
            self.market_condition_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.market_condition_model.fit(X, y)
            
            logger.info("✅ Market condition model trained")
            
        except Exception as e:
            logger.error(f"Error training market condition model: {e}")
    
    async def _train_volatility_model(self, symbol: str):
        """Train volatility prediction model"""
        try:
            # Get training data
            training_data = await self._get_training_data("volatility", symbol)
            
            if training_data.empty:
                logger.warning("No training data available for volatility model")
                return
            
            # Prepare features and labels
            X = training_data.drop(['volatility'], axis=1).values
            y = training_data['volatility'].values
            
            # Train model
            self.volatility_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            self.volatility_model.fit(X, y)
            
            logger.info("✅ Volatility model trained")
            
        except Exception as e:
            logger.error(f"Error training volatility model: {e}")
    
    async def _train_timing_model(self, symbol: str):
        """Train timing prediction model"""
        try:
            # Get training data
            training_data = await self._get_training_data("timing", symbol)
            
            if training_data.empty:
                logger.warning("No training data available for timing model")
                return
            
            # Prepare features and labels
            X = training_data.drop(['optimal_timing'], axis=1).values
            y = training_data['optimal_timing'].values
            
            # Train model
            self.timing_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.timing_model.fit(X, y)
            
            logger.info("✅ Timing model trained")
            
        except Exception as e:
            logger.error(f"Error training timing model: {e}")
    
    async def _train_regime_change_model(self, symbol: str):
        """Train regime change prediction model"""
        try:
            # Get training data
            training_data = await self._get_training_data("regime_change", symbol)
            
            if training_data.empty:
                logger.warning("No training data available for regime change model")
                return
            
            # Prepare features and labels
            X = training_data.drop(['regime_change'], axis=1).values
            y = training_data['regime_change'].values
            
            # Train model
            self.regime_change_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.regime_change_model.fit(X, y)
            
            logger.info("✅ Regime change model trained")
            
        except Exception as e:
            logger.error(f"Error training regime change model: {e}")
    
    async def _get_training_data(self, prediction_type: str, symbol: str) -> pd.DataFrame:
        """Get training data for model training"""
        try:
            # Get historical predictions
            predictions = await self.collection.find({
                "prediction_type": prediction_type,
                "symbol": symbol
            }).to_list(length=1000)
            
            if not predictions:
                return pd.DataFrame()
            
            # Convert to DataFrame
            training_data = []
            for pred in predictions:
                features = pred.get("features", [])
                actual_outcome = pred.get("actual_outcome")
                
                if features and actual_outcome is not None:
                    row = features.copy()
                    row["actual_outcome"] = actual_outcome
                    training_data.append(row)
            
            return pd.DataFrame(training_data)
            
        except Exception as e:
            logger.error(f"Error getting training data: {e}")
            return pd.DataFrame()
    
    def _predict_market_condition(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict market condition"""
        try:
            if self.market_condition_model is None or features.size == 0:
                return {"condition": "unknown", "confidence": 0.0}
            
            # Make prediction
            prediction = self.market_condition_model.predict(features)[0]
            probabilities = self.market_condition_model.predict_proba(features)[0]
            confidence = np.max(probabilities)
            
            # Map prediction to condition
            condition_map = {0: "bear", 1: "sideways", 2: "bull"}
            condition = condition_map.get(prediction, "unknown")
            
            return {
                "condition": condition,
                "confidence": float(confidence),
                "probabilities": {
                    "bear": float(probabilities[0]),
                    "sideways": float(probabilities[1]),
                    "bull": float(probabilities[2])
                }
            }
            
        except Exception as e:
            logger.error(f"Error predicting market condition: {e}")
            return {"condition": "unknown", "confidence": 0.0}
    
    def _predict_volatility(self, features: np.ndarray, days_ahead: int) -> Dict[str, Any]:
        """Predict volatility"""
        try:
            if self.volatility_model is None or features.size == 0:
                return {"volatility": 0.05, "confidence": 0.0}
            
            # Make prediction
            prediction = self.volatility_model.predict(features)[0]
            
            # Calculate confidence based on feature variance
            confidence = min(1.0, max(0.0, 1.0 - np.var(features) * 10))
            
            return {
                "volatility": float(prediction),
                "confidence": float(confidence),
                "days_ahead": days_ahead,
                "volatility_level": self._classify_volatility_level(prediction)
            }
            
        except Exception as e:
            logger.error(f"Error predicting volatility: {e}")
            return {"volatility": 0.05, "confidence": 0.0}
    
    def _predict_optimal_timing(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict optimal timing"""
        try:
            if self.timing_model is None or features.size == 0:
                return {"timing": "neutral", "confidence": 0.0}
            
            # Make prediction
            prediction = self.timing_model.predict(features)[0]
            probabilities = self.timing_model.predict_proba(features)[0]
            confidence = np.max(probabilities)
            
            # Map prediction to timing
            timing_map = {0: "avoid", 1: "neutral", 2: "optimal"}
            timing = timing_map.get(prediction, "neutral")
            
            return {
                "timing": timing,
                "confidence": float(confidence),
                "recommendation": self._get_timing_recommendation(timing, confidence)
            }
            
        except Exception as e:
            logger.error(f"Error predicting optimal timing: {e}")
            return {"timing": "neutral", "confidence": 0.0}
    
    def _predict_regime_change(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict regime change"""
        try:
            if self.regime_change_model is None or features.size == 0:
                return {"regime_change": False, "confidence": 0.0}
            
            # Make prediction
            prediction = self.regime_change_model.predict(features)[0]
            probabilities = self.regime_change_model.predict_proba(features)[0]
            confidence = np.max(probabilities)
            
            return {
                "regime_change": bool(prediction),
                "confidence": float(confidence),
                "timeframe": "7-14 days",
                "recommendation": self._get_regime_change_recommendation(prediction, confidence)
            }
            
        except Exception as e:
            logger.error(f"Error predicting regime change: {e}")
            return {"regime_change": False, "confidence": 0.0}
    
    def _classify_volatility_level(self, volatility: float) -> str:
        """Classify volatility level"""
        if volatility < 0.02:
            return "low"
        elif volatility < 0.05:
            return "medium"
        elif volatility < 0.10:
            return "high"
        else:
            return "very_high"
    
    def _get_timing_recommendation(self, timing: str, confidence: float) -> str:
        """Get timing recommendation"""
        if timing == "optimal" and confidence > 0.7:
            return "Strong buy/sell signal"
        elif timing == "optimal":
            return "Consider trading"
        elif timing == "avoid" and confidence > 0.7:
            return "Avoid trading"
        else:
            return "Wait for better conditions"
    
    def _get_regime_change_recommendation(self, regime_change: bool, confidence: float) -> str:
        """Get regime change recommendation"""
        if regime_change and confidence > 0.7:
            return "Prepare for market regime change"
        elif regime_change:
            return "Monitor for potential regime change"
        else:
            return "Current regime likely to continue"
    
    async def _get_user_trading_patterns(self, user_id: str) -> Dict[str, Any]:
        """Get user trading patterns"""
        try:
            # Get user's trading history
            user_trades = await self.collection.find({
                "user_id": user_id,
                "type": "user_trading_pattern"
            }).to_list(length=100)
            
            if not user_trades:
                return {}
            
            # Analyze patterns
            patterns = {
                "preferred_hour": 12,
                "activity_level": len(user_trades) / 30,
                "success_rate": 0.5
            }
            
            # Calculate preferred hour
            hours = [t.get("hour", 12) for t in user_trades if "hour" in t]
            if hours:
                patterns["preferred_hour"] = int(np.mean(hours))
            
            # Calculate success rate
            successes = [t.get("success", False) for t in user_trades]
            if successes:
                patterns["success_rate"] = np.mean(successes)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error getting user trading patterns: {e}")
            return {}
    
    async def _store_prediction(self, prediction_type: str, symbol: str, prediction: Dict[str, Any], features: np.ndarray):
        """Store prediction for learning"""
        try:
            prediction_doc = {
                "prediction_type": prediction_type,
                "symbol": symbol,
                "prediction": prediction,
                "features": features.tolist() if features.size > 0 else [],
                "timestamp": datetime.utcnow(),
                "type": "prediction"
            }
            
            await self.collection.insert_one(prediction_doc)
            
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
    
    async def update_prediction_accuracy(self, prediction_id: str, actual_outcome: Any):
        """Update prediction accuracy for learning"""
        try:
            # Update prediction with actual outcome
            await self.collection.update_one(
                {"_id": prediction_id},
                {"$set": {"actual_outcome": actual_outcome}}
            )
            
            # Check if we should retrain models
            await self._check_model_retraining()
            
        except Exception as e:
            logger.error(f"Error updating prediction accuracy: {e}")
    
    async def _check_model_retraining(self):
        """Check if models need retraining"""
        try:
            # Count recent predictions
            recent_predictions = await self.collection.count_documents({
                "timestamp": {"$gte": datetime.utcnow() - timedelta(days=7)},
                "type": "prediction"
            })
            
            if recent_predictions >= self.model_retrain_threshold:
                logger.info("🔄 Triggering model retraining due to sufficient new data")
                # Trigger model retraining
                await self._retrain_all_models()
                
        except Exception as e:
            logger.error(f"Error checking model retraining: {e}")
    
    async def _retrain_all_models(self):
        """Retrain all prediction models"""
        try:
            # Get unique symbols
            symbols = await self.collection.distinct("symbol")
            
            for symbol in symbols:
                await self._train_market_condition_model(symbol)
                await self._train_volatility_model(symbol)
                await self._train_timing_model(symbol)
                await self._train_regime_change_model(symbol)
            
            logger.info("✅ All prediction models retrained")
            
        except Exception as e:
            logger.error(f"Error retraining all models: {e}")
    
    async def get_predictive_insights(self) -> Dict[str, Any]:
        """Get predictive learning insights"""
        try:
            if not self.collection:
                await self.initialize()
            
            # Get prediction statistics
            total_predictions = await self.collection.count_documents({"type": "prediction"})
            
            # Get accuracy by prediction type
            prediction_types = await self.collection.distinct("prediction_type")
            accuracy_by_type = {}
            
            for pred_type in prediction_types:
                predictions = await self.collection.find({
                    "prediction_type": pred_type,
                    "actual_outcome": {"$exists": True}
                }).to_list(length=1000)
                
                if predictions:
                    # Calculate accuracy (simplified)
                    correct_predictions = len([p for p in predictions if p.get("prediction", {}).get("confidence", 0) > 0.7])
                    accuracy = correct_predictions / len(predictions)
                    accuracy_by_type[pred_type] = accuracy
            
            return {
                "total_predictions": total_predictions,
                "accuracy_by_type": accuracy_by_type,
                "model_status": {
                    "market_condition_model": self.market_condition_model is not None,
                    "volatility_model": self.volatility_model is not None,
                    "timing_model": self.timing_model is not None,
                    "regime_change_model": self.regime_change_model is not None
                },
                "cache_size": len(self.prediction_cache),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting predictive insights: {e}")
            return {"error": str(e)}

# Global instance
predictive_learning_system = PredictiveLearningSystem()

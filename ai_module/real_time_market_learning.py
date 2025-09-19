#!/usr/bin/env python3
"""
🎯 DEXTER REAL-TIME MARKET LEARNING SYSTEM
============================================================
Learns from live market movements and prediction accuracy
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
import aiohttp
import yfinance as yf
import ccxt

logger = structlog.get_logger()

class RealTimeMarketLearning:
    """
    🎯 Real-Time Market Learning System
    
    Features:
    1. Track prediction accuracy in real-time
    2. Learn from market movements
    3. Adapt to market regime changes
    4. Learn optimal timing for predictions
    5. Track market volatility patterns
    """
    
    def __init__(self, mongodb_url: str = None):
        self.mongodb_url = mongodb_url or os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.db = None
        self.collection = None
        
        # Market data sources
        self.exchanges = {
            'binance': ccxt.binance(),
            'kucoin': ccxt.kucoin()
        }
        
        # Learning parameters
        self.accuracy_threshold = 0.6
        self.volatility_threshold = 0.05
        self.regime_change_threshold = 0.1
        
        # Performance tracking
        self.market_regimes = {
            'bull': {'accuracy': 0.0, 'count': 0},
            'bear': {'accuracy': 0.0, 'count': 0},
            'sideways': {'accuracy': 0.0, 'count': 0}
        }
        
    async def initialize(self):
        """Initialize MongoDB connection"""
        try:
            client = AsyncIOMotorClient(self.mongodb_url)
            self.db = client.dexter
            self.collection = self.db.real_time_market_learning
            
            # Create indexes
            await self.collection.create_index("timestamp")
            await self.collection.create_index("symbol")
            await self.collection.create_index("prediction_id")
            
            logger.info("✅ Real-Time Market Learning initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Real-Time Market Learning: {e}")
            raise
    
    async def track_prediction_accuracy(self, 
                                      prediction_id: str,
                                      symbol: str,
                                      prediction: str,
                                      confidence: float,
                                      timeframe: str = "1h"):
        """Track prediction accuracy in real-time"""
        try:
            if not self.collection:
                await self.initialize()
            
            # Get current price
            current_price = await self._get_current_price(symbol)
            
            # Store prediction for later accuracy check
            prediction_data = {
                "prediction_id": prediction_id,
                "symbol": symbol,
                "prediction": prediction,
                "confidence": confidence,
                "current_price": current_price,
                "timeframe": timeframe,
                "timestamp": datetime.utcnow(),
                "status": "pending"
            }
            
            await self.collection.insert_one(prediction_data)
            
            # Schedule accuracy check
            asyncio.create_task(self._check_prediction_accuracy(prediction_id, timeframe))
            
            logger.info(f"📊 Tracking prediction accuracy for {symbol}")
            
        except Exception as e:
            logger.error(f"Error tracking prediction accuracy: {e}")
    
    async def _check_prediction_accuracy(self, prediction_id: str, timeframe: str):
        """Check prediction accuracy after timeframe"""
        try:
            # Wait for the timeframe
            if timeframe == "1h":
                await asyncio.sleep(3600)  # 1 hour
            elif timeframe == "4h":
                await asyncio.sleep(14400)  # 4 hours
            elif timeframe == "1d":
                await asyncio.sleep(86400)  # 1 day
            
            # Get prediction data
            prediction = await self.collection.find_one({"prediction_id": prediction_id})
            if not prediction:
                return
            
            # Get current price
            current_price = await self._get_current_price(prediction["symbol"])
            original_price = prediction["current_price"]
            
            # Calculate accuracy
            price_change = (current_price - original_price) / original_price
            accuracy = self._calculate_prediction_accuracy(
                prediction["prediction"], 
                price_change, 
                prediction["confidence"]
            )
            
            # Update prediction with accuracy
            await self.collection.update_one(
                {"prediction_id": prediction_id},
                {
                    "$set": {
                        "final_price": current_price,
                        "price_change": price_change,
                        "accuracy": accuracy,
                        "status": "completed",
                        "completed_at": datetime.utcnow()
                    }
                }
            )
            
            # Learn from this prediction
            await self._learn_from_prediction(prediction, accuracy, price_change)
            
            logger.info(f"✅ Prediction accuracy checked: {accuracy:.2f}")
            
        except Exception as e:
            logger.error(f"Error checking prediction accuracy: {e}")
    
    def _calculate_prediction_accuracy(self, prediction: str, price_change: float, confidence: float) -> float:
        """Calculate prediction accuracy based on direction and magnitude"""
        try:
            # Extract direction from prediction
            prediction_lower = prediction.lower()
            
            if any(word in prediction_lower for word in ["bullish", "buy", "up", "rise", "increase"]):
                predicted_direction = 1
            elif any(word in prediction_lower for word in ["bearish", "sell", "down", "fall", "decrease"]):
                predicted_direction = -1
            else:
                predicted_direction = 0  # Neutral
            
            # Calculate actual direction
            actual_direction = 1 if price_change > 0 else -1 if price_change < 0 else 0
            
            # Base accuracy on direction match
            if predicted_direction == actual_direction:
                base_accuracy = 0.8
            elif predicted_direction == 0 or actual_direction == 0:
                base_accuracy = 0.5
            else:
                base_accuracy = 0.2
            
            # Adjust for confidence
            confidence_factor = min(confidence, 1.0)
            
            # Adjust for magnitude (if prediction was about significant movement)
            magnitude_factor = 1.0
            if abs(price_change) > 0.05:  # 5% movement
                magnitude_factor = 1.1
            elif abs(price_change) < 0.01:  # 1% movement
                magnitude_factor = 0.9
            
            final_accuracy = base_accuracy * confidence_factor * magnitude_factor
            return min(1.0, max(0.0, final_accuracy))
            
        except Exception as e:
            logger.error(f"Error calculating prediction accuracy: {e}")
            return 0.5
    
    async def _learn_from_prediction(self, prediction: Dict, accuracy: float, price_change: float):
        """Learn from prediction results"""
        try:
            # Get market regime
            regime = await self._get_market_regime(prediction["symbol"])
            
            # Update regime performance
            self.market_regimes[regime]["accuracy"] = (
                (self.market_regimes[regime]["accuracy"] * self.market_regimes[regime]["count"] + accuracy) /
                (self.market_regimes[regime]["count"] + 1)
            )
            self.market_regimes[regime]["count"] += 1
            
            # Store learning data
            learning_data = {
                "prediction_id": prediction["prediction_id"],
                "symbol": prediction["symbol"],
                "accuracy": accuracy,
                "price_change": price_change,
                "market_regime": regime,
                "confidence": prediction["confidence"],
                "timestamp": datetime.utcnow(),
                "type": "prediction_learning"
            }
            
            await self.collection.insert_one(learning_data)
            
            # Check for regime changes
            await self._check_regime_changes()
            
        except Exception as e:
            logger.error(f"Error learning from prediction: {e}")
    
    async def _get_market_regime(self, symbol: str) -> str:
        """Determine current market regime"""
        try:
            # Get recent price data
            prices = await self._get_recent_prices(symbol, days=30)
            
            if len(prices) < 20:
                return "sideways"
            
            # Calculate trend
            recent_prices = prices[-20:]
            trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            # Calculate volatility
            returns = np.diff(recent_prices) / recent_prices[:-1]
            volatility = np.std(returns)
            
            # Determine regime
            if trend > 0.1 and volatility < 0.05:
                return "bull"
            elif trend < -0.1 and volatility < 0.05:
                return "bear"
            else:
                return "sideways"
                
        except Exception as e:
            logger.error(f"Error getting market regime: {e}")
            return "sideways"
    
    async def _check_regime_changes(self):
        """Check for market regime changes"""
        try:
            # Analyze regime performance
            regime_performance = {}
            for regime, data in self.market_regimes.items():
                if data["count"] > 10:  # Minimum samples
                    regime_performance[regime] = data["accuracy"]
            
            # Check for significant changes
            if len(regime_performance) >= 2:
                best_regime = max(regime_performance, key=regime_performance.get)
                worst_regime = min(regime_performance, key=regime_performance.get)
                
                if regime_performance[best_regime] - regime_performance[worst_regime] > self.regime_change_threshold:
                    logger.info(f"🔄 Market regime change detected: {worst_regime} -> {best_regime}")
                    
                    # Store regime change
                    regime_change_data = {
                        "type": "regime_change",
                        "from_regime": worst_regime,
                        "to_regime": best_regime,
                        "performance_difference": regime_performance[best_regime] - regime_performance[worst_regime],
                        "timestamp": datetime.utcnow()
                    }
                    
                    await self.collection.insert_one(regime_change_data)
                    
        except Exception as e:
            logger.error(f"Error checking regime changes: {e}")
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price from exchanges"""
        try:
            # Try different exchanges
            for exchange_name, exchange in self.exchanges.items():
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    return ticker['last']
                except:
                    continue
            
            # Fallback to yfinance
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                if not hist.empty:
                    return hist['Close'].iloc[-1]
            except:
                pass
            
            logger.warning(f"Could not get price for {symbol}")
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return 0.0
    
    async def _get_recent_prices(self, symbol: str, days: int = 30) -> List[float]:
        """Get recent price history"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d")
            
            if hist.empty:
                return []
            
            return hist['Close'].tolist()
            
        except Exception as e:
            logger.error(f"Error getting recent prices: {e}")
            return []
    
    async def get_market_insights(self) -> Dict[str, Any]:
        """Get current market insights"""
        try:
            if not self.collection:
                await self.initialize()
            
            # Get recent predictions
            recent_predictions = await self.collection.find({
                "timestamp": {"$gte": datetime.utcnow() - timedelta(days=7)},
                "status": "completed"
            }).to_list(length=100)
            
            if not recent_predictions:
                return {"message": "No recent predictions available"}
            
            # Calculate overall accuracy
            accuracies = [p["accuracy"] for p in recent_predictions if "accuracy" in p]
            overall_accuracy = np.mean(accuracies) if accuracies else 0.0
            
            # Calculate accuracy by symbol
            symbol_accuracy = {}
            for pred in recent_predictions:
                symbol = pred["symbol"]
                if symbol not in symbol_accuracy:
                    symbol_accuracy[symbol] = []
                if "accuracy" in pred:
                    symbol_accuracy[symbol].append(pred["accuracy"])
            
            symbol_accuracy = {k: np.mean(v) for k, v in symbol_accuracy.items() if v}
            
            # Get regime performance
            regime_performance = {}
            for regime, data in self.market_regimes.items():
                if data["count"] > 5:
                    regime_performance[regime] = {
                        "accuracy": data["accuracy"],
                        "count": data["count"]
                    }
            
            return {
                "overall_accuracy": overall_accuracy,
                "symbol_accuracy": symbol_accuracy,
                "regime_performance": regime_performance,
                "total_predictions": len(recent_predictions),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting market insights: {e}")
            return {"error": str(e)}
    
    async def get_optimal_timing(self, symbol: str) -> Dict[str, Any]:
        """Get optimal timing for predictions"""
        try:
            if not self.collection:
                await self.initialize()
            
            # Get predictions for this symbol
            symbol_predictions = await self.collection.find({
                "symbol": symbol,
                "status": "completed"
            }).to_list(length=100)
            
            if not symbol_predictions:
                return {"message": f"No predictions available for {symbol}"}
            
            # Analyze timing patterns
            timing_analysis = {}
            
            for pred in symbol_predictions:
                hour = pred["timestamp"].hour
                if hour not in timing_analysis:
                    timing_analysis[hour] = []
                if "accuracy" in pred:
                    timing_analysis[hour].append(pred["accuracy"])
            
            # Calculate best times
            best_times = {}
            for hour, accuracies in timing_analysis.items():
                if len(accuracies) >= 3:  # Minimum samples
                    best_times[hour] = np.mean(accuracies)
            
            # Get best hour
            best_hour = max(best_times, key=best_times.get) if best_times else None
            
            return {
                "best_hour": best_hour,
                "hourly_accuracy": best_times,
                "recommendation": f"Best time to make predictions: {best_hour}:00" if best_hour else "Insufficient data",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting optimal timing: {e}")
            return {"error": str(e)}

# Global instance
real_time_market_learning = RealTimeMarketLearning()

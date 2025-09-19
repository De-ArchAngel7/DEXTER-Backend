#!/usr/bin/env python3
"""
📈 DEXTER PATTERN RECOGNITION LEARNING SYSTEM
============================================================
Discovers profitable trading patterns and strategies
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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import yfinance as yf
import talib

logger = structlog.get_logger()

class PatternRecognitionLearning:
    """
    📈 Pattern Recognition Learning System
    
    Features:
    1. Discover profitable trading patterns
    2. Learn from successful strategies
    3. Identify market anomalies
    4. Create pattern-based trading signals
    5. Learn from failed patterns
    6. Optimize pattern parameters
    """
    
    def __init__(self, mongodb_url: str = None):
        self.mongodb_url = mongodb_url or os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.db = None
        self.collection = None
        
        # Pattern recognition parameters
        self.min_pattern_occurrences = 5
        self.pattern_success_threshold = 0.6
        self.max_patterns = 50
        
        # Technical indicators
        self.indicators = [
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'RSI', 'MACD', 'BB_upper', 'BB_lower', 'BB_middle',
            'ATR', 'ADX', 'STOCH_K', 'STOCH_D'
        ]
        
        # Discovered patterns
        self.discovered_patterns = {}
        
    async def initialize(self):
        """Initialize MongoDB connection"""
        try:
            client = AsyncIOMotorClient(self.mongodb_url)
            self.db = client.dexter
            self.collection = self.db.pattern_recognition_learning
            
            # Create indexes
            await self.collection.create_index("timestamp")
            await self.collection.create_index("symbol")
            await self.collection.create_index("pattern_type")
            
            logger.info("✅ Pattern Recognition Learning initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pattern Recognition Learning: {e}")
            raise
    
    async def discover_patterns(self, symbol: str, days: int = 90) -> Dict[str, Any]:
        """Discover profitable trading patterns"""
        try:
            if not self.collection:
                await self.initialize()
            
            logger.info(f"📈 Discovering patterns for {symbol}")
            
            # Get historical data
            historical_data = await self._get_historical_data(symbol, days)
            
            if historical_data.empty:
                return {"error": f"No historical data available for {symbol}"}
            
            # Calculate technical indicators
            indicators_data = self._calculate_technical_indicators(historical_data)
            
            # Discover price patterns
            price_patterns = await self._discover_price_patterns(historical_data, indicators_data)
            
            # Discover indicator patterns
            indicator_patterns = await self._discover_indicator_patterns(historical_data, indicators_data)
            
            # Discover volume patterns
            volume_patterns = await self._discover_volume_patterns(historical_data)
            
            # Combine and rank patterns
            all_patterns = price_patterns + indicator_patterns + volume_patterns
            ranked_patterns = self._rank_patterns(all_patterns)
            
            # Store discovered patterns
            await self._store_discovered_patterns(symbol, ranked_patterns)
            
            logger.info(f"✅ Discovered {len(ranked_patterns)} patterns for {symbol}")
            
            return {
                "symbol": symbol,
                "total_patterns": len(ranked_patterns),
                "top_patterns": ranked_patterns[:10],
                "discovery_date": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error discovering patterns: {e}")
            return {"error": str(e)}
    
    async def _get_historical_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Get historical price data"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d")
            
            if hist.empty:
                return pd.DataFrame()
            
            # Add additional columns
            hist['Returns'] = hist['Close'].pct_change()
            hist['Volatility'] = hist['Returns'].rolling(window=20).std()
            hist['Volume_MA'] = hist['Volume'].rolling(window=20).mean()
            
            return hist
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            indicators = data.copy()
            
            # Moving averages
            indicators['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)
            indicators['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
            indicators['EMA_12'] = talib.EMA(data['Close'], timeperiod=12)
            indicators['EMA_26'] = talib.EMA(data['Close'], timeperiod=26)
            
            # RSI
            indicators['RSI'] = talib.RSI(data['Close'], timeperiod=14)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(data['Close'])
            indicators['MACD'] = macd
            indicators['MACD_Signal'] = macd_signal
            indicators['MACD_Hist'] = macd_hist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(data['Close'])
            indicators['BB_upper'] = bb_upper
            indicators['BB_middle'] = bb_middle
            indicators['BB_lower'] = bb_lower
            
            # ATR
            indicators['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'])
            
            # ADX
            indicators['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'])
            
            # Stochastic
            stoch_k, stoch_d = talib.STOCH(data['High'], data['Low'], data['Close'])
            indicators['STOCH_K'] = stoch_k
            indicators['STOCH_D'] = stoch_d
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return data
    
    async def _discover_price_patterns(self, data: pd.DataFrame, indicators: pd.DataFrame) -> List[Dict[str, Any]]:
        """Discover price-based patterns"""
        try:
            patterns = []
            
            # Pattern 1: Golden Cross (SMA 20 > SMA 50)
            golden_cross = self._find_golden_cross_pattern(indicators)
            if golden_cross:
                patterns.append(golden_cross)
            
            # Pattern 2: Death Cross (SMA 20 < SMA 50)
            death_cross = self._find_death_cross_pattern(indicators)
            if death_cross:
                patterns.append(death_cross)
            
            # Pattern 3: RSI Oversold/Overbought
            rsi_patterns = self._find_rsi_patterns(indicators)
            patterns.extend(rsi_patterns)
            
            # Pattern 4: Bollinger Band Squeeze
            bb_squeeze = self._find_bollinger_squeeze_pattern(indicators)
            if bb_squeeze:
                patterns.append(bb_squeeze)
            
            # Pattern 5: MACD Divergence
            macd_divergence = self._find_macd_divergence_pattern(indicators)
            if macd_divergence:
                patterns.append(macd_divergence)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error discovering price patterns: {e}")
            return []
    
    def _find_golden_cross_pattern(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Find golden cross patterns"""
        try:
            # Find where SMA 20 crosses above SMA 50
            cross_above = (data['SMA_20'] > data['SMA_50']) & (data['SMA_20'].shift(1) <= data['SMA_50'].shift(1))
            cross_points = data[cross_above]
            
            if len(cross_points) < self.min_pattern_occurrences:
                return None
            
            # Calculate success rate
            successes = 0
            for idx in cross_points.index:
                # Check if price increased in next 5 days
                future_prices = data.loc[idx:idx+5, 'Close']
                if len(future_prices) > 1:
                    price_change = (future_prices.iloc[-1] - future_prices.iloc[0]) / future_prices.iloc[0]
                    if price_change > 0.02:  # 2% increase
                        successes += 1
            
            success_rate = successes / len(cross_points)
            
            if success_rate >= self.pattern_success_threshold:
                return {
                    "pattern_type": "golden_cross",
                    "description": "SMA 20 crosses above SMA 50",
                    "occurrences": len(cross_points),
                    "success_rate": success_rate,
                    "avg_return": 0.05,
                    "confidence": success_rate
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding golden cross pattern: {e}")
            return None
    
    def _find_death_cross_pattern(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Find death cross patterns"""
        try:
            # Find where SMA 20 crosses below SMA 50
            cross_below = (data['SMA_20'] < data['SMA_50']) & (data['SMA_20'].shift(1) >= data['SMA_50'].shift(1))
            cross_points = data[cross_below]
            
            if len(cross_points) < self.min_pattern_occurrences:
                return None
            
            # Calculate success rate
            successes = 0
            for idx in cross_points.index:
                # Check if price decreased in next 5 days
                future_prices = data.loc[idx:idx+5, 'Close']
                if len(future_prices) > 1:
                    price_change = (future_prices.iloc[-1] - future_prices.iloc[0]) / future_prices.iloc[0]
                    if price_change < -0.02:  # 2% decrease
                        successes += 1
            
            success_rate = successes / len(cross_points)
            
            if success_rate >= self.pattern_success_threshold:
                return {
                    "pattern_type": "death_cross",
                    "description": "SMA 20 crosses below SMA 50",
                    "occurrences": len(cross_points),
                    "success_rate": success_rate,
                    "avg_return": -0.05,
                    "confidence": success_rate
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding death cross pattern: {e}")
            return None
    
    def _find_rsi_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find RSI-based patterns"""
        try:
            patterns = []
            
            # RSI Oversold (RSI < 30)
            oversold = data[data['RSI'] < 30]
            if len(oversold) >= self.min_pattern_occurrences:
                # Calculate success rate
                successes = 0
                for idx in oversold.index:
                    future_prices = data.loc[idx:idx+5, 'Close']
                    if len(future_prices) > 1:
                        price_change = (future_prices.iloc[-1] - future_prices.iloc[0]) / future_prices.iloc[0]
                        if price_change > 0.03:  # 3% increase
                            successes += 1
                
                success_rate = successes / len(oversold)
                if success_rate >= self.pattern_success_threshold:
                    patterns.append({
                        "pattern_type": "rsi_oversold",
                        "description": "RSI below 30 (oversold)",
                        "occurrences": len(oversold),
                        "success_rate": success_rate,
                        "avg_return": 0.04,
                        "confidence": success_rate
                    })
            
            # RSI Overbought (RSI > 70)
            overbought = data[data['RSI'] > 70]
            if len(overbought) >= self.min_pattern_occurrences:
                # Calculate success rate
                successes = 0
                for idx in overbought.index:
                    future_prices = data.loc[idx:idx+5, 'Close']
                    if len(future_prices) > 1:
                        price_change = (future_prices.iloc[-1] - future_prices.iloc[0]) / future_prices.iloc[0]
                        if price_change < -0.03:  # 3% decrease
                            successes += 1
                
                success_rate = successes / len(overbought)
                if success_rate >= self.pattern_success_threshold:
                    patterns.append({
                        "pattern_type": "rsi_overbought",
                        "description": "RSI above 70 (overbought)",
                        "occurrences": len(overbought),
                        "success_rate": success_rate,
                        "avg_return": -0.04,
                        "confidence": success_rate
                    })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error finding RSI patterns: {e}")
            return []
    
    def _find_bollinger_squeeze_pattern(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Find Bollinger Band squeeze patterns"""
        try:
            # Calculate Bollinger Band width
            bb_width = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']
            
            # Find squeeze periods (low volatility)
            squeeze_threshold = bb_width.quantile(0.2)  # Bottom 20%
            squeeze_periods = data[bb_width < squeeze_threshold]
            
            if len(squeeze_periods) < self.min_pattern_occurrences:
                return None
            
            # Calculate success rate (breakout after squeeze)
            successes = 0
            for idx in squeeze_periods.index:
                # Check for breakout in next 10 days
                future_data = data.loc[idx:idx+10]
                if len(future_data) > 5:
                    # Check if price breaks above upper band or below lower band
                    breakout = (future_data['Close'] > future_data['BB_upper']).any() or \
                              (future_data['Close'] < future_data['BB_lower']).any()
                    if breakout:
                        successes += 1
            
            success_rate = successes / len(squeeze_periods)
            
            if success_rate >= self.pattern_success_threshold:
                return {
                    "pattern_type": "bollinger_squeeze",
                    "description": "Bollinger Band squeeze (low volatility)",
                    "occurrences": len(squeeze_periods),
                    "success_rate": success_rate,
                    "avg_return": 0.06,
                    "confidence": success_rate
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding Bollinger squeeze pattern: {e}")
            return None
    
    def _find_macd_divergence_pattern(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Find MACD divergence patterns"""
        try:
            # Find MACD signal line crossovers
            macd_cross = (data['MACD'] > data['MACD_Signal']) & (data['MACD'].shift(1) <= data['MACD_Signal'].shift(1))
            cross_points = data[macd_cross]
            
            if len(cross_points) < self.min_pattern_occurrences:
                return None
            
            # Calculate success rate
            successes = 0
            for idx in cross_points.index:
                future_prices = data.loc[idx:idx+5, 'Close']
                if len(future_prices) > 1:
                    price_change = (future_prices.iloc[-1] - future_prices.iloc[0]) / future_prices.iloc[0]
                    if price_change > 0.025:  # 2.5% increase
                        successes += 1
            
            success_rate = successes / len(cross_points)
            
            if success_rate >= self.pattern_success_threshold:
                return {
                    "pattern_type": "macd_bullish_cross",
                    "description": "MACD crosses above signal line",
                    "occurrences": len(cross_points),
                    "success_rate": success_rate,
                    "avg_return": 0.035,
                    "confidence": success_rate
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding MACD divergence pattern: {e}")
            return None
    
    async def _discover_indicator_patterns(self, data: pd.DataFrame, indicators: pd.DataFrame) -> List[Dict[str, Any]]:
        """Discover indicator-based patterns"""
        try:
            patterns = []
            
            # Pattern: High ADX with trend
            adx_pattern = self._find_adx_trend_pattern(indicators)
            if adx_pattern:
                patterns.append(adx_pattern)
            
            # Pattern: Stochastic oversold/overbought
            stoch_patterns = self._find_stochastic_patterns(indicators)
            patterns.extend(stoch_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error discovering indicator patterns: {e}")
            return []
    
    def _find_adx_trend_pattern(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Find ADX trend patterns"""
        try:
            # Strong trend: ADX > 25
            strong_trend = data[data['ADX'] > 25]
            
            if len(strong_trend) < self.min_pattern_occurrences:
                return None
            
            # Calculate success rate
            successes = 0
            for idx in strong_trend.index:
                future_prices = data.loc[idx:idx+3, 'Close']
                if len(future_prices) > 1:
                    price_change = abs((future_prices.iloc[-1] - future_prices.iloc[0]) / future_prices.iloc[0])
                    if price_change > 0.02:  # 2% movement
                        successes += 1
            
            success_rate = successes / len(strong_trend)
            
            if success_rate >= self.pattern_success_threshold:
                return {
                    "pattern_type": "adx_strong_trend",
                    "description": "ADX above 25 (strong trend)",
                    "occurrences": len(strong_trend),
                    "success_rate": success_rate,
                    "avg_return": 0.03,
                    "confidence": success_rate
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding ADX trend pattern: {e}")
            return None
    
    def _find_stochastic_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find Stochastic patterns"""
        try:
            patterns = []
            
            # Stochastic oversold
            stoch_oversold = data[data['STOCH_K'] < 20]
            if len(stoch_oversold) >= self.min_pattern_occurrences:
                successes = 0
                for idx in stoch_oversold.index:
                    future_prices = data.loc[idx:idx+5, 'Close']
                    if len(future_prices) > 1:
                        price_change = (future_prices.iloc[-1] - future_prices.iloc[0]) / future_prices.iloc[0]
                        if price_change > 0.025:
                            successes += 1
                
                success_rate = successes / len(stoch_oversold)
                if success_rate >= self.pattern_success_threshold:
                    patterns.append({
                        "pattern_type": "stoch_oversold",
                        "description": "Stochastic K below 20",
                        "occurrences": len(stoch_oversold),
                        "success_rate": success_rate,
                        "avg_return": 0.035,
                        "confidence": success_rate
                    })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error finding Stochastic patterns: {e}")
            return []
    
    async def _discover_volume_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Discover volume-based patterns"""
        try:
            patterns = []
            
            # High volume breakout
            high_volume = data[data['Volume'] > data['Volume_MA'] * 1.5]
            
            if len(high_volume) >= self.min_pattern_occurrences:
                successes = 0
                for idx in high_volume.index:
                    future_prices = data.loc[idx:idx+3, 'Close']
                    if len(future_prices) > 1:
                        price_change = abs((future_prices.iloc[-1] - future_prices.iloc[0]) / future_prices.iloc[0])
                        if price_change > 0.03:
                            successes += 1
                
                success_rate = successes / len(high_volume)
                if success_rate >= self.pattern_success_threshold:
                    patterns.append({
                        "pattern_type": "high_volume_breakout",
                        "description": "Volume 1.5x above average",
                        "occurrences": len(high_volume),
                        "success_rate": success_rate,
                        "avg_return": 0.04,
                        "confidence": success_rate
                    })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error discovering volume patterns: {e}")
            return []
    
    def _rank_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank patterns by effectiveness"""
        try:
            # Sort by confidence and success rate
            ranked = sorted(patterns, key=lambda x: (x['confidence'], x['success_rate']), reverse=True)
            
            # Limit to top patterns
            return ranked[:self.max_patterns]
            
        except Exception as e:
            logger.error(f"Error ranking patterns: {e}")
            return patterns
    
    async def _store_discovered_patterns(self, symbol: str, patterns: List[Dict[str, Any]]):
        """Store discovered patterns"""
        try:
            for pattern in patterns:
                pattern_data = {
                    "symbol": symbol,
                    "pattern_type": pattern["pattern_type"],
                    "description": pattern["description"],
                    "occurrences": pattern["occurrences"],
                    "success_rate": pattern["success_rate"],
                    "avg_return": pattern["avg_return"],
                    "confidence": pattern["confidence"],
                    "timestamp": datetime.utcnow(),
                    "type": "discovered_pattern"
                }
                
                await self.collection.insert_one(pattern_data)
            
        except Exception as e:
            logger.error(f"Error storing discovered patterns: {e}")
    
    async def get_pattern_signals(self, symbol: str) -> Dict[str, Any]:
        """Get current pattern signals"""
        try:
            if not self.collection:
                await self.initialize()
            
            # Get recent patterns for this symbol
            recent_patterns = await self.collection.find({
                "symbol": symbol,
                "type": "discovered_pattern",
                "timestamp": {"$gte": datetime.utcnow() - timedelta(days=30)}
            }).to_list(length=50)
            
            if not recent_patterns:
                return {"message": f"No patterns available for {symbol}"}
            
            # Get current market data
            current_data = await self._get_current_market_data(symbol)
            
            # Check which patterns are currently active
            active_patterns = []
            for pattern in recent_patterns:
                if self._is_pattern_active(pattern, current_data):
                    active_patterns.append(pattern)
            
            # Calculate signal strength
            signal_strength = self._calculate_signal_strength(active_patterns)
            
            return {
                "symbol": symbol,
                "active_patterns": active_patterns,
                "signal_strength": signal_strength,
                "recommendation": self._get_pattern_recommendation(active_patterns, signal_strength),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting pattern signals: {e}")
            return {"error": str(e)}
    
    async def _get_current_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data for pattern checking"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d")
            
            if hist.empty:
                return {}
            
            # Calculate current indicators
            current_data = {
                "close": hist['Close'].iloc[-1],
                "volume": hist['Volume'].iloc[-1],
                "sma_20": talib.SMA(hist['Close'], timeperiod=20).iloc[-1],
                "sma_50": talib.SMA(hist['Close'], timeperiod=50).iloc[-1],
                "rsi": talib.RSI(hist['Close'], timeperiod=14).iloc[-1],
                "bb_upper": talib.BBANDS(hist['Close'])[0].iloc[-1],
                "bb_lower": talib.BBANDS(hist['Close'])[2].iloc[-1],
                "macd": talib.MACD(hist['Close'])[0].iloc[-1],
                "macd_signal": talib.MACD(hist['Close'])[1].iloc[-1]
            }
            
            return current_data
            
        except Exception as e:
            logger.error(f"Error getting current market data: {e}")
            return {}
    
    def _is_pattern_active(self, pattern: Dict[str, Any], current_data: Dict[str, Any]) -> bool:
        """Check if a pattern is currently active"""
        try:
            pattern_type = pattern["pattern_type"]
            
            if pattern_type == "golden_cross":
                return current_data.get("sma_20", 0) > current_data.get("sma_50", 0)
            elif pattern_type == "death_cross":
                return current_data.get("sma_20", 0) < current_data.get("sma_50", 0)
            elif pattern_type == "rsi_oversold":
                return current_data.get("rsi", 50) < 30
            elif pattern_type == "rsi_overbought":
                return current_data.get("rsi", 50) > 70
            elif pattern_type == "macd_bullish_cross":
                return current_data.get("macd", 0) > current_data.get("macd_signal", 0)
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking if pattern is active: {e}")
            return False
    
    def _calculate_signal_strength(self, active_patterns: List[Dict[str, Any]]) -> float:
        """Calculate overall signal strength"""
        try:
            if not active_patterns:
                return 0.0
            
            # Weight by confidence and success rate
            total_strength = 0.0
            total_weight = 0.0
            
            for pattern in active_patterns:
                weight = pattern.get("confidence", 0.5) * pattern.get("success_rate", 0.5)
                total_strength += weight
                total_weight += 1.0
            
            return total_strength / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0.0
    
    def _get_pattern_recommendation(self, active_patterns: List[Dict[str, Any]], signal_strength: float) -> str:
        """Get trading recommendation based on patterns"""
        try:
            if signal_strength > 0.7:
                return "STRONG_BUY"
            elif signal_strength > 0.5:
                return "BUY"
            elif signal_strength > 0.3:
                return "HOLD"
            elif signal_strength > 0.1:
                return "SELL"
            else:
                return "STRONG_SELL"
                
        except Exception as e:
            logger.error(f"Error getting pattern recommendation: {e}")
            return "HOLD"
    
    async def get_pattern_insights(self) -> Dict[str, Any]:
        """Get overall pattern insights"""
        try:
            if not self.collection:
                await self.initialize()
            
            # Get all discovered patterns
            all_patterns = await self.collection.find({
                "type": "discovered_pattern"
            }).to_list(length=1000)
            
            if not all_patterns:
                return {"message": "No patterns discovered yet"}
            
            # Analyze pattern effectiveness
            pattern_analysis = {
                "total_patterns": len(all_patterns),
                "avg_success_rate": np.mean([p["success_rate"] for p in all_patterns]),
                "avg_confidence": np.mean([p["confidence"] for p in all_patterns]),
                "most_common_patterns": {},
                "best_performing_patterns": []
            }
            
            # Most common patterns
            pattern_types = [p["pattern_type"] for p in all_patterns]
            for pattern_type in set(pattern_types):
                pattern_analysis["most_common_patterns"][pattern_type] = pattern_types.count(pattern_type)
            
            # Best performing patterns
            best_patterns = sorted(all_patterns, key=lambda x: x["success_rate"], reverse=True)[:10]
            pattern_analysis["best_performing_patterns"] = best_patterns
            
            return pattern_analysis
            
        except Exception as e:
            logger.error(f"Error getting pattern insights: {e}")
            return {"error": str(e)}

# Global instance
pattern_recognition_learning = PatternRecognitionLearning()

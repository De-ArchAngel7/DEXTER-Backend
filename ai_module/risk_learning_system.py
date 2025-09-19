#!/usr/bin/env python3
"""
🎲 DEXTER RISK LEARNING SYSTEM
============================================================
Learns optimal risk parameters per user and market conditions
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger()

class RiskLearningSystem:
    """
    🎲 Risk Learning System for DEXTER
    
    Features:
    1. Learn optimal risk parameters per user
    2. Adapt to market volatility
    3. Learn from user behavior patterns
    4. Optimize position sizing
    5. Learn stop-loss and take-profit levels
    6. Risk-adjusted performance tracking
    """
    
    def __init__(self, mongodb_url: str = None):
        self.mongodb_url = mongodb_url or os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.db = None
        self.collection = None
        
        # Risk parameters
        self.default_risk_params = {
            "max_position_size": 0.02,  # 2% of portfolio
            "stop_loss": 0.05,  # 5% stop loss
            "take_profit": 0.10,  # 10% take profit
            "max_daily_loss": 0.05,  # 5% max daily loss
            "risk_tolerance": "medium"
        }
        
        # Learning parameters
        self.min_trades_for_learning = 10
        self.volatility_adjustment_factor = 0.5
        self.performance_weight = 0.3
        
        # User risk profiles
        self.user_risk_profiles = {}
        
    async def initialize(self):
        """Initialize MongoDB connection"""
        try:
            client = AsyncIOMotorClient(self.mongodb_url)
            self.db = client.dexter
            self.collection = self.db.risk_learning
            
            # Create indexes
            await self.collection.create_index("user_id")
            await self.collection.create_index("timestamp")
            await self.collection.create_index("symbol")
            
            logger.info("✅ Risk Learning System initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Risk Learning System: {e}")
            raise
    
    async def learn_from_trade(self, 
                              user_id: str,
                              symbol: str,
                              action: str,
                              position_size: float,
                              entry_price: float,
                              exit_price: float = None,
                              stop_loss: float = None,
                              take_profit: float = None,
                              profit_loss: float = None,
                              market_volatility: float = None,
                              user_behavior: Dict[str, Any] = None):
        """Learn from trade outcomes"""
        try:
            if not self.collection:
                await self.initialize()
            
            # Calculate trade metrics
            trade_metrics = self._calculate_trade_metrics(
                entry_price, exit_price, position_size, profit_loss, market_volatility
            )
            
            # Store trade data
            trade_data = {
                "user_id": user_id,
                "symbol": symbol,
                "action": action,
                "position_size": position_size,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "profit_loss": profit_loss,
                "market_volatility": market_volatility,
                "user_behavior": user_behavior or {},
                "trade_metrics": trade_metrics,
                "timestamp": datetime.utcnow(),
                "type": "trade_learning"
            }
            
            await self.collection.insert_one(trade_data)
            
            # Learn from this trade
            await self._learn_from_trade_data(user_id, trade_data)
            
            logger.info(f"🎲 Learned from trade: {symbol} {action}")
            
        except Exception as e:
            logger.error(f"Error learning from trade: {e}")
    
    def _calculate_trade_metrics(self, 
                               entry_price: float, 
                               exit_price: float, 
                               position_size: float, 
                               profit_loss: float, 
                               market_volatility: float) -> Dict[str, Any]:
        """Calculate trade performance metrics"""
        try:
            if exit_price is None or profit_loss is None:
                return {"status": "open"}
            
            # Calculate returns
            price_return = (exit_price - entry_price) / entry_price
            portfolio_return = price_return * position_size
            
            # Calculate risk metrics
            risk_adjusted_return = portfolio_return / (market_volatility or 0.01)
            
            # Calculate position sizing effectiveness
            position_effectiveness = abs(portfolio_return) / position_size
            
            # Calculate stop-loss effectiveness
            stop_loss_effectiveness = 1.0
            if profit_loss < 0:  # Loss
                stop_loss_effectiveness = min(1.0, abs(profit_loss) / (position_size * 0.05))
            
            return {
                "price_return": price_return,
                "portfolio_return": portfolio_return,
                "risk_adjusted_return": risk_adjusted_return,
                "position_effectiveness": position_effectiveness,
                "stop_loss_effectiveness": stop_loss_effectiveness,
                "success": profit_loss > 0,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade metrics: {e}")
            return {"status": "error"}
    
    async def _learn_from_trade_data(self, user_id: str, trade_data: Dict[str, Any]):
        """Learn from individual trade data"""
        try:
            # Get user's trading history
            user_trades = await self.collection.find({
                "user_id": user_id,
                "type": "trade_learning"
            }).to_list(length=100)
            
            if len(user_trades) < self.min_trades_for_learning:
                return
            
            # Analyze user's risk patterns
            risk_analysis = await self._analyze_user_risk_patterns(user_trades)
            
            # Update user risk profile
            await self._update_user_risk_profile(user_id, risk_analysis)
            
            # Learn optimal parameters
            optimal_params = await self._learn_optimal_parameters(user_trades)
            
            # Store learning results
            learning_result = {
                "user_id": user_id,
                "risk_analysis": risk_analysis,
                "optimal_parameters": optimal_params,
                "timestamp": datetime.utcnow(),
                "type": "risk_learning_result"
            }
            
            await self.collection.insert_one(learning_result)
            
        except Exception as e:
            logger.error(f"Error learning from trade data: {e}")
    
    async def _analyze_user_risk_patterns(self, user_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user's risk patterns"""
        try:
            if not user_trades:
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(user_trades)
            
            # Calculate risk metrics
            risk_metrics = {
                "avg_position_size": df["position_size"].mean(),
                "max_position_size": df["position_size"].max(),
                "position_size_volatility": df["position_size"].std(),
                "success_rate": df["trade_metrics"].apply(lambda x: x.get("success", False)).mean(),
                "avg_return": df["trade_metrics"].apply(lambda x: x.get("portfolio_return", 0)).mean(),
                "return_volatility": df["trade_metrics"].apply(lambda x: x.get("portfolio_return", 0)).std(),
                "risk_adjusted_return": df["trade_metrics"].apply(lambda x: x.get("risk_adjusted_return", 0)).mean(),
                "stop_loss_usage": df["stop_loss"].notna().mean(),
                "take_profit_usage": df["take_profit"].notna().mean()
            }
            
            # Determine risk tolerance
            risk_tolerance = self._determine_risk_tolerance(risk_metrics)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(risk_metrics)
            
            return {
                "risk_metrics": risk_metrics,
                "risk_tolerance": risk_tolerance,
                "risk_score": risk_score,
                "total_trades": len(user_trades)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing user risk patterns: {e}")
            return {}
    
    def _determine_risk_tolerance(self, risk_metrics: Dict[str, Any]) -> str:
        """Determine user's risk tolerance"""
        try:
            avg_position_size = risk_metrics.get("avg_position_size", 0.02)
            return_volatility = risk_metrics.get("return_volatility", 0.05)
            success_rate = risk_metrics.get("success_rate", 0.5)
            
            # Calculate risk score
            risk_score = (avg_position_size * 10) + (return_volatility * 20) + ((1 - success_rate) * 10)
            
            if risk_score < 2:
                return "conservative"
            elif risk_score < 4:
                return "moderate"
            elif risk_score < 6:
                return "aggressive"
            else:
                return "very_aggressive"
                
        except Exception as e:
            logger.error(f"Error determining risk tolerance: {e}")
            return "moderate"
    
    def _calculate_risk_score(self, risk_metrics: Dict[str, Any]) -> float:
        """Calculate overall risk score (0-10)"""
        try:
            # Factors that increase risk score
            position_size_factor = min(risk_metrics.get("avg_position_size", 0.02) * 50, 3)
            volatility_factor = min(risk_metrics.get("return_volatility", 0.05) * 20, 3)
            failure_rate_factor = (1 - risk_metrics.get("success_rate", 0.5)) * 4
            
            risk_score = position_size_factor + volatility_factor + failure_rate_factor
            return min(10.0, max(0.0, risk_score))
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 5.0
    
    async def _update_user_risk_profile(self, user_id: str, risk_analysis: Dict[str, Any]):
        """Update user's risk profile"""
        try:
            self.user_risk_profiles[user_id] = {
                "risk_tolerance": risk_analysis.get("risk_tolerance", "moderate"),
                "risk_score": risk_analysis.get("risk_score", 5.0),
                "risk_metrics": risk_analysis.get("risk_metrics", {}),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating user risk profile: {e}")
    
    async def _learn_optimal_parameters(self, user_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Learn optimal risk parameters from user's trading history"""
        try:
            if len(user_trades) < 5:
                return self.default_risk_params
            
            # Analyze successful trades
            successful_trades = [t for t in user_trades if t.get("profit_loss", 0) > 0]
            
            if not successful_trades:
                return self.default_risk_params
            
            # Calculate optimal parameters
            optimal_params = {}
            
            # Optimal position size
            successful_position_sizes = [t["position_size"] for t in successful_trades]
            optimal_params["max_position_size"] = min(0.05, max(0.01, np.mean(successful_position_sizes)))
            
            # Optimal stop loss
            successful_stop_losses = [t["stop_loss"] for t in successful_trades if t.get("stop_loss")]
            if successful_stop_losses:
                optimal_params["stop_loss"] = np.mean(successful_stop_losses)
            else:
                optimal_params["stop_loss"] = 0.05
            
            # Optimal take profit
            successful_take_profits = [t["take_profit"] for t in successful_trades if t.get("take_profit")]
            if successful_take_profits:
                optimal_params["take_profit"] = np.mean(successful_take_profits)
            else:
                optimal_params["take_profit"] = 0.10
            
            # Risk tolerance
            risk_tolerance = self._determine_risk_tolerance_from_trades(user_trades)
            optimal_params["risk_tolerance"] = risk_tolerance
            
            return optimal_params
            
        except Exception as e:
            logger.error(f"Error learning optimal parameters: {e}")
            return self.default_risk_params
    
    def _determine_risk_tolerance_from_trades(self, user_trades: List[Dict[str, Any]]) -> str:
        """Determine risk tolerance from trading history"""
        try:
            position_sizes = [t["position_size"] for t in user_trades]
            avg_position_size = np.mean(position_sizes)
            
            if avg_position_size < 0.015:
                return "conservative"
            elif avg_position_size < 0.025:
                return "moderate"
            elif avg_position_size < 0.035:
                return "aggressive"
            else:
                return "very_aggressive"
                
        except Exception as e:
            logger.error(f"Error determining risk tolerance from trades: {e}")
            return "moderate"
    
    async def get_user_risk_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user's current risk profile"""
        try:
            if user_id in self.user_risk_profiles:
                return self.user_risk_profiles[user_id]
            
            # Load from database
            user_trades = await self.collection.find({
                "user_id": user_id,
                "type": "trade_learning"
            }).to_list(length=100)
            
            if not user_trades:
                return self.default_risk_params
            
            # Analyze user's risk patterns
            risk_analysis = await self._analyze_user_risk_patterns(user_trades)
            
            # Update profile
            await self._update_user_risk_profile(user_id, risk_analysis)
            
            return self.user_risk_profiles.get(user_id, self.default_risk_params)
            
        except Exception as e:
            logger.error(f"Error getting user risk profile: {e}")
            return self.default_risk_params
    
    async def get_risk_recommendations(self, user_id: str, symbol: str, market_volatility: float = None) -> Dict[str, Any]:
        """Get personalized risk recommendations"""
        try:
            # Get user's risk profile
            risk_profile = await self.get_user_risk_profile(user_id)
            
            # Get market conditions
            market_conditions = await self._get_market_conditions(symbol, market_volatility)
            
            # Calculate recommendations
            recommendations = self._calculate_risk_recommendations(risk_profile, market_conditions)
            
            return {
                "user_risk_profile": risk_profile,
                "market_conditions": market_conditions,
                "recommendations": recommendations,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting risk recommendations: {e}")
            return {"error": str(e)}
    
    async def _get_market_conditions(self, symbol: str, market_volatility: float = None) -> Dict[str, Any]:
        """Get current market conditions"""
        try:
            # Get recent trades for this symbol
            recent_trades = await self.collection.find({
                "symbol": symbol,
                "timestamp": {"$gte": datetime.utcnow() - timedelta(days=7)}
            }).to_list(length=50)
            
            if not recent_trades:
                return {"volatility": market_volatility or 0.05, "trend": "neutral"}
            
            # Calculate market metrics
            volatilities = [t.get("market_volatility", 0.05) for t in recent_trades]
            avg_volatility = np.mean(volatilities) if volatilities else (market_volatility or 0.05)
            
            # Determine trend
            profits = [t.get("profit_loss", 0) for t in recent_trades if t.get("profit_loss")]
            if profits:
                avg_profit = np.mean(profits)
                if avg_profit > 0.01:
                    trend = "bullish"
                elif avg_profit < -0.01:
                    trend = "bearish"
                else:
                    trend = "neutral"
            else:
                trend = "neutral"
            
            return {
                "volatility": avg_volatility,
                "trend": trend,
                "recent_trades": len(recent_trades)
            }
            
        except Exception as e:
            logger.error(f"Error getting market conditions: {e}")
            return {"volatility": market_volatility or 0.05, "trend": "neutral"}
    
    def _calculate_risk_recommendations(self, risk_profile: Dict[str, Any], market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate personalized risk recommendations"""
        try:
            base_params = risk_profile.get("risk_metrics", {})
            market_volatility = market_conditions.get("volatility", 0.05)
            market_trend = market_conditions.get("trend", "neutral")
            
            # Adjust position size based on volatility
            base_position_size = base_params.get("avg_position_size", 0.02)
            volatility_adjustment = 1 - (market_volatility - 0.05) * self.volatility_adjustment_factor
            recommended_position_size = max(0.005, min(0.05, base_position_size * volatility_adjustment))
            
            # Adjust stop loss based on volatility
            base_stop_loss = 0.05
            volatility_stop_loss = base_stop_loss * (1 + market_volatility)
            recommended_stop_loss = min(0.15, max(0.02, volatility_stop_loss))
            
            # Adjust take profit based on trend
            base_take_profit = 0.10
            if market_trend == "bullish":
                recommended_take_profit = base_take_profit * 1.2
            elif market_trend == "bearish":
                recommended_take_profit = base_take_profit * 0.8
            else:
                recommended_take_profit = base_take_profit
            
            return {
                "position_size": recommended_position_size,
                "stop_loss": recommended_stop_loss,
                "take_profit": recommended_take_profit,
                "max_daily_loss": 0.05,
                "risk_level": risk_profile.get("risk_tolerance", "moderate"),
                "confidence": 0.8
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk recommendations: {e}")
            return self.default_risk_params
    
    async def get_risk_insights(self) -> Dict[str, Any]:
        """Get overall risk insights"""
        try:
            if not self.collection:
                await self.initialize()
            
            # Get all risk learning results
            risk_results = await self.collection.find({
                "type": "risk_learning_result"
            }).to_list(length=100)
            
            if not risk_results:
                return {"message": "No risk learning data available"}
            
            # Analyze risk patterns
            risk_analysis = {
                "total_users": len(set(r["user_id"] for r in risk_results)),
                "avg_risk_score": np.mean([r["risk_analysis"].get("risk_score", 5.0) for r in risk_results]),
                "risk_tolerance_distribution": {},
                "common_risk_patterns": []
            }
            
            # Risk tolerance distribution
            for result in risk_results:
                tolerance = result["risk_analysis"].get("risk_tolerance", "moderate")
                risk_analysis["risk_tolerance_distribution"][tolerance] = \
                    risk_analysis["risk_tolerance_distribution"].get(tolerance, 0) + 1
            
            return risk_analysis
            
        except Exception as e:
            logger.error(f"Error getting risk insights: {e}")
            return {"error": str(e)}

# Global instance
risk_learning_system = RiskLearningSystem()

#!/usr/bin/env python3
"""
🧠 DEXTER TRADE LEARNING SYSTEM
============================================================
Learns from past trades to improve AI models and trading strategies
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
from pymongo import MongoClient
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logger = structlog.get_logger()

class TradeLearningSystem:
    """
    🧠 Trade Learning System for DEXTER
    
    Learns from:
    1. Successful vs failed trades
    2. Market conditions during trades
    3. User feedback on AI advice
    4. Price prediction accuracy
    5. Sentiment analysis accuracy
    """
    
    def __init__(self, mongodb_url: str = None):
        self.mongodb_url = mongodb_url or os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.db = None
        self.collection = None
        
        # Learning parameters
        self.learning_rate = 0.01
        self.min_trades_for_learning = 10
        self.retrain_threshold = 100  # Retrain after 100 new trades
        
        # Performance tracking
        self.performance_metrics = {
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "accuracy_rate": 0.0,
            "avg_profit": 0.0,
            "last_retrain": None
        }
        
    async def initialize(self):
        """Initialize MongoDB connection"""
        try:
            client = AsyncIOMotorClient(self.mongodb_url)
            self.db = client.dexter
            self.collection = self.db.trade_learning
            
            # Create indexes for better performance
            await self.collection.create_index("timestamp")
            await self.collection.create_index("symbol")
            await self.collection.create_index("success")
            
            logger.info("✅ Trade Learning System initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Trade Learning System: {e}")
            raise
    
    async def log_trade(self, 
                       user_id: str,
                       symbol: str,
                       action: str,
                       entry_price: float,
                       exit_price: float = None,
                       quantity: float = 1.0,
                       market_conditions: Dict[str, Any] = None,
                       ai_confidence: float = None,
                       ai_reasoning: str = None,
                       success: bool = None,
                       profit_loss: float = None,
                       user_feedback: str = None):
        """
        Log a trade for learning purposes
        
        Args:
            user_id: User who made the trade
            symbol: Trading symbol (e.g., BTC, ETH)
            action: BUY, SELL, HOLD
            entry_price: Price when trade was entered
            exit_price: Price when trade was exited (if completed)
            quantity: Amount traded
            market_conditions: Market data at time of trade
            ai_confidence: AI confidence level
            ai_reasoning: AI reasoning for the trade
            success: Whether trade was successful (if known)
            profit_loss: Profit/loss amount (if known)
            user_feedback: User feedback on the trade
        """
        
        if not self.collection:
            await self.initialize()
        
        trade_data = {
            "user_id": user_id,
            "symbol": symbol,
            "action": action,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "market_conditions": market_conditions or {},
            "ai_confidence": ai_confidence,
            "ai_reasoning": ai_reasoning,
            "success": success,
            "profit_loss": profit_loss,
            "user_feedback": user_feedback,
            "timestamp": datetime.utcnow(),
            "created_at": datetime.utcnow().isoformat()
        }
        
        try:
            result = await self.collection.insert_one(trade_data)
            logger.info(f"📊 Trade logged: {symbol} {action} at {entry_price}")
            
            # Check if we should trigger learning
            await self._check_learning_trigger()
            
            return result.inserted_id
            
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
            return None
    
    async def log_ai_prediction(self,
                               user_id: str,
                               symbol: str,
                               prediction: str,
                               confidence: float,
                               actual_outcome: str = None,
                               accuracy: float = None):
        """Log AI prediction for accuracy tracking"""
        
        if not self.collection:
            await self.initialize()
        
        prediction_data = {
            "user_id": user_id,
            "symbol": symbol,
            "prediction": prediction,
            "confidence": confidence,
            "actual_outcome": actual_outcome,
            "accuracy": accuracy,
            "timestamp": datetime.utcnow(),
            "type": "prediction"
        }
        
        try:
            await self.collection.insert_one(prediction_data)
            logger.info(f"🔮 AI prediction logged: {symbol} - {prediction}")
        except Exception as e:
            logger.error(f"Failed to log AI prediction: {e}")
    
    async def log_user_feedback(self,
                               user_id: str,
                               conversation_id: str,
                               ai_response: str,
                               user_rating: int,  # 1-5 scale
                               feedback_text: str = None):
        """Log user feedback on AI responses"""
        
        if not self.collection:
            await self.initialize()
        
        feedback_data = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "ai_response": ai_response,
            "user_rating": user_rating,
            "feedback_text": feedback_text,
            "timestamp": datetime.utcnow(),
            "type": "feedback"
        }
        
        try:
            await self.collection.insert_one(feedback_data)
            logger.info(f"💬 User feedback logged: Rating {user_rating}/5")
        except Exception as e:
            logger.error(f"Failed to log user feedback: {e}")
    
    async def _check_learning_trigger(self):
        """Check if we should trigger model learning/retraining"""
        try:
            # Count recent trades
            recent_trades = await self.collection.count_documents({
                "timestamp": {"$gte": datetime.utcnow() - timedelta(days=7)},
                "type": {"$ne": "feedback"}
            })
            
            if recent_trades >= self.retrain_threshold:
                logger.info(f"🔄 Learning trigger activated: {recent_trades} recent trades")
                await self.trigger_model_learning()
                
        except Exception as e:
            logger.error(f"Error checking learning trigger: {e}")
    
    async def trigger_model_learning(self):
        """Trigger model learning based on recent trades"""
        try:
            logger.info("🧠 Starting model learning process...")
            
            # Analyze recent trades
            trade_analysis = await self._analyze_recent_trades()
            
            # Update performance metrics
            await self._update_performance_metrics(trade_analysis)
            
            # Generate learning insights
            insights = await self._generate_learning_insights(trade_analysis)
            
            # Store learning results
            await self._store_learning_results(insights)
            
            logger.info("✅ Model learning completed")
            
        except Exception as e:
            logger.error(f"Error in model learning: {e}")
    
    async def _analyze_recent_trades(self) -> Dict[str, Any]:
        """Analyze recent trades for patterns"""
        try:
            # Get trades from last 30 days
            recent_trades = await self.collection.find({
                "timestamp": {"$gte": datetime.utcnow() - timedelta(days=30)},
                "type": {"$ne": "feedback"}
            }).to_list(length=1000)
            
            if not recent_trades:
                return {"error": "No recent trades to analyze"}
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(recent_trades)
            
            # Calculate success rates by symbol
            success_by_symbol = df.groupby('symbol')['success'].mean().to_dict()
            
            # Calculate success rates by market conditions
            success_by_conditions = {}
            if 'market_conditions' in df.columns:
                for idx, row in df.iterrows():
                    if row['market_conditions'] and isinstance(row['market_conditions'], dict):
                        for condition, value in row['market_conditions'].items():
                            if condition not in success_by_conditions:
                                success_by_conditions[condition] = {}
                            if value not in success_by_conditions[condition]:
                                success_by_conditions[condition][value] = []
                            success_by_conditions[condition][value].append(row['success'])
            
            # Calculate AI confidence vs success correlation
            confidence_success = df[['ai_confidence', 'success']].corr().iloc[0, 1] if 'ai_confidence' in df.columns else 0
            
            return {
                "total_trades": len(recent_trades),
                "success_rate": df['success'].mean() if 'success' in df.columns else 0,
                "success_by_symbol": success_by_symbol,
                "success_by_conditions": success_by_conditions,
                "confidence_success_correlation": confidence_success,
                "avg_profit": df['profit_loss'].mean() if 'profit_loss' in df.columns else 0,
                "most_traded_symbols": df['symbol'].value_counts().head(5).to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trades: {e}")
            return {"error": str(e)}
    
    async def _update_performance_metrics(self, analysis: Dict[str, Any]):
        """Update performance metrics based on analysis"""
        try:
            self.performance_metrics.update({
                "total_trades": analysis.get("total_trades", 0),
                "successful_trades": int(analysis.get("total_trades", 0) * analysis.get("success_rate", 0)),
                "failed_trades": int(analysis.get("total_trades", 0) * (1 - analysis.get("success_rate", 0))),
                "accuracy_rate": analysis.get("success_rate", 0),
                "avg_profit": analysis.get("avg_profit", 0),
                "last_retrain": datetime.utcnow().isoformat()
            })
            
            logger.info(f"📊 Performance metrics updated: {self.performance_metrics}")
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _generate_learning_insights(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights for model improvement"""
        try:
            insights = {
                "timestamp": datetime.utcnow().isoformat(),
                "analysis": analysis,
                "recommendations": [],
                "model_updates": []
            }
            
            # Generate recommendations based on analysis
            if analysis.get("success_rate", 0) < 0.6:
                insights["recommendations"].append("Low success rate - consider adjusting risk parameters")
            
            if analysis.get("confidence_success_correlation", 0) < 0.3:
                insights["recommendations"].append("AI confidence not correlating with success - retrain confidence model")
            
            # Find best performing symbols
            best_symbols = [k for k, v in analysis.get("success_by_symbol", {}).items() if v > 0.7]
            if best_symbols:
                insights["recommendations"].append(f"Focus on high-performing symbols: {', '.join(best_symbols)}")
            
            # Model update suggestions
            if analysis.get("total_trades", 0) > 50:
                insights["model_updates"].append("Sufficient data for model retraining")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {"error": str(e)}
    
    async def _store_learning_results(self, insights: Dict[str, Any]):
        """Store learning results for future reference"""
        try:
            learning_data = {
                "insights": insights,
                "performance_metrics": self.performance_metrics,
                "timestamp": datetime.utcnow(),
                "type": "learning_results"
            }
            
            await self.collection.insert_one(learning_data)
            logger.info("💾 Learning results stored")
            
        except Exception as e:
            logger.error(f"Error storing learning results: {e}")
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        try:
            # Get recent performance data
            recent_data = await self.collection.find({
                "timestamp": {"$gte": datetime.utcnow() - timedelta(days=7)},
                "type": {"$ne": "feedback"}
            }).to_list(length=100)
            
            if not recent_data:
                return {"message": "No recent data available"}
            
            df = pd.DataFrame(recent_data)
            
            summary = {
                "total_trades_7d": len(recent_data),
                "success_rate_7d": df['success'].mean() if 'success' in df.columns else 0,
                "avg_profit_7d": df['profit_loss'].mean() if 'profit_loss' in df.columns else 0,
                "top_symbols": df['symbol'].value_counts().head(3).to_dict(),
                "ai_confidence_avg": df['ai_confidence'].mean() if 'ai_confidence' in df.columns else 0,
                "last_learning": self.performance_metrics.get("last_retrain"),
                "overall_metrics": self.performance_metrics
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}
    
    async def export_learning_data(self, days: int = 30) -> Dict[str, Any]:
        """Export learning data for analysis"""
        try:
            # Get data from specified period
            start_date = datetime.utcnow() - timedelta(days=days)
            data = await self.collection.find({
                "timestamp": {"$gte": start_date}
            }).to_list(length=10000)
            
            # Convert to exportable format
            export_data = {
                "export_date": datetime.utcnow().isoformat(),
                "period_days": days,
                "total_records": len(data),
                "data": data,
                "performance_metrics": self.performance_metrics
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting learning data: {e}")
            return {"error": str(e)}

# Global instance
trade_learning_system = TradeLearningSystem()

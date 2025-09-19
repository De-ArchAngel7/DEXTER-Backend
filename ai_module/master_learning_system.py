#!/usr/bin/env python3
"""
🧠 DEXTER MASTER LEARNING SYSTEM
============================================================
Orchestrates all learning systems for maximum intelligence
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import structlog

# Import all learning systems
from .real_time_market_learning import real_time_market_learning
from .risk_learning_system import risk_learning_system
from .pattern_recognition_learning import pattern_recognition_learning
from .meta_learning_system import meta_learning_system
from .collaborative_learning_system import collaborative_learning_system
from .personalized_learning_system import personalized_learning_system
from .predictive_learning_system import predictive_learning_system

logger = structlog.get_logger()

class MasterLearningSystem:
    """
    🧠 Master Learning System for DEXTER
    
    Orchestrates all learning systems:
    1. Real-Time Market Learning
    2. Risk Learning System
    3. Pattern Recognition Learning
    4. Meta-Learning System
    5. Collaborative Learning System
    6. Personalized Learning System
    7. Predictive Learning System
    """
    
    def __init__(self):
        self.learning_systems = {
            "real_time_market": real_time_market_learning,
            "risk_learning": risk_learning_system,
            "pattern_recognition": pattern_recognition_learning,
            "meta_learning": meta_learning_system,
            "collaborative_learning": collaborative_learning_system,
            "personalized_learning": personalized_learning_system,
            "predictive_learning": predictive_learning_system
        }
        
        # Learning orchestration parameters
        self.learning_schedule = {
            "real_time_market": "continuous",
            "risk_learning": "per_trade",
            "pattern_recognition": "daily",
            "meta_learning": "weekly",
            "collaborative_learning": "per_strategy_share",
            "personalized_learning": "per_interaction",
            "predictive_learning": "per_prediction"
        }
        
        # System status
        self.system_status = {}
        self.learning_performance = {}
        
    async def initialize(self):
        """Initialize all learning systems"""
        try:
            logger.info("🧠 Initializing Master Learning System")
            
            # Initialize all learning systems
            for system_name, system in self.learning_systems.items():
                try:
                    await system.initialize()
                    self.system_status[system_name] = "initialized"
                    logger.info(f"✅ {system_name} initialized")
                except Exception as e:
                    self.system_status[system_name] = f"error: {e}"
                    logger.error(f"❌ {system_name} initialization failed: {e}")
            
            logger.info("🧠 Master Learning System initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Master Learning System: {e}")
            raise
    
    async def learn_from_trade(self, 
                              user_id: str,
                              symbol: str,
                              trade_data: Dict[str, Any],
                              outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from a trade across all systems"""
        try:
            learning_results = {}
            
            # Real-time market learning
            if self.system_status.get("real_time_market") == "initialized":
                try:
                    await real_time_market_learning.track_prediction_accuracy(
                        prediction_id=trade_data.get("prediction_id", ""),
                        symbol=symbol,
                        prediction=trade_data.get("prediction", ""),
                        confidence=trade_data.get("confidence", 0.5)
                    )
                    learning_results["real_time_market"] = "success"
                except Exception as e:
                    learning_results["real_time_market"] = f"error: {e}"
            
            # Risk learning
            if self.system_status.get("risk_learning") == "initialized":
                try:
                    await risk_learning_system.learn_from_trade(
                        user_id=user_id,
                        symbol=symbol,
                        action=trade_data.get("action", "BUY"),
                        position_size=trade_data.get("position_size", 0.02),
                        entry_price=trade_data.get("entry_price", 0.0),
                        exit_price=trade_data.get("exit_price"),
                        profit_loss=outcome.get("profit_loss"),
                        market_volatility=trade_data.get("market_volatility")
                    )
                    learning_results["risk_learning"] = "success"
                except Exception as e:
                    learning_results["risk_learning"] = f"error: {e}"
            
            # Pattern recognition learning
            if self.system_status.get("pattern_recognition") == "initialized":
                try:
                    # Share successful strategy if profitable
                    if outcome.get("profit_loss", 0) > 0:
                        await collaborative_learning_system.share_successful_strategy(
                            user_id=user_id,
                            strategy_data=trade_data,
                            performance_metrics=outcome
                        )
                    learning_results["pattern_recognition"] = "success"
                except Exception as e:
                    learning_results["pattern_recognition"] = f"error: {e}"
            
            # Personalized learning
            if self.system_status.get("personalized_learning") == "initialized":
                try:
                    await personalized_learning_system.learn_from_user_interaction(
                        user_id=user_id,
                        interaction_type="trade",
                        interaction_data=trade_data,
                        user_feedback=outcome.get("user_feedback")
                    )
                    learning_results["personalized_learning"] = "success"
                except Exception as e:
                    learning_results["personalized_learning"] = f"error: {e}"
            
            logger.info(f"🧠 Learned from trade: {symbol} {trade_data.get('action', 'BUY')}")
            
            return {
                "learning_results": learning_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error learning from trade: {e}")
            return {"error": str(e)}
    
    async def get_comprehensive_insights(self, user_id: str, symbol: str = None) -> Dict[str, Any]:
        """Get comprehensive insights from all learning systems"""
        try:
            insights = {}
            
            # Real-time market insights
            if self.system_status.get("real_time_market") == "initialized":
                try:
                    insights["market_insights"] = await real_time_market_learning.get_market_insights()
                except Exception as e:
                    insights["market_insights"] = {"error": str(e)}
            
            # Risk insights
            if self.system_status.get("risk_learning") == "initialized":
                try:
                    insights["risk_insights"] = await risk_learning_system.get_risk_insights()
                except Exception as e:
                    insights["risk_insights"] = {"error": str(e)}
            
            # Pattern insights
            if self.system_status.get("pattern_recognition") == "initialized":
                try:
                    insights["pattern_insights"] = await pattern_recognition_learning.get_pattern_insights()
                except Exception as e:
                    insights["pattern_insights"] = {"error": str(e)}
            
            # Meta-learning insights
            if self.system_status.get("meta_learning") == "initialized":
                try:
                    insights["meta_learning_insights"] = await meta_learning_system.get_meta_learning_insights()
                except Exception as e:
                    insights["meta_learning_insights"] = {"error": str(e)}
            
            # Collaborative insights
            if self.system_status.get("collaborative_learning") == "initialized":
                try:
                    insights["collaborative_insights"] = await collaborative_learning_system.get_collaborative_insights()
                except Exception as e:
                    insights["collaborative_insights"] = {"error": str(e)}
            
            # Personalized insights
            if self.system_status.get("personalized_learning") == "initialized":
                try:
                    insights["personalized_insights"] = await personalized_learning_system.get_personalization_insights()
                except Exception as e:
                    insights["personalized_insights"] = {"error": str(e)}
            
            # Predictive insights
            if self.system_status.get("predictive_learning") == "initialized":
                try:
                    insights["predictive_insights"] = await predictive_learning_system.get_predictive_insights()
                except Exception as e:
                    insights["predictive_insights"] = {"error": str(e)}
            
            # User-specific insights
            if user_id:
                user_insights = await self._get_user_specific_insights(user_id, symbol)
                insights["user_insights"] = user_insights
            
            return {
                "comprehensive_insights": insights,
                "system_status": self.system_status,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive insights: {e}")
            return {"error": str(e)}
    
    async def _get_user_specific_insights(self, user_id: str, symbol: str = None) -> Dict[str, Any]:
        """Get user-specific insights"""
        try:
            user_insights = {}
            
            # Risk recommendations
            if self.system_status.get("risk_learning") == "initialized":
                try:
                    risk_recommendations = await risk_learning_system.get_risk_recommendations(user_id, symbol or "BTC")
                    user_insights["risk_recommendations"] = risk_recommendations
                except Exception as e:
                    user_insights["risk_recommendations"] = {"error": str(e)}
            
            # Personalized recommendations
            if self.system_status.get("personalized_learning") == "initialized":
                try:
                    personalized_recs = await personalized_learning_system.get_personalized_recommendations(user_id)
                    user_insights["personalized_recommendations"] = personalized_recs
                except Exception as e:
                    user_insights["personalized_recommendations"] = {"error": str(e)}
            
            # Collaborative recommendations
            if self.system_status.get("collaborative_learning") == "initialized":
                try:
                    collaborative_recs = await collaborative_learning_system.learn_from_community(user_id)
                    user_insights["collaborative_recommendations"] = collaborative_recs
                except Exception as e:
                    user_insights["collaborative_recommendations"] = {"error": str(e)}
            
            # Predictive recommendations
            if symbol and self.system_status.get("predictive_learning") == "initialized":
                try:
                    market_conditions = await predictive_learning_system.predict_market_conditions(symbol)
                    volatility_prediction = await predictive_learning_system.predict_volatility(symbol)
                    optimal_timing = await predictive_learning_system.predict_optimal_timing(symbol, user_id)
                    
                    user_insights["predictive_recommendations"] = {
                        "market_conditions": market_conditions,
                        "volatility_prediction": volatility_prediction,
                        "optimal_timing": optimal_timing
                    }
                except Exception as e:
                    user_insights["predictive_recommendations"] = {"error": str(e)}
            
            return user_insights
            
        except Exception as e:
            logger.error(f"Error getting user-specific insights: {e}")
            return {"error": str(e)}
    
    async def optimize_learning_performance(self) -> Dict[str, Any]:
        """Optimize learning performance across all systems"""
        try:
            logger.info("🧠 Optimizing learning performance")
            
            optimization_results = {}
            
            # Meta-learning optimization
            if self.system_status.get("meta_learning") == "initialized":
                try:
                    # Get performance data from all systems
                    performance_data = await self._collect_performance_data()
                    
                    # Optimize parameters
                    optimization_result = await meta_learning_system.optimize_learning_parameters(
                        "comprehensive", performance_data
                    )
                    optimization_results["meta_learning"] = optimization_result
                except Exception as e:
                    optimization_results["meta_learning"] = {"error": str(e)}
            
            # Pattern discovery
            if self.system_status.get("pattern_recognition") == "initialized":
                try:
                    # Discover patterns for major symbols
                    major_symbols = ["BTC", "ETH", "BNB", "ADA", "SOL"]
                    pattern_results = {}
                    
                    for symbol in major_symbols:
                        pattern_result = await pattern_recognition_learning.discover_patterns(symbol)
                        pattern_results[symbol] = pattern_result
                    
                    optimization_results["pattern_discovery"] = pattern_results
                except Exception as e:
                    optimization_results["pattern_discovery"] = {"error": str(e)}
            
            # Collaborative learning optimization
            if self.system_status.get("collaborative_learning") == "initialized":
                try:
                    # Analyze community performance
                    community_insights = await collaborative_learning_system.get_collaborative_insights()
                    optimization_results["collaborative_optimization"] = community_insights
                except Exception as e:
                    optimization_results["collaborative_optimization"] = {"error": str(e)}
            
            logger.info("✅ Learning performance optimization completed")
            
            return {
                "optimization_results": optimization_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing learning performance: {e}")
            return {"error": str(e)}
    
    async def _collect_performance_data(self) -> List[Dict[str, Any]]:
        """Collect performance data from all systems"""
        try:
            performance_data = []
            
            # Collect from each system
            for system_name, system in self.learning_systems.items():
                try:
                    if hasattr(system, 'get_performance_summary'):
                        performance = await system.get_performance_summary()
                        performance_data.append({
                            "system": system_name,
                            "performance": performance,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                except Exception as e:
                    logger.warning(f"Could not collect performance from {system_name}: {e}")
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Error collecting performance data: {e}")
            return []
    
    async def get_learning_health_status(self) -> Dict[str, Any]:
        """Get overall learning system health status"""
        try:
            health_status = {
                "overall_health": "healthy",
                "system_status": self.system_status,
                "active_systems": 0,
                "failed_systems": 0,
                "recommendations": []
            }
            
            # Count active and failed systems
            for system_name, status in self.system_status.items():
                if status == "initialized":
                    health_status["active_systems"] += 1
                else:
                    health_status["failed_systems"] += 1
                    health_status["recommendations"].append(f"Fix {system_name}: {status}")
            
            # Determine overall health
            total_systems = len(self.system_status)
            if health_status["active_systems"] == total_systems:
                health_status["overall_health"] = "excellent"
            elif health_status["active_systems"] >= total_systems * 0.8:
                health_status["overall_health"] = "good"
            elif health_status["active_systems"] >= total_systems * 0.5:
                health_status["overall_health"] = "fair"
            else:
                health_status["overall_health"] = "poor"
            
            # Add general recommendations
            if health_status["active_systems"] < total_systems:
                health_status["recommendations"].append("Some learning systems are not functioning properly")
            
            if health_status["active_systems"] == 0:
                health_status["recommendations"].append("All learning systems are down - immediate attention required")
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error getting learning health status: {e}")
            return {"overall_health": "unknown", "error": str(e)}
    
    async def trigger_learning_cycle(self) -> Dict[str, Any]:
        """Trigger a complete learning cycle"""
        try:
            logger.info("🧠 Triggering complete learning cycle")
            
            cycle_results = {}
            
            # 1. Collect new data
            cycle_results["data_collection"] = await self._collect_new_learning_data()
            
            # 2. Update all models
            cycle_results["model_updates"] = await self._update_all_models()
            
            # 3. Optimize performance
            cycle_results["optimization"] = await self.optimize_learning_performance()
            
            # 4. Generate insights
            cycle_results["insights"] = await self._generate_cycle_insights()
            
            logger.info("✅ Complete learning cycle completed")
            
            return {
                "cycle_results": cycle_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error triggering learning cycle: {e}")
            return {"error": str(e)}
    
    async def _collect_new_learning_data(self) -> Dict[str, Any]:
        """Collect new learning data"""
        try:
            data_collection = {}
            
            # Collect from each system
            for system_name, system in self.learning_systems.items():
                try:
                    if hasattr(system, 'collect_learning_data'):
                        data = await system.collect_learning_data()
                        data_collection[system_name] = data
                except Exception as e:
                    data_collection[system_name] = {"error": str(e)}
            
            return data_collection
            
        except Exception as e:
            logger.error(f"Error collecting new learning data: {e}")
            return {"error": str(e)}
    
    async def _update_all_models(self) -> Dict[str, Any]:
        """Update all learning models"""
        try:
            model_updates = {}
            
            # Update each system's models
            for system_name, system in self.learning_systems.items():
                try:
                    if hasattr(system, 'update_models'):
                        update_result = await system.update_models()
                        model_updates[system_name] = update_result
                except Exception as e:
                    model_updates[system_name] = {"error": str(e)}
            
            return model_updates
            
        except Exception as e:
            logger.error(f"Error updating all models: {e}")
            return {"error": str(e)}
    
    async def _generate_cycle_insights(self) -> Dict[str, Any]:
        """Generate insights from learning cycle"""
        try:
            insights = {
                "learning_progress": {},
                "performance_improvements": {},
                "new_discoveries": {},
                "recommendations": []
            }
            
            # Generate insights from each system
            for system_name, system in self.learning_systems.items():
                try:
                    if hasattr(system, 'generate_insights'):
                        system_insights = await system.generate_insights()
                        insights["learning_progress"][system_name] = system_insights
                except Exception as e:
                    insights["learning_progress"][system_name] = {"error": str(e)}
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating cycle insights: {e}")
            return {"error": str(e)}

# Global instance
master_learning_system = MasterLearningSystem()

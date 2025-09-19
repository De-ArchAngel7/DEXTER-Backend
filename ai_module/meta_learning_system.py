#!/usr/bin/env python3
"""
🧠 DEXTER META-LEARNING SYSTEM
============================================================
Learns how to improve learning itself and optimize learning parameters
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import optuna

logger = structlog.get_logger()

class MetaLearningSystem:
    """
    🧠 Meta-Learning System for DEXTER
    
    Features:
    1. Learn optimal learning parameters
    2. Optimize model selection
    3. Learn when to retrain vs use existing models
    4. Optimize learning schedules
    5. Learn from learning performance
    6. Adaptive learning strategies
    """
    
    def __init__(self, mongodb_url: str = None):
        self.mongodb_url = mongodb_url or os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.db = None
        self.collection = None
        
        # Meta-learning parameters
        self.learning_parameters = {
            "retrain_threshold": 100,
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 50,
            "patience": 10,
            "min_delta": 0.001
        }
        
        # Performance tracking
        self.learning_performance = {
            "total_learning_cycles": 0,
            "successful_cycles": 0,
            "failed_cycles": 0,
            "avg_improvement": 0.0,
            "best_parameters": {}
        }
        
        # Optimization history
        self.optimization_history = []
        
    async def initialize(self):
        """Initialize MongoDB connection"""
        try:
            client = AsyncIOMotorClient(self.mongodb_url)
            self.db = client.dexter
            self.collection = self.db.meta_learning
            
            # Create indexes
            await self.collection.create_index("timestamp")
            await self.collection.create_index("learning_type")
            await self.collection.create_index("performance_score")
            
            logger.info("✅ Meta-Learning System initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Meta-Learning System: {e}")
            raise
    
    async def optimize_learning_parameters(self, 
                                         learning_type: str,
                                         performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize learning parameters using meta-learning"""
        try:
            if not self.collection:
                await self.initialize()
            
            logger.info(f"🧠 Optimizing learning parameters for {learning_type}")
            
            # Prepare data for optimization
            optimization_data = self._prepare_optimization_data(performance_data)
            
            if not optimization_data:
                return {"error": "No optimization data available"}
            
            # Run optimization
            best_params = await self._run_parameter_optimization(learning_type, optimization_data)
            
            # Store optimization results
            await self._store_optimization_results(learning_type, best_params, optimization_data)
            
            # Update learning parameters
            self.learning_parameters.update(best_params)
            
            logger.info(f"✅ Optimized parameters for {learning_type}: {best_params}")
            
            return {
                "learning_type": learning_type,
                "optimized_parameters": best_params,
                "optimization_date": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing learning parameters: {e}")
            return {"error": str(e)}
    
    def _prepare_optimization_data(self, performance_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare data for parameter optimization"""
        try:
            if not performance_data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(performance_data)
            
            # Extract features and targets
            features = []
            targets = []
            
            for _, row in df.iterrows():
                # Features: learning parameters
                feature = {
                    "retrain_threshold": row.get("retrain_threshold", 100),
                    "learning_rate": row.get("learning_rate", 0.01),
                    "batch_size": row.get("batch_size", 32),
                    "epochs": row.get("epochs", 50),
                    "patience": row.get("patience", 10),
                    "min_delta": row.get("min_delta", 0.001)
                }
                
                # Target: performance improvement
                target = row.get("performance_improvement", 0.0)
                
                features.append(feature)
                targets.append(target)
            
            # Create DataFrame
            optimization_df = pd.DataFrame(features)
            optimization_df["performance_improvement"] = targets
            
            return optimization_df
            
        except Exception as e:
            logger.error(f"Error preparing optimization data: {e}")
            return pd.DataFrame()
    
    async def _run_parameter_optimization(self, learning_type: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Run parameter optimization using Optuna"""
        try:
            if data.empty:
                return self.learning_parameters
            
            # Define objective function
            def objective(trial):
                # Suggest parameters
                retrain_threshold = trial.suggest_int("retrain_threshold", 50, 200)
                learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1, log=True)
                batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
                epochs = trial.suggest_int("epochs", 20, 100)
                patience = trial.suggest_int("patience", 5, 20)
                min_delta = trial.suggest_float("min_delta", 0.0001, 0.01)
                
                # Calculate performance for these parameters
                performance = self._calculate_parameter_performance(
                    data, retrain_threshold, learning_rate, batch_size, epochs, patience, min_delta
                )
                
                return performance
            
            # Run optimization
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=50)
            
            # Get best parameters
            best_params = study.best_params
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error running parameter optimization: {e}")
            return self.learning_parameters
    
    def _calculate_parameter_performance(self, 
                                       data: pd.DataFrame, 
                                       retrain_threshold: int,
                                       learning_rate: float,
                                       batch_size: int,
                                       epochs: int,
                                       patience: int,
                                       min_delta: float) -> float:
        """Calculate performance for given parameters"""
        try:
            # Filter data for similar parameters
            similar_data = data[
                (data["retrain_threshold"] >= retrain_threshold * 0.8) &
                (data["retrain_threshold"] <= retrain_threshold * 1.2) &
                (data["learning_rate"] >= learning_rate * 0.5) &
                (data["learning_rate"] <= learning_rate * 2.0)
            ]
            
            if similar_data.empty:
                return 0.0
            
            # Calculate average performance
            avg_performance = similar_data["performance_improvement"].mean()
            
            return avg_performance
            
        except Exception as e:
            logger.error(f"Error calculating parameter performance: {e}")
            return 0.0
    
    async def _store_optimization_results(self, 
                                        learning_type: str, 
                                        best_params: Dict[str, Any], 
                                        data: pd.DataFrame):
        """Store optimization results"""
        try:
            optimization_result = {
                "learning_type": learning_type,
                "best_parameters": best_params,
                "optimization_data_size": len(data),
                "avg_performance": data["performance_improvement"].mean() if not data.empty else 0.0,
                "timestamp": datetime.utcnow(),
                "type": "optimization_result"
            }
            
            await self.collection.insert_one(optimization_result)
            
        except Exception as e:
            logger.error(f"Error storing optimization results: {e}")
    
    async def learn_optimal_retraining_schedule(self, 
                                              model_performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Learn optimal retraining schedule"""
        try:
            if not self.collection:
                await self.initialize()
            
            logger.info("🧠 Learning optimal retraining schedule")
            
            # Analyze performance degradation patterns
            degradation_analysis = self._analyze_performance_degradation(model_performance_history)
            
            # Learn optimal retraining triggers
            retraining_triggers = self._learn_retraining_triggers(degradation_analysis)
            
            # Create retraining schedule
            retraining_schedule = self._create_retraining_schedule(retraining_triggers)
            
            # Store learning results
            await self._store_retraining_learning(retraining_schedule, degradation_analysis)
            
            return {
                "retraining_schedule": retraining_schedule,
                "degradation_analysis": degradation_analysis,
                "retraining_triggers": retraining_triggers,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error learning optimal retraining schedule: {e}")
            return {"error": str(e)}
    
    def _analyze_performance_degradation(self, performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance degradation patterns"""
        try:
            if not performance_history:
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(performance_history)
            
            # Calculate performance trends
            performance_trends = {
                "avg_degradation_rate": 0.0,
                "degradation_threshold": 0.05,
                "recovery_time": 0,
                "degradation_patterns": []
            }
            
            # Analyze degradation over time
            if "timestamp" in df.columns and "performance" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp")
                
                # Calculate degradation rate
                performance_changes = df["performance"].diff()
                degradation_rate = performance_changes[performance_changes < 0].mean()
                performance_trends["avg_degradation_rate"] = abs(degradation_rate) if not pd.isna(degradation_rate) else 0.0
                
                # Find degradation patterns
                degradation_periods = df[performance_changes < -0.01]  # 1% degradation
                if not degradation_periods.empty:
                    performance_trends["degradation_patterns"] = degradation_periods.to_dict("records")
            
            return performance_trends
            
        except Exception as e:
            logger.error(f"Error analyzing performance degradation: {e}")
            return {}
    
    def _learn_retraining_triggers(self, degradation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Learn optimal retraining triggers"""
        try:
            triggers = {
                "performance_threshold": 0.05,  # 5% performance drop
                "time_threshold": 7,  # 7 days
                "data_threshold": 100,  # 100 new samples
                "volatility_threshold": 0.1  # 10% volatility increase
            }
            
            # Adjust based on degradation analysis
            if degradation_analysis.get("avg_degradation_rate", 0) > 0.02:
                triggers["performance_threshold"] = 0.03  # More sensitive
            elif degradation_analysis.get("avg_degradation_rate", 0) < 0.01:
                triggers["performance_threshold"] = 0.08  # Less sensitive
            
            return triggers
            
        except Exception as e:
            logger.error(f"Error learning retraining triggers: {e}")
            return {"performance_threshold": 0.05, "time_threshold": 7}
    
    def _create_retraining_schedule(self, triggers: Dict[str, Any]) -> Dict[str, Any]:
        """Create optimal retraining schedule"""
        try:
            schedule = {
                "daily_check": True,
                "weekly_analysis": True,
                "monthly_optimization": True,
                "triggers": triggers,
                "retraining_strategy": "adaptive"
            }
            
            return schedule
            
        except Exception as e:
            logger.error(f"Error creating retraining schedule: {e}")
            return {"daily_check": True, "triggers": triggers}
    
    async def _store_retraining_learning(self, 
                                       schedule: Dict[str, Any], 
                                       analysis: Dict[str, Any]):
        """Store retraining learning results"""
        try:
            learning_result = {
                "retraining_schedule": schedule,
                "degradation_analysis": analysis,
                "timestamp": datetime.utcnow(),
                "type": "retraining_learning"
            }
            
            await self.collection.insert_one(learning_result)
            
        except Exception as e:
            logger.error(f"Error storing retraining learning: {e}")
    
    async def optimize_model_selection(self, 
                                     available_models: List[str],
                                     performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize model selection based on performance"""
        try:
            if not self.collection:
                await self.initialize()
            
            logger.info("🧠 Optimizing model selection")
            
            # Analyze model performance
            model_performance = self._analyze_model_performance(available_models, performance_history)
            
            # Learn optimal model selection criteria
            selection_criteria = self._learn_model_selection_criteria(model_performance)
            
            # Create model selection strategy
            selection_strategy = self._create_model_selection_strategy(selection_criteria)
            
            # Store optimization results
            await self._store_model_selection_optimization(selection_strategy, model_performance)
            
            return {
                "model_performance": model_performance,
                "selection_criteria": selection_criteria,
                "selection_strategy": selection_strategy,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing model selection: {e}")
            return {"error": str(e)}
    
    def _analyze_model_performance(self, 
                                 available_models: List[str], 
                                 performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance of different models"""
        try:
            if not performance_history:
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(performance_history)
            
            model_performance = {}
            
            for model in available_models:
                model_data = df[df.get("model", "") == model]
                
                if not model_data.empty:
                    model_performance[model] = {
                        "avg_accuracy": model_data.get("accuracy", pd.Series([0.5])).mean(),
                        "avg_speed": model_data.get("inference_time", pd.Series([1.0])).mean(),
                        "avg_confidence": model_data.get("confidence", pd.Series([0.5])).mean(),
                        "total_usage": len(model_data),
                        "success_rate": model_data.get("success", pd.Series([False])).mean()
                    }
            
            return model_performance
            
        except Exception as e:
            logger.error(f"Error analyzing model performance: {e}")
            return {}
    
    def _learn_model_selection_criteria(self, model_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Learn optimal model selection criteria"""
        try:
            criteria = {
                "accuracy_weight": 0.4,
                "speed_weight": 0.2,
                "confidence_weight": 0.2,
                "success_rate_weight": 0.2,
                "min_accuracy": 0.6,
                "max_inference_time": 2.0
            }
            
            # Adjust weights based on performance
            if model_performance:
                avg_accuracy = np.mean([m.get("avg_accuracy", 0.5) for m in model_performance.values()])
                avg_speed = np.mean([m.get("avg_speed", 1.0) for m in model_performance.values()])
                
                if avg_accuracy < 0.7:
                    criteria["accuracy_weight"] = 0.5
                if avg_speed > 1.5:
                    criteria["speed_weight"] = 0.3
            
            return criteria
            
        except Exception as e:
            logger.error(f"Error learning model selection criteria: {e}")
            return {"accuracy_weight": 0.4, "speed_weight": 0.2}
    
    def _create_model_selection_strategy(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Create model selection strategy"""
        try:
            strategy = {
                "selection_method": "weighted_score",
                "criteria": criteria,
                "fallback_model": "openai",
                "selection_interval": "per_request"
            }
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error creating model selection strategy: {e}")
            return {"selection_method": "weighted_score"}
    
    async def _store_model_selection_optimization(self, 
                                                strategy: Dict[str, Any], 
                                                performance: Dict[str, Any]):
        """Store model selection optimization results"""
        try:
            optimization_result = {
                "selection_strategy": strategy,
                "model_performance": performance,
                "timestamp": datetime.utcnow(),
                "type": "model_selection_optimization"
            }
            
            await self.collection.insert_one(optimization_result)
            
        except Exception as e:
            logger.error(f"Error storing model selection optimization: {e}")
    
    async def get_meta_learning_insights(self) -> Dict[str, Any]:
        """Get meta-learning insights"""
        try:
            if not self.collection:
                await self.initialize()
            
            # Get all meta-learning results
            meta_results = await self.collection.find({
                "type": {"$in": ["optimization_result", "retraining_learning", "model_selection_optimization"]}
            }).to_list(length=100)
            
            if not meta_results:
                return {"message": "No meta-learning data available"}
            
            # Analyze meta-learning performance
            insights = {
                "total_optimizations": len(meta_results),
                "optimization_types": {},
                "avg_improvement": 0.0,
                "best_parameters": {},
                "learning_trends": []
            }
            
            # Count optimization types
            for result in meta_results:
                opt_type = result.get("type", "unknown")
                insights["optimization_types"][opt_type] = insights["optimization_types"].get(opt_type, 0) + 1
            
            # Calculate average improvement
            improvements = [r.get("avg_performance", 0) for r in meta_results if "avg_performance" in r]
            if improvements:
                insights["avg_improvement"] = np.mean(improvements)
            
            # Get best parameters
            best_results = sorted(meta_results, key=lambda x: x.get("avg_performance", 0), reverse=True)
            if best_results:
                insights["best_parameters"] = best_results[0].get("best_parameters", {})
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting meta-learning insights: {e}")
            return {"error": str(e)}
    
    async def should_retrain_model(self, 
                                 model_type: str, 
                                 current_performance: float, 
                                 last_retrain: datetime,
                                 new_data_count: int) -> Dict[str, Any]:
        """Determine if model should be retrained"""
        try:
            # Get retraining triggers
            retraining_data = await self.collection.find_one({
                "type": "retraining_learning"
            }, sort=[("timestamp", -1)])
            
            if not retraining_data:
                # Use default triggers
                triggers = {"performance_threshold": 0.05, "time_threshold": 7, "data_threshold": 100}
            else:
                triggers = retraining_data.get("retraining_schedule", {}).get("triggers", {})
            
            # Check retraining conditions
            should_retrain = False
            reasons = []
            
            # Performance threshold
            if current_performance < (1.0 - triggers.get("performance_threshold", 0.05)):
                should_retrain = True
                reasons.append("Performance below threshold")
            
            # Time threshold
            days_since_retrain = (datetime.utcnow() - last_retrain).days
            if days_since_retrain >= triggers.get("time_threshold", 7):
                should_retrain = True
                reasons.append("Time threshold reached")
            
            # Data threshold
            if new_data_count >= triggers.get("data_threshold", 100):
                should_retrain = True
                reasons.append("Sufficient new data available")
            
            return {
                "should_retrain": should_retrain,
                "reasons": reasons,
                "current_performance": current_performance,
                "days_since_retrain": days_since_retrain,
                "new_data_count": new_data_count,
                "triggers": triggers
            }
            
        except Exception as e:
            logger.error(f"Error determining if model should be retrained: {e}")
            return {"should_retrain": False, "error": str(e)}

# Global instance
meta_learning_system = MetaLearningSystem()

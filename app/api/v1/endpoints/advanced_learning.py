#!/usr/bin/env python3
"""
🧠 DEXTER ADVANCED LEARNING API ENDPOINTS
============================================================
API endpoints for all advanced learning systems
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime
import structlog

from ai_module.master_learning_system import master_learning_system

logger = structlog.get_logger()
router = APIRouter()

# Pydantic models for request/response
class TradeLearningRequest(BaseModel):
    user_id: str
    symbol: str
    trade_data: Dict[str, Any]
    outcome: Dict[str, Any]

class ComprehensiveInsightsRequest(BaseModel):
    user_id: str
    symbol: Optional[str] = None

class LearningOptimizationRequest(BaseModel):
    optimization_type: str = Field(..., pattern="^(performance|parameters|models|all)$")

@router.get("/health")
async def get_learning_health():
    """Get learning systems health status"""
    try:
        health_status = await master_learning_system.get_learning_health_status()
        return {
            "status": "success",
            "data": health_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting learning health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get learning health")

@router.post("/learn-from-trade")
async def learn_from_trade(trade_request: TradeLearningRequest):
    """Learn from a trade across all systems"""
    try:
        result = await master_learning_system.learn_from_trade(
            user_id=trade_request.user_id,
            symbol=trade_request.symbol,
            trade_data=trade_request.trade_data,
            outcome=trade_request.outcome
        )
        
        return {
            "status": "success",
            "data": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error learning from trade: {e}")
        raise HTTPException(status_code=500, detail="Failed to learn from trade")

@router.post("/comprehensive-insights")
async def get_comprehensive_insights(insights_request: ComprehensiveInsightsRequest):
    """Get comprehensive insights from all learning systems"""
    try:
        insights = await master_learning_system.get_comprehensive_insights(
            user_id=insights_request.user_id,
            symbol=insights_request.symbol
        )
        
        return {
            "status": "success",
            "data": insights,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting comprehensive insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to get comprehensive insights")

@router.post("/optimize-learning")
async def optimize_learning(request: LearningOptimizationRequest, background_tasks: BackgroundTasks):
    """Optimize learning performance"""
    try:
        if request.optimization_type == "all":
            # Run full optimization in background
            background_tasks.add_task(
                _run_full_optimization
            )
            
            return {
                "status": "success",
                "message": "Full learning optimization started in background",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            # Run specific optimization
            result = await master_learning_system.optimize_learning_performance()
            
            return {
                "status": "success",
                "data": result,
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        logger.error(f"Error optimizing learning: {e}")
        raise HTTPException(status_code=500, detail="Failed to optimize learning")

@router.post("/trigger-learning-cycle")
async def trigger_learning_cycle(background_tasks: BackgroundTasks):
    """Trigger a complete learning cycle"""
    try:
        # Run learning cycle in background
        background_tasks.add_task(
            _run_learning_cycle
        )
        
        return {
            "status": "success",
            "message": "Learning cycle started in background",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error triggering learning cycle: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger learning cycle")

@router.get("/real-time-market-insights")
async def get_real_time_market_insights():
    """Get real-time market learning insights"""
    try:
        from ai_module.real_time_market_learning import real_time_market_learning
        
        insights = await real_time_market_learning.get_market_insights()
        
        return {
            "status": "success",
            "data": insights,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting real-time market insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to get real-time market insights")

@router.get("/risk-insights")
async def get_risk_insights():
    """Get risk learning insights"""
    try:
        from ai_module.risk_learning_system import risk_learning_system
        
        insights = await risk_learning_system.get_risk_insights()
        
        return {
            "status": "success",
            "data": insights,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting risk insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to get risk insights")

@router.get("/pattern-insights")
async def get_pattern_insights():
    """Get pattern recognition insights"""
    try:
        from ai_module.pattern_recognition_learning import pattern_recognition_learning
        
        insights = await pattern_recognition_learning.get_pattern_insights()
        
        return {
            "status": "success",
            "data": insights,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting pattern insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to get pattern insights")

@router.get("/meta-learning-insights")
async def get_meta_learning_insights():
    """Get meta-learning insights"""
    try:
        from ai_module.meta_learning_system import meta_learning_system
        
        insights = await meta_learning_system.get_meta_learning_insights()
        
        return {
            "status": "success",
            "data": insights,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting meta-learning insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to get meta-learning insights")

@router.get("/collaborative-insights")
async def get_collaborative_insights():
    """Get collaborative learning insights"""
    try:
        from ai_module.collaborative_learning_system import collaborative_learning_system
        
        insights = await collaborative_learning_system.get_collaborative_insights()
        
        return {
            "status": "success",
            "data": insights,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting collaborative insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to get collaborative insights")

@router.get("/personalized-insights")
async def get_personalized_insights():
    """Get personalized learning insights"""
    try:
        from ai_module.personalized_learning_system import personalized_learning_system
        
        insights = await personalized_learning_system.get_personalization_insights()
        
        return {
            "status": "success",
            "data": insights,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting personalized insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to get personalized insights")

@router.get("/predictive-insights")
async def get_predictive_insights():
    """Get predictive learning insights"""
    try:
        from ai_module.predictive_learning_system import predictive_learning_system
        
        insights = await predictive_learning_system.get_predictive_insights()
        
        return {
            "status": "success",
            "data": insights,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting predictive insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to get predictive insights")

@router.get("/user-recommendations/{user_id}")
async def get_user_recommendations(user_id: str, symbol: Optional[str] = None):
    """Get personalized recommendations for a user"""
    try:
        from ai_module.personalized_learning_system import personalized_learning_system
        
        recommendations = await personalized_learning_system.get_personalized_recommendations(user_id)
        
        return {
            "status": "success",
            "data": recommendations,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting user recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user recommendations")

@router.get("/risk-recommendations/{user_id}")
async def get_risk_recommendations(user_id: str, symbol: str = "BTC"):
    """Get risk recommendations for a user"""
    try:
        from ai_module.risk_learning_system import risk_learning_system
        
        recommendations = await risk_learning_system.get_risk_recommendations(user_id, symbol)
        
        return {
            "status": "success",
            "data": recommendations,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting risk recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get risk recommendations")

@router.get("/market-predictions/{symbol}")
async def get_market_predictions(symbol: str):
    """Get market predictions for a symbol"""
    try:
        from ai_module.predictive_learning_system import predictive_learning_system
        
        # Get multiple predictions
        market_conditions = await predictive_learning_system.predict_market_conditions(symbol)
        volatility_prediction = await predictive_learning_system.predict_volatility(symbol)
        optimal_timing = await predictive_learning_system.predict_optimal_timing(symbol)
        regime_change = await predictive_learning_system.predict_regime_change(symbol)
        
        predictions = {
            "market_conditions": market_conditions,
            "volatility_prediction": volatility_prediction,
            "optimal_timing": optimal_timing,
            "regime_change": regime_change
        }
        
        return {
            "status": "success",
            "data": predictions,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting market predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get market predictions")

@router.get("/pattern-signals/{symbol}")
async def get_pattern_signals(symbol: str):
    """Get pattern signals for a symbol"""
    try:
        from ai_module.pattern_recognition_learning import pattern_recognition_learning
        
        signals = await pattern_recognition_learning.get_pattern_signals(symbol)
        
        return {
            "status": "success",
            "data": signals,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting pattern signals: {e}")
        raise HTTPException(status_code=500, detail="Failed to get pattern signals")

# Background task functions
async def _run_full_optimization():
    """Background task for full optimization"""
    try:
        logger.info("🔄 Starting full learning optimization in background")
        
        # Run optimization
        result = await master_learning_system.optimize_learning_performance()
        
        logger.info(f"✅ Full learning optimization completed: {result}")
        
    except Exception as e:
        logger.error(f"Error in background full optimization: {e}")

async def _run_learning_cycle():
    """Background task for learning cycle"""
    try:
        logger.info("🔄 Starting learning cycle in background")
        
        # Run learning cycle
        result = await master_learning_system.trigger_learning_cycle()
        
        logger.info(f"✅ Learning cycle completed: {result}")
        
    except Exception as e:
        logger.error(f"Error in background learning cycle: {e}")

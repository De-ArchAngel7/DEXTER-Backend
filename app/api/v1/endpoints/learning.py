#!/usr/bin/env python3
"""
🧠 DEXTER LEARNING API ENDPOINTS
============================================================
API endpoints for trade learning and model retraining
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime
import structlog

from ai_module.unified_conversation_engine import conversation_engine
from ai_module.trade_learning import trade_learning_system
from ai_module.model_retraining import model_retraining_system

logger = structlog.get_logger()
router = APIRouter()

# Pydantic models for request/response
class TradeLogRequest(BaseModel):
    user_id: str
    symbol: str
    action: str
    entry_price: float
    exit_price: Optional[float] = None
    quantity: float = 1.0
    success: Optional[bool] = None
    profit_loss: Optional[float] = None
    user_feedback: Optional[str] = None

class UserFeedbackRequest(BaseModel):
    user_id: str
    conversation_id: str
    ai_response: str
    user_rating: int = Field(..., ge=1, le=5)
    feedback_text: Optional[str] = None

class RetrainingRequest(BaseModel):
    model_type: str = Field(..., pattern="^(dialoGPT|lstm|both)$")
    days_of_data: int = Field(default=30, ge=1, le=365)

@router.get("/performance")
async def get_learning_performance():
    """Get learning system performance summary"""
    try:
        performance = await conversation_engine.get_learning_performance()
        return {
            "status": "success",
            "data": performance,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting learning performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to get learning performance")

@router.post("/log-trade")
async def log_trade(trade_data: TradeLogRequest):
    """Log a trade for learning purposes"""
    try:
        result = await conversation_engine.log_trade_outcome(
            user_id=trade_data.user_id,
            symbol=trade_data.symbol,
            action=trade_data.action,
            entry_price=trade_data.entry_price,
            exit_price=trade_data.exit_price,
            success=trade_data.success,
            profit_loss=trade_data.profit_loss,
            user_feedback=trade_data.user_feedback
        )
        
        return {
            "status": "success",
            "message": "Trade logged successfully",
            "trade_id": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error logging trade: {e}")
        raise HTTPException(status_code=500, detail="Failed to log trade")

@router.post("/feedback")
async def log_user_feedback(feedback_data: UserFeedbackRequest):
    """Log user feedback on AI responses"""
    try:
        if not trade_learning_system.collection:
            await trade_learning_system.initialize()
        
        await trade_learning_system.log_user_feedback(
            user_id=feedback_data.user_id,
            conversation_id=feedback_data.conversation_id,
            ai_response=feedback_data.ai_response,
            user_rating=feedback_data.user_rating,
            feedback_text=feedback_data.feedback_text
        )
        
        return {
            "status": "success",
            "message": "Feedback logged successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error logging feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to log feedback")

@router.post("/retrain")
async def trigger_model_retraining(request: RetrainingRequest, background_tasks: BackgroundTasks):
    """Trigger model retraining based on learning data"""
    try:
        if request.model_type in ["dialoGPT", "both"]:
            # Trigger DialoGPT retraining in background
            background_tasks.add_task(
                _retrain_dialoGPT_background,
                request.days_of_data
            )
        
        if request.model_type in ["lstm", "both"]:
            # Trigger LSTM retraining in background
            background_tasks.add_task(
                _retrain_lstm_background,
                request.days_of_data
            )
        
        return {
            "status": "success",
            "message": f"Model retraining started for {request.model_type}",
            "data_period_days": request.days_of_data,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error triggering retraining: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger retraining")

@router.get("/retraining-status")
async def get_retraining_status():
    """Get status of ongoing retraining processes"""
    try:
        # This would typically check a job queue or database for retraining status
        # For now, return a simple status
        return {
            "status": "success",
            "data": {
                "dialoGPT_retraining": "idle",
                "lstm_retraining": "idle",
                "last_retrain": None,
                "next_scheduled": None
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting retraining status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get retraining status")

@router.get("/learning-data")
async def export_learning_data(days: int = 30):
    """Export learning data for analysis"""
    try:
        if not trade_learning_system.collection:
            await trade_learning_system.initialize()
        
        data = await trade_learning_system.export_learning_data(days)
        
        return {
            "status": "success",
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error exporting learning data: {e}")
        raise HTTPException(status_code=500, detail="Failed to export learning data")

@router.get("/insights")
async def get_learning_insights():
    """Get insights from learning system"""
    try:
        if not trade_learning_system.collection:
            await trade_learning_system.initialize()
        
        # Get recent analysis
        recent_analysis = await trade_learning_system._analyze_recent_trades()
        
        # Generate insights
        insights = await trade_learning_system._generate_learning_insights(recent_analysis)
        
        return {
            "status": "success",
            "data": insights,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting learning insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to get learning insights")

@router.post("/manual-learning-trigger")
async def manual_learning_trigger():
    """Manually trigger learning analysis"""
    try:
        if not trade_learning_system.collection:
            await trade_learning_system.initialize()
        
        await trade_learning_system.trigger_model_learning()
        
        return {
            "status": "success",
            "message": "Learning analysis triggered successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error triggering learning: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger learning")

# Background task functions
async def _retrain_dialoGPT_background(days_of_data: int):
    """Background task for DialoGPT retraining"""
    try:
        logger.info(f"🔄 Starting background DialoGPT retraining with {days_of_data} days of data")
        
        # Get learning data
        if not trade_learning_system.collection:
            await trade_learning_system.initialize()
        
        recent_data = await trade_learning_system.export_learning_data(days_of_data)
        
        if "error" in recent_data:
            logger.error("No learning data available for retraining")
            return
        
        # Extract conversation and trade data
        conversation_data = []
        trade_data = []
        
        for record in recent_data.get("data", []):
            if record.get("type") == "prediction":
                conversation_data.append({
                    "user_message": f"Question about {record.get('symbol', 'trading')}",
                    "ai_response": record.get("prediction", ""),
                    "accuracy": record.get("accuracy", 0.5)
                })
            elif record.get("type") != "feedback":
                trade_data.append(record)
        
        # Retrain DialoGPT
        results = await model_retraining_system.retrain_dialoGPT(
            conversation_data, trade_data
        )
        
        logger.info(f"✅ Background DialoGPT retraining completed: {results}")
        
    except Exception as e:
        logger.error(f"Error in background DialoGPT retraining: {e}")

async def _retrain_lstm_background(days_of_data: int):
    """Background task for LSTM retraining"""
    try:
        logger.info(f"🔄 Starting background LSTM retraining with {days_of_data} days of data")
        
        # For now, skip LSTM retraining as we need price data
        logger.info("LSTM retraining skipped - no price data available")
        
    except Exception as e:
        logger.error(f"Error in background LSTM retraining: {e}")

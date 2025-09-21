#!/usr/bin/env python3
"""
💬 DEXTER CHAT API ENDPOINTS
============================================================
Chat endpoints for unified conversation engine
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import structlog
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

# Conditional AI imports - prevent blocking during startup
if not os.getenv("DISABLE_AI_IMPORTS", "false").lower() == "true":
    from ai_module.unified_conversation_engine import conversation_engine
else:
    conversation_engine = None
from app.core.security import get_current_user
from app.models.user import User

logger = structlog.get_logger()
router = APIRouter()

class ChatRequest(BaseModel):
    """Chat request model"""
    user_id: str
    message: str
    source: str = "web"  # "telegram" or "web"

class ChatResponse(BaseModel):
    """Chat response model"""
    reply: str
    source: str
    model_used: str
    user_id: str
    timestamp: str

class ConversationHistoryResponse(BaseModel):
    """Conversation history response model"""
    user_id: str
    messages: List[Dict[str, Any]]

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Main chat endpoint for unified conversation engine
    
    Accepts:
    - user_id: Unique user identifier
    - message: User message
    - source: Source of the message ("telegram" or "web")
    
    Returns:
    - reply: AI response
    - source: Source of the request
    - model_used: Which model generated the response
    - user_id: User identifier
    - timestamp: Response timestamp
    """
    try:
        logger.info(f"Chat request from {request.source} for user {request.user_id}")
        
        # Validate source
        if request.source not in ["telegram", "web"]:
            raise HTTPException(status_code=400, detail="Invalid source. Must be 'telegram' or 'web'")
        
        # Get response from unified conversation engine
        response = await conversation_engine.chat(
            user_id=request.user_id,
            message=request.message,
            source=request.source
        )
        
        logger.info(f"Chat response generated using {response['model_used']} model")
        
        return ChatResponse(**response)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/chat/history/{user_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    user_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get conversation history for a user
    """
    try:
        history = conversation_engine.get_conversation_history(user_id)
        
        return ConversationHistoryResponse(
            user_id=user_id,
            messages=history
        )
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/chat/history/{user_id}")
async def clear_conversation_history(
    user_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Clear conversation history for a user
    """
    try:
        conversation_engine.clear_conversation(user_id)
        
        return {"message": f"Conversation history cleared for user {user_id}"}
        
    except Exception as e:
        logger.error(f"Error clearing conversation history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/chat/status")
async def get_chat_status():
    """
    Get status of the unified conversation engine
    """
    try:
        status = conversation_engine.get_engine_status()
        
        return {
            "status": "operational",
            "engines": status,
            "timestamp": "2025-01-01T00:00:00Z"  # You can use datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting chat status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/chat/test")
async def test_chat():
    """
    Test endpoint to verify chat functionality
    """
    try:
        # Test with a simple message
        response = await conversation_engine.chat(
            user_id="test_user",
            message="Hello! Can you help me with trading?",
            source="web"
        )
        
        return {
            "test": "successful",
            "response": response,
            "message": "Chat engine is working correctly"
        }
        
    except Exception as e:
        logger.error(f"Error in test chat: {e}")
        return {
            "test": "failed",
            "error": str(e),
            "message": "Chat engine has issues"
        }

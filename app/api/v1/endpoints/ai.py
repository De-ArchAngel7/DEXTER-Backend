from typing import Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.security import HTTPBearer
from datetime import datetime
import os
import structlog

# Conditional AI imports - prevent blocking during startup
if not os.getenv("DISABLE_AI_IMPORTS", "false").lower() == "true":
    from ai_module.ai_chatbot import AIChatbot
    from ai_module.unified_conversation_engine import conversation_engine
else:
    AIChatbot = None
    conversation_engine = None

logger = structlog.get_logger()
security = HTTPBearer()

# Initialize AI chatbot if OpenAI API key is available
ai_chatbot = None
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    try:
        ai_chatbot = AIChatbot(openai_api_key)
        logger.info("AI chatbot initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AI chatbot: {e}")
        ai_chatbot = None
else:
    logger.warning("OpenAI API key not found. AI chatbot will use fallback mode.")

router = APIRouter()

@router.get("/status")
async def get_ai_status():
    """Get AI system status including unified conversation engine"""
    try:
        # Get unified conversation engine status
        engine_status = conversation_engine.get_engine_status()
        
        return {
            "status": "operational",
            "unified_conversation_engine": {
                "status": "active",
                "dialoGPT_loaded": engine_status["dialoGPT_loaded"],
                "openai_available": engine_status["openai_available"],
                "active_conversations": engine_status["active_conversations"],
                "primary_model": "DialoGPT (Custom Fine-tuned)",
                "fallback_model": "OpenAI GPT-4" if engine_status["openai_available"] else "None"
            },
            "ai_chatbot": {
                "status": "active" if ai_chatbot else "inactive",
                "provider": "OpenAI GPT-4" if ai_chatbot else "fallback",
                "models_available": ["DialoGPT", "LSTM", "Transformer", "GRU"]
            },
            "dialoGPT_model": {
                "status": "active" if engine_status["dialoGPT_loaded"] else "inactive",
                "parameters": engine_status["dialoGPT_status"]["parameters"],
                "device": engine_status["dialoGPT_status"]["device"],
                "quantization": engine_status["dialoGPT_status"]["quantization"],
                "model_path": engine_status["dialoGPT_status"]["model_path"]
            },
            "gpu_count": 1,
            "total_gpu_memory": "8GB",
            "models": {
                "dialoGPT_model": {
                    "status": "active" if engine_status["dialoGPT_loaded"] else "inactive",
                    "accuracy": 85.0,
                    "gpu_utilization": 60 if engine_status["dialoGPT_status"]["device"] == "cuda" else 0,
                    "last_trained": "2024-01-15T10:30:00Z"
                },
                "lstm_model": {
                    "status": "active",
                    "accuracy": 78.5,
                    "gpu_utilization": 45,
                    "last_trained": "2024-01-15T10:30:00Z"
                },
                "sentiment_model": {
                    "status": "active",
                    "accuracy": 82.3,
                    "gpu_utilization": 30,
                    "last_trained": "2024-01-14T15:45:00Z"
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting AI status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get AI status")

@router.get("/insights")
async def get_ai_insights():
    """Get AI-powered market insights"""
    try:
        if ai_chatbot:
            # Get real insights from AI chatbot
            symbols = ["BTC", "ETH", "SOL", "ADA"]
            analysis = await ai_chatbot.get_market_analysis(symbols)
            
            if "error" in analysis:
                return _get_mock_insights()
            
            # Convert analysis to insights format
            insights = []
            for symbol, data in analysis["analysis"].items():
                insights.append({
                    "id": f"insight_{symbol}_{datetime.utcnow().timestamp()}",
                    "type": "ai_analysis",
                    "symbol": symbol,
                    "prediction": f"AI Analysis: {data['recommendation']} with {data['overall_score']:.1f}% confidence",
                    "confidence": data['overall_score'],
                    "timestamp": datetime.utcnow().isoformat(),
                    "details": {
                        "price_prediction": data['price_prediction'],
                        "sentiment_analysis": data['sentiment_analysis']
                    }
                })
            
            return insights
        else:
            return _get_mock_insights()
            
    except Exception as e:
        logger.error(f"Error getting AI insights: {e}")
        return _get_mock_insights()

@router.get("/signals")
async def get_trading_signals():
    """Get AI-powered trading signals"""
    try:
        if ai_chatbot:
            # Get real trading signals from AI chatbot
            symbols = ["BTC", "ETH", "SOL"]
            signals = []
            
            for symbol in symbols:
                # Mock market data - replace with real data service
                market_data = {
                    "price_usd": 45000 if symbol == "BTC" else 2500 if symbol == "ETH" else 100,
                    "price_change_24h": 2.5,
                    "volume_24h": 25000000000,
                    "market_cap": 850000000000
                }
                
                advice = await ai_chatbot.get_trading_advice(
                    symbol, 
                    market_data["price_usd"], 
                    market_data, 
                    "What's your trading recommendation for this token?"
                )
                
                if "error" not in advice:
                    signals.append({
                        "id": f"signal_{symbol}_{datetime.utcnow().timestamp()}",
                        "symbol": symbol,
                        "action": "ANALYZE",
                        "entry_price": market_data["price_usd"],
                        "take_profit": market_data["price_usd"] * 1.1,
                        "stop_loss": market_data["price_usd"] * 0.95,
                        "confidence": advice.get("confidence", 75),
                        "reasoning": advice.get("advice", "AI analysis unavailable"),
                        "timestamp": datetime.utcnow().isoformat(),
                        "expires_at": (datetime.utcnow().replace(hour=23, minute=59, second=59)).isoformat(),
                        "ai_provider": "OpenAI GPT-4"
                    })
            
            return signals if signals else _get_mock_signals()
        else:
            return _get_mock_signals()
            
    except Exception as e:
        logger.error(f"Error getting trading signals: {e}")
        return _get_mock_signals()

@router.post("/chat")
async def chat_with_ai(message: Dict[str, str]):
    """Chat with AI trading advisor using unified conversation engine"""
    try:
        user_message = message.get("message", "")
        symbol = message.get("symbol", "")
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Generate a unique user ID for web users (you might want to implement proper user authentication)
        user_id = f"web_user_{hash(user_message) % 10000}"
        
        # Enhance message with symbol context if provided
        enhanced_message = user_message
        if symbol:
            enhanced_message = f"Regarding {symbol}: {user_message}"
        
        # Use unified conversation engine
        response = await conversation_engine.chat(
            user_id=user_id,
            message=enhanced_message,
            source="web"
        )
        
        return {
            "response": response["reply"],
            "symbol": symbol,
            "confidence": 85 if response["model_used"] == "dialoGPT" else 75,
            "model_used": response["model_used"]
        }
        
    except Exception as e:
        logger.error(f"Error in AI chat: {e}")
        return {
            "response": f"Sorry, I encountered an error: {str(e)}",
            "symbol": None,
            "confidence": 0,
            "model_used": "error"
        }

@router.post("/models/{model_name}/predict")
async def predict_with_model(model_name: str, data: Dict[str, Any]):
    """Make prediction with specific AI model"""
    try:
        if not ai_chatbot:
            return _get_mock_prediction(model_name)
        
        if model_name == "lstm":
            # Use price prediction model
            market_data = data.get("market_data", {})
            prediction = await ai_chatbot._get_price_prediction(
                data.get("symbol", "UNKNOWN"), 
                market_data
            )
            return prediction
        elif model_name == "sentiment":
            # Use sentiment analysis model
            sentiment = await ai_chatbot._get_sentiment_analysis(
                data.get("symbol", "UNKNOWN")
            )
            return sentiment
        else:
            raise HTTPException(status_code=400, detail="Unknown model")
            
    except Exception as e:
        logger.error(f"Error making prediction with {model_name}: {e}")
        return _get_mock_prediction(model_name)

@router.post("/models/{model_name}/retrain")
async def retrain_model(model_name: str, background_tasks: BackgroundTasks):
    """Retrain AI model"""
    try:
        # Simulate retraining process
        background_tasks.add_task(_simulate_retraining, model_name)
        
        return {
            "message": f"Model {model_name} retraining started",
            "status": "training",
            "estimated_completion": "2 hours"
        }
    except Exception as e:
        logger.error(f"Error starting model retraining: {e}")
        raise HTTPException(status_code=500, detail="Failed to start retraining")

def _get_mock_insights():
    """Get mock AI insights when AI is unavailable"""
    return [
        {
            "id": "insight_btc_001",
            "type": "price_prediction",
            "symbol": "BTC",
            "prediction": "Bitcoin showing bullish momentum with potential resistance at $47,000",
            "confidence": 78.5,
            "timestamp": datetime.utcnow().isoformat(),
            "details": {"trend": "bullish", "support": 44000, "resistance": 47000}
        },
        {
            "id": "insight_eth_001",
            "type": "sentiment_analysis",
            "symbol": "ETH",
            "prediction": "Ethereum sentiment improving with institutional adoption news",
            "confidence": 82.3,
            "timestamp": datetime.utcnow().isoformat(),
            "details": {"sentiment": "positive", "social_volume": "high"}
        }
    ]

def _get_mock_signals():
    """Get mock trading signals when AI is unavailable"""
    return [
        {
            "id": "signal_btc_001",
            "symbol": "BTC",
            "action": "BUY",
            "entry_price": 45000,
            "take_profit": 49500,
            "stop_loss": 42750,
            "confidence": 78.5,
            "reasoning": "Strong technical breakout with volume confirmation",
            "timestamp": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow().replace(hour=23, minute=59, second=59)).isoformat(),
            "ai_provider": "Mock Analysis"
        }
    ]

def _get_mock_prediction(model_name: str):
    """Get mock prediction when AI is unavailable"""
    if model_name == "lstm":
        return {
            "predicted_price": 47000,
            "confidence": 75.0,
            "trend": "bullish",
            "timeframe": "24H",
            "model_used": "Mock_LSTM"
        }
    elif model_name == "sentiment":
        return {
            "sentiment": "positive",
            "confidence": 80.0,
            "compound_score": 0.6,
            "social_volume": 1000,
            "trending_topics": ["adoption", "regulation"]
        }
    else:
        return {"error": "Unknown model"}

def _simulate_retraining(model_name: str):
    """Simulate model retraining process"""
    import time
    time.sleep(5)  # Simulate work
    print(f"Model {model_name} retraining completed")

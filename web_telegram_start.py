#!/usr/bin/env python3
"""
WEB + TELEGRAM DEXTER - Both interfaces working
"""
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Get the port
port = os.environ.get("PORT", "10000")
print(f"🌐📱 WEB + TELEGRAM DEXTER START ON PORT: {port}")

# Create FastAPI app
app = FastAPI(
    title="DEXTER AI Trading Bot",
    description="Web + Telegram interface for DEXTER",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    return {"message": "🌐📱 DEXTER Web + Telegram Interface", "status": "operational"}

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "web_telegram"}

# Authentication endpoints
@app.post("/api/v1/auth/login")
async def login(request: Request):
    return {
        "access_token": "demo_token_12345",
        "token_type": "bearer",
        "user": {"id": 1, "email": "demo@dexter.ai", "username": "demo_user"}
    }

@app.post("/api/v1/auth/register")
async def register(request: Request):
    return {
        "message": "Registration successful",
        "user": {"id": 2, "email": "new@dexter.ai", "username": "new_user"}
    }

# Fallback auth endpoints
@app.post("/auth/login")
async def login_fallback(request: Request):
    return await login(request)

@app.post("/auth/register")
async def register_fallback(request: Request):
    return await register(request)

@app.post("/api/v1/auth/logout")
async def logout():
    return {"message": "Logged out successfully"}

# Chat endpoint
@app.post("/api/v1/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        message = body.get("message", "")
        
        return {
            "response": f"Hello! I received your message: '{message}'. I'm DEXTER, your AI trading assistant. I can help you with market analysis, trading strategies, and portfolio management. What specific trading question do you have?",
            "symbol": "",
            "confidence": 0.85,
            "model_used": "dexter_web",
            "timestamp": "2025-09-21T15:00:00Z",
            "ai_provider": "dexter_ai"
        }
        
    except Exception as e:
        return {
            "response": "I'm here to help with your trading questions! How can I assist you today?",
            "symbol": "",
            "confidence": 0.0,
            "model_used": "fallback",
            "timestamp": "2025-09-21T15:00:00Z",
            "ai_provider": "fallback"
        }

# Telegram webhook - SIMPLIFIED
@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    try:
        update = await request.json()
        message = update.get("message", {})
        text = message.get("text", "")
        chat_id = message.get("chat", {}).get("id")
        
        if not text or not chat_id:
            return {"status": "ignored"}
        
        # Simple AI response - no complex imports
        if "hello" in text.lower() or "hi" in text.lower():
            response_text = f"Hello! I'm DEXTER, your AI trading assistant. I received your message: '{text}'. I can help you with cryptocurrency trading, market analysis, and portfolio management. What would you like to know?"
        elif "price" in text.lower() or "btc" in text.lower() or "bitcoin" in text.lower():
            response_text = "I can help you with Bitcoin analysis! Currently, Bitcoin is showing strong momentum. Would you like me to provide detailed market insights or trading recommendations?"
        elif "trade" in text.lower() or "buy" in text.lower() or "sell" in text.lower():
            response_text = "I can assist with trading decisions! For safe trading, I recommend starting with small amounts and using proper risk management. What specific trading question do you have?"
        else:
            response_text = f"I received your message: '{text}'. I'm DEXTER, your AI trading assistant. I can help with market analysis, trading strategies, and risk management. How can I assist you today?"
        
        # Send response to Telegram
        import httpx
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        
        if not telegram_token:
            return {"status": "error", "error": "No bot token"}
            
        telegram_url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(telegram_url, json={
                "chat_id": chat_id,
                "text": response_text
            })
            
            if response.status_code == 200:
                return {"status": "success"}
            else:
                return {"status": "telegram_error", "code": response.status_code}
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Portfolio endpoint
@app.get("/api/v1/portfolio/overview")
async def portfolio_overview():
    return {
        "total_value": 15420.50,
        "change_24h": 340.20,
        "change_percent": 2.25,
        "assets": [
            {"symbol": "BTC", "value": 8500.00, "change": 2.1, "color": "#f7931a"},
            {"symbol": "ETH", "value": 4200.00, "change": 1.8, "color": "#627eea"},
            {"symbol": "SOL", "value": 2720.50, "change": 3.2, "color": "#9945ff"}
        ]
    }

# Trading endpoints
@app.get("/api/v1/trading/orders")
async def get_orders():
    return [
        {"id": "order_1", "symbol": "BTCUSDT", "side": "BUY", "amount": 0.1, "status": "filled"},
        {"id": "order_2", "symbol": "ETHUSDT", "side": "SELL", "amount": 2.5, "status": "pending"}
    ]

@app.get("/api/v1/trading/history")
async def get_trading_history():
    return [
        {
            "id": "trade_001",
            "symbol": "BTCUSDT", 
            "side": "BUY",
            "amount": 0.05,
            "price": 45000.00,
            "status": "completed"
        }
    ]

@app.post("/api/v1/trading/execute")
async def execute_trade(request: Request):
    return {"order_id": "trade_12345", "status": "executed", "message": "Trade executed successfully"}

# AI endpoints
@app.get("/api/v1/ai/status")
async def ai_status():
    return {
        "status": "operational",
        "models": {"dexter_ai": {"status": "ready"}},
        "conversation_engine": "active"
    }

@app.get("/api/v1/ai/trading-signals")
async def get_trading_signals():
    return [
        {
            "id": "signal_001",
            "symbol": "BTCUSDT",
            "action": "BUY",
            "confidence": 0.85,
            "price_target": 52000.00,
            "stop_loss": 48000.00,
            "take_profit": 56000.00,
            "reasoning": "Strong bullish momentum with good support levels"
        }
    ]

@app.get("/api/v1/ai/insights")
async def get_ai_insights():
    return [
        {
            "id": "insight_001",
            "title": "Bitcoin Market Analysis",
            "description": "Strong support levels detected with potential for upward movement",
            "confidence": 0.88,
            "impact": "high",
            "timeframe": "1-3 days"
        }
    ]

# User profile
@app.get("/api/v1/users/profile")
async def get_user_profile():
    return {
        "id": 1,
        "email": "demo@dexter.ai",
        "username": "demo_user"
    }

if __name__ == "__main__":
    print("🌐📱 Starting WEB + TELEGRAM DEXTER...")
    print("✅ Both web and Telegram interfaces active!")
    print("🛡️ Simplified AI for maximum stability!")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(port),
        log_level="info"
    )

#!/usr/bin/env python3
"""
MINIMAL WEB DEXTER - Web interface only, no Telegram
"""
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Get the port
port = os.environ.get("PORT", "10000")
print(f"🌐 MINIMAL WEB DEXTER START ON PORT: {port}")

# Create FastAPI app
app = FastAPI(
    title="DEXTER AI Trading Bot",
    description="Web interface for DEXTER",
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
    return {"message": "🌐 DEXTER Web Interface", "status": "operational"}

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "web_only"}

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
            "response": f"Hello! I received your message: '{message}'. I'm DEXTER, your AI trading assistant. How can I help you today?",
            "symbol": "",
            "confidence": 0.85,
            "model_used": "dexter_web",
            "timestamp": "2025-09-21T15:00:00Z",
            "ai_provider": "dexter_ai"
        }
        
    except Exception as e:
        return {
            "response": "I'm here to help! What would you like to know about trading?",
            "symbol": "",
            "confidence": 0.0,
            "model_used": "fallback",
            "timestamp": "2025-09-21T15:00:00Z",
            "ai_provider": "fallback"
        }

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

# Basic AI endpoints
@app.get("/api/v1/ai/status")
async def ai_status():
    return {
        "status": "operational",
        "models": {"mistral_7b": {"status": "ready"}},
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
            "reasoning": "Strong bullish momentum"
        }
    ]

@app.get("/api/v1/ai/insights")
async def get_ai_insights():
    return [
        {
            "id": "insight_001",
            "title": "Bitcoin Analysis",
            "description": "Strong support levels detected",
            "confidence": 0.88
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
    print("🌐 Starting MINIMAL WEB DEXTER...")
    print("✅ Web interface only - maximum stability!")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(port),
        log_level="info"
    )

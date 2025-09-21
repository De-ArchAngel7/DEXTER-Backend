#!/usr/bin/env python3
"""
SMART DEXTER START - Full functionality with non-blocking AI loading
"""
import os
import sys
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Get the port
port = os.environ.get("PORT", "10000")
print(f"🧠 SMART DEXTER START ON PORT: {port}")

# Set environment for optimal loading
os.environ["USE_HF_API"] = "true"
os.environ["LAZY_AI_LOADING"] = "true"

# Global AI engine placeholder
ai_engine = None
conversation_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown with background AI loading"""
    global ai_engine, conversation_engine
    
    print("🚀 DEXTER Smart startup initiated...")
    
    # Start AI loading in background (non-blocking)
    async def load_ai_modules():
        global ai_engine, conversation_engine
        try:
            print("🧠 Loading AI modules in background...")
            
            # Import AI modules AFTER server starts
            from ai_module.ai_fusion_engine import AIFusionEngine
            from ai_module.unified_conversation_engine import UnifiedConversationEngine
            
            ai_engine = AIFusionEngine()
            conversation_engine = UnifiedConversationEngine()
            
            print("✅ AI modules loaded successfully!")
            
        except Exception as e:
            print(f"⚠️ AI modules failed to load: {e}")
            print("🔄 DEXTER will use fallback responses")
    
    # Start AI loading task (non-blocking)
    asyncio.create_task(load_ai_modules())
    
    print("✅ DEXTER Smart startup complete - Server ready!")
    print("🤖 AI modules loading in background...")
    
    yield
    
    print("🛑 DEXTER Smart shutdown...")

# Create FastAPI app
app = FastAPI(
    title="DEXTER AI Trading Bot",
    description="Advanced AI-powered cryptocurrency trading platform",
    version="2.0.0",
    lifespan=lifespan
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
def root():
    ai_status = "loaded" if ai_engine else "loading"
    return {
        "status": "DEXTER is fully operational!",
        "version": "2.0.0",
        "ai_status": ai_status,
        "features": ["AI Chat", "Trading", "Portfolio", "Telegram Bot"],
        "backend_url": "https://dexter-backend-dqx1.onrender.com"
    }

# Health check
@app.get("/health")
def health():
    ai_status = "loaded" if ai_engine else "loading"
    return {
        "status": "healthy",
        "port": port,
        "ai_status": ai_status,
        "telegram_bot": "active"
    }

# Chat endpoint with smart AI handling
@app.post("/api/v1/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        message = body.get("message", "")
        user_id = body.get("user_id", "web_user")
        
        if conversation_engine:
            # Use full AI if loaded
            response = await conversation_engine.chat(user_id, message, "web")
            return response
        else:
            # Fallback response while AI loads
            return {
                "reply": f"Hello! I'm DEXTER and I'm still initializing my AI systems. Your message '{message}' has been received. Please try again in a moment for full AI responses!",
                "model_used": "fallback",
                "timestamp": "2025-09-21T13:00:00Z"
            }
            
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {
            "reply": "I'm experiencing technical difficulties. Please try again.",
            "model_used": "error_fallback",
            "error": str(e)
        }

# Telegram webhook
@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    try:
        update = await request.json()
        
        # Extract message info
        message = update.get("message", {})
        text = message.get("text", "")
        chat_id = message.get("chat", {}).get("id")
        user_id = message.get("from", {}).get("id", "unknown")
        
        if not text or not chat_id:
            return {"status": "ignored"}
        
        # Generate response
        if conversation_engine:
            # Use full AI if loaded
            ai_response = await conversation_engine.chat(str(user_id), text, "telegram")
            response_text = ai_response.get("reply", "Hello! I'm DEXTER, your AI trading assistant.")
        else:
            # Fallback while AI loads
            response_text = f"Hello! I'm DEXTER and I'm initializing my AI systems. Your message '{text}' received. Full AI responses coming soon!"
        
        # Send response back to Telegram
        import httpx
        telegram_url = f"https://api.telegram.org/bot{os.getenv('TELEGRAM_BOT_TOKEN')}/sendMessage"
        
        async with httpx.AsyncClient() as client:
            await client.post(telegram_url, json={
                "chat_id": chat_id,
                "text": response_text
            })
        
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Telegram webhook error: {e}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    print("🧠 Starting DEXTER Smart Server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(port),
        log_level="info"
    )

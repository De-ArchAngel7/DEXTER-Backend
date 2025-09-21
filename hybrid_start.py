#!/usr/bin/env python3
"""
HYBRID DEXTER START - Essential endpoints + Fast startup
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Get the port
port = os.environ.get("PORT", "10000")
print(f"🔥 HYBRID START ON PORT: {port}")

# Set environment for controlled AI loading
os.environ["USE_HF_API"] = "true"
os.environ["MINIMAL_AI_INIT"] = "true"  # Load AI modules lazily

# Create FastAPI app with essential features
app = FastAPI(
    title="DEXTER AI Trading Bot",
    description="Advanced AI-powered cryptocurrency trading platform",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic endpoints
@app.get("/")
def root():
    return {
        "status": "DEXTER is alive!",
        "version": "2.0.0",
        "features": ["AI Chat", "Trading", "Portfolio", "Telegram Bot"],
        "backend_url": f"https://dexter-backend-dqx1.onrender.com"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "port": port,
        "ai_status": "loading",
        "telegram_bot": "initializing"
    }

# Import essential API routes (lazy loading)
@app.on_event("startup")
async def startup_event():
    print("🚀 DEXTER startup complete - essential services ready")
    print("🤖 Telegram bot will be available after AI initialization")

if __name__ == "__main__":
    print("🔥 Starting hybrid DEXTER server...")
    uvicorn.run(
        app,
        host="0.0.0.0", 
        port=int(port),
        log_level="info"
    )

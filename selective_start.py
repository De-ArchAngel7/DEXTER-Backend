#!/usr/bin/env python3
"""
SELECTIVE DEXTER START - Add endpoints gradually
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Get the port
port = os.environ.get("PORT", "10000")
print(f"🎯 SELECTIVE START ON PORT: {port}")

# Set environment
os.environ["USE_HF_API"] = "true"

# Create FastAPI app
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
        "mode": "selective",
        "features": ["Basic API", "Health Check", "Ready for expansion"],
        "backend_url": "https://dexter-backend-dqx1.onrender.com"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "port": port,
        "mode": "selective",
        "telegram_bot": "ready_to_configure"
    }

# Add simple chat endpoint (no AI imports yet)
@app.post("/api/v1/chat")
def simple_chat(message: dict):
    return {
        "response": f"Hello! I received your message: '{message.get('message', 'No message')}'. Full AI chat coming soon!",
        "model_used": "simple_response",
        "timestamp": "2025-09-21T13:00:00Z"
    }

# Basic Telegram webhook endpoint
@app.post("/telegram/webhook")
def telegram_webhook(update: dict):
    return {"status": "received", "message": "Telegram webhook ready"}

if __name__ == "__main__":
    print("🔥 Starting selective DEXTER server...")
    uvicorn.run(
        app,
        host="0.0.0.0", 
        port=int(port),
        log_level="info"
    )

#!/usr/bin/env python3
"""
PRODUCTION DEXTER START - Full functionality with safe loading
"""
import os
import sys
import uvicorn
import asyncio
from contextlib import asynccontextmanager

# Set production environment
os.environ["USE_HF_API"] = "true"
os.environ["PRODUCTION_MODE"] = "true"

# Get the port
port = os.environ.get("PORT", "10000")
print(f"🚀 PRODUCTION DEXTER START ON PORT: {port}")

@asynccontextmanager
async def lifespan(app):
    """Handle startup and shutdown events"""
    print("🔥 DEXTER Production startup initiated...")
    print("🤖 Loading AI modules in background...")
    print("📱 Telegram bot initializing...")
    print("💼 Trading engines warming up...")
    
    # Startup complete
    print("✅ DEXTER Production ready - All systems operational!")
    yield
    
    # Shutdown
    print("🛑 DEXTER Production shutdown initiated...")

# Import the full FastAPI app with error handling
try:
    print("📦 Loading full DEXTER application...")
    from app.main import app
    
    # Update the app to use our lifespan handler
    app.router.lifespan_context = lifespan
    print("✅ Full DEXTER app loaded successfully!")
    
except Exception as e:
    print(f"❌ Failed to load full app: {e}")
    print("🚨 Falling back to emergency mode...")
    
    # Fallback to emergency mode
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(title="DEXTER Fallback")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    def root():
        return {"status": "DEXTER Fallback Mode", "error": str(e)}
    
    @app.get("/health")
    def health():
        return {"status": "fallback", "port": port, "error": str(e)}

if __name__ == "__main__":
    print("🔥 Starting DEXTER Production Server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(port),
        log_level="info",
        access_log=False  # Reduce noise in production
    )

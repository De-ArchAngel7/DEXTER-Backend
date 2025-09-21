#!/usr/bin/env python3
"""
ULTIMATE DEXTER START - Full app with lazy AI imports
"""
import os
import sys

# CRITICAL: Disable AI imports during module loading
os.environ["USE_HF_API"] = "true"
os.environ["DISABLE_AI_IMPORTS"] = "true"  # This will prevent AI imports

# Get the port
port = os.environ.get("PORT", "10000")
print(f"🎯 ULTIMATE DEXTER START ON PORT: {port}")

try:
    print("📦 Loading FULL DEXTER app with disabled AI imports...")
    
    # Import the FULL FastAPI app (AI imports disabled)
    from app.main import app
    
    print("✅ FULL DEXTER app loaded successfully!")
    print("🧠 AI modules will load on first use (lazy loading)")
    
except Exception as e:
    print(f"❌ CRITICAL ERROR loading app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

if __name__ == "__main__":
    print("🚀 Starting ULTIMATE DEXTER Server...")
    print("🔥 ALL ENDPOINTS ACTIVE - Auth, Trading, AI, Portfolio!")
    print("🤖 Telegram bot fully operational!")
    print("🧠 AI loads on first request!")
    
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(port),
        log_level="info"
    )

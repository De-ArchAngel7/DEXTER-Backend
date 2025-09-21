#!/usr/bin/env python3
"""
ULTRA MINIMAL RENDER START - BYPASS ALL AI INITIALIZATION
"""
import os
import sys

# Get the port
port = os.environ.get("PORT", "10000")
print(f"🚀 MINIMAL START ON PORT: {port}")

# Disable ALL AI modules during startup
os.environ["DISABLE_AI_INIT"] = "true"
os.environ["USE_HF_API"] = "true"

# Import and run uvicorn directly
try:
    print("📦 Importing uvicorn...")
    import uvicorn
    print("✅ Uvicorn imported")
    
    print("🎯 Starting server...")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(port),
        log_level="info",
        access_log=False  # Reduce noise
    )
    
except Exception as e:
    print(f"❌ STARTUP FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

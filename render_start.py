#!/usr/bin/env python3
"""
Render-specific startup script for DEXTER
Ensures proper port binding for Render deployment
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import the FastAPI app
from app.main import app

if __name__ == "__main__":
    # Get port from Render environment
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting DEXTER on {host}:{port}")
    print("✅ Environment configured for Render")
    
    # Start uvicorn with explicit configuration
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )

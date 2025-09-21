#!/usr/bin/env python3
"""
DEAD SIMPLE RENDER STARTUP - NO COMPLEXITY
"""
import os
import sys

# Get the port - this MUST work
port = os.environ.get("PORT", "10000")
print(f"🚀 STARTING ON PORT: {port}")

# Set HF API mode
os.environ["USE_HF_API"] = "true"

# Just run uvicorn directly - no subprocess nonsense
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(port),
        log_level="info"
    )

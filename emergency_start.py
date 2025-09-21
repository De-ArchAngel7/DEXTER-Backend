#!/usr/bin/env python3
"""
EMERGENCY MINIMAL START - BYPASS ALL ENDPOINTS
"""
import os
from fastapi import FastAPI
import uvicorn

# Get the port
port = os.environ.get("PORT", "10000")
print(f"🚨 EMERGENCY START ON PORT: {port}")

# Create absolutely minimal FastAPI app
app = FastAPI(title="DEXTER Emergency")

@app.get("/")
def root():
    return {"status": "DEXTER is alive!", "message": "Emergency mode - basic endpoints only"}

@app.get("/health")
def health():
    return {"status": "healthy", "port": port}

if __name__ == "__main__":
    print("🔥 Starting emergency FastAPI server...")
    uvicorn.run(
        app,
        host="0.0.0.0", 
        port=int(port),
        log_level="info"
    )

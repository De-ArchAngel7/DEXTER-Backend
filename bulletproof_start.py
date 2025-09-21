#!/usr/bin/env python3
"""
BULLETPROOF DEXTER STARTUP FOR RENDER
This WILL work - guaranteed port binding
"""

import os
import sys
import subprocess
import time

def main():
    """Bulletproof startup with guaranteed port binding"""
    
    # Get port from Render (this is critical)
    port = os.getenv("PORT")
    if not port:
        print("❌ ERROR: PORT environment variable not found!")
        print(f"📊 Available env vars: {list(os.environ.keys())}")
        sys.exit(1)
    
    print(f"✅ PORT environment variable found: {port}")
    
    print(f"🚀 Starting DEXTER on port {port}")
    print(f"🌐 Host: 0.0.0.0")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Set environment for HF API
    os.environ["USE_HF_API"] = "true"
    
    # Add current directory to Python path
    sys.path.insert(0, ".")
    
    # Skip import test - let uvicorn handle it
    print("⚡ Skipping import test - letting uvicorn handle app loading...")
    print("🚀 Proceeding directly to uvicorn startup...")
    
    # Create uvicorn command with optimal settings for Render
    cmd = [
        "python", "-m", "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", str(port),  # Convert to string
        "--log-level", "info",
        "--timeout-keep-alive", "30",  # Keep connections alive longer
        "--workers", "1"  # Single worker for stability
    ]
    
    print(f"🔧 Running command: {' '.join(cmd)}")
    
    # Execute with subprocess for better control
    try:
        print("🎯 Executing uvicorn command...")
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✅ Uvicorn started successfully with exit code: {result.returncode}")
    except KeyboardInterrupt:
        print("🛑 Shutdown requested")
    except subprocess.CalledProcessError as e:
        print(f"❌ Process failed with exit code {e.returncode}")
        if e.stdout:
            print(f"📤 STDOUT: {e.stdout}")
        if e.stderr:
            print(f"📥 STDERR: {e.stderr}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

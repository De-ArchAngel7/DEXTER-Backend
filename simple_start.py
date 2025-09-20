#!/usr/bin/env python3
"""
Ultra-simple DEXTER startup for Render
Direct approach to fix port binding issues
"""

import os
import sys

# Add to path
sys.path.append('.')

def main():
    """Start DEXTER with minimal configuration"""
    
    # Get port from Render
    port = int(os.getenv("PORT", 8000))
    
    print(f"Starting DEXTER on port {port}")
    
    # Direct uvicorn command
    os.system(f"uvicorn app.main:app --host 0.0.0.0 --port {port} --log-level info")

if __name__ == "__main__":
    main()

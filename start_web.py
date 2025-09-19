#!/usr/bin/env python3
"""
RedCrowWatch Web Interface Startup Script

This script starts the web interface for easy video upload and analysis.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Start the RedCrowWatch web interface"""
    print("🚦 RedCrowWatch Web Interface")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("web_app.py").exists():
        print("❌ Error: web_app.py not found. Please run this from the RedCrowWatch directory.")
        sys.exit(1)
    
    # Check if templates directory exists
    if not Path("templates").exists():
        print("❌ Error: templates directory not found.")
        sys.exit(1)
    
    # Check if Flask is installed
    try:
        import flask
        print("✅ Flask is installed")
    except ImportError:
        print("❌ Flask not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "flask"], check=True)
        print("✅ Flask installed successfully")
    
    # Create necessary directories
    Path("data/videos/raw").mkdir(parents=True, exist_ok=True)
    Path("data/outputs").mkdir(parents=True, exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    
    print("✅ Directories created")
    print("🌐 Starting web server...")
    print("=" * 50)
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the web app
    try:
        from web_app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Web server stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting web server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
